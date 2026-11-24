import json

import numpy as np
import pandas as pd
import pytest

from aoi import review_model
from aoi.review_model import (
    TARGET_RECALL,
    TEST_START,
    VALID_START,
    build_alarm_dataset,
    choose_threshold,
    run_experiment,
    split_by_time,
    sweep,
    with_product_first_date,
)


def _alarm_frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """rows: (product_id, inspection_time)；每行都是有基准结果的报警。"""
    times = pd.to_datetime([t for _, t in rows])
    return pd.DataFrame(
        {
            "product_id": [p for p, _ in rows],
            "inspection_time": times,
            "date": times.normalize(),
            "is_alarm": True,
            "has_reference": True,
            "is_actual_defect": [i % 2 == 0 for i in range(len(rows))],
        }
    )


def test_sweep_counts_each_outcome_at_each_threshold():
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.3, 0.6, 0.1])
    table = sweep(y, proba, [0.5, 0.95]).set_index("threshold")

    at_half = table.loc[0.5]
    assert (at_half["to_review"], at_half["auto_released"]) == (2, 2)
    assert (at_half["defects_caught"], at_half["defects_missed"]) == (1, 1)
    assert at_half["recall"] == pytest.approx(0.5)
    assert at_half["false_positives_released"] == 1
    assert at_half["workload_reduction"] == pytest.approx(0.5)

    # 阈值高到没人送审：工作量全省，但缺陷全漏
    at_high = table.loc[0.95]
    assert at_high["recall"] == 0
    assert at_high["workload_reduction"] == 1


def test_choose_threshold_is_the_largest_one_meeting_target_recall():
    table = pd.DataFrame({"threshold": [0.1, 0.2, 0.3, 0.4], "recall": [1.0, 0.99, 0.98, 0.90]})
    assert choose_threshold(table, TARGET_RECALL) == 0.3


def test_choose_threshold_fails_loudly_when_target_is_unreachable():
    table = pd.DataFrame({"threshold": [0.1, 0.2], "recall": [0.5, 0.4]})
    with pytest.raises(ValueError, match="Recall"):
        choose_threshold(table, TARGET_RECALL)


def test_retest_after_midnight_stays_in_the_same_split_as_the_first_inspection():
    # 产品 p1 在验证期开始前一天 23:50 首检，复测落在验证期第一天 00:10。
    # 若按记录自己的日期切分，同一产品会同时出现在训练集和验证集。
    clean = _alarm_frame(
        [
            ("p1", "2026-10-20 23:50"),
            ("p1", "2026-10-21 00:10"),
            ("p2", "2026-10-22 10:00"),
            ("p3", "2026-10-27 10:00"),
        ]
    )
    splits = split_by_time(build_alarm_dataset(with_product_first_date(clean)))
    assert set(splits.train["product_id"]) == {"p1"}
    assert len(splits.train) == 2
    assert set(splits.valid["product_id"]) == {"p2"}
    assert set(splits.test["product_id"]) == {"p3"}


def test_split_rejects_a_product_that_lands_in_two_sets():
    clean = _alarm_frame([("p1", "2026-10-01 10:00"), ("p1", "2026-10-22 10:00")])
    # 人为制造不一致的首检日期，模拟切分逻辑出错的情况
    broken = clean.assign(product_first_date=clean["date"])
    with pytest.raises(ValueError, match="多个集合"):
        split_by_time(broken)


# --- 以下使用完整模拟数据（python_run 只在整个测试会话里跑一次） ---


def test_experiment_splits_are_chronological(python_run):
    _, run = python_run
    splits = run.experiment.splits
    assert splits.train["product_first_date"].max() < VALID_START
    assert VALID_START <= splits.valid["product_first_date"].min()
    assert splits.valid["product_first_date"].max() < TEST_START
    assert splits.test["product_first_date"].min() >= TEST_START


def test_chosen_threshold_meets_target_recall_on_validation(python_run):
    _, run = python_run
    exp = run.experiment
    valid_row = exp.valid_sweep[exp.valid_sweep["threshold"] == exp.chosen_threshold].iloc[0]
    assert valid_row["recall"] >= TARGET_RECALL
    assert exp.chosen_threshold == choose_threshold(exp.valid_sweep)


def test_test_set_labels_cannot_influence_model_or_threshold(python_run):
    """把测试期的真实标签全部翻转后，验证集上的预测和选定的阈值必须不变。

    这是对“阈值只在验证集上选、测试集只评估”这条纪律的直接检验。
    """
    _, run = python_run
    flipped = run.clean.copy()
    # 按产品首检日期判断是否属于测试期，与切分口径一致
    in_test = with_product_first_date(flipped)["product_first_date"] >= TEST_START
    flipped.loc[in_test, "is_actual_defect"] = ~flipped.loc[in_test, "is_actual_defect"]

    again = run_experiment(flipped, include_leakage_demo=False)
    assert again.chosen_threshold == run.experiment.chosen_threshold
    np.testing.assert_allclose(again.valid_proba, run.experiment.valid_proba)


def test_metrics_are_internally_consistent(python_run):
    _, run = python_run
    exp = run.experiment
    t = exp.test_row
    assert t["defects_caught"] + t["defects_missed"] == exp.defects_total
    assert t["to_review"] + t["auto_released"] == len(exp.splits.test)
    low, high = exp.recall_ci
    assert low <= t["recall"] <= high
    # 端到端保留率不可能高于 AOI 自己的 Recall（复判模型只会再放走缺陷）
    assert exp.end_to_end.retention <= exp.end_to_end.aoi_recall


def test_row_level_split_inflates_the_score_compared_with_group_split(python_run):
    _, run = python_run
    leak = run.experiment.leakage["PR-AUC (mean)"]
    assert leak["Random split by row (leaky)"] > leak["Random split grouped by product_id"]


def test_metrics_json_matches_the_experiment(python_run):
    paths, run = python_run
    on_disk = json.loads(paths.metrics.read_text())
    assert on_disk == json.loads(run.experiment.metrics_json())
    assert on_disk["model"] == run.experiment.best_name
    assert set(review_model.FEATURES) == set(on_disk["features"])
