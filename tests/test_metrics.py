import numpy as np
import pandas as pd
import pytest

from aoi.metrics import (
    false_call_denominators,
    precision_by_defect_rate,
    quality_metrics,
    rollup,
    trivial_baselines,
)

# 讲义例子：100 个缺陷、9900 个良品，AOI 抓到 95 个，误报 495 个。
ADR = {"tn": 9405, "fp": 495, "fn": 5, "tp": 95}


def test_quality_metrics_match_the_handout_answers():
    m = quality_metrics(**ADR)
    assert m["Recall"] == pytest.approx(0.95)
    assert m["Precision"] == pytest.approx(95 / 590)
    assert m["FPR"] == pytest.approx(0.05)
    assert m["Miss rate"] == pytest.approx(0.05)
    assert m["Accuracy"] == pytest.approx(0.9500)
    assert m["Manual review (all alarms)"] == 590


def test_zero_denominators_give_nan_not_an_exception():
    m = quality_metrics(tn=0, fp=0, fn=0, tp=0)
    for key in ["Precision", "Recall", "FPR", "Miss rate", "Accuracy"]:
        assert np.isnan(m[key])


def test_trivial_baselines_show_accuracy_is_misleading():
    table = trivial_baselines(**ADR)
    all_ok = table.loc["Predict everything OK"]
    assert all_ok["Accuracy"] == pytest.approx(0.99)
    assert all_ok["Recall"] == 0
    all_ng = table.loc["Predict everything NG"]
    assert all_ng["Recall"] == 1
    assert all_ng["FPR"] == 1


def test_false_call_denominators_use_different_bases():
    table = false_call_denominators(**ADR).set_index("Name")["Value"]
    assert table["FPR（良品误报率）"] == pytest.approx(495 / 9900)
    assert table["报警中的误报占比（1 - Precision）"] == pytest.approx(495 / 590)
    assert table["误报占全部检测数"] == pytest.approx(495 / 10000)


def test_precision_falls_with_rarer_defects_for_the_same_machine():
    table = precision_by_defect_rate(0.95, 0.05, [0.001, 0.01, 0.1])
    assert table["Precision"].is_monotonic_increasing
    # 缺陷率 1% 时，与讲义例子的 Precision 一致
    assert table.loc[1, "Precision"] == pytest.approx(0.95 * 0.01 / (0.95 * 0.01 + 0.05 * 0.99))


def test_rollup_counts_and_rates():
    frame = pd.DataFrame(
        {
            "inspection_id": [1, 2, 3, 4],
            "group": ["a", "a", "a", "a"],
            "is_alarm": [True, True, False, False],
            "is_actual_good": [True, False, True, False],
            "is_actual_defect": [False, True, False, True],
            "is_true_positive": [False, True, False, False],
            "is_false_positive": [True, False, False, False],
            "is_miss": [False, False, False, True],
        }
    )
    row = rollup(frame, "group").iloc[0]
    assert (row["inspections"], row["alarms"], row["good"], row["defects"]) == (4, 2, 2, 2)
    assert row["fpr"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(0.5)
    assert row["fp_share_of_alarms"] == pytest.approx(0.5)


def test_rollup_group_without_good_units_has_nan_fpr():
    frame = pd.DataFrame(
        {
            "inspection_id": [1],
            "group": ["only-defects"],
            "is_alarm": [True],
            "is_actual_good": [False],
            "is_actual_defect": [True],
            "is_true_positive": [True],
            "is_false_positive": [False],
            "is_miss": [False],
        }
    )
    assert np.isnan(rollup(frame, "group").iloc[0]["fpr"])
