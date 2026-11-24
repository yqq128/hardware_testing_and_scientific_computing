import pandas as pd
import pytest

from aoi.data import first_pass, with_reference
from aoi.diagnosis import drill_down

# 模拟数据里埋了三条线索（见 scripts/generate_data.py）：
# M03 在 10-14 光源衰减、L03 在 10-27 反光批次、夜班整体偏高。
# 这些断言保证分析代码能把它们找出来，而不是只检查“能跑”。


def test_peak_is_the_injected_machine_and_day(python_run):
    _, run = python_run
    anomaly = run.report.anomaly
    assert anomaly.machine == "M03"
    assert anomaly.day == pd.Timestamp("2026-10-14")
    assert anomaly.peer_machines, "同批次同日应有对照设备"


def test_anomaly_is_statistically_and_practically_significant(python_run):
    _, run = python_run
    anomaly = run.report.anomaly
    assert anomaly.fisher_p < 1e-3
    assert anomaly.target_fpr > 2 * anomaly.peer_fpr


def test_brightness_drops_on_the_anomaly_day(python_run):
    _, run = python_run
    compare = run.report.anomaly.brightness_compare
    assert compare.loc["anomaly day", "mean"] < compare.loc["other days", "mean"]


def test_cross_machine_batch_is_found_and_looks_like_glare(python_run):
    _, run = python_run
    report = run.report
    batch = report.cross_machine_batch
    assert batch is not None
    assert batch in set(report.flagged["batch_id"])
    assert batch not in report.anomaly.batches
    profile = report.cross_machine_profile
    # 反光：亮度偏高、对比度偏低（第 0 行是该批次，其余是对照）
    assert profile.iloc[0]["brightness"] > profile.iloc[1]["brightness"]
    assert profile.iloc[0]["contrast"] < profile.iloc[1]["contrast"]


def test_night_shift_has_higher_false_positive_rate(python_run):
    _, run = python_run
    shift = run.report.shift
    assert shift["night_fpr"] > shift["others_fpr"]
    assert shift["p_value"] < 1e-3


def test_overall_summary_matches_a_direct_count(python_run):
    _, run = python_run
    verified = with_reference(first_pass(run.clean))
    overall = run.report.overall
    assert overall["fp"] == int(verified["is_false_positive"].sum())
    assert overall["fpr"] == pytest.approx(
        verified["is_false_positive"].sum() / verified["is_actual_good"].sum()
    )
    assert overall["recall"] == pytest.approx(
        verified["is_true_positive"].sum() / verified["is_actual_defect"].sum()
    )


def test_drill_down_rejects_a_machine_with_no_records_that_day(python_run):
    _, run = python_run
    verified = with_reference(first_pass(run.clean))
    with pytest.raises(ValueError, match="没有"):
        drill_down(verified, "M99", pd.Timestamp("2026-10-14"))
