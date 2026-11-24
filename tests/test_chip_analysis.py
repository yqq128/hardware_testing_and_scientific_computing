"""分析代码能找出生成器植入的线索（见 scripts/generate_chip_data.py 文档字符串）。

和 tests/test_diagnosis.py 一样，这里断言"线索被找出来了"，而不只是"代码能跑"。
"""

import numpy as np
import pandas as pd
import pytest

from chiptest import efficiency
from chiptest.data import first_pass


def test_top_site_hotspot_is_the_contact_problem(chip_run):
    _, run = chip_run
    top = run.report.hotspots_cell.iloc[0]
    assert (top["lot_id"], top["station"], top["site"]) == ("L04", "T02", 3)
    assert top["delta"] < -0.2


def test_contact_suspect_fails_on_contact_resistance_and_recovers_on_retest(chip_run):
    _, run = chip_run
    s = next(s for s in run.report.suspects if s.kind == "site hotspot")
    assert s.signature["top_test"] == "CONT_VDD"
    assert s.signature["top_test_rest_fail_rate"] < 0.001
    assert s.signature["recovery_rate"] > 3 * s.signature["rest_recovery_rate"]
    assert "接触" in s.hypothesis


def test_timing_offset_hits_station_t01_from_lot_5_and_does_not_recover(chip_run):
    _, run = chip_run
    effects = run.report.station_flags
    shift = effects[(effects["kind"] == "shift") & (effects["test"] == "SCAN_FMAX")]
    t01 = shift[shift["station"] == "T01"]
    assert set(t01["lot_id"]) == {"L05", "L06", "L07", "L08"}
    assert (t01["delta_sigma"] < -0.6).all()
    assert set(shift["station"]) == {"T01"}, "共识基线下不应把正常测试机也标出来"

    timing = [s for s in run.report.suspects if s.station == "T01" and s.kind == "station hotspot"]
    assert timing, "T01 的 FPY 下降应被标为测试机热点"
    for s in timing:
        assert s.signature["top_test"] == "SCAN_FMAX"
        assert s.signature["recovery_rate"] < 0.15
        assert "时序" in s.hypothesis


def test_power_noise_widens_spread_but_barely_moves_the_mean(chip_run):
    _, run = chip_run
    effects = run.report.station_flags
    noise = effects[effects["kind"] == "noise"]
    assert list(zip(noise["lot_id"], noise["test"], noise["station"], strict=True)) == [
        ("L06", "VMIN", "T03")
    ]
    row = noise.iloc[0]
    assert row["std_ratio"] > 2.0
    assert abs(row["delta_sigma"]) < 0.3

    s = next(s for s in run.report.suspects if s.kind == "noise")
    assert s.signature["top_test"] == "VMIN"
    assert "噪声" in s.hypothesis


def test_three_hardware_problems_have_distinguishable_retest_signatures(chip_run):
    _, run = chip_run
    sig = {s.kind + s.station: s.signature["recovery_rate"] for s in run.report.suspects}
    contact, noise = sig["site hotspotT02"], sig["noiseT03"]
    timing = sig["station hotspotT01"]
    assert contact > timing and noise > timing
    assert timing < 0.15


def test_fast_corner_lot_is_flagged_on_independent_tests_in_a_consistent_direction(chip_run):
    _, run = chip_run
    flags = run.report.lot_flags
    assert set(flags["lot_id"]) == {"L07"}
    by_test = flags.set_index("test")["delta_sigma"]
    assert by_test["IDD_STANDBY"] > 0 and by_test["SCAN_FMAX"] > 0 and by_test["IDD_ACTIVE"] > 0
    assert by_test["VMIN"] < 0  # 快角：速度高、漏电大、电流大、Vmin 低


def test_edge_dies_yield_less_than_center_dies(chip_run):
    _, run = chip_run
    radial = run.report.radial.set_index("ring")["fpy"]
    assert radial[5] < radial[1] - 0.03
    assert radial[5] < radial[4] < radial[3] + 0.005


def test_only_the_scratched_wafer_is_a_spatial_cluster(chip_run):
    _, run = chip_run
    clusters = run.report.clusters
    assert list(zip(clusters["lot_id"], clusters["wafer_id"], strict=True)) == [("L03", "07")]
    assert clusters.iloc[0]["largest_cluster"] >= 40


def test_cluster_test_does_not_flag_a_wafer_that_is_merely_low_yield(chip_run):
    """L05 的 10 号晶圆失败率很高、最大集群也不小，但置换检验显示它没有超出随机预期。"""
    _, run = chip_run
    w = run.report.wafers.set_index(["lot_id", "wafer_id"]).loc[("L05", "10")]
    assert w["largest_cluster"] >= 20
    assert not w["flagged"]


def test_cpk_flags_the_capability_limited_tests(chip_run):
    _, run = chip_run
    cpk = run.report.cpk.iloc[0]
    assert cpk["CONT_VDD"] > 5
    assert cpk["SCAN_FMAX"] < 1.0 and cpk["IDD_STANDBY"] < 1.0


def test_pat_residual_finds_weak_dies_that_a_static_limit_cannot(chip_run):
    _, run = chip_run
    pat = run.report.pat
    strict, loose = pat["residual"]["k=4.0"], pat["residual"]["k=3.5"]
    assert strict["precision"] > 0.9 and strict["recall"] > 0.4
    assert loose["recall"] > strict["recall"] and loose["precision"] > 0.8
    assert pat["static_k6"]["true_weak_flagged"] == 0


def test_pat_only_looks_at_dies_that_already_pass(chip_run):
    _, run = chip_run
    frame = run.report.pat_frame
    first = first_pass(run.clean).set_index(["lot_id", "wafer_id", "x", "y"])
    keys = list(zip(frame["lot_id"], frame["wafer_id"], frame["x"], frame["y"], strict=True))
    assert first.loc[keys, "final_ok"].all()


def test_retest_recovers_overkill_but_also_lets_marginal_bad_dies_through(chip_run):
    _, run = chip_run
    o = run.report.evaluation["overall"]
    assert o["overkill_final"] < 0.7 * o["overkill_first"]
    assert o["escape_final"] > o["escape_first"]


def test_retest_fixes_intermittent_contact_but_not_a_systematic_offset(chip_run):
    _, run = chip_run
    by_lot = pd.DataFrame(run.report.evaluation["by_lot"]).set_index("lot_id")
    recovered = 1 - by_lot["overkill_final"] / by_lot["overkill_first"]
    assert recovered["L04"] > 0.6  # 接触不良：复测大量挽回
    assert recovered["L08"] < 0.3  # 时序偏移：挽回很少，过杀留在最终良率里
    assert by_lot.loc["L08", "overkill_rate_final"] > 5 * by_lot.loc["L01", "overkill_rate_final"]


def test_sram_bist_dominates_test_time(chip_run):
    _, run = chip_run
    eff = run.report.efficiency.set_index("test")
    assert eff["time_share"].idxmax() == "SRAM_BIST"
    assert eff.loc["SRAM_BIST", "time_share"] > 0.3
    assert eff.loc["SRAM_BIST", "unique_fails"] > 0, "BIST 有独有失败，不能当冗余项"


def test_vref_and_ldo_out_are_mutual_redundancy_candidates(chip_run):
    _, run = chip_run
    red = run.report.redundancy.set_index("test")
    assert {"VREF", "LDO_OUT"} <= set(red.index)
    assert red.loc["VREF", "covered_by"] == "LDO_OUT"
    assert red.loc["LDO_OUT", "covered_by"] == "VREF"
    assert "SRAM_BIST" not in red.index and "SCAN_FMAX" not in red.index


def test_dropping_one_redundant_test_costs_almost_no_coverage(chip_run):
    _, run = chip_run
    what_if = run.report.drop_what_if.set_index("test")
    assert what_if.loc["LDO_OUT", "new_escapes"] <= 2
    assert what_if.loc["LDO_OUT", "saving"] > 0.01


def test_reordering_tests_saves_little_which_points_at_the_long_tests(chip_run):
    _, run = chip_run
    order = run.report.order
    assert 0 < order["saving"] < 0.05
    assert order["optimized_ms"] < order["current_ms"] < order["full_time_ms"]
    assert order["optimized_order"][:2] == ["CONT_VDD", "CONT_IO"], "接触测试固定在最前"


def test_expected_time_stops_at_the_first_failing_test():
    F = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 1]], dtype=bool)
    times = np.array([10.0, 20.0, 30.0])
    # 芯片1 花 10+20，芯片2 全部通过花 60，芯片3 在第 1 项就停花 10
    assert efficiency.expected_time(F, times, [0, 1, 2]) == pytest.approx((30 + 60 + 10) / 3)
    # 调换顺序后：芯片1 在第 1 项就停(20)，芯片3 要走到 test2 之后的 test0 才停(20+10)
    assert efficiency.expected_time(F, times, [1, 0, 2]) == pytest.approx((20 + 60 + 30) / 3)


def test_greedy_order_puts_cheap_high_yield_killers_first_and_respects_pinned():
    # test0 便宜但几乎不失败；test1 同样便宜、失败率高；test2 很贵
    F = np.zeros((100, 3), dtype=bool)
    F[:30, 1] = True
    F[30:32, 0] = True
    times = np.array([10.0, 10.0, 500.0])
    assert efficiency.greedy_order(F, times, pinned=[]) == [1, 0, 2]
    assert efficiency.greedy_order(F, times, pinned=[0]) == [0, 1, 2]
