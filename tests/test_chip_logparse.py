"""日志解析器：用手工构造的小日志验证脏数据处理，再和生成器的真值对账。"""

import math

import numpy as np
import pytest

from chiptest.logparse import LogParser, si_factor

HEADER = """\
HBR|1|PASS
HBR|2|CONTACT
TSR|1001|CONT_VDD|ohm|0|2|20
TSR|1101|IDD_STANDBY|uA|0|150|60
"""


def parse(tmp_path, body: str, header: str = HEADER):
    path = tmp_path / "L99.log"
    path.write_text(header + body)
    parser = LogParser()
    parser.parse_file(path)
    return parser.result()


def wafer(*parts: str, wir="WIR|L99|01|T01|P1|1|2026-10-01T08:00:00", count=None) -> str:
    n = count if count is not None else sum(1 for p in parts if "PRR|" in p)
    return wir + "\n" + "\n".join(parts) + f"\nWRR|L99|01|1|{n}\n"


def part(pid="0001", cont="0.3", leak="40", hbin=1, x=0, y=0, prr=True) -> str:
    lines = [
        f"PIR|{pid}|{x}|{y}|1|2026-10-01T08:00:00",
        f"PTR|1001|{cont}|P",
        f"PTR|1101|{leak}|P",
    ]
    if prr:
        lines.append(f"PRR|{pid}|{hbin}|{'P' if hbin == 1 else 'F'}")
    return "\n".join(lines)


def test_si_factor_converts_prefixes_and_rejects_different_dimensions():
    assert si_factor("mA", "uA") == 1000.0
    assert si_factor("uA", "mA") == pytest.approx(1e-3)
    assert si_factor("MHz", "GHz") == pytest.approx(1e-3)
    assert si_factor("ohm", "ohm") == 1.0
    assert si_factor("V", "A") is None
    assert si_factor("furlong", "V") is None


def test_clean_wafer_is_parsed_into_one_row_per_part(tmp_path):
    r = parse(tmp_path, wafer(part("0001"), part("0002", cont="0.5", x=1)))
    assert len(r.parts) == 2
    assert list(r.parts["CONT_VDD"]) == [0.3, 0.5]
    assert r.stats.bad_lines == 0
    assert r.stats.wafer_count_mismatch == 0
    assert r.bins == {1: "PASS", 2: "CONTACT"}
    assert list(r.tests["name"]) == ["CONT_VDD", "IDD_STANDBY"]


def test_truncated_ptr_line_is_counted_and_value_left_missing(tmp_path):
    broken = part("0001").replace("PTR|1101|40|P", "PTR|11")
    r = parse(tmp_path, wafer(broken))
    assert r.stats.bad_lines == 1
    assert len(r.parts) == 1  # 芯片仍保留，只是缺这一项
    assert math.isnan(r.parts.loc[0, "IDD_STANDBY"])
    assert r.parts.loc[0, "CONT_VDD"] == 0.3


def test_sentinel_and_nan_values_become_missing(tmp_path):
    body = wafer(part("0001", leak="1.0E+38"), part("0002", leak="nan", x=1))
    r = parse(tmp_path, body)
    assert r.stats.sentinel_values == 1
    assert r.stats.nan_values == 1
    assert r.parts["IDD_STANDBY"].isna().all()


def test_part_without_prr_is_dropped_and_wafer_count_mismatch_is_flagged(tmp_path):
    body = wafer(part("0001"), part("0002", prr=False, x=1), part("0003", x=2), count=3)
    r = parse(tmp_path, body)
    assert list(r.parts["part_id"]) == ["0001", "0003"]
    assert r.stats.incomplete_parts == 1
    assert r.stats.wafer_count_mismatch == 1


def test_trailing_part_without_prr_is_dropped_at_wafer_end(tmp_path):
    r = parse(tmp_path, wafer(part("0001"), part("0002", prr=False, x=1), count=2))
    assert len(r.parts) == 1
    assert r.stats.incomplete_parts == 1


def test_duplicated_part_block_is_kept_once(tmp_path):
    block = part("0001")
    r = parse(tmp_path, wafer(block, block, part("0002", x=1), count=2))
    assert len(r.parts) == 2
    assert r.stats.duplicate_parts == 1


def test_unit_change_is_converted_to_the_first_declared_unit(tmp_path):
    first = wafer(part("0001", leak="40"))
    redeclared = "TSR|1101|IDD_STANDBY|mA|0|0.15|60\n"
    second = wafer(
        part("0001", leak="0.04"), wir="WIR|L99|02|T01|P1|1|2026-10-01T09:00:00"
    ).replace("WRR|L99|01", "WRR|L99|02")
    r = parse(tmp_path, first + redeclared + second)
    assert r.parts["IDD_STANDBY"].tolist() == pytest.approx([40.0, 40.0])
    assert r.stats.unit_converted_values == 1
    assert r.tests.loc[0, "unit"] == "ohm" and r.tests.loc[1, "unit"] == "uA"


def test_parse_is_streaming_one_frame_per_wafer_pass(tmp_path):
    body = wafer(part("0001")) + wafer(
        part("0001"), wir="WIR|L99|02|T02|P1|1|2026-10-01T09:00:00"
    ).replace("WRR|L99|01", "WRR|L99|02")
    r = parse(tmp_path, body)
    assert r.stats.wafer_passes == 2
    assert sorted(r.parts["wafer_id"]) == ["01", "02"]
    assert sorted(r.parts["station"]) == ["T01", "T02"]


def test_parsed_tests_match_the_generator_test_program(chip_run, chip_generator):
    _, run = chip_run
    expected = {t[1]: t for t in chip_generator.TESTS}
    for _, row in run.tests.iterrows():
        _, _, unit, lo, hi, ms, _ = expected[row["name"]]
        assert row["unit"] == unit
        assert row["time_ms"] == ms
        np.testing.assert_allclose([row["lo"], row["hi"]], [lo, hi], equal_nan=True)


def test_almost_every_physical_die_survives_dirty_logs(chip_run, chip_root):
    _, run = chip_run
    _, truth = chip_root
    dies = run.clean.drop_duplicates(["lot_id", "wafer_id", "x", "y"])
    # 只有"首测缺 PRR 且没被复测"的芯片会彻底丢失，应当远少于 1%
    assert 0.99 * len(truth) < len(dies) <= len(truth)
    assert run.parsed.stats.incomplete_parts > 0, "生成器应当植入缺 PRR 的脏数据"


def test_unit_mix_is_repaired_before_analysis(chip_run):
    """L02 的 5 号晶圆以 mA 记录 IDD_STANDBY；不换算的话它会比其他晶圆小 1000 倍。"""
    _, run = chip_run
    lot = run.clean[run.clean["lot_id"] == "L02"]
    odd = lot[lot["wafer_id"] == "05"]["IDD_STANDBY"].median()
    rest = lot[lot["wafer_id"] != "05"]["IDD_STANDBY"].median()
    assert run.parsed.stats.unit_converted_values > 0
    assert 0.7 < odd / rest < 1.4


def test_tester_flags_agree_with_limits_recomputed_from_the_log(chip_run):
    _, run = chip_run
    clean = run.clean
    # 偶有极个别芯片的某项值恰好落在限值上，被 %.5g 取整后翻转；应当极少
    assert clean["flag_mismatch"].mean() < 1e-4
