"""芯片测试日志的质量体检、清洗与派生字段。

统计口径（和 AOI 项目里"以首检为单位"同一个思路）：
- 一颗物理芯片 = (lot, wafer, x, y)。同一颗芯片可能有首测（pass 1）和复测（pass 2）两条记录。
- 首测良率 FPY 以首测为单位，反映测试系统 + 芯片的真实表现；
- 复测后良率 final yield 用最后一次测试结果，反映最终出货口径。
  两者之差是"复测挽回"的部分，它既可能是真的间歇性问题，也可能是在掩盖测试硬件缺陷。
"""

from pathlib import Path

import pandas as pd

from chiptest.logparse import ParsedLogs

DIE_KEY = ["lot_id", "wafer_id", "x", "y"]
META_COLUMNS = [
    "lot_id",
    "wafer_id",
    "pass_no",
    "part_id",
    "x",
    "y",
    "site",
    "station",
    "program",
    "test_time",
    "hbin",
    "tester_pass",
]


def quality_report(parsed: ParsedLogs, clean: pd.DataFrame | None = None) -> pd.Series:
    """日志体检：先交代丢了什么、修了什么，再谈分析。clean 给出时追加跨字段一致性检查。"""
    s = parsed.stats
    report = {
        "files": s.files,
        "lines": s.lines,
        "PTR records": s.ptr_lines,
        "unparseable lines (truncated/orphan)": s.bad_lines,
        "sentinel values (1E+38, treated as missing)": s.sentinel_values,
        "nan values": s.nan_values,
        "values unit-converted (e.g. mA -> uA)": s.unit_converted_values,
        "wafer passes": s.wafer_passes,
        "wafer passes whose part count != WRR": s.wafer_count_mismatch,
        "parts parsed": s.parts,
        "incomplete parts dropped (no PRR)": s.incomplete_parts,
        "duplicate part blocks dropped": s.duplicate_parts,
    }
    if clean is not None:
        names = [c for c in parsed.tests["name"] if c in clean.columns]
        report["parts with a missing test value"] = int(clean[names].isna().any(axis=1).sum())
        report["dies without a first pass (PRR lost)"] = int(
            clean.groupby(DIE_KEY)["is_first_pass"].any().eq(False).sum()
        )
        report["tester flag != recomputed from limits"] = int(clean["flag_mismatch"].sum())
    return pd.Series(report, name="count")


def build_clean(parsed: ParsedLogs) -> pd.DataFrame:
    """排序、标注首测/末次、按限值重算每项是否失败，并校验与测试机判定是否一致。"""
    df = parsed.parts.sort_values(["lot_id", "wafer_id", "pass_no", "part_id"]).reset_index(
        drop=True
    )
    tests = parsed.tests.set_index("name")

    die = df.groupby(DIE_KEY)["pass_no"]
    df["is_first_pass"] = df["pass_no"] == 1
    df["is_final_pass"] = df["pass_no"] == die.transform("max")
    df["retested"] = die.transform("max") > 1
    final_ok = df.loc[df["is_final_pass"], [*DIE_KEY, "tester_pass"]].drop_duplicates(DIE_KEY)
    df = df.merge(final_ok.rename(columns={"tester_pass": "final_ok"}), on=DIE_KEY, how="left")

    df["date"] = df["test_time"].dt.normalize()
    df["hbin_name"] = df["hbin"].map(parsed.bins)

    lo, hi = tests["lo"], tests["hi"]
    fails = {}
    for name in tests.index:
        v = df[name]
        fails[f"fail_{name}"] = (v < lo[name]) | (v > hi[name])
    fail_frame = pd.DataFrame(fails)
    df = pd.concat([df, fail_frame], axis=1)
    df["n_fail_tests"] = fail_frame.sum(axis=1)

    # 有缺失值的芯片无法复核，不计入不一致
    complete = ~df[list(tests.index)].isna().any(axis=1)
    df["flag_mismatch"] = complete & (df["tester_pass"] != (df["n_fail_tests"] == 0))
    return df


def first_pass(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_first_pass"]]


def load_clean(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, parse_dates=["test_time", "date"], dtype={"wafer_id": str, "part_id": str}
    )


def load_tests(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def continuous_tests(tests: pd.DataFrame) -> list[str]:
    """连续量测试项（排除计数类的 BIST/功能测试）。"""
    return list(tests.loc[tests["unit"] != "cnt", "name"])
