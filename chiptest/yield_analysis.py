"""良率、复测与 bin 分布。

三个良率口径不要混：
- FPY（首测良率）：首测一次通过 / 首测芯片数。它同时反映芯片和测试系统。
- 最终良率：复测之后的最终判定。
- 复测恢复率：首测失败且被复测的芯片里，最终通过的比例。它高，说明失败是间歇性的（接触、噪声、
  裕量边缘），而不是芯片本身坏了——这是区分"硬件问题"和"真缺陷"的第一条线索。
"""

import numpy as np
import pandas as pd


def yield_by(first: pd.DataFrame, by) -> pd.DataFrame:
    """first 必须是首测记录（每颗芯片一条）。"""
    failed = ~first["tester_pass"]
    frame = first.assign(
        failed=failed,
        retested_fail=failed & first["retested"],
        recovered=failed & first["retested"] & first["final_ok"],
    )
    out = frame.groupby(by, observed=True).agg(
        dies=("x", "size"),
        passed=("tester_pass", "sum"),
        final_passed=("final_ok", "sum"),
        retested=("retested_fail", "sum"),
        recovered=("recovered", "sum"),
    )
    out["fpy"] = out["passed"] / out["dies"]
    out["final_yield"] = out["final_passed"] / out["dies"]
    out["recovery_rate"] = out["recovered"] / out["retested"].replace(0, np.nan)
    return out.reset_index()


def overall(first: pd.DataFrame) -> dict:
    row = yield_by(first.assign(scope="all"), "scope").iloc[0]
    return {
        "dies": int(row["dies"]),
        "fpy": float(row["fpy"]),
        "final_yield": float(row["final_yield"]),
        "retested": int(row["retested"]),
        "recovery_rate": float(row["recovery_rate"]),
    }


def bin_pareto(first: pd.DataFrame) -> pd.DataFrame:
    """首测失败芯片按硬件 bin 的 Pareto（bin 取第一个失败的测试项所属的 bin）。"""
    fails = first[~first["tester_pass"]]
    out = fails.groupby(["hbin", "hbin_name"]).size().rename("dies").reset_index()
    out = out.sort_values("dies", ascending=False).reset_index(drop=True)
    out["share_of_fails"] = out["dies"] / out["dies"].sum()
    out["cumulative"] = out["share_of_fails"].cumsum()
    return out
