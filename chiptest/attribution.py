"""归因：低良率是测试系统（硬件）的问题，还是芯片/工艺（硅）的问题？

思路与 AOI 项目里"同批次对照设备"一致：同一个 lot 的芯片被随机分到不同的测试机和 site，
所以只要某个 (测试机, site) 的良率显著低于同 lot 里其余部分，问题就更像出在测试系统上，而不是 lot 本身。

定位到格子之后，再看三件事把原因分开：
  1. 哪个测试项在失败（接触电阻？VMIN？SCAN_FMAX？）
  2. 复测恢复率（间歇性问题恢复率高，系统性偏移几乎不恢复）
  3. 是否只出现在某个 site（接触/针卡）还是整台测试机（电源、时序校准）
"""

import numpy as np
import pandas as pd
from scipy import stats

from chiptest.yield_analysis import yield_by


def peer_hotspots(first: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """每个分组的 FPY 对比"同 lot 其余芯片"，用 Fisher 精确检验衡量差异。by 的第一项必须是 lot_id。"""
    table = yield_by(first, by)
    lot_total = first.groupby("lot_id").agg(n=("x", "size"), p=("tester_pass", "sum"))
    rows = []
    for _, row in table.iterrows():
        n_all, p_all = lot_total.loc[row["lot_id"], ["n", "p"]]
        n_rest, p_rest = int(n_all - row["dies"]), int(p_all - row["passed"])
        _, p_value = stats.fisher_exact(
            [
                [int(row["passed"]), int(row["dies"] - row["passed"])],
                [p_rest, n_rest - p_rest],
            ]
        )
        rows.append((p_rest / n_rest, p_value))
    table[["rest_fpy", "fisher_p"]] = rows
    table["delta"] = table["fpy"] - table["rest_fpy"]
    return table


def flag_hotspots(table: pd.DataFrame, p_max: float = 1e-6, min_drop: float = 0.03) -> pd.DataFrame:
    """既要统计显著，也要有实际影响（样本多时微小差异也会显著）。"""
    hit = table[(table["fisher_p"] < p_max) & (table["delta"] <= -min_drop)]
    return hit.sort_values("fisher_p").reset_index(drop=True)


def failing_tests(first: pd.DataFrame, cell: pd.Series, tests: list[str]) -> pd.DataFrame:
    """热点格子里哪些测试项失败得更多：与同 lot 其余芯片的失败率对比（lift = 倍数）。"""
    lot = first.loc[cell, "lot_id"].unique()
    rest = first[first["lot_id"].isin(lot) & ~cell]
    inside = first[cell]
    out = pd.DataFrame(
        {
            "fail_rate": [inside[f"fail_{t}"].mean() for t in tests],
            "rest_fail_rate": [rest[f"fail_{t}"].mean() for t in tests],
        },
        index=pd.Index(tests, name="test"),
    )
    out["lift"] = out["fail_rate"] / out["rest_fail_rate"].replace(0, np.nan)
    return out.sort_values("fail_rate", ascending=False).reset_index()


def hypothesis(sig: dict, site_specific: bool, noisy: bool) -> str:
    """由指纹给出"待验证假设"。它只是排查的起点，不是结论：每一条都要到现场用仪器确认。"""
    # 对照组恢复率本身很低（几个百分点），只比倍数不可靠，所以同时要求绝对值：复测至少挽回四分之一
    recovers = (
        sig["recovery_rate"] >= 0.25 and sig["recovery_rate"] >= 3 * sig["rest_recovery_rate"]
    )
    if site_specific and sig["top_test"].startswith("CONT_") and recovers:
        return (
            "单个 site、接触电阻升高、复测大量恢复：像探针/插座接触不良或针卡磨损（类比虚焊）。"
            "验证：看该 site 的接触电阻分布与探针痕迹，换针卡/插座后对比。"
        )
    if noisy:
        return (
            "只有离散度变大、均值不变，复测部分恢复：像该测试机电源或测量噪声变大。"
            "验证：示波器看电源纹波与地弹，接上解耦电容或换电源板后对比。"
        )
    if not recovers:
        return (
            "整台测试机同一测试项系统性偏移、复测不恢复：像时序/校准偏移（edge placement、"
            "strobe）或程序变更。验证：用同一批样片（golden unit）在各测试机上做相关性对比，重新校准。"
        )
    return "复测恢复偏高但特征不典型：先确认接触与环境，再扩大样本。"


def signature(first: pd.DataFrame, cell: pd.Series, tests: list[str]) -> dict:
    """一个热点的"指纹"：失败率、同 lot 对照、主要失败测试项、复测恢复率。"""
    lot = first.loc[cell, "lot_id"].unique()
    rest = first[first["lot_id"].isin(lot) & ~cell]
    inside = first[cell]
    top = failing_tests(first, cell, tests).iloc[0]
    return {
        "dies": int(len(inside)),
        "fpy": float(inside["tester_pass"].mean()),
        "rest_fpy": float(rest["tester_pass"].mean()),
        "top_test": str(top["test"]),
        "top_test_fail_rate": float(top["fail_rate"]),
        "top_test_rest_fail_rate": float(top["rest_fail_rate"]),
        "recovery_rate": float(yield_by(inside.assign(s=1), "s")["recovery_rate"].iloc[0]),
        "rest_recovery_rate": float(yield_by(rest.assign(s=1), "s")["recovery_rate"].iloc[0]),
    }
