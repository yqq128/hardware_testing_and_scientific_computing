"""参数测试数据：测试机对比、批次偏移、Cpk，以及稳健离群值筛选（PAT）。

三条纪律：
1. 全程用稳健统计（中位数 + MAD），不用均值和标准差。致命缺陷芯片的读数极端，会把均值和标准差拉偏。
2. 比较要在同一个 lot 内跨测试机做，这样 lot 本身的工艺偏移不会混进来。
3. 样本量很大时任何微小差异 p 值都极小，所以同时看效应量（偏移多少个 σ、σ 放大几倍），不只看 p 值。
"""

import numpy as np
import pandas as pd
from scipy import stats

# 服从对数正态的量，在 ln 尺度上分析（漏电随工艺角指数变化）
LOG_TESTS = {"IDD_STANDBY"}


def robust_sigma(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def _scale(values: pd.Series, test: str) -> pd.Series:
    return np.log(values.clip(lower=1e-12)) if test in LOG_TESTS else values


def station_effects(first: pd.DataFrame, tests: list[str], min_dies: int = 300) -> pd.DataFrame:
    """同 lot 内，每台测试机相对"共识"的中位数偏移（以 σ 计）和 σ 放大倍数。

    delta_sigma  ≈ 均值偏移（时序/校准偏移的特征）
    std_ratio    ≈ 离散度放大（电源噪声的特征：均值几乎不变，σ 变大）

    共识 = 各测试机中位数的中位数、各测试机 σ 的中位数。至少要 3 台测试机：一台出问题时，
    共识仍由正常的两台决定；若改用"其余测试机合并"当基线，坏机器会把基线拖偏，
    让正常的测试机看起来也偏了。

    局限：每台测试机测的是不同的几片晶圆，晶圆间本来就有差异，会带来约 0.3~0.45σ 的假偏移；
    要排除它，需要把同一批样片（golden unit）在每台测试机上重复测。
    """
    rows = []
    for lot, lot_frame in first.groupby("lot_id"):
        for test in tests:
            values = _scale(lot_frame[test], test)
            groups = {
                s: values[values.notna() & (lot_frame["station"] == s)]
                for s in lot_frame["station"].unique()
            }
            groups = {s: g for s, g in groups.items() if len(g) >= min_dies}
            if len(groups) < 3:
                continue
            p_var = stats.levene(*groups.values(), center="median").pvalue
            p_mean = stats.kruskal(*groups.values()).pvalue
            medians = {s: g.median() for s, g in groups.items()}
            sigmas = {s: robust_sigma(g) for s, g in groups.items()}
            center, sigma0 = np.median(list(medians.values())), np.median(list(sigmas.values()))
            for station, g in groups.items():
                rows.append(
                    {
                        "lot_id": lot,
                        "test": test,
                        "station": station,
                        "dies": len(g),
                        "delta_sigma": (medians[station] - center) / sigma0,
                        "std_ratio": sigmas[station] / sigma0,
                        "p_mean": p_mean,
                        "p_var": p_var,
                    }
                )
    return pd.DataFrame(rows)


def flag_station_effects(
    table: pd.DataFrame, shift_sigma: float = 0.6, std_ratio: float = 1.5, p_max: float = 1e-6
) -> pd.DataFrame:
    shifted = (table["delta_sigma"].abs() >= shift_sigma) & (table["p_mean"] < p_max)
    noisy = (table["std_ratio"] >= std_ratio) & (table["p_var"] < p_max)
    hit = table.assign(kind=np.select([noisy, shifted], ["noise", "shift"], default=""))
    hit = hit[hit["kind"] != ""]
    return hit.sort_values(["kind", "test", "lot_id"]).reset_index(drop=True)


def lot_effects(first: pd.DataFrame, tests: list[str]) -> pd.DataFrame:
    """每个 lot 的中位数相对其余 lot 偏移多少个 σ：用来发现整批的工艺偏移（快角/慢角）。"""
    rows = []
    for test in tests:
        values = _scale(first[test], test)
        for lot, mask in first.groupby("lot_id").groups.items():
            inside = values.loc[mask].dropna()
            rest = values.drop(index=mask).dropna()
            rows.append(
                {
                    "lot_id": lot,
                    "test": test,
                    "delta_sigma": (inside.median() - rest.median()) / robust_sigma(rest),
                }
            )
    return pd.DataFrame(rows)


def flag_lot_effects(
    table: pd.DataFrame, min_sigma: float = 0.5, min_tests: int = 3
) -> pd.DataFrame:
    """同一批在多个独立测试项上同时显著偏移才算工艺偏移，单项偏移更可能是测试问题。

    只要求偏移的测试项个数，不要求符号一致：快角会表现为速度上升、漏电上升、Vmin 下降。
    """
    big = table[table["delta_sigma"].abs() >= min_sigma]
    counts = big.groupby("lot_id")["test"].transform("size")
    return big[counts >= min_tests].sort_values(["lot_id", "delta_sigma"]).reset_index(drop=True)


def cpk_table(first: pd.DataFrame, tests: pd.DataFrame, by: str | None = "lot_id") -> pd.DataFrame:
    """稳健 Cpk（中位数 + MAD）。下限为 0 的量是物理边界，不当作规格下限。"""
    limits = tests.set_index("name")
    cont = [t for t in tests["name"] if limits.loc[t, "unit"] != "cnt"]
    groups = first.groupby(by) if by else [("all", first)]
    out = {}
    for key, frame in groups:
        row = {}
        for t in cont:
            lo, hi = limits.loc[t, ["lo", "hi"]]
            v = _scale(frame[t], t).dropna()
            if t in LOG_TESTS:
                lo, hi = (np.log(lo) if lo > 0 else np.nan), np.log(hi)
            lo = np.nan if lo == 0 else lo
            mu, sigma = v.median(), robust_sigma(v)
            sides = [
                (hi - mu) if not np.isnan(hi) else np.inf,
                (mu - lo) if not np.isnan(lo) else np.inf,
            ]
            row[t] = min(sides) / (3 * sigma)
        out[key] = row
    return pd.DataFrame(out).T.rename_axis(by or "scope")


def residual_zscores(
    frame: pd.DataFrame, target: str, predictors: list[str], by: list[str], trim: float = 3.0
) -> pd.Series:
    """相对"同速度、同电流下的预期漏电"偏大多少个稳健 σ（残差法 PAT / 邻近残差思路）。

    每个 (lot, 测试机) 单独拟合一条稳健回归：先 OLS，再反复剔除 |残差| 超过 trim 个稳健 σ 的点后重拟合。
    按测试机分组，是为了让测试机之间的系统偏移不进入基线（保守做法；本模拟数据里，
    因为 IDD_ACTIVE 也带有速度信息，结果与只按 lot 分组几乎一致）。
    """
    z = pd.Series(np.nan, index=frame.index)
    for _, g in frame.groupby(by):
        g = g.dropna(subset=[target, *predictors])
        if len(g) < 50:
            continue
        y = np.log(g[target].to_numpy())
        X = np.column_stack([np.ones(len(g)), g[predictors].to_numpy()])
        keep = np.ones(len(g), dtype=bool)
        for _ in range(4):
            beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
            resid = y - X @ beta
            med, sigma = np.median(resid[keep]), robust_sigma(resid[keep])
            keep = np.abs(resid - med) < trim * sigma
        z.loc[g.index] = (resid - med) / sigma
    return z


def static_zscores(frame: pd.DataFrame, tests: list[str], by: list[str]) -> pd.Series:
    """朴素基线：每个测试项在 (lot, 测试机) 内的稳健 z，取各项绝对值的最大值。"""
    worst = pd.Series(0.0, index=frame.index)
    for _, g in frame.groupby(by):
        for t in tests:
            v = _scale(g[t], t)
            sigma = robust_sigma(v)
            if sigma and not np.isnan(sigma):
                worst.loc[g.index] = np.maximum(
                    worst.loc[g.index], ((v - v.median()) / sigma).abs().fillna(0)
                )
    return worst


def passing_complete(first: pd.DataFrame, tests: list[str]) -> pd.DataFrame:
    """PAT 只对已经通过全部规格、且数据完整的芯片做：目标是找出'合格但可疑'的。"""
    return first[first["tester_pass"] & first["final_ok"] & first[tests].notna().all(axis=1)]
