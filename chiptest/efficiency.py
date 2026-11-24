"""测试效率与可测性反馈：测试时间花在哪、哪些测试项冗余、测试顺序能否更省。

注意这份数据是"继续测完"的工程验证模式（失败后不停），所以能算出每个测试项的"独有失败"。
量产时通常是"遇失败即停"，同一份数据可以反推任何测试顺序下的期望测试时间。

数据里看起来冗余，不等于可以删：必须回到故障模型和覆盖率（ATPG/BIST 覆盖了哪些缺陷）去确认，
这里输出的是给 DFT 与测试方案评审的"候选清单"，不是决定。
"""

import numpy as np
import pandas as pd


def fail_matrix(first: pd.DataFrame, tests: pd.DataFrame) -> np.ndarray:
    return first[[f"fail_{n}" for n in tests["name"]]].to_numpy(dtype=bool)


def per_test_summary(first: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    """每个测试项：耗时占比、失败数、独有失败数（只有它抓到）、与其他项的最大重叠。"""
    F = fail_matrix(first, tests)
    only_one = F.sum(axis=1) == 1
    names = list(tests["name"])
    overlap = _coverage_overlap(F)
    overlap.index, overlap.columns = names, names
    table = pd.DataFrame(
        {
            "test": names,
            "time_ms": tests["time_ms"].to_numpy(),
            "time_share": tests["time_ms"].to_numpy() / tests["time_ms"].sum(),
            "fails": F.sum(axis=0),
            "unique_fails": (F & only_one[:, None]).sum(axis=0),
        }
    )
    table["unique_share"] = table["unique_fails"] / table["fails"].replace(0, np.nan)
    best = overlap.max(axis=1)
    table["max_covered_by_other"] = best.to_numpy()
    table["covered_by"] = overlap.idxmax(axis=1).where(best > 0).to_numpy()
    return table


def _coverage_overlap(F: np.ndarray) -> pd.DataFrame:
    """overlap[A, B] = A 失败的芯片里，B 也失败的比例（B 对 A 的"覆盖"）。对角线置 0。"""
    both = F.T.astype(int) @ F.astype(int)
    fails = np.diag(both).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = both / fails[:, None]
    np.fill_diagonal(frac, 0.0)
    return pd.DataFrame(np.nan_to_num(frac))


def redundancy_candidates(
    eff: pd.DataFrame,
    min_covered: float = 0.95,
    max_unique_share: float = 0.05,
    min_fails: int = 20,
) -> pd.DataFrame:
    """失败几乎都被别的测试项同时抓到、自己几乎没有独有失败的测试项。"""
    hit = eff[
        (eff["fails"] >= min_fails)
        & (eff["max_covered_by_other"] >= min_covered)
        & (eff["unique_share"].fillna(0) <= max_unique_share)
    ]
    return hit.sort_values("time_ms", ascending=False).reset_index(drop=True)


def expected_time(F: np.ndarray, times: np.ndarray, order: list[int]) -> float:
    """遇失败即停：每颗芯片花到第一个失败项为止，全部通过则花完全部时间。返回平均每颗 ms。"""
    cum = np.cumsum(times[order])
    fo = F[:, order]
    stop = np.where(fo.any(axis=1), fo.argmax(axis=1), len(order) - 1)
    return float(cum[stop].mean())


def greedy_order(F: np.ndarray, times: np.ndarray, pinned: list[int]) -> list[int]:
    """贪心排序：每一步在"仍存活的芯片"里选 (条件失败率 / 耗时) 最大的测试项。pinned 固定在最前（如接触测试）。"""
    order = list(pinned)
    alive = ~F[:, order].any(axis=1) if order else np.ones(len(F), dtype=bool)
    remaining = [j for j in range(F.shape[1]) if j not in order]
    while remaining:
        score = [F[alive, j].mean() / times[j] if alive.any() else 0.0 for j in remaining]
        pick = remaining[int(np.argmax(score))]
        order.append(pick)
        remaining.remove(pick)
        alive &= ~F[:, pick]
    return order


def order_optimization(
    first: pd.DataFrame, tests: pd.DataFrame, pinned_prefix: str = "CONT_"
) -> dict:
    F = fail_matrix(first, tests)
    times = tests["time_ms"].to_numpy(dtype=float)
    names = list(tests["name"])
    pinned = [i for i, n in enumerate(names) if n.startswith(pinned_prefix)]
    current = expected_time(F, times, list(range(len(names))))
    best_order = greedy_order(F, times, pinned)
    best = expected_time(F, times, best_order)
    return {
        "full_time_ms": float(times.sum()),
        "current_ms": current,
        "optimized_ms": best,
        "saving": 1 - best / current,
        "optimized_order": [names[i] for i in best_order],
    }


def drop_candidates_saving(first: pd.DataFrame, tests: pd.DataFrame, drop: list[str]) -> dict:
    """如果去掉这些测试项，平均每颗省多少时间，以及会多放过多少失败芯片（只有它们抓到的）。"""
    names = list(tests["name"])
    F = fail_matrix(first, tests)
    keep = [i for i, n in enumerate(names) if n not in drop]
    times = tests["time_ms"].to_numpy(dtype=float)
    before = expected_time(F, times, list(range(len(names))))
    after = expected_time(F, times, keep)
    escapes = int((F.any(axis=1) & ~F[:, keep].any(axis=1)).sum())
    return {"dropped": drop, "saving": 1 - after / before, "new_escapes": escapes, "dies": len(F)}
