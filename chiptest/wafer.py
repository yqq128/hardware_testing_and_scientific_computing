"""晶圆图分析：径向（边缘）良率损失，和失败芯片的空间集群（划伤、污染）。

集群的判定不能只看"有多大"：失败率高的晶圆随机也会拼出大块。所以用置换检验——
在同一环、同一 site 内随机打乱失败标签（保留径向趋势和 site 效应），看真实最大集群是否超出随机预期。
"""

import numpy as np
import pandas as pd
from scipy import ndimage

_EIGHT_NEIGHBORS = np.ones((3, 3), dtype=int)


def add_radius(first: pd.DataFrame) -> pd.DataFrame:
    """归一化半径 r（0 圆心，1 晶圆边缘）与等面积环编号（1 最内，5 最外）。"""
    radius = np.hypot(first["x"], first["y"])
    r = radius / (radius.max() + 0.5)
    ring = np.minimum((r**2 * 5).astype(int), 4) + 1
    return first.assign(r=r, ring=ring)


def radial_yield(first: pd.DataFrame) -> pd.DataFrame:
    """等面积分环的 FPY：每环芯片数大致相同，所以各环良率可直接比较。"""
    framed = add_radius(first)
    out = framed.groupby("ring").agg(dies=("x", "size"), fpy=("tester_pass", "mean"))
    return out.reset_index()


def _largest_cluster(grid: np.ndarray) -> int:
    labels, count = ndimage.label(grid, structure=_EIGHT_NEIGHBORS)
    return int(np.bincount(labels.ravel())[1:].max()) if count else 0


def scan_wafers(
    first: pd.DataFrame, n_perm: int = 500, min_size: int = 8, seed: int = 0
) -> pd.DataFrame:
    """逐片晶圆检测失败芯片的最大 8 邻接集群，并与置换基线比较。"""
    framed = add_radius(first)
    rng = np.random.default_rng(seed)
    extent = int(max(framed["x"].abs().max(), framed["y"].abs().max()))
    side = 2 * extent + 1
    rows = []
    for (lot, wafer), g in framed.groupby(["lot_id", "wafer_id"]):
        xs, ys = (g["x"] + extent).to_numpy(), (g["y"] + extent).to_numpy()
        fail = ~g["tester_pass"].to_numpy()
        strata = (g["ring"] * 10 + g["site"]).to_numpy()
        groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]

        def largest(flags, xs=xs, ys=ys):
            grid = np.zeros((side, side), dtype=bool)
            grid[ys, xs] = flags
            return _largest_cluster(grid)

        observed = largest(fail)
        perm = np.empty(n_perm, dtype=int)
        for k in range(n_perm):
            shuffled = fail.copy()
            for idx in groups:
                shuffled[idx] = rng.permutation(fail[idx])
            perm[k] = largest(shuffled)
        p_value = (1 + int((perm >= observed).sum())) / (n_perm + 1)
        rows.append((lot, wafer, int(fail.sum()), observed, float(perm.mean()), p_value))
    table = pd.DataFrame(
        rows,
        columns=["lot_id", "wafer_id", "fails", "largest_cluster", "expected_cluster", "perm_p"],
    )
    table["flagged"] = (table["largest_cluster"] >= min_size) & (
        table["perm_p"] <= 1 / (n_perm + 1)
    )
    return table.sort_values(
        ["flagged", "perm_p", "largest_cluster"], ascending=[False, True, False]
    )
