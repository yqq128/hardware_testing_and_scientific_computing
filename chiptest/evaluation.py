"""仅模拟数据可得的"上帝视角"评估（真实产线没有 truth.csv）。

对应 AOI 项目里的 reference_result：
  过杀 overkill  = 本来合格的芯片被判失败（类比 AOI 误报 FP），拉低良率，浪费产能
  漏测 escape    = 本来不合格的芯片被判通过（类比 AOI 漏检 FN），进入客户端，用 DPPM 衡量
真实产线只能靠复测、可靠性试验、客退来间接估计漏测，没有办法像这里一样直接数。
"""

from pathlib import Path

import pandas as pd

from chiptest.data import DIE_KEY


def load_truth(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"wafer_id": str})


def attach_truth(first: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    return first.merge(truth, on=DIE_KEY, how="left", validate="one_to_one")


def overkill_escape(first_truth: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """首测与最终判定下的过杀率、漏测率。分母分别是"本来合格数""本来不合格数"。"""
    f = first_truth.assign(scope="all") if by is None else first_truth
    key = by or "scope"
    rows = []
    for k, g in f.groupby(key):
        n_good, n_bad = int(g["intrinsic_pass"].sum()), int((~g["intrinsic_pass"]).sum())
        rows.append(
            {
                key: k,
                "intrinsic_good": n_good,
                "overkill_first": int((~g["tester_pass"] & g["intrinsic_pass"]).sum()),
                "overkill_final": int((~g["final_ok"] & g["intrinsic_pass"]).sum()),
                "intrinsic_bad": n_bad,
                "escape_first": int((g["tester_pass"] & ~g["intrinsic_pass"]).sum()),
                "escape_final": int((g["final_ok"] & ~g["intrinsic_pass"]).sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["overkill_rate_first"] = out["overkill_first"] / out["intrinsic_good"]
    out["overkill_rate_final"] = out["overkill_final"] / out["intrinsic_good"]
    out["escape_rate_final"] = out["escape_final"] / out["intrinsic_bad"]
    return out


def pat_evaluation(flagged: pd.Series, weak: pd.Series) -> dict:
    """flagged：被 PAT 标出的布尔序列；weak：真值里的潜在弱芯片。返回精度、召回与标出数量。"""
    hit = int((flagged & weak).sum())
    n_flag = int(flagged.sum())
    return {
        "flagged": n_flag,
        "true_weak_flagged": hit,
        "precision": hit / n_flag if n_flag else float("nan"),
        "recall": hit / int(weak.sum()) if weak.sum() else float("nan"),
        "weak_total": int(weak.sum()),
    }
