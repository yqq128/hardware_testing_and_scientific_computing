"""AOI 误报异常原因定位与多维诊断下钻模块。

在产线发生误报率（FPR）突增时，质量工程师需要快速判断根因归属：
1. 【单设备硬件故障】：某台机台自身的光源衰减、相机镜头污染或位姿偏移；
   - 诊断手段：三维对照（同机历史基线 vs 同日同批次平行设备），通过 Fisher 精确检验量化显著性，
     结合小时时序分桶追踪光学参数（如亮度 brightness）漂移。
2. 【来料/批次工艺缺陷】：整批工件表面处理异常（如阳极氧化反光、油污）；
   - 诊断手段：基于二项分布 Z 分数筛选异常批次，并跨设备核实是否全线机台同批中招。
3. 【人因/班次效应】：夜班操作、光照环境或不同组别保养习惯差异；
   - 诊断手段：班次间 FPR 卡方独立性检验（Chi-square test）。
"""

import textwrap
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from aoi.data import first_pass, with_reference
from aoi.metrics import overall_summary, rollup

SHIFT_ORDER = ["day", "swing", "night"]
# 批次异常筛选的 Z 分数阈值（基于正态分布单侧检验，4 个标准差代表极罕见随机波动）
Z_THRESHOLD = 4.0

# 展示用的列格式，Notebook 与 Streamlit 页面共用
RATE_FORMAT = {
    "alarm_rate": "{:.2%}",
    "fpr": "{:.2%}",
    "recall": "{:.2%}",
    "fp_share_of_alarms": "{:.1%}",
}


def _safe_ratio(a: float, b: float) -> float:
    """浮点数零除保护除法。"""
    return a / b if b else float("nan")


def machine_day_table(verified_first: pd.DataFrame) -> pd.DataFrame:
    """按【设备 x 日期】切分汇总误报指标，用于构建空间-时间二维监控热力图。"""
    return rollup(verified_first, ["machine_id", "date"])


def find_peak(machine_day: pd.DataFrame) -> tuple[str, pd.Timestamp]:
    """定位全局误报率（FPR）最高的异常设备与异常发生日期。"""
    peak = machine_day.nlargest(1, "fpr").iloc[0]
    return peak["machine_id"], peak["date"]


@dataclass
class MachineDayAnomaly:
    """单机台单日异常深度下钻分析报告数据类。"""

    machine: str
    day: pd.Timestamp
    batches: list[str]
    peer_machines: list[str]
    target_fp: int
    target_good: int
    peer_fp: int
    peer_good: int
    fisher_p: float
    comparison: pd.DataFrame
    by_shift: pd.DataFrame
    target_bins: pd.DataFrame
    peer_bins: pd.DataFrame
    brightness_bins: pd.Series
    brightness_compare: pd.DataFrame

    @property
    def target_fpr(self) -> float:
        """异常设备当天的良品误报率。"""
        return _safe_ratio(self.target_fp, self.target_good)

    @property
    def peer_fpr(self) -> float:
        """平行对照设备同批次当天的良品误报率。"""
        return _safe_ratio(self.peer_fp, self.peer_good)


def _by_time_bin(frame: pd.DataFrame) -> pd.DataFrame:
    """按 3 小时时间窗口聚合，平滑小时波动以观察日内趋势演变。"""
    return rollup(frame.assign(hour_bin=(frame["hour"] // 3) * 3), "hour_bin")


def drill_down(verified_first: pd.DataFrame, machine: str, day: pd.Timestamp) -> MachineDayAnomaly:
    """针对特定【设备 x 日期】峰值进行多因子因果下钻。

    对照实验设计（Controlled Experiment）：
    1. 目标组：异常机台当天的检测样本；
    2. 平行对照组（控制来料变量）：当天在同产线加工相同批次（batch_id）的其他设备；
    3. 历史对照组（控制机台固有属性）：该异常机台在其他所有正常日期的表现。

    统计推断：
    采用 Fisher 精确检验（Fisher's Exact Test）验证目标机台与对照机台误报率差异的显著性。
    在工业排障中，单日缺陷频数较小，Fisher 检验比普通卡方检验更准确。
    """
    day_frame = verified_first[verified_first["date"] == day]
    target = day_frame[day_frame["machine_id"] == machine]
    if target.empty:
        raise ValueError(f"{machine} 在 {day:%Y-%m-%d} 没有有基准结果的首检记录")
    batches = list(target["batch_id"].unique())
    peer = day_frame[day_frame["batch_id"].isin(batches) & (day_frame["machine_id"] != machine)]
    peer_machines = sorted(peer["machine_id"].unique())
    baseline = verified_first[
        (verified_first["machine_id"] == machine) & (verified_first["date"] != day)
    ]

    comparison = rollup(
        pd.concat(
            [
                target.assign(group=f"{machine} on anomaly day"),
                peer.assign(group=f"{'/'.join(peer_machines)} same batch same day"),
                baseline.assign(group=f"{machine} other days"),
            ]
        ),
        "group",
    )

    target_good = int(target["is_actual_good"].sum())
    target_fp = int(target["is_false_positive"].sum())
    peer_good = int(peer["is_actual_good"].sum())
    peer_fp = int(peer["is_false_positive"].sum())
    _, fisher_p = stats.fisher_exact(
        [[target_fp, target_good - target_fp], [peer_fp, peer_good - peer_fp]]
    )

    by_shift = (
        rollup(target, "shift").set_index("shift").reindex(["night", "day", "swing"]).reset_index()
    )
    brightness_bins = target.groupby((target["hour"] // 3) * 3)["brightness"].mean()
    brightness_compare = pd.DataFrame(
        {
            "anomaly day": target["brightness"].describe(),
            "other days": baseline["brightness"].describe(),
        }
    ).T[["count", "mean", "std", "min"]]

    return MachineDayAnomaly(
        machine=machine,
        day=day,
        batches=batches,
        peer_machines=peer_machines,
        target_fp=target_fp,
        target_good=target_good,
        peer_fp=peer_fp,
        peer_good=peer_good,
        fisher_p=float(fisher_p),
        comparison=comparison,
        by_shift=by_shift,
        target_bins=_by_time_bin(target),
        peer_bins=_by_time_bin(peer),
        brightness_bins=brightness_bins,
        brightness_compare=brightness_compare,
    )


def batch_scores(verified_first: pd.DataFrame, overall_fpr: float) -> pd.DataFrame:
    """计算各生产批次 FPR 相对全局基线 FPR 的 Z 分数。

    假设在正常受控状态下，良品误报事件服从二项分布 B(n, p0)，其中 p0 为总体 FPR。
    标准差估算为 sqrt(p0 * (1 - p0) / n_good)。
    Z 分数度量了批次误报偏离全局期望的标准差倍数。
    """
    p0 = overall_fpr
    by_batch = rollup(verified_first, "batch_id")
    return by_batch.assign(z=(by_batch["fpr"] - p0) / np.sqrt(p0 * (1 - p0) / by_batch["good"]))


def flag_batches(scores: pd.DataFrame, z_threshold: float = Z_THRESHOLD) -> pd.DataFrame:
    """筛选出 Z 分数超过显著性阈值的严重异常批次列表。"""
    return scores[scores["z"] > z_threshold].sort_values("z", ascending=False)


def flagged_by_machine(verified_first: pd.DataFrame, flagged: pd.DataFrame) -> pd.DataFrame:
    """透视异常批次在各个机台上的分布情况，协助区分是“单机污染”还是“全线共性”。"""
    return rollup(
        verified_first[verified_first["batch_id"].isin(flagged["batch_id"])],
        ["batch_id", "machine_id"],
    )


def pick_cross_machine_batch(flagged: pd.DataFrame, exclude_batches: list[str]) -> str | None:
    """找出首个排除已知单设备故障批次之后的跨设备高误报批次。

    若某个批次在多个机台上同时出现高误报，则极大概率是来料物理属性（如表面反光）引发的共性问题。
    """
    remaining = flagged[~flagged["batch_id"].isin(exclude_batches)]
    return remaining["batch_id"].iloc[0] if len(remaining) else None


def batch_feature_profile(verified_first: pd.DataFrame, batch_id: str) -> pd.DataFrame:
    """提取可疑批次的物理光学特征画像（亮度、对比度）。

    与“同线其他日期”和“同日其他线”进行横向切片对比，锁定是否为工艺环境或来料批次的光学异常。
    """
    batch = verified_first[verified_first["batch_id"] == batch_id]
    line = batch["line_id"].iloc[0]
    day = batch["date"].iloc[0]
    same_line_other_days = verified_first[
        (verified_first["line_id"] == line) & (verified_first["date"] != day)
    ]
    other_lines_same_day = verified_first[
        (verified_first["line_id"] != line) & (verified_first["date"] == day)
    ]
    columns = ["brightness", "contrast"]
    return pd.DataFrame(
        {
            f"{batch_id} ({line}, {day:%m-%d})": batch[columns].mean(),
            f"{line} other days": same_line_other_days[columns].mean(),
            "Other lines same day": other_lines_same_day[columns].mean(),
        }
    ).T


def shift_effect(verified_first: pd.DataFrame) -> dict:
    """通过卡方列联表检验评估夜班（night）与非夜班的良品误报率差异。"""
    night = verified_first[verified_first["shift"] == "night"]
    others = verified_first[verified_first["shift"] != "night"]

    def cells(frame: pd.DataFrame) -> list[int]:
        fp = int(frame["is_false_positive"].sum())
        return [fp, int(frame["is_actual_good"].sum()) - fp]

    _, p_value, _, _ = stats.chi2_contingency([cells(night), cells(others)])
    return {
        "night_fpr": float(night["is_false_positive"].sum() / night["is_actual_good"].sum()),
        "others_fpr": float(others["is_false_positive"].sum() / others["is_actual_good"].sum()),
        "p_value": float(p_value),
    }


def defect_type_recall(verified_first: pd.DataFrame) -> pd.DataFrame:
    """按真实缺陷类型（划伤 scratch、凹坑 dent 等）统计 AOI 报警召回能力。"""
    return rollup(verified_first[verified_first["is_actual_defect"]], "defect_type")


@dataclass
class DiagnosisReport:
    """产线质量根因诊断全景综合报告对象。"""

    overall: dict
    machine_day: pd.DataFrame
    anomaly: MachineDayAnomaly
    batch_scores: pd.DataFrame
    flagged: pd.DataFrame
    flagged_by_machine: pd.DataFrame
    cross_machine_batch: str | None
    cross_machine_profile: pd.DataFrame | None
    shift: dict
    type_recall: pd.DataFrame

    def summary_markdown(self) -> str:
        """生成供业务评审与工程周会汇报的结构化 Markdown 结论。"""
        o, a, s = self.overall, self.anomaly, self.shift
        text = f"""
        **整体（首检，有参考判定）**：Recall {o["recall"]:.1%}，良品误报率 {o["fpr"]:.2%}，
        Precision {o["precision"]:.1%}，即人工看到的报警中约 {1 - o["precision"]:.0%} 是误报。

        **线索 1（设备）**：{a.machine} 在 {a.day:%Y-%m-%d} 良品误报率 {a.target_fpr:.1%}，
        同批次同线的 {"/".join(a.peer_machines)} 为 {a.peer_fpr:.1%}（Fisher p = {a.fisher_p:.1e}），
        异常集中在后半天并伴随亮度下降。建议核查该设备光源状态与维护记录。

        **线索 2（批次）**：{self.cross_machine_batch or "n/a"} 整个批次的所有设备误报率都升高，亮度偏高、对比度偏低，
        更像批次/来源（反光）问题，应联系工艺和来料质量确认。

        **线索 3（班次）**：夜班误报率 {s["night_fpr"]:.2%} 高于其他班次 {s["others_fpr"]:.2%}，原因待现场确认。

        **局限**：数据为模拟数据，且每条记录都有参考判定；真实产线通常只有报警件被复判，漏检需要独立抽检估计。
        """
        return textwrap.dedent(text)


def build_report(clean: pd.DataFrame) -> DiagnosisReport:
    """端到端构建产线根因诊断报告的工厂函数。"""
    verified_first = with_reference(first_pass(clean))
    overall = overall_summary(verified_first)
    machine_day = machine_day_table(verified_first)
    machine, day = find_peak(machine_day)
    anomaly = drill_down(verified_first, machine, day)

    scores = batch_scores(verified_first, overall["fpr"])
    flagged = flag_batches(scores)
    cross_batch = pick_cross_machine_batch(flagged, anomaly.batches)

    return DiagnosisReport(
        overall=overall,
        machine_day=machine_day,
        anomaly=anomaly,
        batch_scores=scores,
        flagged=flagged,
        flagged_by_machine=flagged_by_machine(verified_first, flagged),
        cross_machine_batch=cross_batch,
        cross_machine_profile=(
            batch_feature_profile(verified_first, cross_batch) if cross_batch else None
        ),
        shift=shift_effect(verified_first),
        type_recall=defect_type_recall(verified_first),
    )
