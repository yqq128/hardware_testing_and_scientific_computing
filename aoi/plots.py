"""AOI 检测质量监控图表与评估可视化模块。

包含 9 组核心生产监控图表：
- 01_daily_alarms: 首检每日机检报警件数趋势；
- 02_daily_false_positives: 每日误报量（FP）与误报率（FPR）双轴走势，标注峰值日期；
- 03_fpr_by_machine: 各机台良品误报率横向柱状对比（带全线基准参考线）；
- 04_alarms_by_defect_type: 各缺陷类型报警中真缺陷与假阳性堆叠分布；
- 05_fpr_by_shift_and_batch: 班次误报率与 Top10 高误报批次柱状图；
- 06_fpr_heatmap_machine_day: 【设备 x 日期】空间与时间二维误报率热力图；
- 07_anomaly_drilldown: 异常机台小时级误报率与光源亮度（brightness）双轴下钻；
- 08_review_model_confusion: 二次复判模型测试集混淆矩阵热力图；
- 09_pr_curve_and_threshold_tradeoff: PR 曲线与工作量削减-召回率权衡双联图。
"""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve

from aoi.data import first_pass, with_reference
from aoi.diagnosis import DiagnosisReport, MachineDayAnomaly
from aoi.metrics import overall_summary, rollup
from aoi.review_model import ExperimentResult


def use_style() -> None:
    """统一配置全局绘图风格（Seaborn 白网格主题）。"""
    sns.set_theme(style="whitegrid")


def save_figure(fig: Figure, directory: Path, name: str) -> Path:
    """保存图表至指定目录，确保自动紧凑排版并及时释放图形内存句柄。"""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def daily_alarms(first: pd.DataFrame) -> Figure:
    """绘制首检每日机检报警总数柱状图（反映每日送检峰谷负荷）。"""
    daily = first.groupby("date").agg(
        inspections=("inspection_id", "size"), alarms=("is_alarm", "sum")
    )
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(daily.index, daily["alarms"], color="#4C78A8")
    ax.set_title("Daily AOI alarms (first pass)")
    ax.set_ylabel("Alarms")
    fig.autofmt_xdate()
    return fig


def daily_false_positives(verified_first: pd.DataFrame) -> Figure:
    """绘制每日误报绝对件数（FP）与良品误报率（FPR %）上下双子图，标注异常最高点。"""
    daily = rollup(verified_first, "date")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(daily["date"], daily["fp"], marker="o", color="#E45756")
    axes[0].set_ylabel("False positives")
    axes[0].set_title("Daily false positives and false positive rate")
    axes[1].plot(daily["date"], daily["fpr"] * 100, marker="o", color="#F58518")
    axes[1].set_ylabel("FPR (%)")
    for _, row in daily.nlargest(2, "fpr").iterrows():
        axes[1].annotate(
            row["date"].strftime("%m-%d"),
            (row["date"], row["fpr"] * 100),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )
    fig.autofmt_xdate()
    return fig


def fpr_by_machine(verified_first: pd.DataFrame) -> Figure:
    """绘制各机台 FPR 横向对比柱状图，叠加全线总体平均 FPR 基线。"""
    overall_fpr = overall_summary(verified_first)["fpr"]
    by_machine = rollup(verified_first, "machine_id")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.barplot(data=by_machine, x="machine_id", y=by_machine["fpr"] * 100, color="#4C78A8", ax=ax)
    ax.axhline(overall_fpr * 100, color="red", linestyle="--", label=f"Overall {overall_fpr:.2%}")
    ax.set_ylabel("FPR (%)")
    ax.set_title("False positive rate by machine")
    ax.legend()
    return fig


def alarms_by_defect_type(verified_first: pd.DataFrame) -> Figure:
    """绘制按 AOI 机报缺陷类型的堆叠柱状图，直观展现每种类型中的真缺陷与假阳性误报比例。"""
    alarm_types = (
        verified_first[verified_first["is_alarm"]]
        .groupby(["defect_type", "reference_result"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"NG": "True defect", "OK": "False positive"})
        .loc[:, ["True defect", "False positive"]]
    )
    fig, ax = plt.subplots(figsize=(7, 3.5))
    alarm_types.plot(kind="bar", stacked=True, color=["#54A24B", "#E45756"], ax=ax)
    ax.set_title("Alarms by AOI-reported defect type")
    ax.set_ylabel("Alarms")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=0)
    return fig


def fpr_by_shift_and_batch(verified_first: pd.DataFrame) -> Figure:
    """绘制班次误报率（日班/中班/夜班）与 Top 10 高误报生产批次左右双联图。"""
    overall_fpr = overall_summary(verified_first)["fpr"]
    by_shift = (
        rollup(verified_first, "shift")
        .set_index("shift")
        .loc[["day", "swing", "night"]]
        .reset_index()
    )
    by_batch = rollup(verified_first, "batch_id")

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    sns.barplot(data=by_shift, x="shift", y=by_shift["fpr"] * 100, color="#4C78A8", ax=axes[0])
    axes[0].axhline(overall_fpr * 100, color="red", linestyle="--")
    axes[0].set_ylabel("FPR (%)")
    axes[0].set_title("FPR by shift")

    top_batches = by_batch.nlargest(10, "fpr")
    sns.barplot(
        data=top_batches, x="batch_id", y=top_batches["fpr"] * 100, color="#E45756", ax=axes[1]
    )
    axes[1].axhline(overall_fpr * 100, color="red", linestyle="--")
    axes[1].set_ylabel("FPR (%)")
    axes[1].set_title("Top 10 batches by FPR")
    plt.setp(axes[1].get_xticklabels(), rotation=45)
    return fig


def machine_day_heatmap(machine_day: pd.DataFrame) -> Figure:
    """绘制【机台 x 日期】良品误报率二维热力图，快速锁定特定机台特定日期的突增红区。"""
    heat = machine_day.pivot(index="machine_id", columns="date", values="fpr") * 100
    fig, ax = plt.subplots(figsize=(13, 3.5))
    sns.heatmap(heat, cmap="Reds", cbar_kws={"label": "FPR (%)"}, ax=ax)
    ax.set_xticklabels([d.strftime("%m-%d") for d in heat.columns], rotation=90)
    ax.set_title("False positive rate: machine x date")
    ax.set_xlabel("")
    return fig


def anomaly_drilldown(anomaly: MachineDayAnomaly) -> Figure:
    """绘制异常机台小时级 FPR 趋势与平均成像亮度（brightness）双 Y 轴下钻图。

    直观展示异常机台在后半日误报率飙升与机台光源衰减（亮度骤降）的时序强相关性。
    """
    machine, day = anomaly.machine, anomaly.day
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(
        anomaly.target_bins["hour_bin"],
        anomaly.target_bins["fpr"] * 100,
        marker="o",
        color="#E45756",
        label=f"{machine} FPR",
    )
    ax.plot(
        anomaly.peer_bins["hour_bin"],
        anomaly.peer_bins["fpr"] * 100,
        marker="o",
        linestyle="--",
        color="#4C78A8",
        label="Peer FPR (same batch)",
    )
    ax.set_xlabel("Hour of day (3h bins)")
    ax.set_ylabel("FPR (%)")
    ax2 = ax.twinx()
    ax2.plot(
        anomaly.brightness_bins.index,
        anomaly.brightness_bins.values,
        marker="s",
        color="#54A24B",
        label=f"{machine} brightness",
    )
    ax2.set_ylabel("Mean brightness")
    ax2.grid(False)
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines[0] + lines2[0],
        lines[1] + lines2[1],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
    )
    ax.set_title(f"{machine} on {day:%m-%d}: false positives vs brightness")
    return fig


def review_model_confusion(result: ExperimentResult) -> Figure:
    """绘制二次复判模型在留出测试集上的二分类混淆矩阵热力图。"""
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    sns.heatmap(result.confusion_on_test(), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"Review model on test alarms (threshold {result.chosen_threshold})")
    return fig


def pr_curve_and_threshold_tradeoff(result: ExperimentResult) -> Figure:
    """绘制 PR 曲线（Precision-Recall）与阈值权衡曲线双联图。

    左图：对比复判模型（验证集/测试集）与原生 AOI 打分基线在报警样本上的 Precision-Recall 表现；
    右图：展示随着概率阈值从 0 扫描至 1 时，真实缺陷 Recall 与复检工作量削减率（Workload Reduction）的消长曲线。
    """
    valid, test, row = result.splits.valid, result.splits.test, result.test_row
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for label, frame, proba, style in [
        ("valid", valid, result.valid_proba, "-"),
        ("test", test, result.test_proba, "--"),
    ]:
        precision, recall, _ = precision_recall_curve(frame["y"], proba)
        axes[0].plot(recall, precision, style, label=f"{result.best_name} ({label})")
    precision, recall, _ = precision_recall_curve(test["y"], test["aoi_score"])
    axes[0].plot(recall, precision, ":", color="gray", label="AOI score only (test)")
    axes[0].scatter(
        [row["recall"]],
        [row["defects_caught"] / row["to_review"]],
        color="red",
        zorder=5,
        label="Chosen threshold (test)",
    )
    axes[0].set_xlabel("Recall of real defects")
    axes[0].set_ylabel("Precision among alarms sent to review")
    axes[0].set_title("Precision-recall on alarmed samples")
    axes[0].legend()

    for label, sweep_df, style in [
        ("valid", result.valid_sweep, "-"),
        ("test", result.test_sweep, "--"),
    ]:
        axes[1].plot(
            sweep_df["threshold"],
            sweep_df["recall"] * 100,
            style,
            color="#E45756",
            label=f"Recall ({label})",
        )
        axes[1].plot(
            sweep_df["threshold"],
            sweep_df["workload_reduction"] * 100,
            style,
            color="#4C78A8",
            label=f"Workload reduction ({label})",
        )
    axes[1].axvline(result.chosen_threshold, color="black", linestyle=":", label="Chosen threshold")
    axes[1].set_xlabel("Threshold on P(real defect)")
    axes[1].set_ylabel("%")
    axes[1].set_title("Threshold trade-off")
    axes[1].legend(fontsize=8)
    return fig


def render_all(
    clean: pd.DataFrame,
    report: DiagnosisReport,
    result: ExperimentResult,
    directory: Path,
) -> list[Path]:
    """批量生成并持久化与 Notebook 02、03 完全一致的 9 张质量看板图表。

    性能优化说明：
    每张图表渲染保存后立即调用 plt.close(fig) 释放 Matplotlib 后端图形对象与内存，
    避免在流水线批处理中产生内存堆积。
    """
    use_style()
    first = first_pass(clean)
    verified_first = with_reference(first)
    builders: list[tuple[str, Callable[[], Figure]]] = [
        ("01_daily_alarms.png", lambda: daily_alarms(first)),
        ("02_daily_false_positives.png", lambda: daily_false_positives(verified_first)),
        ("03_fpr_by_machine.png", lambda: fpr_by_machine(verified_first)),
        ("04_alarms_by_defect_type.png", lambda: alarms_by_defect_type(verified_first)),
        ("05_fpr_by_shift_and_batch.png", lambda: fpr_by_shift_and_batch(verified_first)),
        ("06_fpr_heatmap_machine_day.png", lambda: machine_day_heatmap(report.machine_day)),
        ("07_anomaly_drilldown.png", lambda: anomaly_drilldown(report.anomaly)),
        ("08_review_model_confusion.png", lambda: review_model_confusion(result)),
        ("09_pr_curve_and_threshold_tradeoff.png", lambda: pr_curve_and_threshold_tradeoff(result)),
    ]
    return [save_figure(build(), directory, name) for name, build in builders]
