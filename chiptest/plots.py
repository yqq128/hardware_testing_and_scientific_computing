"""芯片测试分析图表（7 张）。风格与保存方式复用 aoi.plots。

c01_yield_by_lot             各 lot 的首测良率与复测后良率（差距 = 复测挽回）
c02_fpy_heatmap_station_site (lot) x (测试机-site) 的 FPY 热力图：先找最异常的格子
c03_wafer_maps               三张晶圆图：划伤集群、接触不良 site、正常对照
c04_radial_yield             等面积分环 FPY：边缘良率损失
c05_station_parametrics      同一 lot 内各测试机的参数分布：电源噪声（σ 变大）与时序偏移（均值平移）
c06_pat_residual             PAT 残差：合格芯片里"相对其速度漏电偏大"的离群点
c07_test_efficiency          测试时间占比与独有失败数：测试时间花在哪
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from aoi.plots import save_figure, use_style
from chiptest.data import first_pass
from chiptest.parametric import LOG_TESTS
from chiptest.report import PAT_K, ChipReport

PASS_COLOR = "#D9D9D9"


def yield_by_lot(report: ChipReport) -> Figure:
    t = report.yield_by_lot
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, t["fpy"] * 100, 0.4, label="First-pass yield", color="#4C78A8")
    ax.bar(x + 0.2, t["final_yield"] * 100, 0.4, label="After retest", color="#72B7B2")
    ax.set_xticks(x, t["lot_id"])
    ax.set_ylim(min(t["fpy"].min() * 100 - 3, 85), 100)
    ax.set_ylabel("Yield (%)")
    ax.set_title("Yield by lot: retest recovers some loss but not all")
    ax.legend(loc="upper right")
    return fig


def fpy_heatmap(first: pd.DataFrame) -> Figure:
    grid = (
        first.assign(cell=first["station"] + "-s" + first["site"].astype(str))
        .pivot_table(index="lot_id", columns="cell", values="tester_pass", aggfunc="mean")
        .mul(100)
    )
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(
        grid,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        vmin=60,
        vmax=96,
        ax=ax,
        cbar_kws={"label": "FPY (%)"},
    )
    ax.set_title("First-pass yield by lot x (station, site)")
    ax.set_xlabel("")
    return fig


def _bin_colors(first: pd.DataFrame) -> dict[int, tuple]:
    """按 bin 号固定配色：同一个 bin 在所有晶圆图里颜色一致。"""
    bins = sorted(first.loc[~first["tester_pass"], "hbin"].unique())
    return dict(zip(bins, sns.color_palette("tab10", len(bins)), strict=True))


def _wafer_ax(ax, wafer_df: pd.DataFrame, title: str, colors: dict[int, tuple]) -> None:
    color = [
        PASS_COLOR if ok else colors[b]
        for ok, b in zip(wafer_df["tester_pass"], wafer_df["hbin"], strict=True)
    ]
    ax.scatter(wafer_df["x"], wafer_df["y"], c=color, marker="s", s=34)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def wafer_maps(first: pd.DataFrame, report: ChipReport) -> Figure:
    picks = []
    if len(report.clusters):
        c = report.clusters.iloc[0]
        picks.append((c["lot_id"], c["wafer_id"], "scratch cluster"))
    site_hot = [s for s in report.suspects if s.kind == "site hotspot"]
    if site_hot:
        s = site_hot[0]
        sub = first[(first["lot_id"] == s.lot_id) & (first["station"] == s.station)]
        wafer_id = sub.groupby("wafer_id")["tester_pass"].mean().idxmin()
        picks.append((s.lot_id, wafer_id, f"contact issue, {s.station} site{s.site}"))
    best = first.groupby(["lot_id", "wafer_id"])["tester_pass"].mean().idxmax()
    picks.append((*best, "healthy reference"))

    colors = _bin_colors(first)
    names = first.drop_duplicates("hbin").set_index("hbin")["hbin_name"]
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 4.8))
    for ax, (lot, wafer_id, label) in zip(np.atleast_1d(axes), picks, strict=True):
        w = first[(first["lot_id"] == lot) & (first["wafer_id"] == wafer_id)]
        _wafer_ax(ax, w, f"{lot} W{wafer_id}: {label}\nFPY {w['tester_pass'].mean():.1%}", colors)
    handles = [plt.Line2D([], [], marker="s", ls="", color=PASS_COLOR, label="PASS")]
    handles += [
        plt.Line2D([], [], marker="s", ls="", color=c, label=names[b]) for b, c in colors.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    return fig


def radial_yield(report: ChipReport) -> Figure:
    t = report.radial
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(t["ring"].astype(str), t["fpy"] * 100, color="#F58518")
    ax.set_ylim(min(t["fpy"].min() * 100 - 3, 85), 100)
    ax.set_xlabel("Ring (1 = center, 5 = edge, equal area)")
    ax.set_ylabel("FPY (%)")
    ax.set_title("Radial yield: edge loss")
    return fig


def single_wafer_map(first: pd.DataFrame, lot: str, wafer_id: str) -> Figure:
    """单片晶圆的 bin 图，配色与 c03 一致。"""
    colors = _bin_colors(first)
    w = first[(first["lot_id"] == lot) & (first["wafer_id"] == wafer_id)]
    fig, ax = plt.subplots(figsize=(5, 5))
    _wafer_ax(ax, w, f"{lot} W{wafer_id}  FPY {w['tester_pass'].mean():.1%}", colors)
    names = first.drop_duplicates("hbin").set_index("hbin")["hbin_name"]
    handles = [plt.Line2D([], [], marker="s", ls="", color=PASS_COLOR, label="PASS")]
    handles += [
        plt.Line2D([], [], marker="s", ls="", color=c, label=names[b]) for b, c in colors.items()
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    return fig


def station_box(ax, first: pd.DataFrame, lot: str, test: str, title: str = "") -> None:
    """同一 lot 内各测试机的参数分布（箱线图，裁掉最极端的 1% 以便看清主体）。"""
    sub = first[first["lot_id"] == lot]
    values = np.log(sub[test]) if test in LOG_TESTS else sub[test]
    lo, hi = values.quantile([0.005, 0.995])
    stations = sorted(sub["station"].unique())
    groups = [values[sub["station"] == s].dropna().clip(lo, hi) for s in stations]
    ax.boxplot(
        groups,
        tick_labels=stations,
        patch_artist=True,
        boxprops={"facecolor": "#9ECAE9"},
        medianprops={"color": "#333"},
        flierprops={"markersize": 2},
    )
    ax.set_title(f"{lot} {test}" + (f"\n{title}" if title else ""), fontsize=10)
    ax.set_ylabel(f"ln({test})" if test in LOG_TESTS else test)


def station_parametrics(first: pd.DataFrame, report: ChipReport) -> Figure:
    flags = report.station_flags
    panels = []
    noise = flags[flags["kind"] == "noise"]
    shift = flags[flags["kind"] == "shift"]
    if len(noise):
        r = noise.sort_values("std_ratio", ascending=False).iloc[0]
        panels.append((r["lot_id"], r["test"], f"noise: {r['station']} std x{r['std_ratio']:.1f}"))
    if len(shift):
        r = shift.loc[shift["delta_sigma"].abs().idxmax()]
        panels.append(
            (r["lot_id"], r["test"], f"shift: {r['station']} {r['delta_sigma']:+.1f} sigma")
        )
    fig, axes = plt.subplots(1, max(len(panels), 1), figsize=(5.5 * max(len(panels), 1), 4))
    for ax, (lot, test, label) in zip(np.atleast_1d(axes), panels, strict=True):
        station_box(ax, first, lot, test, label)
    return fig


def pat_residual(report: ChipReport) -> Figure:
    f = report.pat_frame.dropna(subset=["z_resid"])
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-5, max(f["z_resid"].max(), 8), 90)
    if "is_weak" in f:
        ax.hist(
            f.loc[~f["is_weak"], "z_resid"], bins=bins, color="#4C78A8", label="normal", log=True
        )
        weak = f.loc[f["is_weak"], "z_resid"]
        ax.hist(
            weak,
            bins=bins,
            color="#E45756",
            alpha=0.8,
            label="latent weak (simulation truth)",
            log=True,
        )
    else:
        ax.hist(f["z_resid"], bins=bins, color="#4C78A8", log=True)
    for k, style in zip(PAT_K, [":", "--"], strict=True):
        ax.axvline(k, color="k", ls=style, lw=1, label=f"threshold k={k}")
    ax.set_xlabel("Leakage residual (robust z, given speed and current)")
    ax.set_ylabel("Passing dies (log)")
    ax.set_title("PAT residual screen on dies that pass all limits")
    ax.legend()
    return fig


def efficiency_fig(report: ChipReport) -> Figure:
    e = report.efficiency.sort_values("time_ms", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    axes[0].barh(e["test"], e["time_share"] * 100, color="#4C78A8")
    axes[0].set_xlabel("Share of test time (%)")
    axes[0].invert_yaxis()
    axes[1].barh(e["test"], e["unique_fails"], color="#E45756")
    axes[1].set_xlabel("Unique fails (only this test catches the die)")
    fig.suptitle("Where test time goes vs what each test uniquely catches")
    return fig


def render_all(df: pd.DataFrame, report: ChipReport, out_dir: Path) -> list[Path]:
    use_style()
    first = first_pass(df)
    figures = {
        "c01_yield_by_lot.png": yield_by_lot(report),
        "c02_fpy_heatmap_station_site.png": fpy_heatmap(first),
        "c03_wafer_maps.png": wafer_maps(first, report),
        "c04_radial_yield.png": radial_yield(report),
        "c05_station_parametrics.png": station_parametrics(first, report),
        "c06_pat_residual.png": pat_residual(report),
        "c07_test_efficiency.png": efficiency_fig(report),
    }
    return [save_figure(fig, out_dir, name) for name, fig in figures.items()]
