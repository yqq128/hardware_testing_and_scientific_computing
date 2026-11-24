"""把各个分析串成一份报告：Streamlit 页面、命令行流水线、图表共用同一个结果对象。"""

import json
import textwrap
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from chiptest import attribution, efficiency, evaluation, parametric, wafer, yield_analysis
from chiptest.data import continuous_tests, first_pass

# PAT 残差法：目标是漏电，用速度与工作电流做"同速度下预期漏电"的预测
PAT_TARGET = "IDD_STANDBY"
PAT_PREDICTORS = ["SCAN_FMAX", "IDD_ACTIVE"]
PAT_K = (3.5, 4.0)
PAT_STATIC_TESTS = ["IDD_STANDBY", "IDD_ACTIVE", "SCAN_FMAX", "VMIN"]


@dataclass
class Suspect:
    lot_id: str
    station: str
    site: int | None
    kind: str  # "site hotspot" | "station hotspot" | "noise"
    signature: dict
    hypothesis: str


@dataclass
class ChipReport:
    overall: dict
    yield_by_lot: pd.DataFrame
    bins: pd.DataFrame
    hotspots_cell: pd.DataFrame
    hotspots_station: pd.DataFrame
    suspects: list[Suspect]
    station_effects: pd.DataFrame
    station_flags: pd.DataFrame
    lot_flags: pd.DataFrame
    cpk: pd.DataFrame
    radial: pd.DataFrame
    wafers: pd.DataFrame
    efficiency: pd.DataFrame
    redundancy: pd.DataFrame
    drop_what_if: pd.DataFrame
    order: dict
    pat: dict
    pat_frame: pd.DataFrame
    evaluation: dict | None = None
    extra: dict = field(default_factory=dict)

    @property
    def clusters(self) -> pd.DataFrame:
        return self.wafers[self.wafers["flagged"]]

    def summary_json(self) -> str:
        def clean(o):
            if isinstance(o, dict):
                return {str(k): clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            if isinstance(o, (np.floating, float)):
                return None if np.isnan(o) else round(float(o), 6)
            if isinstance(o, (np.integer,)):
                return int(o)
            return o

        data = {
            "overall": self.overall,
            "suspects": [
                {
                    "lot": s.lot_id,
                    "station": s.station,
                    "site": s.site,
                    "kind": s.kind,
                    **s.signature,
                }
                for s in self.suspects
            ],
            "clusters": self.clusters[["lot_id", "wafer_id", "largest_cluster"]].to_dict("records"),
            "lot_process_shift": sorted(self.lot_flags["lot_id"].unique()),
            "pat": self.pat,
            "order_optimization": {k: v for k, v in self.order.items() if k != "optimized_order"},
            "evaluation": self.evaluation,
        }
        return json.dumps(clean(data), ensure_ascii=False, indent=2)

    def summary_markdown(self) -> str:
        o = self.overall
        lines = [
            f"**整体（首测）**：{o['dies']:,} 颗芯片，FPY {o['fpy']:.2%}，复测后最终良率 {o['final_yield']:.2%}，"
            f"首测失败的芯片被复测后恢复 {o['recovery_rate']:.1%}。",
            "",
        ]
        for i, s in enumerate(self.suspects, 1):
            g = s.signature
            where = f"{s.lot_id} / {s.station}" + (f" / site{s.site}" if s.site else "")
            lines += [
                f"**线索 {i}（{where}）**：FPY {g['fpy']:.1%}，同 lot 其余 {g['rest_fpy']:.1%}；"
                f"主要失败项 {g['top_test']}（{g['top_test_fail_rate']:.1%} vs {g['top_test_rest_fail_rate']:.1%}）；"
                f"复测恢复率 {g['recovery_rate']:.0%}（对照 {g['rest_recovery_rate']:.0%}）。{s.hypothesis}",
                "",
            ]
        if len(self.lot_flags):
            lots = "、".join(sorted(self.lot_flags["lot_id"].unique()))
            lines += [
                f"**批次**：{lots} 在多个独立测试项上同时偏移，更像整批工艺偏移（快角/慢角），而不是测试问题。",
                "",
            ]
        if len(self.clusters):
            c = self.clusters.iloc[0]
            lines += [
                f"**晶圆图**：{c['lot_id']} 的 {c['wafer_id']} 号晶圆有 {int(c['largest_cluster'])} 颗相连失败芯片，"
                f"远超随机预期（约 {c['expected_cluster']:.0f}），像划伤或局部污染。外圈良率低于内圈，见径向图。",
                "",
            ]
        r = self.order
        lines += [
            f"**测试效率**：改测试顺序只能省 {r['saving']:.1%}（遇失败即停口径）。真正的杠杆在耗时最长的测试项，"
            f"例如 {self.efficiency.sort_values('time_ms', ascending=False).iloc[0]['test']} 占 "
            f"{self.efficiency['time_share'].max():.0%} 的测试时间。",
            "",
            "**局限**：数据为模拟数据；日志规模、缺陷分布与测试机行为都是按教学需要设定的，"
            "不代表任何真实芯片或产线。真实场景没有 truth，过杀与漏测只能靠复测、可靠性试验与客退间接估计。",
        ]
        return textwrap.dedent("\n".join(lines))


def _mask(first: pd.DataFrame, lot: str, station: str, site: int | None) -> pd.Series:
    m = (first["lot_id"] == lot) & (first["station"] == station)
    return m & (first["site"] == site) if site is not None else m


def _suspects(first, tests, cells, stations, station_flags) -> list[Suspect]:
    names = list(tests["name"])
    found: list[Suspect] = []
    taken: set[tuple[str, str]] = set()

    if len(cells):
        top = cells.iloc[0]
        sig = attribution.signature(
            first, _mask(first, top["lot_id"], top["station"], int(top["site"])), names
        )
        found.append(
            Suspect(
                top["lot_id"],
                top["station"],
                int(top["site"]),
                "site hotspot",
                sig,
                attribution.hypothesis(sig, site_specific=True, noisy=False),
            )
        )
        taken.add((top["lot_id"], top["station"]))

    for _, row in stations.iterrows():
        if (row["lot_id"], row["station"]) in taken:
            continue
        sig = attribution.signature(first, _mask(first, row["lot_id"], row["station"], None), names)
        found.append(
            Suspect(
                row["lot_id"],
                row["station"],
                None,
                "station hotspot",
                sig,
                attribution.hypothesis(sig, site_specific=False, noisy=False),
            )
        )
        taken.add((row["lot_id"], row["station"]))

    for _, row in station_flags[station_flags["kind"] == "noise"].iterrows():
        if (row["lot_id"], row["station"]) in taken:
            continue
        sig = attribution.signature(first, _mask(first, row["lot_id"], row["station"], None), names)
        found.append(
            Suspect(
                row["lot_id"],
                row["station"],
                None,
                "noise",
                sig,
                attribution.hypothesis(sig, site_specific=False, noisy=True),
            )
        )
        taken.add((row["lot_id"], row["station"]))
    return found


def build_report(
    df: pd.DataFrame, tests: pd.DataFrame, truth: pd.DataFrame | None = None
) -> ChipReport:
    first = first_pass(df)
    cont = continuous_tests(tests)

    cells = attribution.flag_hotspots(
        attribution.peer_hotspots(first, ["lot_id", "station", "site"])
    )
    stations = attribution.flag_hotspots(attribution.peer_hotspots(first, ["lot_id", "station"]))
    effects = parametric.station_effects(first, cont)
    station_flags = parametric.flag_station_effects(effects)
    lot_flags = parametric.flag_lot_effects(parametric.lot_effects(first, cont))

    eff = efficiency.per_test_summary(first, tests)
    redundancy = efficiency.redundancy_candidates(eff)
    what_if = pd.DataFrame(
        [
            efficiency.drop_candidates_saving(first, tests, [t]) | {"test": t}
            for t in redundancy["test"]
        ]
    )

    first_t = evaluation.attach_truth(first, truth) if truth is not None else first
    ok = parametric.passing_complete(first_t, [PAT_TARGET, *PAT_PREDICTORS, *PAT_STATIC_TESTS])
    by = ["lot_id", "station"]
    z_resid = parametric.residual_zscores(ok, PAT_TARGET, PAT_PREDICTORS, by)
    z_static = parametric.static_zscores(ok, PAT_STATIC_TESTS, by)
    pat_frame = ok[["lot_id", "wafer_id", "x", "y", "station", PAT_TARGET, *PAT_PREDICTORS]].assign(
        z_resid=z_resid, z_static=z_static
    )
    pat = {
        "passing_dies": int(len(ok)),
        "residual": {f"k={k}": {"flagged": int((z_resid > k).sum())} for k in PAT_K},
        "static_k6": {"flagged": int((z_static > 6).sum())},
    }

    evaluation_result = None
    if truth is not None:
        weak = ok["is_weak"].astype(bool)
        pat_frame["is_weak"] = weak
        for k in PAT_K:
            pat["residual"][f"k={k}"] = evaluation.pat_evaluation(z_resid > k, weak)
        pat["static_k6"] = evaluation.pat_evaluation(z_static > 6, weak)
        evaluation_result = {
            "overall": evaluation.overkill_escape(first_t).iloc[0].to_dict(),
            "by_lot": evaluation.overkill_escape(first_t, "lot_id").to_dict("records"),
        }

    return ChipReport(
        overall=yield_analysis.overall(first),
        yield_by_lot=yield_analysis.yield_by(first, "lot_id"),
        bins=yield_analysis.bin_pareto(first),
        hotspots_cell=cells,
        hotspots_station=stations,
        suspects=_suspects(first, tests, cells, stations, station_flags),
        station_effects=effects,
        station_flags=station_flags,
        lot_flags=lot_flags,
        cpk=parametric.cpk_table(first, tests, by=None),
        radial=wafer.radial_yield(first),
        wafers=wafer.scan_wafers(first),
        efficiency=eff,
        redundancy=redundancy,
        drop_what_if=what_if,
        order=efficiency.order_optimization(first, tests),
        pat=pat,
        pat_frame=pat_frame,
        evaluation=evaluation_result,
    )
