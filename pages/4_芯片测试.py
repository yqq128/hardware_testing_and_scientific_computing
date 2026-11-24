# 芯片测试方向（华为芯片测试工程）：对 ATE 日志做良率、归因、离群值和测试效率分析。
# 与前三页同样是"读数据 → 画页面"的直线脚本；每次操作都会从头重跑，重活由缓存兜住。

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from chiptest import plots
from chiptest.data import continuous_tests, first_pass
from chiptest.streamlit_support import load_or_stop

st.set_page_config(page_title="芯片测试", layout="wide")
st.title("芯片测试数据分析")
st.caption(
    "ATE 日志与芯片均为模拟数据，仅用于练习，不代表任何真实芯片或产线。"
    "统计口径：每颗芯片只取首测（pass 1），复测单独统计。"
)

clean, tests, report = load_or_stop()
first = first_pass(clean)
plots.use_style()


def show(fig) -> None:
    # st.pyplot 只负责渲染；matplotlib 的 Figure 不会自动释放，页面反复重跑会越积越多，所以画完就关。
    st.pyplot(fig)
    plt.close(fig)


LABELS = {
    "hbin": "硬 bin",
    "hbin_name": "名称",
    "dies": "芯片数",
    "share_of_fails": "占失败比例",
    "cumulative": "累计",
    "station": "测试机",
    "delta_sigma": "均值偏移 (σ)",
    "std_ratio": "离散度比",
    "ring": "环（1=圆心）",
    "fpy": "首测良率",
    "test": "测试项",
    "time_ms": "耗时 ms",
    "time_share": "时间占比",
    "fails": "失败数",
    "unique_fails": "仅此项失败",
    "max_covered_by_other": "与其他项最大重叠",
    "covered_by": "重叠对象",
    "saving": "节省时间",
    "new_escapes": "只被它抓到的失败芯片",
    "flagged": "标出数",
    "true_weak_flagged": "其中真隐患",
    "precision": "精度",
    "recall": "召回",
    "weak_total": "隐患总数",
}


def show_table(
    df: pd.DataFrame, formats: dict[str, str] | None = None, index: bool = False
) -> None:
    # 列名改成中文后再套格式，所以 formats 的键写原列名，这里统一翻译。
    formats = {LABELS.get(k, k): v for k, v in (formats or {}).items()}
    shown = df.rename(columns=LABELS)
    # st.dataframe 的前端会把空值显示成 None，不认 Styler 的 na_rep，所以含空值的列先转成文本。
    for col in shown.columns[shown.isna().any()]:
        fmt = formats.pop(col, "{}")
        shown[col] = ["-" if pd.isna(v) else fmt.format(v) for v in shown[col]]
    st.dataframe(shown.style.format(formats), hide_index=not index, width="stretch")


o = report.overall
cols = st.columns(4)
cols[0].metric("首测芯片数", f"{o['dies']:,}")
cols[1].metric("首测良率 FPY", f"{o['fpy']:.2%}")
cols[2].metric("复测后良率", f"{o['final_yield']:.2%}")
cols[3].metric(
    "复测恢复率", f"{o['recovery_rate']:.1%}", help="首测失败且被复测的芯片中，最终通过的比例。"
)

st.subheader("1. 各 lot 良率与失败 bin")
left, right = st.columns(2)
with left:
    show(plots.yield_by_lot(report))
with right:
    show_table(report.bins, {"share_of_fails": "{:.1%}", "cumulative": "{:.1%}"})

st.subheader("2. 是测试硬件的问题，还是芯片的问题？")
st.caption(
    "同一个 lot 的晶圆被分到不同测试机；某个测试机或 site 的良率明显低于同 lot 其余部分，就更像测试系统的问题。"
)
show(plots.fpy_heatmap(first))
for s in report.suspects:
    where = f"{s.lot_id} / {s.station}" + (f" / site{s.site}" if s.site else "")
    g = s.signature
    with st.expander(f"{where}：{s.kind}，主要失败项 {g['top_test']}", expanded=True):
        st.write(
            f"FPY {g['fpy']:.1%}（同 lot 其余 {g['rest_fpy']:.1%}）；"
            f"{g['top_test']} 失败率 {g['top_test_fail_rate']:.1%}（其余 {g['top_test_rest_fail_rate']:.1%}）；"
            f"复测恢复率 {g['recovery_rate']:.0%}（其余 {g['rest_recovery_rate']:.0%}）。"
        )
        st.info(s.hypothesis)

st.subheader("3. 同 lot 内跨测试机的参数分布")
st.caption(
    "电源噪声表现为离散度变大而均值不变；时序/校准偏移表现为整体平移。默认选中的是 L06 的 VMIN。"
)
cont = continuous_tests(tests)
pick = st.columns(2)
lot = pick[0].selectbox("lot", sorted(first["lot_id"].unique()), index=5)
test = pick[1].selectbox("测试项", cont, index=cont.index("VMIN"))
fig, ax = plt.subplots(figsize=(6, 3.8))
plots.station_box(ax, first, lot, test)
show(fig)
effects = report.station_effects
view = effects[(effects["lot_id"] == lot) & (effects["test"] == test)]
show_table(
    view[["station", "dies", "delta_sigma", "std_ratio"]],
    {"delta_sigma": "{:+.2f}", "std_ratio": "{:.2f}"},
)

st.subheader("4. 晶圆图：边缘损失与划伤集群")
wafer_cols = st.columns(2)
w_lot = wafer_cols[0].selectbox(
    "晶圆所属 lot",
    sorted(first["lot_id"].unique()),
    index=2,
    key="wafer_lot",
)
wafers = sorted(first.loc[first["lot_id"] == w_lot, "wafer_id"].unique())
w_id = wafer_cols[1].selectbox("晶圆", wafers, index=min(6, len(wafers) - 1))
map_col, info_col = st.columns([1, 1])
with map_col:
    show(plots.single_wafer_map(first, w_lot, w_id))
with info_col:
    scan = report.wafers.set_index(["lot_id", "wafer_id"]).loc[(w_lot, w_id)]
    st.write(
        f"最大相连失败集群 {int(scan['largest_cluster'])} 颗，随机预期约 {scan['expected_cluster']:.1f}，"
        f"置换检验 p = {scan['perm_p']:.3f}。"
        + ("**超出随机预期，疑似划伤/局部污染。**" if scan["flagged"] else "未超出随机预期。")
    )
    st.caption("等面积分环的首测良率（1 = 圆心，5 = 边缘）")
    show_table(report.radial, {"fpy": "{:.2%}"})

st.subheader("5. 离群值筛选（PAT）：合格芯片里找可疑的")
st.write(
    "只对已通过全部规格的芯片做。单看限值抓不到的弱芯片，往往表现为"
    "**相对其速度和电流而言漏电偏大**，所以用残差而不是漏电本身。"
)
rows = {"静态限值（6σ）": report.pat["static_k6"]}
rows |= {f"残差法 {k}": v for k, v in report.pat["residual"].items()}
show_table(
    pd.DataFrame(rows).T.astype(float),
    {"flagged": "{:.0f}", "true_weak_flagged": "{:.0f}", "weak_total": "{:.0f}"}
    | {"precision": "{:.1%}", "recall": "{:.1%}"},
    index=True,
)
show(plots.pat_residual(report))
if report.evaluation is None:
    st.info("没有 truth.csv：无法计算精度和召回。真实产线只能靠可靠性试验与客退来验证。")
else:
    with st.expander("仅模拟数据可得：过杀与漏测（真实产线没有）"):
        ev = report.evaluation["overall"]
        st.write(
            f"过杀：首测 {ev['overkill_rate_first']:.2%} → 复测后 {ev['overkill_rate_final']:.2%}；"
            f"漏测（本来不合格却通过）：首测 {int(ev['escape_first'])} 颗 → 复测后 {int(ev['escape_final'])} 颗。"
            "复测挽回了误杀，同时也让一部分边缘的不合格品通过。"
        )

st.subheader("6. 测试效率：时间花在哪、哪些项冗余")
show(plots.efficiency_fig(report))
show_table(
    report.efficiency[
        ["test", "time_ms", "time_share", "fails", "unique_fails", "max_covered_by_other"]
        + ["covered_by"]
    ],
    {"time_ms": "{:.0f}", "time_share": "{:.1%}", "max_covered_by_other": "{:.1%}"},
)
r = report.order
st.write(
    f"遇失败即停口径下，当前顺序平均每颗 {r['current_ms']:.0f} ms，贪心重排后 {r['optimized_ms']:.0f} ms，"
    f"只省 {r['saving']:.1%}。杠杆在耗时最长的测试项，而不在顺序。"
)
if len(report.drop_what_if):
    st.caption(
        "冗余候选：去掉其中一项的假设推演（候选清单，不是决定；必须回到故障模型和覆盖率确认）"
    )
    show_table(report.drop_what_if[["test", "saving", "new_escapes", "dies"]], {"saving": "{:.1%}"})

st.subheader("结论摘要")
st.markdown(report.summary_markdown())
