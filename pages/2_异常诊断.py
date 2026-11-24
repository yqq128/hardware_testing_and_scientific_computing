# 对应 notebooks/aoi/02_aoi_analysis.ipynb 的诊断部分：
# 从“哪台设备哪天误报最高”一路下钻到批次、班次，并给出可验证的线索。

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from aoi import plots
from aoi.data import first_pass, with_reference
from aoi.diagnosis import RATE_FORMAT, drill_down
from aoi.streamlit_support import load_clean_or_stop, load_report_or_stop, pct

st.set_page_config(page_title="异常诊断", layout="wide")
st.title("误报异常诊断")
st.caption("统计口径：每件产品只取首检，且只统计有参考判定的记录（复测会重复计数、放大误报）。")

report = load_report_or_stop()
clean = load_clean_or_stop()
verified_first = with_reference(first_pass(clean))
plots.use_style()

o = report.overall
cols = st.columns(4)
cols[0].metric("Recall", pct(o["recall"]))
cols[1].metric("良品误报率 FPR", pct(o["fpr"], 2))
cols[2].metric("Precision", pct(o["precision"]))
cols[3].metric("报警中的误报占比", pct(o["fp_share_of_alarms"]))

st.subheader("1. 设备 × 日期：先找最异常的格子")
fig = plots.machine_day_heatmap(report.machine_day)
st.pyplot(fig)
plt.close(fig)

st.subheader("2. 下钻某台设备某一天")
# 默认选中热力图上最高的一格，也允许自己换，验证“是不是只有这一个点异常”。
machines = sorted(verified_first["machine_id"].unique())
days = sorted(pd.to_datetime(verified_first["date"].unique()))
pick_cols = st.columns(2)
machine = pick_cols[0].selectbox("设备", machines, index=machines.index(report.anomaly.machine))
day = pick_cols[1].selectbox(
    "日期",
    days,
    index=days.index(report.anomaly.day),
    format_func=lambda d: f"{d:%Y-%m-%d}",
)

try:
    anomaly = drill_down(verified_first, machine, day)
except ValueError as exc:
    st.warning(str(exc))
else:
    st.write(
        f"批次 {anomaly.batches}；同批次同日的对照设备：{anomaly.peer_machines or '无'}。"
        f"**{anomaly.machine}** 良品误报率 {pct(anomaly.target_fpr)}"
        f"（{anomaly.target_fp}/{anomaly.target_good}），"
        f"对照设备 {pct(anomaly.peer_fpr)}（{anomaly.peer_fp}/{anomaly.peer_good}），"
        f"Fisher 精确检验 p = {anomaly.fisher_p:.2e}。"
    )
    st.dataframe(
        anomaly.comparison.style.format(RATE_FORMAT, na_rep="n/a"),
        hide_index=True,
        width="stretch",
    )
    left, right = st.columns(2)
    with left:
        st.caption("按班次拆分")
        st.dataframe(
            anomaly.by_shift.style.format(RATE_FORMAT, na_rep="n/a"),
            hide_index=True,
            width="stretch",
        )
        st.caption("图像亮度：异常日 vs 其它日")
        st.dataframe(anomaly.brightness_compare.style.format("{:.1f}"), width="stretch")
    with right:
        fig = plots.anomaly_drilldown(anomaly)
        st.pyplot(fig)
        plt.close(fig)

st.subheader("3. 批次筛查：是单设备问题，还是整批问题")
st.caption("按批次 FPR 相对整体 FPR 的 z 分数筛选（z > 4）。")
st.dataframe(
    report.flagged[["batch_id", "good", "fp", "fpr", "z"]].style.format(
        {"fpr": "{:.2%}", "z": "{:.1f}"}
    ),
    hide_index=True,
    width="stretch",
)
if report.cross_machine_batch and report.cross_machine_profile is not None:
    st.write(
        f"**{report.cross_machine_batch}** 不属于上面的单设备异常，"
        "对照同线其它日和同日其它线的图像特征："
    )
    st.dataframe(report.cross_machine_profile.style.format("{:.1f}"), width="stretch")

st.subheader("4. 班次与缺陷类型")
s = report.shift
st.write(
    f"夜班 FPR {pct(s['night_fpr'], 2)}，其它班次 {pct(s['others_fpr'], 2)}"
    f"（卡方检验 p = {s['p_value']:.1e}）。"
)
fig = plots.fpr_by_shift_and_batch(verified_first)
st.pyplot(fig)
plt.close(fig)
st.caption("各缺陷类型的 Recall")
st.dataframe(
    report.type_recall[["defect_type", "defects", "tp", "fn", "recall"]].style.format(
        {"recall": "{:.1%}"}
    ),
    hide_index=True,
    width="stretch",
)

st.subheader("结论摘要")
st.markdown(report.summary_markdown())
