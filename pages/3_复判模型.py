# 对应 notebooks/aoi/03_ml_experiment.ipynb：AOI 报警之后，用模型自动放行一部分明显的误报，
# 同时守住缺陷保留率。评估口径：按产品首检日期做时间切分，阈值只在验证集上选，测试集只用一次。

import matplotlib.pyplot as plt
import streamlit as st

from aoi import plots, review_model
from aoi.review_model import DRIFT_FORMAT, DRIFT_LABELS, SWEEP_FORMAT, TARGET_RECALL
from aoi.streamlit_support import load_experiment_or_stop, pct

st.set_page_config(page_title="复判模型", layout="wide")
st.title("误报复判模型")
st.caption("数据为模拟数据。模型只对 AOI 已经报警的样本打分：自动放行低风险误报，其余送人工复判。")

result = load_experiment_or_stop()
splits = result.splits
plots.use_style()
display_grid = sorted({0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, result.chosen_threshold})

st.subheader("1. 数据切分（按时间，同一产品不跨集合）")
st.dataframe(splits.summary(), width="stretch")

st.subheader("2. 模型要先打赢“只看 AOI 分数”的基线")
st.dataframe(result.baseline_table.style.format("{:.3f}"), width="stretch")
st.write(
    f"按验证集 PR-AUC 选择 **{result.best_name}**，相对基线的 PR-AUC 增益 "
    f"**{result.pr_auc_gain:+.3f}**。增益不大时要诚实地说：模型的价值有限，"
    "不要因为用了机器学习就默认它更好。"
)

st.subheader("3. 在验证集上选阈值")
st.caption(
    f"规则：验证集 Recall ≥ {TARGET_RECALL:.0%} 的前提下，取最大的阈值（自动放行最多）。"
    "下面的滑块只用来观察验证集上的权衡，**不会改变**最终阈值，更不会看测试集。"
)
valid_sweep = result.valid_sweep
view_threshold = st.slider(
    "查看验证集上的阈值",
    float(valid_sweep["threshold"].min()),
    float(valid_sweep["threshold"].max()),
    result.chosen_threshold,
    0.01,
)
row = valid_sweep.iloc[(valid_sweep["threshold"] - view_threshold).abs().argmin()]
cols = st.columns(4)
cols[0].metric("验证集 Recall", pct(row["recall"]))
cols[1].metric("人工复判量下降", pct(row["workload_reduction"]))
cols[2].metric("漏掉的真实缺陷", int(row["defects_missed"]))
cols[3].metric(
    "是否满足目标",
    "满足" if row["recall"] >= TARGET_RECALL else "不满足",
    help=f"目标：验证集 Recall ≥ {TARGET_RECALL:.0%}",
)
st.dataframe(
    valid_sweep[valid_sweep["threshold"].isin(display_grid)].style.format(SWEEP_FORMAT),
    hide_index=True,
    width="stretch",
)
st.info(f"最终选定的阈值（只由验证集决定）：**{result.chosen_threshold}**")

st.subheader("4. 测试集：用选定阈值只评估一次")
t = result.test_row
cols = st.columns(4)
cols[0].metric("人工复判量下降", pct(t["workload_reduction"]))
cols[1].metric(
    "测试集 Recall",
    pct(t["recall"]),
    help=f"95% 置信区间 {pct(result.recall_ci[0])} ~ {pct(result.recall_ci[1])}",
)
cols[2].metric(
    "被自动放行的误报",
    f"{result.fp_released} / {result.fp_total}",
    help="放行得越多，节省的人工越多",
)
cols[3].metric("端到端缺陷保留率", pct(result.end_to_end.retention))
st.write(
    f"测试集报警 {len(splits.test)} 条，送人工 {int(t['to_review'])} 条，自动放行 "
    f"{int(t['auto_released'])} 条；{result.defects_total} 个真实缺陷漏掉 "
    f"{int(t['defects_missed'])} 个，Recall {pct(t['recall'])}"
    f"（95% 置信区间 {pct(result.recall_ci[0])} ~ {pct(result.recall_ci[1])}）。"
    "缺陷数量有限，置信区间比点估计更能说明把握有多大。"
)
left, right = st.columns([1, 2])
with left:
    fig = plots.review_model_confusion(result)
    st.pyplot(fig)
    plt.close(fig)
with right:
    fig = plots.pr_curve_and_threshold_tradeoff(result)
    st.pyplot(fig)
    plt.close(fig)

st.caption("测试集上各阈值的表现（选定阈值已在验证集上固定，这里仅供对照）")
st.dataframe(
    result.test_sweep[result.test_sweep["threshold"].isin(display_grid)].style.format(SWEEP_FORMAT),
    hide_index=True,
    width="stretch",
)

e2e = result.end_to_end
st.write(
    f"**端到端**：测试期真实缺陷（首检）{e2e.defects_all} 件，AOI 自己漏检 {e2e.aoi_missed} 件"
    f"（AOI Recall {pct(e2e.aoi_recall)}），复判模型额外放走 {e2e.model_missed_first} 件，"
    f"最终保留率 {pct(e2e.retention)}。评价这个模型，要看的是 AOI 漏检和模型漏放加在一起。"
)

st.subheader("5. 各缺陷类型的 Recall")
st.dataframe(result.by_type.style.format({"recall": "{:.1%}"}), width="stretch")

st.subheader("6. 分布漂移：某个批次突然不一样")
st.dataframe(
    result.drift.rename(index=DRIFT_LABELS).style.format(DRIFT_FORMAT),
    width="stretch",
)
st.markdown(review_model.drift_summary_markdown(result.drift))

st.subheader("7. 数据泄漏演示（可选，约十几秒）")
st.caption("同一产品的复测样本若被随机分到训练集和测试集，指标会虚高。")
if st.button("运行泄漏演示"):
    with st.spinner("训练 16 个随机森林…"):
        st.session_state["leakage"] = review_model.leakage_demo(splits.pool)
if "leakage" in st.session_state:
    leak = st.session_state["leakage"]
    st.dataframe(leak.style.format("{:.3f}"), width="stretch")
    st.markdown(review_model.leakage_summary_markdown(leak, splits.pool))
