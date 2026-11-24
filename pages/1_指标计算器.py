# 对应 notebooks/aoi/01_quality_metrics.ipynb：混淆矩阵 -> 质量指标。
# 这一页不依赖任何数据文件，输入四个格子就能算。

import numpy as np
import pandas as pd
import streamlit as st

from aoi.metrics import (
    RATE_COLUMNS,
    false_call_denominators,
    precision_by_defect_rate,
    quality_metrics,
    trivial_baselines,
)
from aoi.streamlit_support import pct

st.set_page_config(page_title="指标计算器", layout="wide")
st.title("指标计算器")
st.caption("正类 = 真实缺陷。默认值是面试讲义里的例子：100 个缺陷、9900 个良品。")

cols = st.columns(4)
tp = cols[0].number_input("TP：报警且确实有缺陷", min_value=0, value=95, step=1)
fn = cols[1].number_input("FN：放行但确实有缺陷（漏检）", min_value=0, value=5, step=1)
fp = cols[2].number_input("FP：报警但其实是良品（误报）", min_value=0, value=495, step=1)
tn = cols[3].number_input("TN：放行且确实是良品", min_value=0, value=9405, step=1)

m = quality_metrics(int(tn), int(fp), int(fn), int(tp))

metric_cols = st.columns(5)
metric_cols[0].metric("Precision", pct(m["Precision"]), help="报警里有多少是真缺陷：TP / (TP + FP)")
metric_cols[1].metric("Recall", pct(m["Recall"]), help="真缺陷里抓到多少：TP / (TP + FN)")
metric_cols[2].metric("FPR（良品误报率）", pct(m["FPR"]), help="良品里被误报的比例：FP / (FP + TN)")
metric_cols[3].metric("漏检率", pct(m["Miss rate"]), help="FN / (TP + FN) = 1 - Recall")
metric_cols[4].metric("需人工复判", f"{m['Manual review (all alarms)']:,}", help="所有报警件")

st.subheader("为什么不能只看准确率")
st.caption("“全部判 OK”的准确率也很高，但一个缺陷都抓不到。")
baselines = trivial_baselines(int(tn), int(fp), int(fn), int(tp))
st.dataframe(
    baselines.style.format({c: "{:.2%}" for c in RATE_COLUMNS}, na_rep="undefined"),
    width="stretch",
)

st.subheader("误报率到底除以谁")
st.dataframe(
    false_call_denominators(int(tn), int(fp), int(fn), int(tp)).style.format({"Value": "{:.2%}"}),
    hide_index=True,
    width="stretch",
)

st.subheader("同一台机器，缺陷率变了 Precision 会怎样")
st.caption("Recall 和 FPR 是机器自身的性质；Precision 还取决于缺陷有多常见。")
slider_cols = st.columns(2)
recall_fixed = slider_cols[0].slider("Recall", 0.50, 1.00, 0.95, 0.01)
fpr_fixed = slider_cols[1].slider("FPR", 0.001, 0.200, 0.050, 0.001, format="%.3f")

by_rate = precision_by_defect_rate(
    recall_fixed, fpr_fixed, np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.10])
)
left, right = st.columns(2)
left.dataframe(
    by_rate.style.format({"Defect rate": "{:.1%}", "Precision": "{:.1%}"}),
    hide_index=True,
    width="stretch",
)
right.line_chart(
    pd.DataFrame({"Precision": by_rate["Precision"].to_numpy()}, index=by_rate["Defect rate"]),
    x_label="缺陷率",
    y_label="Precision",
)
