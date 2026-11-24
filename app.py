# Streamlit 的核心模型：整个脚本就是页面。
# 每次用户操作（改筛选、点按钮）都会从第一行重跑到底部，页面因此刷新；
# 所以这里写成"读数据 → 画页面"的直线脚本，而不是函数/事件回调式的 UI 代码。
# 副作用只有 st.* 调用本身，没有需要维护的组件状态。

import pandas as pd
import streamlit as st

from aoi.data import load_clean
from aoi.paths import Paths

# 数据位置统一由 aoi.paths 决定：它基于 __file__ 定位，避免依赖启动时的工作目录
# （streamlit run 的 cwd 可能与仓库根不同）；测试时可用环境变量 AOI_ROOT 指向别的目录。
DATA_PATH = Paths.default().clean

# 页面级配置必须是本页第一条 st 调用；重复调用会抛错。设置后不能再改。
st.set_page_config(page_title="AOI 检测质量看板", layout="wide")


# @st.cache_data 缓存"函数的返回值"：脚本每次重跑都会执行这个函数,
# 但参数不变时 Streamlit 会直接返回上次的 DataFrame，不重复读 CSV。
# 参数/代码变化时缓存自动失效；它序列化结果，所以适合纯数据、不适合连接对象。
@st.cache_data
def load_data() -> pd.DataFrame:
    # load_clean 内部用 parse_dates 一次性把时间列转成 datetime，后面才能用 .dt.date / groupby("date")。
    return load_clean(DATA_PATH)


def reviewed_fp_share(frame: pd.DataFrame) -> float | None:
    """已复判报警件中的误报占比。没有已复判样本时返回 None，交给调用方显示 n/a。"""
    reviewed = frame[frame["is_alarm"] & frame["has_reference"]]
    return None if reviewed.empty else reviewed["is_false_positive"].mean()


# --- 前置校验：数据缺失就直接收工 ---
# st.stop() 会立刻终止本次脚本执行，后面的 UI 不再渲染。
# 这比 if/else 包住整页更扁平，重跑时也能正常恢复。
if not DATA_PATH.exists():
    st.error(
        "找不到 data/processed/aoi_clean.csv。请先运行 `uv run python -m aoi.pipeline`（Python 版），"
        "或 notebooks/aoi/02_aoi_analysis.ipynb（Notebook 版）。"
    )
    st.stop()

data = load_data()

# st.title / st.caption 是页面最上方的元素，按调用顺序自上而下排布。
st.title("AOI 检测质量看板")
st.caption("数据为模拟数据，仅用于面试练习。")

# with st.sidebar: 把块内的元素渲染到左侧边栏，而不是主区域。
with st.sidebar:
    st.header("筛选")

    # 控件函数的返回值 = 该控件当前的值；用户改一下就触发整脚本重跑，值随之更新。
    # 不需要绑定变量到某个组件 id，位置即身份（同名同类型的控件要显式给 key 区分）。
    min_date, max_date = data["date"].min().date(), data["date"].max().date()
    picked = st.date_input("日期范围", (min_date, max_date), min_value=min_date, max_value=max_date)

    # multiselect 返回选中项列表；default 只在控件第一次出现时生效，之后由用户的选择接管。
    lines = st.multiselect(
        "产线", sorted(data["line_id"].unique()), default=sorted(data["line_id"].unique())
    )
    # 设备候选项随所选产线联动：每次重跑都用最新的 lines 重新计算，
    # 这就是 Streamlit 里实现"级联筛选"的方式，无需回调。
    machine_options = sorted(data[data["line_id"].isin(lines)]["machine_id"].unique())
    machines = st.multiselect("设备", machine_options, default=machine_options)
    shifts = st.multiselect("班次", ["day", "swing", "night"], default=["day", "swing", "night"])
    # help= 是悬停提示；checkbox/radio/selectbox 等都支持这类通用参数。
    first_pass_only = st.checkbox(
        "仅统计每件产品的首检", value=True, help="复测会重复计数，放大报警和误报数量。"
    )

# date_input 返回 (start, end) 元组，但用户可能只选了一天（返回长度 1 的元组），
# 所以用 picked[0] / picked[-1] 兼容单日选择，并用 len() 兜底空值。
start, end = (picked[0], picked[-1]) if len(picked) else (min_date, max_date)

# 注意筛选是在 Python/pandas 里做的，不是 SQL；重跑就重算一次，几万行内无需额外优化。
# 另外多选框被清空时会返回 []，isin([]) 得到空表而不是"全选"，下面用 st.stop 兜底。
view = data[
    (data["date"].dt.date >= start)
    & (data["date"].dt.date <= end)
    & data["line_id"].isin(lines)
    & data["machine_id"].isin(machines)
    & data["shift"].isin(shifts)
]
# 复测（同一 product 的后续检测）会重复计入报警与误报，默认只看首检。
if first_pass_only:
    view = view[view["is_first_pass"]]

if view.empty:
    st.warning("当前筛选条件下没有数据。")
    st.stop()

reviewed_alarms = view[view["is_alarm"] & view["has_reference"]]
share = reviewed_fp_share(view)

# --- 指标卡 ---
# st.columns(5) 返回 5 个列容器，用下标分别放置元素，实现同一行等宽布局。
cols = st.columns(5)
cols[0].metric("检测总数", f"{len(view):,}")
cols[1].metric("AOI 报警数", f"{int(view['is_alarm'].sum()):,}")
cols[2].metric("已复判报警数", f"{len(reviewed_alarms):,}")
cols[3].metric("已确认误报数", f"{int(reviewed_alarms['is_false_positive'].sum()):,}")
cols[4].metric("已复判报警中的误报占比", "n/a" if share is None else f"{share:.1%}")

# --- 两张并排的图表 ---
# 两列各自 with 一个容器；左列和右列是两个独立的 block。
left, right = st.columns(2)
with left:
    st.subheader("每日 AOI 报警数")
    # st.bar_chart / st.line_chart 直接吃 pandas：Series 用索引当 x 轴、值当 y 轴，
    # rename 的作用只是让图例/轴标签显示中文而不是原始列名。
    st.bar_chart(view.groupby("date")["is_alarm"].sum().rename("报警数"))
with right:
    st.subheader("每日已确认误报数")
    st.line_chart(reviewed_alarms.groupby("date")["is_false_positive"].sum().rename("误报数"))

# 复用变量名 left/right 是安全的：with 块只是作用域语法，两个容器本身互不影响。
left, right = st.columns(2)
with left:
    st.subheader("各设备：已复判报警中的误报占比")
    # mean() 对布尔列等价于"True 的比例"，.mul(100) 转成百分数方便读图。
    machine_share = (
        reviewed_alarms.groupby("machine_id")["is_false_positive"]
        .mean()
        .mul(100)
        .rename("误报占比 (%)")
    )
    st.bar_chart(machine_share)
with right:
    st.subheader("缺陷类型分布（AOI 报出的类型）")
    # 把布尔列映射成中文标签后再 unstack → 形成"缺陷类型 × 结论"的二维表，
    # Streamlit 会把多列 DataFrame 自动画成堆叠/分组柱状图。
    type_mix = (
        reviewed_alarms.assign(
            conclusion=reviewed_alarms["is_false_positive"].map({True: "误报", False: "真实缺陷"})
        )
        .groupby(["defect_type", "conclusion"])
        .size()
        .unstack(fill_value=0)
    )
    st.bar_chart(type_mix)

# st.expander 是可折叠面板：里面的内容仍会执行和渲染，只是默认收起，
# 适合放"可选深入"的内容，避免主视图被次要信息淹没。
with st.expander("仅模拟数据可得的指标（真实产线通常没有全量参考判定）"):
    st.warning("这些指标需要对放行件也有独立参考判定。真实产线只复判报警件时，无法直接计算。")
    verified = view[view["has_reference"]]
    good = int(verified["is_actual_good"].sum())
    defects = int(verified["is_actual_defect"].sum())
    # st.write 能直接渲染 dict，键值对会自动排版；这里用三元表达式在分母为 0 时给 n/a。
    st.write(
        {
            "良品误报率 FP / (FP + TN)": f"{verified['is_false_positive'].sum() / good:.2%}"
            if good
            else "n/a",
            "缺陷召回率 TP / (TP + FN)": f"{verified['is_true_positive'].sum() / defects:.2%}"
            if defects
            else "n/a",
            "漏检数 FN": int(verified["is_miss"].sum()),
        }
    )
    # 透视成 设备 × 日期 的宽表；max(..., 1) 防止某个分组良品数为 0 时除零。
    heat = (
        verified.groupby(["machine_id", "date"])
        .apply(
            lambda g: g["is_false_positive"].sum() / max(g["is_actual_good"].sum(), 1),
            include_groups=False,
        )
        .unstack("date")
        .mul(100)
    )
    # 列名是 Timestamp，转成 MM-DD 更省横向空间。
    heat.columns = [d.strftime("%m-%d") for d in heat.columns]
    st.caption("设备 × 日期 良品误报率 (%)")
    # pandas Styler 也会被 st.dataframe 识别：background_gradient 做热力底色，format 定小数位。
    st.dataframe(heat.style.background_gradient(cmap="Reds", axis=None).format("{:.1f}"))

# --- 明细表 ---
st.subheader("明细")
# 这个 checkbox 放在主区域（不在 sidebar 里）：布局位置由调用位置决定，与逻辑无关。
only_fp = st.checkbox("仅显示已确认误报")
detail = view[view["is_false_positive"]] if only_fp else view
st.dataframe(
    detail[
        [
            "inspection_id",
            "product_id",
            "inspection_time",
            "line_id",
            "machine_id",
            "batch_id",
            "shift",
            "defect_type",
            "aoi_result",
            "reference_result",
            "aoi_score",
            "brightness",
            "contrast",
        ]
    ],
    # width="stretch" 让表格撑满容器宽度；hide_index=True 隐藏 pandas 的行号列。
    width="stretch",
    hide_index=True,
)
