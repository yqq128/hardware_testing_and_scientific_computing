"""Streamlit 多页面看板共用的数据加载、计算缓存与辅助函数模块。

框架特性与设计考量：
1. 【动态缓存失效机制】：
   Streamlit 默认基于函数入参计算缓存哈希。我们将磁盘文件的最后修改时间戳（mtime）作为参数传入，
   当底层流水线重新执行并刷新 CSV 时，页面能在下一次刷新时自动加载最新数据，无需手动重启 Streamlit 进程。
2. 【分级缓存装饰器选型】：
   - `@st.cache_data`: 用于清洗后的结构化数据表格（pd.DataFrame）。其返回结果会被序列化克隆，
     防止不同页面在修改 DataFrame 时产生跨页面竞态污染。
   - `@st.cache_resource`: 用于诊断报告（DiagnosisReport）和训练好的机器学习实验结果（ExperimentResult）。
     这些对象包含 Scikit-Learn Pipeline 和复杂层级结构，做深度序列化开销巨大且无必要；
     以单例资源方式全局常驻，页面层必须遵循“只读不改”契约。
3. 【优雅的缺失拦截】：
   若发现预处理数据文件不存在，使用 `st.error()` 呈现清晰的操作指引，并使用 `st.stop()` 立即中断后续渲染，
   避免产生一连串未初始化的下游 Python 异常。
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from aoi import data, diagnosis, review_model
from aoi.paths import Paths

MISSING_DATA_HINT = (
    "找不到 data/processed/aoi_clean.csv。请先运行 `uv run python -m aoi.pipeline`（Python 版），"
    "或 notebooks/aoi/02_aoi_analysis.ipynb（Notebook 版）。"
)


def _clean_path_or_stop() -> Path:
    """校验已清洗数据文件是否存在；若缺失则直接在 UI 报警并终止 Streamlit 执行。"""
    path = Paths.default().clean
    if not path.exists():
        st.error(MISSING_DATA_HINT)
        st.stop()
    return path


@st.cache_data
def _read_clean(path: str, mtime: float) -> pd.DataFrame:
    """带文件修改时间戳缓存的 DataFrame 加载函数。"""
    return data.load_clean(Path(path))


# 报告/模型结果是带 DataFrame 和 sklearn 对象的复杂结构，用 cache_resource 原样保存，
# 不做序列化；代价是它是全局共享的，所以页面只读不改。
@st.cache_resource
def _build_report(path: str, mtime: float) -> diagnosis.DiagnosisReport:
    """构建产线根因诊断报告的单例缓存工厂函数。"""
    return diagnosis.build_report(_read_clean(path, mtime))


@st.cache_resource
def _run_experiment(path: str, mtime: float) -> review_model.ExperimentResult:
    """训练并评估复判模型的单例缓存工厂函数。

    交互页面默认关闭耗时的泄漏模拟演示（需要训练 16 个随机森林），
    由页面内部的专用按钮按需交互式触发。
    """
    return review_model.run_experiment(_read_clean(path, mtime), include_leakage_demo=False)


def load_clean_or_stop() -> pd.DataFrame:
    """页面层统一入口：安全读取清洗数据，若未就绪则终止页面渲染。"""
    path = _clean_path_or_stop()
    return _read_clean(str(path), path.stat().st_mtime)


def load_report_or_stop() -> diagnosis.DiagnosisReport:
    """页面层统一入口：安全获取根因诊断分析报告。"""
    path = _clean_path_or_stop()
    return _build_report(str(path), path.stat().st_mtime)


def load_experiment_or_stop() -> review_model.ExperimentResult:
    """页面层统一入口：安全获取二次复判模型实验与评估结果。"""
    path = _clean_path_or_stop()
    return _run_experiment(str(path), path.stat().st_mtime)


def pct(value: float, digits: int = 1) -> str:
    """将浮点比率安全格式化为百分比字符串，遇到 NaN 返回 'n/a'。"""
    return "n/a" if pd.isna(value) else f"{value:.{digits}%}"
