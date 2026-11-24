"""芯片测试看板页面共用的加载与缓存（与 aoi.streamlit_support 同一套思路）。

- 缓存键里带文件的修改时间：流水线重新跑完后，页面下次刷新会自动读到新数据。
- 报告对象含多张 DataFrame，用 cache_resource 原样常驻、不序列化；代价是全局共享，所以页面只读不改。
- 数据缺失时用 st.error 给出可操作的提示，再 st.stop() 终止本次脚本，避免后面一串未初始化的异常。
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from chiptest import data, evaluation, report
from chiptest.paths import ChipPaths

MISSING_DATA_HINT = (
    "找不到 data/processed/chip_parts.csv。请先运行 `uv run python scripts/generate_chip_data.py` "
    "生成模拟日志，再运行 `uv run python -m chiptest.pipeline` 解析分析。"
)


def _paths_or_stop() -> ChipPaths:
    paths = ChipPaths.default()
    if not (paths.clean.exists() and paths.tests.exists()):
        st.error(MISSING_DATA_HINT)
        st.stop()
    return paths


@st.cache_data
def _read_clean(path: str, mtime: float) -> pd.DataFrame:
    return data.load_clean(Path(path))


@st.cache_data
def _read_tests(path: str, mtime: float) -> pd.DataFrame:
    return data.load_tests(Path(path))


@st.cache_resource
def _build_report(clean_path: str, tests_path: str, mtime: float, truth_path: str | None):
    truth = evaluation.load_truth(Path(truth_path)) if truth_path else None
    return report.build_report(
        _read_clean(clean_path, mtime), _read_tests(tests_path, mtime), truth
    )


def load_or_stop() -> tuple[pd.DataFrame, pd.DataFrame, report.ChipReport]:
    """页面层统一入口：返回 (清洗数据, 测试项表, 分析报告)；数据未就绪则终止页面。"""
    paths = _paths_or_stop()
    mtime = paths.clean.stat().st_mtime
    truth = str(paths.truth) if paths.truth.exists() else None
    clean = _read_clean(str(paths.clean), mtime)
    tests = _read_tests(str(paths.tests), mtime)
    return clean, tests, _build_report(str(paths.clean), str(paths.tests), mtime, truth)
