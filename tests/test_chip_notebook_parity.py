"""芯片测试：Notebook 版与 Python 版必须产出一致的结果。

在临时目录里真实执行 Notebook 01-03（nbconvert --execute），再与 Python 版流水线的产物比较。
Notebook 用的是自己的内联代码，所以这里比较的是两份独立实现，而不是同一份代码跑两遍。
"""

import json
import shutil
import subprocess
import sys

import pytest

from aoi.paths import PROJECT_ROOT
from chiptest.paths import ChipPaths

pytestmark = pytest.mark.slow

NOTEBOOKS = [
    "01_chip_log_parsing.ipynb",
    "02_chip_yield_attribution.ipynb",
    "03_chip_pat_efficiency.ipynb",
]


@pytest.fixture(scope="module")
def notebook_paths(make_chip_root) -> ChipPaths:
    root = make_chip_root("chip_notebook_version")
    notebook_dir = root / "notebooks" / "chiptest"
    notebook_dir.mkdir(parents=True)
    executed_dir = root / "executed"
    for name in NOTEBOOKS:
        shutil.copy2(PROJECT_ROOT / "notebooks" / "chiptest" / name, notebook_dir / name)
    for name in NOTEBOOKS:  # 01 -> 02 -> 03：后一个读取前一个写出的文件
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--output-dir",
                str(executed_dir),
                "--ExecutePreprocessor.timeout=600",
                str(notebook_dir / name),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{name} 执行失败:\n{result.stderr[-2000:]}"
    return ChipPaths(root)


def _assert_close(left, right, path=""):
    if isinstance(left, dict):
        assert list(left) == list(right), path
        for key in left:
            _assert_close(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        assert len(left) == len(right), path
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            _assert_close(a, b, f"{path}[{i}]")
    elif isinstance(left, float):
        assert left == pytest.approx(right, rel=1e-9, abs=1e-12), path
    else:
        assert left == right, path


def test_clean_data_is_identical(notebook_paths, chip_run):
    python_paths, _ = chip_run
    assert notebook_paths.clean.read_bytes() == python_paths.clean.read_bytes()


def test_test_catalog_is_identical(notebook_paths, chip_run):
    python_paths, _ = chip_run
    assert notebook_paths.tests.read_bytes() == python_paths.tests.read_bytes()


def test_summary_is_identical(notebook_paths, chip_run):
    python_paths, _ = chip_run
    notebook_summary = json.loads(notebook_paths.summary.read_text())
    python_summary = json.loads(python_paths.summary.read_text())
    _assert_close(notebook_summary, python_summary)


def test_both_versions_produce_the_same_figures(notebook_paths, chip_run):
    python_paths, _ = chip_run
    notebook_names = {p.name for p in notebook_paths.figures.glob("*.png")}
    python_names = {p.name for p in python_paths.figures.glob("*.png")}
    assert notebook_names == python_names
    assert len(python_names) == 7
