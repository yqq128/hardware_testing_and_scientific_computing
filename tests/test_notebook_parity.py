"""Notebook 版与 Python 版必须产出一致的结果。

在临时目录里真实执行 Notebook 01-03（nbconvert --execute），再与 Python 版流水线的产物比较。
Notebook 用的是自己的内联代码，所以这里比较的是两份独立实现，而不是同一份代码跑两遍。
"""

import json
import shutil
import subprocess
import sys

import pytest

from aoi.paths import PROJECT_ROOT, Paths

pytestmark = pytest.mark.slow

NOTEBOOKS = [
    "01_quality_metrics.ipynb",
    "02_aoi_analysis.ipynb",
    "03_ml_experiment.ipynb",
]


@pytest.fixture(scope="module")
def notebook_paths(make_root) -> Paths:
    root = make_root("notebook_version")
    notebook_dir = root / "notebooks" / "aoi"
    notebook_dir.mkdir(parents=True)
    executed_dir = root / "executed"
    for name in NOTEBOOKS:
        shutil.copy2(PROJECT_ROOT / "notebooks" / "aoi" / name, notebook_dir / name)
    for name in NOTEBOOKS:  # 02 先于 03，03 读取 02 写出的 aoi_clean.csv
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
    return Paths(root)


def _assert_close(left, right, path=""):
    if isinstance(left, dict):
        assert left.keys() == right.keys(), path
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


def test_clean_data_is_identical(notebook_paths, python_run):
    python_paths, _ = python_run
    assert notebook_paths.clean.read_bytes() == python_paths.clean.read_bytes()


def test_model_metrics_are_identical(notebook_paths, python_run):
    python_paths, _ = python_run
    notebook_metrics = json.loads(notebook_paths.metrics.read_text())
    python_metrics = json.loads(python_paths.metrics.read_text())
    _assert_close(notebook_metrics, python_metrics)


def test_both_versions_produce_the_same_figures(notebook_paths, python_run):
    python_paths, _ = python_run
    notebook_names = {p.name for p in notebook_paths.dashboard.glob("*.png")}
    python_names = {p.name for p in python_paths.dashboard.glob("*.png")}
    assert notebook_names == python_names
    assert len(python_names) == 9
