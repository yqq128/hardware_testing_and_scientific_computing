import importlib.util
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

from aoi import pipeline
from aoi.paths import PROJECT_ROOT, Paths
from chiptest import pipeline as chip_pipeline
from chiptest.paths import ChipPaths


def pytest_configure(config):
    # 无显示环境下用不依赖 GUI 的后端，避免 macOS 在非主线程创建窗口时崩溃。
    matplotlib.use("Agg")


@pytest.fixture(scope="session")
def raw_csv(tmp_path_factory) -> Path:
    """用生成器在临时目录造一份原始数据。

    仓库里的 data/raw/*.csv 被 .gitignore 排除，新克隆的仓库里没有它，
    所以测试不能依赖磁盘上现成的数据。生成器带固定随机种子，结果可复现。
    """
    spec = importlib.util.spec_from_file_location(
        "generate_data", PROJECT_ROOT / "scripts" / "generate_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path_factory.mktemp("raw") / "aoi_inspection.csv"
    module.generate().to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def make_root(tmp_path_factory, raw_csv) -> Callable[[str], Path]:
    """创建一个只含原始数据的独立根目录，产物都写在里面，不碰仓库里的文件。"""

    def _make(name: str) -> Path:
        root = tmp_path_factory.mktemp(name)
        (root / "data" / "raw").mkdir(parents=True)
        shutil.copy2(raw_csv, root / "data" / "raw" / "aoi_inspection.csv")
        return root

    return _make


@pytest.fixture(scope="session")
def python_run(make_root) -> tuple[Paths, pipeline.PipelineResult]:
    """Python 版流水线完整跑一次，供多个测试复用（约 10 秒）。"""
    paths = Paths(make_root("python_version"))
    result = pipeline.run(paths, log=lambda _: None)
    return paths, result


@pytest.fixture(scope="session")
def chip_generator():
    """加载 scripts/generate_chip_data.py（脚本目录不在 sys.path 里，用路径加载）。"""
    spec = importlib.util.spec_from_file_location(
        "generate_chip_data", PROJECT_ROOT / "scripts" / "generate_chip_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass 需要能通过模块名找回自己
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def chip_root(tmp_path_factory, chip_generator) -> tuple[Path, pd.DataFrame]:
    """在临时目录生成完整的模拟 ATE 日志（约 116 万行）和 truth.csv，约 3 秒。"""
    root = tmp_path_factory.mktemp("chip_root")
    truth = chip_generator.write_logs(root / "data" / "raw" / "chip")
    return root, truth


@pytest.fixture(scope="session")
def make_chip_root(tmp_path_factory, chip_root) -> Callable[[str], Path]:
    """复制一份只含原始日志的独立根目录，产物写在里面，不碰其他测试的数据。"""

    def _make(name: str) -> Path:
        root = tmp_path_factory.mktemp(name)
        shutil.copytree(chip_root[0] / "data" / "raw" / "chip", root / "data" / "raw" / "chip")
        return root

    return _make


@pytest.fixture(scope="session")
def chip_run(chip_root) -> tuple[ChipPaths, chip_pipeline.ChipResult]:
    """芯片测试流水线完整跑一次（约 6 秒），供多个测试复用。"""
    paths = ChipPaths(chip_root[0])
    return paths, chip_pipeline.run(paths, log=lambda _: None)
