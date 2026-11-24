"""芯片测试方向的数据路径。与 aoi.paths 共用环境变量 AOI_ROOT，测试可以把根目录指向临时目录。"""

import os
from dataclasses import dataclass
from pathlib import Path

from aoi.paths import PROJECT_ROOT


@dataclass(frozen=True)
class ChipPaths:
    root: Path

    @classmethod
    def default(cls) -> "ChipPaths":
        return cls(Path(os.environ.get("AOI_ROOT", PROJECT_ROOT)))

    @property
    def raw_dir(self) -> Path:
        """原始 ATE 日志目录：每个 lot 一个 *.log，另有仅模拟数据才有的 truth.csv。"""
        return self.root / "data" / "raw" / "chip"

    @property
    def truth(self) -> Path:
        return self.raw_dir / "truth.csv"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def clean(self) -> Path:
        return self.processed_dir / "chip_parts.csv"

    @property
    def tests(self) -> Path:
        return self.processed_dir / "chip_tests.csv"

    @property
    def summary(self) -> Path:
        return self.processed_dir / "chip_summary.json"

    @property
    def figures(self) -> Path:
        return self.root / "reports" / "chip"
