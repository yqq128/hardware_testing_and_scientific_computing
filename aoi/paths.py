"""项目路径配置与管理模块。

统一管理数据输入、中间清洗产物、模型指标与图表输出的目录结构。
支持通过环境变量 AOI_ROOT 重定向根目录，便于单元测试与沙箱运行。
"""

import os
from dataclasses import dataclass
from pathlib import Path

# 当前文件所在目录的上一级即为项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    """工程目录路径配置对象（不可变数据类）。"""

    root: Path

    @classmethod
    def default(cls) -> "Paths":
        """获取默认路径实例。

        优先读取环境变量 AOI_ROOT（用于测试隔离与临时实验），
        若未设置则回退至仓库真实根目录。
        """
        return cls(Path(os.environ.get("AOI_ROOT", PROJECT_ROOT)))

    @property
    def raw(self) -> Path:
        """AOI 原始检测明细 CSV 路径。"""
        return self.root / "data" / "raw" / "aoi_inspection.csv"

    @property
    def processed_dir(self) -> Path:
        """处理后数据及模型指标存放目录。"""
        return self.root / "data" / "processed"

    @property
    def clean(self) -> Path:
        """清洗去重并补充派生统计字段后的明细数据路径。"""
        return self.processed_dir / "aoi_clean.csv"

    @property
    def metrics(self) -> Path:
        """二次复判模型在测试集上的最终评估指标 JSON 文件。"""
        return self.processed_dir / "model_metrics.json"

    @property
    def dashboard(self) -> Path:
        """生成的 9 张质量看板图表产物输出目录。"""
        return self.root / "reports" / "dashboard"
