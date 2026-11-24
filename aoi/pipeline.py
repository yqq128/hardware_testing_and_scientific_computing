"""命令行入口：原始数据 -> 清洗 -> 诊断 -> 复判模型 -> 指标 JSON 与 9 张图。

    uv run python -m aoi.pipeline            # 默认处理仓库内的 data/
    uv run python -m aoi.pipeline --no-plots # 只出数据和指标，跳过图片
    uv run python -m aoi.pipeline --root DIR # 使用另一份数据目录（DIR/data/raw/...）

产物与依次执行 Notebook 02、03 完全一致（tests/test_notebook_parity.py 保证）。
"""

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pandas as pd

from aoi import data, diagnosis, plots, review_model
from aoi.paths import Paths


@dataclass
class PipelineResult:
    """流水线执行产物容器（包含清洗后数据、诊断报告、模型实验结果与生成的图表文件路径列表）。"""

    clean: pd.DataFrame
    report: diagnosis.DiagnosisReport
    experiment: review_model.ExperimentResult
    figures: list[Path]


def run(
    paths: Paths, *, make_plots: bool = True, log: Callable[[str], None] = print
) -> PipelineResult:
    """执行端到端数据分析与建模批处理流水线。

    执行阶段：
    [1/4] 读取原始数据并清洗：校验数据质量，生成清洗数据并落盘；
    [2/4] 根因异常诊断：构建多维透视与 Fisher/Z-score 统计诊断报告；
    [3/4] 机器学习二次复判：时间序列切分建模，以 Recall>=98% 选阈值并输出指标 JSON；
    [4/4] 报表图表生成：使用无头渲染后端（Agg）输出 9 张质量看板图。
    """
    if not paths.raw.exists():
        raise FileNotFoundError(
            f"找不到原始数据 {paths.raw}。先运行 `uv run python scripts/generate_data.py` 生成模拟数据。"
        )

    log(f"[1/4] 读取并清洗 {paths.raw}")
    raw = data.load_raw(paths.raw)
    log(data.quality_report(raw).to_string())
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    data.build_clean(raw).to_csv(paths.clean, index=False)
    # 与 Notebook 03 一致：之后都从落盘的 CSV 读，保证分析/建模/看板用的是同一份数据。
    clean = data.load_clean(paths.clean)
    log(f"      saved {len(clean)} rows -> {paths.clean}")

    log("[2/4] 误报诊断")
    report = diagnosis.build_report(clean)
    o = report.overall
    log(
        f"      Recall={o['recall']:.2%}  FPR={o['fpr']:.2%}  Precision={o['precision']:.2%}  "
        f"异常点 {report.anomaly.machine} @ {report.anomaly.day:%Y-%m-%d}"
    )

    log("[3/4] 复判模型（含泄漏演示）")
    experiment = review_model.run_experiment(clean)
    paths.metrics.write_text(experiment.metrics_json())
    t = experiment.test_row
    log(
        f"      {experiment.best_name}，阈值 {experiment.chosen_threshold}，"
        f"工作量下降 {t['workload_reduction']:.1%}，Recall {t['recall']:.1%}，"
        f"端到端保留率 {experiment.end_to_end.retention:.1%}"
    )
    log(f"      saved -> {paths.metrics}")

    figures: list[Path] = []
    if make_plots:
        log("[4/4] 生成图表")
        # 命令行没有图形界面，强制使用不依赖显示环境的后端。
        matplotlib.use("Agg")
        figures = plots.render_all(clean, report, experiment, paths.dashboard)
        log(f"      saved {len(figures)} figures -> {paths.dashboard}")
    else:
        log("[4/4] 跳过图表（--no-plots）")

    return PipelineResult(clean, report, experiment, figures)


def main(argv: list[str] | None = None) -> int:
    """命令行 CLI 主入口解析函数。"""
    parser = argparse.ArgumentParser(description="AOI 数据分析与复判模型流水线（Python 版）")
    parser.add_argument(
        "--root", type=Path, default=None, help="数据根目录，默认读 AOI_ROOT 或仓库根"
    )
    parser.add_argument("--no-plots", action="store_true", help="跳过图片生成")
    args = parser.parse_args(argv)

    paths = Paths(args.root.resolve()) if args.root else Paths.default()
    try:
        run(paths, make_plots=not args.no_plots)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
