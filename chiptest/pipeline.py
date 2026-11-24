"""命令行入口：原始 ATE 日志 -> 解析清洗 -> 分析报告 -> 指标 JSON 与 7 张图。

uv run python scripts/generate_chip_data.py   # 先生成模拟日志（约 116 万行）
uv run python -m chiptest.pipeline            # 默认处理仓库内的 data/
uv run python -m chiptest.pipeline --no-plots # 只出数据和指标
uv run python -m chiptest.pipeline --root DIR # 使用另一份数据目录（DIR/data/raw/chip/*.log）
"""

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pandas as pd

from chiptest import data, evaluation, logparse, plots
from chiptest import report as report_module
from chiptest.paths import ChipPaths

MISSING_HINT = (
    "找不到 {path} 下的 *.log。先运行 `uv run python scripts/generate_chip_data.py` 生成模拟日志。"
)


@dataclass
class ChipResult:
    clean: pd.DataFrame
    tests: pd.DataFrame
    parsed: logparse.ParsedLogs
    report: report_module.ChipReport
    figures: list[Path]


def run(
    paths: ChipPaths, *, make_plots: bool = True, log: Callable[[str], None] = print
) -> ChipResult:
    logs = sorted(paths.raw_dir.glob("*.log"))
    if not logs:
        raise FileNotFoundError(MISSING_HINT.format(path=paths.raw_dir))

    log(f"[1/4] 解析 {len(logs)} 个日志文件")
    parsed = logparse.parse_logs(logs)
    log(
        f"      {parsed.stats.lines:,} 行 / {parsed.bytes_read / 1e6:.1f} MB，"
        f"用时 {parsed.elapsed_s:.2f}s（{parsed.lines_per_second / 1e6:.2f}M 行/秒）"
    )

    log("[2/4] 清洗与体检")
    clean = data.build_clean(parsed)
    log(data.quality_report(parsed, clean).to_string())
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(paths.clean, index=False)
    parsed.tests.to_csv(paths.tests, index=False)
    log(f"      saved {len(clean):,} rows -> {paths.clean}")

    log("[3/4] 分析")
    truth = evaluation.load_truth(paths.truth) if paths.truth.exists() else None
    report = report_module.build_report(clean, parsed.tests, truth)
    o = report.overall
    log(
        f"      FPY={o['fpy']:.2%}  最终良率={o['final_yield']:.2%}  复测恢复率={o['recovery_rate']:.1%}"
    )
    for s in report.suspects:
        where = f"{s.lot_id}/{s.station}" + (f"/site{s.site}" if s.site else "")
        log(f"      线索 {where}: {s.kind}，主要失败项 {s.signature['top_test']}")
    paths.summary.write_text(report.summary_json())
    log(f"      saved -> {paths.summary}")

    figures: list[Path] = []
    if make_plots:
        log("[4/4] 生成图表")
        matplotlib.use("Agg")
        figures = plots.render_all(clean, report, paths.figures)
        log(f"      saved {len(figures)} figures -> {paths.figures}")
    else:
        log("[4/4] 跳过图表（--no-plots）")
    return ChipResult(clean, parsed.tests, parsed, report, figures)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="芯片测试日志分析流水线")
    parser.add_argument(
        "--root", type=Path, default=None, help="数据根目录，默认读 AOI_ROOT 或仓库根"
    )
    parser.add_argument("--no-plots", action="store_true", help="跳过图片生成")
    args = parser.parse_args(argv)

    paths = ChipPaths(args.root.resolve()) if args.root else ChipPaths.default()
    try:
        run(paths, make_plots=not args.no_plots)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
