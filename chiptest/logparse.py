"""流式解析 STDF 风格的 ATE 文本日志。

设计取舍：
- 逐行读取，按晶圆和测试轮次构建 DataFrame 并保留在内存中，解析结束后再合并。
- PTR 记录按字段拆分、查找测试项、转换测量值，按晶圆批量建 DataFrame。
- 常见异常会计数，例如截断的参数记录、哨兵值、nan、缺失 PRR 和重复芯片块。
- 单位以日志里的 TSR 声明为准：同一测试项在不同晶圆可能换单位（uA/mA），统一换算成第一次见到的单位。
"""

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SENTINEL = 1e30  # 测试机用 1E+38 之类的极大值表示"没测"
_SI_PREFIX = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "M": 1e6, "G": 1e9}
_BASE_UNITS = {"A", "V", "Hz", "s", "ohm", "LSB", "cnt"}


def _split_unit(unit: str) -> tuple[float, str] | None:
    if unit in _BASE_UNITS:
        return 1.0, unit
    if len(unit) > 1 and unit[0] in _SI_PREFIX and unit[1:] in _BASE_UNITS:
        return _SI_PREFIX[unit[0]], unit[1:]
    return None


def si_factor(unit_from: str, unit_to: str) -> float | None:
    """把 unit_from 的数值换算到 unit_to 要乘的系数；量纲不同或不认识的单位返回 None。"""
    a, b = _split_unit(unit_from), _split_unit(unit_to)
    if a is None or b is None or a[1] != b[1]:
        return None
    return float(f"{a[0] / b[0]:.10g}")


@dataclass
class TestInfo:
    __test__ = False  # 名字以 Test 开头，告诉 pytest 这不是测试类

    test_no: int
    name: str
    unit: str
    lo: float
    hi: float
    time_ms: float


@dataclass
class ParseStats:
    files: int = 0
    lines: int = 0
    ptr_lines: int = 0
    bad_lines: int = 0
    sentinel_values: int = 0
    nan_values: int = 0
    unit_converted_values: int = 0
    unknown_unit_values: int = 0
    wafer_passes: int = 0
    wafer_count_mismatch: int = 0
    parts: int = 0
    incomplete_parts: int = 0
    duplicate_parts: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParsedLogs:
    parts: pd.DataFrame
    tests: pd.DataFrame
    bins: dict[int, str]
    stats: ParseStats
    elapsed_s: float = 0.0
    bytes_read: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def lines_per_second(self) -> float:
        return self.stats.lines / self.elapsed_s if self.elapsed_s else float("nan")


def _limit(text: str) -> float:
    return float("nan") if text == "-" else float(text)


class LogParser:
    """跨多个日志文件累积状态的解析器：测试项目录、bin 名称、统计、已解析的晶圆块。"""

    def __init__(self) -> None:
        self.stats = ParseStats()
        self.tests: dict[int, TestInfo] = {}
        self.bins: dict[int, str] = {}
        self._col: dict[str, int] = {}
        self._factor: list[float] = []
        self._frames: list[pd.DataFrame] = []
        self.bytes_read = 0

    def _declare_test(self, rec: list[str]) -> None:
        no, name, unit = int(rec[1]), rec[2], rec[3]
        lo, hi, ms = _limit(rec[4]), _limit(rec[5]), float(rec[6])
        known = self.tests.get(no)
        if known is None:
            self.tests[no] = TestInfo(no, name, unit, lo, hi, ms)
            self._col[rec[1]] = len(self._factor)
            self._factor.append(1.0)
            return
        factor = si_factor(unit, known.unit)
        self._factor[self._col[rec[1]]] = np.nan if factor is None else factor

    def parse_file(self, path: Path) -> None:
        stats = self.stats
        stats.files += 1
        self.bytes_read += path.stat().st_size
        col_of, factor = self._col, self._factor
        wafer: tuple | None = None
        rows: dict[str, tuple] = {}
        part: list | None = None  # [part_id, x, y, site, ts, values]

        with path.open(encoding="ascii", errors="replace") as fh:
            for line in fh:
                stats.lines += 1
                rec = line.rstrip("\n").split("|")
                kind = rec[0]

                if kind == "PTR":
                    stats.ptr_lines += 1
                    j = col_of.get(rec[1]) if len(rec) == 4 and part is not None else None
                    if j is None:
                        stats.bad_lines += 1
                        continue
                    try:
                        v = float(rec[2])
                    except ValueError:
                        stats.bad_lines += 1
                        continue
                    if v != v:
                        stats.nan_values += 1
                    elif abs(v) >= SENTINEL:
                        stats.sentinel_values += 1
                    else:
                        f = factor[j]
                        if f != 1.0:
                            if f != f:
                                stats.unknown_unit_values += 1
                                continue
                            v *= f
                            stats.unit_converted_values += 1
                        part[5][j] = v
                elif kind == "PIR":
                    if part is not None:
                        stats.incomplete_parts += 1
                    try:
                        part = [
                            rec[1],
                            int(rec[2]),
                            int(rec[3]),
                            int(rec[4]),
                            rec[5],
                            [np.nan] * len(factor),
                        ]
                    except (IndexError, ValueError):
                        stats.bad_lines += 1
                        part = None
                elif kind == "PRR":
                    if part is None or len(rec) != 4 or rec[1] != part[0]:
                        stats.bad_lines += 1
                        continue
                    if rec[1] in rows:
                        stats.duplicate_parts += 1
                    else:
                        rows[rec[1]] = (*part[:5], int(rec[2]), rec[3] == "P", part[5])
                    part = None
                elif kind == "TSR":
                    self._declare_test(rec)
                elif kind == "WIR":
                    wafer = (rec[1], rec[2], int(rec[5]), rec[3], rec[4])
                    rows, part = {}, None
                elif kind == "WRR":
                    if part is not None:
                        stats.incomplete_parts += 1
                        part = None
                    if wafer is not None:
                        stats.wafer_passes += 1
                        stats.wafer_count_mismatch += len(rows) != int(rec[4])
                        self._flush(wafer, rows)
                    wafer, rows = None, {}
                elif kind == "HBR":
                    self.bins[int(rec[1])] = rec[2]
                elif kind and kind[0] != "#":
                    stats.bad_lines += 1

    def _flush(self, wafer: tuple, rows: dict[str, tuple]) -> None:
        if not rows:
            return
        lot, wafer_id, pass_no, station, program = wafer
        part_id, x, y, site, ts, hbin, ok, values = zip(*rows.values(), strict=True)
        names = [t.name for t in self.tests.values()]
        matrix = np.full((len(rows), len(names)), np.nan)
        for i, v in enumerate(values):
            matrix[i, : len(v)] = v
        frame = pd.DataFrame(
            {
                "lot_id": lot,
                "wafer_id": wafer_id,
                "pass_no": pass_no,
                "part_id": part_id,
                "x": np.array(x),
                "y": np.array(y),
                "site": np.array(site),
                "station": station,
                "program": program,
                "test_time": ts,
                "hbin": np.array(hbin),
                "tester_pass": np.array(ok),
            }
        )
        self._frames.append(pd.concat([frame, pd.DataFrame(matrix, columns=names)], axis=1))
        self.stats.parts += len(rows)

    def result(self, elapsed_s: float = 0.0) -> ParsedLogs:
        parts = pd.concat(self._frames, ignore_index=True) if self._frames else pd.DataFrame()
        if len(parts):
            parts["test_time"] = pd.to_datetime(
                parts["test_time"], format="%Y-%m-%dT%H:%M:%S", errors="coerce"
            )
        tests = pd.DataFrame([asdict(t) for t in self.tests.values()])
        return ParsedLogs(parts, tests, dict(self.bins), self.stats, elapsed_s, self.bytes_read)


def parse_logs(paths: list[Path]) -> ParsedLogs:
    """按文件名顺序解析一批日志，并记录耗时（用于报告解析吞吐）。"""
    parser = LogParser()
    start = time.perf_counter()
    for path in sorted(paths):
        parser.parse_file(path)
    return parser.result(time.perf_counter() - start)
