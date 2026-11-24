"""生成模拟芯片测试（ATE）日志，全部为虚构，仅用于面试练习。

输出到 data/raw/chip/：每个 lot 一个 STDF 风格的文本日志（L01.log ...）和一份仅模拟数据才有的 truth.csv。
日志只含测试机真实会产出的信息；truth.csv 是"上帝视角"（真实缺陷、潜在弱芯片、无噪声时是否合格），
用来检验分析方法，真实产线没有。

日志记录类型（参照 STDF 的命名，但是纯文本）：
  HBR|bin|name                           硬件 bin 定义
  TSR|test_no|name|unit|lo|hi|exec_ms    测试项、单位、限值、单项耗时（程序变更时会重新声明）
  WIR|lot|wafer|station|program|pass|ts  晶圆开始（pass=1 首测，2 复测）
  PIR|part|x|y|site|ts                   芯片开始
  PTR|test_no|value|flag                 一个参数测试结果（P/F 是测试机当时的判定）
  PRR|part|hbin|flag                     芯片结束与 bin
  WRR|lot|wafer|pass|count               晶圆结束与声明的芯片数

植入的线索（供分析时发现；"硬件"指测试系统，不是芯片本身）：
- 接触不良（类比虚焊）：L04 在 T02 测试机的 site3 上 CONT_VDD 间歇性变大，复测大多恢复。
- 电源噪声：L06 在 T03 上 VMIN、SCAN_FMAX 的测量噪声变大，均值不变。
- 时序偏移：L05 起 T01 的 SCAN_FMAX 整体低 4%，复测不恢复。
- 工艺偏移：L07 整批偏向快角（速度高、漏电大、电流大）。
- 晶圆边缘：所有晶圆外圈的致命缺陷率更高。
- 划伤：L03 的 7 号晶圆上有一条贯穿的划伤，沿线的芯片功能测试全挂。
- 单位混用：L02 的 5 号晶圆把 IDD_STANDBY 以 mA 而不是 uA 记录。
- 脏日志：截断行、哨兵值 1E+38 与 nan、缺失 PRR、重复的芯片块。
- 弱芯片：约 0.5% 的芯片漏电偏大但仍在规格内，单看限值抓不出来，要看"相对其速度而言漏电偏大"。
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
PROGRAM = "PROG_v1.3"
STATIONS = ["T01", "T02", "T03"]
LOT_STARTS = [
    "2026-10-02",
    "2026-10-06",
    "2026-10-10",
    "2026-10-14",
    "2026-10-18",
    "2026-10-22",
    "2026-10-25",
    "2026-10-28",
]
WAFER_RADIUS = 17  # 以芯片为单位的晶圆半径，约 900 颗芯片

NAN = float("nan")
# (test_no, name, unit, lo, hi, exec_ms, hbin)；顺序就是测试程序的执行顺序
TESTS = [
    (1001, "CONT_VDD", "ohm", 0.0, 2.0, 20, 2),
    (1002, "CONT_IO", "ohm", 0.0, 2.0, 20, 2),
    (1101, "IDD_STANDBY", "uA", 0.0, 150.0, 60, 3),
    (1102, "IDD_ACTIVE", "mA", 60.0, 200.0, 80, 4),
    (1201, "VREF", "V", 1.17, 1.23, 50, 5),
    (1202, "LDO_OUT", "V", 1.755, 1.845, 50, 5),
    (1203, "OSC_FREQ", "MHz", 23.4, 24.6, 40, 5),
    (1204, "PLL_LOCK", "us", 0.0, 50.0, 60, 5),
    (1205, "ADC_OFFSET", "LSB", -3.0, 3.0, 120, 5),
    (1301, "SCAN_FMAX", "MHz", 1750.0, NAN, 400, 6),
    (1302, "VMIN", "V", NAN, 0.78, 300, 6),
    (1401, "SRAM_BIST", "cnt", 0.0, 0.0, 800, 7),
    (1402, "FUNC_PAT", "cnt", 0.0, 0.0, 350, 8),
]
BINS = {
    1: "PASS",
    2: "CONTACT",
    3: "LEAKAGE",
    4: "POWER",
    5: "ANALOG",
    6: "SPEED",
    7: "MEMORY",
    8: "FUNCTIONAL",
}
NAMES = [t[1] for t in TESTS]
LO = np.array([t[3] for t in TESTS])
HI = np.array([t[4] for t in TESTS])
HBIN_OF_TEST = np.array([t[6] for t in TESTS])

CONTACT_LOT, CONTACT_STATION, CONTACT_SITE, CONTACT_PROB = 4, "T02", 3, 0.30
NOISE_LOT, NOISE_STATION = 6, "T03"
TIMING_FROM_LOT, TIMING_STATION, TIMING_SCALE = 5, "T01", 0.96
SHIFT_LOT, SHIFT_SIGMA = 7, 1.3
SCRATCH_LOT, SCRATCH_WAFER = 3, 7
UNIT_LOT, UNIT_WAFER = 2, 5

KILLER_RATE = 0.025
EDGE_RATE_GAIN = 3.0
WEAK_RATE = 0.005
DIRTY_DIE_RATE = 0.012

MEAS_NOISE = {
    "CONT_VDD": 0.02,
    "VREF": 0.002,
    "LDO_OUT": 0.0015,
    "OSC_FREQ": 0.03,
    "ADC_OFFSET": 0.2,
    "IDD_ACTIVE": 1.0,
}


@dataclass(frozen=True)
class Config:
    n_lots: int = 8
    wafers_per_lot: int = 10
    seed: int = SEED


def _wafer_geometry() -> pd.DataFrame:
    xs, ys = np.meshgrid(
        np.arange(-WAFER_RADIUS, WAFER_RADIUS + 1), np.arange(-WAFER_RADIUS, WAFER_RADIUS + 1)
    )
    x, y = xs.ravel(), ys.ravel()
    keep = np.hypot(x, y) <= WAFER_RADIUS
    x, y = x[keep], y[keep]
    # 2x2 为一次接触（touchdown），四个角依次是 site1..4
    site = 1 + (x & 1) + 2 * (y & 1)
    r = np.hypot(x, y) / (WAFER_RADIUS + 0.5)
    return pd.DataFrame({"x": x, "y": y, "site": site, "r": r})


def _scratch_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a, b = np.array([-15.0, -8.0]), np.array([15.0, 4.0])
    p = np.stack([x, y], axis=1).astype(float)
    t = np.clip(((p - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
    nearest = a + t[:, None] * (b - a)
    return np.hypot(*(p - nearest).T) < 0.8


def _latents(rng: np.random.Generator, geo: pd.DataFrame, lot: int, wafer: int) -> dict:
    n = len(geo)
    x, y = geo["x"].to_numpy(), geo["y"].to_numpy()
    edge = np.clip((geo["r"].to_numpy() - 0.75) / 0.25, 0, 1)

    proc = rng.normal(0, 0.35) + rng.normal(0, 1, n) + (SHIFT_SIGMA if lot == SHIFT_LOT else 0.0)
    scratch = (
        _scratch_mask(x, y) if (lot, wafer) == (SCRATCH_LOT, SCRATCH_WAFER) else np.zeros(n, bool)
    )
    killer = (rng.random(n) < KILLER_RATE * (1 + EDGE_RATE_GAIN * edge)) | scratch
    kind = rng.choice(["sram", "logic", "analog", "short"], n, p=[0.35, 0.30, 0.20, 0.15])
    kind = np.where(scratch, "scratch", np.where(killer, kind, "none"))
    weak = ~killer & (rng.random(n) < WEAK_RATE)
    sram, analog, short = kind == "sram", kind == "analog", kind == "short"
    logic = (kind == "logic") | scratch

    leak_ln = np.log(40) + 0.45 * proc + rng.normal(0, 0.22, n)
    leak_ln += np.where(weak, rng.normal(1.1, 0.15, n), 0.0)
    leak_ln += np.where(short, np.log(rng.uniform(5, 20, n)), 0.0)
    vref = rng.normal(1.20, 0.007, n) + analog * rng.choice([-1, 1], n) * rng.uniform(0.04, 0.08, n)
    pll_bad = analog & (rng.random(n) < 0.5)
    func = np.where(logic, 1 + rng.poisson(20, n), 0) + np.where(
        sram & (rng.random(n) < 0.5), 1 + rng.poisson(4, n), 0
    )
    return {
        "proc": proc,
        "kind": kind,
        "weak": weak,
        "CONT_VDD": rng.normal(0.35, 0.04, n),
        "CONT_IO": rng.normal(0.40, 0.04, n),
        "IDD_STANDBY": np.exp(leak_ln),
        "IDD_ACTIVE": 120 + 15 * proc + rng.normal(0, 4, n) + short * rng.uniform(40, 80, n),
        "VREF": vref,
        "LDO_OUT": 1.5 * vref + rng.normal(0, 0.001, n),
        "OSC_FREQ": rng.normal(24.0, 0.15, n),
        "PLL_LOCK": np.exp(rng.normal(np.log(20), 0.15, n))
        * np.where(pll_bad, rng.uniform(3, 6, n), 1.0),
        "ADC_OFFSET": rng.normal(0, 0.7, n),
        "SCAN_FMAX": 2000 + 90 * proc + rng.normal(0, 40, n),
        "VMIN": 0.72 - 0.012 * proc + rng.normal(0, 0.010, n),
        "SRAM_BIST": np.where(sram, 1 + rng.poisson(3, n), 0).astype(float),
        "FUNC_PAT": func.astype(float),
    }


def _measure(
    rng, lat: dict, idx: np.ndarray, site: np.ndarray, lot: int, station: str
) -> np.ndarray:
    """对 idx 指定的芯片做一次测量：真值 + 仪器噪声 + 测试系统缺陷。返回 (len(idx), n_tests)。"""
    m = len(idx)
    out = np.empty((m, len(TESTS)))
    noisy = lot == NOISE_LOT and station == NOISE_STATION
    offset = lot >= TIMING_FROM_LOT and station == TIMING_STATION
    contact_bad = (
        lot == CONTACT_LOT
        and station == CONTACT_STATION
        and (site[idx] == CONTACT_SITE) & (rng.random(m) < CONTACT_PROB)
    )
    for j, name in enumerate(NAMES):
        v = lat[name][idx].astype(float)
        if name == "CONT_VDD":
            v = v + rng.normal(0, MEAS_NOISE[name], m) + contact_bad * rng.uniform(1.8, 3.0, m)
        elif name == "IDD_STANDBY":
            v = v * np.exp(rng.normal(0, 0.03, m))
        elif name == "SCAN_FMAX":
            v = v * (TIMING_SCALE if offset else 1.0) + rng.normal(0, 45 if noisy else 15, m)
        elif name == "VMIN":
            v = v + rng.normal(0, 0.035 if noisy else 0.004, m)
        elif name in MEAS_NOISE:
            v = v + rng.normal(0, MEAS_NOISE[name], m)
        elif name == "PLL_LOCK":
            v = v * np.exp(rng.normal(0, 0.03, m))
        out[:, j] = v
    return out


def _fails(values: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (values < LO) | (values > HI)


def _first_fail_bin(fails: np.ndarray) -> np.ndarray:
    first = np.argmax(fails, axis=1)
    return np.where(fails.any(axis=1), HBIN_OF_TEST[first], 1)


def _fmt(value: float, name: str) -> str:
    return str(int(value)) if name in ("SRAM_BIST", "FUNC_PAT") else f"{value:.5g}"


def _limit_text(value: float, scale: float) -> str:
    return "-" if np.isnan(value) else f"{value * scale:g}"


def _tsr_lines(unit_override: dict[str, str] | None = None) -> list[str]:
    """测试项声明。unit_override 把某项改用毫安等更大的单位声明（数值同步缩小 1000 倍）。"""
    lines = []
    for no, name, unit, lo, hi, ms, _ in TESTS:
        scale = 1.0
        if unit_override and name in unit_override:
            unit, scale = unit_override[name], 1e-3
        lines.append(
            f"TSR|{no}|{name}|{unit}|{_limit_text(lo, scale)}|{_limit_text(hi, scale)}|{ms}"
        )
    return lines


def _die_block(i, geo, values, fails, hbin, part_ts, scale_cols) -> list[str]:
    x, y, site = geo["x"].iat[i], geo["y"].iat[i], geo["site"].iat[i]
    block = [f"PIR|{i + 1:04d}|{x}|{y}|{site}|{part_ts[i]}"]
    for j, (no, name, *_rest) in enumerate(TESTS):
        v = values[i, j] * (1e-3 if j in scale_cols else 1.0)
        block.append(f"PTR|{no}|{_fmt(v, name)}|{'F' if fails[i, j] else 'P'}")
    block.append(f"PRR|{i + 1:04d}|{hbin[i]}|{'P' if hbin[i] == 1 else 'F'}")
    return block


def _corrupt(block: list[str], rng: np.random.Generator) -> list[str]:
    kind = rng.choice(
        ["truncate", "sentinel", "nan", "drop_prr", "duplicate"], p=[0.35, 0.25, 0.10, 0.20, 0.10]
    )
    block = list(block)
    k = int(rng.integers(1, len(block) - 1))
    if kind == "truncate":
        block[k] = block[k][: int(rng.integers(4, len(block[k])))]
    elif kind in ("sentinel", "nan"):
        head = block[k].split("|")
        head[2] = "1.0E+38" if kind == "sentinel" else "nan"
        block[k] = "|".join(head)
    elif kind == "drop_prr":
        block = block[:-1]
    else:
        block = block + block
    return block


def _wafer_pass(rng, geo, lat, idx, lot, wafer, station, pass_no, start, values, unit_scale_cols):
    """渲染一个晶圆 pass 的日志行，并返回 (行, 每颗芯片的 hbin)。"""
    fails = _fails(values)
    hbin = _first_fail_bin(fails)
    secs = np.arange(len(idx)) * 0.4
    part_ts = (start + pd.to_timedelta(secs, unit="s")).strftime("%Y-%m-%dT%H:%M:%S").to_numpy()
    sub = geo.iloc[idx].reset_index(drop=True)
    dirty = rng.random(len(idx)) < DIRTY_DIE_RATE

    lines = _tsr_lines({"IDD_STANDBY": "mA"} if unit_scale_cols else None)
    lines.append(
        f"WIR|L{lot:02d}|{wafer:02d}|{station}|{PROGRAM}|{pass_no}|{start:%Y-%m-%dT%H:%M:%S}"
    )
    for i in range(len(idx)):
        block = _die_block(i, sub, values, fails, hbin, part_ts, unit_scale_cols)
        lines.extend(_corrupt(block, rng) if dirty[i] else block)
    lines.append(f"WRR|L{lot:02d}|{wafer:02d}|{pass_no}|{len(idx)}")
    return lines, hbin


def generate_lot(lot: int, cfg: Config, geo: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    rng = np.random.default_rng([cfg.seed, lot])
    lines = ["# DEMO-SOC SYNTHETIC ATE DATALOG v1 (all data simulated)"]
    lines += [f"HBR|{k}|{v}" for k, v in BINS.items()]
    truth_rows = []
    lot_start = pd.Timestamp(LOT_STARTS[lot - 1])
    site = geo["site"].to_numpy()

    for wafer in range(1, cfg.wafers_per_lot + 1):
        station = STATIONS[(wafer - 1) % len(STATIONS)]
        start = lot_start + pd.Timedelta(hours=8 + 5 * ((wafer - 1) // len(STATIONS)))
        lat = _latents(rng, geo, lot, wafer)
        all_idx = np.arange(len(geo))
        unit_cols = (
            {NAMES.index("IDD_STANDBY")} if (lot, wafer) == (UNIT_LOT, UNIT_WAFER) else set()
        )

        values = _measure(rng, lat, all_idx, site, lot, station)
        wafer_lines, hbin = _wafer_pass(
            rng, geo, lat, all_idx, lot, wafer, station, 1, start, values, unit_cols
        )
        lines += wafer_lines

        failed = np.flatnonzero(hbin != 1)
        if len(failed):
            retest = _measure(rng, lat, failed, site, lot, station)
            retest_start = start + pd.Timedelta(minutes=30)
            retest_lines, _ = _wafer_pass(
                rng, geo, lat, failed, lot, wafer, station, 2, retest_start, retest, unit_cols
            )
            lines += retest_lines

        # 无噪声、无测试系统缺陷时这颗芯片本来合不合格（上帝视角）
        clean_values = np.column_stack([lat[name] for name in NAMES])
        intrinsic_pass = ~_fails(clean_values).any(axis=1)
        truth_rows.append(
            pd.DataFrame(
                {
                    "lot_id": f"L{lot:02d}",
                    "wafer_id": f"{wafer:02d}",
                    "x": geo["x"],
                    "y": geo["y"],
                    "defect_type": lat["kind"],
                    "is_weak": lat["weak"],
                    "intrinsic_pass": intrinsic_pass,
                }
            )
        )
    return lines, pd.concat(truth_rows, ignore_index=True)


def write_logs(out_dir: Path, cfg: Config | None = None) -> pd.DataFrame:
    """生成全部 lot 日志与 truth.csv 到 out_dir，返回 truth。"""
    cfg = cfg or Config()
    out_dir.mkdir(parents=True, exist_ok=True)
    geo = _wafer_geometry()
    truths = []
    for lot in range(1, cfg.n_lots + 1):
        lines, truth = generate_lot(lot, cfg, geo)
        (out_dir / f"L{lot:02d}.log").write_text("\n".join(lines) + "\n", encoding="ascii")
        truths.append(truth)
    truth = pd.concat(truths, ignore_index=True)
    truth.to_csv(out_dir / "truth.csv", index=False)
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description="生成模拟芯片测试日志")
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "raw" / "chip"
    )
    args = parser.parse_args()
    truth = write_logs(args.out)
    size = sum(p.stat().st_size for p in args.out.glob("*.log"))
    n_lines = sum(1 for p in args.out.glob("*.log") for _ in p.open("rb"))
    print(
        f"wrote {len(list(args.out.glob('*.log')))} logs to {args.out}: {n_lines:,} lines, {size / 1e6:.1f} MB"
    )
    print(
        f"dies={len(truth):,} killer={(truth['defect_type'] != 'none').mean():.2%} weak={truth['is_weak'].mean():.2%}"
    )


if __name__ == "__main__":
    main()
