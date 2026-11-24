"""生成模拟 AOI 检测数据（全部为虚构，仅用于练习）。

植入的线索（供第 2、3 天分析时发现）：
- M03 在 2026-10-14（周三）06:00 之后光源逐渐衰减，亮度下降、良品误报上升，次日恢复。
- L03 在 2026-10-27 的批次表面反光，亮度偏高、对比度偏低，误报上升（训练期未见过的漂移）。
- 夜班基线误报略高。
- 同一工件被 AOI 报警后可能复测，所以 product_id 重复是合理的；inspection_id 重复才是脏数据。
- 真实缺陷带有独立于 AOI 分数的对比度特征（stain 最弱），所以对比度能辅助区分误报；
  误报只是噪声，没有这个特征。
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START = pd.Timestamp("2026-10-01")
N_DAYS = 30
MEAN_PER_MACHINE_DAY = 330
DEFECT_RATE = 0.01
AOI_THRESHOLD = 0.5

MACHINE_LINE = {
    "M01": "L01",
    "M02": "L01",
    "M03": "L02",
    "M04": "L02",
    "M05": "L03",
    "M06": "L03",
}
LINES = ["L01", "L02", "L03"]
MACHINE_BRIGHTNESS_OFFSET = {
    "M01": 0.0,
    "M02": 1.5,
    "M03": -1.0,
    "M04": 0.5,
    "M05": 2.0,
    "M06": -1.5,
}

DEFECT_TYPES = ["scratch", "dent", "stain", "crack"]
TRUE_TYPE_P = [0.40, 0.25, 0.20, 0.15]
FALSE_CALL_TYPE_P = [0.40, 0.15, 0.35, 0.10]
DEFECT_LATENT_MU = {"scratch": 1.8, "dent": 2.1, "stain": 1.6, "crack": 2.4}
DEFECT_LATENT_SIGMA = 1.2
DEFECT_CONTRAST_GAIN = {"scratch": 4.0, "dent": 5.0, "stain": 2.5, "crack": 5.5}
GOOD_LATENT_MU = -1.8
GOOD_LATENT_SIGMA = 1.1
NIGHT_SHIFT_BIAS = 0.15

DRIFT_MACHINE = "M03"
DRIFT_DATE = pd.Timestamp("2026-10-14")
DRIFT_START_HOUR = 6.0
DRIFT_LATENT_SHIFT = 1.1
DRIFT_BRIGHTNESS_DROP = 24.0

GLARE_LINE = "L03"
GLARE_DATE = pd.Timestamp("2026-10-27")
GLARE_LATENT_SHIFT = 0.5
GLARE_BRIGHTNESS_RISE = 10.0
GLARE_CONTRAST_DROP = 4.0

RETEST_PROB = 0.20
RETEST_SCORE_NOISE = 0.15
RETEST_FEATURE_NOISE = 0.5
MISSING_REFERENCE_RATE = 0.005
MISSING_FEATURE_RATE = 0.004
DUPLICATE_ROW_RATE = 0.002

COLUMNS = [
    "inspection_id",
    "product_id",
    "inspection_time",
    "line_id",
    "machine_id",
    "batch_id",
    "shift",
    "defect_type",
    "aoi_result",
    "reference_result",
    "aoi_score",
    "brightness",
    "contrast",
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def assign_shift(times: pd.Series) -> np.ndarray:
    hour = times.dt.hour.to_numpy()
    return np.select([hour < 8, hour < 16], ["night", "day"], default="swing")


def build_base(rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for day in range(N_DAYS):
        for machine in MACHINE_LINE:
            n = rng.poisson(MEAN_PER_MACHINE_DAY)
            seconds = rng.integers(0, 86400, n)
            parts.append(
                pd.DataFrame(
                    {
                        "day": day,
                        "machine_id": machine,
                        "inspection_time": START
                        + pd.to_timedelta(day, unit="D")
                        + pd.to_timedelta(seconds, unit="s"),
                    }
                )
            )
    df = pd.concat(parts, ignore_index=True)
    df["line_id"] = df["machine_id"].map(MACHINE_LINE)
    line_index = df["line_id"].map({line: i for i, line in enumerate(LINES)})
    df["batch_id"] = "B" + (df["day"] * len(LINES) + line_index + 1).astype(str).str.zfill(3)
    df = df.drop(columns="day").sort_values("inspection_time", kind="stable")
    df = df.reset_index(drop=True)
    df["shift"] = assign_shift(df["inspection_time"])
    df["product_id"] = "P" + (df.index + 1).astype(str).str.zfill(6)
    return df


def simulate_measurements(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    is_defect = rng.random(n) < DEFECT_RATE
    true_type = rng.choice(DEFECT_TYPES, size=n, p=TRUE_TYPE_P)
    claimed_type = rng.choice(DEFECT_TYPES, size=n, p=FALSE_CALL_TYPE_P)

    hours = (df["inspection_time"].dt.hour + df["inspection_time"].dt.minute / 60).to_numpy()
    on_drift_day = (
        (df["machine_id"] == DRIFT_MACHINE) & (df["inspection_time"].dt.normalize() == DRIFT_DATE)
    ).to_numpy()
    drift = np.where(
        on_drift_day, np.clip((hours - DRIFT_START_HOUR) / (24 - DRIFT_START_HOUR), 0, 1), 0.0
    )
    glare = (
        ((df["line_id"] == GLARE_LINE) & (df["inspection_time"].dt.normalize() == GLARE_DATE))
        .to_numpy()
        .astype(float)
    )

    defect_mu = pd.Series(true_type).map(DEFECT_LATENT_MU).to_numpy()
    good_mu = (
        GOOD_LATENT_MU
        + NIGHT_SHIFT_BIAS * (df["shift"] == "night").to_numpy()
        + DRIFT_LATENT_SHIFT * drift
        + GLARE_LATENT_SHIFT * glare
    )
    latent = np.where(
        is_defect,
        rng.normal(defect_mu, DEFECT_LATENT_SIGMA),
        rng.normal(good_mu, GOOD_LATENT_SIGMA),
    )
    score = sigmoid(latent)

    machine_offset = df["machine_id"].map(MACHINE_BRIGHTNESS_OFFSET).to_numpy()
    brightness = (
        125
        + machine_offset
        + rng.normal(0, 4, n)
        - DRIFT_BRIGHTNESS_DROP * drift
        + GLARE_BRIGHTNESS_RISE * glare
    )
    contrast_gain = pd.Series(true_type).map(DEFECT_CONTRAST_GAIN).to_numpy() * is_defect
    contrast = 22 + 0.8 * latent + contrast_gain + rng.normal(0, 3, n) - GLARE_CONTRAST_DROP * glare

    out = df.copy()
    out["aoi_score"] = np.round(score, 4)
    out["aoi_result"] = np.where(out["aoi_score"] >= AOI_THRESHOLD, "NG", "OK")
    out["reference_result"] = np.where(is_defect, "NG", "OK")
    out["defect_type"] = np.select(
        [is_defect, out["aoi_result"].to_numpy() == "NG"],
        [true_type, claimed_type],
        default="none",
    )
    out["brightness"] = np.round(brightness, 1)
    out["contrast"] = np.round(np.clip(contrast, 0, None), 1)
    return out


def add_retests(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    alarms = df[df["aoi_result"] == "NG"]
    picked = alarms[rng.random(len(alarms)) < RETEST_PROB].copy()
    n = len(picked)

    picked["inspection_time"] = picked["inspection_time"] + pd.to_timedelta(
        rng.integers(3 * 60, 15 * 60, n), unit="s"
    )
    picked["shift"] = assign_shift(picked["inspection_time"])
    new_score = sigmoid(
        logit(picked["aoi_score"].to_numpy()) + rng.normal(0, RETEST_SCORE_NOISE, n)
    )
    picked["aoi_score"] = np.round(new_score, 4)
    picked["aoi_result"] = np.where(picked["aoi_score"] >= AOI_THRESHOLD, "NG", "OK")
    picked["brightness"] = np.round(
        picked["brightness"] + rng.normal(0, RETEST_FEATURE_NOISE, n), 1
    )
    picked["contrast"] = np.round(picked["contrast"] + rng.normal(0, RETEST_FEATURE_NOISE, n), 1)

    keeps_type = (picked["reference_result"] == "NG") | (picked["aoi_result"] == "NG")
    picked["defect_type"] = picked["defect_type"].where(keeps_type, "none")
    return pd.concat([df, picked], ignore_index=True)


def inject_dirty_data(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    df.loc[rng.random(n) < MISSING_REFERENCE_RATE, "reference_result"] = np.nan
    df.loc[rng.random(n) < MISSING_FEATURE_RATE, "brightness"] = np.nan
    df.loc[rng.random(n) < MISSING_FEATURE_RATE, "contrast"] = np.nan

    dup = df[rng.random(n) < DUPLICATE_ROW_RATE]
    df = pd.concat([df, dup], ignore_index=True)
    return df.sort_values(["inspection_id"], kind="stable").reset_index(drop=True)


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    df = build_base(rng)
    df = simulate_measurements(df, rng)
    df = add_retests(df, rng)
    df = df.sort_values("inspection_time", kind="stable").reset_index(drop=True)
    df["inspection_id"] = "I" + (df.index + 1).astype(str).str.zfill(6)
    df = inject_dirty_data(df, rng)
    df["inspection_time"] = df["inspection_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df[COLUMNS]


def summarize(df: pd.DataFrame) -> None:
    ref = df.dropna(subset=["reference_result"]).drop_duplicates("inspection_id")
    actual_bad = ref["reference_result"] == "NG"
    called = ref["aoi_result"] == "NG"
    tp = int((actual_bad & called).sum())
    fn = int((actual_bad & ~called).sum())
    fp = int((~actual_bad & called).sum())
    tn = int((~actual_bad & ~called).sum())
    print(f"rows={len(df)} products={df['product_id'].nunique()}")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"recall={tp / (tp + fn):.2%} FPR={fp / (fp + tn):.2%} precision={tp / (tp + fp):.2%}")
    print(f"duplicate inspection_id rows: {int(df.duplicated('inspection_id').sum())}")
    print(df.isna().sum()[lambda s: s > 0].to_dict())


def main() -> None:
    out_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "aoi_inspection.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    summarize(df)


if __name__ == "__main__":
    main()
