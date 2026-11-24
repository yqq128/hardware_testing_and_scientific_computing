import pandas as pd

from aoi import data, pipeline
from aoi.paths import Paths

EXPECTED_FIGURES = {
    "01_daily_alarms.png",
    "02_daily_false_positives.png",
    "03_fpr_by_machine.png",
    "04_alarms_by_defect_type.png",
    "05_fpr_by_shift_and_batch.png",
    "06_fpr_heatmap_machine_day.png",
    "07_anomaly_drilldown.png",
    "08_review_model_confusion.png",
    "09_pr_curve_and_threshold_tradeoff.png",
}


def test_pipeline_writes_all_artifacts(python_run):
    paths, run = python_run
    assert paths.clean.exists()
    assert paths.metrics.exists()
    assert {p.name for p in paths.dashboard.glob("*.png")} == EXPECTED_FIGURES
    assert {p.name for p in run.figures} == EXPECTED_FIGURES


def test_clean_csv_has_one_row_per_distinct_inspection(python_run):
    paths, _ = python_run
    raw = data.load_raw(paths.raw)
    clean = data.load_clean(paths.clean)
    assert len(clean) == raw["inspection_id"].nunique()
    assert clean["inspection_id"].is_unique
    assert pd.api.types.is_datetime64_any_dtype(clean["date"])


def test_missing_raw_data_is_reported_with_the_fix(tmp_path, capsys):
    code = pipeline.main(["--root", str(tmp_path)])
    assert code == 2
    assert "generate_data.py" in capsys.readouterr().err


def test_no_plots_flag_skips_figures_but_still_writes_metrics(make_root):
    paths = Paths(make_root("no_plots"))
    code = pipeline.main(["--root", str(paths.root), "--no-plots"])
    assert code == 0
    assert paths.metrics.exists()
    assert not paths.dashboard.exists()
