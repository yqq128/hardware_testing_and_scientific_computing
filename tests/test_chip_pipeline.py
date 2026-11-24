import json

from chiptest import data, pipeline
from chiptest.paths import ChipPaths

EXPECTED_FIGURES = {
    "c01_yield_by_lot.png",
    "c02_fpy_heatmap_station_site.png",
    "c03_wafer_maps.png",
    "c04_radial_yield.png",
    "c05_station_parametrics.png",
    "c06_pat_residual.png",
    "c07_test_efficiency.png",
}


def test_pipeline_writes_all_artifacts(chip_run):
    paths, run = chip_run
    assert paths.clean.exists() and paths.tests.exists() and paths.summary.exists()
    assert {p.name for p in paths.figures.glob("*.png")} == EXPECTED_FIGURES
    assert {p.name for p in run.figures} == EXPECTED_FIGURES


def test_summary_json_is_valid_and_names_the_planted_problems(chip_run):
    paths, _ = chip_run
    summary = json.loads(paths.summary.read_text())
    assert summary["lot_process_shift"] == ["L07"]
    where = {(s["lot"], s["station"]) for s in summary["suspects"]}
    assert {("L04", "T02"), ("L06", "T03"), ("L05", "T01")} <= where
    assert [(c["lot_id"], c["wafer_id"]) for c in summary["clusters"]] == [("L03", "07")]


def test_clean_csv_round_trips_with_types_restored(chip_run):
    paths, run = chip_run
    loaded = data.load_clean(paths.clean)
    assert len(loaded) == len(run.clean)
    assert loaded["wafer_id"].str.len().eq(2).all()  # "05" 不能被读成整数 5
    assert str(loaded["test_time"].dtype).startswith("datetime64")
    assert loaded["tester_pass"].dtype == bool


def test_first_pass_rows_are_one_per_die(chip_run):
    _, run = chip_run
    first = data.first_pass(run.clean)
    assert not first.duplicated(data.DIE_KEY).any()


def test_missing_logs_are_reported_with_the_fix(tmp_path, capsys):
    code = pipeline.main(["--root", str(tmp_path)])
    assert code == 2
    assert "generate_chip_data.py" in capsys.readouterr().err


def test_no_plots_flag_skips_figures_but_still_writes_results(make_chip_root):
    paths = ChipPaths(make_chip_root("chip_no_plots"))
    assert pipeline.main(["--root", str(paths.root), "--no-plots"]) == 0
    assert paths.summary.exists() and paths.clean.exists()
    assert not paths.figures.exists()


def test_analysis_still_runs_without_truth(make_chip_root):
    """真实日志没有 truth.csv：报告应照常产出，只是没有过杀/漏测与 PAT 召回。"""
    paths = ChipPaths(make_chip_root("chip_no_truth"))
    paths.truth.unlink()
    result = pipeline.run(paths, make_plots=False, log=lambda _: None)
    assert result.report.evaluation is None
    assert "precision" not in result.report.pat["residual"]["k=4.0"]
    assert result.report.pat["residual"]["k=4.0"]["flagged"] > 0
