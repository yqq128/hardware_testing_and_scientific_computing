"""用 Streamlit 自带的 AppTest 在不启动服务器的情况下执行页面脚本。

数据目录通过环境变量 AOI_ROOT 指向 python_run 产出的临时目录，页面不依赖仓库里的文件。
"""

import pytest
from streamlit.testing.v1 import AppTest

from aoi.paths import PROJECT_ROOT

APP = str(PROJECT_ROOT / "app.py")
CALCULATOR = str(PROJECT_ROOT / "pages" / "1_指标计算器.py")
DIAGNOSIS = str(PROJECT_ROOT / "pages" / "2_异常诊断.py")
MODEL = str(PROJECT_ROOT / "pages" / "3_复判模型.py")
CHIP = str(PROJECT_ROOT / "pages" / "4_芯片测试.py")

TIMEOUT = 90


def _run(script: str, root, monkeypatch) -> AppTest:
    monkeypatch.setenv("AOI_ROOT", str(root))
    return AppTest.from_file(script, default_timeout=TIMEOUT).run()


@pytest.mark.parametrize("script", [APP, CALCULATOR, DIAGNOSIS, MODEL])
def test_every_page_renders_without_exceptions(script, python_run, monkeypatch):
    paths, _ = python_run
    at = _run(script, paths.root, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("script", [APP, DIAGNOSIS, MODEL])
def test_pages_that_need_data_tell_the_user_how_to_get_it(script, tmp_path, monkeypatch):
    at = _run(script, tmp_path, monkeypatch)  # 空目录：没有 aoi_clean.csv
    assert not at.exception
    assert len(at.error) == 1
    assert "aoi.pipeline" in at.error[0].value
    assert "02_aoi_analysis" in at.error[0].value  # 两种运行方式都要告诉用户


def test_calculator_does_not_need_any_data_file(tmp_path, monkeypatch):
    at = _run(CALCULATOR, tmp_path, monkeypatch)
    assert not at.exception
    assert not at.error


def test_calculator_defaults_reproduce_the_handout_numbers(tmp_path, monkeypatch):
    at = _run(CALCULATOR, tmp_path, monkeypatch)
    values = {m.label: m.value for m in at.metric}
    assert values["Recall"] == "95.0%"
    assert values["FPR（良品误报率）"] == "5.0%"
    assert values["需人工复判"] == "590"


def test_calculator_shows_na_instead_of_crashing_on_zero_denominators(tmp_path, monkeypatch):
    at = _run(CALCULATOR, tmp_path, monkeypatch)
    labels = {w.label: w for w in at.number_input}
    labels["TP：报警且确实有缺陷"].set_value(0)
    labels["FP：报警但其实是良品（误报）"].set_value(0)
    at.run()
    assert not at.exception
    assert {m.label: m.value for m in at.metric}["Precision"] == "n/a"


def test_dashboard_main_page_shows_five_metric_cards(python_run, monkeypatch):
    paths, _ = python_run
    at = _run(APP, paths.root, monkeypatch)
    assert len(at.metric) == 5


def test_diagnosis_defaults_to_the_peak_cell_and_lets_user_switch(python_run, monkeypatch):
    paths, _ = python_run
    at = _run(DIAGNOSIS, paths.root, monkeypatch)
    machine_box = next(s for s in at.selectbox if s.label == "设备")
    assert machine_box.value == "M03"

    machine_box.select("M01")
    at.run()
    assert not at.exception


def test_model_page_shows_the_fixed_threshold_and_slider_does_not_change_it(
    python_run, monkeypatch
):
    paths, run = python_run
    at = _run(MODEL, paths.root, monkeypatch)
    chosen = run.experiment.chosen_threshold
    assert any(str(chosen) in info.value for info in at.info)

    at.slider[0].set_value(0.50)
    at.run()
    assert not at.exception
    # 滑块只改变验证集的观察窗口，最终选定的阈值和测试集指标保持不变
    assert any(str(chosen) in info.value for info in at.info)
    by_label = {m.label: m.value for m in at.metric}
    assert by_label["端到端缺陷保留率"] == f"{run.experiment.end_to_end.retention:.1%}"


def test_model_page_leakage_demo_runs_only_on_demand(python_run, monkeypatch):
    paths, _ = python_run
    at = _run(MODEL, paths.root, monkeypatch)
    assert "leakage" not in at.session_state

    next(b for b in at.button if b.label == "运行泄漏演示").click()
    at.run()
    assert not at.exception
    assert "leakage" in at.session_state


def test_chip_page_renders_and_defaults_to_the_noise_and_scratch_cases(chip_run, monkeypatch):
    paths, run = chip_run
    at = _run(CHIP, paths.root, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    by_label = {m.label: m.value for m in at.metric}
    assert by_label["首测良率 FPY"] == f"{run.report.overall['fpy']:.2%}"
    boxes = {s.label: s for s in at.selectbox}
    assert (boxes["lot"].value, boxes["测试项"].value) == ("L06", "VMIN")
    assert (boxes["晶圆所属 lot"].value, boxes["晶圆"].value) == ("L03", "07")


def test_chip_page_lets_user_switch_wafer_and_test(chip_run, monkeypatch):
    paths, _ = chip_run
    at = _run(CHIP, paths.root, monkeypatch)
    boxes = {s.label: s for s in at.selectbox}
    boxes["测试项"].select("SCAN_FMAX")
    boxes["晶圆所属 lot"].select("L01")
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_chip_page_tells_the_user_how_to_get_data(tmp_path, monkeypatch):
    at = _run(CHIP, tmp_path, monkeypatch)
    assert not at.exception
    assert len(at.error) == 1
    assert "generate_chip_data.py" in at.error[0].value
    assert "chiptest.pipeline" in at.error[0].value
