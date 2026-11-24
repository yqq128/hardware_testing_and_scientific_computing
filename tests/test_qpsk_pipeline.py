"""从命令行验证 QPSK 产物，再与教学 Notebook 的独立实现对照。"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "qpsk" / "01_qpsk_link_ber.ipynb"


@pytest.fixture(scope="module")
def cli_output(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("qpsk_cli")
    result = subprocess.run(
        [sys.executable, "-m", "qpsk.pipeline", "--output-dir", str(output_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    with (output_dir / "qpsk_ber.csv").open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    return output_dir, rows


def test_cli_writes_ber_and_figures(cli_output):
    output_dir, rows = cli_output
    assert [int(row["eb_n0_db"]) for row in rows] == list(range(13))
    assert float(rows[0]["theory_ber"]) == pytest.approx(0.07864960352514257)
    assert float(rows[0]["simulated_ber"]) == pytest.approx(0.07865, abs=0.002)
    assert all(int(row["total_bits"]) == 400_000 for row in rows)
    assert any(
        0 < float(row["zero_error_upper_95"]) < 0.0000075
        for row in rows
        if int(row["bit_errors"]) == 0
    )
    for name in ("qpsk_ber.png", "qpsk_constellation.png"):
        assert (output_dir / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_cli_rejects_an_incomplete_symbol(tmp_path):
    output_dir = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "qpsk.pipeline",
            "--bits",
            "3",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "正偶数" in result.stderr
    assert not output_dir.exists()


@pytest.mark.slow
def test_cli_matches_executed_notebook(cli_output):
    _, rows = cli_output
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "import json\n"
            "print(json.dumps([error_counts.tolist(), expected_ber.tolist(), "
            "float(zero_upper_bound)]))"
        )
    )
    executed = NotebookClient(
        notebook, timeout=120, resources={"metadata": {"path": str(NOTEBOOK.parent)}}
    ).execute()
    error_counts, theory, zero_upper_bound = json.loads(executed.cells[-1].outputs[-1].text)
    assert [int(row["bit_errors"]) for row in rows] == error_counts
    assert [float(row["theory_ber"]) for row in rows] == pytest.approx(theory, rel=1e-12)
    for row in rows:
        errors, total = int(row["bit_errors"]), int(row["total_bits"])
        assert float(row["simulated_ber"]) == pytest.approx(errors / total, rel=1e-12)
        if errors:
            assert row["zero_error_upper_95"] == ""
        else:
            assert float(row["zero_error_upper_95"]) == pytest.approx(zero_upper_bound, rel=1e-12)
