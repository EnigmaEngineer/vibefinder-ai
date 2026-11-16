import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_harness_runs_and_meets_its_gate(tmp_path):
    out = tmp_path / "results.md"
    proc = subprocess.run([sys.executable, "scripts/evaluate.py", "--out", str(out)],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "== Summary ==" in proc.stdout and out.read_text().startswith("# Evaluation results")


def test_every_case_has_a_query_and_expectations():
    cases = json.loads((ROOT / "eval" / "cases.json").read_text())
    assert len({c["id"] for c in cases}) == len(cases)
    assert all("query" in c and c["expect"] for c in cases)
