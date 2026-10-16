"""A fake-tier run of `python -m evals.run` writes only its two result files.

The runner runs in a child process with its own working directory, temp dir and home,
so anything DeepEval (or anything else) writes outside `--out` shows up there. Its stdout
must be the runner's own progress and summary, with no DeepEval banner in it.
"""

import os
import subprocess
import sys
from pathlib import Path

from evals.run import EXIT_OK
from evals.stack import ROOT

# What deepeval prints from evaluate() and its progress indicator.
BANNERS = ("You're running DeepEval", "Confident AI", "Evaluation completed", "hyperparameters")


def test_a_fake_run_writes_only_its_results(tmp_path: Path) -> None:
    cwd, tmp, home, out = (tmp_path / name for name in ("cwd", "tmp", "home", "out"))
    for directory in (cwd, tmp, home):
        directory.mkdir()
    env = {key: value for key, value in os.environ.items() if not key.startswith("DEEPEVAL_")}
    env |= {
        "TMPDIR": str(tmp),
        "HOME": str(home),
        "PYTHONPATH": str(ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CONFIDENT_API_KEY": "must-not-be-used",
    }
    run = subprocess.run(
        [sys.executable, "-m", "evals.run", "--fake", "--case", "status_question",
         "--out", str(out)],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=300, check=False,
    )  # fmt: skip
    assert run.returncode == EXIT_OK, run.stdout + run.stderr
    assert sorted(path.name for path in out.iterdir()) == ["latest.json", "latest.md"]
    for directory in (cwd, tmp, home):
        assert list(directory.iterdir()) == [], directory.name
    assert not (ROOT / ".deepeval").exists()
    lines = run.stdout.splitlines()
    assert lines[-6].startswith("Pass rate: 1/1 (100%)")
    assert lines[-1] == f"Results: {out}"
    for banner in BANNERS:
        assert banner not in run.stdout + run.stderr, banner
