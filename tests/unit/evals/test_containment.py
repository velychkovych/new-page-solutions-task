"""DeepEval is contained before it is imported: no telemetry, no .env loading, no folder in
the working directory, no Confident AI key. Each check runs in a fresh interpreter, since
this process imported both long ago."""

import json
import os
import subprocess
import sys
from pathlib import Path

import evals
from evals import CONFIDENT_KEY_VARS, DEEPEVAL_ENV, DEEPEVAL_HOME
from evals.stack import ROOT

_PROBE = """
import json, os, sys
import evals.scoring  # imports deepeval
import deepeval.constants, deepeval.telemetry
from deepeval.confident.api import is_confident
print(json.dumps({
    "env": {k: os.environ.get(k) for k in sys.argv[1:]},
    "hidden_dir": deepeval.constants.HIDDEN_DIR,
    "telemetry_opt_out": deepeval.telemetry.telemetry_opt_out(),
    "confident": is_confident(),
    "dotenv_secret": os.environ.get("HELPDESK_TEST_DOTENV_SECRET"),
}))
"""


def _python(code: str, cwd: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    base = {key: value for key, value in os.environ.items() if not key.startswith("DEEPEVAL_")}
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd=cwd,
        env={**base, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1", **env},
        capture_output=True,
        text=True,
        check=False,
    )


def test_importing_evals_sets_the_containment_in_this_process() -> None:
    for name, value in DEEPEVAL_ENV.items():
        assert os.environ[name] == value, name
    assert not any(name in os.environ for name in CONFIDENT_KEY_VARS)
    assert DEEPEVAL_HOME.is_absolute()
    assert not DEEPEVAL_HOME.is_relative_to(ROOT)
    assert evals.__doc__ is not None


def test_deepeval_sees_the_containment_and_writes_nothing_where_it_runs(tmp_path: Path) -> None:
    # A .env in the working directory, as the repo root has: deepeval must not load it.
    (tmp_path / ".env").write_text("HELPDESK_TEST_DOTENV_SECRET=leaked\n", encoding="utf-8")
    result = _python(
        _PROBE, tmp_path, *DEEPEVAL_ENV, "CONFIDENT_API_KEY", CONFIDENT_API_KEY="confident-key"
    )
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout)
    assert seen["env"] == {**DEEPEVAL_ENV, "CONFIDENT_API_KEY": None}
    assert seen["hidden_dir"] == str(DEEPEVAL_HOME)
    assert seen["telemetry_opt_out"] is True
    assert seen["confident"] is False
    assert seen["dotenv_secret"] is None
    assert sorted(p.name for p in tmp_path.iterdir()) == [".env"]


def test_a_planted_keyfile_and_inherited_deepeval_variables_are_ignored(tmp_path: Path) -> None:
    # deepeval merges `<cache folder>/.deepeval` into os.environ after evals has run; on a
    # shared /tmp another user could plant one, Confident AI key included.
    cwd, tmp = tmp_path / "cwd", tmp_path / "tmp"
    cwd.mkdir()
    home = tmp / DEEPEVAL_HOME.name
    home.mkdir(parents=True)
    (home / ".deepeval").write_text('{"confident_api_key": "planted-key"}', encoding="utf-8")
    inherited = {"DEEPEVAL_UPDATE_WARNING_OPT_IN": "1", "CONFIDENT_BASE_URL": "http://x.invalid"}
    result = _python(_PROBE, cwd, "CONFIDENT_API_KEY", *inherited, TMPDIR=str(tmp), **inherited)
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout)
    assert seen["hidden_dir"] == str(home)
    assert seen["confident"] is False
    assert seen["env"] == dict.fromkeys(["CONFIDENT_API_KEY", *inherited])


def test_the_deepeval_folder_is_per_user() -> None:
    if hasattr(os, "getuid"):
        assert DEEPEVAL_HOME.name.endswith(f"-{os.getuid()}")


def test_deepeval_imported_before_evals_is_refused(tmp_path: Path) -> None:
    # Contained by hand here, so this test sends no telemetry either.
    result = _python(
        "import deepeval, evals",
        tmp_path,
        DEEPEVAL_TELEMETRY_OPT_OUT="1",
        DEEPEVAL_DISABLE_DOTENV="1",
        DEEPEVAL_CACHE_FOLDER=str(tmp_path / "cache"),
    )
    assert result.returncode != 0
    assert "deepeval was imported before evals" in result.stderr
