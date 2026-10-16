"""Track F: the eval runner (`python -m evals.run`) and its cases (`evals/cases/*.yaml`).

DeepEval is contained here, in the package's `__init__`, because Python runs it before any
`evals.*` module, and so before the first `import deepeval`. Importing deepeval otherwise:

- merges `.env` from the working directory into `os.environ`, which would put the mock
  tokens and the model key into the runner's environment (the live tier keeps the mock
  tokens out of this process);
- sends usage telemetry and writes a `.deepeval/` folder into the working directory, the
  repository root under `make eval`;
- uploads results whenever `CONFIDENT_API_KEY` is set.

Its cache and results folders point at a per-user directory under the system temp dir. The
runner measures metrics directly (evals.scoring) and never calls `deepeval.evaluate()`, so
nothing is written there either; the path is only the place a stray write would land
instead of the repository. deepeval also merges a legacy keyfile (`<cache folder>/.deepeval`)
into `os.environ`, Confident AI key included, after this module has run; that is turned off,
since on a shared /tmp anyone could leave one there. Every DEEPEVAL_* and CONFIDENT_*
variable the shell set is removed first, so none of them (an update check that calls PyPI,
another Confident AI URL) changes what deepeval does here.
"""

import os
import sys
import tempfile
from pathlib import Path

if "deepeval" in sys.modules:
    # Too late: its settings, cache folder and .env loading were fixed at its import.
    raise ImportError("deepeval was imported before evals; import evals first")

# Per user where there are users, so two accounts on a shared /tmp do not collide on one
# folder. The suffix does not stop another account from creating this folder first; what
# matters is that deepeval never reads a legacy keyfile from it (disabled below).
_USER = f"-{os.getuid()}" if hasattr(os, "getuid") else ""
DEEPEVAL_HOME = Path(tempfile.gettempdir()) / f"helpdesk-evals-deepeval{_USER}"
DEEPEVAL_ENV: dict[str, str] = {
    "DEEPEVAL_TELEMETRY_OPT_OUT": "1",
    "DEEPEVAL_DISABLE_DOTENV": "1",
    "DEEPEVAL_DISABLE_LEGACY_KEYFILE": "1",
    "DEEPEVAL_CACHE_FOLDER": str(DEEPEVAL_HOME),
    "DEEPEVAL_RESULTS_FOLDER": str(DEEPEVAL_HOME / "results"),
    "DEEPEVAL_NO_INSPECT_PROMPT": "1",
}
# Variables deepeval reads that the shell may set: removed, then ours are set. Among them
# CONFIDENT_API_KEY, which uploads test runs and traces to Confident AI, so no eval data
# leaves the machine.
INHERITED_PREFIXES = ("DEEPEVAL_", "CONFIDENT_")
CONFIDENT_KEY_VARS = ("CONFIDENT_API_KEY",)

for _name in [name for name in os.environ if name.startswith(INHERITED_PREFIXES)]:
    del os.environ[_name]
os.environ.update(DEEPEVAL_ENV)
