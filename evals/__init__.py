"""Track F: the eval runner (`python -m evals.run`) and its cases (`evals/cases/*.yaml`).

DeepEval is contained here, in the package's `__init__`, because Python runs it before any
`evals.*` module, and so before the first `import deepeval`. Importing deepeval otherwise:

- merges `.env` from the working directory into `os.environ`, which would put the mock
  tokens and the model key into the runner's environment (the live tier keeps the mock
  tokens out of this process);
- sends usage telemetry and writes a `.deepeval/` folder into the working directory, the
  repository root under `make eval`;
- uploads results whenever `CONFIDENT_API_KEY` is set.

Its cache and results folders point at a directory under the system temp dir. The runner
measures metrics directly (evals.scoring) and never calls `deepeval.evaluate()`, so nothing
is written there either; the path is only the place a stray write would land instead of
the repository.
"""

import os
import sys
import tempfile
from pathlib import Path

if "deepeval" in sys.modules:
    # Too late: its settings, cache folder and .env loading were fixed at its import.
    raise ImportError("deepeval was imported before evals; import evals first")

DEEPEVAL_HOME = Path(tempfile.gettempdir()) / "helpdesk-evals-deepeval"
DEEPEVAL_ENV: dict[str, str] = {
    "DEEPEVAL_TELEMETRY_OPT_OUT": "1",
    "DEEPEVAL_DISABLE_DOTENV": "1",
    "DEEPEVAL_CACHE_FOLDER": str(DEEPEVAL_HOME),
    "DEEPEVAL_RESULTS_FOLDER": str(DEEPEVAL_HOME / "results"),
    "DEEPEVAL_NO_INSPECT_PROMPT": "1",
}
# Read by deepeval to upload test runs and traces to Confident AI; never set by us, and
# removed if the shell has one, so no eval data leaves the machine.
CONFIDENT_KEY_VARS = ("CONFIDENT_API_KEY",)

os.environ.update(DEEPEVAL_ENV)
for _name in CONFIDENT_KEY_VARS:
    os.environ.pop(_name, None)
