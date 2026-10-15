"""Print a trace's span tree: `python scripts/show_trace.py <trace_id>` (see `make trace`).

The logic lives in `helpdesk.telemetry.show_trace`, where mypy checks it.
"""

import sys

from helpdesk.telemetry.show_trace import main

if __name__ == "__main__":
    sys.exit(main())
