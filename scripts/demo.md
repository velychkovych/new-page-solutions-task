# Demo recording script

Covers the PLAN.md §7 submission checklist:

- one run that shows a **read**, a **proposed write**, the **approval**, and a **failure being
  handled** (`make chaos-on` mid-run);
- then `make chaos-off` and `make trace ID=<trace_id>`;
- a terminal screenshot of `make eval`, plus the committed `evals/results/latest.md`.

Save every screenshot or recording in `docs/screenshots/` with the file names below.

Sources for the commands and texts quoted here: `Makefile`, `web/index.html`,
`src/helpdesk/mock/seed_data.py`, `src/helpdesk/agent/proposal.py`,
`src/helpdesk/mcp_server/contract_tool.py`, `src/helpdesk/agent/tool_activity.py`,
`src/helpdesk/telemetry/show_trace.py`.

## Never on screen

- `.env` (it holds `MOCK_API_TOKEN`, `MOCK_ADMIN_TOKEN` and `ANTHROPIC_API_KEY`).
- `docker compose config` output (it prints the interpolated environment, tokens included).

## Pre-flight

1. `make setup`
   - Expected: `Created .env with generated MOCK_API_TOKEN, MOCK_ADMIN_TOKEN. Add ANTHROPIC_API_KEY to it.`
     (or `.env already exists; left unchanged.`)
2. Add `ANTHROPIC_API_KEY=...` to `.env` in an editor, off camera.
3. `make up` (or `make up OBS=1` to get the "open in Jaeger" link next to each trace id)
   - Expected: `mock`, `mcp`, `agent` (and `jaeger`) report healthy; the command returns.
4. `make seed`
   - Expected: `reset and seeded 15 tickets in /app/data/mock.sqlite`
5. `make chaos-off`
   - Expected: the active chaos config as JSON, with `"enabled":false`.
6. Open http://localhost:8080 next to a terminal in the repo root.
   - Expected: header "IT Helpdesk", status line `Enter your user id to start.` (first visit).

## The run (one session, user `alice`)

Seed data used: **INC-0003** "VPN keeps dropping", status `open`, reporter `alice`.

| # | Action | Expected on screen | Screenshot |
|---|---|---|---|
| 1 | Type `alice` in **User id**, press Tab. Click **New session**. | **Session id** gets a new random id. Status line: `New session.` | — |
| 2 | Type `VPN keeps dropping again since this morning` in **Message**. Click **Send** (or Ctrl/Cmd+Enter). | **Tool calls** panel: a `search_tickets` row, then `get_ticket` with `"ticket_id": "INC-0003"` (args, result, latency). This is the **read**. | `01-read.png` |
| 3 | Wait for the turn to end. | **Awaiting your approval** box: `Tool add_comment`, summary `Add a comment to INC-0003: "…"`, the args, and enabled **Approve** / **Decline** buttons. Status line: `Awaiting approval: Add a comment to INC-0003: "…"`. This is the **proposed write**. | `02-proposal.png` |
| 4 | In the terminal: `make chaos-on` | The active chaos config as JSON: `"enabled":true`, `"rate_limit_rate":0.2`, `"error_rate":0.2`, `"slow_rate":0.1`, `"slow_ms":3000`, `"seed":42`. | `03-chaos-on.png` |
| 5 | Click **Approve**. | A user message `Approve: Add a comment to INC-0003: "…"`. Status line and turn notes: `add_comment: Retrying (attempt 2 of 3)`, then `add_comment: Retrying (attempt 3 of 3)`. The `add_comment` row ends without an error, with its latency. The reply confirms the comment. This is the **approval** and the **failure being handled**. | `04-approval-retries.png` |
| 6 | In the terminal: `make chaos-off` | The chaos config JSON with `"enabled":false`. | — |
| 7 | Copy the trace id from the status bar (`Last trace: <id>`) or the turn header. Run `make trace ID=<trace_id>`. | First line `trace <id>: <n> spans, services: agent, mcp, mock`. A `mcp.tool [mcp]` line with `tool=add_comment attempts=3`, and under it `· retry.attempt attempt=… http.status_code=… wait_ms=…` lines. | `05-trace.png` |
| 8 | Only with `make up OBS=1`: click "open in Jaeger" next to the trace id. | The same trace in the Jaeger UI (http://localhost:16686). | `06-jaeger.png` |

Notes on step 2–3:

- The model decides the tool sequence. If the proposal is not `add_comment` on INC-0003,
  click **Decline**, click **New session**, and record again from step 2.
- While a change is pending, typing `approve` or `decline` in **Message** works the same as
  the buttons (page hint: `While a change is pending you can also type "approve" or "decline".`).

Notes on step 5 (why the retries are expected):

- `make chaos-on` sends the default `CHAOS_BODY` from the `Makefile`
  (`{"enabled":true,"rate_limit_rate":0.2,"error_rate":0.2,"slow_rate":0.1,"slow_ms":3000,"seed":42}`).
- Each `make chaos-on` restarts the seeded RNG (`ChaosEngine.configure` in
  `src/helpdesk/mock/chaos.py`). With seed 42 the first three mock requests draw: 429, 500,
  pass (checked by running `ChaosEngine.decide()` in process).
- So when the approved write is the first mock request after `make chaos-on`: attempt 1 gets
  429 (`Retry-After: 1`), attempt 2 gets 500, attempt 3 commits. One comment is written, under
  one idempotency key.
- Any other ticket request to the mock in between shifts the sequence. Causes:
  - a message sent from another tab or session (its tool calls reach the mock through MCP);
  - `make test-e2e`, which talks to the same mock and reconfigures chaos
    (`tests/e2e/test_chaos.py`);
  - `make eval` / `make eval-chaos`, which also turn chaos off, reset the mock DB before
    every case, set each chaos case's own config, and leave chaos off at the end
    (`evals/driver.py`, `evals/run.py`).
- `make seed` does not shift it: it resets the DB inside the mock container, not over HTTP,
  and `/health` and `/admin/*` are exempt from chaos (`is_chaos_exempt` in
  `src/helpdesk/mock/chaos.py`).
- Run `make chaos-on` again right before clicking **Approve**.

## Variant: a write that fails for good (honest failure, no write)

Optional, as a second recording. Same steps 1–3, then in step 4 run:

```bash
make chaos-on CHAOS_BODY='{"enabled":true,"rate_limit_rate":0,"error_rate":1,"slow_rate":0,"slow_ms":0,"seed":7}'
```

(the `ALL_FAIL` config of `tests/e2e/test_chaos.py`). Expected after **Approve**:

- `add_comment: Retrying (attempt 2 of 3)` and `add_comment: Retrying (attempt 3 of 3)`.
- The `add_comment` row is marked failed, with an error starting `ERROR UPSTREAM_UNAVAILABLE:`
  (or `ERROR TIMEOUT:` if the 15 s deadline is reached first).
- The error's next step tells the model to say the change could not be confirmed and not to
  make up a ticket id (`WRITE_NOT_CONFIRMED_NEXT_STEP`, `src/helpdesk/mcp_server/errors.py`).
  Check the reply on screen says so.
- `make chaos-off`, then `make trace ID=<trace_id>`: `tool=add_comment attempts=3
  error_code=UPSTREAM_UNAVAILABLE` (or `error_code=TIMEOUT`).

Screenshot: `04b-honest-failure.png`.

## Evals

1. `make eval` (live model, costs money, `REPEATS=3` by default). Needs `make up`; run it
   after the recording, because it resets the mock DB the browser agent uses.
2. Screenshot the terminal summary: `07-make-eval.png`.
3. Commit the generated results: `evals/results/latest.md` and `evals/results/latest.json`.

## Files in `docs/screenshots/`

| File | Shows |
|---|---|
| `01-read.png` | `search_tickets` + `get_ticket` rows in **Tool calls** |
| `02-proposal.png` | **Awaiting your approval** box for `add_comment` on INC-0003 |
| `03-chaos-on.png` | terminal: `make chaos-on` output |
| `04-approval-retries.png` | retry status lines and the completed `add_comment` row |
| `04b-honest-failure.png` | optional: failed `add_comment` and the honest reply |
| `05-trace.png` | terminal: `make trace ID=<trace_id>` span tree with retry events |
| `06-jaeger.png` | optional: the trace in Jaeger (`make up OBS=1`) |
| `07-make-eval.png` | terminal: `make eval` summary |
