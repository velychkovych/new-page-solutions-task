# AI log

Factual log of AI-assisted work (see CLAUDE.md → Workflow). The author writes the reflection.

---

## 2026-10-15 — Phase 0 (scaffold)

### Delegated to AI (Claude Code, Opus 5.5)

- `uv init`, dependency resolution, and ruff/mypy/pytest config in `pyproject.toml`.
- `Makefile`, `Dockerfile`, `docker-compose.yml`, `.env.example`, `.gitignore`,
  `.dockerignore`, `.pre-commit-config.yaml`, `scripts/init_env.py`.
- `src/helpdesk/shared/` (settings, ids, errors, tickets, tools, approval, api, memory,
  trace_attributes) and its unit tests.
- `docs/CONTRACTS.md` from PLAN §3/§5/§6.
- README.md and AGENTCORE.md skeletons (headings, HUMAN markers, bullet facts, tables).
- A verification sweep of PLAN-named library APIs against the installed versions: 6 parallel
  agents, plus a second agent re-checking every discrepancy. Results are summarised under
  "Library notes" below.

### Independent review (before the gate)

- A second workflow ran 4 reviewers in parallel: contract fidelity, code vs contract, CLAUDE.md
  rules, and infra/gate. A skeptic agent then re-checked each finding.
- 14 findings were confirmed (9 distinct) and 18 were rejected. All confirmed findings were
  fixed:
  - **Blocker:** `Settings()` failed on the shipped `.env` and the compose env. Empty
    `CHAOS_SEED=` / `MODEL_TEMPERATURE=` were parsed as `""`. Fixed with
    `env_ignore_empty=True`, plus tests that load `.env.example` and each compose service's
    resolved environment.
  - **Blocker:** `make lint` failed on an E501 in the e2e test. It was edited after the last
    lint run, so an earlier version of this entry wrongly said lint passed.
  - The chaos RNG could be unseeded, which loosened PLAN's "seeded RNG". Now `CHAOS_SEED`
    defaults to `0` and every run is seeded.
  - `rate_limit_rate + error_rate ≤ 1` was in the contract but not enforced. It is now a
    validator on `ChaosConfig` and on `Settings`.
  - FastMCP's own argument validation adds a prefix that a `CallToolResult` cannot remove.
    CONTRACTS §2.4 now makes Track B responsible, with a required test.
  - SSE: `approval_required` is now required on every stream that ends in
    `awaiting_approval` (re-ask and stale-decision turns included).
  - `result_summary` is capped at 200 chars. `Ticket`/`TicketSummary` enforce the
    title/description length limits.
  - Added the missing HUMAN markers (AGENTCORE "Diagram", README eval subsections).
  - The README bullets "conventional commits" and "uv.lock committed" were false while
    nothing was committed. They are true after the Phase 0 commits below.

### Verified, and how

- `make lint typecheck test` passes. This includes:
  - `tests/unit/shared/*`
  - `tests/unit/scripts/test_init_env.py`
  - `tests/integration/test_compose_isolation.py`, which runs `docker compose config` with
    dummy tokens and `--env-file /dev/null`
- `docker compose config` is valid.
- `docker compose build` succeeds on `python:3.14-slim` (runs Python 3.14.8). The image
  imports `helpdesk`, `langgraph` and `mcp`.
- Network isolation was checked against running containers. The services don't exist yet,
  so a scratch compose override replaced each one with a stand-in HTTP server (the override
  is not committed). `tests/e2e/test_network_isolation.py` passed 5/5:
  - the agent cannot resolve `mock`
  - the agent can resolve `mcp` (positive control)
  - `mcp` can resolve `mock` (positive control)
  - the agent cannot reach the mock through `host.docker.internal`
  - the agent env has no `MOCK_API_TOKEN`
- `make chaos-on` reached the stand-in mock through `docker compose exec`. It got HTTP 501
  because the stand-in has no POST handler, which shows the exec path and tokens work.
- The Jaeger profile container becomes healthy and the UI answers 200.
- `uv.lock` passes `uv lock --check` under both local uv 0.10.0 and the image's uv 0.13.1.
- `.env.example` keys equal the `Settings` fields (unit test). It contains no secret values.

### Resolved versions

| Package | Version |
|---|---|
| Python | 3.14.3 local (uv-managed), 3.14.8 in `python:3.14-slim` |
| langgraph | 1.2.15 |
| langgraph-checkpoint | 4.3.0 |
| langgraph-checkpoint-sqlite | 3.1.2 |
| langchain-core | 1.6.11 |
| langchain | 1.4.6 |
| langchain-anthropic | 1.7.6 |
| anthropic | 1.15.0 |
| langchain-mcp-adapters | 0.3.2 |
| mcp | 1.30.0 |
| fastapi | 0.143.2 |
| pydantic / pydantic-settings | 2.14.1 / 2.16.0 |
| opentelemetry-sdk | 1.45.1 |
| opentelemetry-instrumentation-{fastapi,httpx} | 0.66b1 |
| openinference-instrumentation-langchain | 0.1.79 |
| langchain-openai / langchain-aws | 1.7.2 / 1.9.0 |
| ruff / mypy / pytest / respx | 0.17.1 / 2.4.0 / 9.1.1 / 0.23.1 |
| uv (Docker image) / uv_build | 0.13.1 / 0.13.1 |
| Jaeger image | `jaegertracing/jaeger:2.22.0` |

### Deviations from the plan, and why

1. **CLAUDE.md says Python 3.12; PLAN.md says 3.14.** I followed PLAN §1 (3.14):
   `requires-python`, mypy, ruff and the Docker image are all set to 3.14. The author
   should fix whichever file is stale.
2. **Python 3.15.0 was released on 2026-10-09.** PLAN §1 pins 3.14 explicitly and marks the
   stack "decided", so I did not switch. 3.14 resolved and built with no fallback needed.
3. **`MODEL_TEMPERATURE` defaults to empty (provider default) instead of "low".** The
   installed `langchain-anthropic` 1.7.6 raises at request time for any non-default
   `temperature` on `claude-sonnet-5-5` (`_assert_valid_model_configuration`).
   `claude-haiku-5-5` accepts a low value. **Needs the author's decision** (see "Unsure").
4. **The mock publishes no host port.** The plan implied a host-reachable mock for the chaos
   targets. A live check showed the agent container could reach a loopback-published mock
   port through `host.docker.internal` (Docker Desktop). Changes:
   - The port is not published.
   - `make chaos-on/off/seed` run inside the mock container with `docker compose exec`.
   - The eval runner must do the same (CONTRACTS §7).
   - This is stricter than the plan, not looser.
5. **`/admin/*` requires both the bearer token and `X-Admin-Token`.** I read PLAN §3.1
   ("Bearer on every route except `/health`") literally.
6. **`POST /admin/chaos` also accepts an optional `fail_after_commit_rate`**, in addition to
   the `CHAOS_FAIL_AFTER_COMMIT_RATE` env var, so the e2e test can toggle it at runtime.
7. **No `sse-starlette`.** FastAPI 0.143 ships `fastapi.sse.EventSourceResponse`.
8. **Added `langchain-openai` and `langchain-aws`.** The provider-factory unit test (PLAN §1)
   has to build `ChatOpenAI` and the Bedrock class.
9. **OpenTelemetry instrumentation packages exist only as `0.xxbN`.** That is their normal
   versioning; no non-beta release exists. This is an unavoidable exception to the
   "no pre-release" rule.
10. Additions beyond the plan's named items, all inside Phase 0 scope:
    - `Settings` fields: `CHAOS_RETRY_AFTER_S`, `FAKE_MODEL_SCRIPT`, `SLOW_TOOL_NOTICE_S`,
      `TRACE_RESULT_MAX_BYTES`, `JAEGER_UI_URL`, separate checkpoint/store DB paths.
    - `scripts/init_env.py`, so `make setup` creates `.env` with random tokens.
    - `make up OBS=1` for the Jaeger profile.
11. Contract details that PLAN left open and CONTRACTS.md now fixes (author should review):
    - Search: AND of whitespace terms; `id` ascending; the cursor is bound to `(q, status)`.
    - Chaos draw order: slow first, then one uniform draw for 429/5xx, so each rate is
      marginal; `rate_limit_rate + error_rate ≤ 1`.
    - The chaos RNG is always seeded: `CHAOS_SEED` defaults to `0`, so reproducibility does
      not depend on callers.
    - `/health` and `/admin/*` are exempt from chaos.
    - Fail-after-commit returns 503, only on executed (not replayed) writes.
    - Mock error-body codes.
    - A stale decision returns HTTP 200 with the current state, not 409.
    - Free text while an action is pending gets a deterministic re-ask with no model call.
    - Exact text-decision match (`"yes."` is not an approval).
    - Retry: TIMEOUT means the 15 s deadline was hit; UPSTREAM_UNAVAILABLE means the attempts
      ran out first.

### Unsure — author should review

- Decision 3 (temperature): keep sonnet-5-5 with the provider default, or use haiku-5-5 with a
  low temperature for evals.
- `MODEL_PROVIDER=bedrock` should map to `init_chat_model(..., model_provider="anthropic_bedrock")`
  (`langchain_aws.ChatAnthropicBedrock`, which keeps ChatAnthropic's cache_control and error
  handling) or `bedrock_converse`. Plain `bedrock` is the legacy InvokeModel class.
  Track C chooses, with this note.
- Every item in deviation 11.
- `docker compose` bind-mounts `./traces`, and containers run as uid 10001. This works on
  Docker Desktop (macOS). On a Linux host, the directory may need to be writable by that uid.

### Library notes for later tracks (verified against installed source)

- **LangGraph 1.2.15**
  - `recursion_limit` now defaults to **10007** (`_internal/_config.py:32`). Pass it
    explicitly from Settings in every run config.
  - `GraphRecursionError` is raised inside the `astream` generator.
  - `astream(..., version="v2")` yields typed `{type, ns, data}` parts. With v1, pass
    `stream_mode` as a **list**: a tuple loses the mode tag.
  - `interrupt()` re-runs the node on resume (verified at runtime). `interrupt(...,
    response_schema=...)` is new and optional.
  - Checkpoint serde: set `LANGGRAPH_STRICT_MSGPACK=true`, and type every state field
    concretely. A pydantic object under an `Any`-typed field comes back as a dict in strict
    mode. In default mode it logs "will be blocked in a future version".
  - The newer `context_schema`/`Runtime` and `ToolRuntime` exist as alternatives to
    `RunnableConfig`/`InjectedStore`.
  - `ToolNode`'s default `handle_tool_errors` re-raises everything except arg-validation
    errors.
- **AsyncSqliteSaver**
  - `from_conn_string` is an async context manager. Hold it in the FastAPI lifespan.
  - It uses WAL mode, which writes `*.sqlite-wal`/`-shm` files. `.gitignore` covers
    `*.sqlite*`.
- **AsyncSqliteStore TTL** works, with these caveats:
  - TTL is in minutes: 90 days = 129600.
  - `supports_ttl` wrongly reports False because of the MRO. Don't gate on it.
  - Expired rows are still returned until `sweep_ttl()` runs. Filter on read.
- **langchain-mcp-adapters 0.3.2**
  - An `isError` result becomes `ToolMessage(status="error")` with the original text.
  - Transport failures propagate as exceptions. The agent maps them to
    `UPSTREAM_UNAVAILABLE` without retrying.
  - There is no built-in way to strip injected args. Use
    `tool.model_copy(update={"args_schema": stripped})` for the model and call the full tool
    in `execute_write`.
  - Per-call headers (traceparent) and the `Callbacks.on_progress` contextvar routing only
    work on the per-call-session (`get_tools`/connection) path.
  - Set `sse_read_timeout` above the 15 s tool deadline.
- **mcp 1.30.0 (FastMCP)**
  - `raise ToolError` prefixes "Error executing tool X: ". Return
    `CallToolResult(isError=True)` to keep the exact contract text.
  - `custom_route` is untyped (mypy).
  - `session_manager.run()` can run only once per instance.
  - The DNS-rebinding guard is on for loopback hosts.
  - FastMCP reads `FASTMCP_*` from env and `.env`.
  - `ctx.info`/`report_progress` must be awaited.
  - `json_response=True` would drop mid-call progress.
- **langchain 1.4.6 / langchain-anthropic 1.7.6**
  - `init_chat_model` has no `fake` provider. The factory special-cases it.
  - Build `openai`/`bedrock_converse` with dummy creds + region in tests.
  - Prompt caching: `AnthropicPromptCachingMiddleware` works only with `create_agent`.
    Replicate its 3 steps by hand: `cache_control` on the last system block, on the last
    tool, and at the top level.
  - Cache writes = `cache_creation` + `ephemeral_5m/1h_input_tokens`. `input_tokens`
    already includes cached tokens.
  - The `anthropic` SDK 1.x uses `httpx2`, not `httpx`. httpx instrumentation and respx do
    not see Claude API calls. LLM spans come from OpenInference.
- **Fake chat models**
  - Neither fake implements `bind_tools`. Use a small test subclass.
  - `GenericFakeChatModel` **drops `tool_calls` when streaming**, and LangGraph
    `stream_mode="messages"` makes `ainvoke` stream. Use `FakeMessagesListChatModel`, or
    give the fake a `_stream` that keeps tool calls, for streamed tests.
- **OpenTelemetry**
  - httpx instrumentation does not capture headers unless env-configured. Redaction must be
    enforced in our JSONL exporter.
  - `BaseInstrumentor` is a process singleton.
  - Use `format_trace_id` for 32-hex ids.
  - OpenInference emits `llm.token_count.prompt_details.cache_read`/`cache_write`.
- **FastAPI SSE**
  - The generator is cancelled on client disconnect, so the graph run must be a separate
    task.
  - `: ping` comments arrive every 15 s.
  - All data is JSON-encoded.

### Suggestions

- A `POST /admin/reset` route would let the eval runner reset the mock over HTTP instead of
  `docker compose exec`. Not added, since it is not in PLAN §3.1.

---

## 2026-10-16 — Phase 0 follow-up: author decisions

- **Python:** 3.14 everywhere. `CLAUDE.md` on disk already said 3.14 when it was committed, so
  deviation 1 above no longer applies. `.python-version`, `requires-python`, mypy, ruff and the
  Docker image are all 3.14. Python 3.15 is not used.
- **Temperature:** not used anywhere.
  - `MODEL_TEMPERATURE` was removed from `Settings`, `.env.example` and compose.
  - `test_no_temperature_setting` guards against it coming back.
  - Following the author's instruction ("temperature shouldn't be used anywhere"), the
    PLAN.md §1 line now says "Do **not** set `temperature`".
  - CONTRACTS §7 now says no model gets a temperature.
- **Contracts:** the deviation-11 choices are accepted as written.
- **Bedrock:** `MODEL_PROVIDER=bedrock` maps to `init_chat_model(..., model_provider="anthropic_bedrock")`.
  - Verified against the installed source: `langchain/chat_models/base.py:58`.
  - A runtime check built `langchain_aws.ChatAnthropicBedrock`, an `isinstance` of
    `ChatAnthropic`, with no credentials.
  - The mapping is recorded in CONTRACTS §7 (a provider table), in PLAN.md §1 and in a
    `ModelProvider` comment.
- Verified: `make lint typecheck test` → `All checks passed!`, `Success: no issues found in
  15 source files`, `103 passed`. `docker compose config` is valid.

---

## 2026-10-16 — Phase 1 (tracks A–E)

### Delegated to AI (Claude Code, Opus 5.5)

- A single workflow ran 15 agents. Each track went through three stages:
  1. **Build:** one agent per track, in its own git worktree and branch, building only
     against `docs/CONTRACTS.md` and `src/helpdesk/shared/`.
  2. **Review:** an independent reviewer that changed nothing. It checked the code against
     the contracts, the tests PLAN lists for the track, the CLAUDE.md rules and the installed
     library source, and ran mutation probes on a scratch copy.
  3. **Fix:** a fixer re-checked each finding, fixed the confirmed ones with tests and
     rejected or deferred the rest with a reason.
- Agents were not allowed to edit `shared/`, `docs/`, root config or other tracks. The
  orchestrator cherry-picked the 30 track commits onto `master` with no conflicts and wrote
  this entry.

### Per track

| Track | Paths | Tests | Review findings | Fixed / rejected / deferred |
|---|---|---|---|---|
| A — Mock | `src/helpdesk/mock/`, `tests/unit/mock/`, `tests/integration/mock/` | 210 unit + 2 integration | 6 (1 major, 5 minor) | 5 / 1 partly (A-4: no HMAC) / 1 (A-6) |
| B — MCP server | `src/helpdesk/mcp_server/`, `tests/unit/mcp_server/` | 126 | 3 (1 major, 2 minor) | 3 / 0 / 0 |
| C — Agent core | `src/helpdesk/agent/`, `tests/unit/agent/` | 101 | 4 (2 major, 2 minor) | 4 / 0 / 0 |
| D — Web page | `web/`, `tests/unit/web/` | 129 (45 run the page's JS in node) | 4 minor | 4 / 0 / 0 |
| E — Telemetry | `src/helpdesk/telemetry/`, `tests/unit/telemetry/`, `scripts/show_trace.py` | 64 (1 strict xfail) | 7 (4 major, 3 minor) | 5 / 0 / 2 (E-4, E-7) |

Notable fixes made after review:

- **A:**
  - A body that is not valid UTF-8 returned `400 upstream_error`; it is now
    `422 invalid_request`.
  - Trailing-slash paths now return an ApiError body instead of a 307.
  - A cursor with an offset that was never issued is now rejected.
  - New tests pin the chaos draw order and that a write and its key commit together.
- **B:**
  - Every unexpected exception now stays in the `ERROR <CODE>` format.
  - Arguments are validated in strict mode, so `page_size=true` and `"7"` are rejected.
  - The httpcore-log secrecy test no longer opens a socket.
- **C:**
  - The deadline reply no longer says "nothing was changed" while a write is in flight.
  - The checkpoint-strictness tests now fail under a permissive serializer.
  - A proposal cut off before its `interrupt()` is shown before it can be approved.
  - New tests cover the agent node's prompt caching and history trimming.
- **D:**
  - A turn now ends at `done`.
  - A failed session load clears the stale view.
  - The Jaeger link is hardened, and the `?jaeger=` override was removed.
  - New node-backed tests run the page's own JS.
- **E:**
  - The OTLP tests now inspect the processors actually registered on the provider.
  - `show_trace` skips lines that are not valid UTF-8.
  - One lone-surrogate span no longer drops its whole JSONL batch.
  - The ASGI `http send`/`http receive` spans are excluded.
  - `instrument_app` also accepts the Starlette app that FastMCP builds.

### Verified, and how

- Each track's worktree passed `make lint typecheck test` after its fixes (track
  isolation). Each reviewer re-ran the gate independently.
- On merged `master`:
  - `make lint` → `All checks passed!`, `113 files already formatted`
  - `make typecheck` → `Success: no issues found in 63 source files`
  - `make test` → `734 passed, 1 xfailed, 7 warnings`
  - The suite was run three times with the same result and no flakes. Nothing was skipped
    (node v26 is on PATH locally).
- The 7 warnings are aiosqlite's default-datetime-adapter `DeprecationWarning`. It is
  raised inside langgraph's `AsyncSqliteStore`, not in our code.
- Every track's diff stays inside its own paths (`git diff --name-only` per branch).
- Mutation probes (reviewers):
  - **A:** 12 of 17 mutants were caught. Of the 5 survivors, 2 are equivalent mutants and
    2 are now covered by the fixer's new tests. The fifth, `==` instead of
    `compare_digest`, cannot be seen by a test.
  - **B:** all 9 mutants were caught.
  - **C:** 2 mutants survived: a permissive serializer and the prompt-caching call removed.
    Both are now covered.
  - **E:** 3 OTLP-wiring mutants survived; they are now covered.
- **D**, manual checks: headless Chrome rendered every fixture. An injection string
  `<img src=x onerror=…>` rendered as text, with 0 `<img>` elements. Driving the page over
  CDP, a triple-click on Approve sent exactly one request.

### Deviations from the plan / contracts, and why

- **A:**
  - A 405 returns `method_not_allowed`, which is not in the CONTRACTS §1.7 table.
  - The Bearer scheme is matched case-insensitively.
  - The cursor fingerprint is computed over the normalised filter (casefolded terms, sorted
    statuses); `limit` is not part of it.
  - A malformed path ticket id → 404, not 422.
  - The OpenAPI/docs routes are off.
  - Fail-after-commit is drawn in the write route after the commit (from `chaos.py`, same
    RNG), because middleware cannot tell an executed write from a replay.
- **B:**
  - The tools are `ContractTool` subclasses of FastMCP's `Tool`, registered with
    `FastMCP(tools=[...])`. Their `inputSchema` is the shared model's JSON schema. This is
    how invalid arguments produce the exact `ERROR INVALID_ARGUMENT` text (CONTRACTS §2.4).
  - FastMCP's `pre_parse_json` is not applied.
  - `Retry-After` is honoured only as delta-seconds.
  - Other transport errors and unexpected statuses become `UPSTREAM_UNAVAILABLE` after one
    attempt.
  - Writes get a dedicated next step: "could not be confirmed … do not make up a ticket id".
- **C:**
  - Strict checkpoint deserialisation is enforced in code with
    `JsonPlusSerializer(allowed_msgpack_modules=…)` on a hand-built `AsyncSqliteSaver`.
    `from_conn_string` cannot take a serializer, and `LANGGRAPH_STRICT_MSGPACK` would need
    a compose change.
  - Rejected batches go through the existing `tools` node, so the topology matches
    CONTRACTS §5.1 exactly.
  - `approval` returns `Command(goto=…)` after `interrupt()`.
  - The system prompt is a `SystemMessage` with a fixed id, kept in state.
  - Unanswered tool calls are patched in the model input after a deadline or recursion
    stop.
  - An `execute_write` that never finished is re-run before the next request.
  - `tool_calls` lists only calls that actually ran.
  - `python -m helpdesk.agent` is not created yet; it needs the Phase 2 MCP provider.
- **D:**
  - The page falls back to `/chat` only on a network error, 404/405/501, or a 2xx response
    that is not SSE. It never re-sends after a stream has started.
  - The `.sse` fixtures end with a comment line, because the pre-commit
    `end-of-file-fixer` would otherwise cut off the final frame.
  - Added a CSP meta tag.
  - Ids are kept in `localStorage`.
  - The Jaeger URL comes from `<meta name="helpdesk-jaeger-ui-url">`.
- **E:**
  - The header drop list also covers `Proxy-Authorization` and `X-Api-Key`.
  - Scrubbing also covers span and event names, the status description and links.
  - The JSONL batch delay is 200 ms.
  - `/health` is excluded from server spans.
  - The service name must match `^[a-z0-9][a-z0-9_-]{0,63}$`.
  - bytes values are written as base64.
  - A second, conflicting setup raises `TelemetryConfigError`.

### Unsure — author should review

1. **Cache-write tokens on the streaming path (E-4; blocks Phase 2 tracing).**
   - langchain-anthropic 1.7.6 sets `input_token_details.cache_creation=0` and moves cache
     writes to `ephemeral_5m/1h_input_tokens`.
   - OpenInference 0.1.79 reads only `cache_creation`, so
     `llm.token_count.prompt_details.cache_write` is missing when the agent streams. The
     non-streaming path is correct.
   - It is pinned by a strict xfail. Options:
     - (a) a Track E span processor that derives the count;
     - (b) Track C sets the attribute itself;
     - (c) accept the gap and document it.
2. **CONTRACTS §1.7 gaps.**
   - 405 `method_not_allowed` (A-6).
   - An undecodable body → 422 `invalid_request` (A-1).
3. **Cursor (A-4).** "Only the value of a previous `next_cursor` is valid" (§1.4) is
   enforced as an unsigned cursor bound to the fingerprint plus a check that the offset was
   actually issued. Reading it literally would need a signed cursor and a new secret in
   `Settings`.
4. **MCP error codes.**
   - An unexpected server exception maps to `UPSTREAM_UNAVAILABLE`, because `ErrorCode` has
     no `INTERNAL`.
   - A clipped per-attempt timeout that fires at the deadline gives `TIMEOUT`.
   - Strict validation rejects `page_size=7.0`.
   - Setting `strict=True` in `shared/tools.py` would make the agent and the MCP server
     agree on lax inputs. That is a shared change.
5. **Agent choices left to confirm.**
   - Rejected or unknown tool calls are not in `ChatResponse.tool_calls`.
   - A text decision is not added to the history, which keeps Claude's tool_use/tool_result
     pairing intact.
   - The decline `ToolMessage` has `status="success"`.
   - There are three deadline replies (no write / write unfinished / write saved).
   - The behaviour of a turn cut off between `propose` and `interrupt()`.
   - Should CONTRACTS §3.4 list these?
6. **`trace_id` without telemetry.** With no OTel SDK provider installed, it is 32 zeros.
   Phase 2 must call `setup_telemetry` before the agent serves.
7. **Telemetry lifecycle (E-7).** `setup_telemetry` and `shutdown_telemetry` must be called
   once per process, from each `python -m helpdesk.<svc>` entry point, not from a lifespan.
8. **FastAPI 0.143's built-in telemetry.**
   - It is on by default with the global provider and may duplicate the
     `opentelemetry-instrumentation-fastapi` spans.
   - Its `logs` option records validation input values.
   - Phase 2 should decide on `FastAPI(telemetry=…)`.
9. **Web.**
   - The JS behaviour tests skip when `node` is absent (CI, the slim image). Should node be
     a required dev tool?
   - The `end-of-file-fixer` hook should exclude `web/fixtures/*.sse`.
   - Phase 2 must fill the Jaeger meta tag when it serves `web/`, and decide whether to
     serve `web/fixtures/`.
   - It is not in the contracts that an SSE `error` is always followed by
     `done.status == "error"`.
10. **OpenInference records conversation text untruncated** (`input.value`/`output.value`).
    Secrets are scrubbed, but user text is kept.

### Library notes added in Phase 1 (verified against installed source)

- `AsyncSqliteSaver.from_conn_string` cannot take a serializer. Build
  `AsyncSqliteSaver(conn, serde=…)` on an aiosqlite connection instead.
- mcp 1.30.0:
  - Subclassing `mcp.server.fastmcp.tools.Tool` and passing `FastMCP(tools=[...])` is the
    hook used for exact argument-error text.
  - `Tool.run` is semi-internal; the snapshot test and the real-call-path tests guard it.
- FastAPI 0.143.2 ships built-in OpenTelemetry (`fastapi/telemetry`). It is on by default
  and uses the global provider.
- An OTel `ProxyTracer` from a module-level `trace.get_tracer()` binds permanently to the
  first real global provider. Tests should get the tracer at use time.
- `opentelemetry-instrumentation-httpx` 0.66b1 also ships `HTTPX2ClientInstrumentor`, for
  the anthropic SDK's `httpx2`. It is not used.

### Suggestions

- An `chaos.injected` span event name in `shared/trace_attributes.py`, so chaos failures
  show up in mock traces (A).
- MCP `ToolAnnotations` (`readOnlyHint` / `idempotentHint`) (B).
- `LANGGRAPH_STRICT_MSGPACK=true` in compose, for defence in depth (C).
- Truncate OpenInference inputs and outputs with `TraceConfig`, if conversation text in
  traces is unwanted (E).
- Evals: a "network" search at the default limit puts INC-0010, INC-0014 and INC-0015 on
  page 2. INC-0013 is the prompt-injection fixture (A).

---

## 2026-10-16 — Phase 1 follow-up: the author's answers to the open questions

### Author decisions (numbered as in the Phase 1 "Unsure" list)

| # | Decision | Result |
|---|---|---|
| 1 | Cache-write tokens on the streaming path → a telemetry span processor | Done (Track E) |
| 2 | Add 405 / undecodable body to the contract only if something uses them | No contract change. No client relies on 405, and an undecodable body is already covered by the existing 422 `invalid_request` row |
| 3 | Cursor: change it only if it causes bugs, otherwise leave it | Left as is. The strict offset ≥ 1 still works when `page_size` changes mid-walk and when tickets are created between pages |
| 4 | Explicit errors the agent can handle; strict mode only if it prevents bugs | New `INTERNAL` code; specific error texts (Tracks B, C). No strict shared models: measured on JSON inputs, strict and lax accept the same agent write arguments, and the MCP server already validates strictly |
| 5 | The agent handles bad responses as needed | Done (Track C) |
| 6 | Telemetry is set up when the app starts | Done for mock, MCP and agent (Tracks A, B, C on top of E) |
| 7 | Node required; pre-commit excludes `.sse`; Phase 2 serves `web/fixtures/` | Node from `nodejs-wheel-binaries` (dev dependency); the fixtures exclusion is done; serving the fixtures is a Phase 2 task |
| 8 | Document that traces keep conversation text untruncated | A fact bullet in README "Known limitations" |

### Delegated to AI, and what changed

- **Orchestrator (on master):**
  - `ae9b408`: the page behaviour tests use the pinned `node` 24.19.0 from `.venv`
    (`nodejs_wheel.executable.ROOT_DIR`) and fail instead of skipping when it is missing. The
    pre-commit end-of-file and whitespace hooks now exclude `web/fixtures/*.sse`.
  - `cfa57b3`: `ErrorCode.INTERNAL`; CONTRACTS §2.4 gets an `INTERNAL` row and a
    specific-error-text rule.
  - `3504caf`: README fact about untruncated conversation text in traces.
  - `a4ce2e4`: contract changes:
    - CONTRACTS §3.1: `tool_calls` lists rejected and unparseable calls, with `error` set and
      `latency_ms: 0`.
    - CONTRACTS §5.3: `invalid_tool_calls` are answered, and every error `ToolMessage` and
      the decline have `status="error"`.
    - CONTRACTS §6: telemetry is set up at app start.
- **Two workflows, each running build → review → fix in isolated worktrees, 12 agents in total:**
  - **E (telemetry):**
    - `CacheWriteSpanProcessor` derives the cache-write count in the export path. In SDK
      1.45.1 a span is immutable before any processor hook runs, so it cannot be set
      earlier.
    - A process-global `DelegatingTracerProvider` makes setup and shutdown repeatable and
      reference-counted per service.
    - New `telemetry_lifespan()`.
    - `instrument_app()` is now called from the app factory.
    - New `fastapi_telemetry_config()` turns off FastAPI's native tracing and logs.
    - Review: 5 findings; 4 fixed, 1 was an ai-log request.
  - **A (mock):** telemetry at app start. Tests cover joining an incoming traceparent,
    `/health` not traced, and no token in the JSONL even with all header capture on.
    Review: 1 finding, fixed.
  - **B (MCP):**
    - Telemetry is chained into the FastMCP Starlette app's lifespan.
    - `CaptureTraceContext` middleware: in mcp 1.30.0, tool calls run in the task started by
      the `initialize` request, so without it every tool span joined that request's trace.
    - `INTERNAL` for unexpected failures.
    - Every error text names the tool, the cause, the attempts and one next step. Write
      errors distinguish "not applied" from "could not confirm whether the change was
      applied".
    - Review: 5 findings, all fixed. Model-supplied text is kept out of error texts, the
      write next steps say not to propose the same write again, and FastMCP's
      `Unknown tool` text is mapped into the contract format.
  - **C (agent):**
    - Telemetry at app start, so `trace_id` is real.
    - New modules `model_calls`, `tool_errors`, `tool_results` and `final_reply`.
    - Every tool call is answered exactly once, including `invalid_tool_calls`, a missing
      id (given a generated id) and calls cut off by a truncated reply.
    - Malformed or empty tool results become `INTERNAL`.
    - A blank final answer gets a deterministic reply.
    - Review: 3 findings, all fixed.

### Verified, and how

- Merged master:
  - `make lint` → `All checks passed!`, `131 files already formatted`
  - `make typecheck` → `Success: no issues found in 72 source files`
  - `make test` → `889 passed, 7 warnings` (3 runs, identical)
- No test writes `./traces`, and no test exports OTLP.
- Every branch's diff stays inside its track's paths.
- Reviewers ran mutation probes on scratch copies:
  - E: 16/16 caught, after one test was added.
  - A: dropping `fastapi_telemetry_config()` was not caught; a test was added.
  - B and C: the surviving mutants were covered by the fixes above.

### Deviations, and why

- **Crash rule for writes (C):**
  - Only a `BaseException` (process death, cancellation, deadline) leaves `execute_write`
    to be re-run with the same key.
  - An ordinary exception from the provider is now a failed write: an `INTERNAL` "could not
    be confirmed" `ToolMessage`, `pending_action` cleared, no retry.
  - The crash/restart tests now simulate a crash with a `BaseException` subclass.
- **Non-contract error text (C):** an `is_error` result whose text is not in the contract
  format is never passed to the model; it becomes `INTERNAL`.
- **Truncated reply (C):** a truncated reply has all its calls rejected, not only the last
  one.
- **Cancellation (C):** a `CancelledError` raised by a node gets an `INTERRUPTED_*` reply with
  status `error`, instead of a 500.
- **Entry point (B):** `uvicorn.Config(lifespan="on")`. With `"auto"`, a startup error made
  uvicorn serve without telemetry and without the session manager.
- **Telemetry lifecycle (E):** shutdown no longer uninstruments httpx and LangChain; they are
  instrumented once per process. `instrument_app` raises if the app has already started.

### Library notes added

- opentelemetry-sdk 1.45.1: a span's attributes are frozen before any processor hook runs
  (`_on_ending` is private). A derived attribute has to go on a copy in the export path.
- opentelemetry-api 1.45.1: `ProxyTracer` binds permanently to the first global provider.
  OpenInference's `OITracer` calls `tracer.__class__.start_span(proxy, …)` to use its
  seed-independent id generator.
- mcp 1.30.0: `StreamableHTTPSessionManager` runs tool calls in the task started by the
  `initialize` request, so contextvars (including the OTel context) come from that request.
- uvicorn 0.54: with `lifespan="auto"`, an error raised before `lifespan.startup.failed` is
  sent is treated as "lifespan unsupported".
- FastAPI 0.143.2 copies the `telemetry=` mapping it is given.

### Unsure — author should review

1. **Unconfirmed write, then a user re-ask (B-2 follow-up).** After a write times out or
   exhausts its retries, the next step tells the model not to propose it again unless the
   user asks. If the user does ask, the new proposal gets a new idempotency key. A write that
   did commit could then be applied twice. Preventing this would need agent-side work, for
   example reusing the key for an identical proposal.
2. **Status for unusual endings.** A blank final answer returns status `ok`; `INTERRUPTED_*`
   replies return status `error`. CONTRACTS §3.4 does not list either.
3. **Contract table rows.**
   - FastMCP's "unknown tool" maps to `INVALID_ARGUMENT`.
   - A mock 404 on search/create (only possible with a wrong `MOCK_URL`) stays `NOT_FOUND`,
     with text that says misconfiguration.
   - Neither is a row in CONTRACTS §2.4.
4. **`unnamed_tool` placeholder.** An `invalid_tool_call` with no name is given this name,
   so its tool_use and tool_result stay paired. This is not verified against the live
   Claude API.
5. **Fragile hooks that a library upgrade could break:**
   - B replaces the Starlette `app.router.lifespan_context` (public, but not a documented
     extension point).
   - A mock test reads two FastAPI 0.143 internals.
6. **Phase 2 in-process tests** cannot run the lifespans of two different services at the
   same time (one service per process, by design).

---

## 2026-10-16 — Phase 2 (integration)

### Delegated to AI (Claude Code, Opus 5.5)

- Two workflows, 15 agents in total:
  1. **Build:** three agents, one after the other on `master`. Each built against
     `docs/CONTRACTS.md`, committed per step and passed `make lint typecheck test` before the
     next one started:
     - MCP provider, entry point and tracing;
     - `/chat/stream` and serving `web/`;
     - the e2e stack and tests, run against Docker.
  2. **Review:** five reviewers, each in its own git worktree, with one lens each: contract,
     CLAUDE.md rules and security, streaming and concurrency, the MCP provider, and tests
     (mutation probes). An adversarial verifier then checked each reviewer's findings. One
     fixer applied the confirmed findings on `master` and an independent rechecker checked
     the fix commits.
- Orchestrator:
  - Fixed the rechecker's two findings itself (`0cd0a29`).
  - Removed a reviewer worktree that held only an uncommitted probe file, which made
    `ruff check .` fail.
  - Wrote this entry.
- No agent edited `shared/`, `docs/CONTRACTS.md`, README/AGENTCORE prose, `mcp_server/`,
  `mock/`, `telemetry/` or `web/`.

### What was built (PLAN §4 Phase 2)

| Item | Code | Tests |
|---|---|---|
| 1. MCP tool provider | `agent/mcp_provider.py` (`McpToolProvider`); `ToolProvider.call(..., on_progress=None)` | `tests/unit/agent/test_mcp_provider.py` (real FastMCP app in process) |
| 2. One trace across agent → MCP → mock | A new MCP session per tool call, opened inside the calling node (see library notes) | `tests/integration/test_trace_propagation.py` (three real processes on loopback); `tests/e2e/test_happy_path.py` (compose) |
| — Entry point | `agent/__main__.py` (`python -m helpdesk.agent`) | `tests/unit/agent/test_entry_point.py`; exit 3 when MCP is down at startup (integration) |
| 3. `/chat/stream` | `agent/stream.py` (`TurnEvents`), `agent/tool_activity.py`; `ChatService._run` is the single turn for both endpoints | `tests/unit/agent/test_stream.py` (event order, a /chat-vs-stream parity matrix, disconnect, shutdown) |
| 4. Serve `web/` | `agent/web_page.py`: `GET /` (Jaeger meta filled, escaped) and `GET /fixtures/{name}` | `tests/unit/agent/test_web_page.py` |
| 5. Happy path | `agent-e2e` compose service (profile `e2e`, fake model, port 8081) | `tests/e2e/test_happy_path.py` |
| 6. Restart | — | `tests/e2e/test_restart.py` |
| 7. Chaos | — | `tests/e2e/test_chaos.py`: 100% errors, fail-after-commit, and a seeded mixed case |

**Status events (PLAN item 3 asks for this to be written down): both approaches are used.**

- The installed adapter path does expose MCP progress. The MCP server's retry notifications
  ("Retrying (attempt n of 3)") are forwarded as `status` events of the form
  `"<tool>: <message>"`.
- A single `"Still working on <tool>…"` status is also sent once a tool call has run longer
  than `SLOW_TOOL_NOTICE_S` (3 s). CONTRACTS §4 lists `status` for both "retries" and "slow
  tool", and a slow response with no retry produces no progress notification.
- Both go through LangGraph's custom stream writer, which does nothing on `/chat` runs.

**Fake model:** `ScriptedChatModel` now picks its reply from the conversation (the number
of AI messages after the last human message), not from an in-memory counter. An approve
after `docker compose restart` therefore continues the script.

### Review findings

19 findings; 17 confirmed (2 major), 2 not confirmed.

| Finding | Severity | Outcome |
|---|---|---|
| After a committed write, mcp's `call_tool` sent a follow-up `tools/list` (output-schema validation). If that failed, the saved write was reported as "could not be confirmed". Two reviewers found it. | major | Fixed `b80c7e7`: the provider sends `CallToolRequest` itself with `send_request`, and nothing follows it |
| `initialize` and `tools/list` had no timeout, so a hung MCP server stalled a call, and startup forever | minor | Fixed `b80c7e7`: a session-wide `read_timeout_seconds` |
| The model-visible write schemas kept the `*Input` title and description ("…including the injected arguments") | minor | Fixed `8e8abdc` |
| The no-arguments-in-error-text test could not fail | minor | Fixed `c715204` |
| /chat `tool_calls` gave the wrong args when one request reused a tool call id (older code; the stream was right) | minor | Fixed `8b69be2` |
| A failure in event mapping left `astream` open, so the lock was released while `execute_write` still ran | minor | Fixed `cd65d00` (`aclosing`) |
| The shutdown drain was unbounded (N × turn deadline) | minor | Fixed `722fcc1`; `stop_grace_period: 100s` in compose |
| Rechecker: uvicorn waits for connected streams with no limit before the lifespan drain | minor | Fixed `0cd0a29`: `timeout_graceful_shutdown=5` |
| Rechecker: the drain's bound was untested | minor | Fixed `0cd0a29`; the mutant now fails the test |
| Test gaps: stream spans, slow-notice cancel, trace_id before `begin()`, web_page mutants, an e2e trace wait race, a "no duplicates" check that could not fail | minor | Fixed `e28af20`, `262af83`, `ab1a6ab`, `c59d5c5` |
| No Phase 2 ai-log entry | minor | This entry |
| The agent stays down if MCP is unavailable at agent startup (no `restart:` policy) | minor | **Author decision**, see Unsure 3 |
| More rejected-call rows in the parity matrix | — | Not confirmed: the same code path is already covered by the rejected-batch row |
| E2E chaos for 429 and slow responses | — | Not confirmed: the plan asks for high rates and fail-after-commit; 429 and timeouts are unit-tested in `mcp_server` |

### Verified, and how

- `make lint` → `All checks passed!`, `145 files already formatted`
- `make typecheck` → `Success: no issues found in 77 source files`
- `make test` → `1006 passed, 7 warnings` (the warnings are the existing aiosqlite
  datetime-adapter DeprecationWarnings)
- `ANTHROPIC_API_KEY=placeholder-not-a-real-key make up` → mock, mcp and agent healthy.
  Then `make test-e2e` twice → `18 passed in 28.30s` and `18 passed in 27.62s`.
- Against the running stack:
  - `GET /` → 200 text/html.
  - `GET /fixtures/propose.sse` → 200.
  - `GET /fixtures/../pyproject.toml` → 404.
- The build agent's smoke run on the host (fake model) found one trace id with spans in all
  three services. The parent links ran agent httpx `POST` → MCP `POST /mcp` → `mcp.tool` →
  MCP httpx `GET` → mock `GET /tickets`.
- No token was found in `./traces` or in `docker compose logs`, checked with grep.
- Mutation probes:
  - **Build:** a reused MCP session breaks the trace test. The old counter-based fake model
    breaks the restart e2e test. Without the mcp traces mount, the e2e trace test fails.
  - **Review:** each fixed test gap is backed by a mutant that now fails.
- **Real-model run (after the author added `ANTHROPIC_API_KEY`), through the API:**
  - Setup: `make up`, then `make seed`, model `claude-sonnet-5-5`.
  - Turn 1 (`/chat`, user `bob`), a new issue about a monitor that is not detected after a
    dock firmware update:
    - three `search_tickets` calls, all empty;
    - a proposed `create_ticket` (hardware, medium); status `awaiting_approval`, 0 writes.
  - Turn 2 (`/chat/stream`, text "yes"):
    - events: `tool_start`/`tool_end` for `create_ticket`, then 83 `token` events, then
      `done{status:"ok"}`;
    - the mock holds `INC-0016`, and `/admin/writes` has total 1 under one key.
  - `make trace` for turn 2: 23 spans across agent, mcp and mock. The LLM span shows
    in=3307, out=334, cache_read=2766, cache_write=539.
  - The browser part of this gate step is left to the author (it is also the demo
    recording).

### Deviations from the plan / contracts, and why

- **Provider:**
  - It uses the adapter's `StreamableHttpConnection` / `create_session` and
    `load_mcp_tools` (for list pagination), not the adapter's converted `StructuredTool`s.
    Those turn `isError` into an exception, and the graph builds its own `ToolMessage`s from
    the exact contract text.
  - Calls go through `ClientSession.send_request(CallToolRequest)`, not `call_tool`, so no
    `tools/list` follows a write.
- **One MCP session per tool call:** each call makes about 5 HTTP requests (initialize, initialized, tools/call, a GET stream, DELETE; about 23 ms on
  localhost). This is the cost of correct trace context (see library notes).
- **Client timeout:** `TOOL_DEADLINE_S + TOOL_ATTEMPT_TIMEOUT_S` (20 s), longer than the
  server's 15 s deadline, so the server's own `TIMEOUT` text arrives first. No new Setting.
- **Trace propagation** is checked in a Docker-free integration test, and also in
  `tests/e2e`.
- **E2E runs against a second agent, `agent-e2e`** (profile `e2e`, `MODEL_PROVIDER=fake`,
  127.0.0.1:8081, its own volume, `TRACES_DIR=/app/traces/e2e`), so `agent` stays on the real
  model for the browser.
  - `make test-e2e` starts it.
  - The restart test runs `docker compose restart agent-e2e`, not `agent` as PLAN says.
  - The network-isolation and compose static checks cover both agents.
- **Compose:**
  - The `mcp` service had no `./traces` bind mount, so its spans never reached the host.
    The mount was added, with a static test.
  - `stop_grace_period: 100s` for both agents.
- **Stream text:** model text is streamed as it arrives. That includes text the model writes
  before a tool call, so the tokens can be longer than /chat's `reply`: the intermediate texts,
  then `"\n\n"`, then /chat's exact reply. Text the service adds itself (proposal ask, re-ask,
  stale reply, fallbacks) is sent as tokens. On `error`, the event carries /chat's reply.
- **Unexpected exception:** /chat returns 500, but a stream cannot after its 200, so it
  ends with `error` + `done{status:"error"}`.
- **Status text:** the wording is `"<tool>: Retrying (attempt 2 of 3)"`; the `web/fixtures`
  wording differs slightly.
- **Shutdown:**
  - After shutdown starts, a queued streamed turn ends without running (`INTERRUPTED_REPLY`,
    status error). Its client has already disconnected.
  - The drain is bounded by one turn deadline, and uvicorn's wait for open connections by
    5 s.
- **Write schemas:** the model-visible schemas for `create_ticket` and `add_comment` have no
  top-level title or description; the tool description stays.

### Library notes added in Phase 2 (verified against installed source)

- **mcp 1.30.0:**
  - `ClientSession.call_tool` runs `_validate_tool_result`, which calls `list_tools()` when
    the output schema is not cached (`mcp/client/session.py:386, 421`).
    `BaseSession.send_request` is public (`mcp/shared/session.py:240`).
  - The streamable-HTTP client sends requests from tasks started when the session opens
    (`mcp/client/streamable_http.py:587, 672`). They carry the OTel context of the session's
    opener, hence a session per call.
  - Progress callbacks run in the session's receive loop, and their exceptions are caught
    and logged (`mcp/shared/session.py:405-421`).
- **langchain-mcp-adapters 0.3.2:**
  - `create_session` yields an uninitialised `ClientSession` (`sessions.py:406`) and wires
    only the logging and elicitation callbacks (`sessions.py:434`).
  - `StreamableHttpConnection.session_kwargs` is passed to `ClientSession`
    (`sessions.py:366`).
  - `load_mcp_tools` follows `nextCursor` (`tools.py:320`).
- **opentelemetry-instrumentation-httpx 0.66b1** patches only the real transports
  (`__init__.py:1283`). `ASGITransport` gets no `traceparent`, so propagation can only be
  shown over a socket.
- **langgraph 1.2.15:**
  - With a list `stream_mode`, `astream` yields `(mode, payload)` tuples
    (`pregel/main.py:4428`).
  - An interrupt arrives as an `updates` item `{"__interrupt__": ...}`
    (`pregel/_loop.py:1579`).
  - The custom writer does nothing unless `"custom"` is streamed, and it needs the node's
    contextvars (`pregel/main.py:3355-3395`).
  - `astream` is annotated `AsyncIterator` but is an async generator; `aclosing` needs a
    cast.
- **FastAPI 0.143.2** has native SSE (`fastapi/sse.py`: `EventSourceResponse`,
  `ServerSentEvent`), with `: ping` every 15 s and no-cache / no-buffering headers.
- **langchain-core 1.6.11 / langchain-anthropic 1.7.6:** `BaseMessage.text` keeps only text
  blocks. Anthropic tool arguments stream as `input_json_delta` blocks and never become token
  text.
- **uvicorn 0.54:**
  - With `lifespan="on"`, a failed startup exits with status 3 (`server.py:118`).
  - `shutdown()` waits for open connections before the lifespan shutdown, with no limit
    unless `timeout_graceful_shutdown` is set (`server.py:281-311`).
- **httpx 0.28.1:** `ASGITransport` buffers the whole response and sends `http.disconnect`
  only at the end (`_transports/asgi.py:134-145`). The disconnect tests drive the ASGI app
  directly.

### Unsure — author should review

1. **Real-model run.** It passed through the API (see Verified). The browser click-through
   is left to the author.
   - Plain `make up` fails without a key: the default `agent` exits with
     `MissingSecretError`, so `--wait` fails. The e2e gate was run with a placeholder key in
     the shell.
   - Should `make up` tolerate a missing key, or should the README just state that it needs
     one?
2. **Stream text policy.**
   - Intermediate model text is streamed, so the stream can show more text than /chat's
     `reply`.
   - The alternative is to buffer each reply until it ends: exact parity with /chat, but no
     live text.
   - Should CONTRACTS §4 say which one?
3. **No `restart:` policy in compose.**
   - If MCP is down when the agent starts, the agent exits with 3 and stays down until
     `make up`. A manual `docker compose restart agent` while mcp is restarting is enough.
   - Options: `restart: on-failure` on both agents, or a documented manual recovery.
   - A restart policy is not a tool retry, so the single-retry-layer rule is not affected.
4. **Status of a failed approved write.** It is `ok`, with the honest
   `EMPTY_AFTER_FAILED_WRITE_REPLY`, or with the model's own reply when it writes one.
   CONTRACTS §3.4 does not say. The e2e chaos tests pin `ok`.
5. **Shutdown behaviour.**
   - Shutdown can take up to 5 s plus `TURN_DEADLINE_S` (95 s by default). Compose allows
     100 s.
   - A shorter bound would need a new Setting in `shared/`.
   - Turns queued at shutdown end without running.
6. **Span tree.** The MCP httpx client spans are children of `agent.request`, not of the
   LangGraph `tools` / `execute_write` node span, because OpenInference's LangChain spans are
   not made the current OTel context. The trace id is correct; only the nesting in
   `show_trace` is flatter.
7. **Not in CONTRACTS §4:**
   - A tool call cut off by the deadline has a `tool_start` and no `tool_end`.
   - A stream that hits a server bug ends with `error` + `done{status:error}`, while /chat
     returns 500.
8. **Seeded mixed-chaos e2e test** (`seed=8`). It depends on the mock's chaos draw order.
   If `chaos.py` changes, the test fails on `attempts == 3`; it does not pass silently.

### Suggestions

- An integration test that drives MCP retry progress through the real provider into
  `/chat/stream` with chaos on. Unit tests use the fake provider; the e2e chaos stream test
  covers it on compose.
- Refresh the `web/fixtures/*.sse` status wording to match the real text.
- Cache tool output schemas, or reuse sessions with per-request `traceparent` injection,
  if per-call MCP overhead matters with a real model. This needs a design decision because of
  the trace-context issue.
- Pass `on_progress` only on streamed runs, so /chat does not ask the MCP server for
  progress notifications. Today they are sent and dropped.

## 2026-10-16 — Phase 3 / Track F: DeepEval rewrite, judge model, agent prompt fix

### Delegated to AI (Claude Code, Opus 5.5)

The work the author approved after the first live eval run (`17f4e31`), commits
`8284b3b..556b466`. No live eval was run afterwards and no paid model API was called.

### What changed

| Commit | Change |
|---|---|
| `8284b3b` | `JUDGE_MODEL_PROVIDER` / `JUDGE_MODEL_NAME` in `Settings` (default `anthropic` / `claude-haiku-5-5`), listed in `.env.example`. |
| `268f7f4` | `build_judge_model` in `src/helpdesk/agent/model.py`: the same `init_chat_model` path as the agent; `JUDGE_MODEL_PROVIDER=fake` raises `ModelConfigError`. |
| `e7e547f` | Agent prompt: calling the write tool is the approval request; never ask for approval in plain text. Cause: in the first live run the model wrote the proposal as text and asked "Do you approve?", so no pending action existed. |
| `edcab3f` | PLAN and CONTRACTS §7: judge scope (reply semantics only) and the judge factory. |
| `2f52f80` | `deepeval` 4.3.0 in the dev group; `click` 8.5.0 -> 8.3.3 in the lock (deepeval pins `click<8.4`). `.deepeval/` gitignored. |
| `d16e72d` | `evals/` scored through DeepEval: a `ConversationalGolden` per case, a `ConversationalTestCase` per run, every expectation key an `ExpectationMetric`, `judge:` entries as strict `ConversationalGEval`, `a_measure` instead of `evaluate()`, containment env in `evals/__init__.py`, not-exercised outcome, judge tokens reported apart. |
| `191f6b5` | Judge metrics replace the positive reply-phrasing regexes of cases 3, 7, 12 and 13; the negative guards stay deterministic. |
| `556b466` | PLAN: case-file judge keys and the not-exercised pass rate. |

Docs pass afterwards: `fe48629` (README facts: env rows, judge, DeepEval scoring and
containment, exit status, pytest `addopts`, deepeval version) and `3d817b9` (`make help`
for `eval`). The README eval case table already matched `python -m evals.run --list`
after `52d183f`.

### Verified, and how

- At `3d817b9`:
  - `make lint`: `All checks passed!`, `187 files already formatted`
  - `make typecheck`: `Success: no issues found in 97 source files`
  - `make test`: `1317 passed, 11 warnings`
  - `uv run python -m evals.run --fake --out <scratch>`: `Pass rate: 13/13 (100%)`, judge
    `scripted-judge (provider: fake)`, judge tokens 0
- `--list` output diffed line by line against the README case table: identical.
- README `<!-- HUMAN: write this -->` markers: 23 before, 23 after.

### Deviations from the plan

- `pyproject.toml`: `addopts` disables the pytest plugins deepeval installs (`deepeval`,
  `xdist`, `xdist.looponfail`, `repeat`, `rerunfailures`), and `filterwarnings` ignores
  deepeval's `asyncio.iscoroutinefunction` `DeprecationWarning`.
- The judge is built in the host process (`python -m evals.run`), which reads only the
  judge's fields of `.env` (`evals.judges.JUDGE_FIELDS`, after `5f5bdc4`).
- `JUDGE_MODEL_PROVIDER=fake` passes `Settings` validation but raises `ModelConfigError` in
  the factory; the fake tier passes its own scripted judge.

### Unsure — author should review

- No live run since the prompt fix (`e7e547f`) or the DeepEval rewrite. The committed
  `evals/results/latest.{json,md}` are from `17f4e31` (git `60da65296ad3`, 29/39) and use
  the old report format.
- The Haiku judge's agreement with a human reading of the replies is not measured.
- The Phase 3 Track F/G commits before `17f4e31` (`be13fa3` .. `17f4e31`) have no ai-log
  entry of their own.

## 2026-10-16 — Evals (DeepEval rewrite): review fixes

### Delegated to AI (Claude Code, Opus 5.5)

Fixes for the confirmed review findings on `17f4e31..556b466` (areas evals and platform).
No live eval was run and no paid model API was called.

### What changed

| Commit | Change |
|---|---|
| `784a8a8` | Cases 5, 7, 9 and 10 set `no_invented_ticket_ids`. Cases 5 and 9 require `search_tickets` before the proposal. `known_issue_comment` requires a successful turn-1 `get_ticket`. |
| `52d183f` | README case table regenerated from `python -m evals.run --list`. |
| `03beca8` | Pagination is "not exercised" only when no successful search returned a cursor. Before, only the first search was checked. PLAN wording updated. |
| `d72f3f9` | `LangChainJudge`: JSON-text answers from models without forced tool choice (claude-opus-5-5, claude-sonnet-5-5) go to DeepEval's JSON parser. A `TypeError` is one `JudgeError`, not a second schemaless call. A failed call's tokens are counted. |
| `ff0d780` | Containment: `DEEPEVAL_DISABLE_LEGACY_KEYFILE=1`, a per-user DeepEval folder, and inherited `DEEPEVAL_*` / `CONFIDENT_*` variables removed. |
| `5f5bdc4` | The live judge reads only `JUDGE_FIELDS` from `ROOT/.env`, so the mock tokens stay out of the runner. A settings `ValidationError` prints field and message only. CONTRACTS §7 records this. |
| `fbcf12a` | The `.deepeval/` containment assertions compare before and after the run, not "does not exist". |

### Verified, and how

- Each fix has a test that failed before it, run against the old code (stash or temporary
  revert): fake-tier mutants for the case files; unit tests for the pagination rule, the
  judge (a real `ChatAnthropic` with a canned reply), the keyfile, `judge_settings` and the
  error print.
- `make lint typecheck test`: `1317 passed, 11 warnings`. The stray empty `.deepeval/` was
  still at the repo root during the run.
- `python -m evals.run --fake --out <scratch>`: `Pass rate: 13/13 (100%)`.

### Unsure — author should review

- The JSON-text fallback for Opus/Sonnet judges was tested offline only. How often those
  models answer in text instead of calling the tool is not measured.
- The empty, gitignored `.deepeval/` at the repo root (16:13 today) was not made by tracked
  code. It was left in place; it can be deleted.
