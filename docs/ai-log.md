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
