# CLAUDE.md

This is an IT helpdesk agent built for a take-home assignment. Read `PLAN.md` for the
plan and `docs/CONTRACTS.md` for the interfaces. The contracts are the source
of truth: if your code and a contract disagree, the code is wrong.

## Hard rules

- **Do not write prose for README.md or AGENTCORE.md.** The reviewers want the
  author's own thinking. You may add commands, tables, generated results,
  diagrams and bullet-point facts. Leave every `<!-- HUMAN: write this -->` marker
  in place.
- **Stay inside your track's directory.** Ask before editing `src/helpdesk/shared/`,
  `docs/CONTRACTS.md`, or another track's code.
- **Don't add scope.** Nothing outside PLAN.md: no extra agents, no extra tools,
  no frameworks in the front end. If you think something is missing, write it in
  `docs/ai-log.md` under "Suggestions" and carry on.
- **Verify library APIs against the installed version**, not from memory
  (LangGraph, langchain-mcp-adapters, mcp). If a planned API doesn't exist, stop
  and report it.
- **Never commit secrets.** Tokens come from env. `.env` is gitignored. Never log
  tokens.
- **There is one retry layer for tools**, the MCP server's retry module. Don't add a
  LangGraph `RetryPolicy` to tool nodes or retry loops in the agent. Model calls
  use only the Anthropic SDK's built-in retries, configured via `Settings`.
- **The LLM is the Claude API** (`langchain-anthropic`, default `claude-haiku-5-5`).
  Keep the provider swappable through `init_chat_model`. Never hard-code a
  model name outside `Settings`.
- **No side effects before `interrupt()`**: the node re-runs on resume.
  Idempotency keys are created only in `propose`.
- **`user_id` never comes from model output.** It comes from the request and
  `RunnableConfig` only.

## Code standards

- Python 3.14 and the latest stable libraries (see the version policy in PLAN.md §1). Use modern
  syntax: `type` aliases, PEP 695 generics, `X | None`. Fully type-hinted. `mypy --strict` on `src/`. Pydantic models at all
  boundaries.
- Small modules with clear names. No `utils.py` dumping ground.
- Async I/O everywhere (httpx.AsyncClient, aiosqlite or AsyncSqliteSaver).
- Errors: typed exceptions inside services. At the MCP boundary, convert to
  `ERROR <CODE>: <what>. Next step: <what to do>.`
- Config is read only through `helpdesk.shared.settings.Settings`.
- Comments explain *why*, not *what*.

## Testing

- Every behaviour in PLAN.md §8 has a test. Write the test with the code, not
  afterwards.
- Unit tests use fake models (`GenericFakeChatModel` / `FakeMessagesListChatModel`)
  and respx. They never call a real model or the network.
- Run `make lint typecheck test` before you call a task done. Paste the summary
  line of the output in your final message.

## Workflow

- Commit per logical step with conventional commit messages (`feat(mock): ...`).
- After each phase gate, append to `docs/ai-log.md`:
  - date and phase/track
  - what was delegated to AI
  - what was verified, and how
  - deviations from the plan and why
  - anything you're unsure about that the author should review
  Keep it factual. The author writes the reflection.
- When stuck or when a contract seems wrong: stop and ask. Don't work around it.

## Commands

```
make setup        # uv sync, pre-commit
make lint         # ruff check + format --check
make typecheck    # mypy
make test         # unit + integration (no docker)
make up / down    # docker compose
make test-e2e     # requires `make up`
make eval         # live evals, 3 repeats → evals/results/latest.md
make eval-chaos   # chaos-only cases
```