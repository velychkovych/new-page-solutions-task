# Running this on Amazon Bedrock AgentCore

<!-- HUMAN: write this -->

## Which AgentCore components and why

<!-- HUMAN: write this -->

Components to cover (PLAN.md): Runtime, Gateway, Identity, Memory, Observability.

## What stays and what changes from the local build

<!-- HUMAN: write this -->

Local build inventory (facts). The right-hand column lists the candidates named in
PLAN.md; the author decides.

| Concern | Local build | Candidate named in PLAN.md |
|---|---|---|
| Agent process | `agent` container, FastAPI + LangGraph, `python -m helpdesk.agent` | <!-- HUMAN --> |
| Tool server | `mcp` container, FastMCP streamable HTTP at `/mcp` | <!-- HUMAN --> |
| Mock credential | `MOCK_API_TOKEN` env var in `mcp` (sends it) and `mock` (checks it); never in `agent` | <!-- HUMAN --> |
| Conversation state + pending approvals | `AsyncSqliteSaver`, `data/checkpoints.sqlite` | AgentCore Memory or DynamoDB |
| User facts | LangGraph Store (SQLite), namespace `("users", user_id, "facts")`, 90-day TTL | <!-- HUMAN --> |
| Traces | OpenTelemetry → `traces/*.jsonl`, optional OTLP → Jaeger | CloudWatch via OTel |
| Model | Claude API (`MODEL_PROVIDER=anthropic`, `claude-sonnet-5-5`) | Claude on Bedrock (`MODEL_PROVIDER=bedrock`) |
| Network isolation | Two compose networks; agent has no route to the mock | <!-- HUMAN --> |

## Where it gets difficult

<!-- HUMAN: write this -->

## Diagram

<!-- HUMAN: write this -->

<!-- generated (Phase 3, Track G) from PLAN.md §0, §1, §4 Phase 0 step 7 and the local build (docker-compose.yml, Settings) -->

```mermaid
flowchart LR
  subgraph local_build["Local build: docker compose"]
    l_agent["agent container<br/>FastAPI + LangGraph"]
    l_mcp["mcp container<br/>FastMCP, streamable HTTP, /mcp"]
    l_token["MOCK_API_TOKEN<br/>env var in mcp and mock, not in agent"]
    l_ckpt[("AsyncSqliteSaver<br/>data/checkpoints.sqlite<br/>conversation + pending approvals")]
    l_store[("LangGraph Store, SQLite<br/>user facts, 90-day TTL")]
    l_traces["OpenTelemetry spans<br/>traces/*.jsonl, optional OTLP to Jaeger"]
    l_model["Claude API<br/>MODEL_PROVIDER=anthropic"]
  end

  subgraph aws["AWS"]
    ac_runtime["AgentCore Runtime"]
    ac_gateway["AgentCore Gateway"]
    ac_identity["AgentCore Identity"]
    ac_memory["AgentCore Memory"]
    ac_obs["AgentCore Observability"]
    ddb[("DynamoDB")]
    cloudwatch["CloudWatch via OTel"]
    bedrock["Claude on Bedrock<br/>MODEL_PROVIDER=bedrock"]
  end

  l_ckpt -->|"PLAN: AgentCore Memory or DynamoDB"| ac_memory
  l_ckpt -->|"PLAN: AgentCore Memory or DynamoDB"| ddb
  l_traces -->|"PLAN"| cloudwatch
  l_model -->|"PLAN"| bedrock

  l_agent -.->|"candidate — author to confirm"| ac_runtime
  l_mcp -.->|"candidate — author to confirm"| ac_gateway
  l_token -.->|"candidate — author to confirm"| ac_identity
  l_store -.->|"candidate — author to confirm"| ac_memory
  cloudwatch -.->|"candidate — author to confirm"| ac_obs
```

- Solid edges: mappings PLAN.md names explicitly (§4 Phase 0 step 7):
  - SQLite checkpointer (with pending approvals) → AgentCore Memory or DynamoDB
  - JSONL traces → CloudWatch via OTel
  - Claude API → Claude on Bedrock (`MODEL_PROVIDER=bedrock`, `ChatAnthropicBedrock`;
    CONTRACTS §7)
- Dashed edges: candidates PLAN.md does not name; the author confirms or removes each one.
- Component set (PLAN.md): Runtime, Gateway, Identity, Memory, Observability.
- Not drawn: network isolation (two compose networks); PLAN.md names no target for it.

## What I'd verify in the AWS docs before relying on it

<!-- HUMAN: write this -->
