.DEFAULT_GOAL := help
SHELL := /bin/bash

REPEATS ?= 3
# Optional: `make eval MODEL=claude-haiku-5-5` runs agent-eval with that model.
MODEL ?=
EVAL_MODEL_FLAG := $(if $(MODEL),--model $(MODEL))
# Moderate chaos for the demo recording; override with `make chaos-on CHAOS_BODY='{...}'`.
CHAOS_BODY ?= {"enabled":true,"rate_limit_rate":0.2,"error_rate":0.2,"slow_rate":0.1,"slow_ms":3000,"seed":42}

ifeq ($(OBS),1)
COMPOSE_UP_ENV := OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318 JAEGER_UI_URL=http://localhost:16686
COMPOSE_PROFILES_FLAG := --profile observability
endif

# Calls the mock admin API from inside the mock container, which already holds both
# tokens. The mock publishes no port, and no token ever reaches the host shell.
MOCK_ADMIN = docker compose exec -T mock python -c "import os, sys, urllib.request as u; \
	r = u.Request('http://127.0.0.1:8000' + sys.argv[1], data=sys.argv[2].encode(), method='POST', \
	headers={'Authorization': 'Bearer ' + os.environ['MOCK_API_TOKEN'], \
	'X-Admin-Token': os.environ['MOCK_ADMIN_TOKEN'], 'Content-Type': 'application/json'}); \
	print(u.urlopen(r, timeout=10).read().decode())"

.PHONY: help setup lint typecheck test test-e2e up down eval eval-chaos seed chaos-on chaos-off trace

help: ## List targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Install deps, pre-commit hooks, and create .env with fresh tokens
	uv sync
	uv run pre-commit install
	uv run python scripts/init_env.py

lint: ## ruff check + format --check
	uv run ruff check .
	uv run ruff format --check .

typecheck: ## mypy --strict on src/ and evals/
	uv run mypy

test: ## Unit + integration tests (no docker stack needed)
	uv run pytest tests/unit tests/integration

test-e2e: ## End-to-end tests; requires `make up`. Starts agent-e2e (fake model) on :8081 and agent-eval on :8082
	$(COMPOSE_UP_ENV) docker compose $(COMPOSE_PROFILES_FLAG) --profile e2e --profile eval up -d --wait agent-e2e agent-eval
	uv run pytest -m e2e tests/e2e

up: ## Build and start the stack (OBS=1 adds Jaeger on http://localhost:16686)
	$(COMPOSE_UP_ENV) docker compose $(COMPOSE_PROFILES_FLAG) up -d --build --wait

down: ## Stop the stack (including Jaeger, agent-e2e and agent-eval)
	docker compose --profile observability --profile e2e --profile eval down

eval: ## Live evals (agent-eval on :8082, real model), REPEATS=3 -> evals/results/latest.md
	uv run python -m evals.run --repeats $(REPEATS) $(EVAL_MODEL_FLAG)

eval-chaos: ## Chaos-only live eval cases
	uv run python -m evals.run --repeats $(REPEATS) --chaos $(EVAL_MODEL_FLAG)

seed: ## Reset the mock database to the seed tickets
	docker compose exec mock python -m helpdesk.mock.seed --reset

chaos-on: ## Turn on mock chaos (CHAOS_BODY overrides the rates)
	$(MOCK_ADMIN) /admin/chaos '$(CHAOS_BODY)'

chaos-off: ## Turn off mock chaos
	$(MOCK_ADMIN) /admin/chaos '{"enabled":false,"rate_limit_rate":0,"error_rate":0,"slow_rate":0,"slow_ms":0}'

trace: ## Print a trace's span tree: make trace ID=<trace_id>
	@test -n "$(ID)" || { echo "usage: make trace ID=<trace_id>"; exit 2; }
	uv run python scripts/show_trace.py $(ID)
