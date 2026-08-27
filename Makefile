COMPOSE := docker compose -f infra/docker/docker-compose.yml
PYTHON3 ?= python3

.PHONY: lint fmt typecheck test test-cov up down compose-up-full compose-down compose-up-dsb compose-up-kafka compose-down-dsb demo helm-template gen-proto benchmark-config-check benchmark-run benchmark-config-check-dsb dsb-submodule-init policy-static policy-adaptive

gen-proto:
	chmod +x scripts/gen_proto.sh
	./scripts/gen_proto.sh

lint:
	ruff check arcs-rl services/policy-service services/telemetry-bridge benchmarks tests/integration scripts/gen_dsb_network_override.py
	ruff format --check arcs-rl services/policy-service services/telemetry-bridge benchmarks tests/integration scripts/gen_dsb_network_override.py

fmt:
	ruff format arcs-rl services/policy-service services/telemetry-bridge benchmarks tests/integration scripts/gen_dsb_network_override.py

typecheck:
	mypy arcs-rl/src/arcs_rl

test:
	pytest arcs-rl/tests tests/integration --cov=arcs_rl --cov-fail-under=85 -q

test-cov:
	pytest arcs-rl/tests tests/integration --cov=arcs_rl --cov-report=term-missing --cov-fail-under=85

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

compose-up-full:
	$(COMPOSE) up -d --build

compose-down:
	$(COMPOSE) down

# Recreate policy-service with benchmark YAML (requires configs/ mounted at /config in compose).
policy-static:
	ARCS_CONFIG=/config/benchmark-static.yaml $(COMPOSE) up -d --force-recreate policy-service
	@echo "ARCS_CONFIG=/config/benchmark-static.yaml (inference off)"

policy-adaptive:
	ARCS_CONFIG=/config/benchmark-adaptive.yaml $(COMPOSE) up -d --force-recreate policy-service
	@echo "ARCS_CONFIG=/config/benchmark-adaptive.yaml (needs data/models/policy.ts)"

dsb-submodule-init:
	git submodule update --init --recursive

# DeathStarBench Social Network on the arcs_arcs network + Envoy → nginx-thrift
compose-up-dsb: dsb-submodule-init
	docker compose -f infra/docker/docker-compose.yml \
		-f third_party/deathstarbench/socialNetwork/docker-compose.yml \
		-f infra/docker/docker-compose.dsb.override.yml up -d --build

compose-down-dsb:
	docker compose -f infra/docker/docker-compose.yml \
		-f third_party/deathstarbench/socialNetwork/docker-compose.yml \
		-f infra/docker/docker-compose.dsb.override.yml down

# Optional single-node Kafka (plaintext) + telemetry-bridge consumer wiring.
compose-up-kafka:
	docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.kafka.yml up -d --build

# One command to stand up the practice stack and print where to click in the browser.
demo: compose-up-full
	@echo ""
	@echo "ARCS demo stack is up. Try:"
	@echo "  Echo health:     curl -s http://127.0.0.1:18001/health/live"
	@echo "  Through Envoy:     curl -s 'http://127.0.0.1:10000/echo?message=hi' -H 'x-arcs-route: /echo' -H 'x-arcs-error-rate: 0.01'"
	@echo "  Policy health:   curl -s http://127.0.0.1:18080/health/live"
	@echo "  Prometheus:      http://127.0.0.1:9090"
	@echo "  Grafana:         http://127.0.0.1:3000  (admin / admin)"
	@echo ""

# Validate chart YAML when Helm is installed (no-op message otherwise).
helm-template:
	@if command -v helm >/dev/null 2>&1; then \
		helm template arcs-test infra/helm/arcs >/dev/null && echo "helm template OK"; \
	else \
		echo "helm not installed; skipping template render"; \
	fi

# Validate benchmark YAML shape (no k6 / Docker required).
benchmark-config-check:
	PYTHONPATH=. $(PYTHON3) -c "from pathlib import Path; from benchmarks.harness.bench_config import load_benchmark_config; load_benchmark_config(Path('benchmarks/config/default.yaml')); print('benchmark config OK')"

benchmark-config-check-dsb:
	PYTHONPATH=. $(PYTHON3) -c "from pathlib import Path; from benchmarks.harness.bench_config import load_benchmark_config; load_benchmark_config(Path('benchmarks/config/dsb.yaml')); print('dsb benchmark config OK')"

# Run load test + Prometheus snapshot (needs k6, running stack, reachable Prometheus).
benchmark-run:
	PYTHONPATH=. $(PYTHON3) -m benchmarks.run_experiment -c benchmarks/config/default.yaml
