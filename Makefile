POETRY ?= poetry
PYTHON ?= python
LOAD_TEST_HOST ?= 127.0.0.1
LOAD_TEST_PORT ?= 8000
LOAD_TEST_REQUESTS ?= 100
LOAD_TEST_MAX_RPM ?= 180
LOAD_TEST_RPC_TIMEOUT_MS ?= 2000
LOAD_TEST_REPORT ?= artifacts/load_test/load_test_report.md

.PHONY: install run test precommit migrate load-test compose-up compose-build compose-down compose-logs

install:
	$(POETRY) install

run:
	$(POETRY) run $(PYTHON) -m hackaton.service.main

test:
	$(POETRY) run pytest

precommit:
	$(POETRY) run pre-commit run --all-files

migrate:
	$(POETRY) run $(PYTHON) -m hackaton.service.db migrate

load-test:
	$(POETRY) run $(PYTHON) scripts/load_test.py \
		--host $(LOAD_TEST_HOST) \
		--port $(LOAD_TEST_PORT) \
		--requests $(LOAD_TEST_REQUESTS) \
		--max-rpm $(LOAD_TEST_MAX_RPM) \
		--rpc-timeout-ms $(LOAD_TEST_RPC_TIMEOUT_MS) \
		--report-path $(LOAD_TEST_REPORT)

compose-build:
	docker compose build

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f app
