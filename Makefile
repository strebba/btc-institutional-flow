.PHONY: test test-unit test-integration lint typecheck install run-api run-dashboard update-gex update-flows update-edgar update-all

install:
	pip install -e ".[dev]"

test:
	python3 -m pytest tests/ -v --tb=short

test-unit:
	python3 -m pytest tests/ --ignore=tests/integration/ -v --tb=short -q

test-integration:
	python3 -m pytest tests/integration/ -v --tb=short

lint:
	ruff check src/ tests/

typecheck:
	mypy src/ --ignore-missing-imports || true

update-gex:
	python3 scripts/cron_gex.py

update-flows:
	python3 scripts/run_coinglass.py
	python3 scripts/run_flows.py

update-edgar:
	python3 scripts/run_edgar.py

update-all: update-gex update-flows update-edgar

run-api:
	DB_PATH=data/runtime.db python3 run_api.py

run-dashboard:
	streamlit run src/dashboard/app.py
