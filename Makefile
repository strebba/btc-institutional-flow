.PHONY: test install run-api run-dashboard update-gex update-flows update-edgar update-all

install:
	pip install -e ".[dev]"

test:
	python3 -m pytest tests/ -v --tb=short

update-gex:
	python3 scripts/cron_gex.py

update-flows:
	python3 scripts/run_coinglass.py
	python3 scripts/run_flows.py

update-edgar:
	python3 scripts/run_edgar.py

update-all: update-gex update-flows update-edgar

run-api:
	python3 run_api.py

run-dashboard:
	streamlit run src/dashboard/app.py
