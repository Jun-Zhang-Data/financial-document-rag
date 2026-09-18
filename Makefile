.PHONY: install test lint typecheck format audit run api evaluate compose-up compose-down compose-observe

install:
	uv sync --group dev

test:
	python -m pytest

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

format:
	uv run ruff format src tests

audit:
	uv run pip-audit -r requirements.txt

run:
	PYTHONPATH=src python src/app.py

api:
	PYTHONPATH=src uvicorn api:app --app-dir src --reload --port 8080

evaluate:
	PYTHONPATH=src python src/evaluate.py

compose-up:
	docker compose up --build

compose-observe:
	docker compose --profile observability up --build

compose-down:
	docker compose down
