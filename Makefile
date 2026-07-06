.PHONY: install run test lint fmt typecheck up down

install:        ## install app + dev deps
	pip install -e ".[llm,dev]"

run:            ## run the API locally (offline defaults)
	uvicorn app.main:app --reload --port 8000

test:           ## run the test suite
	pytest -q

coverage:       ## run tests with a coverage report
	pytest --cov=app --cov-report=term-missing

lint:           ## lint with ruff
	ruff check app tests

fmt:            ## auto-fix lint + format
	ruff check --fix app tests
	ruff format app tests

typecheck:      ## static type check
	mypy app

up:             ## run full stack (api + pgvector) in Docker
	docker compose up --build

down:           ## stop the stack
	docker compose down
