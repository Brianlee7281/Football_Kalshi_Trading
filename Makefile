.PHONY: up down migrate test lint

up:
	docker compose -f docker/docker-compose.yml up -d postgres redis

down:
	docker compose -f docker/docker-compose.yml down

migrate:
	PGPASSWORD=postgres psql -h localhost -U postgres -d soccer_trading -f sql/schema.sql

test:
	pytest tests/ -v

lint:
	mypy src/ --strict && ruff check src/
