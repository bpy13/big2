# Run inside the `poker` conda env: `conda activate poker`.
.PHONY: dev test test-rust test-python test-web test-e2e lint fmt bench docs web serve web-dev

dev:            ## Build the Rust extension, install the Python package (editable) and UI deps
	pip install -e ".[dev]"
	npm install --prefix web

test: test-rust test-python test-web

test-rust:
	cargo test --workspace

test-python:
	pytest

test-web:       ## UI unit tests
	cd web && npx vitest run

test-e2e: web   ## Plays a full game in a real browser (needs `npx playwright install chromium` once)
	cd web && npx playwright test

lint:
	cargo fmt --all --check
	cargo clippy --workspace --all-targets -- -D warnings
	ruff check python experiments
	ruff format --check python experiments
	mypy python/poker
	cd web && npx tsc -b && npx oxlint

fmt:
	cargo fmt --all
	ruff format python experiments
	ruff check --fix python experiments

bench:
	cargo bench -p poker-big2 --bench engine

docs:           ## Build Rust API docs into target/doc
	cargo doc --workspace --no-deps

web:            ## Build the UI into web/dist
	npm run build --prefix web

serve: web      ## UI + API on http://127.0.0.1:8000
	poker serve

web-dev:        ## UI with hot reload on :5173 (run `uvicorn poker.server.app:app --reload` alongside)
	npm run dev --prefix web
