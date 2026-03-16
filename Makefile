# MindCastle — root Makefile
# Runs every quality gate; mirror of .github/workflows/ci.yml.
#
# Usage:
#   make test-all          full project verification (blocks merge if any fail)
#   make ui-test           vitest run (all frontend tests)
#   make api-test-unit     pytest -m unit
#   make api-test-integration  pytest -m integration
#   make api-test-load     pytest -m load
#
.PHONY: ui-lint ui-typecheck ui-test \
        api-test-unit api-test-integration api-test-load \
        test-all release-check

# ── Frontend ──────────────────────────────────────────────────────────────────

ui-lint:
	cd tinynet-ui && npm run lint

ui-typecheck:
	cd tinynet-ui && npm run typecheck

ui-test:
	cd tinynet-ui && npm run test:run

# ── Backend ───────────────────────────────────────────────────────────────────

api-test-unit:
	cd tinynet-api && python3 -m pytest -m unit

api-test-integration:
	cd tinynet-api && python3 -m pytest -m integration

api-test-load:
	cd tinynet-api && python3 -m pytest -m load

# ── Gate ──────────────────────────────────────────────────────────────────────

test-all: ui-lint ui-typecheck ui-test api-test-unit api-test-integration api-test-load
	@echo ""
	@echo "✅  All gates passed."

# ── Release check ─────────────────────────────────────────────────────────────

release-check:
	bash scripts/release_gate.sh
