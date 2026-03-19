#!/usr/bin/env bash
# MindCastle — Master Release Gate
# =================================
# Run from the project root. Checks all 5 release gates and prints a go/no-go verdict.
#
# Usage:  bash scripts/release_gate.sh
# Exit:   0 = GO,  1 = NO-GO

cd "$(dirname "$0")/.." || exit 1   # always run from project root

GREEN="\033[92m"
RED="\033[91m"
YELLOW="\033[93m"
DIM="\033[90m"
RESET="\033[0m"
PASS="${GREEN}✓${RESET}"
FAIL="${RED}✗${RESET}"
WARN="${YELLOW}⚠${RESET}"

FAILURES=0

gate() {
  echo ""
  echo -e "${DIM}$(printf '═%.0s' {1..58})${RESET}"
  echo -e "  Gate $1: $2"
  echo -e "${DIM}$(printf '─%.0s' {1..58})${RESET}"
}

run() {
  local label="$1"
  shift
  local tmpfile
  tmpfile=$(mktemp)
  if "$@" > "$tmpfile" 2>&1; then
    echo -e "  ${PASS}  ${label}"
  else
    echo -e "  ${FAIL}  ${label}"
    # Show first 15 lines of output to explain the failure
    head -15 "$tmpfile" | sed 's/^/       /'
    FAILURES=$((FAILURES + 1))
  fi
  rm -f "$tmpfile"
}

check_file() {
  local label="$1"
  local path="$2"
  if [ -f "$path" ]; then
    echo -e "  ${PASS}  ${label}"
  else
    echo -e "  ${FAIL}  ${label}  [not found: ${path}]"
    FAILURES=$((FAILURES + 1))
  fi
}

check_grep() {
  local label="$1"
  local pattern="$2"
  local file="$3"
  if grep -q "$pattern" "$file" 2>/dev/null; then
    echo -e "  ${PASS}  ${label}"
  else
    echo -e "  ${FAIL}  ${label}  [pattern '${pattern}' not found in ${file}]"
    FAILURES=$((FAILURES + 1))
  fi
}

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}  MindCastle Release Gate${RESET}"
echo -e "${DIM}  $(date '+%Y-%m-%d %H:%M:%S')${RESET}"

# ── Gate 1: Product Experience ────────────────────────────────────────────────
gate 1 "Product Experience"
run "UI tests (theme completeness, CSS vars, viewport invariants, perf)" \
    bash -c "cd tinynet-ui && npm run test:run"

# ── Gate 2: Technical Reliability ─────────────────────────────────────────────
gate 2 "Technical Reliability"
run "UI lint — 0 warnings" \
    bash -c "cd tinynet-ui && npm run lint"
run "UI typecheck — 0 errors" \
    bash -c "cd tinynet-ui && npm run typecheck"
check_grep "styles.css uses 100dvh full-screen layout" \
    "100dvh" "tinynet-ui/src/styles.css"
check_grep "styles.css injects safe-area-inset-top (notch support)" \
    "env(safe-area-inset-top)" "tinynet-ui/src/styles.css"
check_grep "styles.css prevents body scroll (overflow:hidden)" \
    "overflow: hidden" "tinynet-ui/src/styles.css"

# ── Gate 3: Backend + Model Production ────────────────────────────────────────
gate 3 "Backend + Model Production"
run "Backend release gate (config / safety policy / governance / degraded mode / ops)" \
    bash -c "cd tinynet-api && python3 scripts/release_gate.py"

# ── Gate 4: Quality (all test suites) ─────────────────────────────────────────
gate 4 "Quality (Test Suites)"
run "API: unit tests" \
    bash -c "cd tinynet-api && python3 -m pytest -m unit -q --tb=short"
run "API: integration tests" \
    bash -c "cd tinynet-api && python3 -m pytest -m integration -q --tb=short"
run "API: load / perf tests (p95 SLO gate)" \
    bash -c "cd tinynet-api && python3 -m pytest -m load -q --tb=short"

# ── Gate 5: Release Ops ────────────────────────────────────────────────────────
gate 5 "Release Ops"
check_file "RELEASE.md at project root"                         "RELEASE.md"
check_file ".github/workflows/ci.yml"                           ".github/workflows/ci.yml"
check_file ".github/pull_request_template.md"                   ".github/pull_request_template.md"
check_file "Root Makefile with test-all"                        "Makefile"
check_file "tinynet-api/.env config"                            "tinynet-api/.env"
check_grep "RELEASE.md has Rollback section"       "## Rollback"            "RELEASE.md"
check_grep "RELEASE.md has Known Limitations"      "## Known Limitations"   "RELEASE.md"
check_grep "Makefile has release-check target"     "release-check"          "Makefile"

# ── Verdict ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${DIM}$(printf '═%.0s' {1..58})${RESET}"
if [ "$FAILURES" -eq 0 ]; then
  echo -e "${GREEN}  ✓ GO — All 5 release gates passed. Safe to deploy.${RESET}"
else
  echo -e "${RED}  ✗ NO-GO — ${FAILURES} gate(s) failed. Fix before releasing.${RESET}"
fi
echo -e "${DIM}$(printf '═%.0s' {1..58})${RESET}"
echo ""

exit "$FAILURES"
