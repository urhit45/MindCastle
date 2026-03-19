#!/usr/bin/env python3
"""
MindCastle Backend Release Gate
================================
Validates all 5 release criteria for the API + model layer.
Run from the tinynet-api directory before any deployment.

Exit codes: 0 = all green (GO), 1 = one or more failures (NO-GO)
"""

import json
import sys
from pathlib import Path
from typing import List

# Resolve project root (tinynet-api/)
PROJECT_ROOT = Path(__file__).parent.parent
REPO_ROOT    = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── ANSI helpers ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
DIM    = "\033[90m"
RESET  = "\033[0m"
PASS   = f"{GREEN}✓{RESET}"
FAIL   = f"{RED}✗{RESET}"
WARN   = f"{YELLOW}⚠{RESET}"
SKIP   = f"{DIM}–{RESET}"

failures: List[str] = []
warnings: List[str] = []


def ok(label: str) -> None:
    print(f"  {PASS}  {label}")


def fail(label: str, detail: str = "") -> None:
    msg = f"{label}" + (f": {detail}" if detail else "")
    failures.append(msg)
    print(f"  {FAIL}  {label}" + (f"  [{detail}]" if detail else ""))


def warn(label: str, detail: str = "") -> None:
    msg = f"{label}" + (f": {detail}" if detail else "")
    warnings.append(msg)
    print(f"  {WARN}  {label}" + (f"  [{detail}]" if detail else ""))


def gate(n: int, title: str) -> None:
    print(f"\n{'─' * 56}")
    print(f"  Gate {n}: {title}")
    print(f"{'─' * 56}")


# ── Gate 1: Configuration ──────────────────────────────────────────────────────
gate(1, "Configuration")

LABELS_PATH = REPO_ROOT / "backend" / "config" / "labels.yaml"
try:
    import yaml  # type: ignore[import]
    with open(LABELS_PATH) as f:
        labels_cfg = yaml.safe_load(f)
    ok("labels.yaml exists and parses")

    for field in ("categories", "states", "next_step_templates"):
        val = labels_cfg.get(field)
        if val:
            ok(f"labels.{field} — {len(val)} entries")
        else:
            fail(f"labels.{field} missing or empty")
except FileNotFoundError:
    fail("labels.yaml exists", f"not found at {LABELS_PATH}")
except Exception as e:
    fail("labels.yaml parses cleanly", str(e))

# ── Gate 2: Safety Policy ──────────────────────────────────────────────────────
gate(2, "Safety Policy")
try:
    from app.ml.policy import PolicyConfig, SafetyPolicy, DecisionMode

    cfg = PolicyConfig()

    if 0.20 <= cfg.state_abstain_threshold <= 0.50:
        ok(f"state_abstain_threshold in [0.20, 0.50] = {cfg.state_abstain_threshold}")
    else:
        fail("state_abstain_threshold out of safe range", str(cfg.state_abstain_threshold))

    if cfg.state_defer_threshold > cfg.state_abstain_threshold:
        ok(f"defer > abstain  ({cfg.state_defer_threshold} > {cfg.state_abstain_threshold})")
    else:
        fail("state_defer_threshold must be > state_abstain_threshold")

    policy = SafetyPolicy()

    d = policy.evaluate(0.99, [0.99], uncertain=True)
    if d.mode == DecisionMode.ABSTAIN:
        ok("uncertain=True → ABSTAIN (safe fallback active)")
    else:
        fail("uncertain=True must → ABSTAIN", f"got {d.mode}")

    d = policy.evaluate(0.90, [0.80], uncertain=False)
    if d.mode == DecisionMode.NORMAL:
        ok("high confidence → NORMAL (no suppression)")
    else:
        fail("high confidence must → NORMAL", f"got {d.mode}")

    d = policy.evaluate(0.20, [0.10], uncertain=False)
    if d.mode == DecisionMode.ABSTAIN:
        ok("both scores below threshold → ABSTAIN")
    else:
        fail("very low scores must → ABSTAIN", f"got {d.mode}")

except ImportError as e:
    fail("app.ml.policy importable", str(e))

# ── Gate 3: Governance Artifacts ──────────────────────────────────────────────
gate(3, "Governance Artifacts")

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
RUNS_DIR   = PROJECT_ROOT / "runs"

if not MODEL_PATH.exists():
    print(f"  {SKIP}  Model not trained yet (models/best.pt absent) — artifacts skipped")
else:
    ok("models/best.pt exists")
    run_dirs = (
        sorted(RUNS_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if RUNS_DIR.exists() else []
    )
    if not run_dirs:
        fail("Training runs exist in runs/", "directory empty or missing")
    else:
        run = run_dirs[0]
        print(f"  {DIM}Checking most recent run: {run.name}{RESET}")

        # run_manifest.json
        manifest_path = run / "run_manifest.json"
        if manifest_path.exists():
            try:
                m = json.loads(manifest_path.read_text())
                for field in ("run_id", "seed", "data_hash", "labels_hash", "thresholds", "metrics"):
                    if field in m:
                        ok(f"run_manifest.{field} present")
                    else:
                        fail(f"run_manifest.{field} missing")
            except Exception as e:
                fail("run_manifest.json valid JSON", str(e))
        else:
            fail("run_manifest.json exists", f"not found in {run.name}/")

        # fairness_report.json
        fairness_path = run / "fairness_report.json"
        if fairness_path.exists():
            try:
                fr = json.loads(fairness_path.read_text())
                if fr.get("overall_pass") is True:
                    ok("fairness_report.overall_pass = true (no parity violations)")
                else:
                    warn("fairness_report.overall_pass = false — bias detected in training data groups")
            except Exception as e:
                fail("fairness_report.json valid JSON", str(e))
        else:
            fail("fairness_report.json exists", f"not found in {run.name}/")

        # model_card.json
        card_path = run / "model_card.json"
        if card_path.exists():
            try:
                card = json.loads(card_path.read_text())
                for field in ("model_name", "version", "intended_use", "limitations"):
                    if field in card:
                        ok(f"model_card.{field} present")
                    else:
                        fail(f"model_card.{field} missing")
            except Exception as e:
                fail("model_card.json valid JSON", str(e))
        else:
            fail("model_card.json exists", f"not found in {run.name}/")

# ── Gate 4: Degraded Mode Safety ──────────────────────────────────────────────
gate(4, "Degraded Mode Safety")
try:
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from app.main import app
    from app.middleware import clear_rate_limits

    clear_rate_limits()
    client = TestClient(app)

    # Health
    resp = client.get("/healthz")
    if resp.status_code == 200:
        ok("GET /healthz → 200")
    else:
        fail("GET /healthz", f"got {resp.status_code}")

    # Uniform error envelope
    resp = client.post("/classify/", json={"text": ""})
    body = resp.json()
    if "error" in body and "code" in body["error"] and "requestId" in body["error"]:
        ok("Validation error → uniform error envelope {error:{code,requestId}}")
    else:
        fail("Validation error envelope shape", f"got: {json.dumps(body)[:120]}")

    # RuntimeError → sanitized 503 (not 500)
    with patch("app.routers.classify.model_service") as mock_svc:
        mock_svc.classify.side_effect = RuntimeError("internal weights raw detail")
        resp = client.post("/classify/", json={"text": "test"})
    if resp.status_code == 503:
        ok("RuntimeError classify → 503 (not 500)")
    else:
        fail("RuntimeError classify must → 503", f"got {resp.status_code}")

    leaked = any(word in resp.text for word in ("weights", "raw detail", "Traceback"))
    if not leaked:
        ok("503 body contains no raw error internals")
    else:
        fail("503 body leaks internal error detail", resp.text[:120])

    # TimeoutError → 503
    import asyncio
    with patch("app.routers.classify.model_service") as mock_svc:
        mock_svc.classify.side_effect = asyncio.TimeoutError()
        resp = client.post("/classify/", json={"text": "slow"})
    if resp.status_code == 503:
        ok("TimeoutError classify → 503")
    else:
        fail("TimeoutError classify must → 503", f"got {resp.status_code}")

except ImportError as e:
    fail("FastAPI app importable for degraded-mode tests", str(e))
except Exception as e:
    fail("Degraded-mode gate raised unexpected error", str(e))

# ── Gate 5: Release Ops ────────────────────────────────────────────────────────
gate(5, "Release Ops")

ops_checks = [
    (REPO_ROOT / "RELEASE.md",                           "RELEASE.md at project root"),
    (REPO_ROOT / ".github" / "workflows" / "ci.yml",     ".github/workflows/ci.yml"),
    (REPO_ROOT / ".github" / "pull_request_template.md", ".github/pull_request_template.md"),
    (REPO_ROOT / "Makefile",                              "Root Makefile"),
    (PROJECT_ROOT / ".env",                               ".env config file"),
]
for path, label in ops_checks:
    if path.exists():
        ok(label)
    else:
        fail(label, f"not found at {path.relative_to(REPO_ROOT) if REPO_ROOT in path.parents else path}")

# Check RELEASE.md has rollback section
release_md = REPO_ROOT / "RELEASE.md"
if release_md.exists():
    content = release_md.read_text()
    if "## Rollback" in content:
        ok("RELEASE.md has Rollback section")
    else:
        warn("RELEASE.md missing ## Rollback section")
    if "## Known Limitations" in content:
        ok("RELEASE.md has Known Limitations section")
    else:
        warn("RELEASE.md missing ## Known Limitations section")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'═' * 56}")

if warnings:
    print(f"{YELLOW}  Warnings ({len(warnings)}):{RESET}")
    for w in warnings:
        print(f"    {WARN}  {w}")

if failures:
    print(f"{RED}  ✗ NO-GO — {len(failures)} check(s) failed:{RESET}")
    for f in failures:
        print(f"    •  {f}")
    print(f"{'═' * 56}\n")
    sys.exit(1)
else:
    print(f"{GREEN}  ✓ GO — all backend release gates passed{RESET}")
    if warnings:
        print(f"{YELLOW}    ({len(warnings)} warning(s) — review before merge){RESET}")
    print(f"{'═' * 56}\n")
    sys.exit(0)
