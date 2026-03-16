"""
Phase III governance tests.
Covers:
  - Unit: SafetyPolicy threshold behaviour (abstain / defer / normal)
  - Unit: DriftMonitor score calculations and alert logic
  - Unit: Fairness metric computation and parity checks
  - Unit: aggregate_epoch_predictions full-epoch correctness
  - Integration: /classify returns decisionMode / reasonCodes / driftAlert fields
  - Integration: governed response on uncertain + degraded model paths
  - Load/perf: sustained classify load stays within latency SLO
"""

import time
import numpy as np
import pytest
import torch
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

# ── ML modules (pure-Python, no FastAPI needed) ───────────────────────────────
from app.ml.policy import SafetyPolicy, PolicyConfig, DecisionMode, PolicyDecision
from app.ml.drift import DriftMonitor, DriftStatus
from app.ml.fairness import (
    compute_group_metrics,
    parity_check,
    generate_fairness_report,
    PARITY_THRESHOLD,
    MIN_GROUP_SAMPLES,
)
from app.ml.train_utils import aggregate_epoch_predictions
from app.middleware import clear_rate_limits

# ── FastAPI TestClient ────────────────────────────────────────────────────────
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_rate_limits():
    clear_rate_limits()
    yield
    clear_rate_limits()


# ─────────────────────────────────────────────────────────────────────────────
# 1. SafetyPolicy — unit
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSafetyPolicy:
    """Verify the three-level decision ladder."""

    def setup_method(self):
        self.policy = SafetyPolicy()
        self.cfg = self.policy.config

    # --- ABSTAIN paths ---

    def test_abstain_when_both_scores_below_threshold(self):
        """Both state and cat below abstain threshold → ABSTAIN."""
        d = self.policy.evaluate(
            state_score=self.cfg.state_abstain_threshold - 0.01,
            cat_scores=[self.cfg.cat_abstain_threshold - 0.01],
            uncertain=False,
        )
        assert d.mode == DecisionMode.ABSTAIN
        assert "low_state_confidence" in d.reason_codes
        assert "low_category_confidence" in d.reason_codes
        assert d.safe_state == SafetyPolicy.SAFE_STATE
        assert d.safe_next_step == SafetyPolicy.SAFE_NEXT_STEP

    def test_abstain_when_uncertain_flag_set(self):
        """uncertain=True overrides any score — still ABSTAIN."""
        d = self.policy.evaluate(
            state_score=0.99,
            cat_scores=[0.99],
            uncertain=True,
        )
        assert d.mode == DecisionMode.ABSTAIN
        assert "model_uncertain_flag" in d.reason_codes

    def test_abstain_overrides_high_cat_score_if_state_low(self):
        """If uncertain=True, high cat score doesn't prevent ABSTAIN."""
        d = self.policy.evaluate(
            state_score=0.10,
            cat_scores=[0.90],
            uncertain=True,
        )
        assert d.mode == DecisionMode.ABSTAIN

    # --- DEFER paths ---

    def test_defer_when_state_in_mid_range(self):
        """state below defer_threshold but above abstain → DEFER."""
        mid_state = (self.cfg.state_abstain_threshold + self.cfg.state_defer_threshold) / 2
        d = self.policy.evaluate(
            state_score=mid_state,
            cat_scores=[0.80],   # cat is strong
            uncertain=False,
        )
        assert d.mode == DecisionMode.DEFER
        assert "state_confidence_below_defer_threshold" in d.reason_codes
        assert d.safe_state == SafetyPolicy.SAFE_STATE
        assert d.safe_next_step == SafetyPolicy.SAFE_NEXT_STEP

    def test_defer_when_no_category_above_threshold(self):
        """Strong state but no category passes threshold → DEFER (no safe_state override)."""
        d = self.policy.evaluate(
            state_score=self.cfg.state_defer_threshold + 0.10,
            cat_scores=[self.cfg.cat_abstain_threshold - 0.01],
            uncertain=False,
        )
        assert d.mode == DecisionMode.DEFER
        assert "no_categories_above_threshold" in d.reason_codes
        assert d.safe_state is None  # state kept, only next_step overridden

    def test_defer_empty_cat_scores_uses_default_zero(self):
        """Empty cat_scores → cat_max=0.0 → no cat above threshold."""
        d = self.policy.evaluate(
            state_score=self.cfg.state_defer_threshold + 0.10,
            cat_scores=[],
            uncertain=False,
        )
        assert d.mode == DecisionMode.DEFER

    # --- NORMAL path ---

    def test_normal_when_all_scores_high(self):
        d = self.policy.evaluate(
            state_score=0.90,
            cat_scores=[0.80, 0.60],
            uncertain=False,
        )
        assert d.mode == DecisionMode.NORMAL
        assert d.reason_codes == []
        assert d.safe_state is None
        assert d.safe_next_step == ""

    def test_normal_at_exact_boundary(self):
        """At exactly the defer threshold → still NORMAL (strictly less than triggers DEFER)."""
        d = self.policy.evaluate(
            state_score=self.cfg.state_defer_threshold,
            cat_scores=[self.cfg.cat_abstain_threshold],
            uncertain=False,
        )
        assert d.mode == DecisionMode.NORMAL

    # --- Custom config ---

    def test_custom_config_changes_thresholds(self):
        strict_cfg = PolicyConfig(
            state_abstain_threshold=0.60,
            cat_abstain_threshold=0.60,
            state_defer_threshold=0.80,
        )
        policy = SafetyPolicy(config=strict_cfg)
        # Scores that would be NORMAL with defaults are ABSTAIN with strict config
        d = policy.evaluate(state_score=0.50, cat_scores=[0.50], uncertain=False)
        assert d.mode == DecisionMode.ABSTAIN


# ─────────────────────────────────────────────────────────────────────────────
# 2. DriftMonitor — unit
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestDriftMonitor:

    def _feed(self, monitor: DriftMonitor, norms: list, states: list):
        for norm, state in zip(norms, states):
            monitor.observe(norm, state)

    def test_insufficient_data_returns_no_alert(self):
        m = DriftMonitor()
        self._feed(m, [1.0] * 5, ["continue"] * 5)
        s = m.status()
        assert not s.alert
        assert "insufficient_data" in s.alert_reasons

    def test_stable_input_no_alert_after_warmup(self):
        m = DriftMonitor()
        # Fill both baseline and recent with identical norms + single state
        self._feed(m, [2.0] * (DriftMonitor.BASELINE_N + DriftMonitor.RECENT_N),
                   ["continue"] * (DriftMonitor.BASELINE_N + DriftMonitor.RECENT_N))
        s = m.status()
        assert not s.alert
        assert s.input_drift_score < DriftMonitor.INPUT_DRIFT_THRESHOLD
        assert s.output_drift_score < DriftMonitor.OUTPUT_DRIFT_THRESHOLD

    def test_large_norm_shift_triggers_input_drift_alert(self):
        m = DriftMonitor()
        # Baseline: small norms
        self._feed(m, [1.0] * DriftMonitor.BASELINE_N, ["continue"] * DriftMonitor.BASELINE_N)
        # Recent: norms shifted up by 60% (well above 25% threshold)
        self._feed(m, [1.6] * DriftMonitor.RECENT_N, ["continue"] * DriftMonitor.RECENT_N)
        s = m.status()
        assert s.alert
        assert any("input_drift" in r for r in s.alert_reasons)

    def test_state_distribution_shift_triggers_output_drift_alert(self):
        m = DriftMonitor()
        # Baseline: all "continue"
        self._feed(m, [1.0] * DriftMonitor.BASELINE_N, ["continue"] * DriftMonitor.BASELINE_N)
        # Recent: all "blocked" — full distribution shift
        self._feed(m, [1.0] * DriftMonitor.RECENT_N, ["blocked"] * DriftMonitor.RECENT_N)
        s = m.status()
        assert s.alert
        assert any("output_drift" in r for r in s.alert_reasons)

    def test_baseline_and_recent_counts_correct(self):
        m = DriftMonitor()
        total = DriftMonitor.BASELINE_N + DriftMonitor.RECENT_N
        self._feed(m, [1.0] * total, ["continue"] * total)
        s = m.status()
        assert s.baseline_n == DriftMonitor.BASELINE_N
        assert s.recent_n == DriftMonitor.RECENT_N

    def test_reset_to_recent_clears_windows(self):
        m = DriftMonitor()
        self._feed(m, [1.0] * DriftMonitor.BASELINE_N, ["continue"] * DriftMonitor.BASELINE_N)
        self._feed(m, [1.0] * DriftMonitor.RECENT_N, ["continue"] * DriftMonitor.RECENT_N)
        m.reset_to_recent()
        s = m.status()
        # After reset, recent window is empty → insufficient data
        assert not s.alert

    def test_input_drift_score_proportional_to_shift(self):
        m = DriftMonitor()
        self._feed(m, [1.0] * DriftMonitor.BASELINE_N, ["continue"] * DriftMonitor.BASELINE_N)
        self._feed(m, [2.0] * DriftMonitor.RECENT_N, ["continue"] * DriftMonitor.RECENT_N)
        s = m.status()
        # Mean shifted from 1.0 to 2.0 → ~100% relative shift
        assert s.input_drift_score > 0.90


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fairness — unit
# ─────────────────────────────────────────────────────────────────────────────

def _make_arrays(n: int, n_cats: int = 3, state_acc: float = 1.0, cat_f1: float = 1.0):
    """Generate fake perfect or imperfect arrays for testing."""
    state_targets = np.zeros(n, dtype=int)
    state_preds   = np.zeros(n, dtype=int) if state_acc == 1.0 else np.ones(n, dtype=int)
    cat_targets   = np.ones((n, n_cats), dtype=int)
    cat_preds     = (np.ones((n, n_cats), dtype=int) if cat_f1 == 1.0
                     else np.zeros((n, n_cats), dtype=int))
    return state_preds, state_targets, cat_preds, cat_targets


@pytest.mark.unit
class TestFairness:

    def test_single_group_metrics_correct(self):
        n = 20
        sp, st, cp, ct = _make_arrays(n)
        groups = ["default"] * n
        result = compute_group_metrics(sp, st, cp, ct, groups)
        assert "default" in result
        assert result["default"]["n_samples"] == n
        assert result["default"]["state_accuracy"] == pytest.approx(1.0)
        assert result["default"]["cat_micro_f1"] == pytest.approx(1.0)

    def test_two_groups_computed_separately(self):
        n = 20
        sp_a, st_a, cp_a, ct_a = _make_arrays(n, state_acc=1.0)
        sp_b, st_b, cp_b, ct_b = _make_arrays(n, state_acc=0.0)
        state_preds   = np.concatenate([sp_a, sp_b])
        state_targets = np.concatenate([st_a, st_b])
        cat_preds     = np.concatenate([cp_a, cp_b])
        cat_targets   = np.concatenate([ct_a, ct_b])
        groups = ["A"] * n + ["B"] * n

        result = compute_group_metrics(state_preds, state_targets, cat_preds, cat_targets, groups)
        assert result["A"]["state_accuracy"] == pytest.approx(1.0)
        assert result["B"]["state_accuracy"] == pytest.approx(0.0)

    def test_parity_check_passes_within_threshold(self):
        group_metrics = {
            "A": {"n_samples": 10, "state_accuracy": 0.90, "cat_micro_f1": 0.85},
            "B": {"n_samples": 10, "state_accuracy": 0.80, "cat_micro_f1": 0.75},
        }
        checks = parity_check(group_metrics)
        # gap = 0.10 < PARITY_THRESHOLD (0.15) → pass
        assert checks["state_accuracy"]["status"] == "pass"
        assert checks["cat_micro_f1"]["status"] == "pass"

    def test_parity_check_fails_above_threshold(self):
        group_metrics = {
            "A": {"n_samples": 10, "state_accuracy": 0.95, "cat_micro_f1": 0.90},
            "B": {"n_samples": 10, "state_accuracy": 0.70, "cat_micro_f1": 0.60},
        }
        checks = parity_check(group_metrics)
        # gap = 0.25 > PARITY_THRESHOLD (0.15) → fail
        assert checks["state_accuracy"]["status"] == "fail"
        assert checks["cat_micro_f1"]["status"] == "fail"

    def test_parity_check_skips_small_groups(self):
        group_metrics = {
            "A": {"n_samples": MIN_GROUP_SAMPLES - 1, "state_accuracy": 0.10, "cat_micro_f1": 0.10},
            "B": {"n_samples": 10, "state_accuracy": 0.90, "cat_micro_f1": 0.90},
        }
        checks = parity_check(group_metrics)
        # Only 1 eligible group → skip
        assert checks["state_accuracy"]["status"] == "skip"

    def test_parity_check_skips_when_single_group(self):
        group_metrics = {
            "only": {"n_samples": 20, "state_accuracy": 0.80, "cat_micro_f1": 0.75},
        }
        checks = parity_check(group_metrics)
        assert all(v["status"] == "skip" for v in checks.values())

    def test_generate_fairness_report_overall_pass(self):
        gm = {"A": {"n_samples": 10, "state_accuracy": 0.90, "cat_micro_f1": 0.85},
              "B": {"n_samples": 10, "state_accuracy": 0.85, "cat_micro_f1": 0.80}}
        checks = parity_check(gm)
        report = generate_fairness_report(gm, checks, run_id="test_run")
        assert report["overall_pass"] is True

    def test_generate_fairness_report_overall_fail(self):
        gm = {"A": {"n_samples": 10, "state_accuracy": 0.99, "cat_micro_f1": 0.99},
              "B": {"n_samples": 10, "state_accuracy": 0.50, "cat_micro_f1": 0.50}}
        checks = parity_check(gm)
        report = generate_fairness_report(gm, checks, run_id="test_run")
        assert report["overall_pass"] is False

    def test_generate_fairness_report_writes_file(self, tmp_path):
        gm = {"default": {"n_samples": 10, "state_accuracy": 0.90, "cat_micro_f1": 0.85}}
        checks = parity_check(gm)
        out = tmp_path / "fairness_report.json"
        report = generate_fairness_report(gm, checks, run_id="test_run", output_path=out)
        assert out.exists()
        assert report["run_id"] == "test_run"


# ─────────────────────────────────────────────────────────────────────────────
# 4. aggregate_epoch_predictions — regression
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestAggregateEpochPredictions:
    """Verify that aggregation collects all batches, not just the last one."""

    def _make_batch_pred(self, batch_size: int, n_cats: int, n_states: int):
        return {
            "categories": {"predictions": torch.zeros(batch_size, n_cats)},
            "state":      {"predictions": torch.zeros(batch_size, n_states)},
        }

    def _make_batch_tgt(self, batch_size: int, n_cats: int):
        return {
            "cat_target":   torch.zeros(batch_size, n_cats),
            "state_target": torch.zeros(batch_size, dtype=torch.long),
        }

    def test_concatenates_all_batches(self):
        n_cats, n_states, batch = 5, 3, 8
        n_batches = 4
        preds = [self._make_batch_pred(batch, n_cats, n_states) for _ in range(n_batches)]
        tgts  = [self._make_batch_tgt(batch, n_cats) for _ in range(n_batches)]

        agg_p, agg_t = aggregate_epoch_predictions(preds, tgts)

        assert agg_p["categories"]["predictions"].shape == (n_batches * batch, n_cats)
        assert agg_p["state"]["predictions"].shape      == (n_batches * batch, n_states)
        assert agg_t["cat_target"].shape                == (n_batches * batch, n_cats)
        assert agg_t["state_target"].shape              == (n_batches * batch,)

    def test_last_batch_only_would_miss_samples(self):
        """Demonstrate the bug that aggregate_epoch_predictions fixes."""
        n_cats, n_states, batch = 5, 3, 8
        n_batches = 4
        all_preds = []
        all_tgts  = []

        # Give each batch a different "fingerprint" value
        for i in range(n_batches):
            p = self._make_batch_pred(batch, n_cats, n_states)
            p["categories"]["predictions"].fill_(float(i))
            t = self._make_batch_tgt(batch, n_cats)
            t["cat_target"].fill_(float(i))
            all_preds.append(p)
            all_tgts.append(t)

        agg_p, agg_t = aggregate_epoch_predictions(all_preds, all_tgts)
        total = n_batches * batch

        # Aggregated result has all N samples; last-batch-only would only have `batch`
        assert agg_p["categories"]["predictions"].shape[0] == total
        # And values from batch 0 are present (would be absent if last-batch-only)
        assert float(agg_p["categories"]["predictions"][0, 0]) == 0.0
        # Values from last batch are also present
        assert float(agg_p["categories"]["predictions"][-1, 0]) == float(n_batches - 1)

    def test_single_batch_round_trips(self):
        n_cats, n_states, batch = 4, 2, 16
        p = [self._make_batch_pred(batch, n_cats, n_states)]
        t = [self._make_batch_tgt(batch, n_cats)]
        agg_p, agg_t = aggregate_epoch_predictions(p, t)
        assert agg_p["categories"]["predictions"].shape == (batch, n_cats)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Integration — /classify returns governance fields
# ─────────────────────────────────────────────────────────────────────────────

def _mock_classify_result(**overrides):
    """Return a minimal valid classify() result dict."""
    base = {
        "categories":    [{"label": "work", "score": 0.85}],
        "state":         {"label": "continue", "score": 0.90},
        "next_step":     {"template": "LogReflection", "confidence": 0.80},
        "uncertain":     False,
        "decision_mode": "normal",
        "reason_codes":  [],
        "drift_alert":   False,
        "inference_ms":  3.5,
    }
    base.update(overrides)
    return base


@pytest.mark.integration
class TestClassifyGovernanceFields:
    """Verify that governance metadata flows from model_service → HTTP response."""

    def test_classify_returns_decision_mode_normal(self):
        with patch("app.routers.classify.model_service") as mock_svc:
            mock_svc.classify.return_value = _mock_classify_result(decision_mode="normal")
            resp = client.post("/classify/", json={"text": "Shipped the PR, feeling great"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["decisionMode"] == "normal"
        assert body["reasonCodes"] == []
        assert body["driftAlert"] is False
        assert body["inferenceMsec"] == pytest.approx(3.5)

    def test_classify_returns_abstain_decision_mode(self):
        with patch("app.routers.classify.model_service") as mock_svc:
            mock_svc.classify.return_value = _mock_classify_result(
                decision_mode="abstain",
                reason_codes=["low_state_confidence", "model_uncertain_flag"],
                uncertain=True,
            )
            resp = client.post("/classify/", json={"text": "hmm"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["decisionMode"] == "abstain"
        assert "low_state_confidence" in body["reasonCodes"]
        assert body["uncertain"] is True

    def test_classify_returns_defer_decision_mode(self):
        with patch("app.routers.classify.model_service") as mock_svc:
            mock_svc.classify.return_value = _mock_classify_result(
                decision_mode="defer",
                reason_codes=["state_confidence_below_defer_threshold"],
            )
            resp = client.post("/classify/", json={"text": "maybe something"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["decisionMode"] == "defer"
        assert "state_confidence_below_defer_threshold" in body["reasonCodes"]

    def test_classify_returns_drift_alert_true(self):
        with patch("app.routers.classify.model_service") as mock_svc:
            mock_svc.classify.return_value = _mock_classify_result(drift_alert=True)
            resp = client.post("/classify/", json={"text": "unusual input pattern"})
        assert resp.status_code == 200
        assert resp.json()["driftAlert"] is True

    def test_classify_timeout_returns_503(self):
        """asyncio.TimeoutError in the executor surfaces as 503, not 500."""
        import asyncio
        with patch("app.routers.classify.model_service") as mock_svc:
            mock_svc.classify.side_effect = asyncio.TimeoutError()
            resp = client.post("/classify/", json={"text": "slow inference"})
        assert resp.status_code == 503

    def test_classify_runtime_error_returns_sanitized_503(self):
        """RuntimeError doesn't leak raw internals."""
        with patch("app.routers.classify.model_service") as mock_svc:
            mock_svc.classify.side_effect = RuntimeError("internal weights corrupt: layer 3")
            resp = client.post("/classify/", json={"text": "trigger error"})
        assert resp.status_code == 503
        body = resp.json()
        assert "corrupt" not in str(body)
        assert "weights" not in str(body)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Load / perf — latency SLO under sustained load
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.load
class TestGovernanceLoadPerf:
    """
    Sustained classify load verifies:
      - p95 wall-clock inference stays under 200 ms (mock removes real ML cost)
      - No 500-series errors (only 200, 429, 503 are acceptable)
      - ABSTAIN responses don't contain raw model internals
    """

    N_REQUESTS = 40
    P95_SLO_MS = 200.0

    def test_sustained_load_no_500_errors(self):
        with patch("app.routers.classify.model_service") as mock_svc:
            mock_svc.classify.return_value = _mock_classify_result()
            statuses = []
            for _ in range(self.N_REQUESTS):
                r = client.post("/classify/", json={"text": "working on feature"})
                statuses.append(r.status_code)

        unexpected = [s for s in statuses if s not in (200, 429, 503)]
        assert unexpected == [], f"Unexpected status codes: {unexpected}"

    def test_p95_latency_within_slo(self):
        """End-to-end HTTP round-trip (mocked ML) must stay under SLO."""
        with patch("app.routers.classify.model_service") as mock_svc:
            mock_svc.classify.return_value = _mock_classify_result(inference_ms=2.0)
            latencies_ms = []
            for _ in range(self.N_REQUESTS):
                t0 = time.monotonic()
                client.post("/classify/", json={"text": "testing latency"})
                latencies_ms.append((time.monotonic() - t0) * 1000)

        p95 = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]
        assert p95 < self.P95_SLO_MS, f"p95 latency {p95:.1f}ms exceeds SLO {self.P95_SLO_MS}ms"

    def test_abstain_responses_have_no_raw_internals(self):
        """Safe fallback mode must not surface internal model details."""
        with patch("app.routers.classify.model_service") as mock_svc:
            mock_svc.classify.return_value = _mock_classify_result(
                decision_mode="abstain",
                reason_codes=["model_uncertain_flag"],
                uncertain=True,
                state={"label": "continue", "score": 0.20},
            )
            resp = client.post("/classify/", json={"text": "unclear"})

        body = str(resp.json())
        for leaked_term in ("tensor", "logit", "weight", "torch", "cuda", "traceback"):
            assert leaked_term not in body.lower(), f"Leaked term '{leaked_term}' in response"
