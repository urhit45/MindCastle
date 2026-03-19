"""
Drift monitor — detects covariate shift and label distribution shift at runtime.

Two signals tracked with a two-window (baseline vs. recent) approach:

  input_drift   : relative change in mean feature-vector L2 norm.
                  Proxy for covariate shift; no stored raw vectors needed.

  output_drift  : max absolute shift in per-state prediction frequency.
                  Proxy for label distribution shift.

The baseline window is seeded from the first BASELINE_N observations.
The recent window is a fixed-size deque (last RECENT_N observations).
"""

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DriftStatus:
    input_drift_score:  float       # 0..∞; relative change from baseline mean
    output_drift_score: float       # 0..1; max absolute state-freq shift
    alert:              bool
    alert_reasons:      List[str]
    baseline_n:         int
    recent_n:           int


class DriftMonitor:
    """
    Call observe(vec_norm, predicted_state) on every inference.
    Call status() to get the current drift assessment.
    """

    BASELINE_N = 100
    RECENT_N   = 50

    INPUT_DRIFT_THRESHOLD  = 0.25   # >25% relative norm shift
    OUTPUT_DRIFT_THRESHOLD = 0.20   # >20% absolute state-freq shift

    def __init__(self) -> None:
        self._baseline_norms:  List[float]        = []
        self._recent_norms:    deque               = deque(maxlen=self.RECENT_N)
        self._baseline_states: Dict[str, int]      = defaultdict(int)
        self._recent_states:   deque               = deque(maxlen=self.RECENT_N)
        self._total_n: int = 0

    # ── public ─────────────────────────────────────────────────────────────────

    def observe(self, vec_norm: float, predicted_state: str) -> None:
        self._total_n += 1
        if len(self._baseline_norms) < self.BASELINE_N:
            self._baseline_norms.append(vec_norm)
            self._baseline_states[predicted_state] += 1
        self._recent_norms.append(vec_norm)
        self._recent_states.append(predicted_state)

    def status(self) -> DriftStatus:
        b_n = len(self._baseline_norms)
        r_n = len(self._recent_norms)

        if b_n < 10 or r_n < 10:
            return DriftStatus(
                input_drift_score=0.0,
                output_drift_score=0.0,
                alert=False,
                alert_reasons=["insufficient_data"],
                baseline_n=b_n,
                recent_n=r_n,
            )

        # Input drift: relative shift in mean norm
        b_mean = sum(self._baseline_norms) / b_n
        r_mean = sum(self._recent_norms) / r_n
        input_score = abs(r_mean - b_mean) / (b_mean + 1e-8)

        # Output drift: max absolute state-frequency shift
        b_total = sum(self._baseline_states.values())
        r_counts: Dict[str, int] = defaultdict(int)
        for s in self._recent_states:
            r_counts[s] += 1

        all_states = set(self._baseline_states) | set(r_counts)
        max_shift = max(
            abs(self._baseline_states.get(s, 0) / b_total
                - r_counts.get(s, 0) / r_n)
            for s in all_states
        )

        alert_reasons: List[str] = []
        if input_score > self.INPUT_DRIFT_THRESHOLD:
            alert_reasons.append(f"input_drift_{input_score:.2f}")
        if max_shift > self.OUTPUT_DRIFT_THRESHOLD:
            alert_reasons.append(f"output_drift_{max_shift:.2f}")

        return DriftStatus(
            input_drift_score=round(input_score, 4),
            output_drift_score=round(max_shift, 4),
            alert=bool(alert_reasons),
            alert_reasons=alert_reasons,
            baseline_n=b_n,
            recent_n=r_n,
        )

    def reset_to_recent(self) -> None:
        """Promote recent window to baseline (call after drift resolves)."""
        if len(self._recent_norms) >= self.RECENT_N:
            self._baseline_norms = list(self._recent_norms)
            self._baseline_states = defaultdict(int)
            for s in self._recent_states:
                self._baseline_states[s] += 1
        self._recent_norms.clear()
        self._recent_states.clear()
