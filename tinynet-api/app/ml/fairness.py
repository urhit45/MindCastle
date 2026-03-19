"""
Offline fairness / bias evaluation.

Computes per-group state accuracy + category micro-F1, then runs a parity
check to verify the gap between best and worst groups is within tolerance.

Usage (called from scripts/train.py after training):

    from app.ml.fairness import compute_group_metrics, parity_check, generate_fairness_report

    group_metrics = compute_group_metrics(
        state_preds, state_targets, cat_preds_bin, cat_targets, groups
    )
    parity = parity_check(group_metrics)
    report = generate_fairness_report(group_metrics, parity, run_id, Path("runs/exp1/fairness_report.json"))
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Max allowed per-metric gap between best and worst group
PARITY_THRESHOLD = 0.15
# Minimum samples for a group to be included in parity checks
MIN_GROUP_SAMPLES = 5


def compute_group_metrics(
    state_preds:   np.ndarray,    # (N,) predicted state indices
    state_targets: np.ndarray,    # (N,) true state indices
    cat_preds_bin: np.ndarray,    # (N, C) binary predicted categories
    cat_targets:   np.ndarray,    # (N, C) binary true categories
    groups:        List[str],     # (N,) group label per sample
) -> Dict[str, Dict]:
    """Return {group_label: {n_samples, state_accuracy, cat_micro_f1}}."""
    unique_groups = sorted(set(groups))
    results: Dict[str, Dict] = {}

    for grp in unique_groups:
        idx = [i for i, g in enumerate(groups) if g == grp]
        n = len(idx)

        s_true = state_targets[idx]
        s_pred = state_preds[idx]
        state_acc = float(accuracy_score(s_true, s_pred))

        c_true = cat_targets[idx]    # (n, C)
        c_pred = cat_preds_bin[idx]  # (n, C)
        cat_f1 = 0.0
        if c_true.sum() > 0:
            cat_f1 = float(
                f1_score(c_true.flatten(), c_pred.flatten(), average="micro", zero_division=0)
            )

        results[grp] = {
            "n_samples":      n,
            "state_accuracy": round(state_acc, 4),
            "cat_micro_f1":   round(cat_f1, 4),
        }

    return results


def parity_check(group_metrics: Dict[str, Dict]) -> Dict:
    """
    For each scalar metric, compute min/max/gap across eligible groups and
    return pass/fail relative to PARITY_THRESHOLD.
    """
    eligible = {g: m for g, m in group_metrics.items() if m["n_samples"] >= MIN_GROUP_SAMPLES}
    checks: Dict[str, Dict] = {}

    for metric in ("state_accuracy", "cat_micro_f1"):
        values = [v[metric] for v in eligible.values()]
        if len(values) < 2:
            checks[metric] = {"status": "skip", "reason": "insufficient_groups"}
            continue
        gap = max(values) - min(values)
        checks[metric] = {
            "status":    "pass" if gap <= PARITY_THRESHOLD else "fail",
            "gap":       round(gap, 4),
            "max":       round(max(values), 4),
            "min":       round(min(values), 4),
            "threshold": PARITY_THRESHOLD,
        }

    return checks


def generate_fairness_report(
    group_metrics:  Dict[str, Dict],
    parity_checks:  Dict,
    run_id:         str,
    output_path:    Optional[Path] = None,
) -> Dict:
    overall_pass = all(
        c.get("status") == "pass"
        for c in parity_checks.values()
        if c.get("status") != "skip"
    )
    report = {
        "run_id":        run_id,
        "overall_pass":  overall_pass,
        "group_metrics": group_metrics,
        "parity_checks": parity_checks,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
    return report
