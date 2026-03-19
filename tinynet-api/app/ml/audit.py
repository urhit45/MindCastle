"""
Run manifest + model card generation for reproducibility and audit.

write_run_manifest() — saved as run_manifest.json alongside each training run.
write_model_card()   — saved as model_card.json; stable across runs unless
                       model architecture / intended use changes.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Optional


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    """First 16 hex chars of SHA-256. Returns 'missing' if file absent."""
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ─── Run manifest ─────────────────────────────────────────────────────────────

def write_run_manifest(
    run_id:        str,
    data_path:     Path,
    labels_path:   Path,
    model_version: str,
    seed:          int,
    thresholds:    Dict,
    metrics:       Dict,
    output_path:   Path,
) -> Dict:
    """Persist a JSON record that makes the run reproducible and traceable."""
    manifest = {
        "run_id":        run_id,
        "created_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_version": model_version,
        "seed":          seed,
        "data_hash":     _file_hash(data_path),
        "labels_hash":   _file_hash(labels_path),
        "thresholds":    thresholds,
        "metrics":       metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ─── Model card ───────────────────────────────────────────────────────────────

_DEFAULT_MODEL_CARD: Dict = {
    "model_name": "TinyNet",
    "version":    "0.1.0",
    "intended_use": (
        "Personal productivity assistant: classify progress log entries into "
        "life-domain categories and momentum states to surface next actions."
    ),
    "out_of_scope": [
        "Medical or mental-health assessment",
        "High-stakes automated decisions",
        "Multi-language text (trained on English only)",
        "Real-time or safety-critical systems",
    ],
    "limitations": [
        "Small training set — bootstrapped from keyword rules",
        "No demographic parity guarantees without labelled group data",
        "Confidence scores are not calibrated probabilities",
        "Drift detection relies on L2 norm proxy, not full distribution test",
    ],
    "fairness_notes": (
        "Per-group parity checks run at training time. "
        "Maximum allowed group gap: 0.15 on state_accuracy and cat_micro_f1. "
        "Runtime drift monitoring active via DriftMonitor."
    ),
    "drift_thresholds": {
        "input_drift_alert":  0.25,
        "output_drift_alert": 0.20,
    },
    "safety_policy": {
        "abstain_state_threshold":  0.35,
        "abstain_cat_threshold":    0.25,
        "defer_state_threshold":    0.50,
        "safe_default_state":       "continue",
        "safe_default_next_step":   "LogReflection",
    },
}


def write_model_card(
    output_path: Path,
    overrides:   Optional[Dict] = None,
) -> Dict:
    card = {**_DEFAULT_MODEL_CARD, **(overrides or {})}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(card, f, indent=2)
    return card
