"""
TinyNet model service — train-on-first-run, singleton loader for classify route.

Phase III additions:
  • SafetyPolicy: translates confidence scores into a governed decision mode.
  • DriftMonitor: tracks input/output distribution drift at runtime.
  • LatencyTracker: rolling p50/p95/p99 of inference wall-clock times.

classify() now returns an extended dict that includes governance metadata so
the classify router can surface it to callers without any extra logic.
"""

import json
import logging
import time
import yaml
import torch
import numpy as np
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tinynet import TinyNet
from .vectorizer import HashingVectorizer512
from .policy import SafetyPolicy, DecisionMode
from .drift import DriftMonitor

logger = logging.getLogger(__name__)

LABELS_PATH     = Path(__file__).parent.parent.parent.parent / "backend" / "config" / "labels.yaml"
TRAIN_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "train.jsonl"
MODEL_DIR       = Path(__file__).parent.parent.parent / "models"
MODEL_PATH      = MODEL_DIR / "best.pt"


# ─── Latency tracker ─────────────────────────────────────────────────────────

class _LatencyTracker:
    """Rolling percentile tracker for inference wall-clock times (ms)."""

    def __init__(self, maxlen: int = 200) -> None:
        self._buf: deque = deque(maxlen=maxlen)

    def record(self, ms: float) -> None:
        self._buf.append(ms)

    def percentiles(self) -> Dict[str, float]:
        if not self._buf:
            return {}
        s = sorted(self._buf)
        n = len(s)
        return {
            "p50_ms": round(s[n // 2], 2),
            "p95_ms": round(s[min(int(n * 0.95), n - 1)], 2),
            "p99_ms": round(s[min(int(n * 0.99), n - 1)], 2),
            "n":      n,
        }


# ─── Training helpers ─────────────────────────────────────────────────────────

def _load_labels() -> tuple[List[str], List[str], List[str]]:
    with open(LABELS_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["categories"], cfg["states"], cfg.get("next_step_templates", [])


def _load_train_data() -> List[Dict]:
    data = []
    with open(TRAIN_DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return data


def _train(model: TinyNet, categories: List[str], states: List[str]) -> None:
    """Quick training run on train.jsonl — completes in < 1s on CPU."""
    from torch.utils.data import DataLoader, Dataset

    data = _load_train_data()
    if not data:
        logger.warning("No training data found, skipping training")
        return

    cat_idx   = {c: i for i, c in enumerate(categories)}
    state_idx = {s: i for i, s in enumerate(states)}
    vectorizer = HashingVectorizer512()

    class _DS(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            row = self.rows[i]
            x = torch.FloatTensor(vectorizer.encode(row["text"]))
            cats = row.get("categories", [])
            cat_t = torch.zeros(len(categories))
            for c in cats:
                if c in cat_idx:
                    cat_t[cat_idx[c]] = 1.0
            state    = row.get("state", "continue")
            state_t  = torch.LongTensor([state_idx.get(state, 0)])
            next_t   = torch.LongTensor([0])
            return x, cat_t, state_t, next_t

    loader = DataLoader(_DS(data), batch_size=16, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(30):
        for x, cat_t, state_t, next_t in loader:
            _, cat_l, state_l, next_l = model(x)
            losses = model.compute_losses(
                cat_l, state_l, next_l,
                cat_t, state_t.squeeze(1), next_t.squeeze(1),
            )
            opt.zero_grad()
            losses["total"].backward()
            opt.step()

    logger.info("TinyNet training complete (%d samples, 30 epochs)", len(data))


# ─── ModelService ─────────────────────────────────────────────────────────────

class ModelService:
    """Singleton that owns the trained TinyNet + vectorizer + governance stack."""

    def __init__(self) -> None:
        self.model:               Optional[TinyNet]              = None
        self.vectorizer:          Optional[HashingVectorizer512] = None
        self.categories:          List[str]                      = []
        self.states:              List[str]                      = []
        self.next_step_templates: List[str]                      = []
        self.cat_idx:             Dict[str, int]                 = {}
        self.state_idx:           Dict[str, int]                 = {}
        # Governance
        self.policy:          SafetyPolicy   = SafetyPolicy()
        self.drift_monitor:   DriftMonitor   = DriftMonitor()
        self.latency_tracker: _LatencyTracker = _LatencyTracker()

    def load(self) -> None:
        categories, states, templates = _load_labels()
        self.categories          = categories
        self.states              = states
        self.next_step_templates = templates
        self.cat_idx             = {c: i for i, c in enumerate(categories)}
        self.state_idx           = {s: i for i, s in enumerate(states)}

        self.vectorizer = HashingVectorizer512()
        self.model      = TinyNet(config_path=str(LABELS_PATH))

        if MODEL_PATH.exists():
            checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded TinyNet weights from %s", MODEL_PATH)
        else:
            logger.info("No saved weights — training TinyNet from train.jsonl")
            _train(self.model, categories, states)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": self.model.state_dict()}, MODEL_PATH)
            logger.info("Saved trained weights to %s", MODEL_PATH)

        self.model.eval()

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Run text through TinyNet + safety policy + drift monitor.

        Returns:
            categories, state, next_step, uncertain  — raw model output
            decision_mode, reason_codes              — policy governance
            drift_alert                              — drift monitor flag
            inference_ms                             — wall-clock time
        """
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("ModelService not loaded")

        t0 = time.monotonic()

        vec = self.vectorizer.encode(text)
        x   = torch.FloatTensor(vec).unsqueeze(0)  # (1, 512)

        with torch.no_grad():
            _, cat_logits, state_logits, next_logits = self.model(x)

        cat_probs   = torch.sigmoid(cat_logits).squeeze(0).tolist()
        state_probs = torch.softmax(state_logits, dim=1).squeeze(0).tolist()
        next_probs  = torch.softmax(next_logits, dim=1).squeeze(0).tolist()

        # ── raw predictions ──────────────────────────────────────────────────
        categories = sorted(
            [{"label": c, "score": round(cat_probs[i], 3)}
             for i, c in enumerate(self.categories) if cat_probs[i] > 0.2],
            key=lambda e: e["score"], reverse=True,
        )[:5]
        if not categories:
            best = int(np.argmax(cat_probs))
            categories = [{"label": self.categories[best], "score": round(cat_probs[best], 3)}]

        best_state_idx = int(np.argmax(state_probs))
        state = {
            "label": self.states[best_state_idx],
            "score": round(state_probs[best_state_idx], 3),
        }

        best_next_idx  = int(np.argmax(next_probs))
        next_step_label = (
            self.next_step_templates[best_next_idx]
            if best_next_idx < len(self.next_step_templates)
            else "LogReflection"
        )
        next_step = {"template": next_step_label, "confidence": round(next_probs[best_next_idx], 3)}

        uncertain = state["score"] < 0.4 and all(c["score"] < 0.4 for c in categories)

        # ── safety policy ────────────────────────────────────────────────────
        decision = self.policy.evaluate(
            state_score=state["score"],
            cat_scores=[c["score"] for c in categories],
            uncertain=uncertain,
        )

        # Apply overrides when policy deems prediction unsafe to surface
        effective_state     = state.copy()
        effective_next_step = next_step.copy()
        if decision.mode != DecisionMode.NORMAL:
            if decision.safe_state:
                effective_state = {"label": decision.safe_state, "score": state["score"]}
            effective_next_step = {"template": decision.safe_next_step, "confidence": 1.0}

        # ── drift monitoring ─────────────────────────────────────────────────
        vec_norm = float(np.linalg.norm(vec))
        self.drift_monitor.observe(vec_norm, effective_state["label"])
        drift_status = self.drift_monitor.status()
        if drift_status.alert:
            logger.warning("drift_alert reasons=%s", drift_status.alert_reasons)

        # ── latency ──────────────────────────────────────────────────────────
        inference_ms = (time.monotonic() - t0) * 1000
        self.latency_tracker.record(inference_ms)

        return {
            "categories":    categories,
            "state":         effective_state,
            "next_step":     effective_next_step,
            "uncertain":     uncertain,
            # governance
            "decision_mode": decision.mode.value,
            "reason_codes":  decision.reason_codes,
            "drift_alert":   drift_status.alert,
            "inference_ms":  round(inference_ms, 2),
        }


# Global singleton
model_service = ModelService()
