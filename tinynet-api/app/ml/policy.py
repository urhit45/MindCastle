"""
Safety policy — translates raw model confidence into governed decision actions.

Decision ladder (checked top-down):
  uncertain OR (state < abstain AND cat_max < cat_abstain)  → ABSTAIN
  state < defer                                             → DEFER
  cat_max < cat_abstain                                     → DEFER
  otherwise                                                 → NORMAL

ABSTAIN / DEFER always carry safe_state + safe_next_step overrides so the
caller never has to propagate an unreviewed prediction.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class DecisionMode(str, Enum):
    NORMAL       = "normal"        # confident — trust prediction
    ABSTAIN      = "abstain"       # too uncertain — suppress prediction
    DEFER        = "defer"         # low-mid confidence — give safe default


@dataclass(frozen=True)
class PolicyConfig:
    # Below this state score *and* cat max score → ABSTAIN
    state_abstain_threshold: float = 0.35
    cat_abstain_threshold:   float = 0.25
    # Below this state score (but above abstain) → DEFER
    state_defer_threshold:   float = 0.50
    # Alert if uncertainty rate in a rolling window exceeds this
    max_uncertainty_rate:    float = 0.30


@dataclass
class PolicyDecision:
    mode:          DecisionMode
    reason_codes:  List[str]
    safe_state:    Optional[str]  # non-None on ABSTAIN/DEFER
    safe_next_step: str           # always present; empty string on NORMAL


class SafetyPolicy:
    """Stateless evaluator — cheap to call per inference."""

    SAFE_STATE     = "continue"
    SAFE_NEXT_STEP = "LogReflection"

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        self.config = config or PolicyConfig()

    def evaluate(
        self,
        state_score: float,
        cat_scores: List[float],
        uncertain: bool,
    ) -> PolicyDecision:
        cfg = self.config
        cat_max = max(cat_scores, default=0.0)
        reasons: List[str] = []

        # --- ABSTAIN: overall signal too weak ---
        if uncertain or (
            state_score < cfg.state_abstain_threshold
            and cat_max < cfg.cat_abstain_threshold
        ):
            if state_score < cfg.state_abstain_threshold:
                reasons.append("low_state_confidence")
            if cat_max < cfg.cat_abstain_threshold:
                reasons.append("low_category_confidence")
            if uncertain:
                reasons.append("model_uncertain_flag")
            return PolicyDecision(
                mode=DecisionMode.ABSTAIN,
                reason_codes=reasons,
                safe_state=self.SAFE_STATE,
                safe_next_step=self.SAFE_NEXT_STEP,
            )

        # --- DEFER: state confidence in mid range ---
        if state_score < cfg.state_defer_threshold:
            reasons.append("state_confidence_below_defer_threshold")
            return PolicyDecision(
                mode=DecisionMode.DEFER,
                reason_codes=reasons,
                safe_state=self.SAFE_STATE,
                safe_next_step=self.SAFE_NEXT_STEP,
            )

        # --- DEFER: no category above threshold ---
        if cat_max < cfg.cat_abstain_threshold:
            reasons.append("no_categories_above_threshold")
            return PolicyDecision(
                mode=DecisionMode.DEFER,
                reason_codes=reasons,
                safe_state=None,          # keep model's state
                safe_next_step=self.SAFE_NEXT_STEP,
            )

        # --- NORMAL ---
        return PolicyDecision(
            mode=DecisionMode.NORMAL,
            reason_codes=[],
            safe_state=None,
            safe_next_step="",
        )
