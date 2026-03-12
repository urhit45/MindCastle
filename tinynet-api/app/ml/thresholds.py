"""
Thresholds and uncertainty detection for TinyNet classification.
"""

import torch
from typing import Union, List


# Threshold constants
TAU_CAT = 0.55      # Category probability threshold
MARGIN_CAT = 0.15   # Category margin threshold
TAU_LINK = 0.35     # Link similarity threshold
TAU_STATE = 0.6     # State confidence threshold


def is_uncertain(
    cat_probs: Union[torch.Tensor, List[float]], 
    margin: float, 
    state_conf: float
) -> bool:
    """
    Determine if classification is uncertain.
    
    Args:
        cat_probs: Category probabilities (tensor or list)
        margin: Difference between top-1 and top-2 category probabilities
        state_conf: State confidence score
        
    Returns:
        True if classification is uncertain
    """
    # Convert tensor to list if needed
    if isinstance(cat_probs, torch.Tensor):
        cat_probs = cat_probs.tolist()
    
    if not cat_probs:
        return True
    
    # Check category probability threshold
    max_cat_prob = max(cat_probs)
    if max_cat_prob <= TAU_CAT:
        return True
    
    # Check category margin threshold
    if margin <= MARGIN_CAT:
        return True
    
    # Check state confidence threshold
    if state_conf <= TAU_STATE:
        return True
    
    return False


def get_category_threshold() -> float:
    """Get the category probability threshold."""
    return TAU_CAT


def get_margin_threshold() -> float:
    """Get the category margin threshold."""
    return MARGIN_CAT


def get_link_threshold() -> float:
    """Get the link similarity threshold."""
    return TAU_LINK


def get_state_threshold() -> float:
    """Get the state confidence threshold."""
    return TAU_STATE


def adjust_thresholds(
    cat_threshold: float = None,
    margin_threshold: float = None,
    link_threshold: float = None,
    state_threshold: float = None
):
    """
    Adjust thresholds dynamically.
    
    Args:
        cat_threshold: New category threshold
        margin_threshold: New margin threshold  
        link_threshold: New link threshold
        state_threshold: New state threshold
    """
    global TAU_CAT, MARGIN_CAT, TAU_LINK, TAU_STATE
    
    if cat_threshold is not None:
        TAU_CAT = cat_threshold
    
    if margin_threshold is not None:
        MARGIN_CAT = margin_threshold
    
    if link_threshold is not None:
        TAU_LINK = link_threshold
    
    if state_threshold is not None:
        TAU_STATE = state_threshold


def get_threshold_summary() -> dict:
    """Get current threshold values."""
    return {
        "category_threshold": TAU_CAT,
        "margin_threshold": MARGIN_CAT,
        "link_threshold": TAU_LINK,
        "state_threshold": TAU_STATE
    }


def determine_route(uncertain: bool, state: str) -> str:
    """
    Determine the UI routing hint based on classification confidence and state.
    
    Args:
        uncertain: Whether the classification is uncertain
        state: The predicted state label
        
    Returns:
        Routing hint: "needs_confirm" | "suggest_plan" | "auto_save_ok"
    """
    if uncertain:
        return "needs_confirm"
    elif state == "blocked":
        return "suggest_plan"
    else:
        return "auto_save_ok"


def get_routing_summary() -> dict:
    """Get routing logic summary for documentation."""
    return {
        "routing_logic": {
            "uncertain == True": "needs_confirm",
            "state == 'blocked'": "suggest_plan", 
            "default": "auto_save_ok"
        },
        "ui_actions": {
            "needs_confirm": "Show confirm chips for top-3 categories and 'Link to...' dropdown",
            "suggest_plan": "Call /reason endpoint for planning suggestions",
            "auto_save_ok": "Automatically save the classification"
        }
    }
