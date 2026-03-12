"""
Unit tests for thresholds and routing logic.
"""

import pytest
import torch
from app.ml.thresholds import (
    is_uncertain, 
    determine_route, 
    get_threshold_summary,
    get_routing_summary,
    adjust_thresholds,
    TAU_CAT,
    MARGIN_CAT,
    TAU_LINK,
    TAU_STATE
)


class TestUncertaintyDetection:
    """Test uncertainty detection logic."""
    
    def test_is_uncertain_with_low_category_probability(self):
        """Test uncertainty when max category probability is below threshold."""
        cat_probs = [0.4, 0.3, 0.2]  # Max is 0.4, below TAU_CAT (0.55)
        margin = 0.2  # Above MARGIN_CAT (0.15)
        state_conf = 0.8  # Above TAU_STATE (0.6)
        
        result = is_uncertain(cat_probs, margin, state_conf)
        assert result is True
    
    def test_is_uncertain_with_low_margin(self):
        """Test uncertainty when category margin is below threshold."""
        cat_probs = [0.6, 0.5, 0.2]  # Max is 0.6, above TAU_CAT (0.55)
        margin = 0.1  # Below MARGIN_CAT (0.15)
        state_conf = 0.8  # Above TAU_STATE (0.6)
        
        result = is_uncertain(cat_probs, margin, state_conf)
        assert result is True
    
    def test_is_uncertain_with_low_state_confidence(self):
        """Test uncertainty when state confidence is below threshold."""
        cat_probs = [0.6, 0.3, 0.2]  # Max is 0.6, above TAU_CAT (0.55)
        margin = 0.3  # Above MARGIN_CAT (0.15)
        state_conf = 0.5  # Below TAU_STATE (0.6)
        
        result = is_uncertain(cat_probs, margin, state_conf)
        assert result is True
    
    def test_is_uncertain_with_high_confidence(self):
        """Test no uncertainty when all thresholds are met."""
        cat_probs = [0.7, 0.4, 0.2]  # Max is 0.7, above TAU_CAT (0.55)
        margin = 0.3  # Above MARGIN_CAT (0.15)
        state_conf = 0.8  # Above TAU_STATE (0.6)
        
        result = is_uncertain(cat_probs, margin, state_conf)
        assert result is False
    
    def test_is_uncertain_with_empty_categories(self):
        """Test uncertainty when no categories are provided."""
        cat_probs = []
        margin = 0.5
        state_conf = 0.8
        
        result = is_uncertain(cat_probs, margin, state_conf)
        assert result is True
    
    def test_is_uncertain_with_torch_tensor(self):
        """Test uncertainty detection with PyTorch tensor input."""
        cat_probs = torch.tensor([0.4, 0.3, 0.2])
        margin = 0.2
        state_conf = 0.8
        
        result = is_uncertain(cat_probs, margin, state_conf)
        assert result is True
    
    def test_is_uncertain_edge_cases(self):
        """Test uncertainty detection with edge case values."""
        # Exactly at thresholds
        cat_probs = [0.55, 0.4, 0.3]  # Max is exactly TAU_CAT
        margin = 0.15  # Exactly MARGIN_CAT
        state_conf = 0.6  # Exactly TAU_STATE
        
        result = is_uncertain(cat_probs, margin, state_conf)
        # Should be uncertain because margin is exactly at threshold (edge case)
        # Note: The current logic considers exactly at threshold as uncertain
        assert result is True


class TestRoutingLogic:
    """Test routing logic for UI guidance."""
    
    def test_route_needs_confirm_when_uncertain(self):
        """Test route when classification is uncertain."""
        route = determine_route(uncertain=True, state="continue")
        assert route == "needs_confirm"
    
    def test_route_suggest_plan_when_blocked(self):
        """Test route when state is blocked."""
        route = determine_route(uncertain=False, state="blocked")
        assert route == "suggest_plan"
    
    def test_route_auto_save_ok_when_confident_and_not_blocked(self):
        """Test route when confident and not blocked."""
        route = determine_route(uncertain=False, state="continue")
        assert route == "auto_save_ok"
    
    def test_route_priority_uncertain_over_blocked(self):
        """Test that uncertain takes priority over blocked state."""
        route = determine_route(uncertain=True, state="blocked")
        assert route == "needs_confirm"
    
    def test_route_with_different_states(self):
        """Test routing with various state values."""
        states = ["start", "continue", "pause", "end", "idea"]
        for state in states:
            route = determine_route(uncertain=False, state=state)
            assert route == "auto_save_ok"
    
    def test_route_with_blocked_state(self):
        """Test routing specifically with blocked state."""
        route = determine_route(uncertain=False, state="blocked")
        assert route == "suggest_plan"


class TestThresholdManagement:
    """Test threshold adjustment and retrieval."""
    
    def test_get_threshold_summary(self):
        """Test getting current threshold values."""
        summary = get_threshold_summary()
        
        assert "category_threshold" in summary
        assert "margin_threshold" in summary
        assert "link_threshold" in summary
        assert "state_threshold" in summary
        
        assert summary["category_threshold"] == TAU_CAT
        assert summary["margin_threshold"] == MARGIN_CAT
        assert summary["link_threshold"] == TAU_LINK
        assert summary["state_threshold"] == TAU_STATE
    
    def test_adjust_thresholds(self):
        """Test dynamic threshold adjustment."""
        # Store original values
        original_cat = TAU_CAT
        original_margin = MARGIN_CAT
        
        try:
            # Adjust thresholds
            adjust_thresholds(cat_threshold=0.6, margin_threshold=0.2)
            
            # Verify changes
            summary = get_threshold_summary()
            assert summary["category_threshold"] == 0.6
            assert summary["margin_threshold"] == 0.2
            
            # Verify other thresholds unchanged
            assert summary["link_threshold"] == TAU_LINK
            assert summary["state_threshold"] == TAU_STATE
            
        finally:
            # Restore original values
            adjust_thresholds(cat_threshold=original_cat, margin_threshold=original_margin)
    
    def test_adjust_thresholds_partial(self):
        """Test adjusting only some thresholds."""
        original_cat = TAU_CAT
        
        try:
            # Adjust only category threshold
            adjust_thresholds(cat_threshold=0.65)
            
            summary = get_threshold_summary()
            assert summary["category_threshold"] == 0.65
            assert summary["margin_threshold"] == MARGIN_CAT  # Unchanged
            
        finally:
            # Restore original value
            adjust_thresholds(cat_threshold=original_cat)
    
    def test_threshold_constants(self):
        """Test that threshold constants have expected values."""
        assert TAU_CAT == 0.55
        assert MARGIN_CAT == 0.15
        assert TAU_LINK == 0.35
        assert TAU_STATE == 0.6


class TestRoutingDocumentation:
    """Test routing documentation and guidance."""
    
    def test_get_routing_summary(self):
        """Test getting routing logic documentation."""
        summary = get_routing_summary()
        
        assert "routing_logic" in summary
        assert "ui_actions" in summary
        
        routing_logic = summary["routing_logic"]
        assert routing_logic["uncertain == True"] == "needs_confirm"
        assert routing_logic["state == 'blocked'"] == "suggest_plan"
        assert routing_logic["default"] == "auto_save_ok"
        
        ui_actions = summary["ui_actions"]
        assert "needs_confirm" in ui_actions
        assert "suggest_plan" in ui_actions
        assert "auto_save_ok" in ui_actions
        
        # Verify UI action descriptions
        assert "confirm chips" in ui_actions["needs_confirm"]
        assert "Link to" in ui_actions["needs_confirm"]
        assert "/reason" in ui_actions["suggest_plan"]
        assert "Automatically save" in ui_actions["auto_save_ok"]


class TestIntegration:
    """Test integration between uncertainty and routing."""
    
    def test_uncertainty_to_routing_flow(self):
        """Test complete flow from uncertainty detection to routing."""
        # Test case 1: Uncertain classification
        cat_probs = [0.4, 0.3, 0.2]  # Low confidence
        margin = 0.2
        state_conf = 0.8
        
        uncertain = is_uncertain(cat_probs, margin, state_conf)
        assert uncertain is True
        
        route = determine_route(uncertain, "continue")
        assert route == "needs_confirm"
        
        # Test case 2: Confident but blocked
        cat_probs = [0.7, 0.3, 0.2]  # High confidence
        margin = 0.4
        state_conf = 0.8
        
        uncertain = is_uncertain(cat_probs, margin, state_conf)
        assert uncertain is False
        
        route = determine_route(uncertain, "blocked")
        assert route == "suggest_plan"
        
        # Test case 3: Confident and not blocked
        route = determine_route(uncertain, "continue")
        assert route == "auto_save_ok"
    
    def test_threshold_impact_on_routing(self):
        """Test how threshold changes affect routing decisions."""
        # Store original thresholds
        original_cat = TAU_CAT
        original_margin = MARGIN_CAT
        
        try:
            # Make thresholds more strict
            adjust_thresholds(cat_threshold=0.7, margin_threshold=0.25)
            
            # Test with values that would pass old thresholds but fail new ones
            cat_probs = [0.65, 0.4, 0.2]  # Would pass old TAU_CAT (0.55) but fail new (0.7)
            margin = 0.2  # Would pass old MARGIN_CAT (0.15) but fail new (0.25)
            state_conf = 0.8
            
            uncertain = is_uncertain(cat_probs, margin, state_conf)
            assert uncertain is True
            
            route = determine_route(uncertain, "continue")
            assert route == "needs_confirm"
            
        finally:
            # Restore original thresholds
            adjust_thresholds(cat_threshold=original_cat, margin_threshold=original_margin)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
