"""
Unit tests for the reasoning service and router.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from app.services.reasoner import RuleBasedReasoner, reason, get_reasoner
from app.schemas import ReasonRequest, ReasonPrediction, ReasonContext, NextStepTemplate, LinkSuggestion, ReasonResponse


class TestRuleBasedReasoner:
    """Test the rule-based reasoning engine."""
    
    def test_initialization(self):
        """Test reasoner initialization and rule loading."""
        reasoner = RuleBasedReasoner()
        
        assert reasoner.rules is not None
        assert len(reasoner.rules) > 0
        
        # Check for specific rules
        assert ("Running", "blocked") in reasoner.rules
        assert ("Fitness", "blocked") in reasoner.rules
        assert ("Admin", "email") in reasoner.rules
    
    def test_running_blocked_rule(self):
        """Test the Running + blocked state rule."""
        reasoner = RuleBasedReasoner()
        
        request = ReasonRequest(
            text="shin splints after 2mi run",
            pred=ReasonPrediction(cats=["Fitness", "Running"], state="blocked"),
            context=ReasonContext()
        )
        
        response = reasoner.reason(request)
        
        assert "form check video" in response.subtasks
        assert "cadence drills 10m" in response.subtasks
        assert "reduce distance to 1.5mi" in response.subtasks
        assert "possible overstride" in response.blockers
        assert response.next_step.template == "PracticeForDuration"
        assert response.next_step.slots["what"] == "cadence drills"
        assert response.next_step.slots["durationMin"] == 10
    
    def test_fitness_blocked_rule(self):
        """Test the Fitness + blocked state rule."""
        reasoner = RuleBasedReasoner()
        
        request = ReasonRequest(
            text="can't complete last set",
            pred=ReasonPrediction(cats=["Fitness"], state="blocked"),
            context=ReasonContext()
        )
        
        response = reasoner.reason(request)
        
        assert "assess pain level 1-10" in response.subtasks
        assert "check form on last exercise" in response.subtasks
        assert "reduce weight by 20%" in response.subtasks
        assert "form breakdown" in response.blockers
        assert response.next_step.template == "ReviewNotes"
    
    def test_admin_email_rule(self):
        """Test the Admin + email pattern rule."""
        reasoner = RuleBasedReasoner()
        
        request = ReasonRequest(
            text="need to send important email",
            pred=ReasonPrediction(cats=["Admin"], state="continue"),
            context=ReasonContext()
        )
        
        response = reasoner.reason(request)
        
        assert "draft response" in response.subtasks
        assert "review tone and content" in response.subtasks
        assert "schedule follow-up reminder" in response.subtasks
        assert "procrastination" in response.blockers
        assert response.next_step.template == "SetReminder"
    
    def test_context_adaptations(self):
        """Test context-based rule adaptations."""
        reasoner = RuleBasedReasoner()
        
        # Test with rehab node context
        request = ReasonRequest(
            text="shin splints after 2mi run",
            pred=ReasonPrediction(cats=["Fitness", "Running"], state="blocked"),
            context=ReasonContext(
                recent_nodes=[{"id": "i7", "title": "Shin-splint rehab"}],
                last_steps=["2mi easy", "stretch calves"],
                constraints={"time_per_day_min": 20, "days_per_week": 3}
            )
        )
        
        response = reasoner.reason(request)
        
        # Should have link to rehab node
        assert len(response.link_to) > 0
        rehab_link = response.link_to[0]
        assert rehab_link.nodeId == "i7"
        assert rehab_link.reason == "rehab protocol"
        
        # Should adapt duration based on time constraints
        assert response.next_step.slots["durationMin"] <= 20
    
    def test_generic_plan_generation(self):
        """Test generic plan generation when no rules match."""
        reasoner = RuleBasedReasoner()
        
        # Test start state
        request = ReasonRequest(
            text="want to learn guitar",
            pred=ReasonPrediction(cats=["Music"], state="start"),
            context=ReasonContext()
        )
        
        response = reasoner.reason(request)
        
        assert "define clear goal" in response.subtasks
        assert "break into 3 sub-goals" in response.subtasks
        assert "set first milestone" in response.subtasks
        assert response.next_step.template == "OutlineThreeBullets"
        
        # Test continue state
        request = ReasonRequest(
            text="continuing guitar practice",
            pred=ReasonPrediction(cats=["Music"], state="continue"),
            context=ReasonContext()
        )
        
        response = reasoner.reason(request)
        
        assert "review progress so far" in response.subtasks
        assert "identify next immediate step" in response.subtasks
        assert response.next_step.template == "ReviewNotes"
    
    def test_rule_priority_and_fallback(self):
        """Test rule priority and fallback behavior."""
        reasoner = RuleBasedReasoner()
        
        # Test that specific rules take priority over generic ones
        request = ReasonRequest(
            text="shin splints after 2mi run",
            pred=ReasonPrediction(cats=["Fitness", "Running"], state="blocked"),
            context=ReasonContext()
        )
        
        response = reasoner.reason(request)
        
        # Should use Running + blocked rule, not generic
        assert "cadence drills" in response.next_step.slots["what"]
        assert response.next_step.template == "PracticeForDuration"
    
    def test_llm_flag_handling(self):
        """Test that LLM flag is handled gracefully (fallback to rules)."""
        reasoner = RuleBasedReasoner()
        
        request = ReasonRequest(
            text="shin splints after 2mi run",
            pred=ReasonPrediction(cats=["Fitness", "Running"], state="blocked"),
            context=ReasonContext()
        )
        
        # Should work the same regardless of LLM flag
        response_rule = reasoner.reason(request, use_llm=False)
        response_llm = reasoner.reason(request, use_llm=True)
        
        # Both should return the same result for now
        assert response_rule.subtasks == response_llm.subtasks
        assert response_rule.next_step.template == response_llm.next_step.template


class TestReasoningService:
    """Test the reasoning service convenience functions."""
    
    def test_get_reasoner_singleton(self):
        """Test that get_reasoner returns the same instance."""
        reasoner1 = get_reasoner()
        reasoner2 = get_reasoner()
        
        assert reasoner1 is reasoner2
        assert isinstance(reasoner1, RuleBasedReasoner)
    
    def test_reason_convenience_function(self):
        """Test the convenience reason function."""
        request = ReasonRequest(
            text="shin splints after 2mi run",
            pred=ReasonPrediction(cats=["Fitness", "Running"], state="blocked"),
            context=ReasonContext()
        )
        
        response = reason(request)
        
        assert isinstance(response, ReasonResponse)
        assert len(response.subtasks) > 0
        assert len(response.blockers) > 0
        assert response.next_step is not None


class TestReasoningRouter:
    """Test the reasoning router endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)
    
    def test_reason_endpoint_success(self, client):
        """Test successful reasoning request."""
        request_data = {
            "text": "shin splints after 2mi run",
            "pred": {
                "cats": ["Fitness", "Running"],
                "state": "blocked"
            },
            "context": {
                "recent_nodes": [{"id": "i7", "title": "Shin-splint rehab"}],
                "last_steps": ["2mi easy", "stretch calves"],
                "constraints": {"time_per_day_min": 20, "days_per_week": 3}
            }
        }
        
        response = client.post("/reason/", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "subtasks" in data
        assert "blockers" in data
        assert "next_step" in data
        assert "link_to" in data
        
        assert len(data["subtasks"]) > 0
        assert len(data["blockers"]) > 0
        assert data["next_step"]["template"] == "PracticeForDuration"
        assert len(data["link_to"]) > 0
    
    def test_reason_endpoint_minimal_request(self, client):
        """Test reasoning with minimal request data."""
        request_data = {
            "text": "need to learn something new",
            "pred": {
                "cats": ["Learning"],
                "state": "start"
            }
        }
        
        response = client.post("/reason/", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "subtasks" in data
        assert "next_step" in data
        assert data["next_step"]["template"] == "OutlineThreeBullets"
    
    def test_reason_endpoint_invalid_request(self, client):
        """Test reasoning with invalid request data."""
        request_data = {
            "text": "test",
            # Missing pred field
        }
        
        response = client.post("/reason/", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_health_check(self, client):
        """Test reasoning service health check."""
        response = client.get("/reason/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "reasoning"
    
    def test_rules_summary(self, client):
        """Test rules summary endpoint."""
        response = client.get("/reason/rules")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "rule_count" in data
        assert "rule_categories" in data
        assert "available_templates" in data
        
        assert data["rule_count"] > 0
        assert len(data["rule_categories"]) > 0
        assert len(data["available_templates"]) > 0
        
        # Check for specific templates
        assert "PracticeForDuration" in data["available_templates"]
        assert "ReviewNotes" in data["available_templates"]


class TestIntegration:
    """Test integration between components."""
    
    def test_end_to_end_reasoning_flow(self):
        """Test complete reasoning flow from request to response."""
        # Create request
        request = ReasonRequest(
            text="shin splints after 2mi run",
            pred=ReasonPrediction(cats=["Fitness", "Running"], state="blocked"),
            context=ReasonContext(
                recent_nodes=[{"id": "i7", "title": "Shin-splint rehab"}],
                last_steps=["2mi easy", "stretch calves"],
                constraints={"time_per_day_min": 20, "days_per_week": 3}
            )
        )
        
        # Get reasoning
        response = reason(request)
        
        # Validate response structure
        assert isinstance(response, ReasonResponse)
        assert isinstance(response.next_step, NextStepTemplate)
        assert all(isinstance(link, LinkSuggestion) for link in response.link_to)
        
        # Validate content
        assert len(response.subtasks) >= 3
        assert len(response.blockers) >= 1
        assert response.next_step.template in ["PracticeForDuration", "ReviewNotes", "OutlineThreeBullets"]
        
        # Check context adaptation
        if response.link_to:
            rehab_link = next((link for link in response.link_to if link.nodeId == "i7"), None)
            if rehab_link:
                assert rehab_link.reason == "rehab protocol"
    
    def test_rule_consistency(self):
        """Test that rules are consistent across multiple calls."""
        reasoner = RuleBasedReasoner()
        
        request = ReasonRequest(
            text="shin splints after 2mi run",
            pred=ReasonPrediction(cats=["Fitness", "Running"], state="blocked"),
            context=ReasonContext()
        )
        
        # Multiple calls should return consistent results
        response1 = reasoner.reason(request)
        response2 = reasoner.reason(request)
        
        assert response1.subtasks == response2.subtasks
        assert response1.blockers == response2.blockers
        assert response1.next_step.template == response2.next_step.template


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
