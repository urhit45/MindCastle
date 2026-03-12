"""
Reasoning service for TinyNet - provides structured plans using rules.
Designed to be pluggable for external LLM integration later.
"""

import logging
from typing import Dict, List, Optional
from ..schemas import ReasonRequest, ReasonResponse, NextStepTemplate, LinkSuggestion

logger = logging.getLogger(__name__)


class RuleBasedReasoner:
    """Rule-based reasoning engine for generating structured plans."""
    
    def __init__(self):
        """Initialize the rule-based reasoner."""
        self.rules = self._build_rules()
        logger.info("RuleBasedReasoner initialized with rule set")
    
    def _build_rules(self) -> Dict:
        """Build the decision table for rule-based reasoning."""
        return {
            # Running + Blocked state rules
            ("Running", "blocked"): {
                "subtasks": [
                    "form check video",
                    "cadence drills 10m", 
                    "reduce distance to 1.5mi"
                ],
                "blockers": ["possible overstride"],
                "next_step": {
                    "template": "PracticeForDuration",
                    "slots": {"what": "cadence drills", "durationMin": 10}
                },
                "link_to": []
            },
            
            # Fitness + Blocked state rules
            ("Fitness", "blocked"): {
                "subtasks": [
                    "assess pain level 1-10",
                    "check form on last exercise",
                    "reduce weight by 20%"
                ],
                "blockers": ["form breakdown", "fatigue"],
                "next_step": {
                    "template": "ReviewNotes",
                    "slots": {"focus": "form and pain assessment"}
                },
                "link_to": []
            },
            
            # Music + Blocked state rules
            ("Music", "blocked"): {
                "subtasks": [
                    "record current piece",
                    "identify difficult passages",
                    "practice at 50% tempo"
                ],
                "blockers": ["technical difficulty", "performance anxiety"],
                "next_step": {
                    "template": "PracticeForDuration",
                    "slots": {"what": "slow tempo practice", "durationMin": 15}
                },
                "link_to": []
            },
            
            # Learning + Blocked state rules
            ("Learning", "blocked"): {
                "subtasks": [
                    "break down complex concept",
                    "find simpler explanation",
                    "take 5-minute break"
                ],
                "blockers": ["cognitive overload", "unclear concepts"],
                "next_step": {
                    "template": "OutlineThreeBullets",
                    "slots": {"topic": "key concepts to review"}
                },
                "link_to": []
            },
            
            # Admin + email rules
            ("Admin", "email"): {
                "subtasks": [
                    "draft response",
                    "review tone and content",
                    "schedule follow-up reminder"
                ],
                "blockers": ["procrastination", "uncertainty about response"],
                "next_step": {
                    "template": "SetReminder",
                    "slots": {"what": "send email", "when": "tomorrow 9am"}
                },
                "link_to": []
            }
        }
    
    def _find_matching_rules(self, categories: List[str], state: str, text: str) -> Optional[Dict]:
        """Find matching rules based on categories, state, and text content."""
        # Check for exact category + state matches
        for (cat, st), rule in self.rules.items():
            if cat in categories and st == state:
                return rule
        
        # Check for Admin + email pattern
        if "Admin" in categories and "email" in text.lower():
            return self.rules.get(("Admin", "email"))
        
        return None
    
    def _apply_context_adaptations(self, base_rule: Dict, context: Dict) -> Dict:
        """Adapt the base rule based on context information."""
        adapted_rule = base_rule.copy()
        
        # Adapt based on recent steps
        if context.get("last_steps"):
            last_steps = context["last_steps"]
            if "stretch" in " ".join(last_steps).lower():
                # If stretching was done, suggest different approach
                if "cadence drills" in str(adapted_rule.get("next_step", {}).get("slots", {})):
                    adapted_rule["next_step"]["slots"]["what"] = "form check and video analysis"
        
        # Adapt based on constraints
        constraints = context.get("constraints", {})
        if constraints.get("time_per_day_min"):
            time_min = constraints["time_per_day_min"]
            if time_min < 30:  # Short time available
                # Suggest shorter activities
                if "durationMin" in adapted_rule.get("next_step", {}).get("slots", {}):
                    adapted_rule["next_step"]["slots"]["durationMin"] = min(
                        adapted_rule["next_step"]["slots"]["durationMin"], 
                        time_min // 2
                    )
        
        # Adapt based on recent nodes
        if context.get("recent_nodes"):
            recent_nodes = context["recent_nodes"]
            for node in recent_nodes:
                if "rehab" in node.get("title", "").lower():
                    # Link to rehab node if available
                    adapted_rule["link_to"].append({
                        "nodeId": node["id"],
                        "reason": "rehab protocol"
                    })
        
        return adapted_rule
    
    def _generate_generic_plan(self, categories: List[str], state: str) -> Dict:
        """Generate a generic plan when no specific rules match."""
        if state == "start":
            return {
                "subtasks": [
                    "define clear goal",
                    "break into 3 sub-goals",
                    "set first milestone"
                ],
                "blockers": ["lack of clarity", "overwhelm"],
                "next_step": {
                    "template": "OutlineThreeBullets",
                    "slots": {"topic": "goal breakdown"}
                },
                "link_to": []
            }
        elif state == "continue":
            return {
                "subtasks": [
                    "review progress so far",
                    "identify next immediate step",
                    "check for obstacles"
                ],
                "blockers": ["momentum loss", "uncertainty"],
                "next_step": {
                    "template": "ReviewNotes",
                    "slots": {"focus": "progress and next steps"}
                },
                "link_to": []
            }
        else:
            return {
                "subtasks": [
                    "assess current situation",
                    "identify key challenges",
                    "plan next action"
                ],
                "blockers": ["analysis paralysis", "lack of direction"],
                "next_step": {
                    "template": "OutlineThreeBullets",
                    "slots": {"topic": "situation analysis"}
                },
                "link_to": []
            }
    
    def reason(self, request: ReasonRequest, use_llm: bool = False) -> ReasonResponse:
        """
        Generate reasoning and structured plan for the given request.
        
        Args:
            request: The reasoning request with text, prediction, and context
            use_llm: Whether to use external LLM (future feature, not implemented)
            
        Returns:
            Structured reasoning response with subtasks, blockers, next step, and links
        """
        if use_llm:
            logger.info("LLM reasoning requested but not implemented yet")
            # TODO: Implement external LLM integration
            # For now, fall back to rule-based reasoning
        
        text = request.text
        categories = request.pred.cats
        state = request.pred.state
        context = request.context or {}
        
        logger.info(f"Reasoning about text: '{text[:50]}...' with categories: {categories}, state: {state}")
        
        # Find matching rules
        matching_rule = self._find_matching_rules(categories, state, text)
        
        if matching_rule:
            # Apply context adaptations
            adapted_rule = self._apply_context_adaptations(matching_rule, context.model_dump())
            logger.info(f"Applied rule-based reasoning with adaptations")
        else:
            # Generate generic plan
            adapted_rule = self._generate_generic_plan(categories, state)
            logger.info(f"Generated generic plan for categories: {categories}, state: {state}")
        
        # Convert to response format
        next_step = NextStepTemplate(
            template=adapted_rule["next_step"]["template"],
            slots=adapted_rule["next_step"].get("slots", {})
        )
        
        link_suggestions = [
            LinkSuggestion(nodeId=link["nodeId"], reason=link["reason"])
            for link in adapted_rule.get("link_to", [])
        ]
        
        response = ReasonResponse(
            subtasks=adapted_rule["subtasks"],
            blockers=adapted_rule["blockers"],
            next_step=next_step,
            link_to=link_suggestions
        )
        
        logger.info(f"Generated plan with {len(response.subtasks)} subtasks and {len(response.blockers)} blockers")
        return response


# Global instance for easy access
_reasoner: Optional[RuleBasedReasoner] = None


def get_reasoner() -> RuleBasedReasoner:
    """Get or create the global reasoner instance."""
    global _reasoner
    if _reasoner is None:
        _reasoner = RuleBasedReasoner()
    return _reasoner


def reason(request: ReasonRequest, use_llm: bool = False) -> ReasonResponse:
    """
    Convenience function to reason about a request.
    
    Args:
        request: The reasoning request
        use_llm: Whether to use external LLM
        
    Returns:
        Structured reasoning response
    """
    reasoner = get_reasoner()
    return reasoner.reason(request, use_llm)
