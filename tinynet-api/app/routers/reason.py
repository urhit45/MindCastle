"""
Reasoning router for TinyNet - provides structured planning and reasoning.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from ..schemas import ReasonRequest, ReasonResponse
from ..services.reasoner import reason

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reason", tags=["reasoning"])


@router.post("/", response_model=ReasonResponse)
async def reason_endpoint(request: ReasonRequest):
    """
    Generate structured reasoning and planning for the given request.
    
    Args:
        request: The reasoning request with text, prediction, and context
        
    Returns:
        Structured reasoning response with subtasks, blockers, next step, and links
    """
    try:
        logger.info(f"Reasoning request received for text: '{request.text[:50]}...'")
        
        # Use the reasoning service
        response = reason(request, use_llm=False)  # Rule-based for now
        
        logger.info(f"Generated reasoning response with {len(response.subtasks)} subtasks")
        return response
        
    except Exception as e:
        logger.error(f"Error in reasoning endpoint: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Reasoning failed: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check for the reasoning service."""
    return {"status": "healthy", "service": "reasoning"}


@router.get("/rules")
async def get_rules_summary():
    """Get a summary of available reasoning rules."""
    from ..services.reasoner import get_reasoner
    
    try:
        reasoner = get_reasoner()
        # Extract rule categories for display
        rule_categories = []
        for (cat, state) in reasoner.rules.keys():
            rule_categories.append({
                "category": cat,
                "state": state,
                "description": f"Specialized rules for {cat} + {state}"
            })
        
        return {
            "rule_count": len(reasoner.rules),
            "rule_categories": rule_categories,
            "available_templates": [
                "PracticeForDuration",
                "ReviewNotes", 
                "OutlineThreeBullets",
                "SetReminder"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting rules summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get rules summary: {str(e)}"
        )
