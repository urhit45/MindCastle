"""
Home router for TinyNet API
"""

from fastapi import APIRouter
from typing import List
from pydantic import BaseModel
import uuid

router = APIRouter(prefix="/home", tags=["home"])


# Response models
class ReviewItem(BaseModel):
    id: str
    title: str
    reason: str  # 'blocked', 'nextStep', 'stale'
    nodeId: str


class HomeReviewResponse(BaseModel):
    items: List[ReviewItem]


@router.get("/review", response_model=HomeReviewResponse)
async def get_home_review():
    """
    Get items that need review on the home page.
    
    This is a mock implementation that returns deterministic results.
    """
    # Mock review items
    mock_items = [
        {
            "id": str(uuid.uuid4()),
            "title": "Running Progress",
            "reason": "blocked",
            "nodeId": str(uuid.uuid4())
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Guitar Practice",
            "reason": "nextStep",
            "nodeId": str(uuid.uuid4())
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Learning Course",
            "reason": "stale",
            "nodeId": str(uuid.uuid4())
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Fitness Goals",
            "reason": "nextStep",
            "nodeId": str(uuid.uuid4())
        }
    ]
    
    return HomeReviewResponse(items=mock_items)
