"""
Nodes router for TinyNet API
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
import uuid
from datetime import datetime

router = APIRouter(prefix="/nodes", tags=["nodes"])


# Request/Response models
class NodeSearchItem(BaseModel):
    id: str
    title: str
    hub: bool
    score: float


class NodeSearchResponse(BaseModel):
    items: List[NodeSearchItem]


class LastCheckpoint(BaseModel):
    text: str
    date: str


class RelatedNode(BaseModel):
    id: str
    title: str


class NodeDetailResponse(BaseModel):
    id: str
    title: str
    hub: bool
    status: str
    lastCheckpoint: Optional[LastCheckpoint] = None
    related: List[RelatedNode]


class NodeLogItem(BaseModel):
    id: int
    time: str
    text: str
    nextStep: Optional[str] = None
    state: str


class NodeLogsResponse(BaseModel):
    items: List[NodeLogItem]


@router.get("/search", response_model=NodeSearchResponse)
async def search_nodes(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results")
):
    """
    Search for nodes by query.
    
    This is a mock implementation that returns deterministic results.
    """
    query = q.lower()
    
    # Mock search results
    mock_nodes = [
        {
            "id": str(uuid.uuid4()),
            "title": "Running Progress",
            "hub": False,
            "score": 0.92
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Fitness Goals",
            "hub": True,
            "score": 0.87
        },
        {
            "id": str(uuid.uuid4()),
            "title": "5K Training Plan",
            "hub": False,
            "score": 0.85
        }
    ]
    
    # Filter by query (simple mock filtering)
    if "run" in query:
        filtered_nodes = [node for node in mock_nodes if "run" in node["title"].lower()]
    elif "fitness" in query:
        filtered_nodes = [node for node in mock_nodes if "fitness" in node["title"].lower()]
    else:
        filtered_nodes = mock_nodes
    
    # Apply limit
    limited_nodes = filtered_nodes[:limit]
    
    return NodeSearchResponse(items=limited_nodes)


@router.get("/{node_id}", response_model=NodeDetailResponse)
async def get_node(node_id: str):
    """
    Get detailed information about a specific node.
    
    This is a mock implementation that returns deterministic results.
    """
    # Validate UUID format
    try:
        uuid.UUID(node_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid node ID format")
    
    # Mock node details
    mock_node = {
        "id": node_id,
        "title": "Running Progress",
        "hub": False,
        "status": "continue",
        "lastCheckpoint": LastCheckpoint(
            text="Ran 2 miles today, feeling good",
            date="2024-01-15T10:30:00Z"
        ),
        "related": [
            RelatedNode(
                id=str(uuid.uuid4()),
                title="Fitness Goals"
            ),
            RelatedNode(
                id=str(uuid.uuid4()),
                title="5K Training Plan"
            )
        ]
    }
    
    return NodeDetailResponse(**mock_node)


@router.get("/{node_id}/logs", response_model=NodeLogsResponse)
async def get_node_logs(
    node_id: str,
    limit: int = Query(50, ge=1, le=200, description="Maximum number of logs")
):
    """
    Get progress logs for a specific node.
    
    This is a mock implementation that returns deterministic results.
    """
    # Validate UUID format
    try:
        uuid.UUID(node_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid node ID format")
    
    # Mock log entries
    mock_logs = [
        {
            "id": 1,
            "time": "2024-01-15T10:30:00Z",
            "text": "Ran 2 miles today, feeling good",
            "nextStep": "PracticeForDuration",
            "state": "continue"
        },
        {
            "id": 2,
            "time": "2024-01-14T09:15:00Z",
            "text": "Shin pain started, need to rest",
            "nextStep": None,
            "state": "blocked"
        },
        {
            "id": 3,
            "time": "2024-01-13T08:45:00Z",
            "text": "Started new running routine",
            "nextStep": "SetReminder",
            "state": "start"
        },
        {
            "id": 4,
            "time": "2024-01-12T07:30:00Z",
            "text": "Completed 5K training plan",
            "nextStep": "ScheduleFollowUp",
            "state": "end"
        }
    ]
    
    # Apply limit
    limited_logs = mock_logs[:limit]
    
    return NodeLogsResponse(items=limited_logs)
