"""
Graph router exposing learned relationships for mind map rendering.
"""

from typing import Dict, List, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..models import GraphEdge, Node, ProgressLog

router = APIRouter(prefix="/graph", tags=["graph"])
SignalKey = Literal["temporal_co_activation", "blocked_propagation", "progress_correlation"]


class GraphNode(BaseModel):
    id: str
    name: str
    val: float
    color: str
    icon: str
    status: str


class GraphLink(BaseModel):
    source: str
    target: str
    weight: float
    width: float
    dominant_signal: SignalKey
    signal_breakdown: Dict[str, float]


class MindMapResponse(BaseModel):
    ok: bool
    nodes: List[GraphNode]
    links: List[GraphLink]


def _node_color(status: str) -> str:
    palette = {
        "blocked": "#ff7d7d",
        "continue": "#31c4d3",
        "start": "#7b8cff",
        "end": "#5fd085",
        "pause": "#b5c1de",
        "idea": "#c9a6ff",
    }
    return palette.get(status, "#8a97b8")


def _node_icon(status: str) -> str:
    mapping = {
        "blocked": "🚧",
        "continue": "◈",
        "start": "✦",
        "end": "◎",
        "pause": "◇",
        "idea": "⟁",
    }
    return mapping.get(status, "◈")


def _dominant_signal(signal_breakdown: Dict[str, float]) -> SignalKey:
    if not signal_breakdown:
        return "temporal_co_activation"
    top = max(signal_breakdown.items(), key=lambda pair: pair[1])[0]
    if top in {"temporal_co_activation", "blocked_propagation", "progress_correlation"}:
        return top
    return "temporal_co_activation"


@router.get("/mind-map", response_model=MindMapResponse)
async def get_mind_map(
    min_weight: float = Query(0.05, ge=0.0, le=1.0),
    session: AsyncSession = Depends(get_session),
):
    nodes_result = await session.execute(select(Node))
    all_nodes = list(nodes_result.scalars().all())

    logs_result = await session.execute(select(ProgressLog).order_by(ProgressLog.created_at.desc()))
    logs = list(logs_result.scalars().all())
    latest_state_by_node: Dict[str, str] = {}
    log_count_by_node: Dict[str, int] = {}
    for log in logs:
        key = str(log.node_id)
        if key not in latest_state_by_node:
            latest_state_by_node[key] = log.state
        log_count_by_node[key] = log_count_by_node.get(key, 0) + 1

    node_payload = [
        GraphNode(
            id=str(node.id),
            name=node.title,
            val=max(1.0, float(log_count_by_node.get(str(node.id), 0)) * 1.3),
            color=_node_color(latest_state_by_node.get(str(node.id), node.status)),
            icon=_node_icon(latest_state_by_node.get(str(node.id), node.status)),
            status=latest_state_by_node.get(str(node.id), node.status),
        )
        for node in all_nodes
    ]

    edge_result = await session.execute(
        select(GraphEdge).where(GraphEdge.weight >= min_weight).order_by(GraphEdge.weight.desc()).limit(350)
    )
    edges = list(edge_result.scalars().all())
    links_payload = []
    for edge in edges:
        breakdown = edge.signal_breakdown or {}
        links_payload.append(
            GraphLink(
                source=edge.source_engine_id,
                target=edge.target_engine_id,
                weight=edge.weight,
                width=max(1.0, edge.weight * 7.0),
                dominant_signal=_dominant_signal(breakdown),
                signal_breakdown=breakdown,
            )
        )

    return MindMapResponse(ok=True, nodes=node_payload, links=links_payload)
