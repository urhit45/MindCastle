"""
Nodes router — real SQLite reads/writes via SQLAlchemy async
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing import List, Optional
from pydantic import BaseModel
import uuid

from ..db import get_session
from ..models import Node, ProgressLog

router = APIRouter(prefix="/nodes", tags=["nodes"])

DEFAULT_USER_ID = 1  # single-user MVP; no auth layer yet


# ─── Request / Response models ───────────────────────────────────────────────

class NodeCreate(BaseModel):
    title: str
    is_hub: bool = False
    status: str = "continue"


class NodeUpdate(BaseModel):
    title: Optional[str] = None
    is_hub: Optional[bool] = None
    status: Optional[str] = None


class NodeResponse(BaseModel):
    id: str
    title: str
    hub: bool
    status: str
    created_at: str


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


class NodeLogCreate(BaseModel):
    text: str
    state: str = "continue"
    next_step: Optional[str] = None


class NodeLogItem(BaseModel):
    id: int
    time: str
    text: str
    nextStep: Optional[str] = None
    state: str


class NodeLogsResponse(BaseModel):
    items: List[NodeLogItem]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _node_response(node: Node) -> NodeResponse:
    return NodeResponse(
        id=str(node.id),
        title=node.title,
        hub=node.is_hub,
        status=node.status,
        created_at=node.created_at.isoformat(),
    )


def _log_item(log: ProgressLog) -> NodeLogItem:
    return NodeLogItem(
        id=log.id,
        time=log.created_at.isoformat(),
        text=log.text,
        nextStep=log.next_step,
        state=log.state,
    )


async def _get_node_or_404(node_id: str, session: AsyncSession) -> Node:
    try:
        uid = uuid.UUID(node_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid node ID")
    result = await session.execute(
        select(Node).where(Node.id == uid, Node.user_id == DEFAULT_USER_ID)
    )
    node = result.scalar_one_or_none()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/", response_model=NodeResponse, status_code=201)
async def create_node(
    request: NodeCreate,
    session: AsyncSession = Depends(get_session),
):
    node = Node(
        title=request.title,
        is_hub=request.is_hub,
        status=request.status,
        user_id=DEFAULT_USER_ID,
    )
    session.add(node)
    await session.commit()
    await session.refresh(node)
    return _node_response(node)


@router.get("/search", response_model=NodeSearchResponse)
async def search_nodes(
    q: str = Query(...),
    limit: int = Query(10, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Node)
        .where(Node.user_id == DEFAULT_USER_ID, Node.title.ilike(f"%{q}%"))
        .limit(limit)
    )
    nodes = result.scalars().all()
    return NodeSearchResponse(
        items=[NodeSearchItem(id=str(n.id), title=n.title, hub=n.is_hub, score=1.0) for n in nodes]
    )


@router.get("/{node_id}", response_model=NodeDetailResponse)
async def get_node(
    node_id: str,
    session: AsyncSession = Depends(get_session),
):
    try:
        uid = uuid.UUID(node_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid node ID")

    result = await session.execute(
        select(Node)
        .where(Node.id == uid, Node.user_id == DEFAULT_USER_ID)
        .options(selectinload(Node.progress_logs))
    )
    node = result.scalar_one_or_none()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    last_checkpoint = None
    if node.progress_logs:
        latest = max(node.progress_logs, key=lambda l: l.created_at)
        last_checkpoint = LastCheckpoint(text=latest.text, date=latest.created_at.isoformat())

    return NodeDetailResponse(
        id=str(node.id),
        title=node.title,
        hub=node.is_hub,
        status=node.status,
        lastCheckpoint=last_checkpoint,
        related=[],
    )


@router.patch("/{node_id}", response_model=NodeResponse)
async def update_node(
    node_id: str,
    request: NodeUpdate,
    session: AsyncSession = Depends(get_session),
):
    node = await _get_node_or_404(node_id, session)
    if request.title is not None:
        node.title = request.title
    if request.is_hub is not None:
        node.is_hub = request.is_hub
    if request.status is not None:
        node.status = request.status
    await session.commit()
    await session.refresh(node)
    return _node_response(node)


@router.delete("/{node_id}")
async def delete_node(
    node_id: str,
    session: AsyncSession = Depends(get_session),
):
    node = await _get_node_or_404(node_id, session)
    await session.delete(node)
    await session.commit()
    return {"ok": True}


@router.post("/{node_id}/logs", response_model=NodeLogItem, status_code=201)
async def add_log(
    node_id: str,
    request: NodeLogCreate,
    session: AsyncSession = Depends(get_session),
):
    node = await _get_node_or_404(node_id, session)
    log = ProgressLog(
        node_id=node.id,
        text=request.text,
        state=request.state,
        next_step=request.next_step,
    )
    session.add(log)
    node.status = request.state  # node status tracks latest log state
    await session.commit()
    await session.refresh(log)
    return _log_item(log)


@router.get("/{node_id}/logs", response_model=NodeLogsResponse)
async def get_node_logs(
    node_id: str,
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    try:
        uid = uuid.UUID(node_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid node ID")

    result = await session.execute(
        select(ProgressLog)
        .where(ProgressLog.node_id == uid)
        .order_by(ProgressLog.created_at.desc())
        .limit(limit)
    )
    logs = result.scalars().all()
    return NodeLogsResponse(items=[_log_item(l) for l in logs])
