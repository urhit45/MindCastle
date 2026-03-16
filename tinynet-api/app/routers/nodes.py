"""
Nodes router — real SQLite reads/writes via SQLAlchemy async.
All writes are scoped to the requesting user via get_current_user().
"""
import logging
import uuid
from typing import Annotated, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..db import get_session
from ..dependencies import CurrentUser, get_current_user, get_request_id
from ..models import Node, ProgressLog

log = logging.getLogger(__name__)

router = APIRouter(prefix="/nodes", tags=["nodes"])

# ─── Shared type aliases ──────────────────────────────────────────────────────

ValidStatus = Literal[
    "start", "continue", "pause", "end", "idea",
    "active", "live", "concept", "blocked", "planned", "planning",
]

BoundedTitle = Annotated[str, Field(min_length=1, max_length=200)]
BoundedText  = Annotated[str, Field(min_length=1, max_length=2000)]

# ─── Request / Response models ────────────────────────────────────────────────

class NodeCreate(BaseModel):
    title: BoundedTitle
    is_hub: bool = False
    status: ValidStatus = "continue"

    @field_validator("title")
    @classmethod
    def strip_title(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("title must not be blank")
        return v


class NodeUpdate(BaseModel):
    title: Optional[BoundedTitle] = None
    is_hub: Optional[bool] = None
    status: Optional[ValidStatus] = None

    @field_validator("title")
    @classmethod
    def strip_title(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("title must not be blank")
        return v


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
    text: BoundedText
    state: ValidStatus = "continue"
    next_step: Optional[Annotated[str, Field(max_length=500)]] = None


class NodeLogItem(BaseModel):
    id: int
    time: str
    text: str
    nextStep: Optional[str] = None
    state: str


class NodeLogsResponse(BaseModel):
    items: List[NodeLogItem]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _node_response(node: Node) -> NodeResponse:
    return NodeResponse(
        id=str(node.id),
        title=node.title,
        hub=node.is_hub,
        status=node.status,
        created_at=node.created_at.isoformat(),
    )


def _log_item(log_obj: ProgressLog) -> NodeLogItem:
    return NodeLogItem(
        id=log_obj.id,
        time=log_obj.created_at.isoformat(),
        text=log_obj.text,
        nextStep=log_obj.next_step,
        state=log_obj.state,
    )


async def _get_node_or_404(
    node_id: str,
    session: AsyncSession,
    user_id: int,
) -> Node:
    try:
        uid = uuid.UUID(node_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid node ID format")
    result = await session.execute(
        select(Node).where(Node.id == uid, Node.user_id == user_id)
    )
    node = result.scalar_one_or_none()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/", response_model=NodeResponse, status_code=201)
async def create_node(
    request: NodeCreate,
    session: AsyncSession = Depends(get_session),
    current_user: CurrentUser = Depends(get_current_user),
    request_id: str = Depends(get_request_id),
):
    log.info("node_create request_id=%s user=%d", request_id, current_user.user_id)
    node = Node(
        title=request.title,
        is_hub=request.is_hub,
        status=request.status,
        user_id=current_user.user_id,
    )
    session.add(node)
    await session.commit()
    await session.refresh(node)
    return _node_response(node)


@router.get("/search", response_model=NodeSearchResponse)
async def search_nodes(
    q: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(10, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
    current_user: CurrentUser = Depends(get_current_user),
):
    result = await session.execute(
        select(Node)
        .where(Node.user_id == current_user.user_id, Node.title.ilike(f"%{q}%"))
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
    current_user: CurrentUser = Depends(get_current_user),
):
    try:
        uid = uuid.UUID(node_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid node ID format")

    result = await session.execute(
        select(Node)
        .where(Node.id == uid, Node.user_id == current_user.user_id)
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
    current_user: CurrentUser = Depends(get_current_user),
    request_id: str = Depends(get_request_id),
):
    node = await _get_node_or_404(node_id, session, current_user.user_id)
    log.info("node_update request_id=%s node=%s user=%d", request_id, node_id, current_user.user_id)
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
    current_user: CurrentUser = Depends(get_current_user),
    request_id: str = Depends(get_request_id),
):
    node = await _get_node_or_404(node_id, session, current_user.user_id)
    log.info("node_delete request_id=%s node=%s user=%d", request_id, node_id, current_user.user_id)
    await session.delete(node)
    await session.commit()
    return {"ok": True}


@router.post("/{node_id}/logs", response_model=NodeLogItem, status_code=201)
async def add_log(
    node_id: str,
    request: NodeLogCreate,
    session: AsyncSession = Depends(get_session),
    current_user: CurrentUser = Depends(get_current_user),
    request_id: str = Depends(get_request_id),
):
    node = await _get_node_or_404(node_id, session, current_user.user_id)
    log.info("log_add request_id=%s node=%s state=%s user=%d", request_id, node_id, request.state, current_user.user_id)
    entry = ProgressLog(
        node_id=node.id,
        text=request.text,
        state=request.state,
        next_step=request.next_step,
    )
    session.add(entry)
    node.status = request.state
    await session.commit()
    await session.refresh(entry)
    return _log_item(entry)


@router.get("/{node_id}/logs", response_model=NodeLogsResponse)
async def get_node_logs(
    node_id: str,
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
    current_user: CurrentUser = Depends(get_current_user),
):
    # Ownership check before reading logs
    node = await _get_node_or_404(node_id, session, current_user.user_id)
    result = await session.execute(
        select(ProgressLog)
        .where(ProgressLog.node_id == node.id)
        .order_by(ProgressLog.created_at.desc())
        .limit(limit)
    )
    logs = result.scalars().all()
    return NodeLogsResponse(items=[_log_item(l) for l in logs])
