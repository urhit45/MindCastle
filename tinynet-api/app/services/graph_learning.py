"""
Incremental graph-edge learning from progress logs.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, List

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import GraphEdge, ProgressLog

SIGNAL_KEYS = (
    "temporal_co_activation",
    "blocked_propagation",
    "progress_correlation",
)

RECENT_WINDOW = timedelta(hours=2)
BLOCKED_WINDOW = timedelta(hours=24)


def _blank_breakdown() -> Dict[str, float]:
    return {k: 0.0 for k in SIGNAL_KEYS}


def _dominant_signal_delta(state: str, current_next_step: str | None, previous_state: str) -> Dict[str, float]:
    temporal = 0.34
    blocked = 0.0
    progress = 0.0

    if state == "blocked" or previous_state == "blocked":
        blocked = 0.68

    if state in {"continue", "end"} and previous_state in {"continue", "end"}:
        progress = 0.42

    if current_next_step:
        progress += 0.06

    return {
        "temporal_co_activation": temporal,
        "blocked_propagation": blocked,
        "progress_correlation": progress,
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _running_average_breakdown(
    existing: Dict[str, float],
    incoming: Dict[str, float],
    sample_count: int,
) -> Dict[str, float]:
    merged = dict(existing or {})
    next_count = max(1, sample_count + 1)
    for key in SIGNAL_KEYS:
        previous_avg = _clamp01(float(merged.get(key, 0.0)))
        incoming_value = _clamp01(float(incoming.get(key, 0.0)))
        merged[key] = round(((previous_avg * sample_count) + incoming_value) / next_count, 4)
    return merged


def _edge_weight(breakdown: Dict[str, float], sample_count: int) -> float:
    temporal = _clamp01(float(breakdown.get("temporal_co_activation", 0.0)))
    blocked = _clamp01(float(breakdown.get("blocked_propagation", 0.0)))
    progress = _clamp01(float(breakdown.get("progress_correlation", 0.0)))
    weighted_mean = (temporal * 0.35) + (blocked * 0.4) + (progress * 0.25)
    confidence = 1.0 - pow(2.718281828, -max(1, sample_count) / 8.0)
    return round(_clamp01(weighted_mean * confidence), 4)


async def _fetch_recent_logs(
    session: AsyncSession,
    *,
    node_id: str,
    created_at: datetime,
) -> List[ProgressLog]:
    result = await session.execute(
        select(ProgressLog)
        .where(
            and_(
                ProgressLog.node_id != node_id,
                ProgressLog.created_at >= created_at - RECENT_WINDOW,
                ProgressLog.created_at <= created_at,
            )
        )
        .order_by(ProgressLog.created_at.desc())
        .limit(20)
    )
    return list(result.scalars().all())


async def _fetch_blocked_logs(
    session: AsyncSession,
    *,
    node_id: str,
    created_at: datetime,
) -> List[ProgressLog]:
    result = await session.execute(
        select(ProgressLog)
        .where(
            and_(
                ProgressLog.node_id != node_id,
                ProgressLog.state == "blocked",
                ProgressLog.created_at >= created_at - BLOCKED_WINDOW,
                ProgressLog.created_at <= created_at,
            )
        )
        .order_by(ProgressLog.created_at.desc())
        .limit(20)
    )
    return list(result.scalars().all())


async def _get_or_create_edge(
    session: AsyncSession,
    *,
    source_engine_id: str,
    target_engine_id: str,
) -> GraphEdge:
    result = await session.execute(
        select(GraphEdge).where(
            GraphEdge.source_engine_id == source_engine_id,
            GraphEdge.target_engine_id == target_engine_id,
        )
    )
    edge = result.scalar_one_or_none()
    if edge:
        return edge

    edge = GraphEdge(
        source_engine_id=source_engine_id,
        target_engine_id=target_engine_id,
        weight=0.0,
        sample_count=0,
        signal_breakdown=_blank_breakdown(),
    )
    session.add(edge)
    return edge


async def _apply_signal(
    session: AsyncSession,
    *,
    source_engine_id: str,
    target_engine_id: str,
    signal_delta: Dict[str, float],
) -> None:
    edge = await _get_or_create_edge(
        session,
        source_engine_id=source_engine_id,
        target_engine_id=target_engine_id,
    )
    current_sample_count = int(edge.sample_count or 0)
    current_breakdown = edge.signal_breakdown or _blank_breakdown()
    merged = _running_average_breakdown(current_breakdown, signal_delta, current_sample_count)
    edge.signal_breakdown = merged
    edge.sample_count = current_sample_count + 1
    edge.weight = _edge_weight(merged, edge.sample_count)


async def update_graph_edges_for_log(
    session: AsyncSession,
    *,
    node_id: str,
    state: str,
    next_step: str | None,
    created_at: datetime | None,
) -> None:
    """
    Incrementally updates graph edge weights for a new log.
    Uses node ids as engine proxies until explicit engine ids are persisted in backend.
    """

    anchor_time = created_at or datetime.utcnow()
    recent_logs = await _fetch_recent_logs(session, node_id=node_id, created_at=anchor_time)
    for prev in recent_logs:
        delta = _dominant_signal_delta(state, next_step, prev.state)
        await _apply_signal(
            session,
            source_engine_id=str(prev.node_id),
            target_engine_id=node_id,
            signal_delta=delta,
        )

    if state == "blocked":
        blocked_sources = await _fetch_blocked_logs(session, node_id=node_id, created_at=anchor_time)
        for prev in blocked_sources:
            await _apply_signal(
                session,
                source_engine_id=str(prev.node_id),
                target_engine_id=node_id,
                signal_delta={
                    "temporal_co_activation": 0.16,
                    "blocked_propagation": 0.85,
                    "progress_correlation": 0.04,
                },
            )
