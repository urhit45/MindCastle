"""
Home router for TinyNet - surfaces actionable items and review tasks.
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from datetime import datetime, timedelta
from typing import List, Optional
import logging

from ..db import get_session
from ..models import ProgressLog, Todo, Node
from ..schemas import ProgressLog, Todo, Node

router = APIRouter(prefix="/home", tags=["home"])

logger = logging.getLogger(__name__)


@router.get("/review")
async def get_review_items(
    session: AsyncSession = Depends(get_session),
    limit: int = 20
):
    """
    Get items that need review:
    - Blocked progress logs
    - Items with next steps
    - Stale items (>7 days old)
    """
    try:
        # Get current time for stale calculations
        now = datetime.utcnow()
        stale_threshold = now - timedelta(days=7)
        
        # 1. Get blocked progress logs
        blocked_query = select(ProgressLog).where(
            ProgressLog.state == "blocked"
        ).order_by(desc(ProgressLog.created_at)).limit(limit // 3)
        
        blocked_result = await session.execute(blocked_query)
        blocked_logs = blocked_result.scalars().all()
        
        # 2. Get progress logs with next steps
        nextstep_query = select(ProgressLog).where(
            and_(
                ProgressLog.next_step.isnot(None),
                ProgressLog.next_step != "",
                ProgressLog.created_at >= stale_threshold  # Not too old
            )
        ).order_by(desc(ProgressLog.created_at)).limit(limit // 3)
        
        nextstep_result = await session.execute(nextstep_query)
        nextstep_logs = nextstep_result.scalars().all()
        
        # 3. Get stale progress logs (>7 days old)
        stale_query = select(ProgressLog).where(
            ProgressLog.created_at < stale_threshold
        ).order_by(ProgressLog.created_at).limit(limit // 3)
        
        stale_result = await session.execute(stale_query)
        stale_logs = stale_result.scalars().all()
        
        # 4. Get overdue todos
        overdue_query = select(Todo).where(
            and_(
                Todo.due_at < now,
                Todo.status.in_(["pending", "in_progress"])
            )
        ).order_by(Todo.due_at).limit(limit // 4)
        
        overdue_result = await session.execute(overdue_query)
        overdue_todos = overdue_result.scalars().all()
        
        # 5. Get todos due soon (next 3 days)
        soon_threshold = now + timedelta(days=3)
        soon_query = select(Todo).where(
            and_(
                Todo.due_at <= soon_threshold,
                Todo.due_at >= now,
                Todo.status.in_(["pending", "in_progress"])
            )
        ).order_by(Todo.due_at).limit(limit // 4)
        
        soon_result = await session.execute(soon_query)
        soon_todos = soon_result.scalars().all()
        
        # Format response items
        review_items = []
        
        # Add blocked items
        for log in blocked_logs:
            review_items.append({
                "id": str(log.id),
                "type": "progress_log",
                "reason": "blocked",
                "title": log.text[:100] + "..." if len(log.text) > 100 else log.text,
                "nodeId": str(log.node_id) if log.node_id else None,
                "created_at": log.created_at.isoformat(),
                "state": log.state,
                "next_step": log.next_step,
                "priority": "high"
            })
        
        # Add next step items
        for log in nextstep_logs:
            review_items.append({
                "id": str(log.id),
                "type": "progress_log",
                "reason": "next_step",
                "title": log.text[:100] + "..." if len(log.text) > 100 else log.text,
                "nodeId": str(log.node_id) if log.node_id else None,
                "created_at": log.created_at.isoformat(),
                "state": log.state,
                "next_step": log.next_step,
                "priority": "medium"
            })
        
        # Add stale items
        for log in stale_logs:
            days_old = (now - log.created_at).days
            review_items.append({
                "id": str(log.id),
                "type": "progress_log",
                "reason": "stale",
                "title": log.text[:100] + "..." if len(log.text) > 100 else log.text,
                "nodeId": str(log.node_id) if log.node_id else None,
                "created_at": log.created_at.isoformat(),
                "days_old": days_old,
                "state": log.state,
                "priority": "low"
            })
        
        # Add overdue todos
        for todo in overdue_todos:
            days_overdue = (now - todo.due_at).days
            review_items.append({
                "id": str(todo.id),
                "type": "todo",
                "reason": "overdue",
                "title": todo.title,
                "nodeId": str(todo.node_id) if todo.node_id else None,
                "due_at": todo.due_at.isoformat(),
                "days_overdue": days_overdue,
                "status": todo.status,
                "priority": "high"
            })
        
        # Add soon due todos
        for todo in soon_todos:
            days_until_due = (todo.due_at - now).days
            review_items.append({
                "id": str(todo.id),
                "type": "todo",
                "reason": "due_soon",
                "title": todo.title,
                "nodeId": str(todo.node_id) if todo.node_id else None,
                "due_at": todo.due_at.isoformat(),
                "days_until_due": days_until_due,
                "status": todo.status,
                "priority": "medium"
            })
        
        # Sort by priority and recency
        priority_order = {"high": 3, "medium": 2, "low": 1}
        review_items.sort(
            key=lambda x: (
                priority_order.get(x.get("priority", "low"), 0),
                x.get("created_at", x.get("due_at", ""))
            ),
            reverse=True
        )
        
        # Limit total items
        review_items = review_items[:limit]
        
        # Get summary stats
        stats = {
            "total_items": len(review_items),
            "blocked_count": len(blocked_logs),
            "next_step_count": len(nextstep_logs),
            "stale_count": len(stale_logs),
            "overdue_count": len(overdue_todos),
            "due_soon_count": len(soon_todos),
            "high_priority": len([item for item in review_items if item.get("priority") == "high"]),
            "medium_priority": len([item for item in review_items if item.get("priority") == "medium"]),
            "low_priority": len([item for item in review_items if item.get("priority") == "low"])
        }
        
        return {
            "ok": True,
            "items": review_items,
            "stats": stats,
            "generated_at": now.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting review items: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get review items: {str(e)}")


@router.get("/dashboard")
async def get_dashboard_summary(
    session: AsyncSession = Depends(get_session)
):
    """Get high-level dashboard summary."""
    try:
        now = datetime.utcnow()
        
        # Get counts
        total_nodes = await session.scalar(select(func.count(Node.id)))
        total_progress = await session.scalar(select(func.count(ProgressLog.id)))
        total_todos = await session.scalar(select(func.count(Todo.id)))
        
        # Get recent activity (last 7 days)
        week_ago = now - timedelta(days=7)
        recent_progress = await session.scalar(
            select(func.count(ProgressLog.id)).where(ProgressLog.created_at >= week_ago)
        )
        recent_todos = await session.scalar(
            select(func.count(Todo.id)).where(Todo.created_at >= week_ago)
        )
        
        # Get completion stats
        completed_todos = await session.scalar(
            select(func.count(Todo.id)).where(Todo.status == "completed")
        )
        completion_rate = (completed_todos / total_todos * 100) if total_todos > 0 else 0
        
        # Get state distribution
        state_counts = {}
        state_query = select(ProgressLog.state, func.count(ProgressLog.id)).group_by(ProgressLog.state)
        state_result = await session.execute(state_query)
        for state, count in state_result:
            state_counts[state] = count
        
        return {
            "ok": True,
            "summary": {
                "total_nodes": total_nodes,
                "total_progress_logs": total_progress,
                "total_todos": total_todos,
                "recent_activity": {
                    "progress_logs_7d": recent_progress,
                    "todos_7d": recent_todos
                },
                "completion": {
                    "completed_todos": completed_todos,
                    "completion_rate_percent": round(completion_rate, 1)
                },
                "state_distribution": state_counts
            },
            "generated_at": now.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard: {str(e)}")


@router.get("/quick-actions")
async def get_quick_actions():
    """Get suggested quick actions based on current state."""
    return {
        "ok": True,
        "quick_actions": [
            {
                "id": "add_progress",
                "title": "Add Progress Note",
                "description": "Log your current progress on any project",
                "action": "navigate_to_add_progress",
                "icon": "📝"
            },
            {
                "id": "create_todo",
                "title": "Create Todo",
                "description": "Add a new task to your list",
                "action": "navigate_to_create_todo",
                "icon": "✅"
            },
            {
                "id": "review_blocked",
                "title": "Review Blocked Items",
                "description": "Check what's holding you back",
                "action": "navigate_to_review",
                "icon": "🚧"
            },
            {
                "id": "plan_next_steps",
                "title": "Plan Next Steps",
                "description": "Define what to do next",
                "action": "navigate_to_planning",
                "icon": "🎯"
            }
        ]
    }
