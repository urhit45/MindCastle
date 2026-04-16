"""
User preferences — optional sync for tinynet-ui (theme, future keys).
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..dependencies import get_current_user, CurrentUser
from ..models import User

router = APIRouter(prefix="/users/me", tags=["preferences"])

ALLOWED_THEMES = frozenset({"tsushima", "transylvania", "frieren", "lofi"})


class PreferencesRead(BaseModel):
    theme: Optional[str] = None


class PreferencesPatch(BaseModel):
    theme: Optional[str] = None

    @field_validator("theme")
    @classmethod
    def theme_must_be_allowed(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if v not in ALLOWED_THEMES:
            raise ValueError(
                "theme must be one of: tsushima, transylvania, frieren, lofi",
            )
        return v


def _normalize_prefs(raw: Any) -> Dict[str, Any]:
    if raw is None or not isinstance(raw, dict):
        return {}
    return dict(raw)


@router.get("/preferences", response_model=PreferencesRead)
async def get_preferences(
    session: AsyncSession = Depends(get_session),
    user: CurrentUser = Depends(get_current_user),
) -> PreferencesRead:
    result = await session.execute(select(User).where(User.id == user.user_id))
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    prefs = _normalize_prefs(row.preferences)
    theme = prefs.get("theme")
    if theme is not None and theme not in ALLOWED_THEMES:
        theme = None
    return PreferencesRead(theme=theme)


@router.patch("/preferences", response_model=PreferencesRead)
async def patch_preferences(
    body: PreferencesPatch,
    session: AsyncSession = Depends(get_session),
    user: CurrentUser = Depends(get_current_user),
) -> PreferencesRead:
    result = await session.execute(select(User).where(User.id == user.user_id))
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    prefs = _normalize_prefs(row.preferences)
    if body.theme is not None:
        prefs["theme"] = body.theme
    row.preferences = prefs
    await session.commit()
    await session.refresh(row)
    out = _normalize_prefs(row.preferences).get("theme")
    if out is not None and out not in ALLOWED_THEMES:
        out = None
    return PreferencesRead(theme=out)
