"""
Shared FastAPI dependencies.

Auth seam: get_current_user() returns the local user for MVP.
Drop-in replacement: swap this function for JWT / session extraction
without touching any route handler signature.
"""
from dataclasses import dataclass

from fastapi import Request


@dataclass(frozen=True)
class CurrentUser:
    user_id: int


def get_current_user() -> CurrentUser:
    """MVP: always the local single user. Replace with real auth later."""
    return CurrentUser(user_id=1)


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")
