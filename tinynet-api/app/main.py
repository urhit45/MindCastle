"""
TinyNet API Main Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from sqlalchemy import select

from .db import init_db, close_db, AsyncSessionLocal
from .config import settings
from .routers import classify, nodes, home
from .models import User
from .ml.model_service import model_service

app = FastAPI(
    title=settings.app_name,
    description="A chat-first mind web API",
    version="0.1.0",
    debug=settings.debug,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
    ],
    allow_origin_regex=r"http://192\.168\.\d+\.\d+:5173",  # LAN phones/tablets
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classify.router, tags=["classify"])
app.include_router(nodes.router, tags=["nodes"])
app.include_router(home.router, tags=["home"])


async def _ensure_default_user():
    """Create user id=1 (local@mindcastle) if it doesn't exist yet."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.id == 1))
        if not result.scalar_one_or_none():
            session.add(User(email="local@mindcastle"))
            await session.commit()


@app.on_event("startup")
async def startup_event():
    await init_db()
    await _ensure_default_user()
    model_service.load()  # train-on-first-run or load from models/best.pt


@app.on_event("shutdown")
async def shutdown_event():
    await close_db()


@app.get("/healthz")
async def health_check():
    return {"status": "healthy", "service": "tinynet-api"}


@app.get("/")
async def root():
    return {
        "message": "Welcome to TinyNet API",
        "version": "0.1.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@app.get("/openapi.json")
async def get_openapi_spec():
    if app.openapi_schema:
        return app.openapi_schema
    app.openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    return app.openapi_schema
