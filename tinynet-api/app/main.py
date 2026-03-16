"""
TinyNet API — main application factory.
Middleware order (outermost → innermost):
  RequestID → BodySizeLimit → RateLimit → CORS → routers
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from sqlalchemy import select

from .db import init_db, close_db, AsyncSessionLocal
from .config import settings
from .routers import classify, nodes, home
from .models import User
from .ml.model_service import model_service
from .errors import register_exception_handlers
from .middleware import RequestIDMiddleware, BodySizeLimitMiddleware, RateLimitMiddleware

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="A chat-first mind web API",
    version="0.1.0",
    debug=settings.debug,
)

# ─── Exception handlers ───────────────────────────────────────────────────────
# Must be registered before middleware so the handlers fire within the ASGI stack.
register_exception_handlers(app)

# ─── Middleware (added in reverse order — last added = outermost) ─────────────
# Innermost: RateLimit (after CORS, before handler)
app.add_middleware(RateLimitMiddleware)
# Body size guard
app.add_middleware(BodySizeLimitMiddleware)
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_origin_regex=r"http://192\.168\.\d+\.\d+:5173",  # LAN phones/tablets
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Outermost: RequestID (generates ID before anything else runs)
app.add_middleware(RequestIDMiddleware)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(classify.router, tags=["classify"])
app.include_router(nodes.router, tags=["nodes"])
app.include_router(home.router, tags=["home"])


# ─── Startup / shutdown ───────────────────────────────────────────────────────

async def _ensure_default_user() -> None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.id == 1))
        if not result.scalar_one_or_none():
            session.add(User(email="local@mindcastle"))
            await session.commit()


@app.on_event("startup")
async def startup_event() -> None:
    await init_db()
    await _ensure_default_user()
    model_service.load()
    log.info("startup_complete service=%s", settings.app_name)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await close_db()
    log.info("shutdown_complete")


# ─── Core routes ──────────────────────────────────────────────────────────────

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
