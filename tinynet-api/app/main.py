"""
TinyNet API Main Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from .db import init_db, close_db
from .config import settings
from .routers import classify, nodes, home

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="A chat-first mind web API",
    version="0.1.0",
    debug=settings.debug
)

# Add CORS middleware - allow frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:5174",  # Alternative Vite port
        "http://localhost:3000",  # Alternative dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(classify.router, tags=["classify"])
app.include_router(nodes.router, tags=["nodes"])
app.include_router(home.router, tags=["home"])


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    await init_db()


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connections on shutdown"""
    await close_db()


@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "tinynet-api"}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to TinyNet API",
        "version": "0.1.0",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


@app.get("/openapi.json")
async def get_openapi_spec():
    """Get OpenAPI specification"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema
