"""
TinyNet API Main Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .db import init_db, close_db
from .config import settings

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="A chat-first mind web API",
    version="0.1.0",
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "docs": "/docs"
    }
