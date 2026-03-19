"""
TinyNet API Configuration
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite+aiosqlite:///./tinynet.db"

    # App
    app_name: str = "TinyNet API"
    debug: bool = False

    # CORS — override via TINYNET_ALLOWED_ORIGINS env var (JSON list)
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
    ]

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_prefix = "TINYNET_"


settings = Settings()
