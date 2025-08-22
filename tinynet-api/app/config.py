"""
TinyNet API Configuration
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./tinynet.db"
    
    # App settings
    app_name: str = "TinyNet API"
    debug: bool = False
    
    class Config:
        env_file = ".env"
        env_prefix = "TINYNET_"


# Global settings instance
settings = Settings()
