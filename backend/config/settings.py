"""
TinyNet Configuration Settings
Loads configuration from environment variables and labels.yaml
"""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field
import yaml


class LabelsConfig(BaseSettings):
    """Configuration for labels loaded from labels.yaml"""
    categories: List[str] = Field(default_factory=list)
    states: List[str] = Field(default_factory=list)
    templates: List[str] = Field(default_factory=list)
    
    class Config:
        env_prefix = "TINYNET_"


class TinyNetSettings(BaseSettings):
    """Main TinyNet application settings"""
    # App settings
    app_name: str = "TinyNet"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Labels configuration
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    
    # Paths
    config_dir: Path = Field(default=Path(__file__).parent)
    labels_file: Path = Field(default=Path(__file__).parent / "labels.yaml")
    
    class Config:
        env_prefix = "TINYNET_"
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def load_labels(self) -> LabelsConfig:
        """Load labels from labels.yaml file"""
        if not self.labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_file}")
        
        with open(self.labels_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Map the YAML structure to our config
        self.labels.categories = data.get('categories', [])
        self.labels.states = data.get('states', [])
        self.labels.templates = data.get('next_step_templates', [])
        
        return self.labels


# Global settings instance
settings = TinyNetSettings()


def load_labels() -> LabelsConfig:
    """Load and return labels configuration"""
    return settings.load_labels()
