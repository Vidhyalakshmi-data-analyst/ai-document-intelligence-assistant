"""
Application configuration management.
Loads configuration settings from environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if it exists
load_dotenv(dotenv_path=BASE_DIR / ".env")


class Settings:
    """Application configuration container."""

    def __init__(self) -> None:
        self.app_name: str = os.getenv(
            "APP_NAME", "AI Document Intelligence Assistant"
        )
        self.app_env: str = os.getenv("APP_ENV", "development")
        self.debug: bool = (
            os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
        )
        self.gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")


def get_settings() -> Settings:
    """Factory function to retrieve application settings."""
    return Settings()


settings: Settings = get_settings()
