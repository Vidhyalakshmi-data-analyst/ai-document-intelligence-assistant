"""
AI Document Intelligence Assistant
Main application entry point.
"""

from config.settings import settings


def main() -> None:
    """Application entry point."""
    print(f"Loaded configuration for {settings.app_name} [{settings.app_env}].")


if __name__ == "__main__":
    main()
