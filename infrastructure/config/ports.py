"""
Centralized port configuration for the WAN2.2 Video Generation System.
This module defines all port-related constants used throughout the application.
"""

import os
from typing import Optional

# Backend server port - where the FastAPI application runs
BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))

# Frontend development server port
FRONTEND_DEV_PORT: int = int(os.getenv("FRONTEND_DEV_PORT", "3000"))

# Gradio UI port (legacy interface)
GRADIO_PORT: int = int(os.getenv("GRADIO_PORT", "7860"))

# Database port (if using external database)
DATABASE_PORT_ENV = os.getenv("DATABASE_PORT")
DATABASE_PORT: Optional[int] = int(DATABASE_PORT_ENV) if DATABASE_PORT_ENV else None

# Redis port (for caching)
REDIS_PORT_ENV = os.getenv("REDIS_PORT")
REDIS_PORT: Optional[int] = int(REDIS_PORT_ENV) if REDIS_PORT_ENV else None

# API prefix
API_PREFIX: str = "/api/v1"

# WebSocket endpoint
WEBSOCKET_ENDPOINT: str = "/ws"


def get_backend_url() -> str:
    """Get the complete backend URL including scheme and port."""
    scheme = os.getenv("BACKEND_SCHEME", "http")
    host = os.getenv("BACKEND_HOST", "localhost")
    return f"{scheme}://{host}:{BACKEND_PORT}"


def get_frontend_url() -> str:
    """Get the complete frontend URL including scheme and port."""
    scheme = os.getenv("FRONTEND_SCHEME", "http")
    host = os.getenv("FRONTEND_HOST", "localhost")
    return f"{scheme}://{host}:{FRONTEND_DEV_PORT}"


def get_api_base_url() -> str:
    """Get the base URL for API endpoints."""
    return f"{get_backend_url()}{API_PREFIX}"


def is_development_mode() -> bool:
    """Check if the application is running in development mode."""
    return os.getenv("NODE_ENV", "development") == "development"


# Export all port configurations
__all__ = [
    "BACKEND_PORT",
    "FRONTEND_DEV_PORT",
    "GRADIO_PORT",
    "DATABASE_PORT",
    "REDIS_PORT",
    "API_PREFIX",
    "WEBSOCKET_ENDPOINT",
    "get_backend_url",
    "get_frontend_url",
    "get_api_base_url",
    "is_development_mode",
]
