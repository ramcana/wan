"""
WebSocket module for real-time updates
"""

from .manager import connection_manager, get_connection_manager, ConnectionManager

__all__ = ["connection_manager", "get_connection_manager", "ConnectionManager"]
