"""
UI Event Handlers Integration Module
Provides a unified interface for enhanced UI event handling
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_event_handlers(config: Optional[Dict[str, Any]] = None):
    """Get the appropriate event handlers based on availability"""
    try:
        # Try to import enhanced event handlers
        from ui_event_handlers_enhanced import get_enhanced_event_handlers
        handlers = get_enhanced_event_handlers(config)
        logger.info("Enhanced UI event handlers loaded successfully")
        return handlers
    except ImportError as e:
        logger.warning(f"Enhanced event handlers not available: {e}")
        # Fallback to basic handlers if needed
        return None
    except Exception as e:
        logger.error(f"Failed to load enhanced event handlers: {e}")
        return None

def setup_event_handlers(ui_components: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
    """Set up event handlers for UI components"""
    try:
        handlers = get_event_handlers(config)
        if handlers:
            handlers.register_components(ui_components)
            handlers.setup_all_event_handlers()
            logger.info("Event handlers set up successfully")
            return handlers
        else:
            logger.warning("No event handlers available - using fallback")
            return None
    except Exception as e:
        logger.error(f"Failed to set up event handlers: {e}")
        return None