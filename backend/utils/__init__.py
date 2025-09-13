"""
Utility modules for the backend
"""

import sys
import importlib.util
from pathlib import Path

from .thumbnail_generator import generate_video_thumbnail, delete_video_thumbnail

# Import ModelManager and get_system_stats from root utils.py using absolute path
def _import_from_root_utils():
    """Import components from root utils.py to avoid circular imports"""
    try:
        # Get absolute path to root utils.py
        root_utils_path = Path(__file__).parent.parent.parent / "utils.py"
        
        # Load the module directly from file path
        spec = importlib.util.spec_from_file_location("root_utils", root_utils_path)
        root_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(root_utils)
        
        return root_utils.ModelManager, root_utils.get_system_stats
    except Exception as e:
        # Fallback if root utils is not available
        return None, None

ModelManager, get_system_stats = _import_from_root_utils()

__all__ = ['generate_video_thumbnail', 'delete_video_thumbnail', 'ModelManager', 'get_system_stats']
