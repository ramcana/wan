"""
Core services for WAN2.2
"""

# Import key services to make them available
try:
    # Try fixed version first
    from .wan_pipeline_loader_fixed import WanPipelineLoader
except ImportError:
    try:
        # Fallback to original
        from .wan_pipeline_loader import WanPipelineLoader
    except ImportError:
        # Create a placeholder
        WanPipelineLoader = None

try:
    from .utils import LoRAManager
except ImportError:
    LoRAManager = None

try:
    from .model_manager import ModelManager
except ImportError:
    ModelManager = None

try:
    from .pipeline_manager import PipelineManager
except ImportError:
    PipelineManager = None

__all__ = [
    'WanPipelineLoader',
    'LoRAManager', 
    'ModelManager',
    'PipelineManager'
]