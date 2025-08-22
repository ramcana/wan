"""
System integration module for FastAPI backend
Integrates with existing Wan2.2 system components
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json

# Add parent directory to path to import existing modules
sys.path.append(str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

class SystemIntegration:
    """Manages integration with existing Wan2.2 system components"""
    
    def __init__(self):
        self.model_manager = None
        self.config = None
        self.system_stats = None
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize system integration with existing components"""
        try:
            # Load configuration
            self.config = self._load_config()
            
            # Initialize model manager
            self.model_manager = self._initialize_model_manager()
            
            # Initialize system monitoring
            self.system_stats = self._initialize_system_stats()
            
            self._initialized = True
            logger.info("System integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system integration: {e}")
            return False
    
    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration from config.json"""
        try:
            config_path = Path(__file__).parent.parent.parent / "config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.warning(f"Could not load config.json: {e}")
            # Return default configuration
            return {
                "directories": {
                    "models_directory": "models",
                    "outputs_directory": "outputs",
                    "loras_directory": "loras"
                },
                "optimization": {
                    "max_vram_usage_gb": 12,
                    "default_quantization": "bf16"
                }
            }
    
    def _initialize_model_manager(self):
        """Initialize model manager from existing utils.py"""
        try:
            from core.services.model_manager import ModelManager
            manager = ModelManager()
            logger.info("Model manager initialized")
            return manager
        except ImportError as e:
            logger.warning(f"Could not import ModelManager: {e}")
            return None
        except Exception as e:
            logger.warning(f"Could not initialize ModelManager: {e}")
            return None
    
    def _initialize_system_stats(self):
        """Initialize system stats monitoring"""
        try:
            from core.services.utils import get_system_stats
            logger.info("System stats monitoring initialized")
            return get_system_stats
        except ImportError as e:
            logger.warning(f"Could not import system stats: {e}")
            return None
        except Exception as e:
            logger.warning(f"Could not initialize system stats: {e}")
            return None
    
    async def validate_gpu_access(self) -> Tuple[bool, str]:
        """Validate GPU access and compatibility"""
        try:
            import torch
            
            # Check if CUDA is available
            if not torch.cuda.is_available():
                return False, "CUDA is not available"
            
            # Get GPU information
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                return False, "No CUDA devices found"
            
            # Get current GPU info
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            
            # Check VRAM
            total_memory = torch.cuda.get_device_properties(current_device).total_memory
            total_memory_gb = total_memory / (1024**3)
            
            # Basic compatibility check for RTX 4080
            if "RTX 4080" in gpu_name or "4080" in gpu_name:
                logger.info(f"RTX 4080 detected: {gpu_name}")
                
                # Check minimum VRAM (RTX 4080 should have 16GB)
                if total_memory_gb < 12:
                    return False, f"Insufficient VRAM: {total_memory_gb:.1f}GB (minimum 12GB required)"
                
                return True, f"RTX 4080 compatible: {gpu_name} with {total_memory_gb:.1f}GB VRAM"
            
            # General GPU compatibility check
            if total_memory_gb < 8:
                return False, f"Insufficient VRAM: {total_memory_gb:.1f}GB (minimum 8GB required)"
            
            return True, f"GPU compatible: {gpu_name} with {total_memory_gb:.1f}GB VRAM"
            
        except ImportError:
            return False, "PyTorch not available"
        except Exception as e:
            return False, f"GPU validation failed: {str(e)}"
    
    async def validate_model_loading(self) -> Tuple[bool, str]:
        """Validate that model loading system is functional"""
        try:
            if not self.model_manager:
                return False, "Model manager not available"
            
            # Test model manager functionality
            # Check if model directories exist
            models_dir = Path(self.config["directories"]["models_directory"])
            if not models_dir.exists():
                models_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created models directory: {models_dir}")
            
            # Test model ID mapping
            test_model_types = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
            for model_type in test_model_types:
                try:
                    model_id = self.model_manager.get_model_id(model_type)
                    if model_id:
                        logger.debug(f"Model mapping validated: {model_type} -> {model_id}")
                except Exception as e:
                    logger.warning(f"Model mapping issue for {model_type}: {e}")
            
            return True, "Model loading system validated successfully"
            
        except Exception as e:
            return False, f"Model loading validation failed: {str(e)}"
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            "initialized": self._initialized,
            "model_manager_available": self.model_manager is not None,
            "system_stats_available": self.system_stats is not None,
            "config_loaded": self.config is not None
        }
        
        if self.config:
            info["directories"] = self.config.get("directories", {})
            info["optimization"] = self.config.get("optimization", {})
        
        return info
    
    async def get_enhanced_system_stats(self) -> Optional[Dict[str, Any]]:
        """Get enhanced system statistics using existing monitoring"""
        try:
            if self.system_stats:
                # Use existing system stats function
                stats = self.system_stats()
                # Convert to dictionary if it's a custom object
                if hasattr(stats, '__dict__'):
                    stats_dict = stats.__dict__.copy()
                    # Convert datetime to string for JSON serialization
                    if 'timestamp' in stats_dict and hasattr(stats_dict['timestamp'], 'isoformat'):
                        stats_dict['timestamp'] = stats_dict['timestamp'].isoformat()
                    return stats_dict
                elif hasattr(stats, '_asdict'):
                    stats_dict = stats._asdict()
                    # Convert datetime to string for JSON serialization
                    if 'timestamp' in stats_dict and hasattr(stats_dict['timestamp'], 'isoformat'):
                        stats_dict['timestamp'] = stats_dict['timestamp'].isoformat()
                    return stats_dict
                else:
                    return stats
            else:
                # Fallback to basic stats
                import psutil
                
                # Get basic system info
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                from datetime import datetime
                stats = {
                    "cpu_percent": cpu_percent,
                    "ram_used_gb": memory.used / (1024**3),
                    "ram_total_gb": memory.total / (1024**3),
                    "ram_percent": memory.percent,
                    "gpu_percent": 0.0,
                    "vram_used_mb": 0.0,
                    "vram_total_mb": 0.0,
                    "vram_percent": 0.0,
                    "timestamp": datetime.now()
                }
                
                # Try to get GPU stats
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = torch.cuda.current_device()
                        total_memory = torch.cuda.get_device_properties(device).total_memory
                        allocated_memory = torch.cuda.memory_allocated(device)
                        
                        stats.update({
                            "gpu_available": True,
                            "vram_used_mb": allocated_memory / (1024**2),
                            "vram_total_mb": total_memory / (1024**2),
                            "vram_percent": (allocated_memory / total_memory) * 100
                        })
                    else:
                        stats.update({
                            "gpu_available": False,
                            "vram_used_mb": 0,
                            "vram_total_mb": 0,
                            "vram_percent": 0
                        })
                except Exception as gpu_error:
                    logger.warning(f"Could not get GPU stats: {gpu_error}")
                    stats.update({
                        "gpu_available": False,
                        "gpu_error": str(gpu_error)
                    })
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return None
    
    async def get_prompt_enhancer(self):
        """Get the prompt enhancer from the existing system"""
        try:
            # Import the prompt enhancement functions from utils.py in parent directory
            import sys
            from pathlib import Path
            
            # Add parent directory to path if not already there
            parent_dir = str(Path(__file__).parent.parent.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            # Import from the root utils.py file
            from core.services import utils
            return utils.get_prompt_enhancer()
        except ImportError as e:
            logger.error(f"Could not import prompt enhancer: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting prompt enhancer: {e}")
            return None

# Global system integration instance
system_integration = SystemIntegration()

async def get_system_integration() -> SystemIntegration:
    """Dependency to get system integration instance"""
    if not system_integration._initialized:
        await system_integration.initialize()
    return system_integration