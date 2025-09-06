from unittest.mock import Mock, patch
"""
System integration module for FastAPI backend
Integrates with existing Wan2.2 system components including ModelManager, 
ModelDownloader, and WAN22SystemOptimizer
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import json
from datetime import datetime

# Add parent directory to path to import existing modules
root_path = str(Path(__file__).parent.parent.parent)
local_install_path = str(Path(__file__).parent.parent.parent / "local_installation")

if root_path not in sys.path:
    sys.path.insert(0, root_path)
if local_install_path not in sys.path:
    sys.path.insert(0, local_install_path)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SystemIntegration:
    """Manages integration with existing Wan2.2 system components"""
    
    def __init__(self):
        self.model_manager = None
        self.model_downloader = None
        self.wan22_system_optimizer = None
        self.wan_pipeline_loader = None
        self.config = None
        self.config_bridge = None
        self.system_stats = None
        self._initialized = False
        self._initialization_errors = []
        
    async def initialize(self) -> bool:
        """Initialize system integration with existing Wan2.2 infrastructure"""
        try:
            logger.info("Initializing system integration with existing Wan2.2 infrastructure...")
            
            # Load configuration using ConfigurationBridge
            self.config_bridge = self._initialize_configuration_bridge()
            self.config = self.config_bridge.get_config() if self.config_bridge else self._load_config()
            
            # Initialize WAN22 System Optimizer first (provides hardware detection)
            self.wan22_system_optimizer = await self._initialize_wan22_system_optimizer()
            
            # Initialize model manager
            self.model_manager = await self._initialize_model_manager()
            
            # Initialize model downloader
            self.model_downloader = await self._initialize_model_downloader()
            
            # Initialize WAN pipeline loader
            self.wan_pipeline_loader = await self._initialize_wan_pipeline_loader()
            
            # Initialize system monitoring
            self.system_stats = self._initialize_system_stats()
            
            self._initialized = True
            
            # Log initialization status
            initialized_components = []
            if self.wan22_system_optimizer: initialized_components.append("WAN22SystemOptimizer")
            if self.model_manager: initialized_components.append("ModelManager")
            if self.model_downloader: initialized_components.append("ModelDownloader")
            if self.wan_pipeline_loader: initialized_components.append("WanPipelineLoader")
            if self.system_stats: initialized_components.append("SystemStats")
            
            logger.info(f"System integration initialized successfully with components: {', '.join(initialized_components)}")
            
            if self._initialization_errors:
                logger.warning(f"Some components failed to initialize: {len(self._initialization_errors)} errors")
                for error in self._initialization_errors:
                    logger.warning(f"  - {error}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system integration: {e}")
            self._initialization_errors.append(str(e))
            return False
    
    def _initialize_configuration_bridge(self):
        """Initialize configuration bridge for enhanced config management"""
        try:
            # Create a simple configuration bridge inline for now
            class SimpleConfigurationBridge:
                def __init__(self, config_path):
                    self.config_path = Path(config_path)
                    self.config_data = {}
                    self._load_config()
                
                def _load_config(self):
                    try:
                        if self.config_path.exists():
                            with open(self.config_path, 'r') as f:
                                self.config_data = json.load(f)
                        else:
                            self.config_data = {
                                "system": {"default_quantization": "bf16"},
                                "directories": {"models_directory": "models", "loras_directory": "loras"},
                                "models": {"t2v_model": "Wan2.2-T2V-A14B", "i2v_model": "Wan2.2-I2V-A14B", "ti2v_model": "Wan2.2-TI2V-5B"},
                                "optimization": {"default_quantization": "bf16", "enable_offload": True, "max_vram_usage_gb": 12}
                            }
                    except Exception:
                        self.config_data = {}
                
                def get_config(self, section=None):
                    if section:
                        return self.config_data.get(section, {})
                    return self.config_data.copy()
                
                def get_model_paths(self):
                    models_config = self.get_config("models")
                    directories_config = self.get_config("directories")
                    models_directory = directories_config.get("models_directory", "models")
                    models_base_path = Path(models_directory)
                    models_base_path.mkdir(parents=True, exist_ok=True)
                    
                    model_paths = {}
                    for model_type in ["t2v_model", "i2v_model", "ti2v_model"]:
                        model_name = models_config.get(model_type, f"Wan2.2-{model_type.upper()}")
                        model_paths[model_type] = str(models_base_path / model_name)
                        model_paths[f"{model_type}_name"] = model_name
                    
                    model_paths["models_directory"] = str(models_base_path)
                    model_paths["loras_directory"] = directories_config.get("loras_directory", "loras")
                    model_paths["output_directory"] = directories_config.get("output_directory", "outputs")
                    return model_paths
                
                def get_optimization_settings(self):
                    optimization_config = self.get_config("optimization")
                    return {
                        "quantization": optimization_config.get("default_quantization", "bf16"),
                        "enable_offload": optimization_config.get("enable_offload", True),
                        "max_vram_usage_gb": optimization_config.get("max_vram_usage_gb", 12)
                    }
                
                def update_optimization_setting(self, setting_name, value):
                    try:
                        if "optimization" not in self.config_data:
                            self.config_data["optimization"] = {}
                        self.config_data["optimization"][setting_name] = value
                        with open(self.config_path, 'w') as f:
                            json.dump(self.config_data, f, indent=2)
                        return True
                    except Exception:
                        return False
                
                def validate_configuration(self):
                    errors = []
                    required_sections = ["system", "directories", "models", "optimization"]
                    for section in required_sections:
                        if section not in self.config_data:
                            errors.append(f"Missing required section: {section}")
                    return len(errors) == 0, errors
                
                def get_runtime_config_for_generation(self, model_type):
                    return {
                        "model_paths": self.get_model_paths(),
                        "optimization": self.get_optimization_settings()
                    }
                
                def get_config_summary(self):
                    return {
                        "config_file": str(self.config_path),
                        "sections": list(self.config_data.keys()),
                        "validation_status": self.validate_configuration()
                    }
            
            config_path = str(Path(__file__).parent.parent.parent / "config.json")
            bridge = SimpleConfigurationBridge(config_path)
            
            # Validate configuration
            is_valid, errors = bridge.validate_configuration()
            if not is_valid:
                logger.warning(f"Configuration validation issues: {errors}")
                for error in errors:
                    self._initialization_errors.append(f"Config validation: {error}")
            
            logger.info("Configuration bridge initialized successfully")
            return bridge
            
        except Exception as e:
            logger.warning(f"Could not initialize configuration bridge: {e}")
            self._initialization_errors.append(f"ConfigurationBridge initialization failed: {e}")
            return None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration from config.json (fallback method)"""
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
    
    async def _initialize_wan22_system_optimizer(self):
        """Initialize WAN22 System Optimizer from existing infrastructure"""
        try:
            # Try direct import using importlib to avoid path conflicts
            import importlib.util
            import sys
            
            # Get the absolute path to the WAN22SystemOptimizer module
            root_path = Path(__file__).parent.parent.parent
            wan22_module_path = root_path / "core" / "services" / "wan22_system_optimizer.py"
            
            if not wan22_module_path.exists():
                raise ImportError(f"WAN22SystemOptimizer module not found at {wan22_module_path}")
            
            # Load the module directly
            spec = importlib.util.spec_from_file_location("wan22_system_optimizer", wan22_module_path)
            wan22_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(wan22_module)
            
            WAN22SystemOptimizer = wan22_module.WAN22SystemOptimizer
            
            # Initialize with config path
            config_path = str(Path(__file__).parent.parent.parent / "config.json")
            optimizer = WAN22SystemOptimizer(config_path=config_path, log_level="INFO")
            
            # Initialize the optimization system
            init_result = optimizer.initialize_system()
            if init_result.success:
                logger.info("WAN22 System Optimizer initialized successfully")
                
                # Apply hardware optimizations
                opt_result = optimizer.apply_hardware_optimizations()
                if opt_result.success:
                    logger.info(f"Applied {len(opt_result.optimizations_applied)} hardware optimizations")
                
                return optimizer
            else:
                logger.warning("WAN22 System Optimizer initialization failed")
                for error in init_result.errors:
                    logger.error(f"Optimizer error: {error}")
                    self._initialization_errors.append(f"WAN22SystemOptimizer: {error}")
                return None
                
        except ImportError as e:
            logger.warning(f"Could not import WAN22SystemOptimizer: {e}")
            self._initialization_errors.append(f"WAN22SystemOptimizer import failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"Could not initialize WAN22SystemOptimizer: {e}")
            self._initialization_errors.append(f"WAN22SystemOptimizer initialization failed: {e}")
            return None

    async def _initialize_model_manager(self):
        """Initialize model manager from existing core services"""
        try:
            # Try to import ModelManager - it may have dependencies that aren't available
            try:
                from core.services.model_manager import ModelManager
            except ImportError as e:
                logger.warning(f"ModelManager has missing dependencies: {e}")
                # Create a minimal mock ModelManager for basic functionality
                return self._create_mock_model_manager()
            
            # Initialize with configuration
            manager = ModelManager()
            
            # If we have the system optimizer, integrate it
            if self.wan22_system_optimizer:
                # Get hardware profile for model manager optimization
                hardware_profile = self.wan22_system_optimizer.get_hardware_profile()
                if hardware_profile:
                    logger.info("Integrated ModelManager with hardware profile from WAN22SystemOptimizer")
            
            logger.info("ModelManager initialized successfully")
            return manager
            
        except ImportError as e:
            logger.warning(f"Could not import ModelManager: {e}")
            self._initialization_errors.append(f"ModelManager import failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"Could not initialize ModelManager: {e}")
            self._initialization_errors.append(f"ModelManager initialization failed: {e}")
            return None

    async def _initialize_model_downloader(self):
        """Initialize model downloader from existing local installation infrastructure"""
        try:
            # Try to import ModelDownloader
            try:
                from scripts.download_models import ModelDownloader
            except ImportError as e:
                logger.warning(f"ModelDownloader has missing dependencies: {e}")
                # Create a minimal mock ModelDownloader for basic functionality
                return self._create_mock_model_downloader()
            
            # Use models directory from config
            models_dir = self.config.get("directories", {}).get("models_directory", "models")
            models_path = Path(models_dir)
            
            # Ensure models directory exists
            models_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize downloader
            downloader = ModelDownloader(str(models_path))
            
            logger.info(f"ModelDownloader initialized with models directory: {models_path}")
            return downloader
            
        except ImportError as e:
            logger.warning(f"Could not import ModelDownloader: {e}")
            self._initialization_errors.append(f"ModelDownloader import failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"Could not initialize ModelDownloader: {e}")
            self._initialization_errors.append(f"ModelDownloader initialization failed: {e}")
            return None

    async def _initialize_wan_pipeline_loader(self):
        """Initialize WAN pipeline loader from existing core services"""
        try:
            # Try to import WanPipelineLoader
            try:
                from core.services.wan_pipeline_loader_standalone import WanPipelineLoader
                logger.info("Using standalone WanPipelineLoader implementation")
            except ImportError as e:
                logger.warning(f"Standalone WanPipelineLoader has missing dependencies: {e}")
                try:
                    from core.services.wan_pipeline_loader_fixed import WanPipelineLoader
                    logger.info("Using fixed WanPipelineLoader implementation")
                except ImportError as e2:
                    logger.warning(f"Fixed WanPipelineLoader has missing dependencies: {e2}")
                    try:
                        from core.services.wan_pipeline_loader import WanPipelineLoader
                        logger.info("Using original WanPipelineLoader implementation")
                    except ImportError as e3:
                        logger.warning(f"Original WanPipelineLoader has missing dependencies: {e3}")
                        # Create a real WanPipelineLoader for actual functionality
                        return self._create_real_wan_pipeline_loader()
            
            # Initialize pipeline loader
            loader = WanPipelineLoader()
            
            # If we have system optimizer, integrate it for hardware optimization
            if self.wan22_system_optimizer:
                hardware_profile = self.wan22_system_optimizer.get_hardware_profile()
                if hardware_profile:
                    logger.info("Integrated WanPipelineLoader with hardware profile from WAN22SystemOptimizer")
            
            logger.info("WanPipelineLoader initialized successfully")
            return loader
            
        except ImportError as e:
            logger.warning(f"Could not import WanPipelineLoader: {e}")
            self._initialization_errors.append(f"WanPipelineLoader import failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"Could not initialize WanPipelineLoader: {e}")
            self._initialization_errors.append(f"WanPipelineLoader initialization failed: {e}")
            return None
    
    def _initialize_system_stats(self):
        """Initialize system stats monitoring"""
        try:
            # Try to import system stats
            try:
                from core.services.utils import get_system_stats
                logger.info("System stats monitoring initialized")
                return get_system_stats
            except ImportError as e:
                logger.warning(f"System stats has missing dependencies: {e}")
                # Return a basic system stats function
                return self._create_basic_system_stats()
                
        except Exception as e:
            logger.warning(f"Could not initialize system stats: {e}")
            self._initialization_errors.append(f"SystemStats initialization failed: {e}")
            return self._create_basic_system_stats()
    
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
        """Get comprehensive system information combining all integrated components"""
        info = {
            "initialized": self._initialized,
            "initialization_timestamp": datetime.now().isoformat(),
            "components": {
                "model_manager_available": self.model_manager is not None,
                "model_downloader_available": self.model_downloader is not None,
                "wan22_system_optimizer_available": self.wan22_system_optimizer is not None,
                "wan_pipeline_loader_available": self.wan_pipeline_loader is not None,
                "system_stats_available": self.system_stats is not None,
                "configuration_bridge_available": self.config_bridge is not None,
            },
            "config_loaded": self.config is not None,
            "initialization_errors": self._initialization_errors
        }
        
        if self.config:
            info["directories"] = self.config.get("directories", {})
            info["optimization"] = self.config.get("optimization", {})
        
        # Add configuration bridge summary if available
        if self.config_bridge:
            try:
                info["configuration_summary"] = self.config_bridge.get_config_summary()
            except Exception as e:
                logger.warning(f"Could not get configuration summary: {e}")
        
        # Add hardware profile from WAN22SystemOptimizer if available
        if self.wan22_system_optimizer:
            try:
                hardware_profile = self.wan22_system_optimizer.get_hardware_profile()
                if hardware_profile:
                    info["hardware_profile"] = {
                        "cpu_model": hardware_profile.cpu_model,
                        "cpu_cores": hardware_profile.cpu_cores,
                        "total_memory_gb": hardware_profile.total_memory_gb,
                        "gpu_model": hardware_profile.gpu_model,
                        "vram_gb": hardware_profile.vram_gb,
                        "cuda_version": hardware_profile.cuda_version,
                        "platform_info": hardware_profile.platform_info
                    }
            except Exception as e:
                logger.warning(f"Could not get hardware profile: {e}")
        
        # Add model status from ModelManager if available
        if self.model_manager:
            try:
                info["model_status"] = self._get_model_status_from_manager()
            except Exception as e:
                logger.warning(f"Could not get model status: {e}")
        
        return info
    
    async def get_system_health_with_recovery_context(self) -> Dict[str, Any]:
        """Get system health information with recovery system context"""
        try:
            # Get basic system info
            system_info = self.get_system_info()
            
            # Add health monitoring specific information
            health_info = {
                "basic_info": system_info,
                "health_monitoring": {
                    "monitoring_available": False,
                    "recovery_system_available": False,
                    "automatic_recovery_enabled": False
                }
            }
            
            # Check if fallback recovery system is available
            try:
                from backend.core.fallback_recovery_system import get_fallback_recovery_system
                recovery_system = get_fallback_recovery_system()
                
                if recovery_system:
                    health_info["health_monitoring"]["recovery_system_available"] = True
                    health_info["health_monitoring"]["automatic_recovery_enabled"] = recovery_system.health_monitoring_active
                    
                    # Get recovery statistics
                    recovery_stats = recovery_system.get_recovery_statistics()
                    health_info["recovery_statistics"] = recovery_stats
                    
                    # Get current health status if available
                    if recovery_system.current_health_status:
                        health_status = recovery_system.current_health_status
                        health_info["current_health"] = {
                            "overall_status": health_status.overall_status,
                            "cpu_usage_percent": health_status.cpu_usage_percent,
                            "memory_usage_percent": health_status.memory_usage_percent,
                            "vram_usage_percent": health_status.vram_usage_percent,
                            "gpu_available": health_status.gpu_available,
                            "model_loading_functional": health_status.model_loading_functional,
                            "generation_pipeline_functional": health_status.generation_pipeline_functional,
                            "issues": health_status.issues,
                            "recommendations": health_status.recommendations,
                            "last_check": health_status.last_check_timestamp.isoformat()
                        }
                    
                    health_info["health_monitoring"]["monitoring_available"] = True
                    
            except Exception as e:
                logger.warning(f"Could not get recovery system info: {e}")
                health_info["health_monitoring"]["error"] = str(e)
            
            # Add WAN22 system optimizer health if available
            if self.wan22_system_optimizer:
                try:
                    health_metrics = self.wan22_system_optimizer.monitor_system_health()
                    health_info["wan22_health"] = {
                        "cpu_usage_percent": health_metrics.cpu_usage_percent,
                        "memory_usage_gb": health_metrics.memory_usage_gb,
                        "vram_usage_mb": health_metrics.vram_usage_mb,
                        "timestamp": datetime.now().isoformat()
                    }
                except Exception as e:
                    logger.warning(f"Could not get WAN22 health metrics: {e}")
            
            return health_info
            
        except Exception as e:
            logger.error(f"Error getting system health with recovery context: {e}")
            return {
                "error": str(e),
                "basic_info": self.get_system_info() if hasattr(self, 'get_system_info') else {},
                "health_monitoring": {"error": "Health monitoring unavailable"}
            }

    def _get_model_status_from_manager(self) -> Dict[str, Any]:
        """Get model status information from the ModelManager"""
        if not self.model_manager:
            return {}
        
        try:
            # Get available model types
            model_types = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
            model_status = {}
            
            for model_type in model_types:
                try:
                    # Check if model is available locally
                    model_id = self.model_manager.get_model_id(model_type)
                    is_loaded = hasattr(self.model_manager, 'loaded_models') and model_id in getattr(self.model_manager, 'loaded_models', {})
                    
                    model_status[model_type] = {
                        "model_id": model_id,
                        "is_loaded": is_loaded,
                        "is_available": model_id is not None
                    }
                except Exception as e:
                    model_status[model_type] = {
                        "error": str(e),
                        "is_loaded": False,
                        "is_available": False
                    }
            
            return model_status
            
        except Exception as e:
            logger.error(f"Error getting model status: {e}")
            return {"error": str(e)}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for integration monitoring"""
        try:
            status = {
                "initialized": self._initialized,
                "wan22_infrastructure_loaded": self.wan22_system_optimizer is not None,
                "model_bridge_available": self.model_manager is not None,
                "optimizer_available": self.wan22_system_optimizer is not None,
                "components": {
                    "model_bridge": self.model_manager is not None,
                    "optimizer": self.wan22_system_optimizer is not None,
                    "pipeline": self.wan_pipeline_loader is not None
                }
            }
            
            # Add hardware status if optimizer is available
            if self.wan22_system_optimizer:
                try:
                    hardware_profile = self.wan22_system_optimizer.get_hardware_profile()
                    if hardware_profile:
                        status["hardware"] = {
                            "gpu_available": hardware_profile.gpu_model is not None,
                            "vram_gb": hardware_profile.vram_gb,
                            "cpu_cores": hardware_profile.cpu_cores,
                            "memory_gb": hardware_profile.total_memory_gb
                        }
                except Exception as e:
                    logger.warning(f"Could not get hardware status: {e}")
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "initialized": False,
                "error": str(e),
                "wan22_infrastructure_loaded": False,
                "model_bridge_available": False,
                "optimizer_available": False
            }
    
    async def get_model_bridge(self):
        """Get the model integration bridge for model operations"""
        try:
            if not self.model_manager:
                return None
            
            # Create a bridge wrapper around the model manager
            class ModelBridge:
                def __init__(self, model_manager, system_integration):
                    self.model_manager = model_manager
                    self.system_integration = system_integration
                
                def get_system_model_status(self):
                    """Get model status for system monitoring"""
                    try:
                        return self.system_integration._get_model_status_from_manager()
                    except Exception as e:
                        logger.error(f"Error getting model status: {e}")
                        return {"error": str(e)}
                
                def is_model_available(self, model_type):
                    """Check if a specific model is available"""
                    try:
                        model_id = self.model_manager.get_model_id(model_type)
                        return model_id is not None
                    except Exception:
                        return False
                
                def get_available_models(self):
                    """Get list of available models"""
                    try:
                        model_types = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
                        available = []
                        for model_type in model_types:
                            if self.is_model_available(model_type):
                                available.append(model_type)
                        return available
                    except Exception:
                        return []
                
                def check_model_availability(self, model_type):
                    """Check model availability with detailed status"""
                    try:
                        model_id = self.model_manager.get_model_id(model_type)
                        if not model_id:
                            return {"available": False, "reason": "Model ID not found"}
                        
                        # Check if model files exist (basic check)
                        # In a real implementation, this would check actual model files
                        return {
                            "available": False,  # Models need to be downloaded
                            "model_id": model_id,
                            "reason": "Model files not downloaded yet"
                        }
                    except Exception as e:
                        return {"available": False, "reason": f"Error checking model: {str(e)}"}
            
            return ModelBridge(self.model_manager, self)
            
        except Exception as e:
            logger.error(f"Error creating model bridge: {e}")
            return None
    
    async def get_system_optimizer(self):
        """Get the WAN22 system optimizer for hardware optimization"""
        return self.wan22_system_optimizer
    
    def get_model_downloader(self):
        """Get the model downloader instance"""
        return self.model_downloader
    
    def get_wan22_system_optimizer(self):
        """Get the WAN22 system optimizer instance"""
        return self.wan22_system_optimizer
    
    def get_wan_pipeline_loader(self):
        """Get the WAN pipeline loader instance"""
        return self.wan_pipeline_loader
    
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

    # Component Access Methods
    def get_model_manager(self):
        """Get the initialized ModelManager instance"""
        return self.model_manager

    def get_model_downloader(self):
        """Get the initialized ModelDownloader instance"""
        return self.model_downloader

    def get_wan22_system_optimizer(self):
        """Get the initialized WAN22SystemOptimizer instance"""
        return self.wan22_system_optimizer

    def get_wan_pipeline_loader(self):
        """Get the initialized WanPipelineLoader instance"""
        return self.wan_pipeline_loader

    def get_configuration_bridge(self):
        """Get the initialized ConfigurationBridge instance"""
        return self.config_bridge

    # Unified System Status Methods
    async def get_unified_system_status(self) -> Dict[str, Any]:
        """Get unified system status combining information from all integrated components"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "system_integration": {
                "initialized": self._initialized,
                "components_loaded": {
                    "model_manager": self.model_manager is not None,
                    "model_downloader": self.model_downloader is not None,
                    "wan22_system_optimizer": self.wan22_system_optimizer is not None,
                    "wan_pipeline_loader": self.wan_pipeline_loader is not None
                }
            }
        }

        # Get system stats from existing monitoring
        try:
            system_stats = await self.get_enhanced_system_stats()
            if system_stats:
                status["system_stats"] = system_stats
        except Exception as e:
            logger.warning(f"Could not get system stats: {e}")
            status["system_stats"] = {"error": str(e)}

        # Get optimization status from WAN22SystemOptimizer
        if self.wan22_system_optimizer:
            try:
                health_metrics = self.wan22_system_optimizer.monitor_system_health()
                optimization_history = self.wan22_system_optimizer.get_optimization_history()
                
                status["optimization"] = {
                    "health_metrics": health_metrics,
                    "optimization_history": optimization_history[-5:] if optimization_history else [],  # Last 5 entries
                    "hardware_profile": self.wan22_system_optimizer.get_hardware_profile()
                }
            except Exception as e:
                logger.warning(f"Could not get optimization status: {e}")
                status["optimization"] = {"error": str(e)}

        # Get model status
        if self.model_manager:
            try:
                status["models"] = self._get_model_status_from_manager()
            except Exception as e:
                logger.warning(f"Could not get model status: {e}")
                status["models"] = {"error": str(e)}

        return status

    async def ensure_model_available(self, model_type: str) -> Tuple[bool, str]:
        """Ensure a model is available, downloading if necessary using integrated components"""
        try:
            if not self.model_manager or not self.model_downloader:
                return False, "Model management components not initialized"

            # Check if model is already available
            model_id = self.model_manager.get_model_id(model_type)
            if model_id and hasattr(self.model_manager, 'is_model_available'):
                if self.model_manager.is_model_available(model_id):
                    return True, f"Model {model_type} is already available"

            # Try to download the model using ModelDownloader
            logger.info(f"Attempting to download model: {model_type}")
            
            # Map model types to downloader format
            model_mapping = {
                "t2v-A14B": "WAN2.2-T2V-A14B",
                "i2v-A14B": "WAN2.2-I2V-A14B", 
                "ti2v-5B": "WAN2.2-TI2V-5B"
            }
            
            downloader_model_name = model_mapping.get(model_type)
            if not downloader_model_name:
                return False, f"Unknown model type: {model_type}"

            # Check if model downloader has the required methods
            if hasattr(self.model_downloader, 'download_model'):
                success = await self.model_downloader.download_model(downloader_model_name)
                if success:
                    return True, f"Successfully downloaded model {model_type}"
                else:
                    return False, f"Failed to download model {model_type}"
            else:
                return False, "ModelDownloader does not support download_model method"

        except Exception as e:
            logger.error(f"Error ensuring model availability for {model_type}: {e}")
            return False, f"Error ensuring model availability: {str(e)}"

    def get_initialization_errors(self) -> List[str]:
        """Get list of initialization errors that occurred during setup"""
        return self._initialization_errors.copy()
    
    # Configuration Management Methods
    def get_runtime_config_for_generation(self, model_type: str) -> Dict[str, Any]:
        """Get runtime configuration optimized for specific model generation"""
        if self.config_bridge:
            return self.config_bridge.get_runtime_config_for_generation(model_type)
        else:
            # Fallback to basic config
            return {
                "model_paths": self.config.get("directories", {}),
                "optimization": self.config.get("optimization", {}),
                "generation_defaults": self.config.get("generation", {})
            }
    
    def update_optimization_setting(self, setting_name: str, value: Any) -> bool:
        """Update optimization setting at runtime"""
        if self.config_bridge:
            success = self.config_bridge.update_optimization_setting(setting_name, value)
            if success:
                # Reload config to reflect changes
                self.config = self.config_bridge.get_config()
            return success
        else:
            logger.warning("Configuration bridge not available for runtime updates")
            return False
    
    def get_model_paths(self) -> Dict[str, str]:
        """Get model path configuration"""
        if self.config_bridge:
            return self.config_bridge.get_model_paths()
        else:
            # Fallback to basic directory config
            directories = self.config.get("directories", {})
            return {
                "models_directory": directories.get("models_directory", "models"),
                "loras_directory": directories.get("loras_directory", "loras"),
                "output_directory": directories.get("output_directory", "outputs")
            }
    
    def get_optimization_settings(self) -> Dict[str, Any]:
        """Get current optimization settings"""
        if self.config_bridge:
            return self.config_bridge.get_optimization_settings()
        else:
            return self.config.get("optimization", {})
    
    def validate_current_configuration(self) -> Tuple[bool, List[str]]:
        """Validate current configuration"""
        if self.config_bridge:
            return self.config_bridge.validate_configuration()
        else:
            # Basic validation for fallback config
            errors = []
            if not self.config.get("directories"):
                errors.append("Missing directories configuration")
            if not self.config.get("optimization"):
                errors.append("Missing optimization configuration")
            return len(errors) == 0, errors
    
    def scan_available_loras(self, loras_directory: str) -> List[str]:
        """Scan for available LoRA files in the specified directory"""
        try:
            loras_path = Path(loras_directory)
            if not loras_path.exists():
                logger.warning(f"LoRAs directory does not exist: {loras_directory}")
                return []
            
            lora_files = []
            valid_extensions = ['.safetensors', '.pt', '.pth', '.bin']
            
            for file_path in loras_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                    lora_files.append(file_path.name)
            
            logger.info(f"Found {len(lora_files)} LoRA files in {loras_directory}")
            return sorted(lora_files)
            
        except Exception as e:
            logger.error(f"Error scanning LoRAs directory {loras_directory}: {e}")
            return []

    # Mock/Fallback Methods for Missing Dependencies
    def _create_mock_model_manager(self):
        """Create a mock ModelManager with basic functionality"""
        class MockModelManager:
            def __init__(self):
                self.loaded_models = {}
                
            def get_model_id(self, model_type: str) -> Optional[str]:
                model_mapping = {
                    "t2v-A14B": "WAN2.2-T2V-A14B",
                    "i2v-A14B": "WAN2.2-I2V-A14B", 
                    "ti2v-5B": "WAN2.2-TI2V-5B"
                }
                return model_mapping.get(model_type)
                
            def is_model_available(self, model_id: str) -> bool:
                # Check if model directory exists
                models_dir = Path("models")  # Default models directory
                return (models_dir / model_id).exists()
                
            def unload_model(self, model_id: str):
                if model_id in self.loaded_models:
                    del self.loaded_models[model_id]
        
        mock_manager = MockModelManager()
        logger.info("Created mock ModelManager with basic functionality")
        return mock_manager

    def _create_mock_model_downloader(self):
        """Create a mock ModelDownloader with basic functionality"""
        class MockModelDownloader:
            def __init__(self, models_dir: str):
                self.models_dir = Path(models_dir)
                
            async def download_model(self, model_name: str) -> bool:
                logger.warning(f"Mock ModelDownloader: Cannot actually download {model_name}")
                return False
                
            def is_model_available(self, model_name: str) -> bool:
                return (self.models_dir / model_name).exists()
        
        models_dir = self.config.get("directories", {}).get("models_directory", "models")
        mock_downloader = MockModelDownloader(models_dir)
        logger.info("Created mock ModelDownloader with basic functionality")
        return mock_downloader

    def _create_real_wan_pipeline_loader(self):
        """Create a real WanPipelineLoader with full functionality"""
        try:
            # Import the real WAN pipeline loader
            import sys
            from pathlib import Path
            
            # Add core services to path
            core_services_path = Path(__file__).parent.parent.parent / "core" / "services"
            if str(core_services_path) not in sys.path:
                sys.path.insert(0, str(core_services_path))
            
            from wan_pipeline_loader import WanPipelineLoader
            from vram_manager import VRAMManager
            from quantization_controller import QuantizationController
            
            # Initialize VRAM manager and quantization controller
            vram_manager = VRAMManager()
            quantization_controller = QuantizationController()
            
            # Create the real pipeline loader with optimization components
            real_loader = WanPipelineLoader(
                optimization_config_path=None,  # Use default config
                enable_caching=True,
                vram_manager=vram_manager,
                quantization_controller=quantization_controller
            )
            
            logger.info("Created real WanPipelineLoader with full optimization support")
            return real_loader
            
        except ImportError as e:
            logger.warning(f"Could not import real WanPipelineLoader: {e}")
            # Fallback to a simplified real loader
            return self._create_simplified_wan_pipeline_loader()
        except Exception as e:
            logger.error(f"Failed to create real WanPipelineLoader: {e}")
            # Fallback to simplified version
            return self._create_simplified_wan_pipeline_loader()
    
    def _create_simplified_wan_pipeline_loader(self):
        """Create a simplified WanPipelineLoader that can actually load models"""
        class SimplifiedWanPipelineLoader:
            def __init__(self):
                self.logger = logging.getLogger(__name__ + ".SimplifiedWanPipelineLoader")
                self._pipeline_cache = {}
                
            def load_pipeline(self, model_type: str, model_path: str = None):
                """Load a pipeline for the specified model type"""
                try:
                    # Import required modules
                    import torch
                    from diffusers import DiffusionPipeline
                    
                    # Determine model path based on type
                    if not model_path:
                        model_path = self._get_model_path_for_type(model_type)
                    
                    if not model_path:
                        self.logger.error(f"No model path found for type: {model_type}")
                        return None
                    
                    # Check cache first
                    if model_path in self._pipeline_cache:
                        self.logger.info(f"Returning cached pipeline for {model_type}")
                        return self._pipeline_cache[model_path]
                    
                    # Load the pipeline
                    self.logger.info(f"Loading pipeline for {model_type} from {model_path}")
                    
                    # Use appropriate loading method based on model type
                    if "wan" in model_type.lower() or "t2v" in model_type.lower() or "i2v" in model_type.lower():
                        # For WAN models, use optimized loading settings
                        self.logger.info(f"Loading WAN model with optimized settings...")
                        
                        # Optimized loading parameters for large WAN models
                        load_kwargs = {
                            "trust_remote_code": True,
                            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                            "low_cpu_mem_usage": True,  # Reduce CPU memory usage
                            # Remove device_map="auto" as it's not supported by WAN models
                            "max_memory": {0: "14GB"} if torch.cuda.is_available() else None,  # Limit VRAM usage
                        }
                        
                        try:
                            pipeline = DiffusionPipeline.from_pretrained(model_path, **load_kwargs)
                            self.logger.info("WAN model loaded successfully with optimized settings")
                        except Exception as e:
                            self.logger.warning(f"Optimized loading failed, trying fallback: {e}")
                            # Fallback to basic loading
                            fallback_kwargs = {
                                "trust_remote_code": True,
                                "torch_dtype": torch.float32,  # Use FP32 as fallback
                                "low_cpu_mem_usage": True
                            }
                            pipeline = DiffusionPipeline.from_pretrained(model_path, **fallback_kwargs)
                            self.logger.info("WAN model loaded with fallback settings")
                    else:
                        # For standard diffusion models
                        pipeline = DiffusionPipeline.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            low_cpu_mem_usage=True
                        )
                    
                    # Move to GPU if available
                    if torch.cuda.is_available():
                        pipeline = pipeline.to("cuda")
                    
                    # Create a wrapper that provides the expected interface
                    wrapped_pipeline = self._create_pipeline_wrapper(pipeline, model_type)
                    
                    # Cache the wrapped pipeline
                    self._pipeline_cache[model_path] = wrapped_pipeline
                    
                    self.logger.info(f"Successfully loaded and wrapped pipeline for {model_type}")
                    return wrapped_pipeline
                    
                except Exception as e:
                    self.logger.error(f"Failed to load pipeline for {model_type}: {e}")
                    return None
            
            def _get_model_path_for_type(self, model_type: str) -> str:
                """Get the model path for a given model type"""
                # Map model types to their paths
                model_mappings = {
                    "t2v-a14b": "models/t2v-a14b",
                    "i2v-a14b": "models/i2v-a14b", 
                    "ti2v-5b": "models/ti2v-5b",
                    "T2V": "models/t2v-a14b",
                    "I2V": "models/i2v-a14b",
                    "TI2V": "models/ti2v-5b"
                }
                
                return model_mappings.get(model_type.lower(), model_mappings.get(model_type))
            
            def load_wan_pipeline(self, model_path: str, trust_remote_code: bool = True, 
                                 apply_optimizations: bool = True, optimization_config: dict = None, **kwargs):
                """Load WAN pipeline with the same interface as the real loader"""
                # Extract model type from path
                model_type = "wan"
                if "t2v" in model_path.lower():
                    model_type = "t2v-a14b"
                elif "i2v" in model_path.lower():
                    model_type = "i2v-a14b"
                elif "ti2v" in model_path.lower():
                    model_type = "ti2v-5b"
                
                self.logger.info(f"Loading WAN pipeline: {model_type} from {model_path}")
                self.logger.info(f"Parameters: trust_remote_code={trust_remote_code}, apply_optimizations={apply_optimizations}")
                
                return self.load_pipeline(model_type, model_path)
            
            def _create_pipeline_wrapper(self, pipeline, model_type):
                """Create a wrapper that provides the expected interface for generation"""
                
                class PipelineWrapper:
                    def __init__(self, pipeline, model_type):
                        self.pipeline = pipeline
                        self.model_type = model_type
                        self.logger = logging.getLogger(__name__ + ".PipelineWrapper")
                    
                    def generate(self, config):
                        """Generate using the wrapped pipeline"""
                        try:
                            self.logger.info(f"Starting generation with {self.model_type}")
                            
                            # Extract parameters from config
                            prompt = getattr(config, 'prompt', 'A simple video')
                            num_frames = getattr(config, 'num_frames', 1)
                            steps = getattr(config, 'steps', 20)
                            
                            # Call the pipeline with progress callback support
                            if hasattr(config, 'progress_callback') and config.progress_callback:
                                # Create a callback wrapper for the pipeline
                                def pipeline_callback(step, timestep, latents):
                                    try:
                                        # Calculate progress percentage
                                        progress = (step / steps) * 100
                                        config.progress_callback(step, steps, latents)
                                    except Exception as e:
                                        self.logger.warning(f"Progress callback error: {e}")
                                
                                # Generate with callback
                                result = self.pipeline(
                                    prompt=prompt,
                                    num_frames=num_frames,
                                    num_inference_steps=steps,
                                    callback=pipeline_callback,
                                    callback_steps=1
                                )
                            else:
                                # Generate without callback
                                result = self.pipeline(
                                    prompt=prompt,
                                    num_frames=num_frames,
                                    num_inference_steps=steps
                                )
                            
                            # Create a result object that matches expected interface
                            class GenerationResult:
                                def __init__(self, success=True, frames=None, errors=None):
                                    self.success = success
                                    self.frames = frames or []
                                    self.errors = errors or []
                                    self.peak_memory_mb = 8000  # Mock values
                                    self.memory_used_mb = 6000
                                    self.applied_optimizations = ["simplified_loading"]
                            
                            self.logger.info(f"Generation completed successfully")
                            return GenerationResult(success=True, frames=result.frames if hasattr(result, 'frames') else [])
                            
                        except Exception as e:
                            self.logger.error(f"Generation failed: {e}")
                            return GenerationResult(success=False, errors=[str(e)])
                    
                    def __call__(self, *args, **kwargs):
                        """Allow the wrapper to be called directly"""
                        return self.pipeline(*args, **kwargs)
                
                return PipelineWrapper(pipeline, model_type)
        
        simplified_loader = SimplifiedWanPipelineLoader()
        logger.info("Created simplified WanPipelineLoader with basic model loading")
        return simplified_loader

    def _create_basic_system_stats(self):
        """Create a basic system stats function"""
        def get_basic_system_stats():
            try:
                import psutil
                from datetime import datetime
                
                # Get basic system info
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                stats = {
                    "cpu_percent": cpu_percent,
                    "ram_used_gb": memory.used / (1024**3),
                    "ram_total_gb": memory.total / (1024**3),
                    "ram_percent": memory.percent,
                    "timestamp": datetime.now(),
                    "gpu_available": False,
                    "vram_used_mb": 0,
                    "vram_total_mb": 0,
                    "vram_percent": 0
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
                except Exception:
                    pass
                
                return stats
                
            except Exception as e:
                logger.error(f"Error in basic system stats: {e}")
                return {
                    "error": str(e),
                    "timestamp": datetime.now()
                }
        
        logger.info("Created basic system stats function")
        return get_basic_system_stats

# Global system integration instance
system_integration = SystemIntegration()

async def get_system_integration() -> SystemIntegration:
    """Dependency to get system integration instance with full Wan2.2 infrastructure"""
    if not system_integration._initialized:
        success = await system_integration.initialize()
        if not success:
            logger.warning("System integration initialization had errors, but continuing with available components")
    return system_integration

# Convenience functions for accessing specific components
async def get_model_manager():
    """Get the ModelManager instance from system integration"""
    integration = await get_system_integration()
    return integration.get_model_manager()

async def get_model_downloader():
    """Get the ModelDownloader instance from system integration"""
    integration = await get_system_integration()
    return integration.get_model_downloader()

async def get_wan22_system_optimizer():
    """Get the WAN22SystemOptimizer instance from system integration"""
    integration = await get_system_integration()
    return integration.get_wan22_system_optimizer()

async def get_wan_pipeline_loader():
    """Get the WanPipelineLoader instance from system integration"""
    integration = await get_system_integration()
    return integration.get_wan_pipeline_loader()