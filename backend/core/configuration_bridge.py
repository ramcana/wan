"""Configuration Bridge for FastAPI backend integration"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfigurationBridge:
    """Configuration adapter for existing config.json structure with FastAPI"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path(__file__).parent.parent.parent / "config.json"
        self.config_data = {}
        self.last_modified = None
        self._load_config()
    
    def _load_config(self) -> bool:
        """Load configuration from config.json file"""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}")
                self._create_default_config()
                return False
            
            with open(self.config_path, 'r') as f:
                self.config_data = json.load(f)
            
            self.last_modified = datetime.fromtimestamp(self.config_path.stat().st_mtime)
            logger.info(f"Configuration loaded from {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self._create_default_config()
            return False
    
    def _create_default_config(self):
        """Create default configuration structure"""
        self.config_data = {
            "system": {"default_quantization": "bf16", "enable_offload": True},
            "directories": {"output_directory": "outputs", "models_directory": "models", "loras_directory": "loras"},
            "models": {"t2v_model": "Wan2.2-T2V-A14B", "i2v_model": "Wan2.2-I2V-A14B", "ti2v_model": "Wan2.2-TI2V-5B"},
            "optimization": {"default_quantization": "bf16", "enable_offload": True, "max_vram_usage_gb": 12}
        }
    
    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration data"""
        if section:
            return self.config_data.get(section, {})
        return self.config_data.copy()
    
    def get_model_paths(self) -> Dict[str, str]:
        """Get model path configuration"""
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
    
    def get_optimization_settings(self) -> Dict[str, Any]:
        """Get optimization settings"""
        optimization_config = self.get_config("optimization")
        return {
            "quantization": optimization_config.get("default_quantization", "bf16"),
            "enable_offload": optimization_config.get("enable_offload", True),
            "max_vram_usage_gb": optimization_config.get("max_vram_usage_gb", 12)
        }
    
    def update_optimization_setting(self, setting_name: str, value: Any) -> bool:
        """Update optimization setting at runtime"""
        try:
            if "optimization" not in self.config_data:
                self.config_data["optimization"] = {}
            self.config_data["optimization"][setting_name] = value
            return self._save_config()
        except Exception as e:
            logger.error(f"Failed to update optimization setting {setting_name}: {e}")
            return False
    
    def _save_config(self) -> bool:
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config_data, f, indent=2)
            self.last_modified = datetime.fromtimestamp(self.config_path.stat().st_mtime)
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate current configuration"""
        errors = []
        required_sections = ["system", "directories", "models", "optimization"]
        for section in required_sections:
            if section not in self.config_data:
                errors.append(f"Missing required section: {section}")
        return len(errors) == 0, errors
    
    def get_runtime_config_for_generation(self, model_type: str) -> Dict[str, Any]:
        """Get runtime configuration for model generation"""
        return {
            "model_paths": self.get_model_paths(),
            "optimization": self.get_optimization_settings()
        }
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            "config_file": str(self.config_path),
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "sections": list(self.config_data.keys()),
            "validation_status": self.validate_configuration()
        }