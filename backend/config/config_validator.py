"""
Configuration validation for ensuring existing config.json works with new system.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

class ConfigValidationResult(BaseModel):
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    migrated_config: Optional[Dict[str, Any]] = None

class ConfigValidator:
    """Validates and migrates existing config.json for new system."""
    
    # Expected configuration schema for new system
    EXPECTED_SCHEMA = {
        "model_settings": {
            "default_model": str,
            "model_cache_dir": str,
            "enable_model_offloading": bool,
            "quantization_mode": str,  # "fp16", "bf16", "int8", "none"
        },
        "generation_settings": {
            "default_resolution": str,
            "default_steps": int,
            "max_queue_size": int,
            "enable_progress_tracking": bool,
        },
        "optimization_settings": {
            "vram_optimization": bool,
            "cpu_offload": bool,
            "vae_tile_size": int,
            "enable_attention_slicing": bool,
        },
        "api_settings": {
            "host": str,
            "port": int,
            "cors_origins": list,
            "max_file_size_mb": int,
        },
        "storage_settings": {
            "outputs_dir": str,
            "thumbnails_dir": str,
            "temp_dir": str,
            "max_storage_gb": int,
        },
        "logging_settings": {
            "log_level": str,
            "log_file": str,
            "enable_performance_logging": bool,
        }
    }
    
    # Mapping from old config keys to new config structure
    CONFIG_MIGRATION_MAP = {
        # Model settings
        "model_type": ("model_settings", "default_model"),
        "model_path": ("model_settings", "model_cache_dir"),
        "offload_model": ("model_settings", "enable_model_offloading"),
        "quantization": ("model_settings", "quantization_mode"),
        
        # Generation settings
        "resolution": ("generation_settings", "default_resolution"),
        "steps": ("generation_settings", "default_steps"),
        "queue_size": ("generation_settings", "max_queue_size"),
        
        # Optimization settings
        "vram_optimize": ("optimization_settings", "vram_optimization"),
        "cpu_offload": ("optimization_settings", "cpu_offload"),
        "vae_tile": ("optimization_settings", "vae_tile_size"),
        "attention_slicing": ("optimization_settings", "enable_attention_slicing"),
        
        # API settings
        "server_host": ("api_settings", "host"),
        "server_port": ("api_settings", "port"),
        "max_file_size": ("api_settings", "max_file_size_mb"),
        
        # Storage settings
        "output_dir": ("storage_settings", "outputs_dir"),
        "temp_dir": ("storage_settings", "temp_dir"),
    }
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.validation_result = ConfigValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[]
        )
    
    def load_existing_config(self) -> Optional[Dict[str, Any]]:
        """Load existing configuration file."""
        try:
            if not self.config_path.exists():
                self.validation_result.warnings.append(
                    f"Configuration file {self.config_path} not found"
                )
                return None
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in configuration file: {e}"
            self.validation_result.errors.append(error_msg)
            logger.error(error_msg)
            return None
        except Exception as e:
            error_msg = f"Failed to load configuration: {e}"
            self.validation_result.errors.append(error_msg)
            logger.error(error_msg)
            return None
    
    def validate_config_structure(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure against expected schema."""
        is_valid = True
        
        # Check for required sections
        required_sections = ["model_settings", "generation_settings", "optimization_settings"]
        for section in required_sections:
            if section not in config:
                self.validation_result.warnings.append(
                    f"Missing configuration section: {section}"
                )
        
        # Validate each section
        for section_name, section_schema in self.EXPECTED_SCHEMA.items():
            if section_name in config:
                section_valid = self._validate_section(
                    config[section_name], 
                    section_schema, 
                    section_name
                )
                is_valid = is_valid and section_valid
        
        return is_valid
    
    def _validate_section(self, section: Dict[str, Any], schema: Dict[str, type], section_name: str) -> bool:
        """Validate a configuration section."""
        is_valid = True
        
        for key, expected_type in schema.items():
            if key in section:
                value = section[key]
                if not isinstance(value, expected_type):
                    error_msg = f"Invalid type for {section_name}.{key}: expected {expected_type.__name__}, got {type(value).__name__}"
                    self.validation_result.errors.append(error_msg)
                    is_valid = False
            else:
                self.validation_result.warnings.append(
                    f"Missing configuration key: {section_name}.{key}"
                )
        
        return is_valid
    
    def migrate_config(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate old configuration format to new format."""
        new_config = {
            "model_settings": {},
            "generation_settings": {},
            "optimization_settings": {},
            "api_settings": {},
            "storage_settings": {},
            "logging_settings": {}
        }
        
        # Apply migration mappings
        for old_key, (new_section, new_key) in self.CONFIG_MIGRATION_MAP.items():
            if old_key in old_config:
                new_config[new_section][new_key] = old_config[old_key]
                logger.info(f"Migrated {old_key} -> {new_section}.{new_key}")
        
        # Set defaults for missing values
        self._set_config_defaults(new_config)
        
        # Handle special cases
        self._handle_special_migrations(old_config, new_config)
        
        return new_config
    
    def _set_config_defaults(self, config: Dict[str, Any]) -> None:
        """Set default values for missing configuration options."""
        defaults = {
            "model_settings": {
                "default_model": "T2V-A14B",
                "model_cache_dir": "models",
                "enable_model_offloading": True,
                "quantization_mode": "fp16",
            },
            "generation_settings": {
                "default_resolution": "1280x720",
                "default_steps": 50,
                "max_queue_size": 10,
                "enable_progress_tracking": True,
            },
            "optimization_settings": {
                "vram_optimization": True,
                "cpu_offload": False,
                "vae_tile_size": 512,
                "enable_attention_slicing": True,
            },
            "api_settings": {
                "host": "0.0.0.0",
                "port": 8000,
                "cors_origins": ["http://localhost:3000", "http://localhost:5173"],
                "max_file_size_mb": 100,
            },
            "storage_settings": {
                "outputs_dir": "backend/outputs",
                "thumbnails_dir": "backend/outputs/thumbnails",
                "temp_dir": "backend/temp",
                "max_storage_gb": 50,
            },
            "logging_settings": {
                "log_level": "INFO",
                "log_file": "backend/logs/app.log",
                "enable_performance_logging": True,
            }
        }
        
        for section_name, section_defaults in defaults.items():
            if section_name not in config:
                config[section_name] = {}
            
            for key, default_value in section_defaults.items():
                if key not in config[section_name]:
                    config[section_name][key] = default_value
    
    def _handle_special_migrations(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        """Handle special migration cases that don't fit the simple mapping."""
        
        # Convert quantization boolean to string mode
        if "quantization" in old_config:
            if isinstance(old_config["quantization"], bool):
                new_config["model_settings"]["quantization_mode"] = "fp16" if old_config["quantization"] else "none"
        
        # Handle model type mapping
        model_type_mapping = {
            "text2video": "T2V-A14B",
            "image2video": "I2V-A14B",
            "textimage2video": "TI2V-5B",
            "t2v": "T2V-A14B",
            "i2v": "I2V-A14B",
            "ti2v": "TI2V-5B"
        }
        
        if "model_type" in old_config:
            old_model = old_config["model_type"].lower()
            if old_model in model_type_mapping:
                new_config["model_settings"]["default_model"] = model_type_mapping[old_model]
        
        # Convert paths to be relative to new structure
        if "output_dir" in old_config:
            old_path = old_config["output_dir"]
            if not old_path.startswith("backend/"):
                new_config["storage_settings"]["outputs_dir"] = f"backend/{old_path}"
    
    def validate_paths(self, config: Dict[str, Any]) -> bool:
        """Validate that configured paths exist or can be created."""
        is_valid = True
        
        paths_to_check = [
            ("storage_settings", "outputs_dir"),
            ("storage_settings", "temp_dir"),
            ("model_settings", "model_cache_dir"),
        ]
        
        for section, key in paths_to_check:
            if section in config and key in config[section]:
                path = Path(config[section][key])
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Validated path: {path}")
                except Exception as e:
                    error_msg = f"Cannot create directory {path}: {e}"
                    self.validation_result.errors.append(error_msg)
                    is_valid = False
        
        return is_valid
    
    def validate_model_compatibility(self, config: Dict[str, Any]) -> bool:
        """Validate that model settings are compatible with existing models."""
        is_valid = True
        
        if "model_settings" in config:
            model_dir = Path(config["model_settings"].get("model_cache_dir", "models"))
            
            if model_dir.exists():
                # Check for existing model files
                model_files = list(model_dir.rglob("*.safetensors")) + list(model_dir.rglob("*.bin"))
                
                if model_files:
                    self.validation_result.suggestions.append(
                        f"Found {len(model_files)} existing model files in {model_dir}"
                    )
                else:
                    self.validation_result.warnings.append(
                        f"No model files found in {model_dir}. Models may need to be downloaded."
                    )
            else:
                self.validation_result.warnings.append(
                    f"Model directory {model_dir} does not exist"
                )
        
        return is_valid
    
    def run_validation(self) -> ConfigValidationResult:
        """Run complete configuration validation."""
        logger.info("Starting configuration validation")
        
        # Load existing config
        old_config = self.load_existing_config()
        
        if old_config is None:
            # Create default config if none exists
            self.validation_result.migrated_config = {}
            self._set_config_defaults(self.validation_result.migrated_config)
            self.validation_result.suggestions.append(
                "Created default configuration. Please review and adjust as needed."
            )
        else:
            # Migrate existing config
            migrated_config = self.migrate_config(old_config)
            
            # Validate migrated config
            structure_valid = self.validate_config_structure(migrated_config)
            paths_valid = self.validate_paths(migrated_config)
            models_valid = self.validate_model_compatibility(migrated_config)
            
            self.validation_result.is_valid = structure_valid and paths_valid and models_valid
            self.validation_result.migrated_config = migrated_config
        
        # Generate suggestions
        if not self.validation_result.errors:
            self.validation_result.suggestions.append(
                "Configuration validation completed successfully"
            )
        
        if self.validation_result.warnings:
            self.validation_result.suggestions.append(
                "Review warnings and update configuration as needed"
            )
        
        logger.info(f"Configuration validation completed. Valid: {self.validation_result.is_valid}")
        return self.validation_result
    
    def save_migrated_config(self, output_path: str = "config_migrated.json") -> bool:
        """Save migrated configuration to file."""
        if not self.validation_result.migrated_config:
            logger.error("No migrated configuration to save")
            return False
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.validation_result.migrated_config, f, indent=2)
            
            logger.info(f"Migrated configuration saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save migrated configuration: {e}")
            return False

def validate_config_cli():
    """CLI entry point for configuration validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate and migrate configuration')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--output', default='config_migrated.json', help='Output path for migrated config')
    parser.add_argument('--validate-only', action='store_true', help='Only validate, do not migrate')
    
    args = parser.parse_args()
    
    validator = ConfigValidator(args.config)
    result = validator.run_validation()
    
    print(f"Configuration validation: {'PASSED' if result.is_valid else 'FAILED'}")
    
    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  ❌ {error}")
    
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  ⚠️  {warning}")
    
    if result.suggestions:
        print("\nSuggestions:")
        for suggestion in result.suggestions:
            print(f"  💡 {suggestion}")
    
    if not args.validate_only and result.migrated_config:
        if validator.save_migrated_config(args.output):
            print(f"\nMigrated configuration saved to {args.output}")

if __name__ == "__main__":
    validate_config_cli()