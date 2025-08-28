#!/usr/bin/env python3
"""
Script to create the model configuration file.
"""

import os

# The complete model configuration implementation
MODEL_CONFIG_CONTENT = '''"""
Model Configuration System
Handles model path configuration, directory structure setup, file organization,
and metadata management for WAN2.2 models.
"""

import os
import json
import time
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

from .interfaces import InstallationError, ErrorCategory, HardwareProfile
from .base_classes import BaseInstallationComponent


class ModelStatus(Enum):
    """Status of a model in the system."""
    MISSING = "missing"
    DOWNLOADING = "downloading"
    AVAILABLE = "available"
    CORRUPTED = "corrupted"
    OUTDATED = "outdated"


class ModelType(Enum):
    """Types of WAN2.2 models."""
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    TEXT_IMAGE_TO_VIDEO = "text_image_to_video"


@dataclass
class ModelMetadata:
    """Metadata for a WAN2.2 model."""
    name: str
    model_type: ModelType
    version: str
    size_gb: float
    path: str
    files: List[str]
    checksum: str
    status: ModelStatus
    last_verified: Optional[float] = None
    download_date: Optional[float] = None
    configuration: Optional[Dict[str, Any]] = None


@dataclass
class ModelDirectoryStructure:
    """Standard directory structure for models."""
    models_root: Path
    model_dirs: Dict[str, Path]
    cache_dir: Path
    temp_dir: Path
    backup_dir: Path
    metadata_file: Path


class ModelConfigurationManager(BaseInstallationComponent):
    """
    Manages model configuration, directory structure, file organization,
    and metadata for WAN2.2 models.
    """
    
    # Standard model configuration
    MODEL_DEFINITIONS = {
        "WAN2.2-T2V-A14B": {
            "type": ModelType.TEXT_TO_VIDEO,
            "version": "v1.0.3",
            "size_gb": 28.5,
            "files": [
                "pytorch_model.bin",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json"
            ],
            "required_files": ["pytorch_model.bin", "config.json"],
            "optional_files": ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
        },
        "WAN2.2-I2V-A14B": {
            "type": ModelType.IMAGE_TO_VIDEO,
            "version": "v1.0.3", 
            "size_gb": 28.5,
            "files": [
                "pytorch_model.bin",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json"
            ],
            "required_files": ["pytorch_model.bin", "config.json"],
            "optional_files": ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
        },
        "WAN2.2-TI2V-5B": {
            "type": ModelType.TEXT_IMAGE_TO_VIDEO,
            "version": "v1.0.2",
            "size_gb": 10.2,
            "files": [
                "pytorch_model.bin",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json"
            ],
            "required_files": ["pytorch_model.bin", "config.json"],
            "optional_files": ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
        }
    }
    
    def __init__(self, installation_path: str, models_dir: Optional[str] = None):
        super().__init__(installation_path)
        self.models_root = Path(models_dir) if models_dir else self.installation_path / "models"
        self.directory_structure = self._setup_directory_structure()
        self.metadata_cache = {}
        self._load_metadata_cache()
    
    def _setup_directory_structure(self) -> ModelDirectoryStructure:
        """Set up the standard directory structure for models."""
        try:
            # Create main directories
            models_root = self.models_root
            cache_dir = models_root / ".cache"
            temp_dir = models_root / ".temp"
            backup_dir = models_root / ".backup"
            metadata_file = models_root / "model_metadata.json"
            
            # Ensure all directories exist
            for directory in [models_root, cache_dir, temp_dir, backup_dir]:
                self.ensure_directory(directory)
            
            # Create model-specific directories
            model_dirs = {}
            for model_name in self.MODEL_DEFINITIONS.keys():
                model_dir = models_root / model_name
                self.ensure_directory(model_dir)
                model_dirs[model_name] = model_dir
            
            structure = ModelDirectoryStructure(
                models_root=models_root,
                model_dirs=model_dirs,
                cache_dir=cache_dir,
                temp_dir=temp_dir,
                backup_dir=backup_dir,
                metadata_file=metadata_file
            )
            
            self.logger.info(f"Set up model directory structure at {models_root}")
            return structure
            
        except Exception as e:
            raise InstallationError(
                f"Failed to set up model directory structure: {str(e)}",
                ErrorCategory.SYSTEM,
                ["Check directory permissions", "Ensure sufficient disk space"]
            )
    
    def configure_model_paths(self, config_path: str, hardware_profile: Optional[HardwareProfile] = None) -> bool:
        """
        Configure model paths in application configuration with hardware optimization.
        
        Args:
            config_path: Path to the application configuration file
            hardware_profile: Hardware profile for optimization
            
        Returns:
            bool: True if configuration was successful
        """
        try:
            config_file = Path(config_path)
            
            # Load existing config or create new one
            if config_file.exists():
                config = self.load_json_file(config_file)
            else:
                config = {}
            
            # Get current model status
            model_status = self.get_all_model_status()
            
            # Configure model paths
            model_paths = {}
            available_models = []
            
            for model_name, status in model_status.items():
                if status.status == ModelStatus.AVAILABLE:
                    model_paths[model_name] = str(status.path)
                    available_models.append(model_name)
            
            # Generate hardware-optimized model configuration
            model_config = self._generate_model_configuration(hardware_profile, available_models)
            
            # Update configuration
            config["models"] = {
                "model_paths": model_paths,
                "models_directory": str(self.models_root),
                "available_models": available_models,
                **model_config
            }
            
            # Save updated configuration
            self.save_json_file(config, config_file)
            
            self.logger.info(f"Configured model paths for {len(available_models)} models")
            return True
            
        except Exception as e:
            self.logger.error(f"Error configuring model paths: {str(e)}")
            raise InstallationError(
                f"Failed to configure model paths: {str(e)}",
                ErrorCategory.CONFIGURATION,
                ["Check configuration file permissions", "Verify models directory exists"]
            )
    
    def _generate_model_configuration(self, hardware_profile: Optional[HardwareProfile], 
                                    available_models: List[str]) -> Dict[str, Any]:
        """Generate hardware-optimized model configuration."""
        
        # Default configuration
        config = {
            "cache_models": False,
            "preload_models": False,
            "model_precision": "fp16",
            "max_models_in_memory": 1,
            "offload_to_cpu": True,
            "use_safetensors": True
        }
        
        if not hardware_profile:
            return config
        
        # High-performance systems (64+ GB RAM, 16+ GB VRAM)
        if (hardware_profile.memory.total_gb >= 64 and 
            hardware_profile.gpu and hardware_profile.gpu.vram_gb >= 16):
            config.update({
                "cache_models": True,
                "preload_models": True,
                "model_precision": "bf16",
                "max_models_in_memory": 2,
                "offload_to_cpu": False
            })
            self.logger.info("Applied high-performance model configuration")
        
        # Mid-range systems (32+ GB RAM, 8+ GB VRAM)
        elif (hardware_profile.memory.total_gb >= 32 and 
              hardware_profile.gpu and hardware_profile.gpu.vram_gb >= 8):
            config.update({
                "cache_models": True,
                "preload_models": False,
                "model_precision": "fp16",
                "max_models_in_memory": 1,
                "offload_to_cpu": True
            })
            self.logger.info("Applied mid-range model configuration")
        
        # Budget systems (16+ GB RAM, 4+ GB VRAM)
        else:
            config.update({
                "cache_models": False,
                "preload_models": False,
                "model_precision": "int8",
                "max_models_in_memory": 1,
                "offload_to_cpu": True
            })
            self.logger.info("Applied budget model configuration")
        
        return config
    
    def organize_model_files(self, model_name: str, source_files: List[str]) -> bool:
        """
        Organize model files into the standard directory structure.
        
        Args:
            model_name: Name of the model
            source_files: List of source file paths to organize
            
        Returns:
            bool: True if organization was successful
        """
        try:
            if model_name not in self.MODEL_DEFINITIONS:
                raise InstallationError(
                    f"Unknown model: {model_name}",
                    ErrorCategory.CONFIGURATION,
                    ["Check model name spelling", "Verify model is supported"]
                )
            
            model_dir = self.directory_structure.model_dirs[model_name]
            model_def = self.MODEL_DEFINITIONS[model_name]
            
            self.logger.info(f"Organizing files for model {model_name}")
            
            # Organize files by type
            organized_files = []
            for source_file in source_files:
                source_path = Path(source_file)
                if not source_path.exists():
                    self.logger.warning(f"Source file not found: {source_file}")
                    continue
                
                # Determine target filename
                target_filename = source_path.name
                if target_filename not in model_def["files"]:
                    self.logger.warning(f"Unexpected file for {model_name}: {target_filename}")
                
                target_path = model_dir / target_filename
                
                # Copy or move file to target location
                if source_path != target_path:
                    if target_path.exists():
                        # Backup existing file
                        backup_path = self.directory_structure.backup_dir / f"{model_name}_{target_filename}.backup"
                        shutil.copy2(target_path, backup_path)
                        self.logger.info(f"Backed up existing file: {target_filename}")
                    
                    shutil.copy2(source_path, target_path)
                    organized_files.append(str(target_path))
                    self.logger.debug(f"Organized file: {target_filename}")
            
            # Update metadata
            self._update_model_metadata(model_name, organized_files)
            
            self.logger.info(f"Successfully organized {len(organized_files)} files for {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error organizing model files: {str(e)}")
            raise InstallationError(
                f"Failed to organize model files for {model_name}: {str(e)}",
                ErrorCategory.SYSTEM,
                ["Check file permissions", "Ensure sufficient disk space"]
            )
    
    def validate_model_files(self, model_name: str) -> Tuple[bool, List[str]]:
        """
        Validate that all required model files are present and valid.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            if model_name not in self.MODEL_DEFINITIONS:
                issues.append(f"Unknown model: {model_name}")
                return False, issues
            
            model_dir = self.directory_structure.model_dirs[model_name]
            model_def = self.MODEL_DEFINITIONS[model_name]
            
            # Check if model directory exists
            if not model_dir.exists():
                issues.append(f"Model directory not found: {model_dir}")
                return False, issues
            
            # Check required files
            for required_file in model_def["required_files"]:
                file_path = model_dir / required_file
                if not file_path.exists():
                    issues.append(f"Required file missing: {required_file}")
                elif file_path.stat().st_size == 0:
                    issues.append(f"Required file is empty: {required_file}")
            
            # Check optional files (warn but don't fail)
            for optional_file in model_def["optional_files"]:
                file_path = model_dir / optional_file
                if not file_path.exists():
                    self.logger.warning(f"Optional file missing for {model_name}: {optional_file}")
            
            # Validate main model file integrity if no issues so far
            if not issues:
                main_model_file = model_dir / "pytorch_model.bin"
                if main_model_file.exists():
                    if not self._validate_model_file_integrity(main_model_file):
                        issues.append("Main model file failed integrity check")
            
            is_valid = len(issues) == 0
            
            if is_valid:
                self.logger.info(f"Model {model_name} validation passed")
            else:
                self.logger.warning(f"Model {model_name} validation failed: {issues}")
            
            return is_valid, issues
            
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
            self.logger.error(f"Error validating model {model_name}: {str(e)}")
            return False, issues
    
    def _validate_model_file_integrity(self, file_path: Path) -> bool:
        """Validate model file integrity using basic checks."""
        try:
            # Check file size (should be > 100MB for model files)
            file_size = file_path.stat().st_size
            if file_size < 100 * 1024 * 1024:  # 100MB
                self.logger.warning(f"Model file seems too small: {file_size} bytes")
                return False
            
            # Try to read first few bytes to ensure file is not corrupted
            with open(file_path, 'rb') as f:
                header = f.read(1024)
                if len(header) < 1024:
                    self.logger.warning("Model file appears to be truncated")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating file integrity: {str(e)}")
            return False
    
    def get_model_status(self, model_name: str) -> ModelMetadata:
        """Get the current status and metadata for a specific model."""
        
        if model_name not in self.MODEL_DEFINITIONS:
            return ModelMetadata(
                name=model_name,
                model_type=ModelType.TEXT_TO_VIDEO,  # Default
                version="unknown",
                size_gb=0.0,
                path="",
                files=[],
                checksum="",
                status=ModelStatus.MISSING
            )
        
        model_def = self.MODEL_DEFINITIONS[model_name]
        model_dir = self.directory_structure.model_dirs[model_name]
        
        # Check if model exists and is valid
        is_valid, issues = self.validate_model_files(model_name)
        
        if is_valid:
            status = ModelStatus.AVAILABLE
        elif model_dir.exists():
            status = ModelStatus.CORRUPTED
        else:
            status = ModelStatus.MISSING
        
        # Get metadata from cache or create new
        metadata = self.metadata_cache.get(model_name)
        if not metadata:
            metadata = ModelMetadata(
                name=model_name,
                model_type=model_def["type"],
                version=model_def["version"],
                size_gb=model_def["size_gb"],
                path=str(model_dir),
                files=model_def["files"],
                checksum="",
                status=status
            )
        else:
            metadata.status = status
        
        return metadata
    
    def get_all_model_status(self) -> Dict[str, ModelMetadata]:
        """Get status for all defined models."""
        return {
            model_name: self.get_model_status(model_name)
            for model_name in self.MODEL_DEFINITIONS.keys()
        }
    
    def _update_model_metadata(self, model_name: str, organized_files: List[str]) -> None:
        """Update metadata for a specific model."""
        try:
            model_def = self.MODEL_DEFINITIONS[model_name]
            model_dir = self.directory_structure.model_dirs[model_name]
            
            # Calculate total size of organized files
            total_size = sum(
                Path(file_path).stat().st_size 
                for file_path in organized_files 
                if Path(file_path).exists()
            )
            size_gb = total_size / (1024 ** 3)
            
            # Create or update metadata
            metadata = ModelMetadata(
                name=model_name,
                model_type=model_def["type"],
                version=model_def["version"],
                size_gb=size_gb,
                path=str(model_dir),
                files=[Path(f).name for f in organized_files],
                checksum="",  # TODO: Calculate actual checksum
                status=ModelStatus.AVAILABLE,
                last_verified=time.time(),
                download_date=time.time()
            )
            
            # Update cache
            self.metadata_cache[model_name] = metadata
            
            # Save to persistent storage
            self._save_metadata_cache()
            
            self.logger.info(f"Updated metadata for model {model_name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to update metadata for {model_name}: {str(e)}")
    
    def _load_metadata_cache(self) -> None:
        """Load model metadata from persistent storage."""
        try:
            if self.directory_structure.metadata_file.exists():
                metadata_data = self.load_json_file(self.directory_structure.metadata_file)
                
                for model_name, data in metadata_data.get("models", {}).items():
                    # Convert dict back to ModelMetadata
                    metadata = ModelMetadata(
                        name=data["name"],
                        model_type=ModelType(data["model_type"]),
                        version=data["version"],
                        size_gb=data["size_gb"],
                        path=data["path"],
                        files=data["files"],
                        checksum=data["checksum"],
                        status=ModelStatus(data["status"]),
                        last_verified=data.get("last_verified"),
                        download_date=data.get("download_date"),
                        configuration=data.get("configuration")
                    )
                    self.metadata_cache[model_name] = metadata
                
                self.logger.info(f"Loaded metadata for {len(self.metadata_cache)} models")
        
        except Exception as e:
            self.logger.warning(f"Failed to load metadata cache: {str(e)}")
            self.metadata_cache = {}
    
    def _save_metadata_cache(self) -> None:
        """Save model metadata to persistent storage."""
        try:
            metadata_data = {
                "last_updated": time.time(),
                "version": "1.0",
                "models": {}
            }
            
            for model_name, metadata in self.metadata_cache.items():
                metadata_data["models"][model_name] = {
                    "name": metadata.name,
                    "model_type": metadata.model_type.value,
                    "version": metadata.version,
                    "size_gb": metadata.size_gb,
                    "path": metadata.path,
                    "files": metadata.files,
                    "checksum": metadata.checksum,
                    "status": metadata.status.value,
                    "last_verified": metadata.last_verified,
                    "download_date": metadata.download_date,
                    "configuration": metadata.configuration
                }
            
            self.save_json_file(metadata_data, self.directory_structure.metadata_file)
            self.logger.debug("Saved metadata cache")
            
        except Exception as e:
            self.logger.warning(f"Failed to save metadata cache: {str(e)}")
    
    def cleanup_temporary_files(self) -> None:
        """Clean up temporary files and directories."""
        try:
            temp_dir = self.directory_structure.temp_dir
            if temp_dir.exists():
                for item in temp_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                        self.logger.debug(f"Removed temporary file: {item.name}")
                    elif item.is_dir():
                        shutil.rmtree(item)
                        self.logger.debug(f"Removed temporary directory: {item.name}")
                
                self.logger.info("Cleaned up temporary files")
        
        except Exception as e:
            self.logger.warning(f"Error cleaning up temporary files: {str(e)}")
    
    def backup_model_configuration(self) -> str:
        """
        Create a backup of the current model configuration.
        
        Returns:
            str: Path to the backup file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"model_config_backup_{timestamp}.json"
            backup_path = self.directory_structure.backup_dir / backup_filename
            
            # Create backup data
            backup_data = {
                "timestamp": time.time(),
                "models_directory": str(self.models_root),
                "metadata": {}
            }
            
            # Include current metadata
            for model_name, metadata in self.metadata_cache.items():
                backup_data["metadata"][model_name] = asdict(metadata)
            
            # Save backup
            self.save_json_file(backup_data, backup_path)
            
            self.logger.info(f"Created model configuration backup: {backup_filename}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {str(e)}")
            raise InstallationError(
                f"Failed to create model configuration backup: {str(e)}",
                ErrorCategory.SYSTEM,
                ["Check backup directory permissions", "Ensure sufficient disk space"]
            )
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of all models and their status."""
        all_status = self.get_all_model_status()
        
        summary = {
            "total_models": len(all_status),
            "available_models": sum(1 for s in all_status.values() if s.status == ModelStatus.AVAILABLE),
            "missing_models": sum(1 for s in all_status.values() if s.status == ModelStatus.MISSING),
            "corrupted_models": sum(1 for s in all_status.values() if s.status == ModelStatus.CORRUPTED),
            "total_size_gb": sum(s.size_gb for s in all_status.values() if s.status == ModelStatus.AVAILABLE),
            "models_directory": str(self.models_root),
            "models": {name: {
                "status": status.status.value,
                "size_gb": status.size_gb,
                "version": status.version,
                "type": status.model_type.value
            } for name, status in all_status.items()}
        }
        
        return summary


def create_model_configuration_manager(installation_path: str, **kwargs) -> ModelConfigurationManager:
    """Factory function to create a ModelConfigurationManager instance."""
    return ModelConfigurationManager(installation_path, **kwargs)
'''

def main():
    """Create the model configuration file."""
    target_file = os.path.join("scripts", "model_configuration.py")
    
    print(f"Creating {target_file}...")
    
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(MODEL_CONFIG_CONTENT)
    
    print(f"✅ Created {target_file} ({len(MODEL_CONFIG_CONTENT)} characters)")
    
    # Verify the file was created correctly
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if len(content) == len(MODEL_CONFIG_CONTENT):
        print("✅ File verification passed")
    else:
        print(f"❌ File verification failed: expected {len(MODEL_CONFIG_CONTENT)}, got {len(content)}")

if __name__ == "__main__":
    main()