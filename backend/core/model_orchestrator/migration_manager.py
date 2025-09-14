"""
Migration Manager - Configuration migration and backward compatibility tools.

This module provides tools for migrating from legacy configuration formats
to the new model orchestrator system, with backward compatibility adapters
and gradual rollout support.
"""

import json
import os
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for older Python versions

try:
    import tomli_w  # For writing TOML files
except ImportError:
    tomli_w = None

from .model_registry import ModelRegistry, ModelSpec, FileSpec, ModelDefaults, VramEstimation
from .exceptions import MigrationError, ValidationError


logger = logging.getLogger(__name__)


@dataclass
class LegacyConfig:
    """Legacy configuration structure from config.json."""
    
    system: Dict[str, Any]
    directories: Dict[str, str]
    models: Dict[str, str]
    optimization: Dict[str, Any]


@dataclass
class MigrationResult:
    """Result of a configuration migration."""
    
    success: bool
    manifest_path: str
    backup_path: Optional[str] = None
    warnings: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


@dataclass
class FeatureFlags:
    """Feature flags for gradual rollout of model orchestrator."""
    
    enable_orchestrator: bool = False
    enable_manifest_validation: bool = True
    enable_legacy_fallback: bool = True
    enable_path_migration: bool = False
    enable_automatic_download: bool = False
    strict_validation: bool = False
    
    @classmethod
    def from_env(cls) -> 'FeatureFlags':
        """Create feature flags from environment variables."""
        return cls(
            enable_orchestrator=os.getenv('WAN_ENABLE_ORCHESTRATOR', 'false').lower() == 'true',
            enable_manifest_validation=os.getenv('WAN_ENABLE_MANIFEST_VALIDATION', 'true').lower() == 'true',
            enable_legacy_fallback=os.getenv('WAN_ENABLE_LEGACY_FALLBACK', 'true').lower() == 'true',
            enable_path_migration=os.getenv('WAN_ENABLE_PATH_MIGRATION', 'false').lower() == 'true',
            enable_automatic_download=os.getenv('WAN_ENABLE_AUTO_DOWNLOAD', 'false').lower() == 'true',
            strict_validation=os.getenv('WAN_STRICT_VALIDATION', 'false').lower() == 'true',
        )


class LegacyPathAdapter:
    """Adapter for resolving legacy model paths to orchestrator paths."""
    
    def __init__(self, legacy_models_dir: str, orchestrator_models_root: str):
        """
        Initialize the legacy path adapter.
        
        Args:
            legacy_models_dir: Legacy models directory path
            orchestrator_models_root: New orchestrator models root path
        """
        self.legacy_models_dir = Path(legacy_models_dir)
        self.orchestrator_models_root = Path(orchestrator_models_root)
        self._path_mappings: Dict[str, str] = {}
        
    def map_legacy_path(self, legacy_model_name: str) -> Optional[str]:
        """
        Map a legacy model name to the new orchestrator path.
        
        Args:
            legacy_model_name: Legacy model name (e.g., "t2v-A14B")
            
        Returns:
            New orchestrator path or None if no mapping exists
        """
        # Map legacy model names to new model IDs
        legacy_to_new_mapping = {
            "t2v-A14B": "t2v-A14B@2.2.0",
            "i2v-A14B": "i2v-A14B@2.2.0", 
            "ti2v-5B": "ti2v-5b@2.2.0",
            "ti2v-5b": "ti2v-5b@2.2.0",
        }
        
        new_model_id = legacy_to_new_mapping.get(legacy_model_name)
        if not new_model_id:
            return None
            
        # Return the new orchestrator path
        return str(self.orchestrator_models_root / "wan22" / new_model_id)
    
    def get_legacy_path(self, legacy_model_name: str) -> str:
        """Get the legacy path for a model."""
        return str(self.legacy_models_dir / legacy_model_name)
    
    def path_exists_in_legacy(self, legacy_model_name: str) -> bool:
        """Check if a model exists in the legacy location."""
        legacy_path = Path(self.get_legacy_path(legacy_model_name))
        return legacy_path.exists() and legacy_path.is_dir()
    
    def migrate_model_files(self, legacy_model_name: str, dry_run: bool = False) -> bool:
        """
        Migrate model files from legacy location to orchestrator location.
        
        Args:
            legacy_model_name: Legacy model name
            dry_run: If True, only simulate the migration
            
        Returns:
            True if migration was successful or would be successful
        """
        legacy_path = Path(self.get_legacy_path(legacy_model_name))
        new_path_str = self.map_legacy_path(legacy_model_name)
        
        if not new_path_str or not legacy_path.exists():
            return False
            
        new_path = Path(new_path_str)
        
        if dry_run:
            logger.info(f"Would migrate {legacy_path} -> {new_path}")
            return True
            
        try:
            # Create parent directories
            new_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the entire model directory
            if new_path.exists():
                logger.warning(f"Target path already exists: {new_path}")
                return False
                
            shutil.copytree(legacy_path, new_path)
            logger.info(f"Migrated model files: {legacy_path} -> {new_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate model files: {e}")
            return False


class ConfigurationMigrator:
    """Tool for migrating legacy configurations to model orchestrator format."""
    
    def __init__(self):
        """Initialize the configuration migrator."""
        self.logger = logging.getLogger(__name__ + ".ConfigurationMigrator")
        
    def load_legacy_config(self, config_path: str) -> LegacyConfig:
        """
        Load legacy configuration from config.json.
        
        Args:
            config_path: Path to the legacy config.json file
            
        Returns:
            LegacyConfig object
            
        Raises:
            MigrationError: If config cannot be loaded or parsed
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise MigrationError(f"Legacy config file not found: {config_path}")
            
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
                
            return LegacyConfig(
                system=data.get('system', {}),
                directories=data.get('directories', {}),
                models=data.get('models', {}),
                optimization=data.get('optimization', {})
            )
            
        except Exception as e:
            raise MigrationError(f"Failed to parse legacy config: {e}")
    
    def generate_manifest_from_legacy(self, legacy_config: LegacyConfig) -> Dict[str, Any]:
        """
        Generate a models.toml manifest structure from legacy configuration.
        
        Args:
            legacy_config: Legacy configuration object
            
        Returns:
            Dictionary representing the new manifest structure
        """
        manifest = {
            "schema_version": 1,
            "models": {}
        }
        
        # Map legacy model names to new specifications
        model_mappings = {
            "t2v_model": {
                "model_id": "t2v-A14B@2.2.0",
                "description": "WAN2.2 Text-to-Video A14B Model",
                "model_type": "t2v",
                "required_components": ["text_encoder", "unet", "vae"]
            },
            "i2v_model": {
                "model_id": "i2v-A14B@2.2.0", 
                "description": "WAN2.2 Image-to-Video A14B Model",
                "model_type": "i2v",
                "required_components": ["image_encoder", "unet", "vae"]
            },
            "ti2v_model": {
                "model_id": "ti2v-5b@2.2.0",
                "description": "WAN2.2 Text+Image-to-Video 5B Model", 
                "model_type": "ti2v",
                "required_components": ["text_encoder", "image_encoder", "unet", "vae"]
            }
        }
        
        # Extract default quantization from legacy config
        default_quantization = legacy_config.optimization.get('default_quantization', 'fp16')
        
        # Generate model specifications
        for legacy_key, mapping in model_mappings.items():
            legacy_model_name = legacy_config.models.get(legacy_key)
            if not legacy_model_name:
                continue
                
            model_id = mapping["model_id"]
            
            # Create basic model specification
            model_spec = {
                "description": mapping["description"],
                "version": "2.2.0",
                "variants": ["fp16", "bf16"],
                "default_variant": default_quantization,
                "resolution_caps": ["720p24", "1080p24"],
                "optional_components": [],
                "lora_required": False,
                "allow_patterns": ["*.safetensors", "*.json", "*.pth", "*.model"],
                "required_components": mapping["required_components"],
                "defaults": {
                    "fps": 24,
                    "frames": 16,
                    "scheduler": "ddim",
                    "precision": default_quantization,
                    "guidance_scale": 7.5,
                    "num_inference_steps": 50
                },
                "vram_estimation": {
                    "params_billion": 14.0 if "A14B" in model_id else 5.0,
                    "family_size": "large" if "A14B" in model_id else "medium",
                    "base_vram_gb": legacy_config.optimization.get('max_vram_usage_gb', 12),
                    "per_frame_vram_mb": 256.0 if "A14B" in model_id else 200.0
                },
                "files": [],  # Will be populated by scanning existing files
                "sources": {
                    "priority": [
                        f"local://wan22/{model_id}",
                        f"s3://ai-models/wan22/{model_id}",
                        f"hf://Wan-AI/Wan2.2-{mapping['model_type'].upper()}-{'A14B' if 'A14B' in model_id else '5B'}"
                    ]
                }
            }
            
            # Add model-specific defaults
            if mapping["model_type"] == "i2v":
                model_spec["defaults"]["image_guidance_scale"] = 1.5
                model_spec["optional_components"] = ["text_encoder"]
            elif mapping["model_type"] == "ti2v":
                model_spec["defaults"]["image_guidance_scale"] = 1.2
                model_spec["defaults"]["text_guidance_scale"] = 7.5
                model_spec["defaults"]["frames"] = 24
            
            manifest["models"][model_id] = model_spec
            
        return manifest
    
    def scan_legacy_model_files(self, models_dir: str, model_name: str) -> List[Dict[str, Any]]:
        """
        Scan legacy model directory to generate file specifications.
        
        Args:
            models_dir: Legacy models directory
            model_name: Model name to scan
            
        Returns:
            List of file specifications
        """
        model_path = Path(models_dir) / model_name
        if not model_path.exists():
            return []
            
        files = []
        
        # Recursively scan for model files
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(model_path)
                file_size = file_path.stat().st_size
                
                # Determine component type based on file path/name
                component = self._determine_component_type(str(relative_path))
                
                # Generate placeholder SHA256 (would need actual computation in real migration)
                placeholder_sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                
                file_spec = {
                    "path": str(relative_path),
                    "size": file_size,
                    "sha256": placeholder_sha256,
                    "component": component
                }
                
                # Check if it's a shard file
                if "diffusion_pytorch_model-" in str(relative_path) and ".safetensors" in str(relative_path):
                    # Extract shard number
                    import re
                    match = re.search(r'-(\d+)-of-\d+\.safetensors', str(relative_path))
                    if match:
                        file_spec["shard_part"] = int(match.group(1))
                elif "index.json" in str(relative_path):
                    file_spec["shard_index"] = True
                    
                files.append(file_spec)
                
        return files
    
    def _determine_component_type(self, file_path: str) -> Optional[str]:
        """Determine the component type based on file path."""
        file_path_lower = file_path.lower()
        
        if "vae" in file_path_lower:
            return "vae"
        elif "text_encoder" in file_path_lower or "t5" in file_path_lower or "umt5" in file_path_lower:
            return "text_encoder"
        elif "image_encoder" in file_path_lower:
            return "image_encoder"
        elif "unet" in file_path_lower or "diffusion_pytorch_model" in file_path_lower:
            return "unet"
        elif "config" in file_path_lower:
            return "config"
        else:
            return None
    
    def write_manifest(self, manifest_data: Dict[str, Any], output_path: str) -> None:
        """
        Write manifest data to a TOML file.
        
        Args:
            manifest_data: Manifest dictionary
            output_path: Output file path
            
        Raises:
            MigrationError: If writing fails
        """
        if tomli_w is None:
            raise MigrationError("tomli_w package required for writing TOML files")
            
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'wb') as f:
                tomli_w.dump(manifest_data, f)
                
            self.logger.info(f"Manifest written to: {output_path}")
            
        except Exception as e:
            raise MigrationError(f"Failed to write manifest: {e}")
    
    def migrate_configuration(
        self, 
        legacy_config_path: str,
        output_manifest_path: str,
        legacy_models_dir: Optional[str] = None,
        backup: bool = True,
        scan_files: bool = False
    ) -> MigrationResult:
        """
        Perform complete configuration migration.
        
        Args:
            legacy_config_path: Path to legacy config.json
            output_manifest_path: Path for new models.toml
            legacy_models_dir: Legacy models directory (for file scanning)
            backup: Whether to create backup of existing files
            scan_files: Whether to scan existing model files
            
        Returns:
            MigrationResult with details of the migration
        """
        result = MigrationResult(
            success=False,
            manifest_path=output_manifest_path
        )
        
        try:
            # Create backup if requested
            if backup and Path(output_manifest_path).exists():
                backup_path = f"{output_manifest_path}.backup"
                shutil.copy2(output_manifest_path, backup_path)
                result.backup_path = backup_path
                result.warnings.append(f"Created backup: {backup_path}")
            
            # Load legacy configuration
            legacy_config = self.load_legacy_config(legacy_config_path)
            
            # Generate manifest structure
            manifest_data = self.generate_manifest_from_legacy(legacy_config)
            
            # Scan existing model files if requested
            if scan_files and legacy_models_dir:
                for model_key in legacy_config.models:
                    model_name = legacy_config.models[model_key]
                    files = self.scan_legacy_model_files(legacy_models_dir, model_name)
                    
                    # Find corresponding model in manifest
                    for model_id, model_spec in manifest_data["models"].items():
                        if model_name.lower() in model_id.lower():
                            model_spec["files"] = files
                            break
            
            # Write the new manifest
            self.write_manifest(manifest_data, output_manifest_path)
            
            result.success = True
            result.warnings.append("Migration completed successfully")
            
            # Add validation warnings
            if not scan_files:
                result.warnings.append("File specifications not generated - run with scan_files=True to include actual files")
            
        except Exception as e:
            result.errors.append(str(e))
            self.logger.error(f"Migration failed: {e}")
            
        return result


class ManifestValidator:
    """Validator for model manifest files and configurations."""
    
    def __init__(self):
        """Initialize the manifest validator."""
        self.logger = logging.getLogger(__name__ + ".ManifestValidator")
        
    def validate_manifest_file(self, manifest_path: str) -> List[ValidationError]:
        """
        Validate a manifest file for correctness and completeness.
        
        Args:
            manifest_path: Path to the manifest file
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            # Try to load with ModelRegistry (includes built-in validation)
            registry = ModelRegistry(manifest_path)
            
            # Additional custom validations
            manifest_errors = self._validate_manifest_structure(manifest_path)
            errors.extend(manifest_errors)
            
        except Exception as e:
            errors.append(ValidationError(f"Failed to load manifest: {e}"))
            
        return errors
    
    def _validate_manifest_structure(self, manifest_path: str) -> List[ValidationError]:
        """Validate manifest structure and content."""
        errors = []
        
        try:
            with open(manifest_path, 'rb') as f:
                data = tomllib.load(f)
                
            # Check required top-level fields
            if 'schema_version' not in data:
                errors.append(ValidationError("Missing required field: schema_version"))
                
            if 'models' not in data:
                errors.append(ValidationError("Missing required field: models"))
                return errors
                
            # Validate each model
            for model_id, model_data in data['models'].items():
                model_errors = self._validate_model_structure(model_id, model_data)
                errors.extend(model_errors)
                
        except Exception as e:
            errors.append(ValidationError(f"Failed to parse manifest: {e}"))
            
        return errors
    
    def _validate_model_structure(self, model_id: str, model_data: Dict[str, Any]) -> List[ValidationError]:
        """Validate individual model structure."""
        errors = []
        
        # Required fields
        required_fields = ['version', 'variants', 'default_variant', 'files', 'sources']
        for field in required_fields:
            if field not in model_data:
                errors.append(ValidationError(f"Model {model_id}: Missing required field '{field}'"))
        
        # Validate files structure
        if 'files' in model_data:
            for i, file_data in enumerate(model_data['files']):
                if not isinstance(file_data, dict):
                    errors.append(ValidationError(f"Model {model_id}: File {i} must be a dictionary"))
                    continue
                    
                file_required = ['path', 'size', 'sha256']
                for field in file_required:
                    if field not in file_data:
                        errors.append(ValidationError(f"Model {model_id}: File {i} missing required field '{field}'"))
        
        return errors
    
    def validate_configuration_compatibility(
        self, 
        manifest_path: str, 
        legacy_config_path: Optional[str] = None
    ) -> List[ValidationError]:
        """
        Validate compatibility between new manifest and legacy configuration.
        
        Args:
            manifest_path: Path to new manifest
            legacy_config_path: Path to legacy config (optional)
            
        Returns:
            List of compatibility errors
        """
        errors = []
        
        try:
            # Load new manifest
            registry = ModelRegistry(manifest_path)
            new_models = set(registry.list_models())
            
            # Load legacy config if provided
            if legacy_config_path and Path(legacy_config_path).exists():
                migrator = ConfigurationMigrator()
                legacy_config = migrator.load_legacy_config(legacy_config_path)
                
                # Check if all legacy models are covered
                legacy_models = set(legacy_config.models.values())
                
                # Map legacy names to new IDs
                legacy_to_new = {
                    "t2v-A14B": "t2v-A14B@2.2.0",
                    "i2v-A14B": "i2v-A14B@2.2.0",
                    "ti2v-5B": "ti2v-5b@2.2.0",
                    "ti2v-5b": "ti2v-5b@2.2.0",
                }
                
                for legacy_model in legacy_models:
                    expected_new_id = legacy_to_new.get(legacy_model)
                    if expected_new_id and expected_new_id not in new_models:
                        errors.append(ValidationError(
                            f"Legacy model '{legacy_model}' not found in new manifest as '{expected_new_id}'"
                        ))
                        
        except Exception as e:
            errors.append(ValidationError(f"Compatibility validation failed: {e}"))
            
        return errors


class RollbackManager:
    """Manager for rolling back migrations and configurations."""
    
    def __init__(self):
        """Initialize the rollback manager."""
        self.logger = logging.getLogger(__name__ + ".RollbackManager")
        
    def create_rollback_point(self, config_paths: List[str], rollback_dir: str) -> str:
        """
        Create a rollback point by backing up current configurations.
        
        Args:
            config_paths: List of configuration file paths to backup
            rollback_dir: Directory to store rollback data
            
        Returns:
            Rollback point identifier
        """
        import time
        
        rollback_id = f"rollback_{int(time.time())}"
        rollback_path = Path(rollback_dir) / rollback_id
        rollback_path.mkdir(parents=True, exist_ok=True)
        
        # Create rollback metadata
        metadata = {
            "rollback_id": rollback_id,
            "timestamp": time.time(),
            "backed_up_files": []
        }
        
        # Backup each configuration file
        for config_path in config_paths:
            config_file = Path(config_path)
            if config_file.exists():
                backup_file = rollback_path / config_file.name
                shutil.copy2(config_file, backup_file)
                metadata["backed_up_files"].append({
                    "original_path": str(config_file),
                    "backup_path": str(backup_file)
                })
                
        # Save metadata
        metadata_file = rollback_path / "rollback_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"Created rollback point: {rollback_id}")
        return rollback_id
    
    def execute_rollback(self, rollback_id: str, rollback_dir: str) -> bool:
        """
        Execute a rollback to a previous configuration state.
        
        Args:
            rollback_id: Rollback point identifier
            rollback_dir: Directory containing rollback data
            
        Returns:
            True if rollback was successful
        """
        rollback_path = Path(rollback_dir) / rollback_id
        metadata_file = rollback_path / "rollback_metadata.json"
        
        if not metadata_file.exists():
            self.logger.error(f"Rollback metadata not found: {rollback_id}")
            return False
            
        try:
            # Load rollback metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            # Restore each backed up file
            for file_info in metadata["backed_up_files"]:
                original_path = Path(file_info["original_path"])
                backup_path = Path(file_info["backup_path"])
                
                if backup_path.exists():
                    # Create parent directories if needed
                    original_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Restore the file
                    shutil.copy2(backup_path, original_path)
                    self.logger.info(f"Restored: {original_path}")
                else:
                    self.logger.warning(f"Backup file not found: {backup_path}")
                    
            self.logger.info(f"Rollback completed: {rollback_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False
    
    def list_rollback_points(self, rollback_dir: str) -> List[Dict[str, Any]]:
        """
        List available rollback points.
        
        Args:
            rollback_dir: Directory containing rollback data
            
        Returns:
            List of rollback point information
        """
        rollback_points = []
        rollback_path = Path(rollback_dir)
        
        if not rollback_path.exists():
            return rollback_points
            
        for rollback_subdir in rollback_path.iterdir():
            if rollback_subdir.is_dir():
                metadata_file = rollback_subdir / "rollback_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        rollback_points.append(metadata)
                    except Exception as e:
                        self.logger.warning(f"Failed to load rollback metadata: {e}")
                        
        # Sort by timestamp (newest first)
        rollback_points.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        return rollback_points