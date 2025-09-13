"""
Model Configuration Validator

This module provides validation for model component configurations including VAE, 
text encoder, and model_index.json files. It handles automatic removal of unsupported 
attributes and model-library compatibility checking.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Set
from dataclasses import dataclass
import logging

from config_validator import (
    ValidationSeverity,
    ValidationMessage,
    ValidationResult,
    CleanupResult,
    ConfigValidator
)


@dataclass
class ModelCompatibilityInfo:
    """Information about model compatibility"""
    model_name: str
    pipeline_class: str
    supported_optimizations: List[str]
    vram_requirements: Dict[str, int]
    trust_remote_code: bool
    min_diffusers_version: str


class ModelConfigValidator(ConfigValidator):
    """
    Model configuration validator for WAN22 system.
    
    Extends ConfigValidator to handle model-specific configurations including
    VAE, text encoder, and pipeline configurations.
    """
    
    def __init__(self, backup_dir: Optional[Path] = None, compatibility_registry_path: Optional[Path] = None):
        """
        Initialize the model configuration validator.
        
        Args:
            backup_dir: Directory for configuration backups
            compatibility_registry_path: Path to compatibility registry file
        """
        super().__init__(backup_dir)
        
        self.compatibility_registry_path = compatibility_registry_path or Path("compatibility_registry.json")
        self.compatibility_registry = self._load_compatibility_registry()
        
        # Define model-specific schemas
        self.model_schemas = self._define_model_schemas()
        
        # Define model-specific cleanup attributes
        self.model_cleanup_attributes = self._define_model_cleanup_attributes()
    
    def _load_compatibility_registry(self) -> Dict[str, Any]:
        """Load the compatibility registry"""
        try:
            if self.compatibility_registry_path.exists():
                with open(self.compatibility_registry_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"Compatibility registry not found: {self.compatibility_registry_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load compatibility registry: {e}")
            return {}
    
    def _define_model_schemas(self) -> Dict[str, Any]:
        """Define expected schemas for model configurations"""
        return {
            "model_index": {
                "type": "object",
                "required": ["_class_name"],
                "properties": {
                    "_class_name": {"type": "string", "enum": ["WanPipeline", "WanImageToVideoPipeline", "StableDiffusionPipeline", "DiffusionPipeline"]},
                    "_diffusers_version": {"type": "string"},
                    "text_encoder": {"type": "array", "items": {"type": "string"}},
                    "tokenizer": {"type": "array", "items": {"type": "string"}},
                    "unet": {"type": "array", "items": {"type": "string"}},
                    "vae": {"type": "array", "items": {"type": "string"}},
                    "scheduler": {"type": "array", "items": {"type": "string"}},
                    "safety_checker": {"type": ["array", "null"]},
                    "feature_extractor": {"type": ["array", "null"]}
                }
            },
            "vae_config": {
                "type": "object",
                "properties": {
                    "_class_name": {"type": "string"},
                    "_diffusers_version": {"type": "string"},
                    "act_fn": {"type": "string"},
                    "block_out_channels": {"type": "array", "items": {"type": "integer"}},
                    "down_block_types": {"type": "array", "items": {"type": "string"}},
                    "in_channels": {"type": "integer", "minimum": 1, "maximum": 16},
                    "latent_channels": {"type": "integer", "minimum": 1, "maximum": 16},
                    "layers_per_block": {"type": "integer", "minimum": 1, "maximum": 10},
                    "norm_num_groups": {"type": "integer", "minimum": 1},
                    "out_channels": {"type": "integer", "minimum": 1, "maximum": 16},
                    "sample_size": {"type": "integer", "minimum": 32},
                    "scaling_factor": {"type": "number", "minimum": 0.1, "maximum": 2.0},
                    "up_block_types": {"type": "array", "items": {"type": "string"}}
                }
            },
            "text_encoder_config": {
                "type": "object",
                "properties": {
                    "_name_or_path": {"type": "string"},
                    "architectures": {"type": "array", "items": {"type": "string"}},
                    "attention_dropout": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "bos_token_id": {"type": "integer"},
                    "eos_token_id": {"type": "integer"},
                    "hidden_act": {"type": "string"},
                    "hidden_size": {"type": "integer", "minimum": 64},
                    "initializer_factor": {"type": "number"},
                    "initializer_range": {"type": "number"},
                    "intermediate_size": {"type": "integer", "minimum": 64},
                    "layer_norm_eps": {"type": "number"},
                    "max_position_embeddings": {"type": "integer", "minimum": 77},
                    "model_type": {"type": "string"},
                    "num_attention_heads": {"type": "integer", "minimum": 1},
                    "num_hidden_layers": {"type": "integer", "minimum": 1},
                    "pad_token_id": {"type": "integer"},
                    "projection_dim": {"type": "integer", "minimum": 64},
                    "torch_dtype": {"type": "string"},
                    "transformers_version": {"type": "string"},
                    "vocab_size": {"type": "integer", "minimum": 1000}
                }
            },
            "unet_config": {
                "type": "object",
                "properties": {
                    "_class_name": {"type": "string"},
                    "_diffusers_version": {"type": "string"},
                    "act_fn": {"type": "string"},
                    "attention_head_dim": {"type": ["integer", "array"]},
                    "block_out_channels": {"type": "array", "items": {"type": "integer"}},
                    "center_input_sample": {"type": "boolean"},
                    "cross_attention_dim": {"type": ["integer", "array"]},
                    "down_block_types": {"type": "array", "items": {"type": "string"}},
                    "downsample_padding": {"type": "integer"},
                    "flip_sin_to_cos": {"type": "boolean"},
                    "freq_shift": {"type": "integer"},
                    "in_channels": {"type": "integer", "minimum": 1},
                    "layers_per_block": {"type": "integer", "minimum": 1},
                    "mid_block_scale_factor": {"type": "number"},
                    "norm_eps": {"type": "number"},
                    "norm_num_groups": {"type": "integer"},
                    "out_channels": {"type": "integer", "minimum": 1},
                    "sample_size": {"type": ["integer", "array"]},
                    "up_block_types": {"type": "array", "items": {"type": "string"}},
                    "use_linear_projection": {"type": "boolean"}
                }
            }
        }
    
    def _define_model_cleanup_attributes(self) -> Dict[str, Set[str]]:
        """Define model-specific attributes that should be cleaned up"""
        return {
            "vae_config": {
                "clip_output",  # Main issue - unsupported in AutoencoderKLWan
                "force_upcast",  # Legacy attribute
                "use_tiling",  # Should be handled by tile_size parameter
                "enable_slicing",  # Should be handled by attention slicing
                "slice_size",  # Deprecated parameter
            },
            "text_encoder_config": {
                "use_attention_mask",  # Model handles this internally
                "return_dict",  # Internal parameter
                "output_attentions",  # Internal parameter
                "output_hidden_states",  # Internal parameter
                "torchscript",  # Compilation parameter
                "use_cache",  # Internal caching parameter
            },
            "unet_config": {
                "use_linear_projection",  # Should be determined by model architecture
                "only_cross_attention",  # Architecture-specific parameter
                "dual_cross_attention",  # Architecture-specific parameter
                "encoder_hid_dim",  # Legacy parameter
                "encoder_hid_dim_type",  # Legacy parameter
            },
            "model_index": {
                "requires_safety_checker",  # Legacy safety checker
                "safety_checker",  # Should be handled separately
                "feature_extractor",  # Deprecated in newer versions
                "_name_or_path",  # Loading parameter, not config
                "cache_dir",  # System parameter
                "local_files_only",  # Loading parameter
                "revision",  # Model version parameter
                "use_safetensors",  # Loading parameter
                "variant",  # Model loading parameter
                "torch_dtype",  # Should be handled by quantization system
            },
            "scheduler_config": {
                "clip_sample",  # Model-specific parameter
                "set_alpha_to_one",  # Legacy parameter
                "skip_prk_steps",  # Scheduler-specific parameter
                "steps_offset",  # Legacy parameter
            }
        }
    
    def validate_model_index(self, model_index_path: Union[str, Path]) -> ValidationResult:
        """
        Validate a model_index.json file.
        
        Args:
            model_index_path: Path to model_index.json file
            
        Returns:
            ValidationResult with validation messages and cleanup info
        """
        model_index_path = Path(model_index_path)
        messages = []
        cleaned_attributes = []
        backup_path = None
        
        try:
            # Check if file exists
            if not model_index_path.exists():
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.CRITICAL,
                    code="MODEL_INDEX_NOT_FOUND",
                    message=f"model_index.json not found: {model_index_path}",
                    field_path=str(model_index_path),
                    help_text="Ensure the model directory contains a valid model_index.json file"
                ))
                return ValidationResult(
                    is_valid=False,
                    messages=messages,
                    cleaned_attributes=cleaned_attributes,
                    backup_path=backup_path
                )
            
            # Load model index
            with open(model_index_path, 'r', encoding='utf-8') as f:
                model_index = json.load(f)
            
            # Create backup
            backup_path = self.create_backup(model_index_path)
            
            # Validate schema
            schema_messages = self._validate_schema(
                {"model_index": model_index}, 
                {"model_index": self.model_schemas["model_index"]},
                "model_index"
            )
            messages.extend(schema_messages)
            
            # Validate pipeline class compatibility (only check registry compatibility, not physical components)
            pipeline_class = model_index.get("_class_name")
            if pipeline_class:
                compatibility_messages = self._validate_pipeline_class_compatibility(pipeline_class)
                messages.extend(compatibility_messages)
            
            # Clean up model index
            cleanup_result = self._cleanup_model_config(model_index, "model_index")
            cleaned_attributes.extend(cleanup_result.cleaned_attributes)
            
            if cleanup_result.cleaned_attributes:
                # Save cleaned model index
                with open(model_index_path, 'w', encoding='utf-8') as f:
                    json.dump(model_index, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"Cleaned model_index.json saved to {model_index_path}")
            
            # Determine if valid
            is_valid = not any(msg.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                             for msg in messages)
            
            return ValidationResult(
                is_valid=is_valid,
                messages=messages,
                cleaned_attributes=cleaned_attributes,
                backup_path=backup_path
            )
            
        except json.JSONDecodeError as e:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.CRITICAL,
                code="INVALID_JSON",
                message=f"Invalid JSON in model_index.json: {e}",
                field_path=str(model_index_path),
                help_text="Fix JSON syntax errors in model_index.json"
            ))
            return ValidationResult(
                is_valid=False,
                messages=messages,
                cleaned_attributes=cleaned_attributes,
                backup_path=backup_path
            )
        except Exception as e:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.CRITICAL,
                code="MODEL_INDEX_VALIDATION_ERROR",
                message=f"Model index validation error: {e}",
                field_path=str(model_index_path)
            ))
            return ValidationResult(
                is_valid=False,
                messages=messages,
                cleaned_attributes=cleaned_attributes,
                backup_path=backup_path
            )
    
    def validate_component_config(self, config_path: Union[str, Path], component_type: str) -> ValidationResult:
        """
        Validate a model component configuration file (VAE, text encoder, etc.).
        
        Args:
            config_path: Path to component config.json file
            component_type: Type of component (vae_config, text_encoder_config, unet_config)
            
        Returns:
            ValidationResult with validation messages and cleanup info
        """
        config_path = Path(config_path)
        messages = []
        cleaned_attributes = []
        backup_path = None
        
        try:
            # Check if file exists
            if not config_path.exists():
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    code="COMPONENT_CONFIG_NOT_FOUND",
                    message=f"Component config not found: {config_path}",
                    field_path=str(config_path),
                    help_text=f"Ensure the {component_type} directory contains a config.json file"
                ))
                return ValidationResult(
                    is_valid=False,
                    messages=messages,
                    cleaned_attributes=cleaned_attributes,
                    backup_path=backup_path
                )
            
            # Load component config
            with open(config_path, 'r', encoding='utf-8') as f:
                component_config = json.load(f)
            
            # Create backup
            backup_path = self.create_backup(config_path)
            
            # Validate schema if available
            if component_type in self.model_schemas:
                schema_messages = self._validate_schema(
                    {component_type: component_config},
                    {component_type: self.model_schemas[component_type]},
                    component_type
                )
                messages.extend(schema_messages)
            
            # Clean up component config
            cleanup_result = self._cleanup_model_config(component_config, component_type)
            cleaned_attributes.extend(cleanup_result.cleaned_attributes)
            
            if cleanup_result.cleaned_attributes:
                # Save cleaned component config
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(component_config, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"Cleaned {component_type} config saved to {config_path}")
            
            # Determine if valid
            is_valid = not any(msg.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                             for msg in messages)
            
            return ValidationResult(
                is_valid=is_valid,
                messages=messages,
                cleaned_attributes=cleaned_attributes,
                backup_path=backup_path
            )
            
        except json.JSONDecodeError as e:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.CRITICAL,
                code="INVALID_JSON",
                message=f"Invalid JSON in {component_type} config: {e}",
                field_path=str(config_path),
                help_text=f"Fix JSON syntax errors in {component_type} config.json"
            ))
            return ValidationResult(
                is_valid=False,
                messages=messages,
                cleaned_attributes=cleaned_attributes,
                backup_path=backup_path
            )
        except Exception as e:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.CRITICAL,
                code="COMPONENT_CONFIG_VALIDATION_ERROR",
                message=f"Component config validation error: {e}",
                field_path=str(config_path)
            ))
            return ValidationResult(
                is_valid=False,
                messages=messages,
                cleaned_attributes=cleaned_attributes,
                backup_path=backup_path
            )
    
    def validate_model_directory(self, model_path: Union[str, Path]) -> ValidationResult:
        """
        Validate an entire model directory including model_index.json and component configs.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            ValidationResult with validation messages and cleanup info
        """
        model_path = Path(model_path)
        all_messages = []
        all_cleaned_attributes = []
        backup_paths = []
        
        # Validate model_index.json
        model_index_path = model_path / "model_index.json"
        index_result = self.validate_model_index(model_index_path)
        all_messages.extend(index_result.messages)
        all_cleaned_attributes.extend(index_result.cleaned_attributes)
        if index_result.backup_path:
            backup_paths.append(index_result.backup_path)
        
        # Additional validation for physical components if model_index exists
        if model_index_path.exists():
            try:
                with open(model_index_path, 'r', encoding='utf-8') as f:
                    model_index = json.load(f)
                
                pipeline_class = model_index.get("_class_name")
                if pipeline_class:
                    # Validate physical component requirements
                    component_messages = self._validate_pipeline_compatibility(pipeline_class, model_path)
                    all_messages.extend(component_messages)
            except Exception as e:
                self.logger.warning(f"Failed to validate physical components: {e}")
        
        # Validate component configs
        component_dirs = {
            "vae": "vae_config",
            "text_encoder": "text_encoder_config", 
            "text_encoder_2": "text_encoder_config",
            "unet": "unet_config",
            "scheduler": "scheduler_config"
        }
        
        for component_dir, config_type in component_dirs.items():
            component_path = model_path / component_dir / "config.json"
            if component_path.exists():
                component_result = self.validate_component_config(component_path, config_type)
                all_messages.extend(component_result.messages)
                all_cleaned_attributes.extend(component_result.cleaned_attributes)
                if component_result.backup_path:
                    backup_paths.append(component_result.backup_path)
        
        # Check for model library compatibility
        compatibility_messages = self._validate_model_library_compatibility(model_path)
        all_messages.extend(compatibility_messages)
        
        # Determine overall validity
        is_valid = not any(msg.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          for msg in all_messages)
        
        return ValidationResult(
            is_valid=is_valid,
            messages=all_messages,
            cleaned_attributes=all_cleaned_attributes,
            backup_path=backup_paths[0] if backup_paths else None
        )
    
    def _validate_pipeline_class_compatibility(self, pipeline_class: str) -> List[ValidationMessage]:
        """Validate pipeline class compatibility (registry-based only)"""
        messages = []
        
        # Check if pipeline class is supported
        supported_classes = ["WanPipeline", "WanImageToVideoPipeline", "StableDiffusionPipeline", "DiffusionPipeline"]
        if pipeline_class not in supported_classes:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="UNSUPPORTED_PIPELINE_CLASS",
                message=f"Pipeline class '{pipeline_class}' may not be fully supported",
                field_path="model_index._class_name",
                current_value=pipeline_class,
                suggested_value="WanPipeline",
                help_text=f"Supported classes: {', '.join(supported_classes)}"
            ))
        
        return messages
    
    def _validate_pipeline_compatibility(self, pipeline_class: str, model_path: Path) -> List[ValidationMessage]:
        """Validate pipeline class compatibility including physical components"""
        messages = []
        
        # First validate class compatibility
        messages.extend(self._validate_pipeline_class_compatibility(pipeline_class))
        
        # Check for WAN-specific requirements (physical components)
        if pipeline_class == "WanPipeline":
            # Check for required WAN components
            required_components = ["vae", "text_encoder", "unet"]
            for component in required_components:
                component_path = model_path / component
                if not component_path.exists():
                    messages.append(ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        code="MISSING_WAN_COMPONENT",
                        message=f"Required WAN component missing: {component}",
                        field_path=f"model_components.{component}",
                        help_text=f"WAN models require {component} component directory"
                    ))
        
        return messages
    
    def _validate_model_library_compatibility(self, model_path: Path) -> List[ValidationMessage]:
        """Validate model library compatibility"""
        messages = []
        
        try:
            # Load model_index.json to get model info
            model_index_path = model_path / "model_index.json"
            if not model_index_path.exists():
                return messages
            
            with open(model_index_path, 'r', encoding='utf-8') as f:
                model_index = json.load(f)
            
            pipeline_class = model_index.get("_class_name")
            diffusers_version = model_index.get("_diffusers_version")
            
            # Check diffusers version compatibility
            if diffusers_version:
                # Parse version and check compatibility
                try:
                    from packaging import version
                    current_version = version.parse(diffusers_version)
                    min_version = version.parse("0.21.0")
                    
                    if current_version < min_version:
                        messages.append(ValidationMessage(
                            severity=ValidationSeverity.WARNING,
                            code="OLD_DIFFUSERS_VERSION",
                            message=f"Model uses old diffusers version: {diffusers_version}",
                            field_path="model_index._diffusers_version",
                            current_value=diffusers_version,
                            suggested_value="0.21.0+",
                            help_text="Consider updating to a newer model version"
                        ))
                except ImportError:
                    # packaging not available, skip version check
                    pass
                except Exception as e:
                    self.logger.warning(f"Failed to parse diffusers version: {e}")
            
            # Check compatibility registry
            model_name = model_path.name
            if model_name in self.compatibility_registry:
                registry_info = self.compatibility_registry[model_name]
                
                # Check pipeline class matches registry
                expected_class = registry_info.get("pipeline_class")
                if expected_class and pipeline_class != expected_class:
                    messages.append(ValidationMessage(
                        severity=ValidationSeverity.WARNING,
                        code="PIPELINE_CLASS_MISMATCH",
                        message=f"Pipeline class mismatch with registry",
                        field_path="model_index._class_name",
                        current_value=pipeline_class,
                        suggested_value=expected_class,
                        help_text="Model may not work as expected with current pipeline class"
                    ))
                
                # Check trust_remote_code requirement
                requires_trust = registry_info.get("trust_remote_code", False)
                if requires_trust:
                    messages.append(ValidationMessage(
                        severity=ValidationSeverity.INFO,
                        code="REQUIRES_TRUST_REMOTE_CODE",
                        message="Model requires trust_remote_code=True",
                        field_path="model_loading.trust_remote_code",
                        suggested_value=True,
                        help_text="This model requires trust_remote_code=True for loading"
                    ))
        
        except Exception as e:
            self.logger.warning(f"Failed to validate model library compatibility: {e}")
        
        return messages
    
    def _cleanup_model_config(self, config_data: Dict[str, Any], config_type: str) -> CleanupResult:
        """
        Clean up model-specific configuration attributes.
        
        Args:
            config_data: Configuration data to clean
            config_type: Type of configuration (model_index, vae_config, etc.)
            
        Returns:
            CleanupResult with list of cleaned attributes
        """
        cleaned_attributes = []
        
        # Get cleanup attributes for this config type
        cleanup_attrs = self.model_cleanup_attributes.get(config_type, set())
        
        # Clean up attributes
        for attr in list(cleanup_attrs):
            if attr in config_data:
                del config_data[attr]
                cleaned_attributes.append(f"{config_type}.{attr}")
                self.logger.info(f"Removed unsupported model attribute: {config_type}.{attr}")
        
        return CleanupResult(
            cleaned_attributes=cleaned_attributes,
            backup_created=True
        )
    
    def get_model_compatibility_info(self, model_name: str) -> Optional[ModelCompatibilityInfo]:
        """
        Get compatibility information for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelCompatibilityInfo if found, None otherwise
        """
        if model_name in self.compatibility_registry:
            info = self.compatibility_registry[model_name]
            return ModelCompatibilityInfo(
                model_name=model_name,
                pipeline_class=info.get("pipeline_class", ""),
                supported_optimizations=info.get("supported_optimizations", []),
                vram_requirements=info.get("vram_requirements", {}),
                trust_remote_code=info.get("trust_remote_code", False),
                min_diffusers_version=info.get("min_diffusers_version", "")
            )
        return None


def validate_model_directory(model_path: Union[str, Path], 
                           backup_dir: Optional[Path] = None,
                           compatibility_registry_path: Optional[Path] = None) -> ValidationResult:
    """
    Convenience function to validate a model directory.
    
    Args:
        model_path: Path to model directory
        backup_dir: Directory for backups (optional)
        compatibility_registry_path: Path to compatibility registry (optional)
        
    Returns:
        ValidationResult with validation messages and cleanup info
    """
    validator = ModelConfigValidator(backup_dir, compatibility_registry_path)
    return validator.validate_model_directory(model_path)


def validate_model_index(model_index_path: Union[str, Path],
                        backup_dir: Optional[Path] = None,
                        compatibility_registry_path: Optional[Path] = None) -> ValidationResult:
    """
    Convenience function to validate a model_index.json file.
    
    Args:
        model_index_path: Path to model_index.json file
        backup_dir: Directory for backups (optional)
        compatibility_registry_path: Path to compatibility registry (optional)
        
    Returns:
        ValidationResult with validation messages and cleanup info
    """
    validator = ModelConfigValidator(backup_dir, compatibility_registry_path)
    return validator.validate_model_index(model_index_path)
