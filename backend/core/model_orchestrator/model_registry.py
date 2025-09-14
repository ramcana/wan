"""
Model Registry - Manifest parsing and model specification management.

This module handles loading and validating the models.toml manifest file,
providing typed access to model specifications with proper validation.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for older Python versions

from .exceptions import (
    ModelNotFoundError,
    VariantNotFoundError,
    InvalidModelIdError,
    ManifestValidationError,
    SchemaVersionError,
)


@dataclass(frozen=True)
class FileSpec:
    """Specification for a single file within a model."""
    
    path: str                         # Relative path within model directory
    size: int                         # Expected file size in bytes
    sha256: str                       # SHA256 checksum for integrity verification
    optional: bool = False            # Whether file is optional
    component: Optional[str] = None   # Component type (text_encoder, image_encoder, unet, vae, config)
    shard_index: bool = False         # Whether this is a shard index file
    shard_part: Optional[int] = None  # Shard part number for sharded models


@dataclass(frozen=True)
class VramEstimation:
    """VRAM estimation parameters for a model."""
    
    params_billion: float             # Model parameters in billions
    family_size: str                  # Size category (small, medium, large)
    base_vram_gb: float              # Base VRAM requirement in GB
    per_frame_vram_mb: float         # Additional VRAM per frame in MB


@dataclass(frozen=True)
class ModelDefaults:
    """Default parameters for model inference."""
    
    fps: int = 24                     # Default frames per second
    frames: int = 16                  # Default number of frames
    scheduler: str = "ddim"           # Default scheduler
    precision: str = "fp16"           # Default precision
    guidance_scale: float = 7.5       # Default guidance scale
    num_inference_steps: int = 50     # Default inference steps
    image_guidance_scale: Optional[float] = None  # Image guidance scale (for i2v/ti2v)
    text_guidance_scale: Optional[float] = None   # Text guidance scale (for ti2v)
    vae_tile_size: int = 512          # VAE tile size for memory optimization
    vae_decode_chunk_size: int = 8    # VAE decode chunk size


@dataclass(frozen=True)
class ModelSpec:
    """Complete specification for a model including all variants and metadata."""
    
    model_id: str                     # Canonical ID (e.g., "t2v-A14B@2.2.0")
    version: str                      # Model version (e.g., "2.2.0")
    variants: List[str]               # Available variants ["fp16", "bf16", "int8"]
    default_variant: str              # Default precision variant
    files: List[FileSpec]             # Required files with metadata
    sources: List[str]                # Priority-ordered source URLs
    allow_patterns: List[str]         # File patterns for selective download
    resolution_caps: List[str]        # Supported resolutions
    optional_components: List[str]    # Optional model components
    lora_required: bool               # Whether LoRA support is required
    description: str = ""             # Human-readable description
    required_components: List[str] = None  # Required model components
    defaults: Optional[ModelDefaults] = None  # Default inference parameters
    vram_estimation: Optional[VramEstimation] = None  # VRAM estimation parameters
    model_type: Optional[str] = None  # Model type (t2v, i2v, ti2v)


@dataclass(frozen=True)
class NormalizedModelId:
    """Normalized model identifier components."""
    
    model_id: str                     # Base model ID without version
    version: str                      # Model version
    variant: Optional[str]            # Precision variant (if specified)


class ModelRegistry:
    """
    Registry for managing model specifications from a TOML manifest.
    
    Provides validation, normalization, and typed access to model definitions
    with support for schema versioning and migration guidance.
    """
    
    SUPPORTED_SCHEMA_VERSIONS = ["1"]
    MODEL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*@[0-9]+\.[0-9]+\.[0-9]+$")
    VARIANT_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
    
    def __init__(self, manifest_path: str):
        """
        Initialize the registry with a manifest file.
        
        Args:
            manifest_path: Path to the models.toml manifest file
            
        Raises:
            FileNotFoundError: If manifest file doesn't exist
            ManifestValidationError: If manifest is invalid
            SchemaVersionError: If schema version is unsupported
        """
        self.manifest_path = Path(manifest_path)
        self._models: Dict[str, ModelSpec] = {}
        self._schema_version: str = ""
        
        self._load_manifest()
    
    def _load_manifest(self) -> None:
        """Load and parse the TOML manifest file."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        
        try:
            with open(self.manifest_path, "rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            raise ManifestValidationError([f"Failed to parse TOML: {e}"])
        
        # Validate schema version
        self._schema_version = str(data.get("schema_version", ""))
        if self._schema_version not in self.SUPPORTED_SCHEMA_VERSIONS:
            raise SchemaVersionError(self._schema_version, self.SUPPORTED_SCHEMA_VERSIONS)
        
        # Parse models
        models_data = data.get("models", {})
        if not models_data:
            raise ManifestValidationError(["No models defined in manifest"])
        
        validation_errors = []
        for model_key, model_data in models_data.items():
            try:
                model_spec = self._parse_model_spec(model_key, model_data)
                self._models[model_spec.model_id] = model_spec
            except Exception as e:
                validation_errors.append(f"Model '{model_key}': {e}")
        
        if validation_errors:
            raise ManifestValidationError(validation_errors)
        
        # Validate the complete manifest
        manifest_validation_errors = self.validate_manifest()
        if manifest_validation_errors:
            raise ManifestValidationError([str(e) for e in manifest_validation_errors])
    
    def _parse_model_spec(self, model_key: str, model_data: dict) -> ModelSpec:
        """Parse a single model specification from manifest data."""
        # Validate model key format
        if not self.MODEL_ID_PATTERN.match(model_key):
            raise ValueError(f"Invalid model ID format: {model_key}")
        
        # Extract version from model key
        model_id_base, version = model_key.rsplit("@", 1)
        
        # Parse required fields
        variants = model_data.get("variants", [])
        if not variants:
            raise ValueError("variants field is required and cannot be empty")
        
        default_variant = model_data.get("default_variant")
        if not default_variant:
            raise ValueError("default_variant field is required")
        
        if default_variant not in variants:
            raise ValueError(f"default_variant '{default_variant}' not in variants list")
        
        # Validate variants
        for variant in variants:
            if not self.VARIANT_PATTERN.match(variant):
                raise ValueError(f"Invalid variant format: {variant}")
        
        # Parse files
        files_data = model_data.get("files", [])
        if not files_data:
            raise ValueError("files field is required and cannot be empty")
        
        files = []
        for file_data in files_data:
            if not isinstance(file_data, dict):
                raise ValueError("Each file entry must be a dictionary")
            
            path = file_data.get("path")
            size = file_data.get("size")
            sha256 = file_data.get("sha256")
            optional = file_data.get("optional", False)
            component = file_data.get("component")
            shard_index = file_data.get("shard_index", False)
            shard_part = file_data.get("shard_part")
            
            if not path or not isinstance(path, str):
                raise ValueError("File path is required and must be a string")
            if not isinstance(size, int) or size < 0:
                raise ValueError("File size must be a non-negative integer")
            if not sha256 or not isinstance(sha256, str):
                raise ValueError("File sha256 is required and must be a string")
            if len(sha256) != 64:
                raise ValueError("File sha256 must be 64 characters long")
            
            files.append(FileSpec(
                path=path, 
                size=size, 
                sha256=sha256, 
                optional=optional,
                component=component,
                shard_index=shard_index,
                shard_part=shard_part
            ))
        
        # Parse sources
        sources_data = model_data.get("sources", {})
        if not sources_data or not isinstance(sources_data, dict):
            raise ValueError("sources field is required and must be a dictionary")
        
        priority_sources = sources_data.get("priority", [])
        if not priority_sources:
            raise ValueError("sources.priority field is required and cannot be empty")
        
        # Parse optional fields with defaults
        allow_patterns = model_data.get("allow_patterns", ["*"])
        resolution_caps = model_data.get("resolution_caps", [])
        optional_components = model_data.get("optional_components", [])
        lora_required = model_data.get("lora_required", False)
        description = model_data.get("description", "")
        required_components = model_data.get("required_components", [])
        
        # Determine model type from model_id
        model_type = None
        if "t2v" in model_id_base.lower():
            model_type = "t2v"
        elif "i2v" in model_id_base.lower():
            model_type = "i2v"
        elif "ti2v" in model_id_base.lower():
            model_type = "ti2v"
        
        # Parse defaults
        defaults_data = model_data.get("defaults", {})
        defaults = None
        if defaults_data:
            defaults = ModelDefaults(
                fps=defaults_data.get("fps", 24),
                frames=defaults_data.get("frames", 16),
                scheduler=defaults_data.get("scheduler", "ddim"),
                precision=defaults_data.get("precision", "fp16"),
                guidance_scale=defaults_data.get("guidance_scale", 7.5),
                num_inference_steps=defaults_data.get("num_inference_steps", 50),
                image_guidance_scale=defaults_data.get("image_guidance_scale"),
                text_guidance_scale=defaults_data.get("text_guidance_scale"),
                vae_tile_size=defaults_data.get("vae_tile_size", 512),
                vae_decode_chunk_size=defaults_data.get("vae_decode_chunk_size", 8)
            )
        
        # Parse VRAM estimation
        vram_data = model_data.get("vram_estimation", {})
        vram_estimation = None
        if vram_data:
            vram_estimation = VramEstimation(
                params_billion=vram_data.get("params_billion", 1.0),
                family_size=vram_data.get("family_size", "medium"),
                base_vram_gb=vram_data.get("base_vram_gb", 8.0),
                per_frame_vram_mb=vram_data.get("per_frame_vram_mb", 128.0)
            )
        
        return ModelSpec(
            model_id=model_key,
            version=version,
            variants=variants,
            default_variant=default_variant,
            files=files,
            sources=priority_sources,
            allow_patterns=allow_patterns,
            resolution_caps=resolution_caps,
            optional_components=optional_components,
            lora_required=lora_required,
            description=description,
            required_components=required_components,
            defaults=defaults,
            vram_estimation=vram_estimation,
            model_type=model_type,
        )
    
    def spec(self, model_id: str, variant: Optional[str] = None) -> ModelSpec:
        """
        Get the specification for a model.
        
        Args:
            model_id: Model identifier (can include @version)
            variant: Optional variant override
            
        Returns:
            ModelSpec for the requested model
            
        Raises:
            ModelNotFoundError: If model is not found
            VariantNotFoundError: If variant is not available
        """
        normalized = self.normalize_model_id(model_id, variant)
        full_model_id = f"{normalized.model_id}@{normalized.version}"
        
        if full_model_id not in self._models:
            raise ModelNotFoundError(full_model_id, list(self._models.keys()))
        
        model_spec = self._models[full_model_id]
        
        # Validate variant if specified
        target_variant = normalized.variant or model_spec.default_variant
        if target_variant not in model_spec.variants:
            raise VariantNotFoundError(full_model_id, target_variant, model_spec.variants)
        
        return model_spec
    
    def list_models(self) -> List[str]:
        """Get a list of all available model IDs."""
        return list(self._models.keys())
    
    def validate_manifest(self) -> List[Exception]:
        """
        Validate the loaded manifest for consistency and safety.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check for path safety issues
        all_paths = []
        for model_spec in self._models.values():
            for file_spec in model_spec.files:
                all_paths.append(file_spec.path)
        
        path_errors = self._validate_path_safety(all_paths)
        errors.extend(path_errors)
        
        # Check for case collisions on case-insensitive systems
        case_errors = self._validate_case_collisions(all_paths)
        errors.extend(case_errors)
        
        return errors
    
    def _validate_path_safety(self, paths: List[str]) -> List[Exception]:
        """Validate paths for safety (no directory traversal)."""
        errors = []
        
        for path in paths:
            # Check for directory traversal
            if ".." in Path(path).parts:
                errors.append(ValueError(f"Path traversal detected in: {path}"))
            
            # Check for absolute paths (handle both Unix and Windows styles)
            if (Path(path).is_absolute() or 
                path.startswith('/') or 
                (len(path) > 1 and path[1] == ':')):
                errors.append(ValueError(f"Absolute path not allowed: {path}"))
            
            # Check for Windows reserved names
            path_parts = Path(path).parts
            windows_reserved = {
                "CON", "PRN", "AUX", "NUL",
                "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
                "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
            }
            
            for part in path_parts:
                # Remove extension for reserved name check
                name_without_ext = part.split('.')[0].upper()
                if name_without_ext in windows_reserved:
                    errors.append(ValueError(f"Windows reserved name in path: {path}"))
        
        return errors
    
    def _validate_case_collisions(self, paths: List[str]) -> List[Exception]:
        """Validate for case collisions on case-insensitive file systems."""
        errors = []
        
        # Group paths by model to check collisions within each model
        model_paths = {}
        for model_spec in self._models.values():
            model_paths[model_spec.model_id] = [f.path for f in model_spec.files]
        
        # Check for case collisions within each model
        for model_id, paths in model_paths.items():
            seen_paths = set()
            for path in paths:
                lower_path = path.lower()
                if lower_path in seen_paths:
                    errors.append(ValueError(f"Case collision detected in model {model_id}: {path}"))
                seen_paths.add(lower_path)
        
        return errors
    
    def get_schema_version(self) -> str:
        """Get the schema version of the loaded manifest."""
        return self._schema_version
    
    @staticmethod
    def normalize_model_id(model_id: str, variant: Optional[str] = None) -> NormalizedModelId:
        """
        Normalize a model ID into canonical components.
        
        Args:
            model_id: Model identifier (may include @version and #variant)
            variant: Optional variant override
            
        Returns:
            NormalizedModelId with separated components
            
        Raises:
            InvalidModelIdError: If model ID format is invalid
        """
        if not model_id or not isinstance(model_id, str):
            raise InvalidModelIdError(model_id, "Model ID must be a non-empty string")
        
        # Handle variant in model_id (e.g., "t2v-A14B@2.2.0#fp16")
        if "#" in model_id:
            model_id, embedded_variant = model_id.rsplit("#", 1)
            if variant is None:
                variant = embedded_variant
        
        # Validate and split model_id@version
        if "@" not in model_id:
            raise InvalidModelIdError(model_id, "Model ID must include version (format: model@version)")
        
        model_base, version = model_id.rsplit("@", 1)
        
        # Validate model base name
        if not model_base or not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", model_base):
            raise InvalidModelIdError(
                model_id, 
                "Model name must start with alphanumeric and contain only alphanumeric, underscore, or hyphen"
            )
        
        # Validate version format (semantic versioning)
        if not re.match(r"^[0-9]+\.[0-9]+\.[0-9]+$", version):
            raise InvalidModelIdError(model_id, "Version must follow semantic versioning (x.y.z)")
        
        # Validate variant if provided
        if variant is not None:
            if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", variant):
                raise InvalidModelIdError(
                    f"{model_id}#{variant}",
                    "Variant must start with alphanumeric and contain only alphanumeric, underscore, or hyphen"
                )
        
        return NormalizedModelId(
            model_id=model_base,
            version=version,
            variant=variant
        )