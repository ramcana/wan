"""
WAN2.2-specific model handling and processing.

This module provides specialized functionality for WAN2.2 models including:
- Sharded model support with selective downloading
- Model-specific conditioning and preprocessing
- Text embedding caching for multi-clip sequences
- VAE optimization and memory management
- Development/production variant switching
- Input validation for different model types
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import hashlib
import time

import torch
import numpy as np
from PIL import Image

from .model_registry import ModelSpec, FileSpec, ModelDefaults
from .exceptions import ModelValidationError, InvalidInputError


logger = logging.getLogger(__name__)


@dataclass
class ShardInfo:
    """Information about a model shard."""
    
    shard_part: int                   # Shard part number
    file_path: str                    # Path to shard file
    size: int                         # Shard file size
    sha256: str                       # Shard checksum
    required: bool = True             # Whether shard is required


@dataclass
class ComponentInfo:
    """Information about a model component."""
    
    component_type: str               # Component type (text_encoder, image_encoder, unet, vae)
    files: List[FileSpec]            # Files belonging to this component
    shards: List[ShardInfo]          # Shards for this component (if sharded)
    optional: bool = False           # Whether component is optional
    loaded: bool = False             # Whether component is currently loaded


@dataclass
class TextEmbeddingCache:
    """Cache for text embeddings to avoid recomputation."""
    
    max_size: int = 100              # Maximum cache entries
    cache: Dict[str, torch.Tensor] = None   # Text hash -> embedding tensor
    access_times: Dict[str, float] = None   # Access timestamps for LRU eviction
    
    def __post_init__(self):
        if self.cache is None:
            self.cache = {}
        if self.access_times is None:
            self.access_times = {}
    
    def get(self, text: str) -> Optional[torch.Tensor]:
        """Get cached embedding for text."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        if text_hash in self.cache:
            self.access_times[text_hash] = time.time()
            return self.cache[text_hash]
        return None
    
    def put(self, text: str, embedding: torch.Tensor) -> None:
        """Cache embedding for text."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        
        # Evict oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[text_hash] = embedding
        self.access_times[text_hash] = time.time()
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_times.clear()


class WAN22ModelHandler:
    """Handler for WAN2.2-specific model operations."""
    
    def __init__(self, model_spec: ModelSpec, model_dir: Path):
        """
        Initialize the WAN2.2 model handler.
        
        Args:
            model_spec: Model specification from registry
            model_dir: Path to model directory
        """
        self.model_spec = model_spec
        self.model_dir = Path(model_dir)
        self.components: Dict[str, ComponentInfo] = {}
        self.text_embedding_cache = TextEmbeddingCache()
        
        # Parse components from file specs
        self._parse_components()
        
        # Validate model type
        if not self.model_spec.model_type:
            raise ModelValidationError(f"Model type not specified for {model_spec.model_id}")
    
    def _parse_components(self) -> None:
        """Parse model components from file specifications."""
        component_files: Dict[str, List[FileSpec]] = {}
        
        # Group files by component
        for file_spec in self.model_spec.files:
            if file_spec.component:
                if file_spec.component not in component_files:
                    component_files[file_spec.component] = []
                component_files[file_spec.component].append(file_spec)
        
        # Create component info objects
        for component_type, files in component_files.items():
            # Check if component is optional
            optional = component_type in self.model_spec.optional_components
            
            # Parse shards for this component
            shards = []
            shard_files = [f for f in files if f.shard_part is not None]
            for file_spec in shard_files:
                shards.append(ShardInfo(
                    shard_part=file_spec.shard_part,
                    file_path=file_spec.path,
                    size=file_spec.size,
                    sha256=file_spec.sha256,
                    required=not file_spec.optional
                ))
            
            # Sort shards by part number
            shards.sort(key=lambda s: s.shard_part)
            
            self.components[component_type] = ComponentInfo(
                component_type=component_type,
                files=files,
                shards=shards,
                optional=optional
            )
    
    def get_required_shards(self, component_type: str, variant: Optional[str] = None) -> List[ShardInfo]:
        """
        Get required shards for a component based on variant and usage.
        
        Args:
            component_type: Type of component (unet, text_encoder, etc.)
            variant: Model variant (fp16, bf16, etc.)
            
        Returns:
            List of required shards for selective downloading
        """
        if component_type not in self.components:
            return []
        
        component = self.components[component_type]
        required_shards = []
        
        # For development variants, we might skip some shards
        if variant and "dev" in variant.lower():
            # For development, only download first few shards for faster iteration
            max_shards = 2 if component_type == "unet" else len(component.shards)
            required_shards = component.shards[:max_shards]
        else:
            # For production, download all required shards
            required_shards = [s for s in component.shards if s.required]
        
        return required_shards
    
    def parse_shard_index(self, index_file_path: Path) -> Dict[str, Any]:
        """
        Parse a shard index file to determine required shards.
        
        Args:
            index_file_path: Path to the index.json file
            
        Returns:
            Parsed index data with shard information
        """
        try:
            with open(index_file_path, 'r') as f:
                index_data = json.load(f)
            
            # Validate index structure
            if "weight_map" not in index_data:
                raise ModelValidationError(f"Invalid shard index: missing weight_map in {index_file_path}")
            
            return index_data
        except Exception as e:
            raise ModelValidationError(f"Failed to parse shard index {index_file_path}: {e}")
    
    def validate_input_for_model_type(self, inputs: Dict[str, Any]) -> None:
        """
        Validate inputs based on model type requirements.
        
        Args:
            inputs: Input dictionary to validate
            
        Raises:
            InvalidInputError: If inputs are invalid for model type
        """
        model_type = self.model_spec.model_type
        
        if model_type == "t2v":
            # Text-to-Video: requires text, no image
            if "prompt" not in inputs or not inputs["prompt"]:
                raise InvalidInputError("t2v models require a text prompt")
            if "image" in inputs and inputs["image"] is not None:
                raise InvalidInputError("t2v models do not accept image input")
        
        elif model_type == "i2v":
            # Image-to-Video: requires image, text optional
            if "image" not in inputs or inputs["image"] is None:
                raise InvalidInputError("i2v models require an image input")
            # Text is optional for i2v models
        
        elif model_type == "ti2v":
            # Text+Image-to-Video: requires both text and image
            if "prompt" not in inputs or not inputs["prompt"]:
                raise InvalidInputError("ti2v models require a text prompt")
            if "image" not in inputs or inputs["image"] is None:
                raise InvalidInputError("ti2v models require an image input")
        
        else:
            logger.warning(f"Unknown model type: {model_type}, skipping input validation")
    
    def preprocess_image_input(self, image: Union[Image.Image, np.ndarray, torch.Tensor], 
                              target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Preprocess image input for i2v/ti2v models.
        
        Args:
            image: Input image in various formats
            target_size: Target size for resizing (width, height)
            
        Returns:
            Preprocessed image tensor
        """
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image
            if image.dim() == 4:  # Batch dimension
                image = image.squeeze(0)
            if image.dim() == 3 and image.shape[0] in [1, 3]:  # CHW format
                image = image.permute(1, 2, 0)
            image = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
        
        # Resize if target size specified
        if target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Normalize to [-1, 1] range for WAN2.2 models
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # HWC -> CHW
        image_tensor = image_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        
        return image_tensor.unsqueeze(0)  # Add batch dimension
    
    def get_cached_text_embedding(self, text: str) -> Optional[torch.Tensor]:
        """Get cached text embedding if available."""
        return self.text_embedding_cache.get(text)
    
    def cache_text_embedding(self, text: str, embedding: torch.Tensor) -> None:
        """Cache text embedding for future use."""
        self.text_embedding_cache.put(text, embedding)
    
    def get_vae_config(self, variant: Optional[str] = None) -> Dict[str, Any]:
        """
        Get VAE configuration optimized for memory usage.
        
        Args:
            variant: Model variant (affects memory optimization)
            
        Returns:
            VAE configuration dictionary
        """
        defaults = self.model_spec.defaults or ModelDefaults()
        
        # Base configuration
        config = {
            "tile_size": defaults.vae_tile_size,
            "decode_chunk_size": defaults.vae_decode_chunk_size,
            "enable_tiling": True,
            "enable_slicing": True,
        }
        
        # Adjust for variant
        if variant:
            if "fp16" in variant:
                config["dtype"] = torch.float16
            elif "bf16" in variant:
                config["dtype"] = torch.bfloat16
            elif "fp32" in variant:
                config["dtype"] = torch.float32
            
            # More aggressive optimization for development variants
            if "dev" in variant.lower():
                config["tile_size"] = min(config["tile_size"], 256)
                config["decode_chunk_size"] = min(config["decode_chunk_size"], 4)
        
        # Model-specific optimizations
        if self.model_spec.vram_estimation:
            vram_gb = self.model_spec.vram_estimation.base_vram_gb
            if vram_gb > 16:  # High VRAM models
                config["tile_size"] = max(config["tile_size"], 1024)
            elif vram_gb < 8:  # Low VRAM models
                config["tile_size"] = min(config["tile_size"], 256)
                config["decode_chunk_size"] = min(config["decode_chunk_size"], 2)
        
        return config
    
    def estimate_memory_usage(self, inputs: Dict[str, Any], variant: Optional[str] = None) -> Dict[str, float]:
        """
        Estimate memory usage for given inputs and variant.
        
        Args:
            inputs: Input parameters (frames, resolution, etc.)
            variant: Model variant
            
        Returns:
            Memory usage estimates in GB
        """
        if not self.model_spec.vram_estimation:
            return {"total": 8.0, "base": 8.0, "dynamic": 0.0}
        
        vram_est = self.model_spec.vram_estimation
        base_vram = vram_est.base_vram_gb
        
        # Calculate dynamic memory based on inputs
        frames = inputs.get("frames", 16)
        width = inputs.get("width", 1024)
        height = inputs.get("height", 576)
        
        # Estimate dynamic memory
        pixel_count = width * height * frames
        dynamic_vram = (pixel_count / (1024 * 1024)) * (vram_est.per_frame_vram_mb / 1024)
        
        # Adjust for variant
        variant_multiplier = 1.0
        if variant:
            if "fp16" in variant:
                variant_multiplier = 0.5
            elif "bf16" in variant:
                variant_multiplier = 0.5
            elif "int8" in variant:
                variant_multiplier = 0.25
        
        base_vram *= variant_multiplier
        
        return {
            "total": base_vram + dynamic_vram,
            "base": base_vram,
            "dynamic": dynamic_vram,
            "variant_multiplier": variant_multiplier
        }
    
    def get_development_variant(self, base_variant: str) -> str:
        """
        Get development variant name for faster iteration.
        
        Args:
            base_variant: Base variant (fp16, bf16, etc.)
            
        Returns:
            Development variant name
        """
        return f"{base_variant}-dev"
    
    def get_production_variant(self, dev_variant: str) -> str:
        """
        Get production variant name from development variant.
        
        Args:
            dev_variant: Development variant name
            
        Returns:
            Production variant name
        """
        if dev_variant.endswith("-dev"):
            return dev_variant[:-4]
        return dev_variant
    
    def is_development_variant(self, variant: str) -> bool:
        """Check if variant is a development variant."""
        return variant.endswith("-dev")
    
    def validate_component_completeness(self, component_type: str) -> Tuple[bool, List[str]]:
        """
        Validate that all required files for a component are present.
        
        Args:
            component_type: Type of component to validate
            
        Returns:
            Tuple of (is_complete, missing_files)
        """
        if component_type not in self.components:
            return False, [f"Component {component_type} not defined"]
        
        component = self.components[component_type]
        missing_files = []
        
        for file_spec in component.files:
            file_path = self.model_dir / file_spec.path
            if not file_path.exists():
                if not file_spec.optional:
                    missing_files.append(file_spec.path)
        
        return len(missing_files) == 0, missing_files
    
    def get_component_files(self, component_type: str, include_optional: bool = False) -> List[FileSpec]:
        """
        Get files for a specific component.
        
        Args:
            component_type: Type of component
            include_optional: Whether to include optional files
            
        Returns:
            List of file specifications for the component
        """
        if component_type not in self.components:
            return []
        
        component = self.components[component_type]
        if include_optional:
            return component.files
        else:
            return [f for f in component.files if not f.optional]
    
    def clear_text_embedding_cache(self) -> None:
        """Clear the text embedding cache."""
        self.text_embedding_cache.clear()
        logger.info("Text embedding cache cleared")