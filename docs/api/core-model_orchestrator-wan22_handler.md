---
title: core.model_orchestrator.wan22_handler
category: api
tags: [api, core]
---

# core.model_orchestrator.wan22_handler

WAN2.2-specific model handling and processing.

This module provides specialized functionality for WAN2.2 models including:
- Sharded model support with selective downloading
- Model-specific conditioning and preprocessing
- Text embedding caching for multi-clip sequences
- VAE optimization and memory management
- Development/production variant switching
- Input validation for different model types

## Classes

### ShardInfo

Information about a model shard.

### ComponentInfo

Information about a model component.

### TextEmbeddingCache

Cache for text embeddings to avoid recomputation.

#### Methods

##### __post_init__(self: Any)



##### get(self: Any, text: str) -> <ast.Subscript object at 0x00000194284A5F30>

Get cached embedding for text.

##### put(self: Any, text: str, embedding: torch.Tensor) -> None

Cache embedding for text.

##### clear(self: Any) -> None

Clear the cache.

### WAN22ModelHandler

Handler for WAN2.2-specific model operations.

#### Methods

##### __init__(self: Any, model_spec: ModelSpec, model_dir: Path)

Initialize the WAN2.2 model handler.

Args:
    model_spec: Model specification from registry
    model_dir: Path to model directory

##### _parse_components(self: Any) -> None

Parse model components from file specifications.

##### get_required_shards(self: Any, component_type: str, variant: <ast.Subscript object at 0x0000019428496B60>) -> <ast.Subscript object at 0x00000194284960E0>

Get required shards for a component based on variant and usage.

Args:
    component_type: Type of component (unet, text_encoder, etc.)
    variant: Model variant (fp16, bf16, etc.)
    
Returns:
    List of required shards for selective downloading

##### parse_shard_index(self: Any, index_file_path: Path) -> <ast.Subscript object at 0x00000194284B2080>

Parse a shard index file to determine required shards.

Args:
    index_file_path: Path to the index.json file
    
Returns:
    Parsed index data with shard information

##### validate_input_for_model_type(self: Any, inputs: <ast.Subscript object at 0x00000194284B1ED0>) -> None

Validate inputs based on model type requirements.

Args:
    inputs: Input dictionary to validate
    
Raises:
    InvalidInputError: If inputs are invalid for model type

##### preprocess_image_input(self: Any, image: <ast.Subscript object at 0x00000194284B0AF0>, target_size: <ast.Subscript object at 0x00000194284B0910>) -> torch.Tensor

Preprocess image input for i2v/ti2v models.

Args:
    image: Input image in various formats
    target_size: Target size for resizing (width, height)
    
Returns:
    Preprocessed image tensor

##### get_cached_text_embedding(self: Any, text: str) -> <ast.Subscript object at 0x00000194284CAE90>

Get cached text embedding if available.

##### cache_text_embedding(self: Any, text: str, embedding: torch.Tensor) -> None

Cache text embedding for future use.

##### get_vae_config(self: Any, variant: <ast.Subscript object at 0x00000194284CA9B0>) -> <ast.Subscript object at 0x00000194284C9030>

Get VAE configuration optimized for memory usage.

Args:
    variant: Model variant (affects memory optimization)
    
Returns:
    VAE configuration dictionary

##### estimate_memory_usage(self: Any, inputs: <ast.Subscript object at 0x00000194284C8E80>, variant: <ast.Subscript object at 0x00000194284C8D60>) -> <ast.Subscript object at 0x000001942A11E560>

Estimate memory usage for given inputs and variant.

Args:
    inputs: Input parameters (frames, resolution, etc.)
    variant: Model variant
    
Returns:
    Memory usage estimates in GB

##### get_development_variant(self: Any, base_variant: str) -> str

Get development variant name for faster iteration.

Args:
    base_variant: Base variant (fp16, bf16, etc.)
    
Returns:
    Development variant name

##### get_production_variant(self: Any, dev_variant: str) -> str

Get production variant name from development variant.

Args:
    dev_variant: Development variant name
    
Returns:
    Production variant name

##### is_development_variant(self: Any, variant: str) -> bool

Check if variant is a development variant.

##### validate_component_completeness(self: Any, component_type: str) -> <ast.Subscript object at 0x0000019427BD8970>

Validate that all required files for a component are present.

Args:
    component_type: Type of component to validate
    
Returns:
    Tuple of (is_complete, missing_files)

##### get_component_files(self: Any, component_type: str, include_optional: bool) -> <ast.Subscript object at 0x0000019427BD91E0>

Get files for a specific component.

Args:
    component_type: Type of component
    include_optional: Whether to include optional files
    
Returns:
    List of file specifications for the component

##### clear_text_embedding_cache(self: Any) -> None

Clear the text embedding cache.

