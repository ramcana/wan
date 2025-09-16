---
title: core.model_orchestrator.test_wan22_handler
category: api
tags: [api, core]
---

# core.model_orchestrator.test_wan22_handler

Tests for WAN2.2-specific model handling.

Tests cover:
- Sharded model support and selective downloading
- Model-specific input validation
- Text embedding caching
- VAE configuration and memory optimization
- Development/production variant switching
- Component validation and file management

## Classes

### TestTextEmbeddingCache

Test text embedding cache functionality.

#### Methods

##### test_cache_basic_operations(self: Any)

Test basic cache operations.

##### test_cache_lru_eviction(self: Any)

Test LRU eviction when cache is full.

##### test_cache_clear(self: Any)

Test cache clearing.

### TestWAN22ModelHandler

Test WAN2.2 model handler functionality.

#### Methods

##### sample_model_spec(self: Any)

Create a sample model specification.

##### temp_model_dir(self: Any)

Create a temporary model directory.

##### test_component_parsing(self: Any, sample_model_spec: Any, temp_model_dir: Any)

Test parsing of model components from file specs.

##### test_required_shards_production(self: Any, sample_model_spec: Any, temp_model_dir: Any)

Test getting required shards for production variant.

##### test_required_shards_development(self: Any, sample_model_spec: Any, temp_model_dir: Any)

Test getting required shards for development variant.

##### test_shard_index_parsing(self: Any, sample_model_spec: Any, temp_model_dir: Any)

Test parsing of shard index files.

##### test_input_validation_t2v(self: Any, sample_model_spec: Any, temp_model_dir: Any)

Test input validation for T2V models.

##### test_input_validation_i2v(self: Any, temp_model_dir: Any)

Test input validation for I2V models.

##### test_input_validation_ti2v(self: Any, temp_model_dir: Any)

Test input validation for TI2V models.

##### test_image_preprocessing(self: Any, sample_model_spec: Any, temp_model_dir: Any)

Test image preprocessing for I2V/TI2V models.

##### test_text_embedding_cache(self: Any, sample_model_spec: Any, temp_model_dir: Any)

Test text embedding caching functionality.

##### test_vae_config_generation(self: Any, sample_model_spec: Any, temp_model_dir: Any)

Test VAE configuration generation.

##### test_memory_estimation(self: Any, sample_model_spec: Any, temp_model_dir: Any)

Test memory usage estimation.

##### test_variant_conversion(self: Any, sample_model_spec: Any, temp_model_dir: Any)

Test development/production variant conversion.

##### test_component_validation(self: Any, sample_model_spec: Any, temp_model_dir: Any)

Test component completeness validation.

##### test_component_file_retrieval(self: Any, sample_model_spec: Any, temp_model_dir: Any)

Test component file retrieval.

### TestWAN22Integration

Integration tests for WAN2.2 model handling.

#### Methods

##### test_full_model_workflow(self: Any, tmp_path: Any)

Test complete model workflow with WAN2.2 handler.

