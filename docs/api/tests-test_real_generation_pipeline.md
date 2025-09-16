---
title: tests.test_real_generation_pipeline
category: api
tags: [api, tests]
---

# tests.test_real_generation_pipeline

Tests for Real Generation Pipeline

## Classes

### TestRealGenerationPipeline

Test suite for RealGenerationPipeline

#### Methods

##### mock_wan_pipeline_loader(self: Any)

Mock WanPipelineLoader

##### mock_websocket_manager(self: Any)

Mock WebSocket manager

##### sample_params(self: Any)

Sample generation parameters

##### pipeline(self: Any, mock_wan_pipeline_loader: Any, mock_websocket_manager: Any)

Create pipeline instance with mocks

##### test_parse_resolution(self: Any, pipeline: Any)

Test resolution parsing

##### test_get_model_path(self: Any, pipeline: Any)

Test model path resolution

##### test_create_optimization_config(self: Any, pipeline: Any, sample_params: Any)

Test optimization config creation

##### test_generation_stats(self: Any, pipeline: Any)

Test generation statistics

##### test_clear_pipeline_cache(self: Any, pipeline: Any)

Test pipeline cache clearing

