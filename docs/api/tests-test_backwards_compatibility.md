---
title: tests.test_backwards_compatibility
category: api
tags: [api, tests]
---

# tests.test_backwards_compatibility

Test backwards compatibility with existing Gradio system.
Ensures model files, LoRA weights, and generation results are identical.

## Classes

### TestBackwardsCompatibility

Test suite for backwards compatibility with existing Gradio system.

#### Methods

##### temp_dirs(self: Any)

Create temporary directories for testing.

##### mock_config(self: Any, temp_dirs: Any)

Create mock configuration matching Gradio setup.

##### mock_model_files(self: Any, temp_dirs: Any)

Create mock model files matching Gradio structure.

##### mock_lora_files(self: Any, temp_dirs: Any)

Create mock LoRA files matching Gradio structure.

##### test_model_file_detection(self: Any, temp_dirs: Any, mock_model_files: Any)

Test that model files are detected correctly.

##### test_lora_file_detection(self: Any, temp_dirs: Any, mock_lora_files: Any)

Test that LoRA files are detected correctly.

##### test_config_compatibility(self: Any, temp_dirs: Any, mock_config: Any)

Test that existing config.json is loaded correctly.

##### test_generation_service_compatibility(self: Any, mock_gpu_props: Any, temp_dirs: Any, mock_config: Any, mock_model_files: Any)

Test that generation service works with existing model structure.

##### test_output_directory_structure(self: Any, temp_dirs: Any, mock_config: Any)

Test that output directory structure is compatible.

##### test_lora_loading_compatibility(self: Any, temp_dirs: Any, mock_lora_files: Any)

Test that LoRA files can be loaded with same interface.

##### test_quantization_compatibility(self: Any, temp_dirs: Any, mock_config: Any)

Test that quantization settings work identically.

##### test_vram_optimization_compatibility(self: Any, temp_dirs: Any, mock_config: Any)

Test that VRAM optimization settings are preserved.

##### test_generation_parameters_compatibility(self: Any, temp_dirs: Any, mock_config: Any)

Test that generation parameters produce identical results.

##### test_error_handling_compatibility(self: Any, temp_dirs: Any, mock_config: Any)

Test that error handling works identically to Gradio.

##### test_file_format_compatibility(self: Any, temp_dirs: Any)

Test that file formats are handled identically.

### TestFullBackwardsCompatibility

Full integration test for backwards compatibility.

#### Methods

##### test_complete_migration_workflow(self: Any, tmp_path: Any)

Test complete migration from Gradio to new system.

