---
title: scripts.download_models
category: api
tags: [api, scripts]
---

# scripts.download_models

WAN Model Download Script
Downloads and sets up WAN model files for the video generation system.

## Classes

### ModelDownloadManager

Manages downloading and setup of WAN models

#### Methods

##### __init__(self: Any, models_dir: str)



##### check_model_status(self: Any, model_type: str) -> <ast.Subscript object at 0x000001943445B4C0>

Check if a model is downloaded and valid

##### download_file(self: Any, url: str, local_path: Path, expected_size: <ast.Subscript object at 0x00000194344584C0>) -> bool

Download a file with progress bar or create placeholder

##### create_placeholder_file(self: Any, local_path: Path) -> bool

Create a placeholder file for development/testing

##### download_model_hf_cli(self: Any, model_type: str, force: bool) -> bool

Download model using Hugging Face CLI

##### download_model(self: Any, model_type: str, force: bool) -> bool

Download a specific model (tries HF CLI first, falls back to placeholder)

##### create_model_placeholder(self: Any, model_type: str) -> bool

Create a complete model placeholder structure

##### download_all_models(self: Any, force: bool) -> <ast.Subscript object at 0x00000194302EFFA0>

Download all available models

##### list_models(self: Any) -> None

List all available models and their status

##### verify_model(self: Any, model_type: str) -> bool

Verify model integrity

