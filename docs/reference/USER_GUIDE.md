---
category: reference
last_updated: '2025-09-15T22:49:59.696011'
original_path: backend\core\model_orchestrator\docs\USER_GUIDE.md
tags:
- configuration
- api
- troubleshooting
- installation
- security
- performance
title: Model Orchestrator User Guide
---

# Model Orchestrator User Guide

## Overview

The Model Orchestrator is a comprehensive model management system for WAN2.2 that eliminates the complexity of model discovery, downloading, and path resolution. It provides a unified approach to managing AI models through a manifest-driven architecture with support for multiple storage backends.

## Key Features

- **Unified Model Management**: Single source of truth for all WAN2.2 model definitions
- **Multi-Source Downloads**: Automatic failover between local, S3/MinIO, and HuggingFace sources
- **Atomic Operations**: Safe, concurrent downloads with integrity verification
- **Cross-Platform Support**: Works on Windows, WSL, and Unix systems
- **Intelligent Caching**: Efficient storage with deduplication and garbage collection
- **CLI and API Access**: Both command-line tools and programmatic interfaces

## Quick Start

### 1. Installation

The Model Orchestrator is included with WAN2.2. Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configuration

Set the required environment variables:

```bash
# Required: Base directory for model storage
export MODELS_ROOT="/path/to/your/models"

# Optional: Custom manifest location
export WAN_MODELS_MANIFEST="/path/to/models.toml"

# Optional: HuggingFace token for private models
export HF_TOKEN="your_hf_token_here"

# Optional: S3/MinIO configuration
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_ENDPOINT_URL="https://your-minio-server.com"
```

### 3. Basic Usage

#### Command Line Interface

Check available models:

```bash
wan models status
```

Download a specific model:

```bash
wan models ensure --only t2v-A14B@2.2.0
```

Download all models:

```bash
wan models ensure --all
```

Check disk usage and clean up:

```bash
wan models gc --dry-run
wan models gc  # Actually perform cleanup
```

#### Python API

```python
from backend.core.model_orchestrator import ModelEnsurer

# Initialize the orchestrator
ensurer = ModelEnsurer.from_config()

# Ensure a model is available (downloads if needed)
model_path = ensurer.ensure("t2v-A14B@2.2.0", variant="fp16")

# Check model status without downloading
status = ensurer.status("t2v-A14B@2.2.0")
print(f"Model state: {status.state}")
print(f"Missing bytes: {status.bytes_needed}")

# Verify model integrity
verification = ensurer.verify_integrity("t2v-A14B@2.2.0")
print(f"Verification passed: {verification.verified}")
```

## Configuration

### Environment Variables

| Variable                    | Required | Description                      | Example                     |
| --------------------------- | -------- | -------------------------------- | --------------------------- |
| `MODELS_ROOT`               | Yes      | Base directory for model storage | `/data/models`              |
| `WAN_MODELS_MANIFEST`       | No       | Path to models.toml file         | `/config/models.toml`       |
| `HF_TOKEN`                  | No       | HuggingFace access token         | `hf_xxxxx`                  |
| `AWS_ACCESS_KEY_ID`         | No       | S3/MinIO access key              | `minioadmin`                |
| `AWS_SECRET_ACCESS_KEY`     | No       | S3/MinIO secret key              | `minioadmin`                |
| `AWS_ENDPOINT_URL`          | No       | Custom S3 endpoint               | `https://minio.example.com` |
| `HF_HUB_ENABLE_HF_TRANSFER` | No       | Enable fast HF downloads         | `1`                         |

### Models Manifest (models.toml)

The manifest defines all available models and their properties:

```toml
schema_version = 1

[models."t2v-A14B@2.2.0"]
description = "WAN2.2 Text-to-Video A14B Model"
version = "2.2.0"
variants = ["fp16", "bf16"]
default_variant = "fp16"
resolution_caps = ["720p24", "1080p24"]
optional_components = []
lora_required = false

[[models."t2v-A14B@2.2.0".files]]
path = "model_index.json"
size = 12345
sha256 = "abc123..."

[[models."t2v-A14B@2.2.0".files]]
path = "unet/diffusion_pytorch_model.safetensors"
size = 5368709120
sha256 = "def456..."

[models."t2v-A14B@2.2.0".sources]
priority = [
    "local://wan22/t2v-A14B@2.2.0",
    "s3://ai-models/wan22/t2v-A14B@2.2.0",
    "hf://Wan-AI/Wan2.2-T2V-A14B"
]
allow_patterns = ["*.safetensors", "*.json", "*.yaml"]
```

## Model Variants

The orchestrator supports multiple model variants for different use cases:

- **fp16**: Half-precision floating point (default, good balance of speed/quality)
- **bf16**: Brain floating point (better numerical stability)
- **int8**: 8-bit quantized (smaller size, faster inference)

Example usage:

```bash
wan models ensure --only t2v-A14B@2.2.0 --variant bf16
```

```python
model_path = ensurer.ensure("t2v-A14B@2.2.0", variant="bf16")
```

## Storage Backends

### Local Storage

Models stored on local filesystem. Fastest access but requires manual management.

```toml
[models."my-model@1.0.0".sources]
priority = ["local://models/my-model@1.0.0"]
```

### HuggingFace Hub

Download from HuggingFace model repositories. Supports private models with tokens.

```toml
[models."my-model@1.0.0".sources]
priority = ["hf://organization/model-name"]
```

### S3/MinIO

Download from S3-compatible storage. Supports custom endpoints for MinIO.

```toml
[models."my-model@1.0.0".sources]
priority = ["s3://bucket-name/path/to/model"]
```

## Disk Space Management

### Automatic Garbage Collection

The orchestrator can automatically clean up old models when disk space is low:

```python
from backend.core.model_orchestrator import GarbageCollector

gc = GarbageCollector.from_config(
    max_total_size=100 * 1024**3,  # 100GB limit
    max_model_age=timedelta(days=30)  # Remove models older than 30 days
)

# Run cleanup
result = gc.collect(dry_run=False)
print(f"Reclaimed {result.bytes_reclaimed} bytes")
```

### Model Pinning

Protect important models from garbage collection:

```bash
wan models pin t2v-A14B@2.2.0
wan models unpin t2v-A14B@2.2.0
```

```python
gc.pin_model("t2v-A14B@2.2.0")
gc.unpin_model("t2v-A14B@2.2.0")
```

## Monitoring and Health Checks

### Health Endpoints

Check system health via HTTP endpoints:

```bash
curl http://localhost:8000/health/models
curl http://localhost:8000/health/models/t2v-A14B@2.2.0
```

### Metrics

Prometheus-compatible metrics are available:

```bash
curl http://localhost:8000/metrics
```

Key metrics include:

- `model_downloads_total`: Number of downloads by model and status
- `model_download_duration_seconds`: Download duration histogram
- `model_storage_bytes_used`: Storage usage by model family
- `model_errors_total`: Error counts by type and model

### Logging

Structured logging with correlation IDs:

```python
import logging
from backend.core.model_orchestrator.logging_config import setup_logging

setup_logging(level=logging.INFO, format="json")
```

## Integration with WAN Pipeline

The orchestrator integrates seamlessly with existing WAN2.2 pipelines:

```python
from backend.services.wan_pipeline_loader import get_wan_paths

# Old way (hardcoded paths)
# model_path = "/data/models/t2v-A14B"

# New way (orchestrated)
model_path = get_wan_paths("t2v-A14B@2.2.0")

# Load pipeline as usual
pipeline = WanT2VPipeline.from_pretrained(model_path)
```

## Best Practices

### 1. Directory Structure

Organize your models directory:

```
/data/models/
├── .tmp/           # Temporary downloads
├── .locks/         # Lock files
├── .state/         # State information
└── wan22/          # Model directories
    ├── t2v-A14B@2.2.0/
    ├── i2v-A14B@2.2.0/
    └── ti2v-5b@2.2.0/
```

### 2. Performance Optimization

- Use SSD storage for `MODELS_ROOT` when possible
- Enable `HF_HUB_ENABLE_HF_TRANSFER=1` for faster HuggingFace downloads
- Configure appropriate `max_concurrent_downloads` for your network
- Use local or S3 sources for production deployments

### 3. Security

- Store sensitive tokens in environment variables or keyring
- Use presigned URLs for temporary S3 access
- Enable at-rest encryption for sensitive models
- Regularly rotate access credentials

### 4. Monitoring

- Set up Prometheus monitoring for production deployments
- Configure log aggregation for structured logs
- Monitor disk usage and set up alerts
- Track download success rates and performance

## Troubleshooting

### Common Issues

#### "Model not found" errors

```bash
# Check available models
wan models status

# Verify manifest syntax
wan models validate-manifest
```

#### Download failures

```bash
# Check network connectivity
wan models test-sources

# Verify credentials
wan models test-auth
```

#### Disk space issues

```bash
# Check current usage
wan models disk-usage

# Run garbage collection
wan models gc --dry-run
wan models gc
```

#### Permission errors

```bash
# Check directory permissions
ls -la $MODELS_ROOT

# Fix permissions (Unix/Linux)
chmod -R 755 $MODELS_ROOT
```

### Platform-Specific Issues

#### Windows Long Paths

Enable long path support in Windows:

1. Open Group Policy Editor (`gpedit.msc`)
2. Navigate to: Computer Configuration > Administrative Templates > System > Filesystem
3. Enable "Enable Win32 long paths"

Or via registry:

```cmd
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1
```

#### WSL Path Issues

Ensure consistent path handling:

```bash
# Use Windows paths in WSL when needed
export MODELS_ROOT="/mnt/c/data/models"
```

### Getting Help

1. Check the logs for detailed error messages
2. Run diagnostics: `wan models diagnose`
3. Consult the troubleshooting guide
4. Report issues with full error logs and system information

## Advanced Usage

### Custom Storage Backends

Implement custom storage backends:

```python
from backend.core.model_orchestrator.storage_backends.base_store import StorageBackend

class CustomStore(StorageBackend):
    def can_handle(self, source_url: str) -> bool:
        return source_url.startswith("custom://")

    def download(self, source_url: str, local_dir: str,
                file_specs: List[FileSpec],
                progress_callback: Optional[Callable] = None) -> DownloadResult:
        # Implement custom download logic
        pass

# Register the backend
ensurer.add_storage_backend(CustomStore())
```

### Migration from Legacy Systems

Use the migration tools to transition from existing setups:

```bash
# Analyze existing model directories
wan models migrate analyze /old/models/path

# Generate migration plan
wan models migrate plan /old/models/path

# Execute migration
wan models migrate execute /old/models/path
```

### Batch Operations

Process multiple models efficiently:

```python
# Batch download
models_to_download = ["t2v-A14B@2.2.0", "i2v-A14B@2.2.0", "ti2v-5b@2.2.0"]
results = ensurer.ensure_batch(models_to_download, max_concurrent=4)

# Batch verification
verification_results = ensurer.verify_batch(models_to_download)
```

## API Reference

For detailed API documentation, see the [API Reference](API_REFERENCE.md).

## Examples

For complete examples and use cases, see the [examples directory](../examples/).
