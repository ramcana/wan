---
category: developer
last_updated: '2025-09-15T22:49:59.693204'
original_path: backend\core\model_orchestrator\docs\API_REFERENCE.md
tags:
- configuration
- api
- troubleshooting
title: Model Orchestrator API Reference
---

# Model Orchestrator API Reference

## Overview

This document provides comprehensive API reference for the WAN2.2 Model Orchestrator. The API is designed to be simple, consistent, and production-ready.

## Core Classes

### ModelEnsurer

The main interface for model management operations.

#### Constructor

```python
ModelEnsurer(
    registry: ModelRegistry,
    resolver: ModelResolver,
    lock_manager: LockManager,
    storage_backends: List[StorageBackend],
    config: Optional[OrchestratorConfig] = None
)
```

#### Class Methods

```python
@classmethod
def from_config(cls, config_path: Optional[str] = None) -> 'ModelEnsurer'
```

Creates a ModelEnsurer instance from configuration file or environment variables.

**Parameters:**

- `config_path` (optional): Path to configuration file

**Returns:** Configured ModelEnsurer instance

**Example:**

```python
ensurer = ModelEnsurer.from_config()
```

#### Instance Methods

##### ensure()

```python
def ensure(
    self,
    model_id: str,
    variant: Optional[str] = None,
    force_redownload: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> str
```

Ensures a model is available locally, downloading if necessary.

**Parameters:**

- `model_id`: Canonical model identifier (e.g., "t2v-A14B@2.2.0")
- `variant` (optional): Model variant ("fp16", "bf16", "int8")
- `force_redownload`: Force re-download even if model exists
- `progress_callback`: Callback for download progress (bytes_downloaded, total_bytes)

**Returns:** Absolute path to model directory

**Raises:**

- `ModelNotFoundError`: Model not found in manifest
- `InsufficientSpaceError`: Not enough disk space
- `ChecksumVerificationError`: File integrity check failed
- `LockTimeoutError`: Could not acquire model lock

**Example:**

```python
# Basic usage
model_path = ensurer.ensure("t2v-A14B@2.2.0")

# With specific variant
model_path = ensurer.ensure("t2v-A14B@2.2.0", variant="bf16")

# With progress callback
def progress(downloaded, total):
    print(f"Progress: {downloaded}/{total} bytes")

model_path = ensurer.ensure("t2v-A14B@2.2.0", progress_callback=progress)
```

##### status()

```python
def status(
    self,
    model_id: str,
    variant: Optional[str] = None
) -> ModelStatus
```

Gets the current status of a model without triggering downloads.

**Parameters:**

- `model_id`: Canonical model identifier
- `variant` (optional): Model variant

**Returns:** ModelStatus object with current state

**Example:**

```python
status = ensurer.status("t2v-A14B@2.2.0")
print(f"State: {status.state}")
print(f"Missing bytes: {status.bytes_needed}")
```

##### verify_integrity()

```python
def verify_integrity(
    self,
    model_id: str,
    variant: Optional[str] = None
) -> VerificationResult
```

Verifies the integrity of a downloaded model.

**Parameters:**

- `model_id`: Canonical model identifier
- `variant` (optional): Model variant

**Returns:** VerificationResult with verification details

**Example:**

```python
result = ensurer.verify_integrity("t2v-A14B@2.2.0")
if result.verified:
    print("Model integrity verified")
else:
    print(f"Verification failed: {result.errors}")
```

##### estimate_download_size()

```python
def estimate_download_size(
    self,
    model_id: str,
    variant: Optional[str] = None
) -> int
```

Estimates the download size for a model.

**Parameters:**

- `model_id`: Canonical model identifier
- `variant` (optional): Model variant

**Returns:** Estimated download size in bytes

**Example:**

```python
size = ensurer.estimate_download_size("t2v-A14B@2.2.0")
print(f"Download size: {size / (1024**3):.2f} GB")
```

### ModelRegistry

Manages model manifest and specifications.

#### Constructor

```python
ModelRegistry(manifest_path: str)
```

**Parameters:**

- `manifest_path`: Path to models.toml manifest file

#### Methods

##### spec()

```python
def spec(
    self,
    model_id: str,
    variant: Optional[str] = None
) -> ModelSpec
```

Gets the specification for a model.

**Parameters:**

- `model_id`: Canonical model identifier
- `variant` (optional): Model variant

**Returns:** ModelSpec object

**Example:**

```python
registry = ModelRegistry("config/models.toml")
spec = registry.spec("t2v-A14B@2.2.0")
print(f"Model version: {spec.version}")
print(f"Available variants: {spec.variants}")
```

##### list_models()

```python
def list_models(self) -> List[str]
```

Lists all available models in the manifest.

**Returns:** List of model identifiers

**Example:**

```python
models = registry.list_models()
for model in models:
    print(f"Available: {model}")
```

##### validate_manifest()

```python
def validate_manifest(self) -> List[ValidationError]
```

Validates the manifest file for errors.

**Returns:** List of validation errors (empty if valid)

**Example:**

```python
errors = registry.validate_manifest()
if errors:
    for error in errors:
        print(f"Validation error: {error}")
```

### ModelResolver

Handles deterministic path resolution.

#### Constructor

```python
ModelResolver(models_root: str)
```

**Parameters:**

- `models_root`: Base directory for model storage

#### Methods

##### local_dir()

```python
def local_dir(
    self,
    model_id: str,
    variant: Optional[str] = None
) -> str
```

Gets the local directory path for a model.

**Parameters:**

- `model_id`: Canonical model identifier
- `variant` (optional): Model variant

**Returns:** Absolute path to model directory

**Example:**

```python
resolver = ModelResolver("/data/models")
path = resolver.local_dir("t2v-A14B@2.2.0", variant="fp16")
print(f"Model path: {path}")
```

##### temp_dir()

```python
def temp_dir(self, model_id: str) -> str
```

Gets a temporary directory path for downloading.

**Parameters:**

- `model_id`: Canonical model identifier

**Returns:** Absolute path to temporary directory

### LockManager

Manages cross-process synchronization.

#### Constructor

```python
LockManager(lock_dir: str)
```

**Parameters:**

- `lock_dir`: Directory for lock files

#### Methods

##### acquire_model_lock()

```python
def acquire_model_lock(
    self,
    model_id: str,
    timeout: float = 300.0
) -> ContextManager
```

Acquires an exclusive lock for a model.

**Parameters:**

- `model_id`: Canonical model identifier
- `timeout`: Lock acquisition timeout in seconds

**Returns:** Context manager for lock

**Example:**

```python
lock_manager = LockManager("/data/models/.locks")

with lock_manager.acquire_model_lock("t2v-A14B@2.2.0", timeout=60.0):
    # Perform exclusive operations on the model
    pass
```

##### is_locked()

```python
def is_locked(self, model_id: str) -> bool
```

Checks if a model is currently locked.

**Parameters:**

- `model_id`: Canonical model identifier

**Returns:** True if model is locked

### GarbageCollector

Manages disk space and model cleanup.

#### Constructor

```python
GarbageCollector(
    resolver: ModelResolver,
    max_total_size: Optional[int] = None,
    max_model_age: Optional[timedelta] = None
)
```

**Parameters:**

- `resolver`: ModelResolver instance
- `max_total_size`: Maximum total storage size in bytes
- `max_model_age`: Maximum age for models before cleanup

#### Methods

##### collect()

```python
def collect(
    self,
    dry_run: bool = False,
    aggressive: bool = False
) -> GCResult
```

Runs garbage collection to free disk space.

**Parameters:**

- `dry_run`: Only report what would be cleaned, don't actually delete
- `aggressive`: More aggressive cleanup (ignore access times)

**Returns:** GCResult with cleanup statistics

**Example:**

```python
gc = GarbageCollector(resolver, max_total_size=100 * 1024**3)  # 100GB

# Dry run to see what would be cleaned
result = gc.collect(dry_run=True)
print(f"Would reclaim: {result.bytes_reclaimable} bytes")

# Actually perform cleanup
result = gc.collect(dry_run=False)
print(f"Reclaimed: {result.bytes_reclaimed} bytes")
```

##### pin_model()

```python
def pin_model(
    self,
    model_id: str,
    variant: Optional[str] = None
) -> None
```

Pins a model to protect it from garbage collection.

**Parameters:**

- `model_id`: Canonical model identifier
- `variant` (optional): Model variant

##### unpin_model()

```python
def unpin_model(
    self,
    model_id: str,
    variant: Optional[str] = None
) -> None
```

Unpins a model, allowing it to be garbage collected.

**Parameters:**

- `model_id`: Canonical model identifier
- `variant` (optional): Model variant

## Data Classes

### ModelSpec

Represents a model specification from the manifest.

```python
@dataclass(frozen=True)
class ModelSpec:
    model_id: str                     # Canonical model ID
    version: str                      # Model version
    variants: List[str]               # Available variants
    default_variant: str              # Default variant
    files: List[FileSpec]             # Required files
    sources: List[str]                # Source URLs in priority order
    allow_patterns: List[str]         # File patterns for selective download
    resolution_caps: List[str]        # Supported resolutions
    optional_components: List[str]    # Optional components
    lora_required: bool               # Whether LoRA is required
```

### FileSpec

Represents a file specification within a model.

```python
@dataclass(frozen=True)
class FileSpec:
    path: str                         # Relative path within model
    size: int                         # Expected file size in bytes
    sha256: str                       # SHA256 checksum
    optional: bool = False            # Whether file is optional
```

### ModelStatus

Represents the current status of a model.

```python
@dataclass
class ModelStatus:
    model_id: str                     # Model identifier
    variant: Optional[str]            # Model variant
    state: str                        # Current state
    bytes_needed: int                 # Bytes needed to complete
    missing_files: List[str]          # List of missing files
    last_verified: Optional[datetime] # Last verification time
```

**State Values:**

- `NOT_PRESENT`: Model not downloaded
- `PARTIAL`: Partially downloaded
- `VERIFYING`: Currently being verified
- `COMPLETE`: Fully downloaded and verified
- `CORRUPT`: Downloaded but failed verification

### VerificationResult

Represents the result of integrity verification.

```python
@dataclass
class VerificationResult:
    verified: bool                    # Overall verification status
    partial_verification: bool        # Whether partial verification was used
    verified_files: List[str]         # Successfully verified files
    failed_files: List[str]           # Files that failed verification
    errors: List[str]                 # Verification errors
```

### GCResult

Represents the result of garbage collection.

```python
@dataclass
class GCResult:
    bytes_reclaimed: int              # Bytes actually reclaimed
    bytes_reclaimable: int            # Bytes that could be reclaimed
    models_removed: List[str]         # Models that were removed
    models_preserved: List[str]       # Models that were preserved
    errors: List[str]                 # Any errors during cleanup
```

## Configuration

### OrchestratorConfig

Configuration class for the orchestrator.

```python
@dataclass
class OrchestratorConfig:
    models_root: str                          # Base directory for models
    manifest_path: str                        # Path to manifest file
    max_concurrent_downloads: int = 4         # Concurrent download limit
    download_timeout: float = 3600.0          # Download timeout in seconds
    retry_attempts: int = 3                   # Number of retry attempts
    retry_backoff_factor: float = 2.0         # Exponential backoff factor
    enable_garbage_collection: bool = True    # Enable automatic GC
    max_total_size: Optional[int] = None      # Maximum total storage size
    max_model_age: Optional[timedelta] = None # Maximum model age
    hf_cache_dir: Optional[str] = None        # HuggingFace cache directory
    s3_endpoint_url: Optional[str] = None     # S3 endpoint URL
    enable_metrics: bool = True               # Enable metrics collection
    log_level: str = "INFO"                   # Logging level
```

## Storage Backends

### StorageBackend (Abstract Base)

Base class for storage backends.

```python
class StorageBackend:
    def can_handle(self, source_url: str) -> bool:
        """Check if this backend can handle the given URL."""
        raise NotImplementedError

    def download(
        self,
        source_url: str,
        local_dir: str,
        file_specs: List[FileSpec],
        progress_callback: Optional[Callable] = None
    ) -> DownloadResult:
        """Download files from the source to local directory."""
        raise NotImplementedError

    def verify_availability(self, source_url: str) -> bool:
        """Verify that the source is available."""
        raise NotImplementedError

    def estimate_download_size(
        self,
        source_url: str,
        file_specs: List[FileSpec]
    ) -> int:
        """Estimate the total download size."""
        raise NotImplementedError
```

### HFStore

HuggingFace Hub storage backend.

```python
class HFStore(StorageBackend):
    def __init__(
        self,
        token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        enable_hf_transfer: bool = True
    ):
        """Initialize HuggingFace store."""
```

**URL Format:** `hf://organization/model-name`

### S3Store

S3/MinIO storage backend.

```python
class S3Store(StorageBackend):
    def __init__(
        self,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region: str = "us-east-1"
    ):
        """Initialize S3 store."""
```

**URL Format:** `s3://bucket-name/path/to/model`

### LocalStore

Local filesystem storage backend.

```python
class LocalStore(StorageBackend):
    def __init__(self, base_path: Optional[str] = None):
        """Initialize local store."""
```

**URL Format:** `local://path/to/model`

## Exceptions

### ModelOrchestratorError

Base exception for all orchestrator errors.

```python
class ModelOrchestratorError(Exception):
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
```

### ModelNotFoundError

Raised when a requested model is not found.

```python
class ModelNotFoundError(ModelOrchestratorError):
    def __init__(self, model_id: str, available_models: Optional[List[str]] = None):
        message = f"Model not found: {model_id}"
        if available_models:
            message += f". Available models: {', '.join(available_models[:5])}"
        super().__init__(message, "MODEL_NOT_FOUND")
        self.model_id = model_id
        self.available_models = available_models
```

### InsufficientSpaceError

Raised when there's not enough disk space for an operation.

```python
class InsufficientSpaceError(ModelOrchestratorError):
    def __init__(self, required_bytes: int, available_bytes: int):
        message = f"Insufficient disk space: need {required_bytes} bytes, have {available_bytes} bytes"
        super().__init__(message, "NO_SPACE")
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes
```

### ChecksumVerificationError

Raised when file integrity verification fails.

```python
class ChecksumVerificationError(ModelOrchestratorError):
    def __init__(self, file_path: str, expected: str, actual: str):
        message = f"Checksum verification failed for {file_path}: expected {expected}, got {actual}"
        super().__init__(message, "CHECKSUM_FAIL")
        self.file_path = file_path
        self.expected_checksum = expected
        self.actual_checksum = actual
```

### LockTimeoutError

Raised when a lock cannot be acquired within the timeout period.

```python
class LockTimeoutError(ModelOrchestratorError):
    def __init__(self, model_id: str, timeout: float):
        message = f"Lock timeout for model {model_id} after {timeout} seconds"
        super().__init__(message, "LOCK_TIMEOUT")
        self.model_id = model_id
        self.timeout = timeout
```

## CLI Interface

The Model Orchestrator provides a command-line interface through the `wan models` command.

### Commands

#### status

Show the status of models.

```bash
wan models status [OPTIONS]
```

**Options:**

- `--model MODEL_ID`: Show status for specific model
- `--json`: Output in JSON format
- `--verbose`: Show detailed information

**Examples:**

```bash
# Show all models
wan models status

# Show specific model
wan models status --model t2v-A14B@2.2.0

# JSON output for automation
wan models status --json
```

#### ensure

Ensure models are downloaded and available.

```bash
wan models ensure [OPTIONS]
```

**Options:**

- `--all`: Ensure all models in manifest
- `--only MODEL_ID`: Ensure only specific model
- `--variant VARIANT`: Specify model variant
- `--force`: Force re-download even if model exists
- `--json`: Output in JSON format

**Examples:**

```bash
# Download all models
wan models ensure --all

# Download specific model
wan models ensure --only t2v-A14B@2.2.0

# Download with specific variant
wan models ensure --only t2v-A14B@2.2.0 --variant bf16
```

#### gc

Run garbage collection to free disk space.

```bash
wan models gc [OPTIONS]
```

**Options:**

- `--dry-run`: Show what would be cleaned without actually deleting
- `--aggressive`: More aggressive cleanup
- `--max-age DAYS`: Maximum age for models in days
- `--max-size SIZE`: Maximum total size (e.g., "100GB")

**Examples:**

```bash
# Dry run to see what would be cleaned
wan models gc --dry-run

# Clean models older than 30 days
wan models gc --max-age 30

# Aggressive cleanup
wan models gc --aggressive
```

#### verify

Verify model integrity.

```bash
wan models verify [OPTIONS]
```

**Options:**

- `--all`: Verify all models
- `--model MODEL_ID`: Verify specific model
- `--fix`: Attempt to fix corrupted models by re-downloading

**Examples:**

```bash
# Verify all models
wan models verify --all

# Verify and fix specific model
wan models verify --model t2v-A14B@2.2.0 --fix
```

## Integration Examples

### Basic Usage

```python
from backend.core.model_orchestrator import ModelEnsurer

# Initialize from configuration
ensurer = ModelEnsurer.from_config()

# Ensure model is available
model_path = ensurer.ensure("t2v-A14B@2.2.0", variant="fp16")

# Use model path with your pipeline
from your_pipeline import WanT2VPipeline
pipeline = WanT2VPipeline.from_pretrained(model_path)
```

### Advanced Usage with Custom Configuration

```python
from backend.core.model_orchestrator import (
    ModelEnsurer, ModelRegistry, ModelResolver,
    LockManager, GarbageCollector
)
from backend.core.model_orchestrator.storage_backends import HFStore, S3Store

# Custom configuration
registry = ModelRegistry("config/models.toml")
resolver = ModelResolver("/data/models")
lock_manager = LockManager("/data/models/.locks")

# Configure storage backends
hf_store = HFStore(token="your_hf_token")
s3_store = S3Store(
    access_key_id="your_key",
    secret_access_key="your_secret",
    endpoint_url="https://your-minio.com"
)

# Create ensurer
ensurer = ModelEnsurer(
    registry=registry,
    resolver=resolver,
    lock_manager=lock_manager,
    storage_backends=[s3_store, hf_store]  # Priority order
)

# Set up garbage collection
gc = GarbageCollector(
    resolver=resolver,
    max_total_size=500 * 1024**3,  # 500GB
    max_model_age=timedelta(days=30)
)

# Use the system
model_path = ensurer.ensure("t2v-A14B@2.2.0")

# Clean up old models
gc_result = gc.collect()
print(f"Reclaimed {gc_result.bytes_reclaimed} bytes")
```

### Error Handling

```python
from backend.core.model_orchestrator import ModelEnsurer
from backend.core.model_orchestrator.exceptions import (
    ModelNotFoundError, InsufficientSpaceError, ChecksumVerificationError
)

ensurer = ModelEnsurer.from_config()

try:
    model_path = ensurer.ensure("t2v-A14B@2.2.0")
    print(f"Model ready at: {model_path}")

except ModelNotFoundError as e:
    print(f"Model not found: {e.model_id}")
    print(f"Available models: {e.available_models}")

except InsufficientSpaceError as e:
    print(f"Need {e.required_bytes} bytes, have {e.available_bytes}")
    # Trigger cleanup or alert administrators

except ChecksumVerificationError as e:
    print(f"Integrity check failed for {e.file_path}")
    # Model will be re-downloaded automatically on retry

except Exception as e:
    print(f"Unexpected error: {e}")
```

### Progress Monitoring

```python
from backend.core.model_orchestrator import ModelEnsurer

def progress_callback(downloaded_bytes, total_bytes):
    percent = (downloaded_bytes / total_bytes) * 100
    print(f"Download progress: {percent:.1f}% ({downloaded_bytes}/{total_bytes} bytes)")

ensurer = ModelEnsurer.from_config()
model_path = ensurer.ensure(
    "t2v-A14B@2.2.0",
    progress_callback=progress_callback
)
```

This API reference provides comprehensive documentation for all public interfaces of the Model Orchestrator system.
