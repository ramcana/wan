---
title: core.model_orchestrator.model_ensurer
category: api
tags: [api, core]
---

# core.model_orchestrator.model_ensurer

Model Ensurer - Atomic download orchestration with preflight checks.

## Classes

### ModelStatus

Status of a model in the local storage.

### VerificationResult

Result of model integrity verification.

### ModelStatusInfo

Detailed status information for a model.

### FailedSource

Information about a failed source.

### ModelEnsurer

Orchestrates atomic model downloads with preflight checks.

#### Methods

##### __init__(self: Any, registry: ModelRegistry, resolver: ModelResolver, lock_manager: LockManager, storage_backends: <ast.Subscript object at 0x00000194344A3190>, safety_margin_bytes: int, negative_cache_ttl: float, retry_config: <ast.Subscript object at 0x00000194302C9930>, enable_deduplication: bool)



##### ensure(self: Any, model_id: str, variant: <ast.BinOp object at 0x0000019431B04E80>) -> str

Ensure a model is available locally, downloading if necessary.

Args:
    model_id: The model identifier
    variant: Optional model variant (defaults to model's default_variant)
    
Returns:
    str: Path to the ensured model directory
    
Raises:
    ModelOrchestratorError: If model cannot be ensured

##### get_model_status(self: Any, model_id: str, variant: <ast.BinOp object at 0x00000194302DA470>) -> ModelStatusInfo

Get the current status of a model.

Args:
    model_id: The model identifier
    variant: Optional model variant
    
Returns:
    ModelStatusInfo: Current status information

##### _download_model(self: Any, spec: ModelSpec, variant: str, local_path: Path, operation_id: str) -> str

Download a model to the local path.

##### _preflight_checks(self: Any, spec: ModelSpec, temp_dir: Path)

Perform preflight checks before downloading.

##### _download_from_source(self: Any, spec: ModelSpec, variant: str, source_url: str, temp_dir: Path, operation_id: str)

Download model files from a specific source.

##### _verify_model_integrity(self: Any, spec: ModelSpec, model_path: Path) -> VerificationResult

Verify the integrity of downloaded model files.

##### _calculate_file_hash(self: Any, file_path: Path) -> str

Calculate SHA256 hash of a file.

##### _is_source_failed(self: Any, source_url: str) -> bool

Check if a source is in the failed cache.

##### _mark_source_failed(self: Any, source_url: str, error_message: str)

Mark a source as failed in the cache.

##### cleanup_failed_downloads(self: Any)

Clean up any failed download artifacts.

##### get_download_progress(self: Any, operation_id: str) -> <ast.Subscript object at 0x000001943007D150>

Get progress information for an ongoing download.

## Constants

### NOT_PRESENT

Type: `str`

Value: `NOT_PRESENT`

### PARTIAL

Type: `str`

Value: `PARTIAL`

### VERIFYING

Type: `str`

Value: `VERIFYING`

### COMPLETE

Type: `str`

Value: `COMPLETE`

### CORRUPT

Type: `str`

Value: `CORRUPT`

