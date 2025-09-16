---
title: core.model_orchestrator.exceptions
category: api
tags: [api, core]
---

# core.model_orchestrator.exceptions

Exception classes for the Model Orchestrator system.

## Classes

### ErrorCode

Structured error codes for consistent error handling.

### ModelOrchestratorError

Base exception for all Model Orchestrator errors.

#### Methods

##### __init__(self: Any, message: str, error_code: ErrorCode, details: <ast.Subscript object at 0x00000194340978B0>)



### ModelNotFoundError

Raised when a requested model is not found in the manifest.

#### Methods

##### __init__(self: Any, model_id: str, available_models: <ast.Subscript object at 0x0000019434097460>)



### VariantNotFoundError

Raised when a requested variant is not available for a model.

#### Methods

##### __init__(self: Any, model_id: str, variant: str, available_variants: <ast.Subscript object at 0x0000019434097160>)



### InvalidModelIdError

Raised when a model ID format is invalid.

#### Methods

##### __init__(self: Any, model_id: str, reason: str)



### ManifestValidationError

Raised when manifest validation fails.

#### Methods

##### __init__(self: Any, errors: <ast.Subscript object at 0x0000019434096110>)



### SchemaVersionError

Raised when manifest schema version is incompatible.

#### Methods

##### __init__(self: Any, found_version: str, supported_versions: <ast.Subscript object at 0x0000019434115930>)



### LockTimeoutError

Raised when a lock cannot be acquired within the specified timeout.

#### Methods

##### __init__(self: Any, message: str, model_id: <ast.Subscript object at 0x0000019434114C40>, timeout: <ast.Subscript object at 0x0000019434114E50>)



### LockError

Raised when a lock operation fails.

#### Methods

##### __init__(self: Any, message: str, model_id: <ast.Subscript object at 0x0000019434114B80>)



### NoSpaceError

Raised when insufficient disk space is available.

#### Methods

##### __init__(self: Any, bytes_needed: int, bytes_available: int, path: str)



### ChecksumError

Raised when file checksum verification fails.

#### Methods

##### __init__(self: Any, file_path: str, expected: str, actual: str)



### SizeMismatchError

Raised when file size doesn't match expected size.

#### Methods

##### __init__(self: Any, file_path: str, expected: int, actual: int)



### IncompleteDownloadError

Raised when a download is incomplete.

#### Methods

##### __init__(self: Any, message: str, missing_files: <ast.Subscript object at 0x0000019434116DA0>)



### IntegrityVerificationError

Raised when comprehensive integrity verification fails.

#### Methods

##### __init__(self: Any, message: str, failed_files: <ast.Subscript object at 0x0000019434272EC0>, missing_files: <ast.Subscript object at 0x0000019434272FE0>)



### ManifestSignatureError

Raised when manifest signature verification fails.

#### Methods

##### __init__(self: Any, message: str, model_id: <ast.Subscript object at 0x0000019434273460>)



### ModelValidationError

Raised when model validation fails.

#### Methods

##### __init__(self: Any, message: str, model_id: <ast.Subscript object at 0x0000019434273AF0>)



### InvalidInputError

Raised when input validation fails for a model type.

#### Methods

##### __init__(self: Any, message: str, model_type: <ast.Subscript object at 0x0000019434273FA0>, input_data: <ast.Subscript object at 0x0000019434272D70>)



### MigrationError

Raised when configuration migration fails.

#### Methods

##### __init__(self: Any, message: str, details: <ast.Subscript object at 0x000001942FBE2530>)



### ValidationError

Raised when validation fails.

#### Methods

##### __init__(self: Any, message: str, details: <ast.Subscript object at 0x000001942FBE20E0>)



## Constants

### INVALID_CONFIG

Type: `str`

Value: `INVALID_CONFIG`

### MISSING_MANIFEST

Type: `str`

Value: `MISSING_MANIFEST`

### SCHEMA_VERSION_MISMATCH

Type: `str`

Value: `SCHEMA_VERSION_MISMATCH`

### MODEL_NOT_FOUND

Type: `str`

Value: `MODEL_NOT_FOUND`

### VARIANT_NOT_FOUND

Type: `str`

Value: `VARIANT_NOT_FOUND`

### INVALID_MODEL_ID

Type: `str`

Value: `INVALID_MODEL_ID`

### MANIFEST_VALIDATION_ERROR

Type: `str`

Value: `MANIFEST_VALIDATION_ERROR`

### PATH_TRAVERSAL_DETECTED

Type: `str`

Value: `PATH_TRAVERSAL_DETECTED`

### CASE_COLLISION_DETECTED

Type: `str`

Value: `CASE_COLLISION_DETECTED`

### NO_SPACE

Type: `str`

Value: `NO_SPACE`

### PERMISSION_DENIED

Type: `str`

Value: `PERMISSION_DENIED`

### PATH_TOO_LONG

Type: `str`

Value: `PATH_TOO_LONG`

### FILESYSTEM_ERROR

Type: `str`

Value: `FILESYSTEM_ERROR`

### AUTH_FAIL

Type: `str`

Value: `AUTH_FAIL`

### RATE_LIMIT

Type: `str`

Value: `RATE_LIMIT`

### NETWORK_TIMEOUT

Type: `str`

Value: `NETWORK_TIMEOUT`

### SOURCE_UNAVAILABLE

Type: `str`

Value: `SOURCE_UNAVAILABLE`

### CHECKSUM_FAIL

Type: `str`

Value: `CHECKSUM_FAIL`

### SIZE_MISMATCH

Type: `str`

Value: `SIZE_MISMATCH`

### INCOMPLETE_DOWNLOAD

Type: `str`

Value: `INCOMPLETE_DOWNLOAD`

### INTEGRITY_VERIFICATION_ERROR

Type: `str`

Value: `INTEGRITY_VERIFICATION_ERROR`

### MANIFEST_SIGNATURE_ERROR

Type: `str`

Value: `MANIFEST_SIGNATURE_ERROR`

### LOCK_TIMEOUT

Type: `str`

Value: `LOCK_TIMEOUT`

### LOCK_ERROR

Type: `str`

Value: `LOCK_ERROR`

### CONCURRENT_MODIFICATION

Type: `str`

Value: `CONCURRENT_MODIFICATION`

### UNKNOWN_ERROR

Type: `str`

Value: `UNKNOWN_ERROR`

