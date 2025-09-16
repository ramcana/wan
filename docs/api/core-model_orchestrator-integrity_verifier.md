---
title: core.model_orchestrator.integrity_verifier
category: api
tags: [api, core]
---

# core.model_orchestrator.integrity_verifier

Comprehensive integrity verification system for model files.

This module provides enhanced integrity verification including SHA256 checksums,
HuggingFace ETag verification, manifest signature verification, and retry logic
for handling various integrity failure scenarios.

## Classes

### VerificationMethod

Available verification methods in order of preference.

### FileVerificationResult

Result of verifying a single file.

### IntegrityVerificationResult

Result of comprehensive integrity verification.

### HFFileMetadata

HuggingFace file metadata for ETag verification.

### IntegrityVerifier

Comprehensive integrity verification system.

Provides multiple verification methods with fallback support:
1. SHA256 checksum verification (preferred)
2. HuggingFace ETag verification (fallback)
3. Size-only verification (last resort)

#### Methods

##### __init__(self: Any, retry_config: <ast.Subscript object at 0x000001943459CA30>)

Initialize the integrity verifier.

Args:
    retry_config: Configuration for retry logic on verification failures

##### verify_model_integrity(self: Any, spec: ModelSpec, model_dir: Path, hf_metadata: <ast.Subscript object at 0x000001943459C040>, manifest_signature: <ast.Subscript object at 0x000001943459D870>, public_key: <ast.Subscript object at 0x000001943459DA20>) -> IntegrityVerificationResult

Perform comprehensive integrity verification for a model.

Args:
    spec: Model specification with file requirements
    model_dir: Local directory containing the model files
    hf_metadata: Optional HuggingFace metadata for ETag verification
    manifest_signature: Optional manifest signature for verification
    public_key: Optional public key for signature verification
    
Returns:
    IntegrityVerificationResult with detailed verification results

##### _verify_single_file(self: Any, file_spec: FileSpec, file_path: Path, hf_metadata: <ast.Subscript object at 0x00000194319C4550>) -> FileVerificationResult

Verify a single file using the best available method.

Args:
    file_spec: File specification with expected values
    file_path: Path to the actual file
    hf_metadata: Optional HuggingFace metadata for ETag verification
    
Returns:
    FileVerificationResult with verification details

##### _calculate_sha256(self: Any, file_path: Path) -> str

Calculate SHA256 checksum of a file with optimized chunking.

##### _calculate_etag_equivalent(self: Any, file_path: Path) -> str

Calculate ETag equivalent (typically MD5) for HuggingFace verification.

Note: HuggingFace ETags are usually MD5 hashes, but can be more complex
for multipart uploads. This implementation handles simple cases.

##### _verify_manifest_signature(self: Any, spec: ModelSpec, signature: str, public_key: str) -> bool

Verify manifest signature using HMAC-SHA256.

Args:
    spec: Model specification to verify
    signature: Base64-encoded signature
    public_key: Public key for verification
    
Returns:
    True if signature is valid, False otherwise

##### _create_canonical_spec_data(self: Any, spec: ModelSpec) -> str

Create a canonical string representation of model spec for signing.

This ensures consistent signature verification across different
serialization formats and field ordering.

##### extract_hf_metadata_from_download(self: Any, downloaded_files: <ast.Subscript object at 0x000001942FC41ED0>, hf_cache_dir: <ast.Subscript object at 0x000001942FC41F90>) -> <ast.Subscript object at 0x000001942FC43DC0>

Extract HuggingFace metadata from downloaded files.

This method attempts to extract ETag and other metadata from
HuggingFace cache files or HTTP headers if available.

Args:
    downloaded_files: List of downloaded file paths
    hf_cache_dir: Optional HuggingFace cache directory
    
Returns:
    Dictionary mapping file paths to HFFileMetadata

## Constants

### SHA256_CHECKSUM

Type: `str`

Value: `sha256_checksum`

### HF_ETAG

Type: `str`

Value: `hf_etag`

### SIZE_ONLY

Type: `str`

Value: `size_only`

