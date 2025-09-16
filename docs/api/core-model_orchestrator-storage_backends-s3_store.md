---
title: core.model_orchestrator.storage_backends.s3_store
category: api
tags: [api, core]
---

# core.model_orchestrator.storage_backends.s3_store

S3/MinIO storage backend with parallel downloads and resume capability.

## Classes

### S3Config

Configuration for S3/MinIO backend.

### S3Store

S3/MinIO storage backend with parallel downloads and resume capability.

#### Methods

##### __init__(self: Any, config: <ast.Subscript object at 0x0000019431AFAE00>, credential_manager: <ast.Subscript object at 0x0000019431AFAD40>)

Initialize S3 store.

Args:
    config: S3 configuration. If None, will use environment variables.
    credential_manager: Secure credential manager for handling credentials

##### _get_client(self: Any)

Get or create S3 client (thread-safe).

##### can_handle(self: Any, source_url: str) -> bool

Check if this backend can handle S3 URLs.

##### _parse_s3_url(self: Any, source_url: str) -> <ast.Subscript object at 0x0000019431B23280>

Parse S3 URL to extract bucket and key prefix.

Args:
    source_url: URL in format s3://bucket/key/prefix
    
Returns:
    Tuple of (bucket, key_prefix)

##### _list_objects(self: Any, bucket: str, prefix: str, allow_patterns: <ast.Subscript object at 0x0000019431B23010>) -> <ast.Subscript object at 0x0000019431B21C90>

List objects in S3 bucket with optional pattern filtering.

##### _download_file_with_resume(self: Any, bucket: str, key: str, local_path: Path, expected_size: <ast.Subscript object at 0x0000019431B21930>, progress_callback: <ast.Subscript object at 0x0000019431B21870>) -> int

Download a single file with resume capability using HTTP Range requests.

Args:
    bucket: S3 bucket name
    key: S3 object key
    local_path: Local file path to download to
    expected_size: Expected file size for validation
    progress_callback: Optional progress callback
    
Returns:
    Number of bytes downloaded

##### download(self: Any, source_url: str, local_dir: Path, file_specs: <ast.Subscript object at 0x000001942FCF7D60>, allow_patterns: <ast.Subscript object at 0x000001942FCF76D0>, progress_callback: <ast.Subscript object at 0x000001942FCF7EE0>) -> DownloadResult

Download files from S3/MinIO with parallel downloads and resume capability.

Args:
    source_url: S3 URL (s3://bucket/prefix)
    local_dir: Local directory to download to
    file_specs: List of FileSpec objects for validation
    allow_patterns: List of file patterns to download
    progress_callback: Optional progress callback
    
Returns:
    DownloadResult with download statistics

##### verify_availability(self: Any, source_url: str) -> bool

Verify if the S3 source is available.

##### estimate_download_size(self: Any, source_url: str, file_specs: <ast.Subscript object at 0x000001942F7741F0>, allow_patterns: <ast.Subscript object at 0x000001942F774130>) -> int

Estimate download size from S3 source.

