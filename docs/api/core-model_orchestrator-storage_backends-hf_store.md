---
title: core.model_orchestrator.storage_backends.hf_store
category: api
tags: [api, core]
---

# core.model_orchestrator.storage_backends.hf_store

HuggingFace Hub storage backend.

## Classes

### HFFileMetadata

HuggingFace file metadata for integrity verification.

### HFStore

HuggingFace Hub storage backend using huggingface_hub.

#### Methods

##### __init__(self: Any, token: <ast.Subscript object at 0x000001942F777550>, enable_hf_transfer: bool, credential_manager: <ast.Subscript object at 0x000001942F777430>)

Initialize HuggingFace store.

Args:
    token: HuggingFace API token (if None, will try to get from credential manager or environment)
    enable_hf_transfer: Whether to enable hf_transfer for faster downloads
    credential_manager: Secure credential manager for handling credentials

##### can_handle(self: Any, source_url: str) -> bool

Check if this backend can handle HuggingFace URLs.

##### _parse_hf_url(self: Any, source_url: str) -> <ast.Subscript object at 0x000001942F826B30>

Parse HuggingFace URL to extract repo_id and revision.

Args:
    source_url: URL in format hf://repo_id or hf://repo_id@revision
    
Returns:
    Tuple of (repo_id, revision)

##### download(self: Any, source_url: str, local_dir: Path, file_specs: <ast.Subscript object at 0x000001942F826740>, allow_patterns: <ast.Subscript object at 0x000001942F826710>, progress_callback: <ast.Subscript object at 0x000001942F8254B0>) -> DownloadResult

Download model from HuggingFace Hub.

Args:
    source_url: HuggingFace URL (hf://repo_id or hf://repo_id@revision)
    local_dir: Local directory to download to
    file_specs: List of FileSpec objects (not used for HF, patterns used instead)
    allow_patterns: List of file patterns to download
    progress_callback: Optional progress callback
    
Returns:
    DownloadResult with download statistics

##### verify_availability(self: Any, source_url: str) -> bool

Verify if the HuggingFace repository is available.

##### estimate_download_size(self: Any, source_url: str, file_specs: <ast.Subscript object at 0x00000194344C31F0>, allow_patterns: <ast.Subscript object at 0x00000194344C32B0>) -> int

Estimate download size from HuggingFace repository.

Note: This is a rough estimate as HuggingFace doesn't provide
easy access to file sizes without downloading. Returns 0 for now.

##### _calculate_directory_size(self: Any, directory: Path) -> int

Calculate total size of all files in directory.

##### _count_files(self: Any, directory: Path) -> int

Count total number of files in directory.

##### _extract_file_metadata(self: Any, directory: Path) -> <ast.Subscript object at 0x000001942FCA1360>

Extract file metadata from downloaded HuggingFace files.

This attempts to extract ETags and other metadata that can be used
for integrity verification when SHA256 checksums are not available.

##### get_file_metadata(self: Any, source_url: str, file_path: str) -> <ast.Subscript object at 0x000001942FCA0850>

Get metadata for a specific file from HuggingFace Hub.

This can be used to get ETag information before downloading
for integrity verification purposes.

