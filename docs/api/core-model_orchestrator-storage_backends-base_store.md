---
title: core.model_orchestrator.storage_backends.base_store
category: api
tags: [api, core]
---

# core.model_orchestrator.storage_backends.base_store

Base storage backend interface.

## Classes

### DownloadResult

Result of a download operation.

### StorageBackend

Abstract base class for storage backends.

#### Methods

##### can_handle(self: Any, source_url: str) -> bool

Check if this backend can handle the given source URL.

##### download(self: Any, source_url: str, local_dir: Path, file_specs: <ast.Subscript object at 0x000001942FCF7EE0>, allow_patterns: <ast.Subscript object at 0x000001942FCF7E20>, progress_callback: <ast.Subscript object at 0x000001942FCF7D00>) -> DownloadResult

Download files from source to local directory.

##### verify_availability(self: Any, source_url: str) -> bool

Verify if the source is available.

##### estimate_download_size(self: Any, source_url: str, file_specs: <ast.Subscript object at 0x000001942FCF76D0>, allow_patterns: <ast.Subscript object at 0x000001942FCF7610>) -> int

Estimate the total download size in bytes.

