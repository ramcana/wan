---
title: core.model_orchestrator.test_concurrent_ensure
category: api
tags: [api, core]
---

# core.model_orchestrator.test_concurrent_ensure

Test for concurrent ensure() calls - validates that two concurrent ensure() calls
for the same model result in one downloading while the other waits, and both succeed.

## Classes

### MockStorageBackend

Mock storage backend that simulates slow downloads.

#### Methods

##### __init__(self: Any, download_delay: float)



##### can_handle(self: Any, source_url: str) -> bool



##### download(self: Any, source_url: str, local_dir: Path, file_specs: Any, allow_patterns: Any, progress_callback: Any) -> DownloadResult

Simulate a slow download with configurable delay.

##### verify_availability(self: Any, source_url: str) -> bool



##### estimate_download_size(self: Any, source_url: str, file_specs: Any, allow_patterns: Any) -> int



### TestConcurrentEnsure

Test concurrent ensure() calls for the same model.

#### Methods

##### test_concurrent_ensure_same_model_one_downloads_one_waits(self: Any, model_ensurer: Any, mock_backend: Any, temp_dir: Any)

Test that two concurrent ensure() calls for the same model result in:
1. One call performing the download
2. The other call waiting for the first to complete
3. Both calls succeeding and returning the same path

##### test_concurrent_ensure_different_variants(self: Any, model_ensurer: Any, mock_backend: Any, temp_dir: Any)

Test concurrent ensure() calls for the same model but different variants.
Each variant should be downloaded separately.

##### test_concurrent_ensure_with_existing_complete_model(self: Any, model_ensurer: Any, mock_backend: Any, temp_dir: Any, mock_registry: Any)

Test concurrent ensure() calls when the model is already complete.
Neither call should trigger a download.

##### test_concurrent_ensure_with_force_redownload(self: Any, model_ensurer: Any, mock_backend: Any, temp_dir: Any)

Test concurrent ensure() calls with force_redownload=True.
Should still only download once due to locking.

##### test_sequential_ensure_calls_use_cached_result(self: Any, model_ensurer: Any, mock_backend: Any, temp_dir: Any)

Test that sequential ensure() calls use the cached result from the first call.

