---
title: core.model_orchestrator.test_model_ensurer
category: api
tags: [api, core]
---

# core.model_orchestrator.test_model_ensurer

Integration tests for ModelEnsurer - atomic download orchestration with preflight checks.

## Classes

### MockStorageBackend

Mock storage backend for testing.

#### Methods

##### __init__(self: Any, url_prefix: str, should_fail: bool)



##### can_handle(self: Any, source_url: str) -> bool



##### download(self: Any, source_url: str, local_dir: Path, file_specs: <ast.Subscript object at 0x000001942CC88400>, allow_patterns: <ast.Subscript object at 0x000001942CC8AA10>, progress_callback: <ast.Subscript object at 0x000001942CC88190>) -> DownloadResult



##### verify_availability(self: Any, source_url: str) -> bool



##### estimate_download_size(self: Any, source_url: str, file_specs: <ast.Subscript object at 0x000001942CC8B970>, allow_patterns: <ast.Subscript object at 0x000001942CC8A800>) -> int



##### _generate_content_for_spec(self: Any, file_spec: FileSpec) -> bytes

Generate content that matches the file spec.

### TestModelEnsurer

Test cases for ModelEnsurer.

#### Methods

##### test_ensure_new_model_success(self: Any, model_ensurer: Any, mock_backend: Any, temp_dir: Any)

Test successful download of a new model.

##### test_ensure_existing_complete_model(self: Any, model_ensurer: Any, mock_backend: Any, temp_dir: Any)

Test that existing complete model is not re-downloaded.

##### test_ensure_force_redownload(self: Any, model_ensurer: Any, mock_backend: Any, temp_dir: Any)

Test force redownload of existing model.

##### test_ensure_disk_space_check_failure(self: Any, model_ensurer: Any, mock_resolver: Any, temp_dir: Any)

Test that insufficient disk space raises NoSpaceError.

##### test_ensure_disk_space_with_garbage_collection(self: Any, model_ensurer: Any, mock_resolver: Any, temp_dir: Any)

Test that garbage collection is triggered when disk space is insufficient.

##### test_ensure_disk_space_gc_insufficient_reclaim(self: Any, model_ensurer: Any, mock_resolver: Any, temp_dir: Any)

Test that NoSpaceError is still raised if GC doesn't free enough space.

##### test_ensure_download_failure_fallback(self: Any, model_ensurer: Any, mock_registry: Any, temp_dir: Any)

Test fallback to secondary source when primary fails.

##### test_ensure_all_sources_fail(self: Any, model_ensurer: Any)

Test that failure of all sources raises appropriate error.

##### test_ensure_checksum_verification_failure(self: Any, model_ensurer: Any, mock_backend: Any, temp_dir: Any)

Test that checksum verification failure raises ChecksumError.

##### test_ensure_size_mismatch_failure(self: Any, model_ensurer: Any, mock_backend: Any, temp_dir: Any)

Test that size mismatch raises SizeMismatchError.

##### test_ensure_missing_file_failure(self: Any, model_ensurer: Any, mock_backend: Any, temp_dir: Any)

Test that missing files after download raise IncompleteDownloadError.

##### test_status_not_present(self: Any, model_ensurer: Any)

Test status for non-existent model.

##### test_status_complete_with_verification_marker(self: Any, model_ensurer: Any, temp_dir: Any)

Test status for complete model with verification marker.

##### test_status_partial(self: Any, model_ensurer: Any, temp_dir: Any)

Test status for partially downloaded model.

##### test_status_corrupt(self: Any, model_ensurer: Any, temp_dir: Any)

Test status for model with corrupt files.

##### test_verify_integrity_success(self: Any, model_ensurer: Any, temp_dir: Any)

Test successful integrity verification.

##### test_verify_integrity_missing_files(self: Any, model_ensurer: Any, temp_dir: Any)

Test integrity verification with missing files.

##### test_verify_integrity_checksum_mismatch(self: Any, model_ensurer: Any, temp_dir: Any)

Test integrity verification with checksum mismatch.

##### test_estimate_download_size(self: Any, model_ensurer: Any)

Test download size estimation.

##### test_concurrent_downloads(self: Any, model_ensurer: Any, mock_backend: Any, mock_lock_manager: Any, temp_dir: Any)

Test that concurrent downloads are properly synchronized.

##### test_progress_callback(self: Any, model_ensurer: Any, mock_backend: Any, temp_dir: Any)

Test that progress callback is called during download.

##### test_temp_directory_cleanup(self: Any, model_ensurer: Any, mock_backend: Any, temp_dir: Any)

Test that temporary directories are cleaned up after download.

##### test_temp_directory_cleanup_on_failure(self: Any, model_ensurer: Any, mock_backend: Any, temp_dir: Any)

Test that temporary directories are cleaned up even on failure.

##### test_negative_cache_skips_recently_failed_sources(self: Any, model_ensurer: Any, mock_registry: Any, temp_dir: Any)

Test that negative cache skips recently failed sources.

##### test_negative_cache_expires_after_ttl(self: Any, model_ensurer: Any, mock_registry: Any, temp_dir: Any)

Test that negative cache entries expire after TTL.

##### test_negative_cache_clears_on_success(self: Any, model_ensurer: Any, mock_registry: Any, temp_dir: Any)

Test that negative cache is cleared when a source succeeds.

##### test_negative_cache_cleanup_expired_entries(self: Any, model_ensurer: Any)

Test that expired entries are cleaned up from negative cache.

##### test_status_verifying_state(self: Any, model_ensurer: Any, temp_dir: Any)

Test status returns VERIFYING when verification is in progress.

##### test_verify_integrity_model_not_exist(self: Any, model_ensurer: Any)

Test verify_integrity when model directory doesn't exist.

##### test_verify_integrity_size_mismatch(self: Any, model_ensurer: Any, temp_dir: Any)

Test verify_integrity with size mismatch.

##### test_api_error_codes_structure(self: Any, model_ensurer: Any)

Test that API methods return structured error codes.

##### test_ensure_with_variant_parameter(self: Any, model_ensurer: Any, mock_backend: Any, temp_dir: Any)

Test ensure method with explicit variant parameter.

##### test_status_with_variant_parameter(self: Any, model_ensurer: Any)

Test status method with explicit variant parameter.

##### test_verify_integrity_with_variant_parameter(self: Any, model_ensurer: Any)

Test verify_integrity method with explicit variant parameter.

