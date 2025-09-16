---
title: core.model_orchestrator.test_enhanced_integrity
category: api
tags: [api, core]
---

# core.model_orchestrator.test_enhanced_integrity

Tests for enhanced integrity verification in ModelEnsurer.

## Classes

### MockStorageBackend

Mock storage backend for testing.

#### Methods

##### __init__(self: Any, should_succeed: Any, metadata: Any)



##### can_handle(self: Any, source_url: str) -> bool



##### download(self: Any, source_url: Any, local_dir: Any, file_specs: Any, allow_patterns: Any, progress_callback: Any)



##### verify_availability(self: Any, source_url: str) -> bool



##### estimate_download_size(self: Any, source_url: Any, file_specs: Any, allow_patterns: Any) -> int



### TestEnhancedIntegrityVerification

Test enhanced integrity verification in ModelEnsurer.

#### Methods

##### temp_dir(self: Any)



##### mock_registry(self: Any)



##### mock_resolver(self: Any, temp_dir: Any)



##### mock_lock_manager(self: Any)



##### test_successful_download_with_sha256_verification(self: Any, temp_dir: Any, mock_registry: Any, mock_resolver: Any, mock_lock_manager: Any)

Test successful download with SHA256 verification.

##### test_download_with_checksum_failure_and_retry(self: Any, temp_dir: Any, mock_registry: Any, mock_resolver: Any, mock_lock_manager: Any)

Test download with checksum failure that triggers retry.

##### test_download_with_hf_etag_fallback(self: Any, temp_dir: Any, mock_registry: Any, mock_resolver: Any, mock_lock_manager: Any)

Test download with HuggingFace ETag fallback verification.

##### test_download_with_size_mismatch_failure(self: Any, temp_dir: Any, mock_registry: Any, mock_resolver: Any, mock_lock_manager: Any)

Test download failure due to size mismatch.

##### test_verify_integrity_method_with_comprehensive_verification(self: Any, temp_dir: Any, mock_registry: Any, mock_resolver: Any, mock_lock_manager: Any)

Test the verify_integrity method uses comprehensive verification.

##### test_verify_integrity_with_missing_files(self: Any, temp_dir: Any, mock_registry: Any, mock_resolver: Any, mock_lock_manager: Any)

Test verify_integrity with missing files.

##### test_verify_integrity_with_corrupted_files(self: Any, temp_dir: Any, mock_registry: Any, mock_resolver: Any, mock_lock_manager: Any)

Test verify_integrity with corrupted files.

##### test_status_method_with_comprehensive_verification(self: Any, temp_dir: Any, mock_registry: Any, mock_resolver: Any, mock_lock_manager: Any)

Test that status method works with comprehensive verification.

##### test_integrity_verification_with_manifest_signature(self: Any, temp_dir: Any, mock_registry: Any, mock_resolver: Any, mock_lock_manager: Any)

Test integrity verification with manifest signature validation.

##### test_error_recovery_during_integrity_verification(self: Any, temp_dir: Any, mock_registry: Any, mock_resolver: Any, mock_lock_manager: Any)

Test error recovery during integrity verification.

##### test_metrics_recording_for_integrity_failures(self: Any, temp_dir: Any, mock_registry: Any, mock_resolver: Any, mock_lock_manager: Any)

Test that integrity failures are properly recorded in metrics.

### TestIntegrityVerificationEdgeCases

Test edge cases for integrity verification.

#### Methods

##### test_empty_file_verification(self: Any, tmp_path: Any)

Test verification of empty files.

##### test_very_large_file_verification(self: Any, tmp_path: Any)

Test verification of very large files.

##### test_unicode_filename_verification(self: Any, tmp_path: Any)

Test verification with Unicode filenames.

##### test_concurrent_verification_safety(self: Any, tmp_path: Any)

Test that concurrent verification operations are safe.

### CorrectContentBackend



#### Methods

##### download(self: Any, source_url: Any, local_dir: Any, file_specs: Any)



### RetryableBackend



#### Methods

##### __init__(self: Any)



##### download(self: Any, source_url: Any, local_dir: Any, file_specs: Any)



### HFMetadataBackend



#### Methods

##### download(self: Any, source_url: Any, local_dir: Any, file_specs: Any)



### WrongSizeBackend



#### Methods

##### download(self: Any, source_url: Any, local_dir: Any, file_specs: Any)



### FailingVerificationBackend



#### Methods

##### __init__(self: Any)



##### download(self: Any, source_url: Any, local_dir: Any, file_specs: Any)



