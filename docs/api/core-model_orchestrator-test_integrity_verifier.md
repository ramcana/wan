---
title: core.model_orchestrator.test_integrity_verifier
category: api
tags: [api, core]
---

# core.model_orchestrator.test_integrity_verifier

Tests for comprehensive integrity verification system.

## Classes

### TestIntegrityVerifier

Test suite for IntegrityVerifier.

#### Methods

##### temp_dir(self: Any)

Create a temporary directory for testing.

##### sample_files(self: Any, temp_dir: Any)

Create sample files with known content and checksums.

##### sample_model_spec(self: Any, sample_files: Any)

Create a sample model spec for testing.

##### verifier(self: Any)

Create an IntegrityVerifier instance.

##### test_sha256_verification_success(self: Any, verifier: Any, temp_dir: Any, sample_files: Any, sample_model_spec: Any)

Test successful SHA256 checksum verification.

##### test_sha256_verification_checksum_mismatch(self: Any, verifier: Any, temp_dir: Any, sample_files: Any, sample_model_spec: Any)

Test SHA256 verification with checksum mismatch.

##### test_size_mismatch_verification(self: Any, verifier: Any, temp_dir: Any, sample_files: Any, sample_model_spec: Any)

Test verification with size mismatch.

##### test_missing_files_verification(self: Any, verifier: Any, temp_dir: Any, sample_model_spec: Any)

Test verification with missing files.

##### test_hf_etag_fallback_verification(self: Any, verifier: Any, temp_dir: Any, sample_files: Any)

Test HuggingFace ETag fallback verification when SHA256 is not available.

##### test_hf_etag_mismatch(self: Any, verifier: Any, temp_dir: Any, sample_files: Any)

Test HuggingFace ETag verification with mismatch.

##### test_size_only_fallback(self: Any, verifier: Any, temp_dir: Any, sample_files: Any)

Test size-only verification as last resort.

##### test_manifest_signature_verification_success(self: Any, verifier: Any, temp_dir: Any, sample_model_spec: Any)

Test successful manifest signature verification.

##### test_manifest_signature_verification_failure(self: Any, verifier: Any, temp_dir: Any, sample_model_spec: Any)

Test failed manifest signature verification.

##### test_retry_logic_on_verification_failure(self: Any, temp_dir: Any, sample_files: Any, sample_model_spec: Any)

Test retry logic when verification initially fails.

##### test_verification_performance_metrics(self: Any, verifier: Any, temp_dir: Any, sample_files: Any, sample_model_spec: Any)

Test that verification includes performance metrics.

##### test_canonical_spec_data_creation(self: Any, verifier: Any, sample_model_spec: Any)

Test creation of canonical spec data for signature verification.

##### test_extract_hf_metadata_from_download(self: Any, verifier: Any, temp_dir: Any, sample_files: Any)

Test extraction of HF metadata from downloaded files.

##### test_comprehensive_failure_scenario(self: Any, verifier: Any, temp_dir: Any, sample_model_spec: Any)

Test comprehensive failure scenario with multiple issues.

##### test_large_file_chunked_processing(self: Any, verifier: Any, temp_dir: Any)

Test that large files are processed in chunks efficiently.

##### test_concurrent_verification_safety(self: Any, verifier: Any, temp_dir: Any, sample_files: Any, sample_model_spec: Any)

Test that verification is safe for concurrent access.

##### test_error_handling_and_logging(self: Any, verifier: Any, temp_dir: Any, sample_model_spec: Any)

Test proper error handling and logging during verification.

### TestIntegrityVerifierEdgeCases

Test edge cases and error conditions for IntegrityVerifier.

#### Methods

##### verifier(self: Any)



##### test_empty_model_directory(self: Any, verifier: Any, tmp_path: Any)

Test verification of empty model directory.

##### test_nonexistent_model_directory(self: Any, verifier: Any)

Test verification of nonexistent model directory.

##### test_zero_byte_files(self: Any, verifier: Any, tmp_path: Any)

Test verification of zero-byte files.

##### test_unicode_file_paths(self: Any, verifier: Any, tmp_path: Any)

Test verification with Unicode file paths.

