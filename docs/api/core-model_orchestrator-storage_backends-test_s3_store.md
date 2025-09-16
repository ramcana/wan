---
title: core.model_orchestrator.storage_backends.test_s3_store
category: api
tags: [api, core]
---

# core.model_orchestrator.storage_backends.test_s3_store

Tests for S3/MinIO storage backend.

## Classes

### MockFileSpec

Mock FileSpec for testing.

### TestS3Store

Test suite for S3Store.

#### Methods

##### setup_method(self: Any)

Set up test fixtures.

##### test_can_handle_s3_urls(self: Any)

Test that S3Store can handle S3 URLs.

##### test_parse_s3_url(self: Any)

Test S3 URL parsing.

##### test_client_initialization(self: Any, mock_boto3_client: Any)

Test S3 client initialization.

##### test_client_initialization_with_env_vars(self: Any, mock_boto3_client: Any)

Test S3 client initialization with environment variables.

##### test_list_objects(self: Any, mock_boto3_client: Any)

Test listing S3 objects.

##### test_list_objects_with_patterns(self: Any, mock_boto3_client: Any)

Test listing S3 objects with allow patterns.

##### test_download_file_with_resume_new_file(self: Any, mock_boto3_client: Any)

Test downloading a new file.

##### test_download_file_with_resume_partial_file(self: Any, mock_boto3_client: Any)

Test resuming download of a partial file.

##### test_download_file_already_complete(self: Any, mock_boto3_client: Any)

Test handling of already complete file.

##### test_download_with_retry(self: Any, mock_boto3_client: Any)

Test download retry logic.

##### test_download_max_retries_exceeded(self: Any, mock_boto3_client: Any)

Test download failure after max retries.

##### test_download_success(self: Any, mock_boto3_client: Any)

Test successful download of multiple files.

##### test_download_with_file_specs_validation(self: Any, mock_boto3_client: Any)

Test download with file specs validation.

##### test_download_no_objects_found(self: Any, mock_boto3_client: Any)

Test download when no objects are found.

##### test_download_partial_failure(self: Any, mock_boto3_client: Any)

Test download with some files failing.

##### test_verify_availability_success(self: Any, mock_boto3_client: Any)

Test successful availability verification.

##### test_verify_availability_no_credentials(self: Any, mock_boto3_client: Any)

Test availability verification with no credentials.

##### test_verify_availability_client_error(self: Any, mock_boto3_client: Any)

Test availability verification with client errors.

##### test_estimate_download_size(self: Any, mock_boto3_client: Any)

Test download size estimation.

##### test_estimate_download_size_with_patterns(self: Any, mock_boto3_client: Any)

Test download size estimation with allow patterns.

##### test_config_from_environment(self: Any)

Test configuration from environment variables.

##### test_thread_safety(self: Any, mock_boto3_client: Any)

Test that client initialization is thread-safe.

##### test_missing_boto3_dependency(self: Any)

Test handling of missing boto3 dependency.

### TestS3Config

Test suite for S3Config.

#### Methods

##### test_default_config(self: Any)

Test default configuration values.

##### test_custom_config(self: Any)

Test custom configuration values.

