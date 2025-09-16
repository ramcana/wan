---
title: core.model_orchestrator.storage_backends.test_hf_store
category: api
tags: [api, core]
---

# core.model_orchestrator.storage_backends.test_hf_store

Tests for HuggingFace storage backend.

## Classes

### TestHFStore

Test cases for HFStore.

#### Methods

##### setup_method(self: Any)

Set up test fixtures.

##### teardown_method(self: Any)

Clean up test fixtures.

##### test_can_handle_hf_urls(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test that HFStore can handle HuggingFace URLs.

##### test_parse_hf_url(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test parsing of HuggingFace URLs.

##### test_parse_hf_url_invalid(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test parsing of invalid HuggingFace URLs.

##### test_download_success(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test successful download from HuggingFace.

##### test_download_with_revision(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test download with specific revision.

##### test_download_hf_hub_error(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test handling of HuggingFace Hub errors.

##### test_download_generic_error(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test handling of generic errors during download.

##### test_verify_availability_success(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test successful repository availability check.

##### test_verify_availability_with_revision(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test repository availability check with revision.

##### test_verify_availability_not_found(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test repository availability check for non-existent repo.

##### test_verify_availability_auth_required(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test repository availability check for private repo.

##### test_estimate_download_size(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test download size estimation.

##### test_calculate_directory_size(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test directory size calculation.

##### test_count_files(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test file counting.

##### test_token_from_environment(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test token loading from environment variable.

##### test_token_from_parameter(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test token from constructor parameter.

##### test_no_token(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test behavior when no token is provided.

##### test_missing_huggingface_hub(self: Any)

Test error when huggingface_hub is not installed.

##### test_hf_transfer_disabled(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test behavior when hf_transfer is explicitly disabled.

##### test_hf_transfer_fallback_warning(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test that warning is logged once when hf_transfer is unavailable.

##### test_hf_transfer_available(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test behavior when hf_transfer is available.

##### test_hf_transfer_environment_variable_set(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test that HF_HUB_ENABLE_HF_TRANSFER environment variable is set when hf_transfer is available.

##### test_download_continues_without_hf_transfer(self: Any, mock_repo_info: Any, mock_snapshot_download: Any)

Test that download continues normally even when hf_transfer is unavailable.

