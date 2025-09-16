---
title: core.model_orchestrator.test_security_credentials
category: api
tags: [api, core]
---

# core.model_orchestrator.test_security_credentials

Comprehensive security tests for credential management in the Model Orchestrator.

These tests verify secure credential storage, retrieval, masking, and handling
of presigned URLs and temporary access tokens.

## Classes

### TestCredentialMasker

Test credential masking functionality.

#### Methods

##### test_mask_api_keys(self: Any)

Test masking of API keys in text.

##### test_mask_aws_credentials(self: Any)

Test masking of AWS credentials.

##### test_mask_urls_with_credentials(self: Any)

Test masking of URLs containing credentials.

##### test_mask_presigned_urls(self: Any)

Test masking of presigned URL signatures.

##### test_mask_hf_tokens(self: Any)

Test masking of HuggingFace tokens.

##### test_mask_dict_values(self: Any)

Test masking of sensitive values in dictionaries.

##### test_mask_short_values(self: Any)

Test masking of short sensitive values.

##### test_preserve_non_sensitive_data(self: Any)

Test that non-sensitive data is preserved.

### TestCredentialStore

Test secure credential storage.

#### Methods

##### temp_credential_file(self: Any)

Create temporary credential file.

##### credential_config(self: Any, temp_credential_file: Any)

Create credential configuration for testing.

##### test_store_and_retrieve_credential(self: Any, credential_config: Any)

Test storing and retrieving credentials.

##### test_credential_not_found(self: Any, credential_config: Any)

Test retrieving non-existent credential.

##### test_delete_credential(self: Any, credential_config: Any)

Test deleting credentials.

##### test_environment_fallback(self: Any, credential_config: Any)

Test fallback to environment variables.

##### test_list_credentials(self: Any, credential_config: Any)

Test listing stored credentials.

##### test_keyring_storage(self: Any, credential_config: Any)

Test keyring-based credential storage.

### TestPresignedURLManager

Test presigned URL management.

#### Methods

##### url_manager(self: Any)

Create presigned URL manager.

##### test_generate_presigned_url(self: Any, url_manager: Any)

Test generating presigned URLs.

##### test_url_expiration_check(self: Any, url_manager: Any)

Test URL expiration checking.

##### test_refresh_url_if_needed(self: Any, url_manager: Any)

Test URL refresh when expiring soon.

##### test_clean_expired_cache(self: Any, url_manager: Any)

Test cleaning expired URLs from cache.

### TestSecureCredentialManager

Test the main credential manager.

#### Methods

##### credential_manager(self: Any)

Create secure credential manager.

##### test_get_s3_credentials(self: Any, credential_manager: Any)

Test getting S3 credentials.

##### test_get_hf_credentials(self: Any, credential_manager: Any)

Test getting HuggingFace credentials.

##### test_mask_sensitive_info_string(self: Any, credential_manager: Any)

Test masking sensitive information in strings.

##### test_mask_sensitive_info_dict(self: Any, credential_manager: Any)

Test masking sensitive information in dictionaries.

##### test_setup_credentials(self: Any, credential_manager: Any)

Test setting up multiple credentials.

##### test_https_validation(self: Any, credential_manager: Any)

Test HTTPS requirement validation.

##### test_secure_context(self: Any, credential_manager: Any)

Test secure credential context manager.

### TestSecureCredentialContext

Test secure credential context manager.

#### Methods

##### test_temporary_environment_variables(self: Any)

Test temporary setting of environment variables.

##### test_cleanup_on_exception(self: Any)

Test that environment is cleaned up even on exceptions.

### TestModelEncryption

Test model encryption functionality.

#### Methods

##### temp_dir(self: Any)

Create temporary directory for testing.

##### encryption_config(self: Any)

Create encryption configuration for testing.

##### encryption_manager(self: Any, encryption_config: Any)

Create encryption manager for testing.

##### test_should_encrypt_file(self: Any, encryption_manager: Any)

Test file encryption pattern matching.

##### test_encrypt_decrypt_file(self: Any, encryption_manager: Any, temp_dir: Any)

Test file encryption and decryption.

##### test_encrypt_model_directory(self: Any, encryption_manager: Any, temp_dir: Any)

Test encrypting an entire model directory.

##### test_decrypt_model_directory(self: Any, encryption_manager: Any, temp_dir: Any)

Test decrypting an entire model directory.

##### test_temporary_decrypt_context(self: Any, encryption_manager: Any, temp_dir: Any)

Test temporary decryption context manager.

### TestIntegrationSecurity

Integration tests for security features.

#### Methods

##### test_end_to_end_credential_flow(self: Any)

Test complete credential management flow.

##### test_logging_credential_masking(self: Any)

Test that credentials are masked in log output.

