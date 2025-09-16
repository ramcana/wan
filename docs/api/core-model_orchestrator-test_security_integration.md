---
title: core.model_orchestrator.test_security_integration
category: api
tags: [api, core]
---

# core.model_orchestrator.test_security_integration

Integration tests for security features in the Model Orchestrator.

These tests verify that all security components work together correctly,
including credential management, encryption, logging masking, and storage backend integration.

## Classes

### TestSecurityIntegration

Integration tests for security features.

#### Methods

##### temp_dir(self: Any)

Create temporary directory for testing.

##### credential_config(self: Any, temp_dir: Any)

Create credential configuration for testing.

##### encryption_config(self: Any)

Create encryption configuration for testing.

##### credential_manager(self: Any, credential_config: Any)

Create secure credential manager.

##### encryption_manager(self: Any, encryption_config: Any)

Create encryption manager.

##### test_end_to_end_s3_with_security(self: Any, credential_manager: Any, temp_dir: Any)

Test complete S3 workflow with security features.

##### test_end_to_end_hf_with_security(self: Any, credential_manager: Any, temp_dir: Any)

Test complete HuggingFace workflow with security features.

##### test_model_encryption_with_credential_security(self: Any, encryption_manager: Any, credential_manager: Any, temp_dir: Any)

Test model encryption with secure credential management.

##### test_logging_security_integration(self: Any, credential_manager: Any, temp_dir: Any)

Test that logging properly masks credentials across all components.

##### test_presigned_url_security(self: Any, credential_manager: Any)

Test presigned URL generation and security.

##### test_storage_backend_credential_integration(self: Any, credential_manager: Any, temp_dir: Any)

Test storage backends with integrated credential management.

##### test_error_handling_with_security(self: Any, credential_manager: Any, temp_dir: Any)

Test error handling maintains security.

##### test_configuration_security(self: Any, credential_manager: Any, temp_dir: Any)

Test configuration handling maintains security.

##### test_concurrent_access_security(self: Any, credential_manager: Any, temp_dir: Any)

Test that concurrent access maintains security.

### TestSecurityCompliance

Test security compliance and best practices.

#### Methods

##### test_no_hardcoded_secrets(self: Any)

Test that no secrets are hardcoded in the codebase.

##### test_secure_defaults(self: Any)

Test that security features are enabled by default.

##### test_environment_variable_security(self: Any)

Test that environment variables are handled securely.

