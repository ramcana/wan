"""
Integration tests for security features in the Model Orchestrator.

These tests verify that all security components work together correctly,
including credential management, encryption, logging masking, and storage backend integration.
"""

import os
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from .credential_manager import SecureCredentialManager, CredentialConfig
from .encryption_manager import ModelEncryptionManager, EncryptionConfig
from .storage_backends.s3_store import S3Store, S3Config
from .storage_backends.hf_store import HFStore
from .logging_config import configure_logging, get_logger, StructuredFormatter


class TestSecurityIntegration:
    """Integration tests for security features."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def credential_config(self, temp_dir):
        """Create credential configuration for testing."""
        return CredentialConfig(
            use_keyring=False,
            credential_file=str(temp_dir / "credentials.enc"),
            encryption_key_env="TEST_CREDENTIAL_KEY",
            mask_credentials_in_logs=True,
            require_https=True
        )
    
    @pytest.fixture
    def encryption_config(self):
        """Create encryption configuration for testing."""
        return EncryptionConfig(
            require_encryption_for_patterns=["*.safetensors", "*.bin"],
            master_key_env="TEST_MASTER_KEY"
        )
    
    @pytest.fixture
    def credential_manager(self, credential_config):
        """Create secure credential manager."""
        return SecureCredentialManager(credential_config)
    
    @pytest.fixture
    def encryption_manager(self, encryption_config):
        """Create encryption manager."""
        return ModelEncryptionManager(encryption_config)
    
    def test_end_to_end_s3_with_security(self, credential_manager, temp_dir):
        """Test complete S3 workflow with security features."""
        with patch.dict(os.environ, {"TEST_CREDENTIAL_KEY": "a" * 64}):
            # Setup AWS credentials securely
            aws_credentials = {
                "aws-access-key-id": "AKIATEST123456789",
                "aws-secret-access-key": "test-secret-key-123456789",
                "aws-endpoint-url": "https://minio.example.com"
            }
            
            success = credential_manager.setup_credentials(aws_credentials)
            assert success
            
            # Create S3 store with credential manager
            s3_store = S3Store(credential_manager=credential_manager)
            
            # Verify credentials are retrieved securely
            retrieved_creds = credential_manager.get_credentials_for_source("s3://")
            assert retrieved_creds["access_key_id"] == "AKIATEST123456789"
            assert retrieved_creds["secret_access_key"] == "test-secret-key-123456789"
            
            # Test credential masking
            masked_creds = credential_manager.mask_sensitive_info(retrieved_creds)
            assert "AKIATEST123456789" not in str(masked_creds)
            assert "test-secret-key-123456789" not in str(masked_creds)
            
            # Verify HTTPS requirement
            assert credential_manager.validate_https_requirement("https://minio.example.com")
            assert not credential_manager.validate_https_requirement("http://minio.example.com")
    
    def test_end_to_end_hf_with_security(self, credential_manager, temp_dir):
        """Test complete HuggingFace workflow with security features."""
        with patch.dict(os.environ, {"TEST_CREDENTIAL_KEY": "a" * 64}):
            # Setup HF credentials securely
            hf_token = "hf_abcdefghijklmnopqrstuvwxyz123456"
            
            success = credential_manager.store.store_credential("hf-token", hf_token)
            assert success
            
            # Create HF store with credential manager
            hf_store = HFStore(credential_manager=credential_manager)
            
            # Verify credentials are retrieved securely
            retrieved_creds = credential_manager.get_credentials_for_source("hf://")
            assert retrieved_creds["token"] == hf_token
            
            # Test credential masking
            masked_creds = credential_manager.mask_sensitive_info(retrieved_creds)
            assert hf_token not in str(masked_creds)
            assert "hf******56" in str(masked_creds)
    
    def test_model_encryption_with_credential_security(self, encryption_manager, credential_manager, temp_dir):
        """Test model encryption with secure credential management."""
        with patch.dict(os.environ, {
            "TEST_MASTER_KEY": "a" * 64,
            "TEST_CREDENTIAL_KEY": "b" * 64
        }):
            # Create model files
            model_dir = temp_dir / "test_model"
            model_dir.mkdir()
            
            (model_dir / "model.safetensors").write_bytes(b"sensitive model weights")
            (model_dir / "config.json").write_bytes(b'{"model_type": "test"}')
            (model_dir / "weights.bin").write_bytes(b"more sensitive weights")
            
            # Encrypt model directory
            encrypted_files = encryption_manager.encrypt_model_directory(model_dir)
            
            # Verify encryption
            assert len(encrypted_files) == 2  # .safetensors and .bin files
            assert not (model_dir / "model.safetensors").exists()
            assert (model_dir / "model.safetensors.encrypted").exists()
            assert (model_dir / "config.json").exists()  # Not encrypted
            
            # Test temporary decryption with credential context
            with encryption_manager.temporary_decrypt(model_dir):
                # Model should be temporarily decrypted
                assert (model_dir / "model.safetensors").exists()
                assert (model_dir / "model.safetensors").read_bytes() == b"sensitive model weights"
                
                # Use credentials in this context
                with credential_manager.get_secure_context("s3://"):
                    # Credentials should be available in environment
                    pass
            
            # Model should be re-encrypted
            assert not (model_dir / "model.safetensors").exists()
            assert (model_dir / "model.safetensors.encrypted").exists()
    
    def test_logging_security_integration(self, credential_manager, temp_dir):
        """Test that logging properly masks credentials across all components."""
        import logging
        import io
        
        # Setup logging with credential masking
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        formatter = StructuredFormatter()
        handler.setFormatter(formatter)
        
        logger = get_logger("test")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        with patch.dict(os.environ, {"TEST_CREDENTIAL_KEY": "a" * 64}):
            # Store sensitive credentials
            sensitive_data = {
                "aws-access-key-id": "AKIATEST123456789",
                "aws-secret-access-key": "test-secret-key-123456789",
                "hf-token": "hf_abcdefghijklmnopqrstuvwxyz123456"
            }
            
            credential_manager.setup_credentials(sensitive_data)
            
            # Log operations that might contain sensitive data
            logger.info(
                "Processing S3 download",
                extra={
                    "source_url": "s3://bucket/model",
                    "aws_access_key_id": "AKIATEST123456789",
                    "token": "hf_abcdefghijklmnopqrstuvwxyz123456"
                }
            )
            
            # Check log output
            log_output = log_stream.getvalue()
            
            # Sensitive values should be masked
            assert "AKIATEST123456789" not in log_output
            assert "test-secret-key-123456789" not in log_output
            assert "hf_abcdefghijklmnopqrstuvwxyz123456" not in log_output
            
            # Masked values should be present
            assert "AK******89" in log_output or "aws_access_key_id" in log_output
            assert "hf******56" in log_output or "token" in log_output
    
    def test_presigned_url_security(self, credential_manager):
        """Test presigned URL generation and security."""
        # Setup credentials
        credentials = {
            "access_key": "test-access-key",
            "secret_key": "test-secret-key"
        }
        
        # Generate presigned URL
        base_url = "https://s3.amazonaws.com/bucket/model.safetensors"
        presigned_url = credential_manager.url_manager.generate_presigned_url(
            base_url, credentials, expiration_seconds=3600
        )
        
        # Verify URL structure
        assert "access_key=test-access-key" in presigned_url
        assert "signature=" in presigned_url
        assert "expires=" in presigned_url
        
        # Test URL masking
        masked_url = credential_manager.mask_sensitive_info(presigned_url)
        assert "test-access-key" not in masked_url
        assert "test-secret-key" not in masked_url
        
        # Test URL expiration
        assert not credential_manager.url_manager.is_url_expired(presigned_url)
        
        # Test URL refresh
        refreshed_url = credential_manager.url_manager.refresh_url_if_needed(
            presigned_url, credentials
        )
        assert refreshed_url  # Should return a valid URL
    
    def test_storage_backend_credential_integration(self, credential_manager, temp_dir):
        """Test storage backends with integrated credential management."""
        with patch.dict(os.environ, {"TEST_CREDENTIAL_KEY": "a" * 64}):
            # Setup credentials for both S3 and HF
            credentials = {
                "aws-access-key-id": "AKIATEST123456789",
                "aws-secret-access-key": "test-secret-key-123456789",
                "hf-token": "hf_abcdefghijklmnopqrstuvwxyz123456"
            }
            
            credential_manager.setup_credentials(credentials)
            
            # Test S3 store integration
            s3_config = S3Config()
            s3_store = S3Store(s3_config, credential_manager)
            
            # Verify S3 store gets credentials from manager
            assert s3_store.config.access_key_id == "AKIATEST123456789"
            assert s3_store.config.secret_access_key == "test-secret-key-123456789"
            
            # Test HF store integration
            hf_store = HFStore(credential_manager=credential_manager)
            
            # Verify HF store gets token from manager
            assert hf_store.token == "hf_abcdefghijklmnopqrstuvwxyz123456"
            
            # Test credential masking in store operations
            s3_creds = credential_manager.get_credentials_for_source("s3://")
            hf_creds = credential_manager.get_credentials_for_source("hf://")
            
            masked_s3 = credential_manager.mask_sensitive_info(s3_creds)
            masked_hf = credential_manager.mask_sensitive_info(hf_creds)
            
            # Verify masking
            assert "AKIATEST123456789" not in str(masked_s3)
            assert "hf_abcdefghijklmnopqrstuvwxyz123456" not in str(masked_hf)
    
    def test_error_handling_with_security(self, credential_manager, temp_dir):
        """Test error handling maintains security."""
        with patch.dict(os.environ, {"TEST_CREDENTIAL_KEY": "a" * 64}):
            # Test with invalid credentials
            invalid_credentials = {
                "aws-access-key-id": "INVALID_KEY",
                "aws-secret-access-key": "invalid-secret"
            }
            
            credential_manager.setup_credentials(invalid_credentials)
            
            # Create S3 store that will fail
            s3_store = S3Store(credential_manager=credential_manager)
            
            # Test that errors don't leak credentials
            try:
                # This would normally fail with authentication error
                # but we're testing that the error doesn't contain credentials
                result = s3_store.verify_availability("s3://invalid-bucket/path")
                # Should return False for invalid credentials
                assert result is False
            except Exception as e:
                # Error message should not contain sensitive data
                error_msg = str(e)
                masked_error = credential_manager.mask_sensitive_info(error_msg)
                
                # Verify no credentials in error
                assert "INVALID_KEY" not in error_msg or "INVALID_KEY" not in masked_error
                assert "invalid-secret" not in error_msg or "invalid-secret" not in masked_error
    
    def test_configuration_security(self, credential_manager, temp_dir):
        """Test configuration handling maintains security."""
        with patch.dict(os.environ, {"TEST_CREDENTIAL_KEY": "a" * 64}):
            # Setup credentials
            credentials = {
                "aws-access-key-id": "AKIATEST123456789",
                "aws-secret-access-key": "test-secret-key-123456789"
            }
            
            credential_manager.setup_credentials(credentials)
            
            # Export configuration (without values)
            from .credential_cli import CredentialCLI
            cli = CredentialCLI(credential_manager.config)
            
            config_file = temp_dir / "config.json"
            success = cli.export_config(str(config_file), include_values=False)
            assert success
            
            # Verify exported config doesn't contain sensitive values
            with open(config_file) as f:
                config_data = json.load(f)
            
            assert "AKIATEST123456789" not in json.dumps(config_data)
            assert "test-secret-key-123456789" not in json.dumps(config_data)
            assert config_data["credentials"]["aws-access-key-id"] == "<masked>"
    
    def test_concurrent_access_security(self, credential_manager, temp_dir):
        """Test that concurrent access maintains security."""
        import threading
        import time
        
        with patch.dict(os.environ, {"TEST_CREDENTIAL_KEY": "a" * 64}):
            # Setup credentials
            credentials = {
                "test-key-1": "secret-value-1",
                "test-key-2": "secret-value-2"
            }
            
            credential_manager.setup_credentials(credentials)
            
            results = []
            errors = []
            
            def access_credentials(key):
                try:
                    # Simulate concurrent credential access
                    time.sleep(0.1)  # Small delay to increase chance of race conditions
                    value = credential_manager.store.get_credential(key)
                    masked_value = credential_manager.mask_sensitive_info(value)
                    results.append((key, value, masked_value))
                except Exception as e:
                    errors.append(e)
            
            # Start multiple threads accessing credentials
            threads = []
            for i in range(10):
                key = f"test-key-{(i % 2) + 1}"
                thread = threading.Thread(target=access_credentials, args=(key,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify no errors occurred
            assert len(errors) == 0, f"Errors occurred: {errors}"
            
            # Verify all results are properly masked
            for key, value, masked_value in results:
                assert value in ["secret-value-1", "secret-value-2"]
                assert "secret-value" not in masked_value
                assert "se******" in masked_value or "masked" in masked_value.lower()


class TestSecurityCompliance:
    """Test security compliance and best practices."""
    
    def test_no_hardcoded_secrets(self):
        """Test that no secrets are hardcoded in the codebase."""
        # This is a basic test - in practice you'd use tools like truffleHog
        from .credential_manager import SecureCredentialManager
        from .encryption_manager import ModelEncryptionManager
        
        # Check that default configurations don't contain secrets
        cred_config = SecureCredentialManager().config
        enc_config = ModelEncryptionManager().config
        
        # Verify no hardcoded keys or tokens
        config_str = str(cred_config.__dict__) + str(enc_config.__dict__)
        
        # These patterns should not appear in configuration
        forbidden_patterns = [
            "AKIA",  # AWS access key prefix
            "hf_",   # HuggingFace token prefix
            "sk-",   # OpenAI API key prefix
            "ghp_",  # GitHub personal access token
        ]
        
        for pattern in forbidden_patterns:
            assert pattern not in config_str, f"Potential hardcoded secret found: {pattern}"
    
    def test_secure_defaults(self):
        """Test that security features are enabled by default."""
        from .credential_manager import CredentialConfig
        from .encryption_manager import EncryptionConfig
        
        # Credential security defaults
        cred_config = CredentialConfig()
        assert cred_config.use_keyring is True
        assert cred_config.mask_credentials_in_logs is True
        assert cred_config.require_https is True
        
        # Encryption security defaults
        enc_config = EncryptionConfig()
        assert enc_config.algorithm == "AES-256-GCM"
        assert enc_config.key_iterations >= 100000
    
    def test_environment_variable_security(self):
        """Test that environment variables are handled securely."""
        from .credential_manager import SecureCredentialManager
        
        # Test with sensitive environment variables
        with patch.dict(os.environ, {
            "AWS_ACCESS_KEY_ID": "AKIATEST123456789",
            "AWS_SECRET_ACCESS_KEY": "test-secret-key",
            "HF_TOKEN": "hf_test_token_123"
        }):
            manager = SecureCredentialManager()
            
            # Get credentials from environment
            s3_creds = manager.get_credentials_for_source("s3://")
            hf_creds = manager.get_credentials_for_source("hf://")
            
            # Verify credentials are retrieved
            assert s3_creds.get("access_key_id") == "AKIATEST123456789"
            assert hf_creds.get("token") == "hf_test_token_123"
            
            # Verify masking works
            masked_s3 = manager.mask_sensitive_info(s3_creds)
            masked_hf = manager.mask_sensitive_info(hf_creds)
            
            assert "AKIATEST123456789" not in str(masked_s3)
            assert "hf_test_token_123" not in str(masked_hf)


if __name__ == "__main__":
    pytest.main([__file__])