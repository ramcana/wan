"""
Comprehensive security tests for credential management in the Model Orchestrator.

These tests verify secure credential storage, retrieval, masking, and handling
of presigned URLs and temporary access tokens.
"""

import os
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from .credential_manager import (
    CredentialConfig,
    CredentialStore,
    CredentialMasker,
    PresignedURLManager,
    SecureCredentialManager,
    secure_credential_context
)
from .encryption_manager import (
    EncryptionConfig,
    ModelEncryptionManager,
    KeyManager,
    FileEncryptor
)


class TestCredentialMasker:
    """Test credential masking functionality."""
    
    def test_mask_api_keys(self):
        """Test masking of API keys in text."""
        text = "API_KEY=sk-1234567890abcdef"
        masked = CredentialMasker.mask_sensitive_data(text)
        assert "sk-1234567890abcdef" not in masked
        assert "API_KEY=sk******ef" in masked
    
    def test_mask_aws_credentials(self):
        """Test masking of AWS credentials."""
        text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        masked = CredentialMasker.mask_sensitive_data(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in masked
        assert "AWS_ACCESS_KEY_ID=AK******LE" in masked
    
    def test_mask_urls_with_credentials(self):
        """Test masking of URLs containing credentials."""
        text = "https://user:password123@example.com/path"
        masked = CredentialMasker.mask_sensitive_data(text)
        assert "password123" not in masked
        assert "https://user:pa******23@example.com/path" in masked
    
    def test_mask_presigned_urls(self):
        """Test masking of presigned URL signatures."""
        text = "https://s3.amazonaws.com/bucket/file?X-Amz-Signature=abcdef123456"
        masked = CredentialMasker.mask_sensitive_data(text)
        assert "abcdef123456" not in masked
        assert "X-Amz-Signature=ab******56" in masked
    
    def test_mask_hf_tokens(self):
        """Test masking of HuggingFace tokens."""
        text = "token=hf_abcdefghijklmnopqrstuvwxyz123456"
        masked = CredentialMasker.mask_sensitive_data(text)
        assert "hf_abcdefghijklmnopqrstuvwxyz123456" not in masked
        assert "hf******56" in masked
    
    def test_mask_dict_values(self):
        """Test masking of sensitive values in dictionaries."""
        data = {
            "api_key": "secret123",
            "username": "user",
            "password": "pass456",
            "config": {
                "token": "nested_secret"
            }
        }
        
        masked = CredentialMasker.mask_dict(data)
        
        assert masked["api_key"] == "se******23"
        assert masked["username"] == "user"  # Not sensitive
        assert masked["password"] == "pa******56"
        assert masked["config"]["token"] == "ne******et"
    
    def test_mask_short_values(self):
        """Test masking of short sensitive values."""
        text = "key=abc"
        masked = CredentialMasker.mask_sensitive_data(text)
        assert "abc" not in masked
        assert "key=******" in masked
    
    def test_preserve_non_sensitive_data(self):
        """Test that non-sensitive data is preserved."""
        text = "This is normal text with numbers 123 and symbols @#$"
        masked = CredentialMasker.mask_sensitive_data(text)
        assert masked == text


class TestCredentialStore:
    """Test secure credential storage."""
    
    @pytest.fixture
    def temp_credential_file(self):
        """Create temporary credential file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.enc') as f:
            yield f.name
        # Cleanup
        if os.path.exists(f.name):
            os.unlink(f.name)
    
    @pytest.fixture
    def credential_config(self, temp_credential_file):
        """Create credential configuration for testing."""
        return CredentialConfig(
            use_keyring=False,  # Disable keyring for testing
            credential_file=temp_credential_file,
            encryption_key_env="TEST_CREDENTIAL_KEY"
        )
    
    def test_store_and_retrieve_credential(self, credential_config):
        """Test storing and retrieving credentials."""
        with patch.dict(os.environ, {"TEST_CREDENTIAL_KEY": "a" * 64}):
            store = CredentialStore(credential_config)
            
            # Store credential
            success = store.store_credential("test-key", "test-value")
            assert success
            
            # Retrieve credential
            value = store.get_credential("test-key")
            assert value == "test-value"
    
    def test_credential_not_found(self, credential_config):
        """Test retrieving non-existent credential."""
        store = CredentialStore(credential_config)
        value = store.get_credential("non-existent")
        assert value is None
    
    def test_delete_credential(self, credential_config):
        """Test deleting credentials."""
        with patch.dict(os.environ, {"TEST_CREDENTIAL_KEY": "a" * 64}):
            store = CredentialStore(credential_config)
            
            # Store and verify
            store.store_credential("test-key", "test-value")
            assert store.get_credential("test-key") == "test-value"
            
            # Delete and verify
            success = store.delete_credential("test-key")
            assert success
            assert store.get_credential("test-key") is None
    
    def test_environment_fallback(self, credential_config):
        """Test fallback to environment variables."""
        credential_config.use_env_fallback = True
        store = CredentialStore(credential_config)
        
        with patch.dict(os.environ, {"TEST_KEY": "env-value"}):
            value = store.get_credential("test-key")
            assert value == "env-value"
    
    def test_list_credentials(self, credential_config):
        """Test listing stored credentials."""
        with patch.dict(os.environ, {"TEST_CREDENTIAL_KEY": "a" * 64}):
            store = CredentialStore(credential_config)
            
            # Store multiple credentials
            store.store_credential("key1", "value1")
            store.store_credential("key2", "value2")
            
            # List credentials
            keys = store.list_credentials()
            assert "key1" in keys
            assert "key2" in keys
    
    def test_keyring_storage(self, credential_config):
        """Test keyring-based credential storage."""
        credential_config.use_keyring = True
        
        # Mock keyring operations
        mock_keyring = Mock()
        mock_keyring.set_password = Mock()
        mock_keyring.get_password = Mock(return_value="keyring-value")
        mock_keyring.delete_password = Mock()
        
        store = CredentialStore(credential_config)
        store._keyring = mock_keyring
        
        # Test store
        success = store.store_credential("test-key", "test-value")
        assert success
        mock_keyring.set_password.assert_called_once()
        
        # Test retrieve
        value = store.get_credential("test-key")
        assert value == "keyring-value"
        mock_keyring.get_password.assert_called_once()
        
        # Test delete
        success = store.delete_credential("test-key")
        assert success
        mock_keyring.delete_password.assert_called_once()


class TestPresignedURLManager:
    """Test presigned URL management."""
    
    @pytest.fixture
    def url_manager(self):
        """Create presigned URL manager."""
        return PresignedURLManager()
    
    def test_generate_presigned_url(self, url_manager):
        """Test generating presigned URLs."""
        base_url = "https://example.com/file.txt"
        credentials = {
            "access_key": "test-access-key",
            "secret_key": "test-secret-key"
        }
        
        presigned_url = url_manager.generate_presigned_url(
            base_url, credentials, expiration_seconds=3600
        )
        
        assert "access_key=test-access-key" in presigned_url
        assert "signature=" in presigned_url
        assert "expires=" in presigned_url
        assert "timestamp=" in presigned_url
    
    def test_url_expiration_check(self, url_manager):
        """Test URL expiration checking."""
        import time
        
        # Create expired URL
        expired_url = "https://example.com/file.txt?expires=1000000000"
        assert url_manager.is_url_expired(expired_url)
        
        # Create future URL
        future_timestamp = int(time.time()) + 3600
        future_url = f"https://example.com/file.txt?expires={future_timestamp}"
        assert not url_manager.is_url_expired(future_url)
    
    def test_refresh_url_if_needed(self, url_manager):
        """Test URL refresh when expiring soon."""
        import time
        
        # Create URL that expires in 2 minutes (should be refreshed)
        expires_soon = int(time.time()) + 120
        url = f"https://example.com/file.txt?expires={expires_soon}"
        
        credentials = {
            "access_key": "test-access-key",
            "secret_key": "test-secret-key"
        }
        
        refreshed_url = url_manager.refresh_url_if_needed(url, credentials)
        
        # Should be different (refreshed)
        assert refreshed_url != url
        assert "signature=" in refreshed_url
    
    def test_clean_expired_cache(self, url_manager):
        """Test cleaning expired URLs from cache."""
        import time
        
        # Add expired entry to cache
        url_manager._url_cache["test"] = {
            "url": "https://example.com/test",
            "expires": int(time.time()) - 3600  # Expired 1 hour ago
        }
        
        # Add valid entry
        url_manager._url_cache["valid"] = {
            "url": "https://example.com/valid",
            "expires": int(time.time()) + 3600  # Expires in 1 hour
        }
        
        url_manager.clean_expired_cache()
        
        assert "test" not in url_manager._url_cache
        assert "valid" in url_manager._url_cache


class TestSecureCredentialManager:
    """Test the main credential manager."""
    
    @pytest.fixture
    def credential_manager(self):
        """Create secure credential manager."""
        config = CredentialConfig(use_keyring=False)
        return SecureCredentialManager(config)
    
    def test_get_s3_credentials(self, credential_manager):
        """Test getting S3 credentials."""
        # Mock credential store
        credential_manager.store.get_credential = Mock(side_effect=lambda key: {
            'aws-access-key-id': 'test-access-key',
            'aws-secret-access-key': 'test-secret-key'
        }.get(key))
        
        credentials = credential_manager.get_credentials_for_source("s3://bucket/path")
        
        assert credentials['access_key_id'] == 'test-access-key'
        assert credentials['secret_access_key'] == 'test-secret-key'
    
    def test_get_hf_credentials(self, credential_manager):
        """Test getting HuggingFace credentials."""
        # Mock credential store
        credential_manager.store.get_credential = Mock(return_value='hf-token-123')
        
        credentials = credential_manager.get_credentials_for_source("hf://model/repo")
        
        assert credentials['token'] == 'hf-token-123'
    
    def test_mask_sensitive_info_string(self, credential_manager):
        """Test masking sensitive information in strings."""
        sensitive_text = "API_KEY=secret123 and PASSWORD=pass456"
        masked = credential_manager.mask_sensitive_info(sensitive_text)
        
        assert "secret123" not in masked
        assert "pass456" not in masked
    
    def test_mask_sensitive_info_dict(self, credential_manager):
        """Test masking sensitive information in dictionaries."""
        sensitive_data = {
            "token": "secret123",
            "config": "normal_value"
        }
        
        masked = credential_manager.mask_sensitive_info(sensitive_data)
        
        assert masked["token"] == "se******23"
        assert masked["config"] == "normal_value"
    
    def test_setup_credentials(self, credential_manager):
        """Test setting up multiple credentials."""
        # Mock credential store
        credential_manager.store.store_credential = Mock(return_value=True)
        
        credentials = {
            "aws-access-key-id": "test-access",
            "aws-secret-access-key": "test-secret"
        }
        
        success = credential_manager.setup_credentials(credentials)
        assert success
        assert credential_manager.store.store_credential.call_count == 2
    
    def test_https_validation(self, credential_manager):
        """Test HTTPS requirement validation."""
        # HTTPS required by default
        assert credential_manager.validate_https_requirement("https://example.com")
        assert not credential_manager.validate_https_requirement("http://example.com")
        
        # Disable HTTPS requirement
        credential_manager.config.require_https = False
        assert credential_manager.validate_https_requirement("http://example.com")
    
    def test_secure_context(self, credential_manager):
        """Test secure credential context manager."""
        # Mock credential store
        credential_manager.store.get_credential = Mock(return_value='test-value')
        
        with credential_manager.get_secure_context("s3://bucket/path"):
            # Credentials should be temporarily available in environment
            pass
        
        # Credentials should be cleaned up after context


class TestSecureCredentialContext:
    """Test secure credential context manager."""
    
    def test_temporary_environment_variables(self):
        """Test temporary setting of environment variables."""
        original_value = os.environ.get("TEST_CREDENTIAL")
        
        # Mock credential store
        mock_store = Mock()
        credentials = {"test-credential": "temp-value"}
        
        with secure_credential_context(mock_store, credentials):
            # Environment variable should be set
            assert os.environ.get("TEST_CREDENTIAL") == "temp-value"
        
        # Environment variable should be restored
        assert os.environ.get("TEST_CREDENTIAL") == original_value
    
    def test_cleanup_on_exception(self):
        """Test that environment is cleaned up even on exceptions."""
        original_value = os.environ.get("TEST_CREDENTIAL")
        
        mock_store = Mock()
        credentials = {"test-credential": "temp-value"}
        
        try:
            with secure_credential_context(mock_store, credentials):
                assert os.environ.get("TEST_CREDENTIAL") == "temp-value"
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Environment should still be cleaned up
        assert os.environ.get("TEST_CREDENTIAL") == original_value


class TestModelEncryption:
    """Test model encryption functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def encryption_config(self):
        """Create encryption configuration for testing."""
        return EncryptionConfig(
            require_encryption_for_patterns=["*.safetensors", "*.bin"],
            master_key_env="TEST_MASTER_KEY"
        )
    
    @pytest.fixture
    def encryption_manager(self, encryption_config):
        """Create encryption manager for testing."""
        return ModelEncryptionManager(encryption_config)
    
    def test_should_encrypt_file(self, encryption_manager):
        """Test file encryption pattern matching."""
        assert encryption_manager.should_encrypt_file(Path("model.safetensors"))
        assert encryption_manager.should_encrypt_file(Path("weights.bin"))
        assert not encryption_manager.should_encrypt_file(Path("config.json"))
    
    def test_encrypt_decrypt_file(self, encryption_manager, temp_dir):
        """Test file encryption and decryption."""
        with patch.dict(os.environ, {"TEST_MASTER_KEY": "a" * 64}):
            # Create test file
            test_file = temp_dir / "test.safetensors"
            test_content = b"This is test model data"
            test_file.write_bytes(test_content)
            
            # Encrypt file
            encrypted_file = temp_dir / "test.safetensors.encrypted"
            metadata = encryption_manager.file_encryptor.encrypt_file(test_file, encrypted_file)
            
            assert encrypted_file.exists()
            assert metadata.original_size == len(test_content)
            
            # Decrypt file
            decrypted_file = temp_dir / "test_decrypted.safetensors"
            success = encryption_manager.file_encryptor.decrypt_file(
                encrypted_file, decrypted_file, metadata
            )
            
            assert success
            assert decrypted_file.read_bytes() == test_content
    
    def test_encrypt_model_directory(self, encryption_manager, temp_dir):
        """Test encrypting an entire model directory."""
        with patch.dict(os.environ, {"TEST_MASTER_KEY": "a" * 64}):
            # Create model files
            (temp_dir / "model.safetensors").write_bytes(b"model weights")
            (temp_dir / "config.json").write_bytes(b'{"model": "config"}')
            (temp_dir / "weights.bin").write_bytes(b"more weights")
            
            # Encrypt directory
            encrypted_files = encryption_manager.encrypt_model_directory(temp_dir)
            
            # Should encrypt .safetensors and .bin files, but not .json
            assert len(encrypted_files) == 2
            assert "model.safetensors" in encrypted_files
            assert "weights.bin" in encrypted_files
            
            # Original files should be removed, encrypted files should exist
            assert not (temp_dir / "model.safetensors").exists()
            assert (temp_dir / "model.safetensors.encrypted").exists()
            assert (temp_dir / "config.json").exists()  # Not encrypted
    
    def test_decrypt_model_directory(self, encryption_manager, temp_dir):
        """Test decrypting an entire model directory."""
        with patch.dict(os.environ, {"TEST_MASTER_KEY": "a" * 64}):
            # Create and encrypt model files
            (temp_dir / "model.safetensors").write_bytes(b"model weights")
            (temp_dir / "weights.bin").write_bytes(b"more weights")
            
            # Encrypt directory
            encryption_manager.encrypt_model_directory(temp_dir)
            
            # Decrypt directory
            success = encryption_manager.decrypt_model_directory(temp_dir)
            
            assert success
            assert (temp_dir / "model.safetensors").exists()
            assert (temp_dir / "weights.bin").exists()
            assert not (temp_dir / "model.safetensors.encrypted").exists()
    
    def test_temporary_decrypt_context(self, encryption_manager, temp_dir):
        """Test temporary decryption context manager."""
        with patch.dict(os.environ, {"TEST_MASTER_KEY": "a" * 64}):
            # Create and encrypt model
            (temp_dir / "model.safetensors").write_bytes(b"model weights")
            encryption_manager.encrypt_model_directory(temp_dir)
            
            # Use temporary decrypt context
            with encryption_manager.temporary_decrypt(temp_dir) as decrypted_dir:
                # Model should be decrypted
                assert (decrypted_dir / "model.safetensors").exists()
                assert not (decrypted_dir / "model.safetensors.encrypted").exists()
            
            # Model should be re-encrypted
            assert not (temp_dir / "model.safetensors").exists()
            assert (temp_dir / "model.safetensors.encrypted").exists()


class TestIntegrationSecurity:
    """Integration tests for security features."""
    
    def test_end_to_end_credential_flow(self):
        """Test complete credential management flow."""
        config = CredentialConfig(use_keyring=False)
        manager = SecureCredentialManager(config)
        
        # Setup credentials
        credentials = {
            "aws-access-key-id": "AKIATEST123",
            "aws-secret-access-key": "secret123",
            "hf-token": "hf_token123"
        }
        
        success = manager.setup_credentials(credentials)
        assert success
        
        # Get credentials for different sources
        s3_creds = manager.get_credentials_for_source("s3://bucket/path")
        hf_creds = manager.get_credentials_for_source("hf://model/repo")
        
        assert "access_key_id" in s3_creds
        assert "token" in hf_creds
        
        # Test masking
        sensitive_data = {"token": "hf_token123", "key": "secret123"}
        masked = manager.mask_sensitive_info(sensitive_data)
        
        assert "hf_token123" not in str(masked)
        assert "secret123" not in str(masked)
    
    def test_logging_credential_masking(self):
        """Test that credentials are masked in log output."""
        from .logging_config import StructuredFormatter
        import logging
        
        formatter = StructuredFormatter()
        
        # Create log record with sensitive data
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Processing with token=%s",
            args=("hf_secret123",),
            exc_info=None
        )
        
        # Add sensitive extra data
        record.api_key = "secret_key_123"
        record.normal_field = "normal_value"
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        # Sensitive data should be masked
        assert "hf_secret123" not in formatted
        assert "secret_key_123" not in formatted
        assert log_data["api_key"] == "se******23"
        assert log_data["normal_field"] == "normal_value"


if __name__ == "__main__":
    pytest.main([__file__])