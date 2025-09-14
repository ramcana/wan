"""
Secure credential management system for the Model Orchestrator.

This module provides secure storage and retrieval of credentials using system keyring,
environment variables, and secure configuration files. It also handles credential masking
in logs and supports presigned URLs and temporary access tokens.
"""

import os
import re
import json
import logging
import hashlib
import secrets
from typing import Dict, Optional, Any, Union, List
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class CredentialConfig:
    """Configuration for credential storage and retrieval."""
    
    # Keyring settings
    use_keyring: bool = True
    keyring_service: str = "wan-model-orchestrator"
    
    # Environment variable fallback
    use_env_fallback: bool = True
    
    # Credential file settings (encrypted)
    credential_file: Optional[str] = None
    encryption_key_env: str = "WAN_CREDENTIAL_KEY"
    
    # Security settings
    mask_credentials_in_logs: bool = True
    credential_rotation_days: int = 90
    require_https: bool = True


class CredentialMasker:
    """Utility class for masking sensitive information in logs and outputs."""
    
    # Patterns for detecting sensitive information
    SENSITIVE_PATTERNS = [
        # API Keys and tokens (more specific patterns)
        (r'(?i)(api[_-]?key|token|secret|password|credential)["\s]*[:=]["\s]*([^\s"\'&]+)', 'API_KEY'),
        # AWS credentials
        (r'(?i)(aws[_-]?access[_-]?key[_-]?id)["\s]*[:=]["\s]*([A-Z0-9]{20})', 'AWS_ACCESS_KEY'),
        (r'(?i)(aws[_-]?secret[_-]?access[_-]?key)["\s]*[:=]["\s]*([A-Za-z0-9/+=]{40})', 'AWS_SECRET_KEY'),
        # URLs with credentials - more precise pattern
        (r'(https?://[^:]+):([^@]+)(@)', 'URL_CREDENTIAL'),
        # Presigned URLs (query parameters) - more precise
        (r'([?&])(X-Amz-Signature|Signature|signature)=([^&\s]+)', 'SIGNATURE'),
        (r'([?&])(X-Amz-Credential|AWSAccessKeyId|aws_access_key_id)=([^&\s]+)', 'AWS_CREDENTIAL'),
        # HuggingFace tokens
        (r'(hf_[a-zA-Z0-9]{34})', 'HF_TOKEN'),
    ]
    
    @classmethod
    def mask_sensitive_data(cls, text: str, mask_char: str = '*') -> str:
        """
        Mask sensitive information in text.
        
        Args:
            text: Text that may contain sensitive information
            mask_char: Character to use for masking
            
        Returns:
            Text with sensitive information masked
        """
        if not text:
            return text
            
        masked_text = text
        
        for pattern, label in cls.SENSITIVE_PATTERNS:
            def replace_match(match):
                if label == 'URL_CREDENTIAL':
                    # Special handling for URL credentials
                    prefix = match.group(1)  # https://user
                    sensitive = match.group(2)  # password
                    suffix = match.group(3)  # @
                    masked = cls._mask_value(sensitive, mask_char)
                    return f"{prefix}:{masked}{suffix}"
                elif label in ['SIGNATURE', 'AWS_CREDENTIAL']:
                    # Special handling for query parameters
                    prefix = match.group(1)  # ? or &
                    param_name = match.group(2)  # parameter name
                    sensitive = match.group(3)  # parameter value
                    masked = cls._mask_value(sensitive, mask_char)
                    return f"{prefix}{param_name}={masked}"
                elif len(match.groups()) >= 2:
                    # Keep prefix and mask the sensitive part
                    prefix = match.group(1)
                    sensitive = match.group(2)
                    masked = cls._mask_value(sensitive, mask_char)
                    return f"{prefix}={masked}"
                else:
                    # Mask the entire match
                    sensitive = match.group(1)
                    return cls._mask_value(sensitive, mask_char)
            
            masked_text = re.sub(pattern, replace_match, masked_text)
        
        return masked_text
    
    @classmethod
    def mask_dict(cls, data: Dict[str, Any], mask_char: str = '*') -> Dict[str, Any]:
        """
        Mask sensitive information in a dictionary.
        
        Args:
            data: Dictionary that may contain sensitive information
            mask_char: Character to use for masking
            
        Returns:
            Dictionary with sensitive values masked
        """
        if not isinstance(data, dict):
            return data
            
        masked_data = {}
        
        for key, value in data.items():
            if cls._is_sensitive_key(key):
                if isinstance(value, str):
                    masked_data[key] = cls._mask_value(value, mask_char)
                else:
                    masked_data[key] = f"<{type(value).__name__}:masked>"
            elif isinstance(value, dict):
                masked_data[key] = cls.mask_dict(value, mask_char)
            elif isinstance(value, str):
                masked_data[key] = cls.mask_sensitive_data(value, mask_char)
            else:
                masked_data[key] = value
                
        return masked_data
    
    @classmethod
    def _is_sensitive_key(cls, key: str) -> bool:
        """Check if a key name indicates sensitive information."""
        sensitive_keys = [
            'token', 'key', 'secret', 'password', 'credential', 'auth',
            'signature', 'access_key', 'secret_key', 'api_key'
        ]
        key_lower = key.lower()
        return any(sensitive in key_lower for sensitive in sensitive_keys)
    
    @classmethod
    def _mask_value(cls, value: str, mask_char: str = '*') -> str:
        """Mask a sensitive value."""
        if not value:
            return mask_char * 6
        if len(value) <= 4:
            return mask_char * 6
        elif len(value) <= 8:
            return value[:2] + mask_char * 4
        else:
            return value[:2] + mask_char * 6 + value[-2:]


class CredentialStore:
    """Secure credential storage using system keyring with fallbacks."""
    
    def __init__(self, config: Optional[CredentialConfig] = None):
        """
        Initialize credential store.
        
        Args:
            config: Credential configuration
        """
        self.config = config or CredentialConfig()
        self._keyring = None
        self._encryption_key = None
        
        # Try to import keyring
        if self.config.use_keyring:
            try:
                import keyring
                self._keyring = keyring
                logger.info("System keyring available for credential storage")
            except ImportError:
                logger.warning(
                    "keyring package not available, falling back to environment variables. "
                    "Install keyring for secure credential storage: pip install keyring"
                )
                self.config.use_keyring = False
    
    def store_credential(self, key: str, value: str, username: str = "default") -> bool:
        """
        Store a credential securely.
        
        Args:
            key: Credential identifier
            value: Credential value
            username: Username for keyring storage
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            if self.config.use_keyring and self._keyring:
                self._keyring.set_password(self.config.keyring_service, f"{username}:{key}", value)
                logger.info(f"Stored credential '{key}' in system keyring")
                return True
            elif self.config.credential_file:
                return self._store_in_file(key, value, username)
            else:
                # If no secure storage is available, we can still store in memory for testing
                # or use environment variables as a fallback
                logger.warning(f"No secure storage available for credential '{key}', using environment fallback")
                env_key = key.upper().replace('-', '_').replace(' ', '_')
                os.environ[env_key] = value
                return True
                
        except Exception as e:
            logger.error(f"Failed to store credential '{key}': {e}")
            return False
    
    def get_credential(self, key: str, username: str = "default") -> Optional[str]:
        """
        Retrieve a credential.
        
        Args:
            key: Credential identifier
            username: Username for keyring storage
            
        Returns:
            Credential value if found, None otherwise
        """
        try:
            # Try keyring first
            if self.config.use_keyring and self._keyring:
                value = self._keyring.get_password(self.config.keyring_service, f"{username}:{key}")
                if value:
                    logger.debug(f"Retrieved credential '{key}' from system keyring")
                    return value
            
            # Try credential file
            if self.config.credential_file:
                value = self._get_from_file(key, username)
                if value:
                    logger.debug(f"Retrieved credential '{key}' from credential file")
                    return value
            
            # Try environment variables as fallback
            if self.config.use_env_fallback:
                env_key = key.upper().replace('-', '_').replace(' ', '_')
                value = os.getenv(env_key)
                if value:
                    logger.debug(f"Retrieved credential '{key}' from environment variable")
                    return value
            
            logger.debug(f"Credential '{key}' not found in any storage")
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve credential '{key}': {e}")
            return None
    
    def delete_credential(self, key: str, username: str = "default") -> bool:
        """
        Delete a credential.
        
        Args:
            key: Credential identifier
            username: Username for keyring storage
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            success = False
            
            # Delete from keyring
            if self.config.use_keyring and self._keyring:
                try:
                    self._keyring.delete_password(self.config.keyring_service, f"{username}:{key}")
                    success = True
                    logger.info(f"Deleted credential '{key}' from system keyring")
                except Exception:
                    pass  # Credential might not exist in keyring
            
            # Delete from credential file
            if self.config.credential_file:
                if self._delete_from_file(key, username):
                    success = True
                    logger.info(f"Deleted credential '{key}' from credential file")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete credential '{key}': {e}")
            return False
    
    def list_credentials(self) -> List[str]:
        """
        List available credential keys.
        
        Returns:
            List of credential keys
        """
        credentials = set()
        
        try:
            # List from credential file
            if self.config.credential_file:
                file_creds = self._list_from_file()
                credentials.update(file_creds)
            
            # Note: keyring doesn't provide a way to list all stored credentials
            # so we can only list from file storage
            
        except Exception as e:
            logger.error(f"Failed to list credentials: {e}")
        
        return list(credentials)
    
    def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key for credential file."""
        if self._encryption_key:
            return self._encryption_key
        
        # Try to get key from environment
        key_str = os.getenv(self.config.encryption_key_env)
        if key_str:
            self._encryption_key = key_str.encode('utf-8')[:32].ljust(32, b'\0')
        else:
            # Generate a new key (this should be stored securely)
            self._encryption_key = secrets.token_bytes(32)
            logger.warning(
                f"Generated new encryption key. Set {self.config.encryption_key_env} "
                "environment variable to persist credentials across restarts."
            )
        
        return self._encryption_key
    
    def _store_in_file(self, key: str, value: str, username: str) -> bool:
        """Store credential in encrypted file."""
        try:
            from cryptography.fernet import Fernet
            import base64
            
            # Get or create encryption key
            encryption_key = self._get_encryption_key()
            fernet_key = base64.urlsafe_b64encode(encryption_key)
            fernet = Fernet(fernet_key)
            
            # Load existing credentials
            credentials = {}
            if Path(self.config.credential_file).exists():
                with open(self.config.credential_file, 'rb') as f:
                    encrypted_data = f.read()
                    if encrypted_data:
                        decrypted_data = fernet.decrypt(encrypted_data)
                        credentials = json.loads(decrypted_data.decode('utf-8'))
            
            # Add new credential
            credential_key = f"{username}:{key}"
            credentials[credential_key] = value
            
            # Encrypt and save
            data_str = json.dumps(credentials)
            encrypted_data = fernet.encrypt(data_str.encode('utf-8'))
            
            # Ensure parent directory exists
            Path(self.config.credential_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config.credential_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions
            os.chmod(self.config.credential_file, 0o600)
            
            return True
            
        except ImportError:
            logger.error(
                "cryptography package required for encrypted credential storage. "
                "Install with: pip install cryptography"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to store credential in file: {e}")
            return False
    
    def _get_from_file(self, key: str, username: str) -> Optional[str]:
        """Get credential from encrypted file."""
        try:
            if not Path(self.config.credential_file).exists():
                return None
            
            from cryptography.fernet import Fernet
            import base64
            
            # Get encryption key
            encryption_key = self._get_encryption_key()
            fernet_key = base64.urlsafe_b64encode(encryption_key)
            fernet = Fernet(fernet_key)
            
            # Load and decrypt credentials
            with open(self.config.credential_file, 'rb') as f:
                encrypted_data = f.read()
                if not encrypted_data:
                    return None
                
                decrypted_data = fernet.decrypt(encrypted_data)
                credentials = json.loads(decrypted_data.decode('utf-8'))
            
            credential_key = f"{username}:{key}"
            return credentials.get(credential_key)
            
        except ImportError:
            logger.error("cryptography package required for encrypted credential storage")
            return None
        except Exception as e:
            logger.debug(f"Failed to get credential from file: {e}")
            return None
    
    def _delete_from_file(self, key: str, username: str) -> bool:
        """Delete credential from encrypted file."""
        try:
            if not Path(self.config.credential_file).exists():
                return False
            
            from cryptography.fernet import Fernet
            import base64
            
            # Get encryption key
            encryption_key = self._get_encryption_key()
            fernet_key = base64.urlsafe_b64encode(encryption_key)
            fernet = Fernet(fernet_key)
            
            # Load and decrypt credentials
            with open(self.config.credential_file, 'rb') as f:
                encrypted_data = f.read()
                if not encrypted_data:
                    return False
                
                decrypted_data = fernet.decrypt(encrypted_data)
                credentials = json.loads(decrypted_data.decode('utf-8'))
            
            # Remove credential
            credential_key = f"{username}:{key}"
            if credential_key in credentials:
                del credentials[credential_key]
                
                # Encrypt and save
                data_str = json.dumps(credentials)
                encrypted_data = fernet.encrypt(data_str.encode('utf-8'))
                
                with open(self.config.credential_file, 'wb') as f:
                    f.write(encrypted_data)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete credential from file: {e}")
            return False
    
    def _list_from_file(self) -> List[str]:
        """List credentials from encrypted file."""
        try:
            if not Path(self.config.credential_file).exists():
                return []
            
            from cryptography.fernet import Fernet
            import base64
            
            # Get encryption key
            encryption_key = self._get_encryption_key()
            fernet_key = base64.urlsafe_b64encode(encryption_key)
            fernet = Fernet(fernet_key)
            
            # Load and decrypt credentials
            with open(self.config.credential_file, 'rb') as f:
                encrypted_data = f.read()
                if not encrypted_data:
                    return []
                
                decrypted_data = fernet.decrypt(encrypted_data)
                credentials = json.loads(decrypted_data.decode('utf-8'))
            
            # Extract keys (remove username prefix)
            keys = []
            for credential_key in credentials.keys():
                if ':' in credential_key:
                    _, key = credential_key.split(':', 1)
                    keys.append(key)
                else:
                    keys.append(credential_key)
            
            return keys
            
        except Exception as e:
            logger.debug(f"Failed to list credentials from file: {e}")
            return []


class PresignedURLManager:
    """Manager for handling presigned URLs and temporary access tokens."""
    
    def __init__(self, credential_store: Optional[CredentialStore] = None):
        """
        Initialize presigned URL manager.
        
        Args:
            credential_store: Credential store for caching tokens
        """
        self.credential_store = credential_store
        self._url_cache = {}  # In-memory cache for temporary URLs
    
    def generate_presigned_url(
        self,
        base_url: str,
        credentials: Dict[str, str],
        expiration_seconds: int = 3600,
        method: str = "GET"
    ) -> str:
        """
        Generate a presigned URL for temporary access.
        
        Args:
            base_url: Base URL to sign
            credentials: Credentials for signing
            expiration_seconds: URL expiration time in seconds
            method: HTTP method for the URL
            
        Returns:
            Presigned URL
        """
        try:
            import time
            import hmac
            from urllib.parse import quote
            
            # Parse URL components
            parsed = urlparse(base_url)
            
            # Generate timestamp and expiration
            timestamp = int(time.time())
            expires = timestamp + expiration_seconds
            
            # Create signature payload
            string_to_sign = f"{method}\n{parsed.path}\n{timestamp}\n{expires}"
            
            # Sign with secret key
            secret_key = credentials.get('secret_key', '')
            signature = hmac.new(
                secret_key.encode('utf-8'),
                string_to_sign.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Add query parameters
            query_params = parse_qs(parsed.query) if parsed.query else {}
            query_params.update({
                'access_key': [credentials.get('access_key', '')],
                'timestamp': [str(timestamp)],
                'expires': [str(expires)],
                'signature': [signature]
            })
            
            # Rebuild URL
            new_query = urlencode(query_params, doseq=True)
            presigned_url = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                new_query,
                parsed.fragment
            ))
            
            # Cache the URL
            cache_key = hashlib.md5(base_url.encode()).hexdigest()
            self._url_cache[cache_key] = {
                'url': presigned_url,
                'expires': expires
            }
            
            logger.info(f"Generated presigned URL (expires in {expiration_seconds}s)")
            return presigned_url
            
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise
    
    def is_url_expired(self, url: str) -> bool:
        """
        Check if a presigned URL has expired.
        
        Args:
            url: Presigned URL to check
            
        Returns:
            True if expired, False otherwise
        """
        try:
            import time
            
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            
            expires_list = query_params.get('expires', [])
            if not expires_list:
                return True  # No expiration info, assume expired
            
            expires = int(expires_list[0])
            current_time = int(time.time())
            
            return current_time >= expires
            
        except Exception as e:
            logger.warning(f"Failed to check URL expiration: {e}")
            return True  # Assume expired on error
    
    def refresh_url_if_needed(self, url: str, credentials: Dict[str, str]) -> str:
        """
        Refresh a presigned URL if it's expired or about to expire.
        
        Args:
            url: Current presigned URL
            credentials: Credentials for signing
            
        Returns:
            Valid presigned URL (refreshed if necessary)
        """
        try:
            import time
            
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            
            expires_list = query_params.get('expires', [])
            if expires_list:
                expires = int(expires_list[0])
                current_time = int(time.time())
                
                # Refresh if expires within 5 minutes
                if (expires - current_time) < 300:
                    logger.info("Refreshing presigned URL (expires soon)")
                    
                    # Remove signature parameters to get base URL
                    base_query_params = {
                        k: v for k, v in query_params.items()
                        if k not in ['access_key', 'timestamp', 'expires', 'signature']
                    }
                    base_query = urlencode(base_query_params, doseq=True)
                    base_url = urlunparse((
                        parsed.scheme,
                        parsed.netloc,
                        parsed.path,
                        parsed.params,
                        base_query,
                        parsed.fragment
                    ))
                    
                    return self.generate_presigned_url(base_url, credentials)
            
            return url  # URL is still valid
            
        except Exception as e:
            logger.warning(f"Failed to refresh URL: {e}")
            return url  # Return original URL on error
    
    def clean_expired_cache(self):
        """Clean expired URLs from cache."""
        try:
            import time
            current_time = int(time.time())
            
            expired_keys = [
                key for key, data in self._url_cache.items()
                if data['expires'] <= current_time
            ]
            
            for key in expired_keys:
                del self._url_cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned {len(expired_keys)} expired URLs from cache")
                
        except Exception as e:
            logger.warning(f"Failed to clean URL cache: {e}")


@contextmanager
def secure_credential_context(credential_store: CredentialStore, credentials: Dict[str, str]):
    """
    Context manager for temporarily setting credentials in environment.
    
    This is useful for libraries that only read from environment variables.
    The credentials are removed from the environment when the context exits.
    """
    original_env = {}
    
    try:
        # Set credentials in environment
        for key, value in credentials.items():
            env_key = key.upper().replace('-', '_').replace(' ', '_')
            original_env[env_key] = os.environ.get(env_key)
            os.environ[env_key] = value
        
        yield
        
    finally:
        # Restore original environment
        for key in credentials.keys():
            env_key = key.upper().replace('-', '_').replace(' ', '_')
            if original_env[env_key] is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = original_env[env_key]


class SecureCredentialManager:
    """
    Main credential manager that combines all security features.
    
    This class provides a high-level interface for secure credential management,
    including storage, retrieval, masking, and presigned URL handling.
    """
    
    def __init__(self, config: Optional[CredentialConfig] = None):
        """
        Initialize secure credential manager.
        
        Args:
            config: Credential configuration
        """
        self.config = config or CredentialConfig()
        self.store = CredentialStore(self.config)
        self.masker = CredentialMasker()
        self.url_manager = PresignedURLManager(self.store)
    
    def get_credentials_for_source(self, source_url: str) -> Dict[str, Optional[str]]:
        """
        Get credentials for a specific source URL.
        
        Args:
            source_url: Source URL to get credentials for
            
        Returns:
            Dictionary of credentials
        """
        credentials = {}
        
        if source_url.startswith('s3://'):
            credentials.update({
                'access_key_id': self.store.get_credential('aws-access-key-id'),
                'secret_access_key': self.store.get_credential('aws-secret-access-key'),
                'session_token': self.store.get_credential('aws-session-token'),
                'endpoint_url': self.store.get_credential('aws-endpoint-url')
            })
        elif source_url.startswith('hf://'):
            credentials.update({
                'token': self.store.get_credential('hf-token')
            })
        
        # Filter out None values
        return {k: v for k, v in credentials.items() if v is not None}
    
    def mask_sensitive_info(self, data: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """
        Mask sensitive information for logging.
        
        Args:
            data: Data to mask (string or dictionary)
            
        Returns:
            Data with sensitive information masked
        """
        if not self.config.mask_credentials_in_logs:
            return data
        
        if isinstance(data, str):
            return self.masker.mask_sensitive_data(data)
        elif isinstance(data, dict):
            return self.masker.mask_dict(data)
        else:
            return data
    
    def setup_credentials(self, credentials: Dict[str, str]) -> bool:
        """
        Set up multiple credentials at once.
        
        Args:
            credentials: Dictionary of credential key-value pairs
            
        Returns:
            True if all credentials were stored successfully
        """
        success = True
        
        for key, value in credentials.items():
            if not self.store.store_credential(key, value):
                success = False
                logger.error(f"Failed to store credential: {key}")
        
        return success
    
    def validate_https_requirement(self, url: str) -> bool:
        """
        Validate that URLs use HTTPS when required.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid, False otherwise
        """
        if not self.config.require_https:
            return True
        
        parsed = urlparse(url)
        if parsed.scheme == 'http':
            logger.error(f"HTTP URLs not allowed when HTTPS is required: {url}")
            return False
        
        return True
    
    def get_secure_context(self, source_url: str):
        """
        Get a secure context manager for credential handling.
        
        Args:
            source_url: Source URL to get credentials for
            
        Returns:
            Context manager for secure credential handling
        """
        credentials = self.get_credentials_for_source(source_url)
        return secure_credential_context(self.store, credentials)