---
title: core.model_orchestrator.credential_manager
category: api
tags: [api, core]
---

# core.model_orchestrator.credential_manager

Secure credential management system for the Model Orchestrator.

This module provides secure storage and retrieval of credentials using system keyring,
environment variables, and secure configuration files. It also handles credential masking
in logs and supports presigned URLs and temporary access tokens.

## Classes

### CredentialConfig

Configuration for credential storage and retrieval.

### CredentialMasker

Utility class for masking sensitive information in logs and outputs.

#### Methods

##### mask_sensitive_data(cls: Any, text: str, mask_char: str) -> str

Mask sensitive information in text.

Args:
    text: Text that may contain sensitive information
    mask_char: Character to use for masking
    
Returns:
    Text with sensitive information masked

##### mask_dict(cls: Any, data: <ast.Subscript object at 0x000001942F7A2A10>, mask_char: str) -> <ast.Subscript object at 0x000001942F800850>

Mask sensitive information in a dictionary.

Args:
    data: Dictionary that may contain sensitive information
    mask_char: Character to use for masking
    
Returns:
    Dictionary with sensitive values masked

##### _is_sensitive_key(cls: Any, key: str) -> bool

Check if a key name indicates sensitive information.

##### _mask_value(cls: Any, value: str, mask_char: str) -> str

Mask a sensitive value.

### CredentialStore

Secure credential storage using system keyring with fallbacks.

#### Methods

##### __init__(self: Any, config: <ast.Subscript object at 0x000001942F7FF3D0>)

Initialize credential store.

Args:
    config: Credential configuration

##### store_credential(self: Any, key: str, value: str, username: str) -> bool

Store a credential securely.

Args:
    key: Credential identifier
    value: Credential value
    username: Username for keyring storage
    
Returns:
    True if stored successfully, False otherwise

##### get_credential(self: Any, key: str, username: str) -> <ast.Subscript object at 0x000001942F825780>

Retrieve a credential.

Args:
    key: Credential identifier
    username: Username for keyring storage
    
Returns:
    Credential value if found, None otherwise

##### delete_credential(self: Any, key: str, username: str) -> bool

Delete a credential.

Args:
    key: Credential identifier
    username: Username for keyring storage
    
Returns:
    True if deleted successfully, False otherwise

##### list_credentials(self: Any) -> <ast.Subscript object at 0x0000019431970910>

List available credential keys.

Returns:
    List of credential keys

##### _get_encryption_key(self: Any) -> bytes

Get or generate encryption key for credential file.

##### _store_in_file(self: Any, key: str, value: str, username: str) -> bool

Store credential in encrypted file.

##### _get_from_file(self: Any, key: str, username: str) -> <ast.Subscript object at 0x000001942F768EE0>

Get credential from encrypted file.

##### _delete_from_file(self: Any, key: str, username: str) -> bool

Delete credential from encrypted file.

##### _list_from_file(self: Any) -> <ast.Subscript object at 0x0000019434095090>

List credentials from encrypted file.

### PresignedURLManager

Manager for handling presigned URLs and temporary access tokens.

#### Methods

##### __init__(self: Any, credential_store: <ast.Subscript object at 0x0000019434094E80>)

Initialize presigned URL manager.

Args:
    credential_store: Credential store for caching tokens

##### generate_presigned_url(self: Any, base_url: str, credentials: <ast.Subscript object at 0x0000019434094A90>, expiration_seconds: int, method: str) -> str

Generate a presigned URL for temporary access.

Args:
    base_url: Base URL to sign
    credentials: Credentials for signing
    expiration_seconds: URL expiration time in seconds
    method: HTTP method for the URL
    
Returns:
    Presigned URL

##### is_url_expired(self: Any, url: str) -> bool

Check if a presigned URL has expired.

Args:
    url: Presigned URL to check
    
Returns:
    True if expired, False otherwise

##### refresh_url_if_needed(self: Any, url: str, credentials: <ast.Subscript object at 0x000001942F816CE0>) -> str

Refresh a presigned URL if it's expired or about to expire.

Args:
    url: Current presigned URL
    credentials: Credentials for signing
    
Returns:
    Valid presigned URL (refreshed if necessary)

##### clean_expired_cache(self: Any)

Clean expired URLs from cache.

### SecureCredentialManager

Main credential manager that combines all security features.

This class provides a high-level interface for secure credential management,
including storage, retrieval, masking, and presigned URL handling.

#### Methods

##### __init__(self: Any, config: <ast.Subscript object at 0x000001942FBED1E0>)

Initialize secure credential manager.

Args:
    config: Credential configuration

##### get_credentials_for_source(self: Any, source_url: str) -> <ast.Subscript object at 0x000001942FBEC310>

Get credentials for a specific source URL.

Args:
    source_url: Source URL to get credentials for
    
Returns:
    Dictionary of credentials

##### mask_sensitive_info(self: Any, data: <ast.Subscript object at 0x000001942FBEDD80>) -> <ast.Subscript object at 0x000001942FC0C0D0>

Mask sensitive information for logging.

Args:
    data: Data to mask (string or dictionary)
    
Returns:
    Data with sensitive information masked

##### setup_credentials(self: Any, credentials: <ast.Subscript object at 0x000001942FC0C340>) -> bool

Set up multiple credentials at once.

Args:
    credentials: Dictionary of credential key-value pairs
    
Returns:
    True if all credentials were stored successfully

##### validate_https_requirement(self: Any, url: str) -> bool

Validate that URLs use HTTPS when required.

Args:
    url: URL to validate
    
Returns:
    True if URL is valid, False otherwise

##### get_secure_context(self: Any, source_url: str)

Get a secure context manager for credential handling.

Args:
    source_url: Source URL to get credentials for
    
Returns:
    Context manager for secure credential handling

## Constants

### SENSITIVE_PATTERNS

Type: `unknown`

