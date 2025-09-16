---
title: core.model_orchestrator.encryption_manager
category: api
tags: [api, core]
---

# core.model_orchestrator.encryption_manager

At-rest encryption support for sensitive models in the Model Orchestrator.

This module provides encryption and decryption capabilities for model files
that contain sensitive information or require additional security measures.

## Classes

### EncryptionConfig

Configuration for model encryption.

### EncryptionMetadata

Metadata for encrypted files.

#### Methods

##### __init__(self: Any, algorithm: str, key_id: str, salt: bytes, nonce: bytes, tag: bytes, original_size: int, original_hash: str, encrypted_at: str)



##### to_dict(self: Any) -> <ast.Subscript object at 0x00000194344472B0>

Convert metadata to dictionary for JSON serialization.

##### from_dict(cls: Any, data: <ast.Subscript object at 0x0000019434447E20>) -> EncryptionMetadata

Create metadata from dictionary.

### KeyManager

Manages encryption keys with rotation support.

#### Methods

##### __init__(self: Any, config: EncryptionConfig)

Initialize key manager.

Args:
    config: Encryption configuration

##### get_master_key(self: Any) -> bytes

Get or generate master encryption key.

##### derive_key(self: Any, key_id: str, salt: bytes) -> bytes

Derive encryption key from master key.

Args:
    key_id: Unique identifier for the key
    salt: Salt for key derivation
    
Returns:
    Derived encryption key

##### generate_key_id(self: Any, file_path: str) -> str

Generate a unique key ID for a file.

Args:
    file_path: Path to the file
    
Returns:
    Unique key identifier

##### rotate_key(self: Any, old_key_id: str) -> str

Rotate encryption key for a file.

Args:
    old_key_id: Current key ID
    
Returns:
    New key ID

### FileEncryptor

Handles encryption and decryption of individual files.

#### Methods

##### __init__(self: Any, key_manager: KeyManager, config: EncryptionConfig)

Initialize file encryptor.

Args:
    key_manager: Key manager instance
    config: Encryption configuration

##### encrypt_file(self: Any, input_path: Path, output_path: Path) -> EncryptionMetadata

Encrypt a file.

Args:
    input_path: Path to input file
    output_path: Path to encrypted output file
    
Returns:
    Encryption metadata

##### decrypt_file(self: Any, input_path: Path, output_path: Path, metadata: EncryptionMetadata) -> bool

Decrypt a file.

Args:
    input_path: Path to encrypted input file
    output_path: Path to decrypted output file
    metadata: Encryption metadata
    
Returns:
    True if decryption successful, False otherwise

##### _calculate_file_hash(self: Any, file_path: Path) -> str

Calculate SHA256 hash of a file.

##### _verify_file_integrity(self: Any, file_path: Path, metadata: EncryptionMetadata) -> bool

Verify file integrity after decryption.

### ModelEncryptionManager

Main encryption manager for model files.

This class provides high-level encryption and decryption operations
for model files, with support for selective encryption based on patterns.

#### Methods

##### __init__(self: Any, config: <ast.Subscript object at 0x0000019432DD2920>)

Initialize model encryption manager.

Args:
    config: Encryption configuration

##### should_encrypt_file(self: Any, file_path: Path) -> bool

Determine if a file should be encrypted based on patterns.

Args:
    file_path: Path to the file
    
Returns:
    True if file should be encrypted, False otherwise

##### encrypt_model_directory(self: Any, model_dir: Path) -> <ast.Subscript object at 0x0000019431AD1930>

Encrypt all files in a model directory that match encryption patterns.

Args:
    model_dir: Path to model directory
    
Returns:
    Dictionary mapping file paths to encryption metadata

##### decrypt_model_directory(self: Any, model_dir: Path) -> bool

Decrypt all encrypted files in a model directory.

Args:
    model_dir: Path to model directory
    
Returns:
    True if all files decrypted successfully, False otherwise

##### is_model_encrypted(self: Any, model_dir: Path) -> bool

Check if a model directory contains encrypted files.

Args:
    model_dir: Path to model directory
    
Returns:
    True if model contains encrypted files, False otherwise

##### temporary_decrypt(self: Any, model_dir: Path)

Context manager for temporarily decrypting a model directory.

The model is decrypted on entry and re-encrypted on exit.

Args:
    model_dir: Path to model directory

