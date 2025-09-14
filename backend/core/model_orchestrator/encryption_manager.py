"""
At-rest encryption support for sensitive models in the Model Orchestrator.

This module provides encryption and decryption capabilities for model files
that contain sensitive information or require additional security measures.
"""

import os
import json
import logging
import hashlib
import secrets
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class EncryptionConfig:
    """Configuration for model encryption."""
    
    # Encryption settings
    algorithm: str = "AES-256-GCM"
    key_derivation: str = "PBKDF2"
    key_iterations: int = 100000
    
    # Key management
    master_key_env: str = "WAN_MASTER_ENCRYPTION_KEY"
    key_rotation_days: int = 365
    
    # File settings
    encrypted_extension: str = ".encrypted"
    metadata_extension: str = ".enc_meta"
    chunk_size: int = 64 * 1024  # 64KB chunks
    
    # Security settings
    require_encryption_for_patterns: List[str] = None
    allow_unencrypted_fallback: bool = False


class EncryptionMetadata:
    """Metadata for encrypted files."""
    
    def __init__(
        self,
        algorithm: str,
        key_id: str,
        salt: bytes,
        nonce: bytes,
        tag: bytes,
        original_size: int,
        original_hash: str,
        encrypted_at: str
    ):
        self.algorithm = algorithm
        self.key_id = key_id
        self.salt = salt
        self.nonce = nonce
        self.tag = tag
        self.original_size = original_size
        self.original_hash = original_hash
        self.encrypted_at = encrypted_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        return {
            'algorithm': self.algorithm,
            'key_id': self.key_id,
            'salt': self.salt.hex(),
            'nonce': self.nonce.hex(),
            'tag': self.tag.hex(),
            'original_size': self.original_size,
            'original_hash': self.original_hash,
            'encrypted_at': self.encrypted_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EncryptionMetadata':
        """Create metadata from dictionary."""
        return cls(
            algorithm=data['algorithm'],
            key_id=data['key_id'],
            salt=bytes.fromhex(data['salt']),
            nonce=bytes.fromhex(data['nonce']),
            tag=bytes.fromhex(data['tag']),
            original_size=data['original_size'],
            original_hash=data['original_hash'],
            encrypted_at=data['encrypted_at']
        )


class KeyManager:
    """Manages encryption keys with rotation support."""
    
    def __init__(self, config: EncryptionConfig):
        """
        Initialize key manager.
        
        Args:
            config: Encryption configuration
        """
        self.config = config
        self._master_key = None
        self._derived_keys = {}  # Cache for derived keys
    
    def get_master_key(self) -> bytes:
        """Get or generate master encryption key."""
        if self._master_key:
            return self._master_key
        
        # Try to get key from environment
        key_str = os.getenv(self.config.master_key_env)
        if key_str:
            # Decode hex key
            try:
                self._master_key = bytes.fromhex(key_str)
                if len(self._master_key) != 32:  # 256 bits
                    raise ValueError("Master key must be 32 bytes (256 bits)")
                logger.info("Loaded master encryption key from environment")
                return self._master_key
            except ValueError as e:
                logger.error(f"Invalid master key format: {e}")
                raise
        
        # Generate new master key
        self._master_key = secrets.token_bytes(32)
        logger.warning(
            f"Generated new master encryption key. Set {self.config.master_key_env} "
            f"environment variable to: {self._master_key.hex()}"
        )
        
        return self._master_key
    
    def derive_key(self, key_id: str, salt: bytes) -> bytes:
        """
        Derive encryption key from master key.
        
        Args:
            key_id: Unique identifier for the key
            salt: Salt for key derivation
            
        Returns:
            Derived encryption key
        """
        cache_key = f"{key_id}:{salt.hex()}"
        if cache_key in self._derived_keys:
            return self._derived_keys[cache_key]
        
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            
            master_key = self.get_master_key()
            
            # Derive key using PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,  # 256 bits
                salt=salt,
                iterations=self.config.key_iterations,
            )
            
            # Use master key + key_id as password
            password = master_key + key_id.encode('utf-8')
            derived_key = kdf.derive(password)
            
            # Cache the derived key
            self._derived_keys[cache_key] = derived_key
            
            return derived_key
            
        except ImportError:
            raise ImportError(
                "cryptography package required for encryption. "
                "Install with: pip install cryptography"
            )
    
    def generate_key_id(self, file_path: str) -> str:
        """
        Generate a unique key ID for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Unique key identifier
        """
        # Use file path hash as key ID
        path_hash = hashlib.sha256(file_path.encode('utf-8')).hexdigest()
        return f"file:{path_hash[:16]}"
    
    def rotate_key(self, old_key_id: str) -> str:
        """
        Rotate encryption key for a file.
        
        Args:
            old_key_id: Current key ID
            
        Returns:
            New key ID
        """
        import time
        timestamp = int(time.time())
        return f"{old_key_id}:rotated:{timestamp}"


class FileEncryptor:
    """Handles encryption and decryption of individual files."""
    
    def __init__(self, key_manager: KeyManager, config: EncryptionConfig):
        """
        Initialize file encryptor.
        
        Args:
            key_manager: Key manager instance
            config: Encryption configuration
        """
        self.key_manager = key_manager
        self.config = config
    
    def encrypt_file(self, input_path: Path, output_path: Path) -> EncryptionMetadata:
        """
        Encrypt a file.
        
        Args:
            input_path: Path to input file
            output_path: Path to encrypted output file
            
        Returns:
            Encryption metadata
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            import time
            
            logger.info(f"Encrypting file: {input_path} -> {output_path}")
            
            # Generate encryption parameters
            key_id = self.key_manager.generate_key_id(str(input_path))
            salt = secrets.token_bytes(32)
            nonce = secrets.token_bytes(12)  # 96 bits for GCM
            
            # Derive encryption key
            encryption_key = self.key_manager.derive_key(key_id, salt)
            
            # Calculate original file hash and size
            original_hash = self._calculate_file_hash(input_path)
            original_size = input_path.stat().st_size
            
            # Initialize AES-GCM cipher
            aesgcm = AESGCM(encryption_key)
            
            # Encrypt file in chunks
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(input_path, 'rb') as infile, open(output_path, 'wb') as outfile:
                # Write header with nonce
                outfile.write(nonce)
                
                # Encrypt file content
                chunk_num = 0
                tag = None
                
                while True:
                    chunk = infile.read(self.config.chunk_size)
                    if not chunk:
                        break
                    
                    # For GCM, we need to encrypt all data at once to get the tag
                    # So we'll read the entire file for now (can be optimized for large files)
                    if chunk_num == 0:
                        # Read entire file
                        infile.seek(0)
                        entire_content = infile.read()
                        
                        # Encrypt entire content
                        encrypted_content = aesgcm.encrypt(nonce, entire_content, None)
                        
                        # Extract ciphertext and tag
                        ciphertext = encrypted_content[:-16]  # All but last 16 bytes
                        tag = encrypted_content[-16:]  # Last 16 bytes
                        
                        # Write encrypted content
                        outfile.write(ciphertext)
                        break
                    
                    chunk_num += 1
            
            # Create metadata
            metadata = EncryptionMetadata(
                algorithm=self.config.algorithm,
                key_id=key_id,
                salt=salt,
                nonce=nonce,
                tag=tag,
                original_size=original_size,
                original_hash=original_hash,
                encrypted_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            )
            
            # Save metadata
            metadata_path = output_path.with_suffix(output_path.suffix + self.config.metadata_extension)
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            logger.info(f"File encrypted successfully: {output_path}")
            return metadata
            
        except ImportError:
            raise ImportError(
                "cryptography package required for encryption. "
                "Install with: pip install cryptography"
            )
        except Exception as e:
            logger.error(f"Failed to encrypt file {input_path}: {e}")
            raise
    
    def decrypt_file(self, input_path: Path, output_path: Path, metadata: EncryptionMetadata) -> bool:
        """
        Decrypt a file.
        
        Args:
            input_path: Path to encrypted input file
            output_path: Path to decrypted output file
            metadata: Encryption metadata
            
        Returns:
            True if decryption successful, False otherwise
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            logger.info(f"Decrypting file: {input_path} -> {output_path}")
            
            # Derive decryption key
            decryption_key = self.key_manager.derive_key(metadata.key_id, metadata.salt)
            
            # Initialize AES-GCM cipher
            aesgcm = AESGCM(decryption_key)
            
            # Read encrypted file
            with open(input_path, 'rb') as infile:
                # Read nonce (first 12 bytes)
                nonce = infile.read(12)
                if nonce != metadata.nonce:
                    raise ValueError("Nonce mismatch in encrypted file")
                
                # Read encrypted content
                ciphertext = infile.read()
            
            # Reconstruct encrypted data with tag
            encrypted_data = ciphertext + metadata.tag
            
            # Decrypt content
            decrypted_content = aesgcm.decrypt(nonce, encrypted_data, None)
            
            # Write decrypted file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as outfile:
                outfile.write(decrypted_content)
            
            # Verify integrity
            if not self._verify_file_integrity(output_path, metadata):
                logger.error("File integrity verification failed after decryption")
                return False
            
            logger.info(f"File decrypted successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to decrypt file {input_path}: {e}")
            return False
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _verify_file_integrity(self, file_path: Path, metadata: EncryptionMetadata) -> bool:
        """Verify file integrity after decryption."""
        try:
            # Check file size
            actual_size = file_path.stat().st_size
            if actual_size != metadata.original_size:
                logger.error(f"Size mismatch: expected {metadata.original_size}, got {actual_size}")
                return False
            
            # Check file hash
            actual_hash = self._calculate_file_hash(file_path)
            if actual_hash != metadata.original_hash:
                logger.error(f"Hash mismatch: expected {metadata.original_hash}, got {actual_hash}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify file integrity: {e}")
            return False


class ModelEncryptionManager:
    """
    Main encryption manager for model files.
    
    This class provides high-level encryption and decryption operations
    for model files, with support for selective encryption based on patterns.
    """
    
    def __init__(self, config: Optional[EncryptionConfig] = None):
        """
        Initialize model encryption manager.
        
        Args:
            config: Encryption configuration
        """
        self.config = config or EncryptionConfig()
        self.key_manager = KeyManager(self.config)
        self.file_encryptor = FileEncryptor(self.key_manager, self.config)
    
    def should_encrypt_file(self, file_path: Path) -> bool:
        """
        Determine if a file should be encrypted based on patterns.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file should be encrypted, False otherwise
        """
        if not self.config.require_encryption_for_patterns:
            return False
        
        import fnmatch
        
        file_str = str(file_path)
        for pattern in self.config.require_encryption_for_patterns:
            if fnmatch.fnmatch(file_str, pattern):
                logger.debug(f"File matches encryption pattern '{pattern}': {file_path}")
                return True
        
        return False
    
    def encrypt_model_directory(self, model_dir: Path) -> Dict[str, EncryptionMetadata]:
        """
        Encrypt all files in a model directory that match encryption patterns.
        
        Args:
            model_dir: Path to model directory
            
        Returns:
            Dictionary mapping file paths to encryption metadata
        """
        encrypted_files = {}
        
        try:
            logger.info(f"Encrypting model directory: {model_dir}")
            
            # Find files to encrypt
            files_to_encrypt = []
            for file_path in model_dir.rglob("*"):
                if file_path.is_file() and self.should_encrypt_file(file_path):
                    files_to_encrypt.append(file_path)
            
            if not files_to_encrypt:
                logger.info("No files require encryption in model directory")
                return encrypted_files
            
            logger.info(f"Encrypting {len(files_to_encrypt)} files")
            
            # Encrypt each file
            for file_path in files_to_encrypt:
                try:
                    # Create encrypted file path
                    encrypted_path = file_path.with_suffix(
                        file_path.suffix + self.config.encrypted_extension
                    )
                    
                    # Encrypt file
                    metadata = self.file_encryptor.encrypt_file(file_path, encrypted_path)
                    
                    # Store metadata
                    relative_path = file_path.relative_to(model_dir)
                    encrypted_files[str(relative_path)] = metadata
                    
                    # Remove original file
                    file_path.unlink()
                    
                    logger.debug(f"Encrypted file: {relative_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to encrypt file {file_path}: {e}")
                    # Continue with other files
            
            logger.info(f"Successfully encrypted {len(encrypted_files)} files")
            return encrypted_files
            
        except Exception as e:
            logger.error(f"Failed to encrypt model directory {model_dir}: {e}")
            raise
    
    def decrypt_model_directory(self, model_dir: Path) -> bool:
        """
        Decrypt all encrypted files in a model directory.
        
        Args:
            model_dir: Path to model directory
            
        Returns:
            True if all files decrypted successfully, False otherwise
        """
        try:
            logger.info(f"Decrypting model directory: {model_dir}")
            
            # Find encrypted files
            encrypted_files = []
            for file_path in model_dir.rglob(f"*{self.config.encrypted_extension}"):
                if file_path.is_file():
                    encrypted_files.append(file_path)
            
            if not encrypted_files:
                logger.info("No encrypted files found in model directory")
                return True
            
            logger.info(f"Decrypting {len(encrypted_files)} files")
            
            success_count = 0
            
            # Decrypt each file
            for encrypted_path in encrypted_files:
                try:
                    # Load metadata
                    metadata_path = encrypted_path.with_suffix(
                        encrypted_path.suffix + self.config.metadata_extension
                    )
                    
                    if not metadata_path.exists():
                        logger.error(f"Metadata file not found: {metadata_path}")
                        continue
                    
                    with open(metadata_path, 'r') as f:
                        metadata_dict = json.load(f)
                    
                    metadata = EncryptionMetadata.from_dict(metadata_dict)
                    
                    # Create decrypted file path (remove encrypted extension)
                    decrypted_path = encrypted_path.with_suffix(
                        encrypted_path.suffix.replace(self.config.encrypted_extension, '')
                    )
                    
                    # Decrypt file
                    if self.file_encryptor.decrypt_file(encrypted_path, decrypted_path, metadata):
                        # Remove encrypted file and metadata
                        encrypted_path.unlink()
                        metadata_path.unlink()
                        success_count += 1
                        
                        logger.debug(f"Decrypted file: {decrypted_path.relative_to(model_dir)}")
                    else:
                        logger.error(f"Failed to decrypt file: {encrypted_path}")
                
                except Exception as e:
                    logger.error(f"Failed to decrypt file {encrypted_path}: {e}")
                    # Continue with other files
            
            logger.info(f"Successfully decrypted {success_count}/{len(encrypted_files)} files")
            return success_count == len(encrypted_files)
            
        except Exception as e:
            logger.error(f"Failed to decrypt model directory {model_dir}: {e}")
            return False
    
    def is_model_encrypted(self, model_dir: Path) -> bool:
        """
        Check if a model directory contains encrypted files.
        
        Args:
            model_dir: Path to model directory
            
        Returns:
            True if model contains encrypted files, False otherwise
        """
        try:
            for file_path in model_dir.rglob(f"*{self.config.encrypted_extension}"):
                if file_path.is_file():
                    return True
            return False
        except Exception:
            return False
    
    @contextmanager
    def temporary_decrypt(self, model_dir: Path):
        """
        Context manager for temporarily decrypting a model directory.
        
        The model is decrypted on entry and re-encrypted on exit.
        
        Args:
            model_dir: Path to model directory
        """
        was_encrypted = self.is_model_encrypted(model_dir)
        
        try:
            if was_encrypted:
                logger.info(f"Temporarily decrypting model: {model_dir}")
                if not self.decrypt_model_directory(model_dir):
                    raise RuntimeError("Failed to decrypt model directory")
            
            yield model_dir
            
        finally:
            if was_encrypted:
                logger.info(f"Re-encrypting model: {model_dir}")
                try:
                    self.encrypt_model_directory(model_dir)
                except Exception as e:
                    logger.error(f"Failed to re-encrypt model directory: {e}")
                    # Don't raise here to avoid masking original exceptions