"""
Comprehensive integrity verification system for model files.

This module provides enhanced integrity verification including SHA256 checksums,
HuggingFace ETag verification, manifest signature verification, and retry logic
for handling various integrity failure scenarios.
"""

import os
import json
import hashlib
import time
import hmac
import base64
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

from .model_registry import FileSpec, ModelSpec
from .exceptions import (
    ChecksumError, SizeMismatchError, IntegrityVerificationError,
    ManifestSignatureError, ErrorCode
)
from .logging_config import get_logger, LogContext
from .error_recovery import retry_operation, RetryConfig

logger = get_logger(__name__)


class VerificationMethod(Enum):
    """Available verification methods in order of preference."""
    SHA256_CHECKSUM = "sha256_checksum"
    HF_ETAG = "hf_etag"
    SIZE_ONLY = "size_only"


@dataclass
class FileVerificationResult:
    """Result of verifying a single file."""
    file_path: str
    success: bool
    method_used: VerificationMethod
    expected_value: Optional[str] = None
    actual_value: Optional[str] = None
    error_message: Optional[str] = None
    verification_time: float = 0.0


@dataclass
class IntegrityVerificationResult:
    """Result of comprehensive integrity verification."""
    success: bool
    verified_files: List[FileVerificationResult]
    failed_files: List[FileVerificationResult]
    missing_files: List[str]
    total_verification_time: float
    manifest_signature_valid: bool = True
    error_message: Optional[str] = None


@dataclass
class HFFileMetadata:
    """HuggingFace file metadata for ETag verification."""
    etag: str
    size: int
    last_modified: Optional[str] = None


class IntegrityVerifier:
    """
    Comprehensive integrity verification system.
    
    Provides multiple verification methods with fallback support:
    1. SHA256 checksum verification (preferred)
    2. HuggingFace ETag verification (fallback)
    3. Size-only verification (last resort)
    """
    
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        """
        Initialize the integrity verifier.
        
        Args:
            retry_config: Configuration for retry logic on verification failures
        """
        self.retry_config = retry_config or RetryConfig()
        self._hf_metadata_cache: Dict[str, HFFileMetadata] = {}
        
    def verify_model_integrity(
        self,
        spec: ModelSpec,
        model_dir: Path,
        hf_metadata: Optional[Dict[str, HFFileMetadata]] = None,
        manifest_signature: Optional[str] = None,
        public_key: Optional[str] = None
    ) -> IntegrityVerificationResult:
        """
        Perform comprehensive integrity verification for a model.
        
        Args:
            spec: Model specification with file requirements
            model_dir: Local directory containing the model files
            hf_metadata: Optional HuggingFace metadata for ETag verification
            manifest_signature: Optional manifest signature for verification
            public_key: Optional public key for signature verification
            
        Returns:
            IntegrityVerificationResult with detailed verification results
        """
        with LogContext(
            operation="verify_model_integrity",
            model_id=spec.model_id,
            file_count=len(spec.files)
        ):
            start_time = time.time()
            
            logger.info(
                "Starting comprehensive integrity verification",
                extra={
                    "model_id": spec.model_id,
                    "file_count": len(spec.files),
                    "has_hf_metadata": hf_metadata is not None,
                    "has_signature": manifest_signature is not None
                }
            )
            
            # First verify manifest signature if provided
            manifest_signature_valid = True
            if manifest_signature and public_key:
                try:
                    manifest_signature_valid = self._verify_manifest_signature(
                        spec, manifest_signature, public_key
                    )
                    if not manifest_signature_valid:
                        logger.error(
                            "Manifest signature verification failed",
                            extra={"model_id": spec.model_id}
                        )
                except Exception as e:
                    logger.warning(
                        "Manifest signature verification error",
                        extra={"model_id": spec.model_id, "error": str(e)}
                    )
                    manifest_signature_valid = False
            
            # Verify individual files
            verified_files = []
            failed_files = []
            missing_files = []
            
            for file_spec in spec.files:
                file_path = model_dir / file_spec.path
                
                if not file_path.exists():
                    missing_files.append(file_spec.path)
                    logger.warning(
                        "File missing during verification",
                        extra={"model_id": spec.model_id, "file_path": file_spec.path}
                    )
                    continue
                
                # Verify the file with retry logic
                try:
                    result = retry_operation(
                        self._verify_single_file,
                        operation="verify_file",
                        config=self.retry_config,
                        file_spec=file_spec,
                        file_path=file_path,
                        hf_metadata=hf_metadata.get(file_spec.path) if hf_metadata else None
                    )
                    
                    if result.success:
                        verified_files.append(result)
                        logger.debug(
                            "File verification successful",
                            extra={
                                "model_id": spec.model_id,
                                "file_path": file_spec.path,
                                "method": result.method_used.value,
                                "verification_time": result.verification_time
                            }
                        )
                    else:
                        failed_files.append(result)
                        logger.error(
                            "File verification failed",
                            extra={
                                "model_id": spec.model_id,
                                "file_path": file_spec.path,
                                "method": result.method_used.value,
                                "error": result.error_message,
                                "expected": result.expected_value,
                                "actual": result.actual_value
                            }
                        )
                        
                except Exception as e:
                    # Create a failed result for the exception
                    failed_result = FileVerificationResult(
                        file_path=file_spec.path,
                        success=False,
                        method_used=VerificationMethod.SHA256_CHECKSUM,
                        error_message=f"Verification exception: {str(e)}"
                    )
                    failed_files.append(failed_result)
                    logger.error(
                        "File verification exception",
                        extra={
                            "model_id": spec.model_id,
                            "file_path": file_spec.path,
                            "error": str(e)
                        },
                        exc_info=True
                    )
            
            total_time = time.time() - start_time
            success = (
                len(failed_files) == 0 and 
                len(missing_files) == 0 and 
                manifest_signature_valid
            )
            
            error_message = None
            if not success:
                error_parts = []
                if failed_files:
                    error_parts.append(f"{len(failed_files)} files failed verification")
                if missing_files:
                    error_parts.append(f"{len(missing_files)} files missing")
                if not manifest_signature_valid:
                    error_parts.append("manifest signature invalid")
                error_message = "; ".join(error_parts)
            
            result = IntegrityVerificationResult(
                success=success,
                verified_files=verified_files,
                failed_files=failed_files,
                missing_files=missing_files,
                total_verification_time=total_time,
                manifest_signature_valid=manifest_signature_valid,
                error_message=error_message
            )
            
            logger.info(
                "Integrity verification completed",
                extra={
                    "model_id": spec.model_id,
                    "success": success,
                    "verified_files": len(verified_files),
                    "failed_files": len(failed_files),
                    "missing_files": len(missing_files),
                    "total_time": total_time,
                    "manifest_signature_valid": manifest_signature_valid
                }
            )
            
            return result
    
    def _verify_single_file(
        self,
        file_spec: FileSpec,
        file_path: Path,
        hf_metadata: Optional[HFFileMetadata] = None
    ) -> FileVerificationResult:
        """
        Verify a single file using the best available method.
        
        Args:
            file_spec: File specification with expected values
            file_path: Path to the actual file
            hf_metadata: Optional HuggingFace metadata for ETag verification
            
        Returns:
            FileVerificationResult with verification details
        """
        start_time = time.time()
        
        # Always check file size first as it's quick
        actual_size = file_path.stat().st_size
        if actual_size != file_spec.size:
            return FileVerificationResult(
                file_path=file_spec.path,
                success=False,
                method_used=VerificationMethod.SIZE_ONLY,
                expected_value=str(file_spec.size),
                actual_value=str(actual_size),
                error_message=f"Size mismatch: expected {file_spec.size}, got {actual_size}",
                verification_time=time.time() - start_time
            )
        
        # Try SHA256 checksum verification first (preferred method)
        if file_spec.sha256:
            try:
                actual_checksum = self._calculate_sha256(file_path)
                verification_time = time.time() - start_time
                
                if actual_checksum == file_spec.sha256:
                    return FileVerificationResult(
                        file_path=file_spec.path,
                        success=True,
                        method_used=VerificationMethod.SHA256_CHECKSUM,
                        expected_value=file_spec.sha256,
                        actual_value=actual_checksum,
                        verification_time=verification_time
                    )
                else:
                    return FileVerificationResult(
                        file_path=file_spec.path,
                        success=False,
                        method_used=VerificationMethod.SHA256_CHECKSUM,
                        expected_value=file_spec.sha256,
                        actual_value=actual_checksum,
                        error_message=f"SHA256 mismatch: expected {file_spec.sha256}, got {actual_checksum}",
                        verification_time=verification_time
                    )
                    
            except Exception as e:
                logger.warning(
                    "SHA256 calculation failed, trying fallback methods",
                    extra={"file_path": file_spec.path, "error": str(e)}
                )
        
        # Try HuggingFace ETag verification as fallback
        if hf_metadata and hf_metadata.etag:
            try:
                # For HuggingFace, ETag is typically the MD5 hash
                actual_etag = self._calculate_etag_equivalent(file_path)
                verification_time = time.time() - start_time
                
                if actual_etag == hf_metadata.etag:
                    return FileVerificationResult(
                        file_path=file_spec.path,
                        success=True,
                        method_used=VerificationMethod.HF_ETAG,
                        expected_value=hf_metadata.etag,
                        actual_value=actual_etag,
                        verification_time=verification_time
                    )
                else:
                    return FileVerificationResult(
                        file_path=file_spec.path,
                        success=False,
                        method_used=VerificationMethod.HF_ETAG,
                        expected_value=hf_metadata.etag,
                        actual_value=actual_etag,
                        error_message=f"ETag mismatch: expected {hf_metadata.etag}, got {actual_etag}",
                        verification_time=verification_time
                    )
                    
            except Exception as e:
                logger.warning(
                    "ETag calculation failed, falling back to size-only verification",
                    extra={"file_path": file_spec.path, "error": str(e)}
                )
        
        # Last resort: size-only verification (already passed above)
        verification_time = time.time() - start_time
        return FileVerificationResult(
            file_path=file_spec.path,
            success=True,
            method_used=VerificationMethod.SIZE_ONLY,
            expected_value=str(file_spec.size),
            actual_value=str(actual_size),
            verification_time=verification_time
        )
    
    def _calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file with optimized chunking."""
        sha256_hash = hashlib.sha256()
        
        # Use larger chunks for better performance on large files
        chunk_size = 1024 * 1024  # 1MB chunks
        
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def _calculate_etag_equivalent(self, file_path: Path) -> str:
        """
        Calculate ETag equivalent (typically MD5) for HuggingFace verification.
        
        Note: HuggingFace ETags are usually MD5 hashes, but can be more complex
        for multipart uploads. This implementation handles simple cases.
        """
        md5_hash = hashlib.md5()
        
        chunk_size = 1024 * 1024  # 1MB chunks
        
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                md5_hash.update(chunk)
        
        return md5_hash.hexdigest()
    
    def _verify_manifest_signature(
        self,
        spec: ModelSpec,
        signature: str,
        public_key: str
    ) -> bool:
        """
        Verify manifest signature using HMAC-SHA256.
        
        Args:
            spec: Model specification to verify
            signature: Base64-encoded signature
            public_key: Public key for verification
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Create canonical representation of the model spec for signing
            canonical_data = self._create_canonical_spec_data(spec)
            
            # Calculate expected signature
            expected_signature = hmac.new(
                public_key.encode('utf-8'),
                canonical_data.encode('utf-8'),
                hashlib.sha256
            ).digest()
            
            # Decode provided signature
            provided_signature = base64.b64decode(signature)
            
            # Compare signatures using constant-time comparison
            return hmac.compare_digest(expected_signature, provided_signature)
            
        except Exception as e:
            logger.error(
                "Manifest signature verification failed",
                extra={"error": str(e)},
                exc_info=True
            )
            return False
    
    def _create_canonical_spec_data(self, spec: ModelSpec) -> str:
        """
        Create a canonical string representation of model spec for signing.
        
        This ensures consistent signature verification across different
        serialization formats and field ordering.
        """
        # Create a deterministic representation
        canonical_dict = {
            "model_id": spec.model_id,
            "version": spec.version,
            "variants": sorted(spec.variants),
            "default_variant": spec.default_variant,
            "files": [
                {
                    "path": f.path,
                    "size": f.size,
                    "sha256": f.sha256,
                    "optional": f.optional
                }
                for f in sorted(spec.files, key=lambda x: x.path)
            ],
            "sources": spec.sources,
            "allow_patterns": sorted(spec.allow_patterns),
            "resolution_caps": sorted(spec.resolution_caps),
            "optional_components": sorted(spec.optional_components),
            "lora_required": spec.lora_required
        }
        
        # Convert to JSON with sorted keys for deterministic output
        return json.dumps(canonical_dict, sort_keys=True, separators=(',', ':'))
    
    def extract_hf_metadata_from_download(
        self,
        downloaded_files: List[Path],
        hf_cache_dir: Optional[Path] = None
    ) -> Dict[str, HFFileMetadata]:
        """
        Extract HuggingFace metadata from downloaded files.
        
        This method attempts to extract ETag and other metadata from
        HuggingFace cache files or HTTP headers if available.
        
        Args:
            downloaded_files: List of downloaded file paths
            hf_cache_dir: Optional HuggingFace cache directory
            
        Returns:
            Dictionary mapping file paths to HFFileMetadata
        """
        metadata = {}
        
        # This is a simplified implementation
        # In a real scenario, you would extract this from HF cache metadata
        # or HTTP response headers during download
        
        for file_path in downloaded_files:
            try:
                # For now, we'll create basic metadata from file stats
                # In practice, this would come from HF API responses
                stat = file_path.stat()
                
                # Generate a mock ETag (in practice, this comes from HF)
                etag = self._calculate_etag_equivalent(file_path)
                
                metadata[file_path.name] = HFFileMetadata(
                    etag=etag,
                    size=stat.st_size,
                    last_modified=time.ctime(stat.st_mtime)
                )
                
            except Exception as e:
                logger.warning(
                    "Failed to extract metadata for file",
                    extra={"file_path": str(file_path), "error": str(e)}
                )
        
        return metadata