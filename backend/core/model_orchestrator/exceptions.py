"""
Exception classes for the Model Orchestrator system.
"""

from enum import Enum
from typing import List, Optional


class ErrorCode(Enum):
    """Structured error codes for consistent error handling."""
    
    # Configuration Errors
    INVALID_CONFIG = "INVALID_CONFIG"
    MISSING_MANIFEST = "MISSING_MANIFEST"
    SCHEMA_VERSION_MISMATCH = "SCHEMA_VERSION_MISMATCH"
    
    # Model Errors
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    VARIANT_NOT_FOUND = "VARIANT_NOT_FOUND"
    INVALID_MODEL_ID = "INVALID_MODEL_ID"
    
    # Validation Errors
    MANIFEST_VALIDATION_ERROR = "MANIFEST_VALIDATION_ERROR"
    PATH_TRAVERSAL_DETECTED = "PATH_TRAVERSAL_DETECTED"
    CASE_COLLISION_DETECTED = "CASE_COLLISION_DETECTED"
    
    # Storage Errors
    NO_SPACE = "NO_SPACE"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    PATH_TOO_LONG = "PATH_TOO_LONG"
    FILESYSTEM_ERROR = "FILESYSTEM_ERROR"
    
    # Network Errors
    AUTH_FAIL = "AUTH_FAIL"
    RATE_LIMIT = "RATE_LIMIT"
    NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    
    # Integrity Errors
    CHECKSUM_FAIL = "CHECKSUM_FAIL"
    SIZE_MISMATCH = "SIZE_MISMATCH"
    INCOMPLETE_DOWNLOAD = "INCOMPLETE_DOWNLOAD"
    INTEGRITY_VERIFICATION_ERROR = "INTEGRITY_VERIFICATION_ERROR"
    MANIFEST_SIGNATURE_ERROR = "MANIFEST_SIGNATURE_ERROR"
    
    # Concurrency Errors
    LOCK_TIMEOUT = "LOCK_TIMEOUT"
    LOCK_ERROR = "LOCK_ERROR"
    CONCURRENT_MODIFICATION = "CONCURRENT_MODIFICATION"
    
    # Generic Errors
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class ModelOrchestratorError(Exception):
    """Base exception for all Model Orchestrator errors."""
    
    def __init__(self, message: str, error_code: ErrorCode, details: Optional[dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class ModelNotFoundError(ModelOrchestratorError):
    """Raised when a requested model is not found in the manifest."""
    
    def __init__(self, model_id: str, available_models: Optional[List[str]] = None):
        message = f"Model '{model_id}' not found in manifest"
        if available_models:
            message += f". Available models: {', '.join(available_models)}"
        super().__init__(
            message,
            ErrorCode.MODEL_NOT_FOUND,
            {"model_id": model_id, "available_models": available_models}
        )


class VariantNotFoundError(ModelOrchestratorError):
    """Raised when a requested variant is not available for a model."""
    
    def __init__(self, model_id: str, variant: str, available_variants: Optional[List[str]] = None):
        message = f"Variant '{variant}' not found for model '{model_id}'"
        if available_variants:
            message += f". Available variants: {', '.join(available_variants)}"
        super().__init__(
            message,
            ErrorCode.VARIANT_NOT_FOUND,
            {"model_id": model_id, "variant": variant, "available_variants": available_variants}
        )


class InvalidModelIdError(ModelOrchestratorError):
    """Raised when a model ID format is invalid."""
    
    def __init__(self, model_id: str, reason: str):
        message = f"Invalid model ID '{model_id}': {reason}"
        super().__init__(
            message,
            ErrorCode.INVALID_MODEL_ID,
            {"model_id": model_id, "reason": reason}
        )


class ManifestValidationError(ModelOrchestratorError):
    """Raised when manifest validation fails."""
    
    def __init__(self, errors: List[str]):
        message = f"Manifest validation failed: {'; '.join(errors)}"
        super().__init__(
            message,
            ErrorCode.MANIFEST_VALIDATION_ERROR,
            {"validation_errors": errors}
        )


class SchemaVersionError(ModelOrchestratorError):
    """Raised when manifest schema version is incompatible."""
    
    def __init__(self, found_version: str, supported_versions: List[str]):
        message = f"Unsupported schema version '{found_version}'. Supported versions: {', '.join(supported_versions)}"
        super().__init__(
            message,
            ErrorCode.SCHEMA_VERSION_MISMATCH,
            {"found_version": found_version, "supported_versions": supported_versions}
        )


class LockTimeoutError(ModelOrchestratorError):
    """Raised when a lock cannot be acquired within the specified timeout."""
    
    def __init__(self, message: str, model_id: Optional[str] = None, timeout: Optional[float] = None):
        super().__init__(
            message,
            ErrorCode.LOCK_TIMEOUT,
            {"model_id": model_id, "timeout": timeout}
        )


class LockError(ModelOrchestratorError):
    """Raised when a lock operation fails."""
    
    def __init__(self, message: str, model_id: Optional[str] = None):
        super().__init__(
            message,
            ErrorCode.LOCK_ERROR,
            {"model_id": model_id}
        )


class NoSpaceError(ModelOrchestratorError):
    """Raised when insufficient disk space is available."""
    
    def __init__(self, bytes_needed: int, bytes_available: int, path: str):
        message = f"Insufficient disk space. Need {bytes_needed} bytes, have {bytes_available} bytes at {path}"
        super().__init__(
            message,
            ErrorCode.NO_SPACE,
            {"bytes_needed": bytes_needed, "bytes_available": bytes_available, "path": path}
        )


class ChecksumError(ModelOrchestratorError):
    """Raised when file checksum verification fails."""
    
    def __init__(self, file_path: str, expected: str, actual: str):
        message = f"Checksum mismatch for {file_path}. Expected: {expected}, Got: {actual}"
        super().__init__(
            message,
            ErrorCode.CHECKSUM_FAIL,
            {"file_path": file_path, "expected": expected, "actual": actual}
        )


class SizeMismatchError(ModelOrchestratorError):
    """Raised when file size doesn't match expected size."""
    
    def __init__(self, file_path: str, expected: int, actual: int):
        message = f"Size mismatch for {file_path}. Expected: {expected} bytes, Got: {actual} bytes"
        super().__init__(
            message,
            ErrorCode.SIZE_MISMATCH,
            {"file_path": file_path, "expected": expected, "actual": actual}
        )


class IncompleteDownloadError(ModelOrchestratorError):
    """Raised when a download is incomplete."""
    
    def __init__(self, message: str, missing_files: Optional[List[str]] = None):
        super().__init__(
            message,
            ErrorCode.INCOMPLETE_DOWNLOAD,
            {"missing_files": missing_files or []}
        )


class IntegrityVerificationError(ModelOrchestratorError):
    """Raised when comprehensive integrity verification fails."""
    
    def __init__(self, message: str, failed_files: Optional[List[str]] = None, missing_files: Optional[List[str]] = None):
        super().__init__(
            message,
            ErrorCode.INTEGRITY_VERIFICATION_ERROR,
            {"failed_files": failed_files or [], "missing_files": missing_files or []}
        )


class ManifestSignatureError(ModelOrchestratorError):
    """Raised when manifest signature verification fails."""
    
    def __init__(self, message: str, model_id: Optional[str] = None):
        super().__init__(
            message,
            ErrorCode.MANIFEST_SIGNATURE_ERROR,
            {"model_id": model_id}
        )


class ModelValidationError(ModelOrchestratorError):
    """Raised when model validation fails."""
    
    def __init__(self, message: str, model_id: Optional[str] = None):
        super().__init__(
            message,
            ErrorCode.MANIFEST_VALIDATION_ERROR,
            {"model_id": model_id}
        )


class InvalidInputError(ModelOrchestratorError):
    """Raised when input validation fails for a model type."""
    
    def __init__(self, message: str, model_type: Optional[str] = None, input_data: Optional[dict] = None):
        super().__init__(
            message,
            ErrorCode.INVALID_CONFIG,
            {"model_type": model_type, "input_data": input_data}
        )


class MigrationError(ModelOrchestratorError):
    """Raised when configuration migration fails."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message,
            ErrorCode.INVALID_CONFIG,
            details
        )


class ValidationError(ModelOrchestratorError):
    """Raised when validation fails."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message,
            ErrorCode.MANIFEST_VALIDATION_ERROR,
            details
        )