"""
Model Orchestrator - Unified model management system for WAN2.2

This package provides manifest-driven model discovery, downloading, and path resolution
with support for multiple storage backends and atomic operations.
"""

from .model_registry import ModelRegistry, ModelSpec, FileSpec
from .model_resolver import ModelResolver
from .lock_manager import LockManager
from .model_ensurer import ModelEnsurer, ModelStatus, ModelStatusInfo, VerificationResult
from .garbage_collector import GarbageCollector, GCConfig, GCResult, GCTrigger, ModelInfo, DiskUsage
from .exceptions import (
    ModelOrchestratorError,
    ModelNotFoundError,
    VariantNotFoundError,
    InvalidModelIdError,
    ManifestValidationError,
    SchemaVersionError,
    LockTimeoutError,
    LockError,
    NoSpaceError,
    ChecksumError,
    SizeMismatchError,
    IncompleteDownloadError,
)

__version__ = "0.1.0"

__all__ = [
    "ModelRegistry",
    "ModelSpec", 
    "FileSpec",
    "ModelResolver",
    "LockManager",
    "ModelEnsurer",
    "ModelStatus",
    "ModelStatusInfo",
    "VerificationResult",
    "GarbageCollector",
    "GCConfig",
    "GCResult",
    "GCTrigger",
    "ModelInfo",
    "DiskUsage",
    "ModelOrchestratorError",
    "ModelNotFoundError",
    "VariantNotFoundError", 
    "InvalidModelIdError",
    "ManifestValidationError",
    "SchemaVersionError",
    "LockTimeoutError",
    "LockError",
    "NoSpaceError",
    "ChecksumError",
    "SizeMismatchError",
    "IncompleteDownloadError",
]