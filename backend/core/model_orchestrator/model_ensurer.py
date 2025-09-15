"""
Model Ensurer - Atomic download orchestration with preflight checks.
"""

import os
import json
import shutil
import hashlib
import time
import uuid
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable
from collections import defaultdict

from .model_registry import ModelRegistry, ModelSpec, FileSpec
from .model_resolver import ModelResolver
from .lock_manager import LockManager
from .storage_backends.base_store import StorageBackend, DownloadResult
from .exceptions import (
    ModelOrchestratorError, ErrorCode, NoSpaceError, ChecksumError, 
    SizeMismatchError, IncompleteDownloadError
)
from .integrity_verifier import IntegrityVerifier, IntegrityVerificationResult
from .error_recovery import ErrorRecoveryManager, RetryConfig, retry_operation
from .logging_config import get_logger, LogContext, log_operation_start, log_operation_success, log_operation_failure
from .metrics import get_metrics_collector
from .component_deduplicator import ComponentDeduplicator
from .wan22_handler import WAN22ModelHandler

logger = get_logger(__name__)


class ModelStatus(Enum):
    """Status of a model in the local storage."""
    NOT_PRESENT = "NOT_PRESENT"
    PARTIAL = "PARTIAL"
    VERIFYING = "VERIFYING"
    COMPLETE = "COMPLETE"
    CORRUPT = "CORRUPT"


@dataclass
class VerificationResult:
    """Result of model integrity verification."""
    success: bool
    verified_files: List[str]
    failed_files: List[str]
    missing_files: List[str]
    error_message: Optional[str] = None


@dataclass
class ModelStatusInfo:
    """Detailed status information for a model."""
    status: ModelStatus
    local_path: Optional[str] = None
    missing_files: List[str] = None
    bytes_needed: int = 0
    verification_result: Optional[VerificationResult] = None


@dataclass
class FailedSource:
    """Information about a failed source."""
    url: str
    failed_at: float
    error_message: str
    retry_count: int = 0


class ModelEnsurer:
    """Orchestrates atomic model downloads with preflight checks."""
    
    def __init__(
        self,
        registry: ModelRegistry,
        resolver: ModelResolver,
        lock_manager: LockManager,
        storage_backends: List[StorageBackend],
        safety_margin_bytes: int = 1024 * 1024 * 1024,
        negative_cache_ttl: float = 300.0,  # 5 minutes
        retry_config: Optional[RetryConfig] = None,
        enable_deduplication: bool = True
    ):
        self.registry = registry
        self.resolver = resolver
        self.lock_manager = lock_manager
        self.storage_backends = storage_backends
        self.safety_margin_bytes = safety_margin_bytes
        self.negative_cache_ttl = negative_cache_ttl
        self.enable_deduplication = enable_deduplication
        
        # Initialize error recovery manager
        self.error_recovery = ErrorRecoveryManager(retry_config)
        
        # Initialize metrics collector
        self.metrics = get_metrics_collector()
        
        # Initialize integrity verifier
        self.integrity_verifier = IntegrityVerifier()
        
        # Initialize component deduplicator if enabled
        if self.enable_deduplication:
            self.deduplicator = ComponentDeduplicator()
        else:
            self.deduplicator = None
            
        # Initialize WAN22 handler
        self.wan22_handler = WAN22ModelHandler()
        
        # Failed sources cache (negative cache)
        self.failed_sources: Dict[str, FailedSource] = {}
        
        # Storage backend mapping
        self.backend_map = {backend.scheme: backend for backend in storage_backends}
        
        logger.info(f"ModelEnsurer initialized with {len(storage_backends)} backends")

    def ensure(self, model_id: str, variant: str | None = None) -> str:
        """
        Ensure a model is available locally, downloading if necessary.
        
        Args:
            model_id: The model identifier
            variant: Optional model variant (defaults to model's default_variant)
            
        Returns:
            str: Path to the ensured model directory
            
        Raises:
            ModelOrchestratorError: If model cannot be ensured
        """
        operation_id = str(uuid.uuid4())
        
        with LogContext(operation_id=operation_id, model_id=model_id, variant=variant):
            log_operation_start("model_ensure", {
                "model_id": model_id,
                "variant": variant
            })
            
            try:
                # Get model specification
                spec = self.registry.get_model_spec(model_id)
                if not spec:
                    raise ModelOrchestratorError(
                        f"Model {model_id} not found in registry",
                        ErrorCode.MODEL_NOT_FOUND
                    )
                
                # Resolve variant
                if variant is None:
                    variant = spec.default_variant
                
                if variant not in spec.variants:
                    raise ModelOrchestratorError(
                        f"Variant {variant} not available for model {model_id}",
                        ErrorCode.VARIANT_NOT_FOUND
                    )
                
                # Get local path
                local_path = self.resolver.get_local_path(model_id, variant)
                
                # Check current status
                status_info = self.get_model_status(model_id, variant)
                
                if status_info.status == ModelStatus.COMPLETE:
                    logger.info(f"Model {model_id}@{variant} already available at {local_path}")
                    log_operation_success("model_ensure", {"local_path": str(local_path)})
                    return str(local_path)
                
                # Acquire lock for this model
                lock_key = f"{model_id}@{variant}"
                with self.lock_manager.acquire_lock(lock_key):
                    # Re-check status after acquiring lock
                    status_info = self.get_model_status(model_id, variant)
                    if status_info.status == ModelStatus.COMPLETE:
                        logger.info(f"Model {model_id}@{variant} became available while waiting for lock")
                        log_operation_success("model_ensure", {"local_path": str(local_path)})
                        return str(local_path)
                    
                    # Download the model
                    result_path = self._download_model(spec, variant, local_path, operation_id)
                    
                    log_operation_success("model_ensure", {"local_path": result_path})
                    return result_path
                    
            except Exception as e:
                log_operation_failure("model_ensure", str(e))
                self.metrics.increment_counter("model_ensure_failures", {
                    "model_id": model_id,
                    "variant": variant or "default",
                    "error_type": type(e).__name__
                })
                raise

    def get_model_status(self, model_id: str, variant: str | None = None) -> ModelStatusInfo:
        """
        Get the current status of a model.
        
        Args:
            model_id: The model identifier
            variant: Optional model variant
            
        Returns:
            ModelStatusInfo: Current status information
        """
        try:
            # Get model specification
            spec = self.registry.get_model_spec(model_id)
            if not spec:
                return ModelStatusInfo(
                    status=ModelStatus.NOT_PRESENT,
                    error_message=f"Model {model_id} not found in registry"
                )
            
            # Resolve variant
            if variant is None:
                variant = spec.default_variant
            
            # Get local path
            local_path = Path(self.resolver.get_local_path(model_id, variant))
            
            if not local_path.exists():
                return ModelStatusInfo(
                    status=ModelStatus.NOT_PRESENT,
                    local_path=str(local_path)
                )
            
            # Check file completeness
            missing_files = []
            corrupt_files = []
            bytes_needed = 0
            
            for file_spec in spec.files:
                file_path = local_path / file_spec.path
                
                if not file_path.exists():
                    missing_files.append(file_spec.path)
                    bytes_needed += file_spec.size
                elif file_path.stat().st_size != file_spec.size:
                    corrupt_files.append(file_spec.path)
                    bytes_needed += file_spec.size
            
            if missing_files or corrupt_files:
                return ModelStatusInfo(
                    status=ModelStatus.PARTIAL,
                    local_path=str(local_path),
                    missing_files=missing_files + corrupt_files,
                    bytes_needed=bytes_needed
                )
            
            # Verify integrity
            verification_result = self._verify_model_integrity(spec, local_path)
            
            if verification_result.success:
                return ModelStatusInfo(
                    status=ModelStatus.COMPLETE,
                    local_path=str(local_path),
                    verification_result=verification_result
                )
            else:
                return ModelStatusInfo(
                    status=ModelStatus.CORRUPT,
                    local_path=str(local_path),
                    missing_files=verification_result.failed_files,
                    verification_result=verification_result
                )
                
        except Exception as e:
            logger.error(f"Error checking model status: {e}")
            return ModelStatusInfo(
                status=ModelStatus.NOT_PRESENT,
                error_message=str(e)
            )

    def _download_model(self, spec: ModelSpec, variant: str, local_path: Path, operation_id: str) -> str:
        """Download a model to the local path."""
        logger.info(f"Starting download of {spec.model_id}@{variant} to {local_path}")
        
        # Create temporary directory for atomic download
        temp_dir = local_path.parent / f".tmp_{spec.model_id}_{variant}_{operation_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Preflight checks
            self._preflight_checks(spec, temp_dir)
            
            # Get sources for this model
            sources = spec.sources.priority if spec.sources else []
            
            if not sources:
                raise ModelOrchestratorError(
                    f"No sources configured for model {spec.model_id}",
                    ErrorCode.NO_SOURCES_AVAILABLE
                )
            
            # Try each source until one succeeds
            last_error = None
            for source_url in sources:
                if self._is_source_failed(source_url):
                    logger.debug(f"Skipping failed source: {source_url}")
                    continue
                
                try:
                    logger.info(f"Attempting download from source: {source_url}")
                    self._download_from_source(spec, variant, source_url, temp_dir, operation_id)
                    
                    # Verify downloaded files
                    verification_result = self._verify_model_integrity(spec, temp_dir)
                    if not verification_result.success:
                        raise ChecksumError(
                            f"Integrity verification failed: {verification_result.error_message}"
                        )
                    
                    # Atomic move to final location
                    if local_path.exists():
                        shutil.rmtree(local_path)
                    
                    shutil.move(str(temp_dir), str(local_path))
                    
                    logger.info(f"Successfully downloaded {spec.model_id}@{variant}")
                    self.metrics.increment_counter("model_downloads_success", {
                        "model_id": spec.model_id,
                        "variant": variant,
                        "source": source_url
                    })
                    
                    return str(local_path)
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"Download failed from {source_url}: {e}")
                    self._mark_source_failed(source_url, str(e))
                    
                    # Clean up partial download
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                        temp_dir.mkdir(parents=True, exist_ok=True)
                    
                    continue
            
            # All sources failed
            raise ModelOrchestratorError(
                f"All sources failed for model {spec.model_id}@{variant}. Last error: {last_error}",
                ErrorCode.ALL_SOURCES_FAILED
            )
            
        finally:
            # Clean up temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def _preflight_checks(self, spec: ModelSpec, temp_dir: Path):
        """Perform preflight checks before downloading."""
        # Calculate total size needed
        total_size = sum(file_spec.size for file_spec in spec.files)
        
        # Check available disk space
        available_space = shutil.disk_usage(temp_dir.parent).free
        
        if available_space < total_size + self.safety_margin_bytes:
            raise NoSpaceError(
                f"Insufficient disk space. Need {total_size + self.safety_margin_bytes} bytes, "
                f"have {available_space} bytes"
            )
        
        logger.info(f"Preflight checks passed. Will download {total_size} bytes")

    def _download_from_source(self, spec: ModelSpec, variant: str, source_url: str, 
                            temp_dir: Path, operation_id: str):
        """Download model files from a specific source."""
        # Parse source URL to get backend
        scheme = source_url.split("://")[0]
        backend = self.backend_map.get(scheme)
        
        if not backend:
            raise ModelOrchestratorError(
                f"No backend available for scheme: {scheme}",
                ErrorCode.BACKEND_NOT_AVAILABLE
            )
        
        # Prepare file list for download
        files_to_download = []
        for file_spec in spec.files:
            # Apply deduplication if enabled
            if self.deduplicator and self.deduplicator.is_component_available(file_spec.component):
                logger.debug(f"Skipping {file_spec.path} - component {file_spec.component} already available")
                continue
            
            files_to_download.append(file_spec)
        
        if not files_to_download:
            logger.info("All files already available through deduplication")
            return
        
        # Create progress callback
        def progress_callback(downloaded: int, total: int, filename: str):
            self.metrics.set_gauge("download_progress", downloaded / total * 100, {
                "model_id": spec.model_id,
                "variant": variant,
                "operation_id": operation_id
            })
        
        # Download with retry logic
        download_start_time = time.time()
        
        try:
            retry_operation(
                backend.download,
                operation="download_from_source",
                config=self.error_recovery.config,
                source_url=source_url,
                local_dir=temp_dir,
                file_specs=files_to_download,
                allow_patterns=getattr(spec, 'allow_patterns', None),
                progress_callback=progress_callback
            )
            
            download_duration = time.time() - download_start_time
            self.metrics.record_histogram("download_duration_seconds", download_duration, {
                "model_id": spec.model_id,
                "variant": variant,
                "source": source_url
            })
            
        except Exception as e:
            self.metrics.increment_counter("download_failures", {
                "model_id": spec.model_id,
                "variant": variant,
                "source": source_url,
                "error_type": type(e).__name__
            })
            raise

    def _verify_model_integrity(self, spec: ModelSpec, model_path: Path) -> VerificationResult:
        """Verify the integrity of downloaded model files."""
        logger.info(f"Verifying integrity of {spec.model_id} at {model_path}")
        
        verified_files = []
        failed_files = []
        missing_files = []
        
        for file_spec in spec.files:
            file_path = model_path / file_spec.path
            
            if not file_path.exists():
                missing_files.append(file_spec.path)
                continue
            
            # Check file size
            actual_size = file_path.stat().st_size
            if actual_size != file_spec.size:
                logger.error(f"Size mismatch for {file_spec.path}: expected {file_spec.size}, got {actual_size}")
                failed_files.append(file_spec.path)
                continue
            
            # Check SHA256 if available
            if file_spec.sha256:
                try:
                    actual_hash = self._calculate_file_hash(file_path)
                    if actual_hash != file_spec.sha256:
                        logger.error(f"Hash mismatch for {file_spec.path}: expected {file_spec.sha256}, got {actual_hash}")
                        failed_files.append(file_spec.path)
                        continue
                except Exception as e:
                    logger.error(f"Error calculating hash for {file_spec.path}: {e}")
                    failed_files.append(file_spec.path)
                    continue
            
            verified_files.append(file_spec.path)
        
        success = len(failed_files) == 0 and len(missing_files) == 0
        
        return VerificationResult(
            success=success,
            verified_files=verified_files,
            failed_files=failed_files,
            missing_files=missing_files,
            error_message=None if success else f"Verification failed: {len(failed_files)} corrupt, {len(missing_files)} missing"
        )

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _is_source_failed(self, source_url: str) -> bool:
        """Check if a source is in the failed cache."""
        failed_source = self.failed_sources.get(source_url)
        if not failed_source:
            return False
        
        # Check if cache entry has expired
        if time.time() - failed_source.failed_at > self.negative_cache_ttl:
            del self.failed_sources[source_url]
            return False
        
        return True

    def _mark_source_failed(self, source_url: str, error_message: str):
        """Mark a source as failed in the cache."""
        existing = self.failed_sources.get(source_url)
        retry_count = existing.retry_count + 1 if existing else 1
        
        self.failed_sources[source_url] = FailedSource(
            url=source_url,
            failed_at=time.time(),
            error_message=error_message,
            retry_count=retry_count
        )
        
        logger.debug(f"Marked source as failed: {source_url} (retry #{retry_count})")

    def cleanup_failed_downloads(self):
        """Clean up any failed download artifacts."""
        # This would clean up temporary directories and partial downloads
        # Implementation depends on specific cleanup requirements
        pass

    def get_download_progress(self, operation_id: str) -> Dict[str, Any]:
        """Get progress information for an ongoing download."""
        # This would return progress information for a specific operation
        # Implementation depends on how progress tracking is implemented
        return {}