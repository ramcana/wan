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
        self.integrity_verifier = IntegrityVerifier(retry_config)
        
        # Initialize component deduplicator if enabled
        self.component_deduplicator = None
        if enable_deduplication:
            self.component_deduplicator = ComponentDeduplicator(resolver.models_root)
        
        # Optional garbage collector for disk space management
        self.garbage_collector = None
        
        # Negative cache for recently failed sources
        # Key: (model_id, source_url), Value: FailedSource
        self._failed_sources: Dict[tuple, FailedSource] = {}
        
        # Cache for WAN2.2 handlers
        self._wan22_handlers: Dict[str, WAN22ModelHandler] = {}
    
    def _extract_source_type(self, source_url: str) -> str:
        """Extract source type from URL for metrics labeling."""
        if not source_url:
            return 'unknown'
        
        if source_url.startswith('local://'):
            return 'local'
        elif source_url.startswith('s3://'):
            return 's3'
        elif source_url.startswith('hf://'):
            return 'huggingface'
        elif source_url.startswith('http://') or source_url.startswith('https://'):
            return 'http'
        else:
            return 'unknown'
    
    def set_garbage_collector(self, garbage_collector) -> None:
        """Set the garbage collector for automatic disk space management."""
        self.garbage_collector = garbage_collector
    
    def add_model_reference(self, model_id: str, model_path: str) -> None:
        """
        Add component references for a model that uses shared components.
        
        Args:
            model_id: Identifier for the model
            model_path: Path to the model directory
        """
        if self.component_deduplicator:
            self.component_deduplicator.add_model_reference(model_id, Path(model_path))
    
    def remove_model_reference(self, model_id: str) -> int:
        """
        Remove component references for a model and clean up orphaned components.
        
        Args:
            model_id: Identifier for the model being removed
            
        Returns:
            Number of bytes reclaimed from cleaning up orphaned components
        """
        if not self.component_deduplicator:
            return 0
        
        # Remove references and get orphaned components
        orphaned_components = self.component_deduplicator.remove_model_reference(model_id)
        
        # Clean up orphaned components
        bytes_reclaimed = 0
        if orphaned_components:
            bytes_reclaimed = self.component_deduplicator.cleanup_orphaned_components(orphaned_components)
            
            logger.info(
                f"Cleaned up orphaned components for model {model_id}",
                extra={
                    "orphaned_components": len(orphaned_components),
                    "bytes_reclaimed": bytes_reclaimed
                }
            )
        
        return bytes_reclaimed
    
    def deduplicate_across_models(self, model_ids: List[str]) -> Optional[Dict[str, any]]:
        """
        Perform cross-model deduplication for the specified models.
        
        Args:
            model_ids: List of model identifiers to deduplicate across
            
        Returns:
            Deduplication result statistics or None if deduplication is disabled
        """
        if not self.component_deduplicator:
            return None
        
        # Build model paths dictionary
        model_paths = {}
        for model_id in model_ids:
            try:
                model_path = Path(self.resolver.local_dir(model_id))
                if model_path.exists():
                    model_paths[model_id] = model_path
            except Exception as e:
                logger.warning(f"Failed to get path for model {model_id}: {e}")
        
        if not model_paths:
            logger.warning("No valid model paths found for cross-model deduplication")
            return None
        
        # Perform cross-model deduplication
        result = self.component_deduplicator.deduplicate_across_models(model_paths)
        
        return {
            "files_processed": result.total_files_processed,
            "duplicates_found": result.duplicates_found,
            "bytes_saved": result.bytes_saved,
            "links_created": result.links_created,
            "processing_time": result.processing_time,
            "errors": result.errors
        }
    
    def get_component_stats(self) -> Optional[Dict[str, any]]:
        """
        Get statistics about shared components.
        
        Returns:
            Component statistics or None if deduplication is disabled
        """
        if not self.component_deduplicator:
            return None
        
        return self.component_deduplicator.get_component_stats()
    
    def get_wan22_handler(self, model_spec: ModelSpec, model_dir: Path) -> Optional[WAN22ModelHandler]:
        """
        Get or create a WAN2.2 handler for the model.
        
        Args:
            model_spec: Model specification
            model_dir: Path to model directory
            
        Returns:
            WAN22ModelHandler if model is WAN2.2 compatible, None otherwise
        """
        # Only create handlers for WAN2.2 models
        if not model_spec.model_type or model_spec.model_type not in ["t2v", "i2v", "ti2v"]:
            return None
        
        cache_key = f"{model_spec.model_id}:{model_dir}"
        
        if cache_key not in self._wan22_handlers:
            try:
                self._wan22_handlers[cache_key] = WAN22ModelHandler(model_spec, model_dir)
                logger.debug(f"Created WAN2.2 handler for {model_spec.model_id}")
            except Exception as e:
                logger.warning(f"Failed to create WAN2.2 handler for {model_spec.model_id}: {e}")
                return None
        
        return self._wan22_handlers[cache_key]
    
    def get_selective_files_for_variant(self, model_spec: ModelSpec, variant: Optional[str] = None) -> List[FileSpec]:
        """
        Get selective files for download based on variant and WAN2.2 optimizations.
        
        Args:
            model_spec: Model specification
            variant: Model variant
            
        Returns:
            List of files to download (may be subset for development variants)
        """
        all_files = model_spec.files
        
        # For non-WAN2.2 models, return all files
        if not model_spec.model_type or model_spec.model_type not in ["t2v", "i2v", "ti2v"]:
            return all_files
        
        # Create temporary handler to get shard information
        temp_dir = Path("/tmp")  # Temporary path for handler creation
        try:
            handler = WAN22ModelHandler(model_spec, temp_dir)
            
            # If this is a development variant, get selective shards
            if variant and handler.is_development_variant(variant):
                selective_files = []
                
                # Always include non-shard files
                for file_spec in all_files:
                    if file_spec.shard_part is None:
                        selective_files.append(file_spec)
                
                # Add selective shards for each component
                for component_type in handler.components:
                    required_shards = handler.get_required_shards(component_type, variant)
                    for shard_info in required_shards:
                        # Find the corresponding file spec
                        for file_spec in all_files:
                            if (file_spec.component == component_type and 
                                file_spec.shard_part == shard_info.shard_part):
                                selective_files.append(file_spec)
                                break
                
                logger.info(f"Selective download for {variant}: {len(selective_files)}/{len(all_files)} files")
                return selective_files
            
        except Exception as e:
            logger.warning(f"Failed to create WAN2.2 handler for selective download: {e}")
        
        return all_files
    
    def ensure(
        self,
        model_id: str,
        variant: Optional[str] = None,
        force_redownload: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> str:
        """Ensure a model is available locally with comprehensive error recovery."""
        with LogContext(
            operation="ensure_model",
            model_id=model_id,
            variant=variant,
            force_redownload=force_redownload
        ) as ctx:
            log_operation_start("model_ensure", model_id=model_id, variant=variant)
            start_time = time.time()
            
            # Record download start metrics
            self.metrics.record_download_started(model_id, variant or "default", "orchestrator")
            
            try:
                # Call the internal method directly with retry logic
                with self.error_recovery.recovery_context(
                    "ensure_model",
                    model_id=model_id,
                    variant=variant
                ) as context:
                    result = self.error_recovery.retry_with_recovery(
                        self._ensure_with_recovery,
                        context,
                        model_id,
                        variant,
                        force_redownload,
                        progress_callback
                    )
                
                duration = time.time() - start_time
                
                # Record successful completion metrics
                # Calculate approximate bytes (would be more accurate in real implementation)
                try:
                    spec = self.registry.spec(model_id, variant)
                    total_bytes = sum(f.size for f in spec.files)
                    self.metrics.record_download_completed(
                        model_id, variant or "default", "orchestrator", duration, total_bytes
                    )
                except Exception:
                    # Fallback if we can't get size info
                    self.metrics.record_download_completed(
                        model_id, variant or "default", "orchestrator", duration, 0
                    )
                
                log_operation_success("model_ensure", duration=duration, model_id=model_id, local_path=result)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Record failure metrics
                error_code = getattr(e, 'error_code', ErrorCode.UNKNOWN_ERROR)
                self.metrics.record_download_failed(
                    model_id, variant or "default", "orchestrator", error_code.value
                )
                
                log_operation_failure("model_ensure", e, duration=duration, model_id=model_id)
                raise
    
    def _ensure_with_recovery(
        self,
        model_id: str,
        variant: Optional[str] = None,
        force_redownload: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> str:
        """Internal ensure method with error recovery."""
        spec = self.registry.spec(model_id, variant)
        local_path = Path(self.resolver.local_dir(model_id, variant))
        
        if not force_redownload:
            status_info = self.status(model_id, variant)
            if status_info.status == ModelStatus.COMPLETE:
                return str(local_path)
        
        with self.lock_manager.acquire_model_lock(model_id, timeout=300.0):
            if not force_redownload:
                status_info = self.status(model_id, variant)
                if status_info.status == ModelStatus.COMPLETE:
                    return str(local_path)
            
            self._preflight_checks(spec, self.garbage_collector)
            temp_dir = self._create_temp_directory(model_id)
            
            try:
                # Get selective files for this variant
                selective_files = self.get_selective_files_for_variant(spec, variant)
                download_metadata = self._download_to_temp(spec, temp_dir, progress_callback, selective_files)
                self._verify_integrity(spec, temp_dir, download_metadata, selective_files)
                self._atomic_move(temp_dir, local_path, spec.model_id)
                self._create_verification_marker(local_path, spec)
                return str(local_path)
            finally:
                self._cleanup_temp_directory(temp_dir)
    
    def status(self, model_id: str, variant: Optional[str] = None) -> ModelStatusInfo:
        """Get the current status of a model."""
        try:
            spec = self.registry.spec(model_id, variant)
            local_path = Path(self.resolver.local_dir(model_id, variant))
            
            if not local_path.exists():
                return ModelStatusInfo(
                    status=ModelStatus.NOT_PRESENT,
                    local_path=str(local_path),
                    missing_files=[f.path for f in spec.files],
                    bytes_needed=sum(f.size for f in spec.files)
                )
            
            verification_file = local_path / ".verified.json"
            if verification_file.exists():
                try:
                    with open(verification_file, 'r') as f:
                        verification_data = json.load(f)
                    
                    if self._quick_completeness_check(spec, local_path):
                        return ModelStatusInfo(
                            status=ModelStatus.COMPLETE,
                            local_path=str(local_path),
                            missing_files=[],
                            bytes_needed=0
                        )
                except (json.JSONDecodeError, IOError):
                    pass
            
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
                # Determine status based on the nature of the issues
                if corrupt_files and not missing_files:
                    # Only corrupt files, no missing files
                    status = ModelStatus.CORRUPT
                else:
                    # Missing files (with or without corrupt files) - treat as PARTIAL
                    # since we need to download/re-download files
                    status = ModelStatus.PARTIAL
                
                return ModelStatusInfo(
                    status=status,
                    local_path=str(local_path),
                    missing_files=missing_files + corrupt_files,
                    bytes_needed=bytes_needed
                )
            
            return ModelStatusInfo(
                status=ModelStatus.COMPLETE,
                local_path=str(local_path),
                missing_files=[],
                bytes_needed=0
            )
            
        except Exception as e:
            logger.error(f"Failed to get status for model {model_id}: {e}")
            raise
    
    def verify_integrity(self, model_id: str, variant: Optional[str] = None) -> VerificationResult:
        """Verify the integrity of a model using comprehensive verification."""
        try:
            spec = self.registry.spec(model_id, variant)
            local_path = Path(self.resolver.local_dir(model_id, variant))
            
            if not local_path.exists():
                return VerificationResult(
                    success=False,
                    verified_files=[],
                    failed_files=[],
                    missing_files=[f.path for f in spec.files],
                    error_message="Model directory does not exist"
                )
            
            # Use comprehensive integrity verification
            result = self.integrity_verifier.verify_model_integrity(
                spec=spec,
                model_dir=local_path
            )
            
            # Record integrity failures in metrics
            for failed_file in result.failed_files:
                self.metrics.record_integrity_failure(model_id, failed_file.file_path)
            
            # Convert to legacy VerificationResult format for compatibility
            return VerificationResult(
                success=result.success,
                verified_files=[f.file_path for f in result.verified_files],
                failed_files=[f.file_path for f in result.failed_files],
                missing_files=result.missing_files,
                error_message=result.error_message
            )
            
        except Exception as e:
            logger.error(f"Failed to verify integrity for model {model_id}: {e}")
            return VerificationResult(
                success=False,
                verified_files=[],
                failed_files=[],
                missing_files=[],
                error_message=str(e)
            )
    
    def estimate_download_size(self, model_id: str, variant: Optional[str] = None) -> int:
        """Estimate the download size for a model."""
        try:
            spec = self.registry.spec(model_id, variant)
            return sum(f.size for f in spec.files)
        except Exception as e:
            logger.error(f"Failed to estimate download size for model {model_id}: {e}")
            raise
    
    def _preflight_checks(self, spec: ModelSpec, garbage_collector=None) -> None:
        """Perform preflight checks before downloading."""
        total_bytes = sum(f.size for f in spec.files)
        models_root = Path(self.resolver.models_root)
        stat = shutil.disk_usage(models_root)
        available_bytes = stat.free
        
        logger.info(
            "Performing preflight checks",
            extra={
                "model_id": spec.model_id,
                "total_bytes": total_bytes,
                "available_bytes": available_bytes,
                "safety_margin_bytes": self.safety_margin_bytes
            }
        )
        
        if total_bytes + self.safety_margin_bytes > available_bytes:
            # Try garbage collection if available
            if garbage_collector:
                logger.info(
                    "Insufficient disk space, attempting garbage collection",
                    extra={
                        "model_id": spec.model_id,
                        "bytes_needed": total_bytes + self.safety_margin_bytes,
                        "bytes_available": available_bytes
                    }
                )
                
                try:
                    from .garbage_collector import GCTrigger
                    gc_result = garbage_collector.collect(dry_run=False, trigger=GCTrigger.LOW_DISK_SPACE)
                    
                    if gc_result.bytes_reclaimed > 0:
                        # Re-check disk space after GC
                        stat = shutil.disk_usage(models_root)
                        available_bytes = stat.free
                        
                        logger.info(
                            "Garbage collection completed",
                            extra={
                                "bytes_reclaimed": gc_result.bytes_reclaimed,
                                "models_removed": len(gc_result.models_removed),
                                "new_available_bytes": available_bytes
                            }
                        )
                        
                        # Check if we now have enough space
                        if total_bytes + self.safety_margin_bytes <= available_bytes:
                            return
                    
                except Exception as gc_error:
                    logger.warning(
                        "Garbage collection failed",
                        extra={"model_id": spec.model_id, "error": str(gc_error)},
                        exc_info=True
                    )
            
            # Still insufficient space after GC (or GC not available)
            raise NoSpaceError(
                bytes_needed=total_bytes + self.safety_margin_bytes,
                bytes_available=available_bytes,
                path=str(models_root)
            )
    
    def _create_temp_directory(self, model_id: str) -> Path:
        """Create a temporary directory for atomic downloads."""
        temp_name = f"{model_id}_{uuid.uuid4().hex[:8]}_{int(time.time())}.partial"
        temp_dir = Path(self.resolver.models_root) / ".tmp" / temp_name
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(
            "Created temporary directory",
            extra={"model_id": model_id, "temp_dir": str(temp_dir)}
        )
        
        return temp_dir
    
    def _download_to_temp(self, spec: ModelSpec, temp_dir: Path, progress_callback: Optional[Callable[[int, int], None]] = None, selective_files: Optional[List[FileSpec]] = None) -> Optional[dict]:
        """Download model files to temporary directory with error recovery."""
        # Clean up expired failures before starting
        self._cleanup_expired_failures()
        
        def attempt_download_from_source(source_url: str) -> Optional[dict]:
            """Attempt download from a single source with retry logic."""
            # Check negative cache first
            if self._is_source_recently_failed(spec.model_id, source_url):
                failed_source = self._failed_sources.get((spec.model_id, source_url))
                logger.info(
                    "Skipping recently failed source",
                    extra={
                        "model_id": spec.model_id,
                        "source_url": source_url,
                        "retry_count": failed_source.retry_count,
                        "last_error": failed_source.error_message
                    }
                )
                return None
            
            backend = None
            for b in self.storage_backends:
                if b.can_handle(source_url):
                    backend = b
                    break
            
            if not backend:
                logger.warning(
                    "No backend available for source",
                    extra={"model_id": spec.model_id, "source_url": source_url}
                )
                return None
            
            logger.info(
                "Attempting download from source",
                extra={"model_id": spec.model_id, "source_url": source_url}
            )
            
            # Record download start for this specific source
            source_type = self._extract_source_type(source_url)
            download_start_time = time.time()
            
            try:
                # Use selective files if provided, otherwise use all files
                files_to_download = selective_files if selective_files is not None else spec.files
                
                result = retry_operation(
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
                
                if result.success:
                    # Clear any previous failures for this source
                    self._clear_source_failure(spec.model_id, source_url)
                    
                    # Record successful download metrics
                    files_downloaded = selective_files if selective_files is not None else spec.files
                    total_bytes = sum(f.size for f in files_downloaded)
                    self.metrics.record_download_completed(
                        spec.model_id, spec.default_variant, source_type, 
                        download_duration, total_bytes
                    )
                    
                    logger.info(
                        "Successfully downloaded from source",
                        extra={
                            "model_id": spec.model_id, 
                            "source_url": source_url,
                            "duration": download_duration,
                            "bytes": total_bytes
                        }
                    )
                    # Return metadata for integrity verification
                    return result.metadata
                else:
                    self._mark_source_failed(spec.model_id, source_url, result.error_message)
                    
                    # Record failed download metrics
                    error_code = getattr(result, 'error_code', 'DOWNLOAD_FAILED')
                    self.metrics.record_download_failed(
                        spec.model_id, spec.default_variant, source_type, error_code
                    )
                    
                    logger.warning(
                        "Download failed from source",
                        extra={
                            "model_id": spec.model_id,
                            "source_url": source_url,
                            "error": result.error_message
                        }
                    )
                    return None
                    
            except Exception as e:
                error_message = str(e)
                self._mark_source_failed(spec.model_id, source_url, error_message)
                logger.warning(
                    "Download failed from source with exception",
                    extra={
                        "model_id": spec.model_id,
                        "source_url": source_url,
                        "error": error_message
                    },
                    exc_info=True
                )
                return None
        
        # Try each source in order
        attempted_sources = []
        for source_url in spec.sources:
            attempted_sources.append(source_url)
            metadata = attempt_download_from_source(source_url)
            if metadata is not None:
                return metadata
        
        # If we get here, all sources failed
        raise ModelOrchestratorError(
            f"Failed to download from all sources. Attempted: {', '.join(attempted_sources)}",
            ErrorCode.SOURCE_UNAVAILABLE,
            {"model_id": spec.model_id, "sources": spec.sources, "attempted_sources": attempted_sources}
        )
    
    def _verify_integrity(
        self, 
        spec: ModelSpec, 
        temp_dir: Path, 
        download_metadata: Optional[dict] = None,
        selective_files: Optional[List[FileSpec]] = None
    ) -> None:
        """Verify integrity of downloaded files with comprehensive verification."""
        files_to_verify = selective_files if selective_files is not None else spec.files
        logger.info(
            "Starting comprehensive integrity verification",
            extra={"model_id": spec.model_id, "file_count": len(files_to_verify)}
        )
        
        # Extract HF metadata if available from download result
        hf_metadata = None
        if download_metadata:
            hf_metadata = download_metadata.get('hf_metadata', {})
        
        # Create a temporary spec with selective files for verification
        if selective_files is not None:
            temp_spec = ModelSpec(
                model_id=spec.model_id,
                version=spec.version,
                variants=spec.variants,
                default_variant=spec.default_variant,
                files=selective_files,
                sources=spec.sources,
                allow_patterns=spec.allow_patterns,
                resolution_caps=spec.resolution_caps,
                optional_components=spec.optional_components,
                lora_required=spec.lora_required,
                description=spec.description,
                required_components=getattr(spec, 'required_components', []),
                defaults=getattr(spec, 'defaults', None),
                vram_estimation=getattr(spec, 'vram_estimation', None),
                model_type=getattr(spec, 'model_type', None)
            )
        else:
            temp_spec = spec
        
        # Perform comprehensive verification
        result = self.integrity_verifier.verify_model_integrity(
            spec=temp_spec,
            model_dir=temp_dir,
            hf_metadata=hf_metadata
        )
        
        if not result.success:
            # Log detailed failure information
            logger.error(
                "Integrity verification failed",
                extra={
                    "model_id": spec.model_id,
                    "failed_files": len(result.failed_files),
                    "missing_files": len(result.missing_files),
                    "error_message": result.error_message
                }
            )
            
            # Raise appropriate exceptions based on failure type
            if result.missing_files:
                raise IncompleteDownloadError(
                    f"Missing files after download: {', '.join(result.missing_files)}",
                    missing_files=result.missing_files
                )
            
            if result.failed_files:
                # Find the first checksum or size error to raise
                for failed_file in result.failed_files:
                    if "checksum" in failed_file.error_message.lower():
                        raise ChecksumError(
                            file_path=failed_file.file_path,
                            expected=failed_file.expected_value,
                            actual=failed_file.actual_value
                        )
                    elif "size" in failed_file.error_message.lower():
                        raise SizeMismatchError(
                            file_path=failed_file.file_path,
                            expected=int(failed_file.expected_value) if failed_file.expected_value else 0,
                            actual=int(failed_file.actual_value) if failed_file.actual_value else 0
                        )
                
                # Generic integrity error if we can't classify
                raise ModelOrchestratorError(
                    f"Integrity verification failed: {result.error_message}",
                    ErrorCode.CHECKSUM_FAIL,
                    {"model_id": spec.model_id, "failed_files": [f.file_path for f in result.failed_files]}
                )
        
        logger.info(
            "Comprehensive integrity verification completed successfully",
            extra={
                "model_id": spec.model_id,
                "verified_files": len(result.verified_files),
                "total_time": result.total_verification_time
            }
        )
    
    def _atomic_move(self, temp_dir: Path, target_dir: Path, model_id: str = None) -> None:
        """Atomically move from temporary to final location with error handling and deduplication."""
        logger.info(
            "Performing atomic move with deduplication",
            extra={
                "temp_dir": str(temp_dir), 
                "target_dir": str(target_dir),
                "deduplication_enabled": self.enable_deduplication and self.component_deduplicator is not None
            }
        )
        
        def perform_move() -> None:
            """Perform the actual move operation."""
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            
            if target_dir.exists():
                logger.debug("Removing existing target directory")
                shutil.rmtree(target_dir)
            
            shutil.move(str(temp_dir), str(target_dir))
            
            # Ensure data is written to disk
            if hasattr(os, 'sync'):
                os.sync()
        
        # Retry the move operation in case of transient filesystem issues
        retry_operation(
            perform_move,
            operation="atomic_move",
            config=self.error_recovery.config
        )
        
        # Perform deduplication after successful move
        if self.enable_deduplication and self.component_deduplicator is not None and model_id:
            try:
                logger.info(f"Starting component deduplication for model {model_id}")
                dedup_result = self.component_deduplicator.deduplicate_model(model_id, target_dir)
                
                logger.info(
                    "Component deduplication completed",
                    extra={
                        "model_id": model_id,
                        "files_processed": dedup_result.total_files_processed,
                        "duplicates_found": dedup_result.duplicates_found,
                        "bytes_saved": dedup_result.bytes_saved,
                        "links_created": dedup_result.links_created,
                        "processing_time": dedup_result.processing_time,
                        "errors": len(dedup_result.errors)
                    }
                )
                
                # Record deduplication metrics
                if hasattr(self.metrics, 'record_deduplication_completed'):
                    self.metrics.record_deduplication_completed(
                        model_id, dedup_result.bytes_saved, dedup_result.links_created
                    )
                
            except Exception as e:
                logger.warning(
                    f"Component deduplication failed for model {model_id}: {e}",
                    exc_info=True
                )
                # Don't fail the entire operation if deduplication fails
        
        logger.info(
            "Atomic move completed successfully",
            extra={"target_dir": str(target_dir)}
        )
    
    def _create_verification_marker(self, model_dir: Path, spec: ModelSpec) -> None:
        """Create a verification marker file for fast status checks."""
        verification_data = {
            "model_id": spec.model_id,
            "version": spec.version,
            "verified_at": time.time(),
            "files": [
                {
                    "path": f.path,
                    "size": f.size,
                    "sha256": f.sha256
                }
                for f in spec.files
            ]
        }
        
        def write_verification_file() -> None:
            """Write the verification marker file."""
            verification_file = model_dir / ".verified.json"
            with open(verification_file, 'w') as f:
                json.dump(verification_data, f, indent=2)
        
        # Retry verification file creation in case of transient issues
        retry_operation(
            write_verification_file,
            operation="create_verification_marker",
            config=self.error_recovery.config
        )
        
        logger.debug(
            "Created verification marker",
            extra={"model_id": spec.model_id, "model_dir": str(model_dir)}
        )
    
    def _cleanup_temp_directory(self, temp_dir: Path) -> None:
        """Clean up temporary directory with error handling."""
        try:
            if temp_dir.exists():
                logger.debug(
                    "Cleaning up temporary directory",
                    extra={"temp_dir": str(temp_dir)}
                )
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(
                "Failed to cleanup temp directory",
                extra={"temp_dir": str(temp_dir), "error": str(e)},
                exc_info=True
            )
    
    def _quick_completeness_check(self, spec: ModelSpec, model_dir: Path) -> bool:
        """Quick check that all files exist with correct sizes."""
        try:
            for file_spec in spec.files:
                file_path = model_dir / file_spec.path
                if not file_path.exists():
                    return False
                if file_path.stat().st_size != file_spec.size:
                    return False
            return True
        except Exception:
            return False
    
    def _calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _is_source_recently_failed(self, model_id: str, source_url: str) -> bool:
        """Check if a source has recently failed and is in negative cache."""
        cache_key = (model_id, source_url)
        failed_source = self._failed_sources.get(cache_key)
        
        if not failed_source:
            return False
        
        # Check if the failure is still within TTL
        if time.time() - failed_source.failed_at > self.negative_cache_ttl:
            # Remove expired entry
            del self._failed_sources[cache_key]
            return False
        
        return True
    
    def _mark_source_failed(self, model_id: str, source_url: str, error_message: str) -> None:
        """Mark a source as failed in the negative cache."""
        cache_key = (model_id, source_url)
        existing = self._failed_sources.get(cache_key)
        
        if existing:
            # Update existing entry
            existing.failed_at = time.time()
            existing.error_message = error_message
            existing.retry_count += 1
        else:
            # Create new entry
            self._failed_sources[cache_key] = FailedSource(
                url=source_url,
                failed_at=time.time(),
                error_message=error_message,
                retry_count=1
            )
    
    def _clear_source_failure(self, model_id: str, source_url: str) -> None:
        """Clear a source from the negative cache after successful download."""
        cache_key = (model_id, source_url)
        if cache_key in self._failed_sources:
            del self._failed_sources[cache_key]
    
    def _cleanup_expired_failures(self) -> None:
        """Clean up expired entries from the negative cache."""
        current_time = time.time()
        expired_keys = [
            key for key, failed_source in self._failed_sources.items()
            if current_time - failed_source.failed_at > self.negative_cache_ttl
        ]
        
        for key in expired_keys:
            del self._failed_sources[key]