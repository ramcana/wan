"""
Garbage Collector - Disk space management with LRU-based cleanup.
"""

import os
import json
import shutil
import time
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple
from collections import defaultdict

from .model_registry import ModelRegistry
from .model_resolver import ModelResolver
from .exceptions import ModelOrchestratorError, ErrorCode
from .logging_config import get_logger, LogContext, log_operation_start, log_operation_success, log_operation_failure

logger = get_logger(__name__)


class GCTrigger(Enum):
    """Reasons for triggering garbage collection."""
    MANUAL = "manual"
    QUOTA_EXCEEDED = "quota_exceeded"
    LOW_DISK_SPACE = "low_disk_space"
    SCHEDULED = "scheduled"


@dataclass
class ModelInfo:
    """Information about a model for garbage collection."""
    model_id: str
    variant: Optional[str]
    path: Path
    size_bytes: int
    last_accessed: float
    is_pinned: bool = False
    verification_time: Optional[float] = None


@dataclass
class GCConfig:
    """Configuration for garbage collection."""
    max_total_size: Optional[int] = None  # Maximum total size in bytes
    max_model_age: Optional[float] = None  # Maximum age in seconds
    low_disk_threshold: float = 0.1  # Trigger GC when disk usage > 90%
    safety_margin_bytes: int = 1024 * 1024 * 1024  # 1GB safety margin
    pin_recent_models: bool = True  # Auto-pin models accessed in last 24h
    recent_access_threshold: float = 24 * 3600  # 24 hours
    enable_auto_gc: bool = True  # Enable automatic garbage collection


@dataclass
class GCResult:
    """Result of garbage collection operation."""
    trigger: GCTrigger
    dry_run: bool
    models_removed: List[str] = field(default_factory=list)
    models_preserved: List[str] = field(default_factory=list)
    bytes_reclaimed: int = 0
    bytes_preserved: int = 0
    component_bytes_reclaimed: int = 0
    orphaned_components_cleaned: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class DiskUsage:
    """Disk usage information."""
    total_bytes: int
    used_bytes: int
    free_bytes: int
    models_bytes: int
    usage_percentage: float


class GarbageCollector:
    """Manages disk space through configurable retention policies."""
    
    def __init__(
        self,
        registry: ModelRegistry,
        resolver: ModelResolver,
        config: Optional[GCConfig] = None,
        component_deduplicator=None
    ):
        self.registry = registry
        self.resolver = resolver
        self.config = config or GCConfig()
        self.component_deduplicator = component_deduplicator
        
        # Pin file to track pinned models
        self.pin_file = Path(self.resolver.models_root) / ".gc" / "pinned_models.json"
        self.pin_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load pinned models
        self._pinned_models: Set[str] = self._load_pinned_models()
    
    def collect(self, dry_run: bool = False, trigger: GCTrigger = GCTrigger.MANUAL) -> GCResult:
        """Perform garbage collection with configurable policies."""
        with LogContext(
            operation="garbage_collection",
            dry_run=dry_run,
            trigger=trigger.value
        ):
            log_operation_start("garbage_collection", dry_run=dry_run, trigger=trigger.value)
            start_time = time.time()
            
            try:
                result = self._perform_collection(dry_run, trigger)
                duration = time.time() - start_time
                result.duration_seconds = duration
                
                log_operation_success(
                    "garbage_collection",
                    duration=duration,
                    bytes_reclaimed=result.bytes_reclaimed,
                    models_removed=len(result.models_removed),
                    dry_run=dry_run
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                log_operation_failure("garbage_collection", e, duration=duration, dry_run=dry_run)
                raise
    
    def pin_model(self, model_id: str, variant: Optional[str] = None) -> None:
        """Pin a model to protect it from garbage collection."""
        model_key = self._get_model_key(model_id, variant)
        
        logger.info(
            "Pinning model",
            extra={"model_id": model_id, "variant": variant, "model_key": model_key}
        )
        
        self._pinned_models.add(model_key)
        self._save_pinned_models()
    
    def unpin_model(self, model_id: str, variant: Optional[str] = None) -> None:
        """Unpin a model to allow garbage collection."""
        model_key = self._get_model_key(model_id, variant)
        
        logger.info(
            "Unpinning model",
            extra={"model_id": model_id, "variant": variant, "model_key": model_key}
        )
        
        self._pinned_models.discard(model_key)
        self._save_pinned_models()
    
    def is_pinned(self, model_id: str, variant: Optional[str] = None) -> bool:
        """Check if a model is pinned."""
        model_key = self._get_model_key(model_id, variant)
        return model_key in self._pinned_models
    
    def get_disk_usage(self) -> DiskUsage:
        """Get current disk usage information."""
        models_root = Path(self.resolver.models_root)
        
        # Get overall disk usage
        stat = shutil.disk_usage(models_root)
        total_bytes = stat.total
        free_bytes = stat.free
        used_bytes = total_bytes - free_bytes
        usage_percentage = (used_bytes / total_bytes) * 100
        
        # Calculate models directory usage
        models_bytes = self._calculate_directory_size(models_root)
        
        return DiskUsage(
            total_bytes=total_bytes,
            used_bytes=used_bytes,
            free_bytes=free_bytes,
            models_bytes=models_bytes,
            usage_percentage=usage_percentage
        )
    
    def estimate_reclaimable_space(self) -> int:
        """Estimate how much space could be reclaimed by garbage collection."""
        models = self._discover_models()
        candidates = self._select_removal_candidates(models, dry_run=True)
        return sum(model.size_bytes for model in candidates)
    
    def should_trigger_gc(self) -> Tuple[bool, Optional[GCTrigger]]:
        """Check if garbage collection should be triggered automatically."""
        if not self.config.enable_auto_gc:
            return False, None
        
        disk_usage = self.get_disk_usage()
        
        # Check disk space threshold
        if disk_usage.usage_percentage > (1.0 - self.config.low_disk_threshold) * 100:
            return True, GCTrigger.LOW_DISK_SPACE
        
        # Check total size quota
        if self.config.max_total_size and disk_usage.models_bytes > self.config.max_total_size:
            return True, GCTrigger.QUOTA_EXCEEDED
        
        return False, None
    
    def _perform_collection(self, dry_run: bool, trigger: GCTrigger) -> GCResult:
        """Perform the actual garbage collection."""
        result = GCResult(trigger=trigger, dry_run=dry_run)
        
        # Discover all models
        models = self._discover_models()
        
        logger.info(
            "Starting garbage collection",
            extra={
                "total_models": len(models),
                "dry_run": dry_run,
                "trigger": trigger.value
            }
        )
        
        # Auto-pin recently accessed models if configured
        if self.config.pin_recent_models:
            self._auto_pin_recent_models(models)
        
        # Select models for removal
        removal_candidates = self._select_removal_candidates(models, dry_run)
        
        # Remove selected models and clean up component references
        removed_model_ids = []
        for model in removal_candidates:
            try:
                model_key = self._get_model_key(model.model_id, model.variant)
                
                if not dry_run:
                    self._remove_model(model)
                    # Track removed model for component cleanup
                    removed_model_ids.append(model.model_id)
                
                result.models_removed.append(model_key)
                result.bytes_reclaimed += model.size_bytes
                
                logger.info(
                    "Model removed" if not dry_run else "Model would be removed",
                    extra={
                        "model_id": model.model_id,
                        "variant": model.variant,
                        "size_bytes": model.size_bytes,
                        "last_accessed": model.last_accessed,
                        "dry_run": dry_run
                    }
                )
                
            except Exception as e:
                error_msg = f"Failed to remove model {model.model_id}: {e}"
                result.errors.append(error_msg)
                logger.error(error_msg, exc_info=True)
        
        # Clean up orphaned components after model removal
        if self.component_deduplicator and removed_model_ids and not dry_run:
            try:
                total_component_bytes_reclaimed = 0
                total_orphaned_components = 0
                
                for model_id in removed_model_ids:
                    orphaned_components = self.component_deduplicator.remove_model_reference(model_id)
                    if orphaned_components:
                        component_bytes_reclaimed = self.component_deduplicator.cleanup_orphaned_components(orphaned_components)
                        total_component_bytes_reclaimed += component_bytes_reclaimed
                        total_orphaned_components += len(orphaned_components)
                
                result.component_bytes_reclaimed = total_component_bytes_reclaimed
                result.orphaned_components_cleaned = total_orphaned_components
                
                if total_orphaned_components > 0:
                    logger.info(
                        "Cleaned up orphaned components",
                        extra={
                            "orphaned_components": total_orphaned_components,
                            "component_bytes_reclaimed": total_component_bytes_reclaimed
                        }
                    )
                
            except Exception as e:
                error_msg = f"Failed to clean up orphaned components: {e}"
                result.errors.append(error_msg)
                logger.error(error_msg, exc_info=True)
        
        # Track preserved models
        for model in models:
            if model not in removal_candidates:
                result.models_preserved.append(self._get_model_key(model.model_id, model.variant))
                result.bytes_preserved += model.size_bytes
        
        logger.info(
            "Garbage collection completed",
            extra={
                "models_removed": len(result.models_removed),
                "models_preserved": len(result.models_preserved),
                "bytes_reclaimed": result.bytes_reclaimed,
                "bytes_preserved": result.bytes_preserved,
                "component_bytes_reclaimed": result.component_bytes_reclaimed,
                "orphaned_components_cleaned": result.orphaned_components_cleaned,
                "errors": len(result.errors),
                "dry_run": dry_run
            }
        )
        
        return result
    
    def _discover_models(self) -> List[ModelInfo]:
        """Discover all models in the models directory."""
        models = []
        models_root = Path(self.resolver.models_root)
        wan22_dir = models_root / "wan22"
        
        if not wan22_dir.exists():
            return models
        
        for model_dir in wan22_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            try:
                model_info = self._analyze_model_directory(model_dir)
                if model_info:
                    models.append(model_info)
            except Exception as e:
                logger.warning(
                    "Failed to analyze model directory",
                    extra={"model_dir": str(model_dir), "error": str(e)},
                    exc_info=True
                )
        
        return models
    
    def _analyze_model_directory(self, model_dir: Path) -> Optional[ModelInfo]:
        """Analyze a model directory to extract information."""
        # Parse model ID and variant from directory name
        dir_name = model_dir.name
        if "@" in dir_name:
            model_id, variant = dir_name.rsplit("@", 1)
        else:
            model_id = dir_name
            variant = None
        
        # Calculate directory size
        size_bytes = self._calculate_directory_size(model_dir)
        
        # Get last access time
        last_accessed = self._get_last_access_time(model_dir)
        
        # Check if pinned
        model_key = self._get_model_key(model_id, variant)
        is_pinned = model_key in self._pinned_models
        
        # Get verification time if available
        verification_time = self._get_verification_time(model_dir)
        
        return ModelInfo(
            model_id=model_id,
            variant=variant,
            path=model_dir,
            size_bytes=size_bytes,
            last_accessed=last_accessed,
            is_pinned=is_pinned,
            verification_time=verification_time
        )
    
    def _select_removal_candidates(self, models: List[ModelInfo], dry_run: bool) -> List[ModelInfo]:
        """Select models for removal based on configured policies."""
        candidates = []
        current_time = time.time()
        
        # Filter out pinned models
        unpinned_models = [m for m in models if not m.is_pinned]
        
        logger.debug(
            "Selecting removal candidates",
            extra={
                "total_models": len(models),
                "unpinned_models": len(unpinned_models),
                "pinned_models": len(models) - len(unpinned_models)
            }
        )
        
        # Apply age-based removal
        if self.config.max_model_age:
            age_threshold = current_time - self.config.max_model_age
            for model in unpinned_models:
                if model.last_accessed < age_threshold:
                    candidates.append(model)
                    logger.debug(
                        "Model selected for age-based removal",
                        extra={
                            "model_id": model.model_id,
                            "variant": model.variant,
                            "age_hours": (current_time - model.last_accessed) / 3600,
                            "max_age_hours": self.config.max_model_age / 3600
                        }
                    )
        
        # Apply size-based removal (LRU)
        if self.config.max_total_size:
            total_size = sum(m.size_bytes for m in models)
            if total_size > self.config.max_total_size:
                # Sort by last accessed time (oldest first)
                remaining_models = [m for m in unpinned_models if m not in candidates]
                remaining_models.sort(key=lambda x: x.last_accessed)
                
                bytes_to_remove = total_size - self.config.max_total_size
                bytes_removed = sum(m.size_bytes for m in candidates)
                
                for model in remaining_models:
                    if bytes_removed >= bytes_to_remove:
                        break
                    
                    candidates.append(model)
                    bytes_removed += model.size_bytes
                    
                    logger.debug(
                        "Model selected for size-based removal",
                        extra={
                            "model_id": model.model_id,
                            "variant": model.variant,
                            "size_bytes": model.size_bytes,
                            "bytes_removed": bytes_removed,
                            "bytes_to_remove": bytes_to_remove
                        }
                    )
        
        # Apply disk space-based removal
        disk_usage = self.get_disk_usage()
        if disk_usage.usage_percentage > (1.0 - self.config.low_disk_threshold) * 100:
            # Need to free up space
            target_free_bytes = disk_usage.total_bytes * self.config.low_disk_threshold + self.config.safety_margin_bytes
            current_free_bytes = disk_usage.free_bytes
            bytes_to_free = max(0, target_free_bytes - current_free_bytes)
            
            if bytes_to_free > 0:
                remaining_models = [m for m in unpinned_models if m not in candidates]
                remaining_models.sort(key=lambda x: x.last_accessed)
                
                bytes_freed = sum(m.size_bytes for m in candidates)
                
                for model in remaining_models:
                    if bytes_freed >= bytes_to_free:
                        break
                    
                    if model not in candidates:
                        candidates.append(model)
                        bytes_freed += model.size_bytes
                        
                        logger.debug(
                            "Model selected for disk space-based removal",
                            extra={
                                "model_id": model.model_id,
                                "variant": model.variant,
                                "size_bytes": model.size_bytes,
                                "bytes_freed": bytes_freed,
                                "bytes_to_free": bytes_to_free
                            }
                        )
        
        return candidates
    
    def _auto_pin_recent_models(self, models: List[ModelInfo]) -> None:
        """Automatically pin recently accessed models."""
        current_time = time.time()
        threshold = current_time - self.config.recent_access_threshold
        
        for model in models:
            if model.last_accessed > threshold and not model.is_pinned:
                model_key = self._get_model_key(model.model_id, model.variant)
                self._pinned_models.add(model_key)
                model.is_pinned = True
                
                logger.debug(
                    "Auto-pinned recently accessed model",
                    extra={
                        "model_id": model.model_id,
                        "variant": model.variant,
                        "hours_since_access": (current_time - model.last_accessed) / 3600
                    }
                )
        
        # Save updated pinned models
        self._save_pinned_models()
    
    def _remove_model(self, model: ModelInfo) -> None:
        """Remove a model from disk."""
        logger.info(
            "Removing model from disk",
            extra={
                "model_id": model.model_id,
                "variant": model.variant,
                "path": str(model.path),
                "size_bytes": model.size_bytes
            }
        )
        
        if model.path.exists():
            shutil.rmtree(model.path)
        
        # Also remove from pinned models if present
        model_key = self._get_model_key(model.model_id, model.variant)
        self._pinned_models.discard(model_key)
        self._save_pinned_models()
    
    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate the total size of a directory."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = Path(dirpath) / filename
                    try:
                        total_size += filepath.stat().st_size
                    except (OSError, IOError):
                        # Skip files that can't be accessed
                        pass
        except (OSError, IOError):
            # Skip directories that can't be accessed
            pass
        
        return total_size
    
    def _get_last_access_time(self, model_dir: Path) -> float:
        """Get the last access time for a model directory."""
        try:
            # Check for verification file first (most recent activity)
            verification_file = model_dir / ".verified.json"
            if verification_file.exists():
                return verification_file.stat().st_mtime
            
            # Fall back to directory modification time
            return model_dir.stat().st_mtime
        except (OSError, IOError):
            # Fall back to current time if we can't get stats
            return time.time()
    
    def _get_verification_time(self, model_dir: Path) -> Optional[float]:
        """Get the verification time from the verification file."""
        try:
            verification_file = model_dir / ".verified.json"
            if verification_file.exists():
                with open(verification_file, 'r') as f:
                    data = json.load(f)
                    return data.get('verified_at')
        except (OSError, IOError, json.JSONDecodeError):
            pass
        
        return None
    
    def _get_model_key(self, model_id: str, variant: Optional[str]) -> str:
        """Get a unique key for a model."""
        if variant and variant.strip():  # Check for non-empty variant
            return f"{model_id}@{variant}"
        return model_id
    
    def _load_pinned_models(self) -> Set[str]:
        """Load pinned models from disk."""
        try:
            if self.pin_file.exists():
                with open(self.pin_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('pinned_models', []))
        except (OSError, IOError, json.JSONDecodeError) as e:
            logger.warning(
                "Failed to load pinned models, starting with empty set",
                extra={"pin_file": str(self.pin_file), "error": str(e)}
            )
        
        return set()
    
    def _save_pinned_models(self) -> None:
        """Save pinned models to disk."""
        try:
            data = {
                'pinned_models': list(self._pinned_models),
                'updated_at': time.time()
            }
            
            with open(self.pin_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except (OSError, IOError) as e:
            logger.error(
                "Failed to save pinned models",
                extra={"pin_file": str(self.pin_file), "error": str(e)},
                exc_info=True
            )