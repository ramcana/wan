"""
Model Health Service - Enhanced health monitoring with observability features.

Provides comprehensive model health checking including GPU validation,
performance metrics, and detailed diagnostics with structured logging.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from backend.core.model_orchestrator import (
    ModelRegistry, ModelResolver, ModelEnsurer, ModelStatus, ModelStatusInfo
)
from backend.core.model_orchestrator.exceptions import ModelOrchestratorError
from backend.core.model_orchestrator.gpu_health import GPUHealthChecker, HealthStatus as GPUHealthStatus
from backend.core.model_orchestrator.metrics import get_metrics_collector
from backend.core.model_orchestrator.logging_config import (
    get_logger, performance_timer, log_with_context, set_correlation_id, generate_correlation_id
)

logger = get_logger("health_service")


@dataclass
class ModelHealthInfo:
    """Enhanced health information for a single model."""
    model_id: str
    variant: Optional[str]
    status: str  # NOT_PRESENT, PARTIAL, COMPLETE, CORRUPT
    local_path: Optional[str]
    missing_files: List[str]
    bytes_needed: int
    last_verified: Optional[float] = None
    error_message: Optional[str] = None
    gpu_health: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    component_details: Optional[Dict[str, Any]] = None


@dataclass
class OrchestratorHealthResponse:
    """Enhanced orchestrator health response with observability metrics."""
    status: str  # healthy, degraded, error
    timestamp: float
    models: Dict[str, ModelHealthInfo]
    total_models: int
    healthy_models: int
    missing_models: int
    partial_models: int
    corrupt_models: int
    total_bytes_needed: int
    response_time_ms: float
    correlation_id: str
    system_metrics: Optional[Dict[str, Any]] = None
    gpu_system_health: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class ModelHealthService:
    """Enhanced service for providing comprehensive model health monitoring."""
    
    def __init__(
        self,
        registry: ModelRegistry,
        resolver: ModelResolver,
        ensurer: ModelEnsurer,
        timeout_ms: float = 100.0,
        enable_gpu_checks: bool = True,
        enable_detailed_diagnostics: bool = False
    ):
        self.registry = registry
        self.resolver = resolver
        self.ensurer = ensurer
        self.timeout_ms = timeout_ms
        self.enable_gpu_checks = enable_gpu_checks
        self.enable_detailed_diagnostics = enable_detailed_diagnostics
        
        # Initialize observability components
        self.metrics = get_metrics_collector()
        self.gpu_checker = GPUHealthChecker() if enable_gpu_checks else None
        
        # Performance tracking
        self._health_check_count = 0
        self._total_response_time = 0.0
    
    async def get_health_status(
        self, 
        dry_run: bool = True,
        include_gpu_checks: bool = None,
        include_detailed_diagnostics: bool = None
    ) -> OrchestratorHealthResponse:
        """
        Get comprehensive health status of all models in the orchestrator.
        
        Args:
            dry_run: If True, only check status without triggering downloads
            include_gpu_checks: Override GPU check setting for this call
            include_detailed_diagnostics: Include detailed component diagnostics
            
        Returns:
            Enhanced OrchestratorHealthResponse with observability metrics
        """
        # Set correlation ID for this health check
        correlation_id = generate_correlation_id()
        set_correlation_id(correlation_id)
        
        with performance_timer("orchestrator_health_check", dry_run=dry_run):
            start_time = time.time()
            
            try:
                # Get all models from registry
                model_ids = self.registry.list_models()
                
                log_with_context(
                    logger, logging.INFO,
                    f"Starting health check for {len(model_ids)} models",
                    model_count=len(model_ids),
                    dry_run=dry_run,
                    gpu_checks_enabled=include_gpu_checks or self.enable_gpu_checks
                )
                
                models_health = {}
                healthy_count = 0
                missing_count = 0
                partial_count = 0
                corrupt_count = 0
                total_bytes_needed = 0
                
                for model_id in model_ids:
                    try:
                        # Check if we're approaching timeout
                        elapsed_ms = (time.time() - start_time) * 1000
                        if elapsed_ms > self.timeout_ms * 0.8:  # 80% of timeout
                            logger.warning(f"Approaching timeout, skipping remaining models")
                            break
                        
                        model_health = await self._get_model_health(
                            model_id, dry_run, include_gpu_checks, include_detailed_diagnostics
                        )
                        models_health[model_id] = model_health
                        
                        # Update counters
                        if model_health.status == ModelStatus.COMPLETE.value:
                            healthy_count += 1
                        elif model_health.status == ModelStatus.NOT_PRESENT.value:
                            missing_count += 1
                        elif model_health.status == ModelStatus.PARTIAL.value:
                            partial_count += 1
                        elif model_health.status == ModelStatus.CORRUPT.value:
                            corrupt_count += 1
                        # ERROR status counts as corrupt for overall health calculation
                        
                        total_bytes_needed += model_health.bytes_needed
                        
                    except Exception as e:
                        logger.error(f"Error checking health for model {model_id}: {e}")
                        models_health[model_id] = ModelHealthInfo(
                            model_id=model_id,
                            variant=None,
                            status="ERROR",
                            local_path=None,
                            missing_files=[],
                            bytes_needed=0,
                            error_message=str(e)
                        )
                
                # Determine overall status
                total_models = len(models_health)
                error_count = sum(1 for health in models_health.values() if health.status == "ERROR")
                
                if total_models == 0:
                    overall_status = "error"
                elif healthy_count == total_models:
                    overall_status = "healthy"
                elif missing_count > 0 or partial_count > 0 or corrupt_count > 0 or error_count > 0:
                    overall_status = "degraded"
                else:
                    overall_status = "healthy"
                
                response_time_ms = (time.time() - start_time) * 1000
                
                # Collect system metrics
                system_metrics = {
                    "total_health_checks": self._health_check_count,
                    "average_response_time_ms": (
                        self._total_response_time / self._health_check_count 
                        if self._health_check_count > 0 else 0
                    ),
                    "models_processed": len(models_health),
                    "timeout_ms": self.timeout_ms
                }
                
                # Get GPU system health if available
                gpu_system_health = None
                if self.gpu_checker:
                    try:
                        gpu_system_health = self.gpu_checker.get_system_health()
                    except Exception as e:
                        logger.warning(f"Failed to get GPU system health: {e}")
                
                # Update performance tracking
                self._health_check_count += 1
                self._total_response_time += response_time_ms
                
                # Record metrics
                self.metrics.record_storage_usage("all_models", total_bytes_needed, total_models)
                
                log_with_context(
                    logger, logging.INFO,
                    f"Health check completed",
                    overall_status=overall_status,
                    total_models=total_models,
                    healthy_models=healthy_count,
                    response_time_ms=response_time_ms
                )
                
                return OrchestratorHealthResponse(
                    status=overall_status,
                    timestamp=time.time(),
                    models=models_health,
                    total_models=total_models,
                    healthy_models=healthy_count,
                    missing_models=missing_count,
                    partial_models=partial_count,
                    corrupt_models=corrupt_count,
                    total_bytes_needed=total_bytes_needed,
                    response_time_ms=response_time_ms,
                    correlation_id=correlation_id,
                    system_metrics=system_metrics,
                    gpu_system_health=gpu_system_health
                )
                
            except Exception as e:
                response_time_ms = (time.time() - start_time) * 1000
                logger.error(f"Error getting orchestrator health status: {e}", exc_info=True)
                
                return OrchestratorHealthResponse(
                    status="error",
                    timestamp=time.time(),
                    models={},
                    total_models=0,
                    healthy_models=0,
                    missing_models=0,
                    partial_models=0,
                    corrupt_models=0,
                    total_bytes_needed=0,
                    response_time_ms=response_time_ms,
                    correlation_id=correlation_id,
                    error_message=str(e)
                )
    
    async def _get_model_health(
        self, 
        model_id: str, 
        dry_run: bool = True,
        include_gpu_check: bool = None,
        include_detailed_diagnostics: bool = None
    ) -> ModelHealthInfo:
        """Get comprehensive health information for a single model."""
        with performance_timer("model_health_check", model_id=model_id):
            try:
                # Use the ensurer's status method for consistent results
                status_info = self.ensurer.status(model_id)
                
                # Get last verification time from .verified.json if available
                last_verified = None
                try:
                    local_path = Path(self.resolver.local_dir(model_id))
                    verification_file = local_path / ".verified.json"
                    
                    if verification_file.exists():
                        with open(verification_file, 'r') as f:
                            verification_data = json.load(f)
                        last_verified = verification_data.get('verified_at')
                except Exception as e:
                    logger.debug(f"Could not read verification file for {model_id}: {e}")
                
                # GPU health check if enabled and model is available
                gpu_health = None
                if ((include_gpu_check if include_gpu_check is not None else self.enable_gpu_checks) 
                    and self.gpu_checker 
                    and status_info.status == ModelStatus.COMPLETE):
                    try:
                        model_path = self.resolver.local_dir(model_id)
                        gpu_result = self.gpu_checker.check_model_health(model_id, model_path)
                        gpu_health = asdict(gpu_result)
                    except Exception as e:
                        logger.warning(f"GPU health check failed for {model_id}: {e}")
                        gpu_health = {"error": str(e), "status": "unknown"}
                
                # Detailed component diagnostics if enabled
                component_details = None
                if (include_detailed_diagnostics if include_detailed_diagnostics is not None 
                    else self.enable_detailed_diagnostics):
                    component_details = await self._get_component_details(model_id)
                
                # Performance metrics
                performance_metrics = {
                    "status_check_duration": 0.0,  # Would be measured in real implementation
                    "last_health_check": time.time()
                }
                
                return ModelHealthInfo(
                    model_id=model_id,
                    variant=None,  # Default variant
                    status=status_info.status.value,
                    local_path=status_info.local_path,
                    missing_files=status_info.missing_files or [],
                    bytes_needed=status_info.bytes_needed,
                    last_verified=last_verified,
                    gpu_health=gpu_health,
                    performance_metrics=performance_metrics,
                    component_details=component_details
                )
                
            except Exception as e:
                logger.error(f"Error getting health for model {model_id}: {e}", exc_info=True)
                return ModelHealthInfo(
                    model_id=model_id,
                    variant=None,
                    status="ERROR",
                    local_path=None,
                    missing_files=[],
                    bytes_needed=0,
                    error_message=str(e)
                )
    
    async def get_model_health(
        self, 
        model_id: str, 
        variant: Optional[str] = None,
        include_gpu_check: bool = None,
        include_detailed_diagnostics: bool = None
    ) -> ModelHealthInfo:
        """Get comprehensive health information for a specific model."""
        correlation_id = generate_correlation_id()
        set_correlation_id(correlation_id)
        
        with performance_timer("single_model_health_check", model_id=model_id, variant=variant):
            try:
                status_info = self.ensurer.status(model_id, variant)
                
                # Get last verification time
                last_verified = None
                try:
                    local_path = Path(self.resolver.local_dir(model_id, variant))
                    verification_file = local_path / ".verified.json"
                    
                    if verification_file.exists():
                        with open(verification_file, 'r') as f:
                            verification_data = json.load(f)
                        last_verified = verification_data.get('verified_at')
                except Exception as e:
                    logger.debug(f"Could not read verification file for {model_id}: {e}")
                
                # GPU health check if enabled and model is available
                gpu_health = None
                if ((include_gpu_check if include_gpu_check is not None else self.enable_gpu_checks) 
                    and self.gpu_checker 
                    and status_info.status == ModelStatus.COMPLETE):
                    try:
                        model_path = self.resolver.local_dir(model_id, variant)
                        gpu_result = self.gpu_checker.check_model_health(model_id, model_path)
                        gpu_health = asdict(gpu_result)
                    except Exception as e:
                        logger.warning(f"GPU health check failed for {model_id}: {e}")
                        gpu_health = {"error": str(e), "status": "unknown"}
                
                # Detailed component diagnostics if enabled
                component_details = None
                if (include_detailed_diagnostics if include_detailed_diagnostics is not None 
                    else self.enable_detailed_diagnostics):
                    component_details = await self._get_component_details(model_id, variant)
                
                # Performance metrics
                performance_metrics = {
                    "status_check_duration": 0.0,
                    "last_health_check": time.time(),
                    "correlation_id": correlation_id
                }
                
                return ModelHealthInfo(
                    model_id=model_id,
                    variant=variant,
                    status=status_info.status.value,
                    local_path=status_info.local_path,
                    missing_files=status_info.missing_files or [],
                    bytes_needed=status_info.bytes_needed,
                    last_verified=last_verified,
                    gpu_health=gpu_health,
                    performance_metrics=performance_metrics,
                    component_details=component_details
                )
                
            except Exception as e:
                logger.error(f"Error getting health for model {model_id}: {e}", exc_info=True)
                return ModelHealthInfo(
                    model_id=model_id,
                    variant=variant,
                    status="ERROR",
                    local_path=None,
                    missing_files=[],
                    bytes_needed=0,
                    error_message=str(e)
                )
    
    async def _get_component_details(
        self, 
        model_id: str, 
        variant: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get detailed component information for a model."""
        try:
            spec = self.registry.spec(model_id, variant)
            local_path = Path(self.resolver.local_dir(model_id, variant))
            
            component_info = {
                "expected_files": len(spec.files),
                "expected_size": sum(f.size for f in spec.files),
                "file_details": []
            }
            
            if local_path.exists():
                for file_spec in spec.files:
                    file_path = local_path / file_spec.path
                    file_info = {
                        "path": file_spec.path,
                        "expected_size": file_spec.size,
                        "exists": file_path.exists()
                    }
                    
                    if file_path.exists():
                        stat = file_path.stat()
                        file_info.update({
                            "actual_size": stat.st_size,
                            "size_match": stat.st_size == file_spec.size,
                            "modified_time": stat.st_mtime
                        })
                    
                    component_info["file_details"].append(file_info)
            
            return component_info
            
        except Exception as e:
            logger.warning(f"Failed to get component details for {model_id}: {e}")
            return {"error": str(e)}
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance and health metrics summary."""
        return {
            "health_service": {
                "total_checks": self._health_check_count,
                "average_response_time_ms": (
                    self._total_response_time / self._health_check_count 
                    if self._health_check_count > 0 else 0
                ),
                "timeout_ms": self.timeout_ms,
                "gpu_checks_enabled": self.enable_gpu_checks,
                "detailed_diagnostics_enabled": self.enable_detailed_diagnostics
            },
            "orchestrator_metrics": self.metrics.get_metrics_dict(),
            "gpu_system": (
                self.gpu_checker.get_system_health() 
                if self.gpu_checker else None
            )
        }
    
    def to_dict(self, response: OrchestratorHealthResponse) -> Dict[str, Any]:
        """Convert health response to dictionary for JSON serialization."""
        result = asdict(response)
        
        # Convert ModelHealthInfo objects to dictionaries
        models_dict = {}
        for model_id, health_info in response.models.items():
            models_dict[model_id] = asdict(health_info)
        
        result['models'] = models_dict
        return result


# Global service instance (will be initialized by the application)
_health_service: Optional[ModelHealthService] = None


def get_model_health_service() -> Optional[ModelHealthService]:
    """Get the global model health service instance."""
    return _health_service


def initialize_model_health_service(
    registry: ModelRegistry,
    resolver: ModelResolver,
    ensurer: ModelEnsurer,
    timeout_ms: float = 100.0
) -> ModelHealthService:
    """Initialize the global model health service."""
    global _health_service
    _health_service = ModelHealthService(registry, resolver, ensurer, timeout_ms)
    return _health_service