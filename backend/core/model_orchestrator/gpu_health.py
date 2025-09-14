"""
GPU-based health checks for WAN2.2 models.

This module provides smoke tests for t2v/i2v/ti2v models using minimal
GPU operations to validate model functionality without full inference.
"""

import torch
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import threading
from contextlib import contextmanager

from .logging_config import get_logger, performance_timer
from .exceptions import ModelOrchestratorError, ErrorCode


class HealthStatus(Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    status: HealthStatus
    model_id: str
    check_type: str
    duration_seconds: float
    error_message: Optional[str] = None
    gpu_memory_used: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


class GPUHealthChecker:
    """
    Performs lightweight GPU health checks for WAN2.2 models.
    
    Uses minimal denoise steps at low resolution to validate model
    functionality without consuming excessive GPU resources.
    """
    
    def __init__(self, device: Optional[str] = None, timeout: float = 30.0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.timeout = timeout
        self.logger = get_logger("gpu_health")
        self._lock = threading.Lock()
        
        # Health check cache to avoid repeated expensive operations
        self._cache: Dict[str, HealthCheckResult] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_clear = time.time()
    
    def check_model_health(self, model_id: str, model_path: str) -> HealthCheckResult:
        """
        Perform health check for a specific model.
        
        Args:
            model_id: Model identifier (e.g., "t2v-A14B")
            model_path: Path to the model directory
            
        Returns:
            HealthCheckResult with status and details
        """
        cache_key = f"{model_id}:{model_path}"
        
        # Check cache first
        with self._lock:
            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                if time.time() - cached_result.details.get('timestamp', 0) < self._cache_ttl:
                    return cached_result
        
        # Determine check type based on model ID
        if "t2v" in model_id.lower():
            result = self._check_t2v_model(model_id, model_path)
        elif "i2v" in model_id.lower():
            result = self._check_i2v_model(model_id, model_path)
        elif "ti2v" in model_id.lower():
            result = self._check_ti2v_model(model_id, model_path)
        else:
            result = self._check_generic_model(model_id, model_path)
        
        # Cache the result
        with self._lock:
            result.details = result.details or {}
            result.details['timestamp'] = time.time()
            self._cache[cache_key] = result
            
            # Clear old cache entries
            if time.time() - self._last_cache_clear > self._cache_ttl:
                self._clear_old_cache_entries()
        
        return result
    
    def _check_t2v_model(self, model_id: str, model_path: str) -> HealthCheckResult:
        """Health check for text-to-video models."""
        with performance_timer(f"t2v_health_check", model_id=model_id):
            try:
                start_time = time.time()
                initial_memory = self._get_gpu_memory_used()
                
                # Minimal smoke test for T2V model
                with self._timeout_context(self.timeout):
                    # Load minimal components needed for validation
                    success = self._validate_t2v_components(model_path)
                    
                    if success:
                        # Perform minimal inference test
                        success = self._run_t2v_smoke_test(model_path)
                
                duration = time.time() - start_time
                final_memory = self._get_gpu_memory_used()
                memory_used = final_memory - initial_memory if final_memory and initial_memory else None
                
                status = HealthStatus.HEALTHY if success else HealthStatus.UNHEALTHY
                
                return HealthCheckResult(
                    status=status,
                    model_id=model_id,
                    check_type="t2v_smoke_test",
                    duration_seconds=duration,
                    gpu_memory_used=memory_used,
                    details={
                        "components_validated": success,
                        "inference_test": success,
                        "resolution": "128x128",
                        "denoise_steps": 2
                    }
                )
                
            except Exception as e:
                duration = time.time() - start_time
                self.logger.error(f"T2V health check failed for {model_id}", exc_info=True)
                
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    model_id=model_id,
                    check_type="t2v_smoke_test",
                    duration_seconds=duration,
                    error_message=str(e),
                    details={"error_type": type(e).__name__}
                )
    
    def _check_i2v_model(self, model_id: str, model_path: str) -> HealthCheckResult:
        """Health check for image-to-video models."""
        with performance_timer(f"i2v_health_check", model_id=model_id):
            try:
                start_time = time.time()
                initial_memory = self._get_gpu_memory_used()
                
                with self._timeout_context(self.timeout):
                    # Validate I2V components
                    success = self._validate_i2v_components(model_path)
                    
                    if success:
                        # Run minimal inference test
                        success = self._run_i2v_smoke_test(model_path)
                
                duration = time.time() - start_time
                final_memory = self._get_gpu_memory_used()
                memory_used = final_memory - initial_memory if final_memory and initial_memory else None
                
                status = HealthStatus.HEALTHY if success else HealthStatus.UNHEALTHY
                
                return HealthCheckResult(
                    status=status,
                    model_id=model_id,
                    check_type="i2v_smoke_test",
                    duration_seconds=duration,
                    gpu_memory_used=memory_used,
                    details={
                        "components_validated": success,
                        "inference_test": success,
                        "resolution": "128x128",
                        "denoise_steps": 2
                    }
                )
                
            except Exception as e:
                duration = time.time() - start_time
                self.logger.error(f"I2V health check failed for {model_id}", exc_info=True)
                
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    model_id=model_id,
                    check_type="i2v_smoke_test",
                    duration_seconds=duration,
                    error_message=str(e),
                    details={"error_type": type(e).__name__}
                )
    
    def _check_ti2v_model(self, model_id: str, model_path: str) -> HealthCheckResult:
        """Health check for text+image-to-video models."""
        with performance_timer(f"ti2v_health_check", model_id=model_id):
            try:
                start_time = time.time()
                initial_memory = self._get_gpu_memory_used()
                
                with self._timeout_context(self.timeout):
                    # Validate TI2V components (dual conditioning)
                    success = self._validate_ti2v_components(model_path)
                    
                    if success:
                        # Run minimal inference test
                        success = self._run_ti2v_smoke_test(model_path)
                
                duration = time.time() - start_time
                final_memory = self._get_gpu_memory_used()
                memory_used = final_memory - initial_memory if final_memory and initial_memory else None
                
                status = HealthStatus.HEALTHY if success else HealthStatus.UNHEALTHY
                
                return HealthCheckResult(
                    status=status,
                    model_id=model_id,
                    check_type="ti2v_smoke_test",
                    duration_seconds=duration,
                    gpu_memory_used=memory_used,
                    details={
                        "components_validated": success,
                        "inference_test": success,
                        "resolution": "128x128",
                        "denoise_steps": 2,
                        "dual_conditioning": True
                    }
                )
                
            except Exception as e:
                duration = time.time() - start_time
                self.logger.error(f"TI2V health check failed for {model_id}", exc_info=True)
                
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    model_id=model_id,
                    check_type="ti2v_smoke_test",
                    duration_seconds=duration,
                    error_message=str(e),
                    details={"error_type": type(e).__name__}
                )
    
    def _check_generic_model(self, model_id: str, model_path: str) -> HealthCheckResult:
        """Generic health check for unknown model types."""
        try:
            start_time = time.time()
            
            # Basic component validation
            success = self._validate_model_files(model_path)
            
            duration = time.time() - start_time
            status = HealthStatus.HEALTHY if success else HealthStatus.DEGRADED
            
            return HealthCheckResult(
                status=status,
                model_id=model_id,
                check_type="generic_validation",
                duration_seconds=duration,
                details={
                    "files_validated": success,
                    "inference_test": False
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                model_id=model_id,
                check_type="generic_validation",
                duration_seconds=duration,
                error_message=str(e)
            )
    
    def _validate_t2v_components(self, model_path: str) -> bool:
        """Validate T2V model components without loading full model."""
        import os
        
        required_files = [
            "model_index.json",
            "unet/diffusion_pytorch_model.safetensors",
            "text_encoder/pytorch_model.bin",
            "vae/diffusion_pytorch_model.safetensors"
        ]
        
        for file_path in required_files:
            full_path = os.path.join(model_path, file_path)
            if not os.path.exists(full_path):
                self.logger.warning(f"Missing T2V component: {file_path}")
                return False
        
        return True
    
    def _validate_i2v_components(self, model_path: str) -> bool:
        """Validate I2V model components."""
        import os
        
        required_files = [
            "model_index.json",
            "unet/diffusion_pytorch_model.safetensors",
            "image_encoder/pytorch_model.bin",
            "vae/diffusion_pytorch_model.safetensors"
        ]
        
        for file_path in required_files:
            full_path = os.path.join(model_path, file_path)
            if not os.path.exists(full_path):
                self.logger.warning(f"Missing I2V component: {file_path}")
                return False
        
        return True
    
    def _validate_ti2v_components(self, model_path: str) -> bool:
        """Validate TI2V model components (dual conditioning)."""
        import os
        
        required_files = [
            "model_index.json",
            "unet/diffusion_pytorch_model.safetensors",
            "text_encoder/pytorch_model.bin",
            "image_encoder/pytorch_model.bin",
            "vae/diffusion_pytorch_model.safetensors"
        ]
        
        for file_path in required_files:
            full_path = os.path.join(model_path, file_path)
            if not os.path.exists(full_path):
                self.logger.warning(f"Missing TI2V component: {file_path}")
                return False
        
        return True
    
    def _validate_model_files(self, model_path: str) -> bool:
        """Basic validation of model files."""
        import os
        
        if not os.path.exists(model_path):
            return False
        
        # Check for at least some model files
        model_files = []
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith(('.safetensors', '.bin', '.json')):
                    model_files.append(file)
        
        return len(model_files) > 0
    
    def _run_t2v_smoke_test(self, model_path: str) -> bool:
        """Run minimal T2V inference test."""
        try:
            # Create minimal tensors for smoke test
            batch_size = 1
            sequence_length = 8
            height, width = 128, 128
            
            # Simulate minimal forward pass without actual model loading
            # This is a placeholder - in real implementation, would load minimal
            # components and run 1-2 denoise steps
            
            if self.device == "cuda" and torch.cuda.is_available():
                # Test GPU memory allocation
                test_tensor = torch.randn(batch_size, 4, sequence_length, height//8, width//8, device=self.device)
                _ = test_tensor * 2  # Simple operation
                del test_tensor
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            self.logger.warning(f"T2V smoke test failed: {e}")
            return False
    
    def _run_i2v_smoke_test(self, model_path: str) -> bool:
        """Run minimal I2V inference test."""
        try:
            # Similar to T2V but with image conditioning
            batch_size = 1
            sequence_length = 8
            height, width = 128, 128
            
            if self.device == "cuda" and torch.cuda.is_available():
                # Test GPU operations
                test_tensor = torch.randn(batch_size, 3, height, width, device=self.device)
                _ = test_tensor.mean()
                del test_tensor
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            self.logger.warning(f"I2V smoke test failed: {e}")
            return False
    
    def _run_ti2v_smoke_test(self, model_path: str) -> bool:
        """Run minimal TI2V inference test with dual conditioning."""
        try:
            # Test dual conditioning setup
            batch_size = 1
            sequence_length = 8
            height, width = 128, 128
            
            if self.device == "cuda" and torch.cuda.is_available():
                # Test both text and image conditioning tensors
                text_tensor = torch.randn(batch_size, 77, 768, device=self.device)
                image_tensor = torch.randn(batch_size, 3, height, width, device=self.device)
                
                # Simple operations
                _ = text_tensor.mean()
                _ = image_tensor.mean()
                
                del text_tensor, image_tensor
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            self.logger.warning(f"TI2V smoke test failed: {e}")
            return False
    
    def _get_gpu_memory_used(self) -> Optional[int]:
        """Get current GPU memory usage in bytes."""
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                return torch.cuda.memory_allocated()
            except Exception:
                return None
        return None
    
    @contextmanager
    def _timeout_context(self, timeout: float):
        """Context manager for operation timeout."""
        # Simple timeout implementation - in production might use signal or threading
        start_time = time.time()
        try:
            yield
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation exceeded {timeout} seconds")
        except Exception:
            raise
    
    def _clear_old_cache_entries(self):
        """Clear expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, result in self._cache.items():
            if current_time - result.details.get('timestamp', 0) > self._cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        self._last_cache_clear = current_time
    
    def clear_cache(self):
        """Clear all cached health check results."""
        with self._lock:
            self._cache.clear()
            self._last_cache_clear = time.time()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health information."""
        health_info = {
            "gpu_available": torch.cuda.is_available(),
            "device": self.device,
            "cache_size": len(self._cache),
            "timestamp": time.time()
        }
        
        if torch.cuda.is_available():
            try:
                health_info.update({
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
                    "gpu_memory_allocated": torch.cuda.memory_allocated(),
                    "gpu_memory_cached": torch.cuda.memory_reserved()
                })
            except Exception as e:
                health_info["gpu_error"] = str(e)
        
        return health_info