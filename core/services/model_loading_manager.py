#!/usr/bin/env python3
"""
Model Loading Manager for WAN22 System Optimization

This module provides comprehensive model loading optimization with detailed progress tracking,
parameter caching, and intelligent error handling for large models like TI2V-5B.

Requirements addressed:
- 7.1: Detailed progress tracking for large model loading
- 7.2: Specific error messages and suggested solutions for loading issues  
- 7.3: Loading parameter caching for faster subsequent loads
"""

import json
import logging
import os
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
import hashlib

try:
    import torch
    import psutil
    from diffusers import DiffusionPipeline
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


class ModelLoadingPhase(Enum):
    """Phases of model loading process"""
    INITIALIZATION = "initialization"
    VALIDATION = "validation"
    CACHE_CHECK = "cache_check"
    DOWNLOAD = "download"
    LOADING = "loading"
    OPTIMIZATION = "optimization"
    FINALIZATION = "finalization"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelLoadingError(Exception):
    """Custom exception for model loading errors"""
    def __init__(self, message: str, error_code: str = None, suggestions: List[str] = None):
        super().__init__(message)
        self.error_code = error_code or "UNKNOWN"
        self.suggestions = suggestions or []


@dataclass
class ModelLoadingProgress:
    """Progress information for model loading"""
    phase: ModelLoadingPhase
    progress_percent: float
    current_step: str
    estimated_time_remaining: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    download_speed_mbps: Optional[float] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class LoadingParameters:
    """Parameters used for model loading"""
    model_path: str
    torch_dtype: str
    device_map: Optional[str] = None
    low_cpu_mem_usage: bool = True
    trust_remote_code: bool = False
    variant: Optional[str] = None
    use_safetensors: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    custom_pipeline: Optional[str] = None
    
    def get_cache_key(self) -> str:
        """Generate cache key for these parameters"""
        # Create a hash of the parameters for caching
        param_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()


@dataclass
class ModelLoadingResult:
    """Result of model loading operation"""
    success: bool
    model: Optional[Any] = None
    loading_time: float = 0.0
    memory_usage_mb: float = 0.0
    cache_hit: bool = False
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    suggestions: List[str] = None
    parameters_used: Optional[LoadingParameters] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class ModelLoadingManager:
    """
    Comprehensive model loading manager with progress tracking, caching, and error handling.
    
    Features:
    - Detailed progress tracking with time estimates
    - Parameter caching for faster subsequent loads
    - Specific error messages with suggested solutions
    - Memory usage monitoring
    - Hardware-aware optimization
    """
    
    def __init__(self, cache_dir: str = "model_cache", enable_logging: bool = True):
        """
        Initialize the ModelLoadingManager.
        
        Args:
            cache_dir: Directory for caching model parameters and metadata
            enable_logging: Whether to enable detailed logging
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if enable_logging:
            self._setup_logging()
        
        # Progress tracking
        self._progress_callbacks: List[Callable[[ModelLoadingProgress], None]] = []
        self._current_progress = ModelLoadingProgress(
            phase=ModelLoadingPhase.INITIALIZATION,
            progress_percent=0.0,
            current_step="Initializing"
        )
        
        # Cache management
        self._parameter_cache: Dict[str, Dict[str, Any]] = {}
        self._load_parameter_cache()
        
        # Error handling
        self._error_solutions = self._initialize_error_solutions()
        
        # Performance tracking
        self._loading_stats: Dict[str, List[float]] = {}
        
        self.logger.info("ModelLoadingManager initialized")
    
    def _setup_logging(self):
        """Setup detailed logging for model loading operations"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _initialize_error_solutions(self) -> Dict[str, List[str]]:
        """Initialize common error solutions"""
        return {
            "CUDA_OUT_OF_MEMORY": [
                "Reduce model precision (use torch.float16 or torch.bfloat16)",
                "Enable CPU offloading with device_map='auto'",
                "Use gradient checkpointing to reduce memory usage",
                "Close other GPU-intensive applications",
                "Try loading with load_in_8bit=True for quantization"
            ],
            "MODEL_NOT_FOUND": [
                "Verify the model path or repository name is correct",
                "Check internet connection for remote models",
                "Ensure you have access to private repositories",
                "Try using the full repository path (e.g., 'username/model-name')",
                "Check if the model requires authentication"
            ],
            "TRUST_REMOTE_CODE_ERROR": [
                "Set trust_remote_code=True in loading parameters",
                "Verify the model source is trustworthy",
                "Check model documentation for required parameters",
                "Consider using a local copy of the model"
            ],
            "INSUFFICIENT_DISK_SPACE": [
                "Free up disk space (models can be 10GB+)",
                "Change cache directory to a drive with more space",
                "Clear old model cache files",
                "Use symbolic links to store models on different drives"
            ],
            "NETWORK_ERROR": [
                "Check internet connection stability",
                "Try again later (server may be temporarily unavailable)",
                "Use a VPN if accessing from restricted regions",
                "Download model manually and load from local path"
            ],
            "INCOMPATIBLE_HARDWARE": [
                "Check CUDA compatibility with your GPU",
                "Update GPU drivers to latest version",
                "Verify PyTorch CUDA version matches your setup",
                "Consider using CPU-only mode for testing"
            ]
        }
    
    def add_progress_callback(self, callback: Callable[[ModelLoadingProgress], None]):
        """Add a callback function to receive progress updates"""
        self._progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable[[ModelLoadingProgress], None]):
        """Remove a progress callback"""
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)
    
    def _update_progress(self, phase: ModelLoadingPhase, progress_percent: float, 
                        current_step: str, **kwargs):
        """Update loading progress and notify callbacks"""
        self._current_progress = ModelLoadingProgress(
            phase=phase,
            progress_percent=progress_percent,
            current_step=current_step,
            **kwargs
        )
        
        # Notify all callbacks
        for callback in self._progress_callbacks:
            try:
                callback(self._current_progress)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")
        
        self.logger.info(f"Loading progress: {phase.value} - {progress_percent:.1f}% - {current_step}")
    
    def _load_parameter_cache(self):
        """Load cached parameters from disk"""
        cache_file = self.cache_dir / "parameter_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self._parameter_cache = json.load(f)
                self.logger.info(f"Loaded {len(self._parameter_cache)} cached parameter sets")
            except Exception as e:
                self.logger.warning(f"Failed to load parameter cache: {e}")
                self._parameter_cache = {}
    
    def _save_parameter_cache(self):
        """Save parameter cache to disk"""
        cache_file = self.cache_dir / "parameter_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self._parameter_cache, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save parameter cache: {e}")
    
    def _get_cached_parameters(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached parameters if available"""
        return self._parameter_cache.get(cache_key)
    
    def _cache_parameters(self, cache_key: str, parameters: LoadingParameters, 
                         loading_time: float, memory_usage: float):
        """Cache successful loading parameters"""
        self._parameter_cache[cache_key] = {
            "parameters": asdict(parameters),
            "loading_time": loading_time,
            "memory_usage_mb": memory_usage,
            "last_used": datetime.now().isoformat(),
            "use_count": self._parameter_cache.get(cache_key, {}).get("use_count", 0) + 1
        }
        self._save_parameter_cache()
    
    def _estimate_loading_time(self, model_path: str, parameters: LoadingParameters) -> Optional[float]:
        """Estimate loading time based on historical data"""
        cache_key = parameters.get_cache_key()
        cached = self._get_cached_parameters(cache_key)
        
        if cached:
            return cached.get("loading_time", None)
        
        # Fallback estimates based on model type
        if "5B" in model_path or "5b" in model_path:
            return 300.0  # 5 minutes for large models
        elif "1B" in model_path or "1b" in model_path:
            return 120.0  # 2 minutes for medium models
        else:
            return 60.0   # 1 minute for smaller models
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if not DEPENDENCIES_AVAILABLE:
            return 0.0
        
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _validate_model_path(self, model_path: str) -> bool:
        """Validate that model path exists or is a valid repository"""
        if os.path.exists(model_path):
            return True
        
        # Check if it looks like a HuggingFace repository
        if "/" in model_path and not model_path.startswith("/"):
            return True  # Assume it's a valid repo path
        
        return False
    
    def _handle_loading_error(self, error: Exception, parameters: LoadingParameters) -> ModelLoadingResult:
        """Handle and categorize loading errors with suggestions"""
        error_message = str(error)
        error_code = "UNKNOWN"
        suggestions = []
        
        # Categorize common errors
        if "CUDA out of memory" in error_message or "OutOfMemoryError" in error_message:
            error_code = "CUDA_OUT_OF_MEMORY"
        elif "not found" in error_message.lower() or "does not exist" in error_message.lower():
            error_code = "MODEL_NOT_FOUND"
        elif "trust_remote_code" in error_message.lower():
            error_code = "TRUST_REMOTE_CODE_ERROR"
        elif "No space left" in error_message or "disk full" in error_message.lower():
            error_code = "INSUFFICIENT_DISK_SPACE"
        elif "connection" in error_message.lower() or "network" in error_message.lower():
            error_code = "NETWORK_ERROR"
        elif "CUDA" in error_message and "not available" in error_message:
            error_code = "INCOMPATIBLE_HARDWARE"
        
        suggestions = self._error_solutions.get(error_code, [
            "Check the error message for specific details",
            "Verify your system meets the model requirements",
            "Try loading a smaller model first to test your setup",
            "Check the model documentation for specific requirements"
        ])
        
        self.logger.error(f"Model loading failed: {error_code} - {error_message}")
        
        return ModelLoadingResult(
            success=False,
            error_message=error_message,
            error_code=error_code,
            suggestions=suggestions,
            parameters_used=parameters
        )
    
    def load_model(self, model_path: str, **kwargs) -> ModelLoadingResult:
        """
        Load a model with comprehensive progress tracking and error handling.
        
        Args:
            model_path: Path to model or HuggingFace repository
            **kwargs: Additional loading parameters
            
        Returns:
            ModelLoadingResult with success status and details
        """
        if not DEPENDENCIES_AVAILABLE:
            return ModelLoadingResult(
                success=False,
                error_message="Required dependencies not available (torch, diffusers, psutil)",
                error_code="MISSING_DEPENDENCIES",
                suggestions=["Install required packages: pip install torch diffusers psutil"]
            )
        
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        # Create loading parameters
        parameters = LoadingParameters(
            model_path=model_path,
            torch_dtype=kwargs.get('torch_dtype', 'auto'),
            device_map=kwargs.get('device_map'),
            low_cpu_mem_usage=kwargs.get('low_cpu_mem_usage', True),
            trust_remote_code=kwargs.get('trust_remote_code', False),
            variant=kwargs.get('variant'),
            use_safetensors=kwargs.get('use_safetensors', True),
            load_in_8bit=kwargs.get('load_in_8bit', False),
            load_in_4bit=kwargs.get('load_in_4bit', False),
            custom_pipeline=kwargs.get('custom_pipeline')
        )
        
        try:
            # Phase 1: Initialization
            self._update_progress(
                ModelLoadingPhase.INITIALIZATION, 
                5.0, 
                "Initializing model loading",
                estimated_time_remaining=self._estimate_loading_time(model_path, parameters)
            )
            
            # Phase 2: Validation
            self._update_progress(ModelLoadingPhase.VALIDATION, 10.0, "Validating model path")
            if not self._validate_model_path(model_path):
                raise ModelLoadingError(
                    f"Model path not found: {model_path}",
                    "MODEL_NOT_FOUND"
                )
            
            # Phase 3: Cache check
            self._update_progress(ModelLoadingPhase.CACHE_CHECK, 15.0, "Checking parameter cache")
            cache_key = parameters.get_cache_key()
            cached_params = self._get_cached_parameters(cache_key)
            cache_hit = cached_params is not None
            
            if cache_hit:
                self.logger.info("Using cached loading parameters")
                # Update parameters with cached optimizations if available
                cached_param_data = cached_params["parameters"]
                for key, value in cached_param_data.items():
                    if hasattr(parameters, key) and getattr(parameters, key) is None:
                        setattr(parameters, key, value)
            
            # Phase 4: Model loading
            self._update_progress(ModelLoadingPhase.LOADING, 25.0, "Loading model components")
            
            # Prepare loading arguments
            load_kwargs = {
                'torch_dtype': getattr(torch, parameters.torch_dtype) if hasattr(torch, parameters.torch_dtype) else torch.float16,
                'low_cpu_mem_usage': parameters.low_cpu_mem_usage,
                'trust_remote_code': parameters.trust_remote_code,
            }
            
            if parameters.device_map:
                load_kwargs['device_map'] = parameters.device_map
            if parameters.variant:
                load_kwargs['variant'] = parameters.variant
            if parameters.use_safetensors:
                load_kwargs['use_safetensors'] = parameters.use_safetensors
            if parameters.load_in_8bit:
                load_kwargs['load_in_8bit'] = parameters.load_in_8bit
            if parameters.load_in_4bit:
                load_kwargs['load_in_4bit'] = parameters.load_in_4bit
            
            # Load the model
            self._update_progress(ModelLoadingPhase.LOADING, 50.0, "Loading pipeline from pretrained")
            
            if parameters.custom_pipeline:
                model = DiffusionPipeline.from_pretrained(
                    model_path,
                    custom_pipeline=parameters.custom_pipeline,
                    **load_kwargs
                )
            else:
                model = DiffusionPipeline.from_pretrained(model_path, **load_kwargs)
            
            # Phase 5: Optimization
            self._update_progress(ModelLoadingPhase.OPTIMIZATION, 80.0, "Applying optimizations")
            
            # Apply memory optimizations if needed
            if hasattr(model, 'enable_model_cpu_offload'):
                model.enable_model_cpu_offload()
            
            if hasattr(model, 'enable_attention_slicing'):
                model.enable_attention_slicing()
            
            # Phase 6: Finalization
            self._update_progress(ModelLoadingPhase.FINALIZATION, 95.0, "Finalizing model setup")
            
            loading_time = time.time() - start_time
            final_memory = self._get_memory_usage()
            memory_usage = final_memory - initial_memory
            
            # Cache successful parameters
            self._cache_parameters(cache_key, parameters, loading_time, memory_usage)
            
            # Phase 7: Completed
            self._update_progress(ModelLoadingPhase.COMPLETED, 100.0, "Model loading completed")
            
            self.logger.info(f"Model loaded successfully in {loading_time:.2f}s, memory usage: {memory_usage:.1f}MB")
            
            return ModelLoadingResult(
                success=True,
                model=model,
                loading_time=loading_time,
                memory_usage_mb=memory_usage,
                cache_hit=cache_hit,
                parameters_used=parameters
            )
            
        except Exception as e:
            self._update_progress(
                ModelLoadingPhase.FAILED, 
                0.0, 
                f"Loading failed: {str(e)[:50]}...",
                error_message=str(e)
            )
            return self._handle_loading_error(e, parameters)
    
    def get_loading_statistics(self) -> Dict[str, Any]:
        """Get statistics about model loading performance"""
        stats = {
            "total_cached_parameters": len(self._parameter_cache),
            "cache_hit_rate": 0.0,
            "average_loading_times": {},
            "memory_usage_stats": {}
        }
        
        if self._parameter_cache:
            total_uses = sum(cache.get("use_count", 1) for cache in self._parameter_cache.values())
            cache_hits = sum(cache.get("use_count", 1) - 1 for cache in self._parameter_cache.values())
            stats["cache_hit_rate"] = cache_hits / total_uses if total_uses > 0 else 0.0
            
            # Calculate average loading times by model type
            loading_times = [cache.get("loading_time", 0) for cache in self._parameter_cache.values()]
            if loading_times:
                stats["average_loading_times"]["overall"] = sum(loading_times) / len(loading_times)
            
            # Memory usage statistics
            memory_usages = [cache.get("memory_usage_mb", 0) for cache in self._parameter_cache.values()]
            if memory_usages:
                stats["memory_usage_stats"] = {
                    "average_mb": sum(memory_usages) / len(memory_usages),
                    "max_mb": max(memory_usages),
                    "min_mb": min(memory_usages)
                }
        
        return stats
    
    def clear_cache(self, older_than_days: int = None):
        """Clear parameter cache, optionally only entries older than specified days"""
        if older_than_days is None:
            self._parameter_cache.clear()
            self.logger.info("Cleared all parameter cache")
        else:
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            keys_to_remove = []
            
            for key, cache_data in self._parameter_cache.items():
                last_used_str = cache_data.get("last_used")
                if last_used_str:
                    try:
                        last_used = datetime.fromisoformat(last_used_str)
                        if last_used < cutoff_date:
                            keys_to_remove.append(key)
                    except ValueError:
                        keys_to_remove.append(key)  # Remove invalid dates
            
            for key in keys_to_remove:
                del self._parameter_cache[key]
            
            self.logger.info(f"Cleared {len(keys_to_remove)} cache entries older than {older_than_days} days")
        
        self._save_parameter_cache()
    
    def get_current_progress(self) -> ModelLoadingProgress:
        """Get current loading progress"""
        return self._current_progress


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    manager = ModelLoadingManager()
    
    # Add a progress callback
    def progress_callback(progress: ModelLoadingProgress):
        print(f"Progress: {progress.phase.value} - {progress.progress_percent:.1f}% - {progress.current_step}")
    
    manager.add_progress_callback(progress_callback)
    
    # Example model loading (this would fail without actual model)
    result = manager.load_model(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype="float16",
        device_map="auto"
    )
    
    if result.success:
        print(f"Model loaded successfully in {result.loading_time:.2f}s")
        print(f"Memory usage: {result.memory_usage_mb:.1f}MB")
        print(f"Cache hit: {result.cache_hit}")
    else:
        print(f"Loading failed: {result.error_message}")
        print(f"Error code: {result.error_code}")
        print("Suggestions:")
        for suggestion in result.suggestions:
            print(f"  - {suggestion}")
    
    # Print statistics
    stats = manager.get_loading_statistics()
    print(f"Loading statistics: {stats}")