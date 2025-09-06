"""
Intelligent Quantization Controller for WAN22 System Optimization

This module provides comprehensive quantization management with hardware-aware
strategy selection, timeout handling, progress monitoring, and user preference
persistence.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple
import torch

# Configure logging
logger = logging.getLogger(__name__)


class QuantizationMethod(Enum):
    """Supported quantization methods"""
    NONE = "none"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    FP8 = "fp8"  # Experimental


class QuantizationStatus(Enum):
    """Status of quantization operation"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class HardwareProfile:
    """Hardware profile for quantization strategy determination"""
    gpu_model: str
    vram_gb: int
    cuda_version: str
    driver_version: str
    compute_capability: Tuple[int, int]
    supports_bf16: bool = False
    supports_fp8: bool = False
    supports_int8: bool = False


@dataclass
class ModelInfo:
    """Information about the model being quantized"""
    name: str
    size_gb: float
    architecture: str
    components: List[str]
    estimated_vram_usage: float


@dataclass
class QuantizationStrategy:
    """Strategy for quantization based on hardware and model"""
    method: QuantizationMethod
    timeout_seconds: int
    skip_large_components: bool
    component_priorities: Dict[str, int]
    fallback_method: Optional[QuantizationMethod]
    quality_threshold: float


@dataclass
class QuantizationProgress:
    """Progress information for quantization operation"""
    current_component: str
    components_completed: int
    total_components: int
    elapsed_seconds: float
    estimated_remaining_seconds: float
    memory_usage_mb: float
    status: QuantizationStatus


@dataclass
class QuantizationResult:
    """Result of quantization operation"""
    success: bool
    method_used: QuantizationMethod
    components_quantized: List[str]
    memory_saved_mb: float
    time_taken_seconds: float
    quality_score: Optional[float]
    warnings: List[str]
    errors: List[str]
    status: QuantizationStatus


@dataclass
class UserPreferences:
    """User preferences for quantization"""
    preferred_method: QuantizationMethod
    auto_fallback_enabled: bool
    timeout_seconds: int
    skip_quality_check: bool
    remember_model_settings: bool
    model_specific_preferences: Dict[str, QuantizationMethod]


class QuantizationController:
    """
    Intelligent quantization controller with hardware-aware strategy selection,
    timeout management, progress monitoring, and user preference persistence.
    """
    
    def __init__(self, config_path: str = "config.json", preferences_path: str = "quantization_preferences.json"):
        """
        Initialize the quantization controller.
        
        Args:
            config_path: Path to main configuration file
            preferences_path: Path to user preferences file
        """
        self.config_path = Path(config_path)
        self.preferences_path = Path(preferences_path)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration and preferences
        self.config = self._load_config()
        self.preferences = self._load_preferences()
        
        # Hardware detection
        self.hardware_profile = self._detect_hardware_profile()
        
        # Progress tracking
        self._progress_callback: Optional[Callable[[QuantizationProgress], None]] = None
        self._cancellation_event = threading.Event()
        self._current_operation: Optional[threading.Thread] = None
        
        # Quality validation
        self._quality_validator = None
        
        self.logger.info(f"QuantizationController initialized with hardware: {self.hardware_profile.gpu_model}")
    
    def set_progress_callback(self, callback: Callable[[QuantizationProgress], None]) -> None:
        """Set callback function for progress updates"""
        self._progress_callback = callback
    
    def determine_optimal_strategy(self, model_info: ModelInfo) -> QuantizationStrategy:
        """
        Determine optimal quantization strategy based on hardware capabilities and model requirements.
        
        Args:
            model_info: Information about the model to be quantized
            
        Returns:
            Optimal quantization strategy
        """
        self.logger.info(f"Determining quantization strategy for {model_info.name}")
        
        # Check user preferences first
        if model_info.name in self.preferences.model_specific_preferences:
            preferred_method = self.preferences.model_specific_preferences[model_info.name]
            self.logger.info(f"Using model-specific preference: {preferred_method.value}")
        else:
            preferred_method = self.preferences.preferred_method
        
        # Validate hardware compatibility
        compatible_method = self._validate_hardware_compatibility(preferred_method, model_info)
        
        # Determine timeout based on model size and method
        timeout_seconds = self._calculate_timeout(compatible_method, model_info)
        
        # Determine component handling strategy
        skip_large_components = self._should_skip_large_components(compatible_method, model_info)
        
        # Set component priorities
        component_priorities = self._get_component_priorities(compatible_method)
        
        # Determine fallback method
        fallback_method = self._get_fallback_method(compatible_method)
        
        # Set quality threshold
        quality_threshold = self._get_quality_threshold(compatible_method)
        
        strategy = QuantizationStrategy(
            method=compatible_method,
            timeout_seconds=timeout_seconds,
            skip_large_components=skip_large_components,
            component_priorities=component_priorities,
            fallback_method=fallback_method,
            quality_threshold=quality_threshold
        )
        
        self.logger.info(f"Selected strategy: {strategy.method.value} with {timeout_seconds}s timeout")
        return strategy
    
    def apply_quantization_with_monitoring(self, model: Any, strategy: QuantizationStrategy) -> QuantizationResult:
        """
        Apply quantization with progress monitoring and timeout handling.
        
        Args:
            model: Model to quantize
            strategy: Quantization strategy to apply
            
        Returns:
            Quantization result with detailed information
        """
        self.logger.info(f"Starting quantization with method: {strategy.method.value}")
        
        # Reset cancellation event
        self._cancellation_event.clear()
        
        # Initialize progress tracking
        start_time = time.time()
        components = self._get_model_components(model)
        
        progress = QuantizationProgress(
            current_component="initializing",
            components_completed=0,
            total_components=len(components),
            elapsed_seconds=0.0,
            estimated_remaining_seconds=strategy.timeout_seconds,
            memory_usage_mb=self._get_current_vram_usage(),
            status=QuantizationStatus.IN_PROGRESS
        )
        
        # Start quantization in separate thread with timeout
        result_container = {}
        quantization_thread = threading.Thread(
            target=self._quantize_with_progress,
            args=(model, strategy, components, start_time, result_container)
        )
        
        self._current_operation = quantization_thread
        quantization_thread.start()
        
        # Wait for completion or timeout
        quantization_thread.join(timeout=strategy.timeout_seconds)
        
        if quantization_thread.is_alive():
            # Timeout occurred
            self.logger.warning(f"Quantization timed out after {strategy.timeout_seconds} seconds")
            self._cancellation_event.set()
            quantization_thread.join(timeout=5)  # Give it a few seconds to cleanup
            
            # Apply fallback if available
            if strategy.fallback_method and self.preferences.auto_fallback_enabled:
                self.logger.info(f"Applying fallback method: {strategy.fallback_method.value}")
                fallback_strategy = QuantizationStrategy(
                    method=strategy.fallback_method,
                    timeout_seconds=strategy.timeout_seconds // 2,
                    skip_large_components=True,
                    component_priorities=strategy.component_priorities,
                    fallback_method=None,
                    quality_threshold=strategy.quality_threshold
                )
                return self.apply_quantization_with_monitoring(model, fallback_strategy)
            
            return QuantizationResult(
                success=False,
                method_used=strategy.method,
                components_quantized=[],
                memory_saved_mb=0.0,
                time_taken_seconds=strategy.timeout_seconds,
                quality_score=None,
                warnings=[f"Quantization timed out after {strategy.timeout_seconds} seconds"],
                errors=["Quantization timeout"],
                status=QuantizationStatus.TIMEOUT
            )
        
        # Get result from thread
        if 'result' in result_container:
            result = result_container['result']
            
            # Update user preferences if successful
            if result.success and self.preferences.remember_model_settings:
                self._update_model_preference(model, strategy.method)
            
            return result
        else:
            return QuantizationResult(
                success=False,
                method_used=strategy.method,
                components_quantized=[],
                memory_saved_mb=0.0,
                time_taken_seconds=time.time() - start_time,
                quality_score=None,
                warnings=[],
                errors=["Unknown error during quantization"],
                status=QuantizationStatus.FAILED
            )
    
    def cancel_quantization(self) -> bool:
        """
        Cancel current quantization operation.
        
        Returns:
            True if cancellation was successful
        """
        if self._current_operation and self._current_operation.is_alive():
            self.logger.info("Cancelling quantization operation")
            self._cancellation_event.set()
            self._current_operation.join(timeout=10)
            return not self._current_operation.is_alive()
        return True
    
    def validate_quantization_compatibility(self, model_info: ModelInfo, method: QuantizationMethod) -> Dict[str, Any]:
        """
        Validate quantization compatibility with current model and hardware.
        
        Args:
            model_info: Information about the model
            method: Quantization method to validate
            
        Returns:
            Validation result with compatibility information
        """
        compatibility = {
            "compatible": True,
            "warnings": [],
            "recommendations": [],
            "estimated_memory_usage": 0.0,
            "estimated_quality_impact": "minimal"
        }
        
        # Hardware compatibility checks
        if method == QuantizationMethod.BF16 and not self.hardware_profile.supports_bf16:
            compatibility["compatible"] = False
            compatibility["warnings"].append("Hardware does not support BF16 quantization")
            compatibility["recommendations"].append("Use FP16 instead")
        
        if method == QuantizationMethod.FP8 and not self.hardware_profile.supports_fp8:
            compatibility["compatible"] = False
            compatibility["warnings"].append("Hardware does not support FP8 quantization")
            compatibility["recommendations"].append("Use BF16 or INT8 instead")
        
        if method == QuantizationMethod.INT8 and not self.hardware_profile.supports_int8:
            compatibility["compatible"] = False
            compatibility["warnings"].append("Hardware does not support INT8 quantization")
            compatibility["recommendations"].append("Use BF16 instead")
        
        # VRAM usage estimation
        if method == QuantizationMethod.NONE:
            estimated_usage = model_info.estimated_vram_usage
            compatibility["estimated_quality_impact"] = "none"
        elif method == QuantizationMethod.BF16:
            estimated_usage = model_info.estimated_vram_usage * 0.7
            compatibility["estimated_quality_impact"] = "minimal"
        elif method == QuantizationMethod.INT8:
            estimated_usage = model_info.estimated_vram_usage * 0.5
            compatibility["estimated_quality_impact"] = "moderate"
        elif method == QuantizationMethod.FP8:
            estimated_usage = model_info.estimated_vram_usage * 0.4
            compatibility["estimated_quality_impact"] = "significant"
        else:
            estimated_usage = model_info.estimated_vram_usage * 0.8
        
        compatibility["estimated_memory_usage"] = estimated_usage
        
        # VRAM capacity check
        if estimated_usage > self.hardware_profile.vram_gb * 1024 * 0.9:  # 90% threshold
            compatibility["warnings"].append("Estimated VRAM usage exceeds 90% of available memory")
            compatibility["recommendations"].append("Consider more aggressive quantization")
        
        # Model-specific recommendations
        if "transformer" in model_info.components and method == QuantizationMethod.INT8:
            compatibility["warnings"].append("INT8 quantization of transformer may impact quality")
            compatibility["recommendations"].append("Consider BF16 for better quality")
        
        return compatibility
    
    def get_supported_methods(self) -> List[QuantizationMethod]:
        """Get list of quantization methods supported by current hardware"""
        supported = [QuantizationMethod.NONE]
        
        if torch.cuda.is_available():
            supported.append(QuantizationMethod.FP16)
            
            if self.hardware_profile.supports_bf16:
                supported.append(QuantizationMethod.BF16)
            
            if self.hardware_profile.supports_int8:
                supported.append(QuantizationMethod.INT8)
            
            if self.hardware_profile.supports_fp8:
                supported.append(QuantizationMethod.FP8)
        
        return supported
    
    def update_preferences(self, preferences: UserPreferences) -> None:
        """Update user preferences and save to file"""
        self.preferences = preferences
        self._save_preferences()
        self.logger.info("User preferences updated")
    
    def get_preferences(self) -> UserPreferences:
        """Get current user preferences"""
        return self.preferences
    
    def _quantize_with_progress(self, model: Any, strategy: QuantizationStrategy, 
                              components: List[str], start_time: float, result_container: Dict) -> None:
        """Internal method to perform quantization with progress tracking"""
        try:
            initial_vram = self._get_current_vram_usage()
            components_quantized = []
            warnings = []
            errors = []
            
            # Sort components by priority
            sorted_components = sorted(
                components, 
                key=lambda c: strategy.component_priorities.get(c, 0), 
                reverse=True
            )
            
            for i, component_name in enumerate(sorted_components):
                if self._cancellation_event.is_set():
                    result_container['result'] = QuantizationResult(
                        success=False,
                        method_used=strategy.method,
                        components_quantized=components_quantized,
                        memory_saved_mb=0.0,
                        time_taken_seconds=time.time() - start_time,
                        quality_score=None,
                        warnings=warnings,
                        errors=["Operation cancelled by user"],
                        status=QuantizationStatus.CANCELLED
                    )
                    return
                
                # Update progress
                elapsed = time.time() - start_time
                remaining = max(0, strategy.timeout_seconds - elapsed)
                
                progress = QuantizationProgress(
                    current_component=component_name,
                    components_completed=i,
                    total_components=len(sorted_components),
                    elapsed_seconds=elapsed,
                    estimated_remaining_seconds=remaining,
                    memory_usage_mb=self._get_current_vram_usage(),
                    status=QuantizationStatus.IN_PROGRESS
                )
                
                if self._progress_callback:
                    self._progress_callback(progress)
                
                # Quantize component
                try:
                    component = getattr(model, component_name, None)
                    if component is not None:
                        if strategy.skip_large_components and self._is_large_component(component):
                            warnings.append(f"Skipped large component: {component_name}")
                            continue
                        
                        self._quantize_component(component, strategy.method)
                        components_quantized.append(component_name)
                        self.logger.info(f"Quantized component: {component_name}")
                    
                except Exception as e:
                    error_msg = f"Failed to quantize {component_name}: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            # Calculate final metrics
            final_vram = self._get_current_vram_usage()
            memory_saved = max(0, initial_vram - final_vram)
            total_time = time.time() - start_time
            
            # Quality validation (if enabled)
            quality_score = None
            if not self.preferences.skip_quality_check and self._quality_validator:
                try:
                    quality_score = self._quality_validator.validate_quality(model, strategy.method)
                except Exception as e:
                    warnings.append(f"Quality validation failed: {str(e)}")
            
            result_container['result'] = QuantizationResult(
                success=len(components_quantized) > 0,
                method_used=strategy.method,
                components_quantized=components_quantized,
                memory_saved_mb=memory_saved,
                time_taken_seconds=total_time,
                quality_score=quality_score,
                warnings=warnings,
                errors=errors,
                status=QuantizationStatus.COMPLETED if len(components_quantized) > 0 else QuantizationStatus.FAILED
            )
            
        except Exception as e:
            result_container['result'] = QuantizationResult(
                success=False,
                method_used=strategy.method,
                components_quantized=[],
                memory_saved_mb=0.0,
                time_taken_seconds=time.time() - start_time,
                quality_score=None,
                warnings=[],
                errors=[f"Quantization failed: {str(e)}"],
                status=QuantizationStatus.FAILED
            )
    
    def _detect_hardware_profile(self) -> HardwareProfile:
        """Detect hardware capabilities for quantization"""
        try:
            if not torch.cuda.is_available():
                return HardwareProfile(
                    gpu_model="CPU",
                    vram_gb=0,
                    cuda_version="N/A",
                    driver_version="N/A",
                    compute_capability=(0, 0)
                )
            
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            compute_capability = torch.cuda.get_device_capability(0)
            
            # Detect quantization support
            supports_bf16 = torch.cuda.is_bf16_supported()
            supports_int8 = self._check_int8_support()
            supports_fp8 = self._check_fp8_support()
            
            return HardwareProfile(
                gpu_model=gpu_name,
                vram_gb=int(vram_gb),
                cuda_version=torch.version.cuda or "Unknown",
                driver_version="Unknown",  # Would need nvidia-ml-py for this
                compute_capability=compute_capability,
                supports_bf16=supports_bf16,
                supports_int8=supports_int8,
                supports_fp8=supports_fp8
            )
            
        except Exception as e:
            self.logger.error(f"Hardware detection failed: {e}")
            return HardwareProfile(
                gpu_model="Unknown",
                vram_gb=8,  # Conservative default
                cuda_version="Unknown",
                driver_version="Unknown",
                compute_capability=(0, 0)
            )
    
    def _check_int8_support(self) -> bool:
        """Check if INT8 quantization is supported"""
        try:
            import bitsandbytes
            return True
        except ImportError:
            return False
    
    def _check_fp8_support(self) -> bool:
        """Check if FP8 quantization is supported"""
        # FP8 support is limited to newer hardware and experimental
        compute_capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
        return compute_capability >= (8, 9)  # Ada Lovelace and newer
    
    def _validate_hardware_compatibility(self, method: QuantizationMethod, model_info: ModelInfo) -> QuantizationMethod:
        """Validate and adjust quantization method based on hardware compatibility"""
        if method == QuantizationMethod.NONE:
            return method
        
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, disabling quantization")
            return QuantizationMethod.NONE
        
        if method == QuantizationMethod.BF16 and not self.hardware_profile.supports_bf16:
            self.logger.warning("BF16 not supported, falling back to FP16")
            return QuantizationMethod.FP16
        
        if method == QuantizationMethod.INT8 and not self.hardware_profile.supports_int8:
            self.logger.warning("INT8 not supported, falling back to BF16")
            return QuantizationMethod.BF16 if self.hardware_profile.supports_bf16 else QuantizationMethod.FP16
        
        if method == QuantizationMethod.FP8 and not self.hardware_profile.supports_fp8:
            self.logger.warning("FP8 not supported, falling back to BF16")
            return QuantizationMethod.BF16 if self.hardware_profile.supports_bf16 else QuantizationMethod.FP16
        
        return method
    
    def _calculate_timeout(self, method: QuantizationMethod, model_info: ModelInfo) -> int:
        """Calculate appropriate timeout based on method and model size"""
        base_timeout = self.preferences.timeout_seconds
        
        # Adjust based on model size
        size_multiplier = max(1.0, model_info.size_gb / 5.0)  # 5GB baseline
        
        # Adjust based on quantization method complexity
        method_multipliers = {
            QuantizationMethod.NONE: 0.1,
            QuantizationMethod.FP16: 1.0,
            QuantizationMethod.BF16: 1.5,
            QuantizationMethod.INT8: 3.0,
            QuantizationMethod.FP8: 2.0
        }
        
        method_multiplier = method_multipliers.get(method, 1.0)
        
        return int(base_timeout * size_multiplier * method_multiplier)
    
    def _should_skip_large_components(self, method: QuantizationMethod, model_info: ModelInfo) -> bool:
        """Determine if large components should be skipped"""
        # Skip large components for aggressive quantization or large models
        if method in [QuantizationMethod.INT8, QuantizationMethod.FP8]:
            return True
        
        if model_info.size_gb > 10:  # Large models
            return True
        
        return False
    
    def _get_component_priorities(self, method: QuantizationMethod) -> Dict[str, int]:
        """Get component quantization priorities"""
        # Higher numbers = higher priority (quantized first)
        base_priorities = {
            "unet": 10,
            "transformer": 9,
            "transformer_2": 8,
            "text_encoder": 7,
            "text_encoder_2": 6,
            "vae": 3,  # Lower priority due to quality impact
            "scheduler": 1
        }
        
        # Adjust priorities based on method
        if method == QuantizationMethod.INT8:
            # Be more conservative with VAE for INT8
            base_priorities["vae"] = 1
        
        return base_priorities
    
    def _get_fallback_method(self, method: QuantizationMethod) -> Optional[QuantizationMethod]:
        """Get fallback method for given quantization method"""
        fallback_chain = {
            QuantizationMethod.FP8: QuantizationMethod.BF16,
            QuantizationMethod.INT8: QuantizationMethod.BF16,
            QuantizationMethod.BF16: QuantizationMethod.FP16,
            QuantizationMethod.FP16: QuantizationMethod.NONE,
            QuantizationMethod.NONE: None
        }
        
        return fallback_chain.get(method)
    
    def _get_quality_threshold(self, method: QuantizationMethod) -> float:
        """Get quality threshold for given quantization method"""
        thresholds = {
            QuantizationMethod.NONE: 1.0,
            QuantizationMethod.FP16: 0.95,
            QuantizationMethod.BF16: 0.95,
            QuantizationMethod.INT8: 0.85,
            QuantizationMethod.FP8: 0.80
        }
        
        return thresholds.get(method, 0.90)
    
    def _get_model_components(self, model: Any) -> List[str]:
        """Get list of quantizable components from model"""
        components = []
        
        # Common component names
        component_names = [
            "unet", "transformer", "transformer_2",
            "text_encoder", "text_encoder_2",
            "vae", "scheduler"
        ]
        
        for name in component_names:
            if hasattr(model, name) and getattr(model, name) is not None:
                components.append(name)
        
        return components
    
    def _quantize_component(self, component: Any, method: QuantizationMethod) -> None:
        """Quantize a single model component"""
        if method == QuantizationMethod.NONE:
            return
        
        if method == QuantizationMethod.FP16:
            if hasattr(component, 'half'):
                component.half()
            elif hasattr(component, 'to'):
                component.to(dtype=torch.float16)
        
        elif method == QuantizationMethod.BF16:
            if hasattr(component, 'to'):
                component.to(dtype=torch.bfloat16)
        
        elif method == QuantizationMethod.INT8:
            # This would require bitsandbytes integration
            # For now, fall back to BF16
            self.logger.warning("INT8 quantization not fully implemented, using BF16")
            if hasattr(component, 'to'):
                component.to(dtype=torch.bfloat16)
        
        elif method == QuantizationMethod.FP8:
            # FP8 is experimental and would require specific library support
            self.logger.warning("FP8 quantization not implemented, using BF16")
            if hasattr(component, 'to'):
                component.to(dtype=torch.bfloat16)
    
    def _is_large_component(self, component: Any) -> bool:
        """Check if component is considered large"""
        try:
            if hasattr(component, 'parameters'):
                param_count = sum(p.numel() for p in component.parameters())
                return param_count > 1e9  # 1B parameters
            return False
        except:
            return False
    
    def _get_current_vram_usage(self) -> float:
        """Get current VRAM usage in MB"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
            return 0.0
        except:
            return 0.0
    
    def _update_model_preference(self, model: Any, method: QuantizationMethod) -> None:
        """Update model-specific preference"""
        model_name = getattr(model, 'name', 'unknown')
        self.preferences.model_specific_preferences[model_name] = method
        self._save_preferences()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load main configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
        
        # Default configuration
        return {
            "optimization": {
                "default_quantization": "bf16",
                "quantization_levels": ["fp16", "bf16", "int8"]
            }
        }
    
    def _load_preferences(self) -> UserPreferences:
        """Load user preferences"""
        try:
            if self.preferences_path.exists():
                with open(self.preferences_path, 'r') as f:
                    data = json.load(f)
                    return UserPreferences(
                        preferred_method=QuantizationMethod(data.get('preferred_method', 'bf16')),
                        auto_fallback_enabled=data.get('auto_fallback_enabled', True),
                        timeout_seconds=data.get('timeout_seconds', 300),
                        skip_quality_check=data.get('skip_quality_check', False),
                        remember_model_settings=data.get('remember_model_settings', True),
                        model_specific_preferences={
                            k: QuantizationMethod(v) for k, v in data.get('model_specific_preferences', {}).items()
                        }
                    )
        except Exception as e:
            self.logger.error(f"Failed to load preferences: {e}")
        
        # Default preferences
        return UserPreferences(
            preferred_method=QuantizationMethod.BF16,
            auto_fallback_enabled=True,
            timeout_seconds=300,
            skip_quality_check=False,
            remember_model_settings=True,
            model_specific_preferences={}
        )
    
    def _save_preferences(self) -> None:
        """Save user preferences"""
        try:
            data = {
                'preferred_method': self.preferences.preferred_method.value,
                'auto_fallback_enabled': self.preferences.auto_fallback_enabled,
                'timeout_seconds': self.preferences.timeout_seconds,
                'skip_quality_check': self.preferences.skip_quality_check,
                'remember_model_settings': self.preferences.remember_model_settings,
                'model_specific_preferences': {
                    k: v.value for k, v in self.preferences.model_specific_preferences.items()
                }
            }
            
            with open(self.preferences_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save preferences: {e}")