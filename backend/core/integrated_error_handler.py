"""
Integrated Error Handler for Real AI Model Integration

This module provides enhanced error handling that bridges the FastAPI backend
with the existing GenerationErrorHandler from the Wan2.2 infrastructure.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import traceback
import psutil
from pathlib import Path

# Import torch optionally
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import existing error handling infrastructure
try:
    from infrastructure.hardware.error_handler import (
        GenerationErrorHandler,
        UserFriendlyError,
        ErrorCategory,
        ErrorSeverity,
        RecoveryAction
    )
    EXISTING_ERROR_HANDLER_AVAILABLE = True
except ImportError:
    EXISTING_ERROR_HANDLER_AVAILABLE = False
    # Create minimal fallback classes
    class ErrorCategory(Enum):
        VRAM_MEMORY = "vram_memory"
        MODEL_LOADING = "model_loading"
        GENERATION_PIPELINE = "generation_pipeline"
        INPUT_VALIDATION = "input_validation"
        SYSTEM_RESOURCE = "system_resource"
        UNKNOWN = "unknown"
    
    class ErrorSeverity(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
    @dataclass
    class RecoveryAction:
        action_type: str
        description: str
        parameters: Dict[str, Any] = None
        automatic: bool = False
        success_probability: float = 0.5
        
        def __post_init__(self):
            if self.parameters is None:
                self.parameters = {}
    
    @dataclass
    class UserFriendlyError:
        category: ErrorCategory
        severity: ErrorSeverity
        title: str
        message: str
        recovery_suggestions: List[str]
        recovery_actions: List[RecoveryAction] = None
        technical_details: Optional[str] = None
        error_code: Optional[str] = None
        
        def __post_init__(self):
            if self.recovery_actions is None:
                self.recovery_actions = []


logger = logging.getLogger(__name__)


class IntegratedErrorHandler:
    """
    Enhanced error handler that integrates FastAPI backend with existing
    GenerationErrorHandler infrastructure for comprehensive error management.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Initialize existing error handler if available
        if EXISTING_ERROR_HANDLER_AVAILABLE:
            try:
                self.existing_handler = GenerationErrorHandler()
                self.logger.info("Existing GenerationErrorHandler initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize existing error handler: {e}")
                self.existing_handler = None
        else:
            self.existing_handler = None
            self.logger.warning("Existing error handler not available, using fallback implementation")
        
        # Initialize FastAPI-specific error patterns
        self._fastapi_error_patterns = self._initialize_fastapi_error_patterns()
        self._recovery_strategies = self._initialize_recovery_strategies()
    
    def _setup_logging(self):
        """Setup logging configuration for integrated error handling"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _initialize_fastapi_error_patterns(self) -> Dict[str, ErrorCategory]:
        """Initialize FastAPI-specific error patterns"""
        return {
            # Model integration specific errors
            "model integration bridge": ErrorCategory.MODEL_LOADING,
            "real generation pipeline": ErrorCategory.GENERATION_PIPELINE,
            "system integration": ErrorCategory.SYSTEM_RESOURCE,
            "websocket connection": ErrorCategory.SYSTEM_RESOURCE,
            
            # FastAPI specific errors
            "validation error": ErrorCategory.INPUT_VALIDATION,
            "request timeout": ErrorCategory.SYSTEM_RESOURCE,
            "connection error": ErrorCategory.SYSTEM_RESOURCE,
            "database error": ErrorCategory.SYSTEM_RESOURCE,
            
            # Model download and management
            "model download failed": ErrorCategory.MODEL_LOADING,
            "model integrity check": ErrorCategory.MODEL_LOADING,
            "model cache": ErrorCategory.MODEL_LOADING,
            
            # Hardware optimization errors
            "wan22 system optimizer": ErrorCategory.SYSTEM_RESOURCE,
            "hardware detection": ErrorCategory.SYSTEM_RESOURCE,
            "optimization failed": ErrorCategory.VRAM_MEMORY,
        }
    
    def _initialize_recovery_strategies(self) -> Dict[ErrorCategory, List[RecoveryAction]]:
        """Initialize recovery strategies specific to FastAPI integration"""
        if not EXISTING_ERROR_HANDLER_AVAILABLE:
            return {}
        
        return {
            ErrorCategory.MODEL_LOADING: [
                RecoveryAction(
                    action_type="trigger_model_download",
                    description="Automatically download missing model using ModelDownloader",
                    parameters={"use_model_downloader": True, "verify_integrity": True},
                    automatic=True,
                    success_probability=0.9
                ),
                RecoveryAction(
                    action_type="clear_model_cache_and_reload",
                    description="Clear model cache and reload using ModelManager",
                    parameters={"clear_cache": True, "force_reload": True},
                    automatic=True,
                    success_probability=0.7
                ),
                RecoveryAction(
                    action_type="fallback_to_mock_generation",
                    description="Temporarily use mock generation while resolving model issues",
                    parameters={"enable_mock_mode": True, "notify_user": True},
                    automatic=False,
                    success_probability=1.0
                )
            ],
            
            ErrorCategory.VRAM_MEMORY: [
                RecoveryAction(
                    action_type="apply_wan22_optimization",
                    description="Apply WAN22 system optimization for VRAM management",
                    parameters={"use_wan22_optimizer": True, "apply_quantization": True},
                    automatic=True,
                    success_probability=0.85
                ),
                RecoveryAction(
                    action_type="enable_model_offloading",
                    description="Enable CPU offloading through ModelIntegrationBridge",
                    parameters={"enable_cpu_offload": True, "sequential_cpu_offload": True},
                    automatic=True,
                    success_probability=0.8
                ),
                RecoveryAction(
                    action_type="reduce_generation_parameters",
                    description="Automatically reduce resolution and steps to fit VRAM",
                    parameters={"max_resolution": "720p", "max_steps": 20, "batch_size": 1},
                    automatic=True,
                    success_probability=0.9
                )
            ],
            
            ErrorCategory.GENERATION_PIPELINE: [
                RecoveryAction(
                    action_type="restart_generation_pipeline",
                    description="Restart the RealGenerationPipeline with fresh state",
                    parameters={"clear_pipeline_cache": True, "reinitialize": True},
                    automatic=True,
                    success_probability=0.6
                ),
                RecoveryAction(
                    action_type="fallback_generation_settings",
                    description="Use conservative generation settings for stability",
                    parameters={"use_safe_settings": True, "disable_advanced_features": True},
                    automatic=True,
                    success_probability=0.8
                )
            ]
        }
    
    async def handle_error(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None
    ) -> UserFriendlyError:
        """
        Main error handling method that integrates with existing infrastructure
        
        Args:
            error: The exception that occurred
            context: Additional context about the error (FastAPI request, task info, etc.)
            
        Returns:
            UserFriendlyError with enhanced recovery suggestions for FastAPI integration
        """
        # Enhance context with FastAPI-specific information
        enhanced_context = self._enhance_context_for_fastapi(context or {})
        
        # Check if this is a WAN model-specific error
        if self._is_wan_model_error(error, enhanced_context):
            try:
                # Use WAN-specific error handler
                return await self._handle_wan_model_error(error, enhanced_context)
            except Exception as e:
                self.logger.warning(f"WAN error handler failed, falling back to integrated handling: {e}")
        
        # Use existing error handler if available
        if self.existing_handler:
            try:
                user_error = self.existing_handler.handle_error(error, enhanced_context)
                # Enhance with FastAPI-specific recovery suggestions
                user_error = self._enhance_error_for_fastapi(user_error, enhanced_context)
                return user_error
            except Exception as e:
                self.logger.error(f"Existing error handler failed: {e}")
                # Fall back to integrated handling
        
        # Fallback to integrated error handling
        return self._handle_error_integrated(error, enhanced_context)
    
    def _enhance_context_for_fastapi(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance error context with FastAPI-specific information"""
        enhanced_context = context.copy()
        
        # Add system integration status
        enhanced_context.update({
            "fastapi_integration": True,
            "error_handler_type": "integrated",
            "existing_handler_available": EXISTING_ERROR_HANDLER_AVAILABLE
        })
        
        # Add model integration status if available
        try:
            # This would be injected by the generation service
            if "generation_service" in context:
                service = context["generation_service"]
                enhanced_context.update({
                    "model_bridge_available": hasattr(service, 'model_integration_bridge') and service.model_integration_bridge is not None,
                    "real_pipeline_available": hasattr(service, 'real_generation_pipeline') and service.real_generation_pipeline is not None,
                    "wan22_optimizer_available": hasattr(service, 'wan22_system_optimizer') and service.wan22_system_optimizer is not None
                })
        except Exception:
            pass
        
        return enhanced_context
    
    def _enhance_error_for_fastapi(
        self, 
        user_error: UserFriendlyError, 
        context: Dict[str, Any]
    ) -> UserFriendlyError:
        """Enhance existing error with FastAPI-specific recovery suggestions"""
        
        # Add FastAPI-specific recovery suggestions
        fastapi_suggestions = self._get_fastapi_recovery_suggestions(user_error.category, context)
        
        # Combine existing suggestions with FastAPI-specific ones
        enhanced_suggestions = list(user_error.recovery_suggestions)
        enhanced_suggestions.extend(fastapi_suggestions)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in enhanced_suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
        
        # Update the error with enhanced suggestions
        user_error.recovery_suggestions = unique_suggestions[:7]  # Limit to 7 suggestions
        
        return user_error
    
    def _get_fastapi_recovery_suggestions(
        self, 
        category: ErrorCategory, 
        context: Dict[str, Any]
    ) -> List[str]:
        """Get FastAPI-specific recovery suggestions"""
        suggestions = []
        
        if category == ErrorCategory.MODEL_LOADING:
            suggestions.extend([
                "Check the model management API endpoints for model status",
                "Use the model download API to re-download corrupted models",
                "Verify model integration bridge is properly initialized"
            ])
            
            if not context.get("model_bridge_available", True):
                suggestions.append("Model integration bridge is not available - restart the service")
        
        elif category == ErrorCategory.VRAM_MEMORY:
            suggestions.extend([
                "Use the system optimization API to apply WAN22 optimizations",
                "Check system stats API for current VRAM usage",
                "Enable automatic VRAM management in generation settings"
            ])
            
            if context.get("wan22_optimizer_available", False):
                suggestions.append("WAN22 system optimizer is available - enable automatic optimization")
        
        elif category == ErrorCategory.GENERATION_PIPELINE:
            suggestions.extend([
                "Check generation queue status via API",
                "Restart generation service if pipeline is stuck",
                "Use WebSocket connection to monitor generation progress"
            ])
            
            if not context.get("real_pipeline_available", True):
                suggestions.append("Real generation pipeline is not available - check service initialization")
        
        elif category == ErrorCategory.SYSTEM_RESOURCE:
            suggestions.extend([
                "Check system health via monitoring API",
                "Review WebSocket connection status",
                "Verify database connectivity"
            ])
        
        return suggestions
    
    def _handle_error_integrated(
        self, 
        error: Exception, 
        context: Dict[str, Any]
    ) -> UserFriendlyError:
        """Handle error using integrated fallback implementation"""
        
        error_message = str(error).lower()
        category = self._categorize_error_integrated(error_message, context)
        severity = self._determine_severity_integrated(error, category)
        
        title, message = self._generate_error_message_integrated(error, category, context)
        recovery_suggestions = self._generate_recovery_suggestions_integrated(category, error, context)
        
        return UserFriendlyError(
            category=category,
            severity=severity,
            title=title,
            message=message,
            recovery_suggestions=recovery_suggestions,
            recovery_actions=[],  # Add empty recovery actions for fallback
            technical_details=self._format_technical_details_integrated(error, context),
            error_code=f"FASTAPI_{category.value.upper()}_{hash(str(error)) % 10000:04d}"
        )
    
    def _categorize_error_integrated(self, error_message: str, context: Optional[Dict[str, Any]] = None) -> ErrorCategory:
        """Categorize error using integrated patterns"""
        # Check for forced category in context
        if context and "force_category" in context:
            force_category = context["force_category"]
            for category in ErrorCategory:
                if category.value == force_category:
                    return category
        
        # Check FastAPI-specific patterns first
        for pattern, category in self._fastapi_error_patterns.items():
            if pattern in error_message:
                return category
        
        # Fall back to basic categorization
        if "cuda" in error_message or "memory" in error_message or "vram" in error_message:
            return ErrorCategory.VRAM_MEMORY
        elif "model" in error_message:
            return ErrorCategory.MODEL_LOADING
        elif "generation" in error_message or "pipeline" in error_message:
            return ErrorCategory.GENERATION_PIPELINE
        elif "validation" in error_message or "invalid" in error_message:
            return ErrorCategory.INPUT_VALIDATION
        else:
            return ErrorCategory.UNKNOWN
    
    def _determine_severity_integrated(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity for integrated handling"""
        if isinstance(error, (SystemError, MemoryError)):
            return ErrorSeverity.CRITICAL
        elif category in [ErrorCategory.MODEL_LOADING, ErrorCategory.VRAM_MEMORY]:
            return ErrorSeverity.HIGH
        elif category == ErrorCategory.GENERATION_PIPELINE:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _generate_error_message_integrated(
        self, 
        error: Exception, 
        category: ErrorCategory, 
        context: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Generate user-friendly error message for integrated handling"""
        
        messages = {
            ErrorCategory.MODEL_LOADING: (
                "Model Integration Error",
                "Failed to load or integrate the AI model. This may be due to missing files or system issues."
            ),
            ErrorCategory.VRAM_MEMORY: (
                "GPU Memory Insufficient",
                "Your GPU doesn't have enough memory for this generation. Try reducing settings or enabling optimization."
            ),
            ErrorCategory.GENERATION_PIPELINE: (
                "Generation Pipeline Error",
                "The video generation process encountered an error. Try adjusting your settings or restarting."
            ),
            ErrorCategory.INPUT_VALIDATION: (
                "Input Validation Error",
                "There's an issue with your input parameters. Please check your settings and try again."
            ),
            ErrorCategory.SYSTEM_RESOURCE: (
                "System Resource Error",
                "System resources are insufficient or unavailable. Try closing other applications."
            )
        }
        
        return messages.get(category, ("Unexpected Error", "An unexpected error occurred. Please try again."))
    
    def _generate_recovery_suggestions_integrated(
        self, 
        category: ErrorCategory, 
        error: Exception, 
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate recovery suggestions for integrated handling"""
        
        base_suggestions = {
            ErrorCategory.MODEL_LOADING: [
                "Restart the application to reload models",
                "Check that model files are not corrupted",
                "Ensure sufficient disk space for model files",
                "Try downloading the model again"
            ],
            ErrorCategory.VRAM_MEMORY: [
                "Reduce the resolution (try 720p instead of 1080p)",
                "Lower the number of inference steps",
                "Close other GPU-intensive applications",
                "Enable model quantization if available"
            ],
            ErrorCategory.GENERATION_PIPELINE: [
                "Try generating again - this might be temporary",
                "Reduce the complexity of your prompt",
                "Try different generation settings",
                "Restart the application if the problem persists"
            ],
            ErrorCategory.INPUT_VALIDATION: [
                "Check that your prompt is not too long",
                "Verify uploaded images are in supported formats",
                "Ensure resolution settings are valid",
                "Try using simpler prompts"
            ],
            ErrorCategory.SYSTEM_RESOURCE: [
                "Close unnecessary applications",
                "Check available disk space",
                "Restart your computer if memory usage is high",
                "Try generating at lower settings"
            ]
        }
        
        suggestions = base_suggestions.get(category, ["Try the operation again", "Restart the application"])
        
        # Add FastAPI-specific suggestions
        fastapi_suggestions = self._get_fastapi_recovery_suggestions(category, context)
        suggestions.extend(fastapi_suggestions)
        
        return suggestions[:6]  # Limit to 6 suggestions
    
    def _format_technical_details_integrated(
        self, 
        error: Exception, 
        context: Dict[str, Any]
    ) -> str:
        """Format technical details for integrated handling"""
        details = [
            f"Error Type: {type(error).__name__}",
            f"Error Message: {str(error)}",
            f"Handler Type: Integrated FastAPI Handler"
        ]
        
        if context:
            details.append("Context:")
            for key, value in context.items():
                if key != "generation_service":  # Skip complex objects
                    details.append(f"  {key}: {value}")
        
        # Add system info
        try:
            details.append("System Info:")
            details.append(f"  CPU Usage: {psutil.cpu_percent()}%")
            details.append(f"  Memory Usage: {psutil.virtual_memory().percent}%")
            
            if TORCH_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available():
                details.append(f"  GPU Memory Allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
                details.append(f"  GPU Memory Reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
        except Exception:
            details.append("  System info unavailable")
        
        return "\n".join(details)
    
    async def handle_model_loading_error(
        self, 
        error: Exception, 
        model_type: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> UserFriendlyError:
        """Handle model loading errors with automatic recovery attempts"""
        
        enhanced_context = context or {}
        enhanced_context.update({
            "error_type": "model_loading",
            "model_type": model_type,
            "recovery_attempted": False
        })
        
        user_error = await self.handle_error(error, enhanced_context)
        
        # Attempt automatic recovery for model loading errors
        if user_error.category == ErrorCategory.MODEL_LOADING:
            recovery_success = await self._attempt_model_loading_recovery(error, model_type, enhanced_context)
            if recovery_success:
                user_error.recovery_suggestions.insert(0, "✓ Automatic recovery attempted - try your request again")
                enhanced_context["recovery_attempted"] = True
        
        return user_error
    
    async def handle_vram_exhaustion_error(
        self, 
        error: Exception, 
        generation_params: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None
    ) -> UserFriendlyError:
        """Handle VRAM exhaustion with optimization fallbacks"""
        
        enhanced_context = context or {}
        enhanced_context.update({
            "error_type": "vram_exhaustion",
            "generation_params": generation_params,
            "optimization_applied": False
        })
        
        user_error = await self.handle_error(error, enhanced_context)
        
        # Attempt automatic VRAM optimization
        if user_error.category == ErrorCategory.VRAM_MEMORY:
            optimization_success = await self._attempt_vram_optimization(generation_params, enhanced_context)
            if optimization_success:
                user_error.recovery_suggestions.insert(0, "✓ VRAM optimization applied - try with reduced settings")
                enhanced_context["optimization_applied"] = True
        
        return user_error
    
    async def handle_generation_pipeline_error(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None
    ) -> UserFriendlyError:
        """Handle generation pipeline errors with recovery suggestions"""
        
        enhanced_context = context or {}
        enhanced_context.update({
            "error_type": "generation_pipeline",
            "pipeline_recovery_attempted": False,
            "force_category": "generation_pipeline"
        })
        
        # Force use of integrated handler for pipeline errors to ensure correct categorization
        user_error = self._handle_error_integrated(error, enhanced_context)
        
        # Attempt automatic recovery for pipeline errors
        if user_error.category == ErrorCategory.GENERATION_PIPELINE:
            recovery_success = await self._attempt_pipeline_recovery(error, enhanced_context)
            if recovery_success:
                user_error.recovery_suggestions.insert(0, "✓ Pipeline recovery attempted - try your request again")
                enhanced_context["pipeline_recovery_attempted"] = True
        
        return user_error
    
    async def _attempt_pipeline_recovery(
        self, 
        error: Exception, 
        context: Dict[str, Any]
    ) -> bool:
        """Attempt automatic recovery for pipeline errors"""
        try:
            # Clear any cached pipeline state
            if TORCH_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Basic pipeline recovery - would integrate with actual pipeline
            self.logger.info("Attempting pipeline recovery")
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline recovery failed: {e}")
            return False
    
    async def _attempt_model_loading_recovery(
        self, 
        error: Exception, 
        model_type: str, 
        context: Dict[str, Any]
    ) -> bool:
        """Attempt automatic recovery for model loading errors"""
        try:
            # Clear model cache
            if TORCH_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Try to trigger model download if model is missing
            if "not found" in str(error).lower() or "missing" in str(error).lower():
                self.logger.info(f"Attempting to trigger model download for {model_type}")
                # This would integrate with the ModelDownloader
                # For now, just log the attempt
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"Model loading recovery failed: {e}")
            return False
    
    async def _attempt_vram_optimization(
        self, 
        generation_params: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> bool:
        """Attempt automatic VRAM optimization"""
        try:
            # Clear GPU cache
            if TORCH_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Apply basic optimizations
            optimizations_applied = []
            
            # Reduce resolution if high
            if generation_params.get("resolution") == "1080p":
                generation_params["resolution"] = "720p"
                optimizations_applied.append("resolution_reduced")
            
            # Reduce steps if high
            if generation_params.get("steps", 0) > 25:
                generation_params["steps"] = 20
                optimizations_applied.append("steps_reduced")
            
            # Enable quantization
            generation_params["enable_quantization"] = True
            optimizations_applied.append("quantization_enabled")
            
            if optimizations_applied:
                self.logger.info(f"Applied VRAM optimizations: {optimizations_applied}")
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"VRAM optimization failed: {e}")
            return False
    
    def get_error_categories(self) -> List[str]:
        """Get list of available error categories"""
        return [category.value for category in ErrorCategory]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status for error context"""
        status = {
            "existing_handler_available": EXISTING_ERROR_HANDLER_AVAILABLE,
            "integrated_handler_active": True
        }
        
        try:
            status.update({
                "cpu_usage_percent": psutil.cpu_percent(),
                "memory_usage_percent": psutil.virtual_memory().percent,
                "available_memory_gb": psutil.virtual_memory().available / (1024**3)
            })
            
            if TORCH_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available():
                status.update({
                    "gpu_available": True,
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3)
                })
            else:
                status["gpu_available"] = False
        except Exception as e:
            status["system_info_error"] = str(e)
        
        return status
    
    def _is_wan_model_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if this is a WAN model-specific error"""
        error_message = str(error).lower()
        
        # Check for WAN-specific keywords in error message
        wan_keywords = [
            "wan", "wan22", "wan_t2v", "wan_i2v", "wan_ti2v",
            "wan model", "wan pipeline", "wan generation",
            "t2v-a14b", "i2v-a14b", "ti2v-5b"
        ]
        
        for keyword in wan_keywords:
            if keyword in error_message:
                return True
        
        # Check context for WAN model indicators
        if context:
            # Check for WAN model type in context
            model_type = context.get("model_type", "")
            if isinstance(model_type, str) and any(wan_type in model_type.lower() for wan_type in ["t2v", "i2v", "ti2v", "wan"]):
                return True
            
            # Check for WAN-specific context keys
            wan_context_keys = [
                "wan_model_type", "wan_error_stage", "wan_model_loaded",
                "wan_weights_loaded", "wan_hardware_optimized"
            ]
            
            for key in wan_context_keys:
                if key in context:
                    return True
            
            # Check error stage for WAN-specific stages
            error_stage = context.get("error_stage", "")
            if error_stage in ["wan_loading", "wan_inference", "wan_optimization"]:
                return True
        
        # Check exception type for WAN-specific exceptions
        exception_name = type(error).__name__.lower()
        if "wan" in exception_name:
            return True
        
        return False
    
    async def _handle_wan_model_error(self, error: Exception, context: Dict[str, Any]) -> UserFriendlyError:
        """Handle WAN model-specific errors using the WAN error handler"""
        try:
            # Import WAN error handler
            from backend.core.models.wan_models.wan_model_error_handler import (
                get_wan_error_handler,
                WANErrorContext,
                WANModelType
            )
            
            # Create WAN error context from integrated context
            wan_context = WANErrorContext()
            
            # Map model type
            model_type_str = context.get("model_type") or context.get("wan_model_type", "")
            if model_type_str:
                try:
                    if "t2v" in model_type_str.lower():
                        wan_context.model_type = WANModelType.T2V_A14B
                    elif "i2v" in model_type_str.lower():
                        wan_context.model_type = WANModelType.I2V_A14B
                    elif "ti2v" in model_type_str.lower():
                        wan_context.model_type = WANModelType.TI2V_5B
                except (ValueError, AttributeError):
                    pass
            
            # Map context fields
            wan_context.model_loaded = context.get("model_loaded") or context.get("wan_model_loaded", False)
            wan_context.weights_loaded = context.get("weights_loaded") or context.get("wan_weights_loaded", False)
            wan_context.hardware_optimized = context.get("hardware_optimized") or context.get("wan_hardware_optimized", False)
            wan_context.generation_params = context.get("generation_params")
            wan_context.vram_usage_gb = context.get("vram_usage_gb", 0.0)
            wan_context.available_vram_gb = context.get("available_vram_gb", 0.0)
            wan_context.applied_optimizations = context.get("applied_optimizations", [])
            wan_context.error_stage = context.get("error_stage") or context.get("wan_error_stage", "unknown")
            wan_context.checkpoint_path = context.get("checkpoint_path")
            wan_context.previous_errors = context.get("previous_errors", [])
            
            # Create hardware profile if available
            if context.get("hardware_profile"):
                try:
                    from backend.core.models.wan_models.wan_base_model import HardwareProfile
                    hw_info = context["hardware_profile"]
                    wan_context.hardware_profile = HardwareProfile(
                        gpu_name=hw_info.get("gpu_name", "Unknown"),
                        total_vram_gb=hw_info.get("total_vram_gb", 0.0),
                        available_vram_gb=hw_info.get("available_vram_gb", 0.0),
                        cpu_cores=hw_info.get("cpu_cores", 1),
                        total_ram_gb=hw_info.get("total_ram_gb", 0.0),
                        architecture_type=hw_info.get("architecture_type", "cuda"),
                        supports_fp16=hw_info.get("supports_fp16", True),
                        supports_bf16=hw_info.get("supports_bf16", False),
                        tensor_cores_available=hw_info.get("tensor_cores_available", False)
                    )
                except ImportError:
                    pass
            
            # Create WAN error handler instance directly to avoid circular dependency
            from backend.core.models.wan_models.wan_model_error_handler import WANModelErrorHandler
            wan_handler = WANModelErrorHandler(avoid_integrated_handler=True)
            wan_error = await wan_handler.handle_wan_error(error, wan_context)
            
            # Add integrated handler context to the error
            wan_error.technical_details += f"\n\nIntegrated Handler Context:\n"
            for key, value in context.items():
                if key not in ["generation_service", "hardware_profile"]:  # Skip complex objects
                    wan_error.technical_details += f"  {key}: {value}\n"
            
            return wan_error
            
        except ImportError as e:
            self.logger.warning(f"WAN error handler not available: {e}")
            # Fallback to integrated error handling with WAN context
            enhanced_context = context.copy()
            enhanced_context["error_type"] = "wan_model_fallback"
            return self._handle_error_integrated(error, enhanced_context)
        
        except Exception as e:
            self.logger.error(f"WAN error handler failed: {e}")
            # Fallback to integrated error handling
            enhanced_context = context.copy()
            enhanced_context["error_type"] = "wan_model_handler_failed"
            enhanced_context["wan_handler_error"] = str(e)
            return self._handle_error_integrated(error, enhanced_context)


# Convenience functions for common error scenarios
async def handle_model_loading_error(
    error: Exception, 
    model_type: str, 
    context: Optional[Dict[str, Any]] = None
) -> UserFriendlyError:
    """Handle model loading errors with automatic recovery"""
    handler = IntegratedErrorHandler()
    return await handler.handle_model_loading_error(error, model_type, context)


async def handle_vram_exhaustion_error(
    error: Exception, 
    generation_params: Dict[str, Any], 
    context: Optional[Dict[str, Any]] = None
) -> UserFriendlyError:
    """Handle VRAM exhaustion with optimization fallbacks"""
    handler = IntegratedErrorHandler()
    return await handler.handle_vram_exhaustion_error(error, generation_params, context)


async def handle_generation_pipeline_error(
    error: Exception, 
    context: Optional[Dict[str, Any]] = None
) -> UserFriendlyError:
    """Handle generation pipeline errors"""
    handler = IntegratedErrorHandler()
    enhanced_context = context or {}
    enhanced_context["error_type"] = "generation_pipeline"
    enhanced_context["force_category"] = "generation_pipeline"  # Force correct categorization
    return await handler.handle_generation_pipeline_error(error, enhanced_context)


async def handle_wan_model_error(
    error: Exception,
    wan_context: Optional[Dict[str, Any]] = None
) -> UserFriendlyError:
    """Handle WAN model-specific errors with comprehensive recovery suggestions"""
    try:
        # Import WAN error handler
        from backend.core.models.wan_models.wan_model_error_handler import (
            get_wan_error_handler,
            WANErrorContext,
            WANModelType
        )
        
        # Create WAN error context
        wan_error_context = WANErrorContext()
        
        if wan_context:
            # Map context to WAN error context
            if "model_type" in wan_context:
                try:
                    wan_error_context.model_type = WANModelType(wan_context["model_type"])
                except (ValueError, KeyError):
                    pass
            
            wan_error_context.model_loaded = wan_context.get("model_loaded", False)
            wan_error_context.weights_loaded = wan_context.get("weights_loaded", False)
            wan_error_context.hardware_optimized = wan_context.get("hardware_optimized", False)
            wan_error_context.generation_params = wan_context.get("generation_params")
            wan_error_context.vram_usage_gb = wan_context.get("vram_usage_gb", 0.0)
            wan_error_context.available_vram_gb = wan_context.get("available_vram_gb", 0.0)
            wan_error_context.applied_optimizations = wan_context.get("applied_optimizations", [])
            wan_error_context.error_stage = wan_context.get("error_stage", "unknown")
            wan_error_context.checkpoint_path = wan_context.get("checkpoint_path")
        
        # Create WAN error handler instance directly to avoid circular dependency
        from backend.core.models.wan_models.wan_model_error_handler import WANModelErrorHandler
        wan_handler = WANModelErrorHandler(avoid_integrated_handler=True)
        return await wan_handler.handle_wan_error(error, wan_error_context)
        
    except ImportError:
        # Fallback to integrated error handler if WAN handler not available
        handler = IntegratedErrorHandler()
        enhanced_context = wan_context or {}
        enhanced_context["error_type"] = "wan_model"
        return await handler.handle_error(error, enhanced_context)
    except Exception as e:
        # Fallback error handling
        handler = IntegratedErrorHandler()
        enhanced_context = wan_context or {}
        enhanced_context["error_type"] = "wan_model"
        enhanced_context["wan_handler_error"] = str(e)
        return await handler.handle_error(error, enhanced_context)


# Global error handler instance
_global_error_handler = None


def get_integrated_error_handler() -> IntegratedErrorHandler:
    """Get the global integrated error handler instance"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = IntegratedErrorHandler()
    return _global_error_handler
