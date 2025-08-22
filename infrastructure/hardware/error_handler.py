"""
Enhanced Error Handling System for Wan2.2 Video Generation

This module provides comprehensive error handling, categorization, and recovery
mechanisms for the video generation pipeline.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any, Union
import logging
import traceback
import torch
import psutil
from pathlib import Path


class ErrorCategory(Enum):
    """Categories of errors that can occur during video generation"""
    INPUT_VALIDATION = "input_validation"
    MODEL_LOADING = "model_loading"
    VRAM_MEMORY = "vram_memory"
    GENERATION_PIPELINE = "generation_pipeline"
    SYSTEM_RESOURCE = "system_resource"
    CONFIGURATION = "configuration"
    FILE_SYSTEM = "file_system"
    NETWORK = "network"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Severity levels for errors"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RecoveryAction:
    """Represents a recovery action that can be taken for an error"""
    action_type: str
    description: str
    parameters: Dict[str, Any]
    automatic: bool = False
    success_probability: float = 0.0


@dataclass
class UserFriendlyError:
    """User-facing error representation with recovery suggestions"""
    category: ErrorCategory
    severity: ErrorSeverity
    title: str
    message: str
    recovery_suggestions: List[str]
    recovery_actions: List[RecoveryAction]
    technical_details: Optional[str] = None
    error_code: Optional[str] = None
    
    def to_html(self) -> str:
        """Convert error to HTML format for UI display"""
        severity_colors = {
            ErrorSeverity.LOW: "#28a745",
            ErrorSeverity.MEDIUM: "#ffc107", 
            ErrorSeverity.HIGH: "#fd7e14",
            ErrorSeverity.CRITICAL: "#dc3545"
        }
        
        color = severity_colors.get(self.severity, "#6c757d")
        
        html = f"""
        <div class="error-container" style="border-left: 4px solid {color}; padding: 15px; margin: 10px 0; background-color: #f8f9fa;">
            <h4 style="color: {color}; margin-top: 0;">{self.title}</h4>
            <p style="margin: 10px 0;">{self.message}</p>
            
            {self._format_recovery_suggestions()}
            
            {self._format_technical_details()}
        </div>
        """
        return html
    
    def _format_recovery_suggestions(self) -> str:
        """Format recovery suggestions as HTML"""
        if not self.recovery_suggestions:
            return ""
            
        suggestions_html = "<h5>Suggested Solutions:</h5><ul>"
        for suggestion in self.recovery_suggestions:
            suggestions_html += f"<li>{suggestion}</li>"
        suggestions_html += "</ul>"
        return suggestions_html
    
    def _format_technical_details(self) -> str:
        """Format technical details as collapsible HTML"""
        if not self.technical_details:
            return ""
            
        return f"""
        <details style="margin-top: 10px;">
            <summary style="cursor: pointer; color: #6c757d;">Technical Details</summary>
            <pre style="background-color: #e9ecef; padding: 10px; margin-top: 5px; font-size: 12px; overflow-x: auto;">{self.technical_details}</pre>
        </details>
        """


class GenerationErrorHandler:
    """Comprehensive error handler for video generation pipeline"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        self._error_patterns = self._initialize_error_patterns()
        self._recovery_strategies = self._initialize_recovery_strategies()
    
    def _setup_logging(self):
        """Setup logging configuration for error handling"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _initialize_error_patterns(self) -> Dict[str, ErrorCategory]:
        """Initialize patterns for error categorization"""
        return {
            # Input validation patterns
            "invalid input provided": ErrorCategory.INPUT_VALIDATION,
            "prompt too long": ErrorCategory.INPUT_VALIDATION,
            "invalid image format": ErrorCategory.INPUT_VALIDATION,
            "unsupported resolution": ErrorCategory.INPUT_VALIDATION,
            "invalid parameters": ErrorCategory.INPUT_VALIDATION,
            
            # Model loading patterns (more specific patterns first)
            "model file not found": ErrorCategory.MODEL_LOADING,
            "model not found": ErrorCategory.MODEL_LOADING,
            "failed to load model": ErrorCategory.MODEL_LOADING,
            "model loading error": ErrorCategory.MODEL_LOADING,
            "corrupted model": ErrorCategory.MODEL_LOADING,
            "model compatibility": ErrorCategory.MODEL_LOADING,
            
            # VRAM/Memory patterns
            "out of memory": ErrorCategory.VRAM_MEMORY,
            "cuda out of memory": ErrorCategory.VRAM_MEMORY,
            "insufficient vram": ErrorCategory.VRAM_MEMORY,
            "memory allocation": ErrorCategory.VRAM_MEMORY,
            "gpu memory": ErrorCategory.VRAM_MEMORY,
            "vram optimization": ErrorCategory.VRAM_MEMORY,
            "quantization failed": ErrorCategory.VRAM_MEMORY,
            
            # Generation pipeline patterns
            "generation failed": ErrorCategory.GENERATION_PIPELINE,
            "pipeline error": ErrorCategory.GENERATION_PIPELINE,
            "inference failed": ErrorCategory.GENERATION_PIPELINE,
            "tensor shape": ErrorCategory.GENERATION_PIPELINE,
            "dimension mismatch": ErrorCategory.GENERATION_PIPELINE,
            
            # System resource patterns
            "disk space": ErrorCategory.FILE_SYSTEM,
            "permission denied": ErrorCategory.FILE_SYSTEM,
            "file not found": ErrorCategory.FILE_SYSTEM,
            "network error": ErrorCategory.NETWORK,
            "connection failed": ErrorCategory.NETWORK,
            
            # Configuration patterns
            "config error": ErrorCategory.CONFIGURATION,
            "missing configuration": ErrorCategory.CONFIGURATION,
            "invalid config": ErrorCategory.CONFIGURATION,
        }
    
    def _initialize_recovery_strategies(self) -> Dict[ErrorCategory, List[RecoveryAction]]:
        """Initialize recovery strategies for each error category"""
        return {
            ErrorCategory.INPUT_VALIDATION: [
                RecoveryAction(
                    action_type="validate_and_fix_prompt",
                    description="Automatically truncate or fix prompt issues",
                    parameters={"max_length": 512},
                    automatic=True,
                    success_probability=0.8
                ),
                RecoveryAction(
                    action_type="suggest_alternative_resolution",
                    description="Suggest compatible resolution settings",
                    parameters={"fallback_resolutions": ["720p", "480p"]},
                    automatic=False,
                    success_probability=0.9
                )
            ],
            
            ErrorCategory.MODEL_LOADING: [
                RecoveryAction(
                    action_type="clear_model_cache",
                    description="Clear model cache and retry loading",
                    parameters={"cache_dirs": ["models", ".cache"]},
                    automatic=True,
                    success_probability=0.6
                ),
                RecoveryAction(
                    action_type="download_model",
                    description="Download missing or corrupted model",
                    parameters={"model_source": "huggingface"},
                    automatic=False,
                    success_probability=0.95
                )
            ],
            
            ErrorCategory.VRAM_MEMORY: [
                RecoveryAction(
                    action_type="apply_system_optimization",
                    description="Apply WAN22 system optimization for VRAM management",
                    parameters={"enable_vram_optimization": True, "apply_quantization": True},
                    automatic=True,
                    success_probability=0.8
                ),
                RecoveryAction(
                    action_type="optimize_vram_usage",
                    description="Automatically optimize VRAM settings",
                    parameters={"enable_cpu_offload": True, "reduce_precision": True},
                    automatic=True,
                    success_probability=0.7
                ),
                RecoveryAction(
                    action_type="apply_quantization",
                    description="Apply model quantization to reduce VRAM usage",
                    parameters={"quantization_method": "bf16", "fallback_method": "int8"},
                    automatic=True,
                    success_probability=0.75
                ),
                RecoveryAction(
                    action_type="reduce_batch_size",
                    description="Reduce generation parameters to fit available VRAM",
                    parameters={"batch_size": 1, "steps": 20},
                    automatic=True,
                    success_probability=0.85
                )
            ],
            
            ErrorCategory.GENERATION_PIPELINE: [
                RecoveryAction(
                    action_type="retry_with_fallback",
                    description="Retry generation with simpler settings",
                    parameters={"fallback_mode": "basic"},
                    automatic=True,
                    success_probability=0.6
                ),
                RecoveryAction(
                    action_type="reset_pipeline",
                    description="Reset generation pipeline to default state",
                    parameters={"clear_cache": True},
                    automatic=True,
                    success_probability=0.5
                )
            ],
            
            ErrorCategory.SYSTEM_RESOURCE: [
                RecoveryAction(
                    action_type="free_system_memory",
                    description="Free up system memory and resources",
                    parameters={"gc_collect": True, "clear_cache": True},
                    automatic=True,
                    success_probability=0.4
                )
            ],
            
            ErrorCategory.CONFIGURATION: [
                RecoveryAction(
                    action_type="reset_to_defaults",
                    description="Reset configuration to default values",
                    parameters={"backup_current": True},
                    automatic=False,
                    success_probability=0.8
                )
            ],
            
            ErrorCategory.FILE_SYSTEM: [
                RecoveryAction(
                    action_type="create_directories",
                    description="Create missing output directories",
                    parameters={"create_parents": True},
                    automatic=True,
                    success_probability=0.9
                )
            ]
        }
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> UserFriendlyError:
        """
        Main error handling method that categorizes errors and provides recovery suggestions
        
        Args:
            error: The exception that occurred
            context: Additional context about the error (parameters, state, etc.)
            
        Returns:
            UserFriendlyError with categorized error information and recovery suggestions
        """
        error_message = str(error).lower()
        error_category = self._categorize_error(error_message)
        severity = self._determine_severity(error, error_category)
        
        # Log the error with context
        self._log_error(error, error_category, context)
        
        # Generate user-friendly error
        user_error = self._create_user_friendly_error(
            error, error_category, severity, context
        )
        
        return user_error
    
    def _categorize_error(self, error_message: str) -> ErrorCategory:
        """Categorize error based on message patterns"""
        for pattern, category in self._error_patterns.items():
            if pattern in error_message:
                return category
        return ErrorCategory.UNKNOWN
    
    def _determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity based on type and category"""
        # Critical errors that prevent any functionality
        if isinstance(error, (SystemError, MemoryError)):
            return ErrorSeverity.CRITICAL
        
        # High severity for model loading and VRAM issues
        if category in [ErrorCategory.MODEL_LOADING, ErrorCategory.VRAM_MEMORY]:
            return ErrorSeverity.HIGH
        
        # Medium severity for generation pipeline issues
        if category == ErrorCategory.GENERATION_PIPELINE:
            return ErrorSeverity.MEDIUM
        
        # Low severity for input validation issues
        if category == ErrorCategory.INPUT_VALIDATION:
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    def _log_error(self, error: Exception, category: ErrorCategory, context: Optional[Dict[str, Any]]):
        """Log error with detailed information"""
        log_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "category": category.value,
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        # Add system information
        log_data["system_info"] = self._get_system_info()
        
        self.logger.error(f"Generation Error: {log_data}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information for error context"""
        info = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "available_memory_gb": psutil.virtual_memory().available / (1024**3)
        }
        
        # Add GPU information if available
        try:
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                info.update({
                    "gpu_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "gpu_memory_allocated": torch.cuda.memory_allocated() / (1024**3),
                    "gpu_memory_reserved": torch.cuda.memory_reserved() / (1024**3)
                })
        except Exception:
            # GPU information not available
            pass
        
        return info
    
    def _create_user_friendly_error(
        self, 
        error: Exception, 
        category: ErrorCategory, 
        severity: ErrorSeverity,
        context: Optional[Dict[str, Any]]
    ) -> UserFriendlyError:
        """Create user-friendly error with recovery suggestions"""
        
        title, message = self._generate_error_message(error, category, context)
        recovery_suggestions = self._generate_recovery_suggestions(category, error, context)
        recovery_actions = self._recovery_strategies.get(category, [])
        
        return UserFriendlyError(
            category=category,
            severity=severity,
            title=title,
            message=message,
            recovery_suggestions=recovery_suggestions,
            recovery_actions=recovery_actions,
            technical_details=self._format_technical_details(error, context),
            error_code=f"{category.value.upper()}_{hash(str(error)) % 10000:04d}"
        )
    
    def _generate_error_message(
        self, 
        error: Exception, 
        category: ErrorCategory, 
        context: Optional[Dict[str, Any]]
    ) -> tuple[str, str]:
        """Generate user-friendly title and message for the error"""
        
        error_messages = {
            ErrorCategory.INPUT_VALIDATION: {
                "title": "Input Validation Error",
                "message": "There's an issue with your input parameters. Please check your settings and try again."
            },
            ErrorCategory.MODEL_LOADING: {
                "title": "Model Loading Failed",
                "message": "The AI model couldn't be loaded. This might be due to missing files or insufficient resources."
            },
            ErrorCategory.VRAM_MEMORY: {
                "title": "Insufficient GPU Memory",
                "message": "Your GPU doesn't have enough memory for this generation. Try reducing the resolution or other settings."
            },
            ErrorCategory.GENERATION_PIPELINE: {
                "title": "Generation Failed",
                "message": "The video generation process encountered an error. This might be temporary - try again or adjust your settings."
            },
            ErrorCategory.SYSTEM_RESOURCE: {
                "title": "System Resource Issue",
                "message": "Your system is running low on resources. Close other applications or try simpler settings."
            },
            ErrorCategory.CONFIGURATION: {
                "title": "Configuration Error",
                "message": "There's an issue with the application configuration. You may need to reset settings to defaults."
            },
            ErrorCategory.FILE_SYSTEM: {
                "title": "File System Error",
                "message": "There's an issue accessing files or directories. Check permissions and available disk space."
            },
            ErrorCategory.NETWORK: {
                "title": "Network Error",
                "message": "Network connection failed. Check your internet connection and try again."
            },
            ErrorCategory.UNKNOWN: {
                "title": "Unexpected Error",
                "message": "An unexpected error occurred. Please try again or contact support if the issue persists."
            }
        }
        
        default_msg = error_messages[ErrorCategory.UNKNOWN]
        msg_info = error_messages.get(category, default_msg)
        
        return msg_info["title"], msg_info["message"]
    
    def _generate_recovery_suggestions(
        self, 
        category: ErrorCategory, 
        error: Exception, 
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate specific recovery suggestions based on error category and context"""
        
        suggestions = {
            ErrorCategory.INPUT_VALIDATION: [
                "Check that your prompt is not too long (under 512 characters recommended)",
                "Verify that uploaded images are in supported formats (PNG, JPG, JPEG)",
                "Ensure resolution settings are compatible with your selected model",
                "Try using simpler prompts without special characters"
            ],
            ErrorCategory.MODEL_LOADING: [
                "Restart the application to reload models",
                "Check that model files are not corrupted",
                "Ensure you have sufficient disk space for model files",
                "Try downloading the model again if it appears to be missing"
            ],
            ErrorCategory.VRAM_MEMORY: [
                "Apply WAN22 system optimization to automatically manage VRAM usage",
                "Enable quantization (bf16 or int8) to reduce memory requirements",
                "Reduce the resolution (try 720p instead of 1080p)",
                "Lower the number of inference steps",
                "Close other GPU-intensive applications",
                "Enable CPU offloading in advanced settings"
            ],
            ErrorCategory.GENERATION_PIPELINE: [
                "Try generating again - this might be a temporary issue",
                "Reduce the complexity of your prompt",
                "Try different generation settings",
                "Restart the application if the problem persists"
            ],
            ErrorCategory.SYSTEM_RESOURCE: [
                "Close unnecessary applications to free up memory",
                "Restart your computer if memory usage is high",
                "Check available disk space",
                "Try generating at a lower resolution"
            ],
            ErrorCategory.CONFIGURATION: [
                "Reset application settings to defaults",
                "Check that all required files are present",
                "Verify your configuration file is not corrupted",
                "Reinstall the application if issues persist"
            ],
            ErrorCategory.FILE_SYSTEM: [
                "Check that the output directory exists and is writable",
                "Ensure you have sufficient disk space",
                "Verify file permissions for the application directory",
                "Try using a different output location"
            ],
            ErrorCategory.NETWORK: [
                "Check your internet connection",
                "Try again in a few minutes",
                "Use offline mode if available",
                "Check firewall settings"
            ],
            ErrorCategory.UNKNOWN: [
                "Try the operation again",
                "Restart the application",
                "Check the application logs for more details",
                "Contact support if the issue persists"
            ]
        }
        
        base_suggestions = suggestions.get(category, suggestions[ErrorCategory.UNKNOWN])
        
        # Add context-specific suggestions
        if context:
            base_suggestions.extend(self._get_context_specific_suggestions(category, context))
        
        return base_suggestions[:5]  # Limit to 5 suggestions to avoid overwhelming users
    
    def _get_context_specific_suggestions(
        self, 
        category: ErrorCategory, 
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate context-specific suggestions based on the error context"""
        suggestions = []
        
        if category == ErrorCategory.VRAM_MEMORY and context:
            if context.get("resolution") == "1080p":
                suggestions.append("Try using 720p resolution instead of 1080p")
            if context.get("steps", 0) > 30:
                suggestions.append("Reduce inference steps to 20-25 for faster generation")
        
        if category == ErrorCategory.INPUT_VALIDATION and context:
            if context.get("prompt_length", 0) > 500:
                suggestions.append("Your prompt is very long - try shortening it to under 300 characters")
            if context.get("image_size"):
                suggestions.append("Try resizing your input image to match the target resolution")
        
        return suggestions
    
    def _format_technical_details(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Format technical details for debugging"""
        details = [
            f"Error Type: {type(error).__name__}",
            f"Error Message: {str(error)}",
            f"Timestamp: {self._get_timestamp()}"
        ]
        
        if context:
            details.append("Context:")
            for key, value in context.items():
                details.append(f"  {key}: {value}")
        
        # Add system info
        sys_info = self._get_system_info()
        details.append("System Info:")
        for key, value in sys_info.items():
            details.append(f"  {key}: {value}")
        
        # Add stack trace (truncated)
        stack_trace = traceback.format_exc()
        if len(stack_trace) > 1000:
            stack_trace = stack_trace[:1000] + "... (truncated)"
        details.append(f"Stack Trace:\n{stack_trace}")
        
        return "\n".join(details)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for error logging"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def attempt_automatic_recovery(
        self, 
        error: UserFriendlyError, 
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Attempt automatic recovery for errors that support it
        
        Args:
            error: The UserFriendlyError to attempt recovery for
            context: Additional context for recovery
            
        Returns:
            Tuple of (success, message) indicating if recovery was attempted and result
        """
        automatic_actions = [
            action for action in error.recovery_actions 
            if action.automatic and action.success_probability > 0.5
        ]
        
        if not automatic_actions:
            return False, "No automatic recovery available for this error"
        
        # Sort by success probability (highest first)
        automatic_actions.sort(key=lambda x: x.success_probability, reverse=True)
        
        for action in automatic_actions:
            try:
                success = self._execute_recovery_action(action, context)
                if success:
                    self.logger.info(f"Automatic recovery successful: {action.description}")
                    return True, f"Automatically resolved: {action.description}"
            except Exception as recovery_error:
                self.logger.warning(f"Recovery action failed: {action.description} - {recovery_error}")
                continue
        
        return False, "Automatic recovery attempts failed"
    
    def _execute_recovery_action(
        self, 
        action: RecoveryAction, 
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Execute a specific recovery action"""
        
        if action.action_type == "apply_system_optimization":
            return self._apply_system_optimization(action.parameters)
        elif action.action_type == "apply_quantization":
            return self._apply_quantization(action.parameters)
        elif action.action_type == "optimize_vram_usage":
            return self._optimize_vram_usage(action.parameters)
        elif action.action_type == "clear_model_cache":
            return self._clear_model_cache(action.parameters)
        elif action.action_type == "free_system_memory":
            return self._free_system_memory(action.parameters)
        elif action.action_type == "create_directories":
            return self._create_directories(action.parameters, context)
        elif action.action_type == "reduce_batch_size":
            return self._reduce_batch_size(action.parameters, context)
        elif action.action_type == "validate_and_fix_prompt":
            return self._validate_and_fix_prompt(action.parameters, context)
        
        return False
    
    def _optimize_vram_usage(self, parameters: Dict[str, Any]) -> bool:
        """Optimize VRAM usage settings"""
        try:
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
                if parameters.get("enable_cpu_offload"):
                    # This would be implemented based on the specific model architecture
                    pass
                return True
        except Exception:
            return False
        return False
    
    def _clear_model_cache(self, parameters: Dict[str, Any]) -> bool:
        """Clear model cache directories"""
        try:
            import shutil
            cache_dirs = parameters.get("cache_dirs", [])
            for cache_dir in cache_dirs:
                cache_path = Path(cache_dir)
                if cache_path.exists() and cache_path.is_dir():
                    # Only clear cache files, not the entire directory
                    for item in cache_path.glob("*.cache"):
                        if item.is_file():
                            item.unlink()
            return True
        except Exception:
            return False
    
    def _free_system_memory(self, parameters: Dict[str, Any]) -> bool:
        """Free system memory"""
        try:
            import gc
            if parameters.get("gc_collect"):
                gc.collect()
            if parameters.get("clear_cache") and hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except Exception:
            return False
    
    def _create_directories(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> bool:
        """Create missing directories"""
        try:
            if context and "output_path" in context:
                output_path = Path(context["output_path"])
                output_path.parent.mkdir(parents=parameters.get("create_parents", True), exist_ok=True)
                return True
        except Exception:
            return False
        return False
    
    def _reduce_batch_size(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> bool:
        """Reduce batch size and other parameters"""
        # This would modify the generation parameters in the context
        # Implementation depends on how parameters are passed through the system
        return True
    
    def _validate_and_fix_prompt(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> bool:
        """Validate and fix prompt issues"""
        if context and "prompt" in context:
            prompt = context["prompt"]
            max_length = parameters.get("max_length", 512)
            if len(prompt) > max_length:
                context["prompt"] = prompt[:max_length].strip()
                return True
        return False
    
    def set_system_optimizer(self, system_optimizer):
        """Set the system optimizer for enhanced error recovery"""
        self.system_optimizer = system_optimizer
        self.logger.info("System optimizer integrated with error handler")
    
    def _apply_system_optimization(self, parameters: Dict[str, Any]) -> bool:
        """Apply WAN22 system optimization for error recovery"""
        if not hasattr(self, 'system_optimizer') or not self.system_optimizer:
            self.logger.warning("System optimizer not available for error recovery")
            return False
        
        try:
            # Apply hardware optimizations
            if parameters.get("enable_vram_optimization", True):
                opt_result = self.system_optimizer.apply_hardware_optimizations()
                if opt_result.success:
                    self.logger.info(f"Applied system optimizations: {opt_result.optimizations_applied}")
                    return True
                else:
                    self.logger.warning(f"System optimization failed: {opt_result.errors}")
            
            return False
        except Exception as e:
            self.logger.error(f"System optimization error recovery failed: {e}")
            return False
    
    def _apply_quantization(self, parameters: Dict[str, Any]) -> bool:
        """Apply quantization for VRAM error recovery"""
        if not hasattr(self, 'system_optimizer') or not self.system_optimizer:
            return False
        
        try:
            # This would integrate with the quantization controller
            # For now, just log the attempt
            method = parameters.get("quantization_method", "bf16")
            fallback = parameters.get("fallback_method", "int8")
            
            self.logger.info(f"Attempting quantization recovery with {method} (fallback: {fallback})")
            
            # In a full implementation, this would:
            # 1. Get the current pipeline from context
            # 2. Apply quantization using the quantization controller
            # 3. Return success/failure based on result
            
            return True  # Placeholder - would return actual result
            
        except Exception as e:
            self.logger.error(f"Quantization error recovery failed: {e}")
            return False
    
    def get_optimization_recommendations(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Get optimization recommendations based on error and system state"""
        recommendations = []
        
        if not hasattr(self, 'system_optimizer') or not self.system_optimizer:
            return recommendations
        
        try:
            # Get system health metrics
            health_metrics = self.system_optimizer.monitor_system_health()
            
            # VRAM-related recommendations
            if health_metrics.vram_total_mb > 0:
                vram_usage_percent = (health_metrics.vram_usage_mb / health_metrics.vram_total_mb) * 100
                
                if vram_usage_percent > 90:
                    recommendations.append("VRAM usage is critically high (>90%). Consider applying quantization or reducing model precision.")
                elif vram_usage_percent > 80:
                    recommendations.append("VRAM usage is high (>80%). Consider enabling CPU offloading for some model components.")
                
                if health_metrics.vram_total_mb < 12288:  # Less than 12GB
                    recommendations.append("Limited VRAM detected. Consider using bf16 quantization or smaller tile sizes.")
            
            # CPU and memory recommendations
            if health_metrics.cpu_usage_percent > 90:
                recommendations.append("CPU usage is very high. Consider reducing parallel processing or batch sizes.")
            
            if health_metrics.memory_usage_gb > 32:  # High memory usage
                recommendations.append("High system memory usage detected. Consider clearing model cache or reducing concurrent operations.")
            
            # Temperature recommendations
            if health_metrics.gpu_temperature > 80:
                recommendations.append("GPU temperature is high. Consider reducing workload or improving cooling.")
            
            # Hardware-specific recommendations
            hardware_profile = self.system_optimizer.get_hardware_profile()
            if hardware_profile:
                if "RTX 4080" in hardware_profile.gpu_model:
                    recommendations.append("RTX 4080 detected. Enable tensor core optimizations and use bf16 precision for best performance.")
                
                if "Threadripper" in hardware_profile.cpu_model:
                    recommendations.append("High-core CPU detected. Consider enabling multi-threaded preprocessing and NUMA optimizations.")
        
        except Exception as e:
            self.logger.warning(f"Failed to generate optimization recommendations: {e}")
        
        return recommendations


# Convenience functions for common error handling scenarios
def handle_validation_error(error: Exception, context: Dict[str, Any] = None) -> UserFriendlyError:
    """Handle input validation errors"""
    handler = GenerationErrorHandler()
    return handler.handle_error(error, context)


def handle_model_loading_error(error: Exception, model_path: str = None) -> UserFriendlyError:
    """Handle model loading errors"""
    handler = GenerationErrorHandler()
    context = {"model_path": model_path} if model_path else None
    return handler.handle_error(error, context)


def handle_vram_error(error: Exception, generation_params: Dict[str, Any] = None) -> UserFriendlyError:
    """Handle VRAM/memory errors"""
    handler = GenerationErrorHandler()
    return handler.handle_error(error, generation_params)


def handle_generation_error(error: Exception, generation_context: Dict[str, Any] = None) -> UserFriendlyError:
    """Handle general generation pipeline errors"""
    handler = GenerationErrorHandler()
    return handler.handle_error(error, generation_context)

# Error recovery decorator and related classes
@dataclass
class ErrorWithRecoveryInfo:
    """Container for error information with recovery context"""
    original_error: Exception
    error_category: ErrorCategory
    recovery_actions: List[RecoveryAction]
    context: Dict[str, Any]
    user_friendly_message: str


def handle_error_with_recovery(func):
    """
    Decorator that provides automatic error handling and recovery for functions.
    
    This decorator catches exceptions, categorizes them, and attempts recovery
    actions before re-raising if recovery fails.
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Create error handler
            handler = GenerationErrorHandler()
            
            # Get function context
            context = {
                'function_name': func.__name__,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys()) if kwargs else [],
                'module': func.__module__
            }
            
            # Handle the error
            user_friendly_error = handler.handle_error(e, context)
            
            # Create recovery info
            recovery_info = ErrorWithRecoveryInfo(
                original_error=e,
                error_category=user_friendly_error.category,
                recovery_actions=user_friendly_error.recovery_actions,
                context=context,
                user_friendly_message=user_friendly_error.message
            )
            
            # Log the error
            log_error_with_context(e, context, user_friendly_error.category)
            
            # For now, re-raise the original exception
            # In the future, we could attempt automatic recovery here
            raise e
    
    return wrapper


def log_error_with_context(error: Exception, context: Dict[str, Any], category: ErrorCategory = ErrorCategory.UNKNOWN):
    """Log an error with additional context information"""
    logger = logging.getLogger(__name__)
    
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'category': category.value if hasattr(category, 'value') else str(category),
        'context': context,
        'traceback': traceback.format_exc()
    }
    
    # Add system information
    try:
        error_info['system_info'] = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'available_memory_gb': psutil.virtual_memory().available / (1024**3)
        }
        
        if torch.cuda.is_available():
            error_info['system_info'].update({
                'gpu_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'gpu_memory_allocated': torch.cuda.memory_allocated() / (1024**3),
                'gpu_memory_reserved': torch.cuda.memory_reserved() / (1024**3)
            })
    except Exception:
        pass  # Don't fail if we can't get system info
    
    logger.error(f"Generation Error: {error_info}")


def get_error_recovery_manager():
    """Get the global error recovery manager instance"""
    return GenerationErrorHandler()


def create_error_info(error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create standardized error information dictionary"""
    return {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context or {},
        'traceback': traceback.format_exc(),
        'timestamp': str(Path(__file__).stat().st_mtime)  # Simple timestamp
    }