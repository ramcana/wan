"""
Comprehensive Error Messaging System for Wan Model Compatibility

This module provides user-friendly error messages with progressive disclosure,
specific guidance for common compatibility issues, and actionable recovery steps.

Requirements covered: 1.4, 2.4, 3.4, 4.4, 6.4, 7.4
"""

import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import traceback
import sys
import platform


class ErrorSeverity(Enum):
    """Error severity levels for progressive disclosure"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for specific handling"""
    ARCHITECTURE_DETECTION = "architecture_detection"
    PIPELINE_LOADING = "pipeline_loading"
    VAE_COMPATIBILITY = "vae_compatibility"
    PIPELINE_INITIALIZATION = "pipeline_initialization"
    RESOURCE_CONSTRAINTS = "resource_constraints"
    DEPENDENCY_MANAGEMENT = "dependency_management"
    VIDEO_PROCESSING = "video_processing"
    SECURITY_VALIDATION = "security_validation"


@dataclass
class ErrorContext:
    """Context information for error analysis"""
    model_path: Optional[str] = None
    model_name: Optional[str] = None
    pipeline_class: Optional[str] = None
    system_info: Optional[Dict[str, Any]] = None
    attempted_operation: Optional[str] = None
    user_inputs: Optional[Dict[str, Any]] = None


@dataclass
class RecoveryAction:
    """Actionable recovery step"""
    title: str
    description: str
    command: Optional[str] = None
    url: Optional[str] = None
    priority: int = 1  # 1=high, 2=medium, 3=low
    estimated_time: Optional[str] = None
    requires_restart: bool = False


@dataclass
class ErrorMessage:
    """Comprehensive error message with progressive disclosure"""
    # Basic level - always shown
    title: str
    summary: str
    severity: ErrorSeverity
    category: ErrorCategory
    
    # Detailed level - shown on request
    detailed_description: Optional[str] = None
    technical_details: Optional[str] = None
    
    # Diagnostic level - for developers/advanced users
    stack_trace: Optional[str] = None
    system_context: Optional[Dict[str, Any]] = None
    debug_info: Optional[Dict[str, Any]] = None
    
    # Recovery guidance
    recovery_actions: List[RecoveryAction] = None
    related_errors: List[str] = None
    documentation_links: List[str] = None
    
    # Metadata
    error_code: Optional[str] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.recovery_actions is None:
            self.recovery_actions = []
        if self.related_errors is None:
            self.related_errors = []
        if self.documentation_links is None:
            self.documentation_links = []


class ErrorMessageGenerator:
    """Generate user-friendly error messages for different failure types"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._load_error_templates()
    
    def _load_error_templates(self):
        """Load error message templates"""
        self.templates = {
            # Architecture Detection Errors
            "unrecognized_model_format": {
                "title": "Model Format Not Recognized",
                "summary": "The model format could not be automatically detected",
                "category": ErrorCategory.ARCHITECTURE_DETECTION,
                "severity": ErrorSeverity.WARNING
            },
            "corrupted_model_index": {
                "title": "Model Configuration Corrupted",
                "summary": "The model_index.json file is corrupted or missing",
                "category": ErrorCategory.ARCHITECTURE_DETECTION,
                "severity": ErrorSeverity.ERROR
            },
            "missing_components": {
                "title": "Model Components Missing",
                "summary": "Required model components are missing or inaccessible",
                "category": ErrorCategory.ARCHITECTURE_DETECTION,
                "severity": ErrorSeverity.ERROR
            },
            
            # Pipeline Loading Errors
            "missing_pipeline_class": {
                "title": "Custom Pipeline Not Available",
                "summary": "The required WanPipeline class is not installed",
                "category": ErrorCategory.PIPELINE_LOADING,
                "severity": ErrorSeverity.ERROR
            },
            "pipeline_args_mismatch": {
                "title": "Pipeline Arguments Invalid",
                "summary": "The pipeline initialization arguments are incorrect",
                "category": ErrorCategory.PIPELINE_INITIALIZATION,
                "severity": ErrorSeverity.ERROR
            },
            "remote_code_blocked": {
                "title": "Remote Code Execution Blocked",
                "summary": "Security settings prevent downloading required pipeline code",
                "category": ErrorCategory.SECURITY_VALIDATION,
                "severity": ErrorSeverity.WARNING
            },
            
            # VAE Compatibility Errors
            "vae_shape_mismatch": {
                "title": "VAE Architecture Incompatible",
                "summary": "The VAE component has incompatible dimensions for this model",
                "category": ErrorCategory.VAE_COMPATIBILITY,
                "severity": ErrorSeverity.ERROR
            },
            "vae_random_init": {
                "title": "VAE Weights Not Loaded",
                "summary": "VAE fell back to random initialization instead of loading weights",
                "category": ErrorCategory.VAE_COMPATIBILITY,
                "severity": ErrorSeverity.CRITICAL
            },
            
            # Resource Constraint Errors
            "insufficient_vram": {
                "title": "Insufficient GPU Memory",
                "summary": "Not enough VRAM available for this model",
                "category": ErrorCategory.RESOURCE_CONSTRAINTS,
                "severity": ErrorSeverity.ERROR
            },
            "memory_allocation_failed": {
                "title": "Memory Allocation Failed",
                "summary": "System ran out of memory during model loading",
                "category": ErrorCategory.RESOURCE_CONSTRAINTS,
                "severity": ErrorSeverity.CRITICAL
            },
            
            # Dependency Management Errors
            "dependency_missing": {
                "title": "Required Dependencies Missing",
                "summary": "Some required packages are not installed",
                "category": ErrorCategory.DEPENDENCY_MANAGEMENT,
                "severity": ErrorSeverity.ERROR
            },
            "version_conflict": {
                "title": "Package Version Conflict",
                "summary": "Installed package versions are incompatible",
                "category": ErrorCategory.DEPENDENCY_MANAGEMENT,
                "severity": ErrorSeverity.ERROR
            },
            
            # Video Processing Errors
            "frame_processing_failed": {
                "title": "Video Frame Processing Failed",
                "summary": "Error occurred while processing generated video frames",
                "category": ErrorCategory.VIDEO_PROCESSING,
                "severity": ErrorSeverity.ERROR
            },
            "encoding_failed": {
                "title": "Video Encoding Failed",
                "summary": "Could not encode frames to video file",
                "category": ErrorCategory.VIDEO_PROCESSING,
                "severity": ErrorSeverity.WARNING
            },
            "ffmpeg_missing": {
                "title": "FFmpeg Not Available",
                "summary": "FFmpeg is required for video encoding but not found",
                "category": ErrorCategory.VIDEO_PROCESSING,
                "severity": ErrorSeverity.WARNING
            }
        }
    
    def generate_error_message(self, 
                             error_type: str,
                             context: ErrorContext,
                             original_exception: Optional[Exception] = None) -> ErrorMessage:
        """Generate comprehensive error message for specific error type"""
        
        template = self.templates.get(error_type, {})
        if not template:
            return self._generate_generic_error(error_type, context, original_exception)
        
        # Create base error message from template
        error_msg = ErrorMessage(
            title=template["title"],
            summary=template["summary"],
            category=template["category"],
            severity=template["severity"],
            error_code=error_type,
            timestamp=self._get_timestamp()
        )
        
        # Add context-specific details
        self._add_detailed_description(error_msg, error_type, context)
        self._add_technical_details(error_msg, context, original_exception)
        self._add_recovery_actions(error_msg, error_type, context)
        self._add_diagnostic_info(error_msg, context, original_exception)
        
        return error_msg
    
    def _generate_generic_error(self, 
                              error_type: str,
                              context: ErrorContext,
                              original_exception: Optional[Exception]) -> ErrorMessage:
        """Generate generic error message for unknown error types"""
        
        return ErrorMessage(
            title="Unexpected Error",
            summary=f"An unexpected error occurred: {error_type}",
            category=ErrorCategory.ARCHITECTURE_DETECTION,
            severity=ErrorSeverity.ERROR,
            detailed_description=str(original_exception) if original_exception else None,
            technical_details=traceback.format_exc() if original_exception else None,
            error_code=error_type,
            timestamp=self._get_timestamp(),
            recovery_actions=[
                RecoveryAction(
                    title="Check System Requirements",
                    description="Verify that your system meets the minimum requirements",
                    priority=1
                ),
                RecoveryAction(
                    title="Report Issue",
                    description="Report this error to the development team",
                    url="https://github.com/wan-ai/wan22/issues",
                    priority=3
                )
            ]
        )
    
    def _add_detailed_description(self, error_msg: ErrorMessage, error_type: str, context: ErrorContext):
        """Add detailed description based on error type and context"""
        
        descriptions = {
            "missing_pipeline_class": f"""
The Wan model at '{context.model_path}' requires a custom WanPipeline class that is not currently available in your environment. 

Wan models use specialized 3D transformers and custom VAE components that are incompatible with standard Diffusers pipelines. The WanPipeline class handles these custom components properly.

This typically happens when:
- The WanPipeline package is not installed
- The package version is incompatible with the model
- Remote code downloading is disabled
            """.strip(),
            
            "vae_shape_mismatch": f"""
The VAE (Variational Autoencoder) component in this Wan model has a 3D architecture with shape dimensions that don't match standard 2D VAE expectations.

Model: {context.model_name or 'Unknown'}
Expected: 3D VAE with shape [384, ...] for video generation
Found: Incompatible VAE dimensions

This mismatch can cause the VAE to fall back to random weight initialization, which will produce poor quality or corrupted video output.
            """.strip(),
            
            "insufficient_vram": f"""
The model requires more GPU memory (VRAM) than is currently available on your system.

Model: {context.model_name or 'Unknown'}
Estimated VRAM needed: 12-16 GB for full quality
Available VRAM: {self._get_available_vram()} GB

Large video generation models like Wan 2.2 require significant GPU memory due to their 3D transformer architecture and temporal processing requirements.
            """.strip(),
            
            "remote_code_blocked": f"""
The model requires downloading and executing custom pipeline code from Hugging Face, but this is currently blocked by security settings.

Model: {context.model_name or 'Unknown'}
Pipeline source: {context.pipeline_class or 'WanPipeline'}

Remote code execution is disabled either by:
- trust_remote_code=False setting
- System security policies
- Network restrictions
            """.strip()
        }
        
        error_msg.detailed_description = descriptions.get(error_type, 
            f"Detailed information for {error_type} is not available.")
    
    def _add_technical_details(self, error_msg: ErrorMessage, context: ErrorContext, exception: Optional[Exception]):
        """Add technical details for developers"""
        
        details = []
        
        if context.model_path:
            details.append(f"Model Path: {context.model_path}")
        if context.pipeline_class:
            details.append(f"Pipeline Class: {context.pipeline_class}")
        if context.attempted_operation:
            details.append(f"Operation: {context.attempted_operation}")
        
        if exception:
            details.append(f"Exception Type: {type(exception).__name__}")
            details.append(f"Exception Message: {str(exception)}")
        
        if context.system_info:
            details.append("System Information:")
            for key, value in context.system_info.items():
                details.append(f"  {key}: {value}")
        
        error_msg.technical_details = "\n".join(details) if details else None
    
    def _add_recovery_actions(self, error_msg: ErrorMessage, error_type: str, context: ErrorContext):
        """Add specific recovery actions based on error type"""
        
        recovery_actions = {
            "missing_pipeline_class": [
                RecoveryAction(
                    title="Install WanPipeline Package",
                    description="Install the required WanPipeline package",
                    command="pip install wan-pipeline",
                    priority=1,
                    estimated_time="2-5 minutes"
                ),
                RecoveryAction(
                    title="Enable Remote Code Download",
                    description="Allow automatic downloading of pipeline code",
                    priority=2,
                    estimated_time="1 minute"
                ),
                RecoveryAction(
                    title="Manual Pipeline Installation",
                    description="Download and install pipeline code manually",
                    url="https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                    priority=3,
                    estimated_time="5-10 minutes"
                )
            ],
            
            "vae_shape_mismatch": [
                RecoveryAction(
                    title="Update Model Components",
                    description="Download the correct VAE component for this model",
                    priority=1,
                    estimated_time="5-10 minutes"
                ),
                RecoveryAction(
                    title="Use Compatible Model Version",
                    description="Switch to a model version with compatible VAE",
                    priority=2
                ),
                RecoveryAction(
                    title="Force VAE Compatibility",
                    description="Override VAE loading with compatibility mode (may reduce quality)",
                    priority=3
                )
            ],
            
            "insufficient_vram": [
                RecoveryAction(
                    title="Enable CPU Offloading",
                    description="Move some model components to CPU to reduce VRAM usage",
                    priority=1,
                    estimated_time="1 minute"
                ),
                RecoveryAction(
                    title="Use Mixed Precision",
                    description="Enable half-precision (fp16) to reduce memory usage",
                    priority=1,
                    estimated_time="1 minute"
                ),
                RecoveryAction(
                    title="Enable Chunked Processing",
                    description="Process video frames in smaller chunks",
                    priority=2,
                    estimated_time="1 minute"
                ),
                RecoveryAction(
                    title="Close Other Applications",
                    description="Free up GPU memory by closing other GPU-intensive applications",
                    priority=2,
                    estimated_time="2-5 minutes"
                ),
                RecoveryAction(
                    title="Use Smaller Model",
                    description="Switch to a smaller Wan model variant (e.g., Wan2.2-Mini)",
                    priority=3
                )
            ],
            
            "remote_code_blocked": [
                RecoveryAction(
                    title="Enable Trust Remote Code",
                    description="Allow downloading of custom pipeline code",
                    priority=1,
                    estimated_time="1 minute"
                ),
                RecoveryAction(
                    title="Install Pipeline Locally",
                    description="Download and install the pipeline code manually",
                    command="git clone https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                    priority=2,
                    estimated_time="5-10 minutes"
                ),
                RecoveryAction(
                    title="Use Pre-installed Pipeline",
                    description="Install WanPipeline package instead of remote code",
                    command="pip install wan-pipeline",
                    priority=2,
                    estimated_time="2-5 minutes"
                )
            ],
            
            "encoding_failed": [
                RecoveryAction(
                    title="Install FFmpeg",
                    description="Install FFmpeg for video encoding support",
                    priority=1,
                    estimated_time="5-10 minutes",
                    url="https://ffmpeg.org/download.html"
                ),
                RecoveryAction(
                    title="Use Frame Sequence Output",
                    description="Save individual frames instead of encoded video",
                    priority=2,
                    estimated_time="1 minute"
                ),
                RecoveryAction(
                    title="Try Different Video Format",
                    description="Use a different video format (MP4, WebM, AVI)",
                    priority=2,
                    estimated_time="1 minute"
                )
            ],
            
            "dependency_missing": [
                RecoveryAction(
                    title="Install Missing Dependencies",
                    description="Install all required packages",
                    command="pip install -r requirements.txt",
                    priority=1,
                    estimated_time="5-15 minutes"
                ),
                RecoveryAction(
                    title="Update Package Manager",
                    description="Update pip to the latest version",
                    command="pip install --upgrade pip",
                    priority=2,
                    estimated_time="2-5 minutes"
                ),
                RecoveryAction(
                    title="Use Virtual Environment",
                    description="Create a clean virtual environment",
                    priority=3,
                    estimated_time="10-20 minutes",
                    requires_restart=True
                )
            ]
        }
        
        error_msg.recovery_actions = recovery_actions.get(error_type, [
            RecoveryAction(
                title="Check Documentation",
                description="Consult the documentation for troubleshooting steps",
                url="https://github.com/wan-ai/wan22/docs",
                priority=1
            )
        ])
    
    def _add_diagnostic_info(self, error_msg: ErrorMessage, context: ErrorContext, exception: Optional[Exception]):
        """Add diagnostic information for advanced troubleshooting"""
        
        diagnostic_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "context": asdict(context) if context else None
        }
        
        if exception:
            diagnostic_info["stack_trace"] = traceback.format_exc()
            diagnostic_info["exception_args"] = getattr(exception, 'args', None)
        
        error_msg.debug_info = diagnostic_info
        error_msg.system_context = self._collect_system_context()
    
    def _collect_system_context(self) -> Dict[str, Any]:
        """Collect system context information"""
        try:
            import torch
            import diffusers
            
            context = {
                "torch_version": torch.__version__,
                "torch_cuda_available": torch.cuda.is_available(),
                "diffusers_version": diffusers.__version__,
            }
            
            if torch.cuda.is_available():
                context.update({
                    "cuda_version": torch.version.cuda,
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    "total_vram": [torch.cuda.get_device_properties(i).total_memory // (1024**3) 
                                 for i in range(torch.cuda.device_count())]
                })
            
            return context
        except Exception as e:
            return {"error_collecting_context": str(e)}
    
    def _get_available_vram(self) -> str:
        """Get available VRAM information"""
        try:
            import torch
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                return f"{total}"
            return "N/A (No CUDA)"
        except:
            return "Unknown"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


class ErrorMessageFormatter:
    """Format error messages for different output contexts"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def format_for_console(self, error_msg: ErrorMessage, detail_level: str = "basic") -> str:
        """Format error message for console output"""
        
        lines = []
        
        # Header with severity indicator
        severity_icons = {
            ErrorSeverity.INFO: "ℹ️",
            ErrorSeverity.WARNING: "⚠️",
            ErrorSeverity.ERROR: "❌",
            ErrorSeverity.CRITICAL: "🚨"
        }
        
        icon = severity_icons.get(error_msg.severity, "❓")
        lines.append(f"{icon} {error_msg.title}")
        lines.append("=" * (len(error_msg.title) + 3))
        lines.append("")
        
        # Basic information
        lines.append(error_msg.summary)
        lines.append("")
        
        if detail_level in ["detailed", "diagnostic"] and error_msg.detailed_description:
            lines.append("Details:")
            lines.append("-" * 8)
            lines.append(error_msg.detailed_description)
            lines.append("")
        
        # Recovery actions
        if error_msg.recovery_actions:
            lines.append("Recommended Actions:")
            lines.append("-" * 20)
            for i, action in enumerate(sorted(error_msg.recovery_actions, key=lambda x: x.priority), 1):
                lines.append(f"{i}. {action.title}")
                lines.append(f"   {action.description}")
                if action.command:
                    lines.append(f"   Command: {action.command}")
                if action.estimated_time:
                    lines.append(f"   Estimated time: {action.estimated_time}")
                lines.append("")
        
        # Technical details for diagnostic level
        if detail_level == "diagnostic":
            if error_msg.technical_details:
                lines.append("Technical Details:")
                lines.append("-" * 17)
                lines.append(error_msg.technical_details)
                lines.append("")
            
            if error_msg.debug_info:
                lines.append("Debug Information:")
                lines.append("-" * 17)
                lines.append(json.dumps(error_msg.debug_info, indent=2))
                lines.append("")
        
        return "\n".join(lines)
    
    def format_for_ui(self, error_msg: ErrorMessage) -> Dict[str, Any]:
        """Format error message for UI display"""
        
        return {
            "title": error_msg.title,
            "summary": error_msg.summary,
            "severity": error_msg.severity.value,
            "category": error_msg.category.value,
            "detailed_description": error_msg.detailed_description,
            "recovery_actions": [
                {
                    "title": action.title,
                    "description": action.description,
                    "command": action.command,
                    "url": action.url,
                    "priority": action.priority,
                    "estimated_time": action.estimated_time,
                    "requires_restart": action.requires_restart
                }
                for action in error_msg.recovery_actions
            ],
            "error_code": error_msg.error_code,
            "timestamp": error_msg.timestamp
        }
    
    def format_for_json(self, error_msg: ErrorMessage, include_diagnostics: bool = False) -> str:
        """Format error message as JSON"""
        
        data = asdict(error_msg)
        
        if not include_diagnostics:
            # Remove diagnostic information for basic JSON output
            data.pop("stack_trace", None)
            data.pop("system_context", None)
            data.pop("debug_info", None)
        
        return json.dumps(data, indent=2, default=str)


class ProgressiveErrorDisclosure:
    """Handle progressive error disclosure (basic → detailed → diagnostic)"""
    
    def __init__(self):
        self.generator = ErrorMessageGenerator()
        self.formatter = ErrorMessageFormatter()
        self.logger = logging.getLogger(__name__)
    
    def handle_error(self, 
                    error_type: str,
                    context: ErrorContext,
                    original_exception: Optional[Exception] = None,
                    output_format: str = "console",
                    detail_level: str = "basic") -> Union[str, Dict[str, Any]]:
        """Handle error with progressive disclosure"""
        
        # Generate comprehensive error message
        error_msg = self.generator.generate_error_message(error_type, context, original_exception)
        
        # Log the error
        self._log_error(error_msg, original_exception)
        
        # Format for requested output
        if output_format == "console":
            return self.formatter.format_for_console(error_msg, detail_level)
        elif output_format == "ui":
            return self.formatter.format_for_ui(error_msg)
        elif output_format == "json":
            return self.formatter.format_for_json(error_msg, detail_level == "diagnostic")
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _log_error(self, error_msg: ErrorMessage, original_exception: Optional[Exception]):
        """Log error message appropriately"""
        
        log_level = {
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error_msg.severity, logging.ERROR)
        
        log_message = f"{error_msg.title}: {error_msg.summary}"
        
        if original_exception:
            self.logger.log(log_level, log_message, exc_info=True)
        else:
            self.logger.log(log_level, log_message)


# Convenience functions for common error scenarios
def create_architecture_error(model_path: str, error_type: str, exception: Optional[Exception] = None) -> str:
    """Create architecture detection error message"""
    context = ErrorContext(
        model_path=model_path,
        attempted_operation="architecture_detection"
    )
    
    disclosure = ProgressiveErrorDisclosure()
    return disclosure.handle_error(error_type, context, exception)


def create_pipeline_error(model_path: str, pipeline_class: str, error_type: str, exception: Optional[Exception] = None) -> str:
    """Create pipeline loading error message"""
    context = ErrorContext(
        model_path=model_path,
        pipeline_class=pipeline_class,
        attempted_operation="pipeline_loading"
    )
    
    disclosure = ProgressiveErrorDisclosure()
    return disclosure.handle_error(error_type, context, exception)


def create_vae_error(model_path: str, error_type: str, exception: Optional[Exception] = None) -> str:
    """Create VAE compatibility error message"""
    context = ErrorContext(
        model_path=model_path,
        attempted_operation="vae_loading"
    )
    
    disclosure = ProgressiveErrorDisclosure()
    return disclosure.handle_error(error_type, context, exception)


def create_resource_error(error_type: str, system_info: Dict[str, Any], exception: Optional[Exception] = None) -> str:
    """Create resource constraint error message"""
    context = ErrorContext(
        system_info=system_info,
        attempted_operation="resource_allocation"
    )
    
    disclosure = ProgressiveErrorDisclosure()
    return disclosure.handle_error(error_type, context, exception)


def create_dependency_error(error_type: str, missing_deps: List[str], exception: Optional[Exception] = None) -> str:
    """Create dependency management error message"""
    context = ErrorContext(
        user_inputs={"missing_dependencies": missing_deps},
        attempted_operation="dependency_management"
    )
    
    disclosure = ProgressiveErrorDisclosure()
    return disclosure.handle_error(error_type, context, exception)


def create_video_error(error_type: str, output_path: str, exception: Optional[Exception] = None) -> str:
    """Create video processing error message"""
    context = ErrorContext(
        user_inputs={"output_path": output_path},
        attempted_operation="video_processing"
    )
    
    disclosure = ProgressiveErrorDisclosure()
    return disclosure.handle_error(error_type, context, exception)


class ErrorGuidanceSystem:
    """Provide contextual guidance for error resolution"""
    
    def __init__(self):
        self.generator = ErrorMessageGenerator()
        self.formatter = ErrorMessageFormatter()
        self.logger = logging.getLogger(__name__)
    
    def get_guided_resolution(self, error_type: str, context: ErrorContext) -> Dict[str, Any]:
        """Get step-by-step guided resolution for an error"""
        
        error_msg = self.generator.generate_error_message(error_type, context)
        
        # Create guided resolution steps
        guided_steps = []
        for action in sorted(error_msg.recovery_actions, key=lambda x: x.priority):
            step = {
                "title": action.title,
                "description": action.description,
                "type": self._determine_step_type(action),
                "command": action.command,
                "url": action.url,
                "estimated_time": action.estimated_time,
                "requires_restart": action.requires_restart,
                "validation": self._get_step_validation(action, error_type)
            }
            guided_steps.append(step)
        
        return {
            "error_summary": error_msg.summary,
            "guided_steps": guided_steps,
            "alternative_solutions": self._get_alternative_solutions(error_type),
            "prevention_tips": self._get_prevention_tips(error_type)
        }
    
    def _determine_step_type(self, action: RecoveryAction) -> str:
        """Determine the type of recovery step"""
        if action.command:
            return "command"
        elif action.url:
            return "download"
        elif "install" in action.title.lower():
            return "installation"
        elif "enable" in action.title.lower() or "disable" in action.title.lower():
            return "configuration"
        else:
            return "manual"
    
    def _get_step_validation(self, action: RecoveryAction, error_type: str) -> Dict[str, str]:
        """Get validation criteria for a recovery step"""
        
        validations = {
            "missing_pipeline_class": {
                "Install WanPipeline Package": "python -c 'import wan_pipeline; print(\"Success\")'",
                "Enable Remote Code Download": "Check that trust_remote_code=True in configuration"
            },
            "insufficient_vram": {
                "Enable CPU Offloading": "Check that model components are moved to CPU",
                "Use Mixed Precision": "Verify fp16 is enabled in pipeline configuration"
            },
            "encoding_failed": {
                "Install FFmpeg": "ffmpeg -version",
                "Use Frame Sequence Output": "Check that individual frames are saved"
            }
        }
        
        error_validations = validations.get(error_type, {})
        return {"validation_command": error_validations.get(action.title, "Manual verification required")}
    
    def _get_alternative_solutions(self, error_type: str) -> List[Dict[str, str]]:
        """Get alternative solutions for common errors"""
        
        alternatives = {
            "missing_pipeline_class": [
                {
                    "title": "Use Docker Container",
                    "description": "Run the application in a pre-configured Docker container with all dependencies"
                },
                {
                    "title": "Use Cloud Service",
                    "description": "Use a cloud-based inference service that handles dependencies automatically"
                }
            ],
            "insufficient_vram": [
                {
                    "title": "Use Cloud GPU",
                    "description": "Rent a cloud GPU with more VRAM for video generation"
                },
                {
                    "title": "Batch Processing",
                    "description": "Process multiple shorter videos instead of one long video"
                }
            ],
            "vae_shape_mismatch": [
                {
                    "title": "Use Different Model Variant",
                    "description": "Try a different version of the Wan model with compatible VAE"
                },
                {
                    "title": "Custom VAE Conversion",
                    "description": "Convert the VAE to the expected format using conversion tools"
                }
            ]
        }
        
        return alternatives.get(error_type, [])
    
    def _get_prevention_tips(self, error_type: str) -> List[str]:
        """Get tips to prevent similar errors in the future"""
        
        tips = {
            "missing_pipeline_class": [
                "Always check model requirements before downloading",
                "Keep a list of installed pipeline packages",
                "Use virtual environments for different model types"
            ],
            "insufficient_vram": [
                "Monitor VRAM usage during generation",
                "Close unnecessary applications before generation",
                "Consider upgrading GPU for regular video generation"
            ],
            "dependency_missing": [
                "Use requirements.txt files for consistent environments",
                "Regularly update package dependencies",
                "Test installations in clean environments"
            ],
            "encoding_failed": [
                "Install video codecs during initial setup",
                "Test video encoding with sample files",
                "Keep backup encoding options available"
            ]
        }
        
        return tips.get(error_type, ["Check documentation for best practices"])


class ErrorAnalytics:
    """Track and analyze error patterns for system improvement"""
    
    def __init__(self, analytics_file: str = "error_analytics.json"):
        self.analytics_file = Path(analytics_file)
        self.analytics_data = self._load_analytics()
    
    def _load_analytics(self) -> Dict[str, Any]:
        """Load existing analytics data"""
        if self.analytics_file.exists():
            try:
                with open(self.analytics_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            "error_counts": {},
            "resolution_success": {},
            "common_contexts": {},
            "user_feedback": []
        }
    
    def record_error(self, error_type: str, context: ErrorContext, resolved: bool = False):
        """Record an error occurrence"""
        
        # Update error counts
        self.analytics_data["error_counts"][error_type] = \
            self.analytics_data["error_counts"].get(error_type, 0) + 1
        
        # Track resolution success
        if error_type not in self.analytics_data["resolution_success"]:
            self.analytics_data["resolution_success"][error_type] = {"attempts": 0, "successes": 0}
        
        self.analytics_data["resolution_success"][error_type]["attempts"] += 1
        if resolved:
            self.analytics_data["resolution_success"][error_type]["successes"] += 1
        
        # Track common contexts
        if context.model_path:
            model_key = Path(context.model_path).name
            if model_key not in self.analytics_data["common_contexts"]:
                self.analytics_data["common_contexts"][model_key] = {}
            self.analytics_data["common_contexts"][model_key][error_type] = \
                self.analytics_data["common_contexts"][model_key].get(error_type, 0) + 1
        
        self._save_analytics()
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for analysis"""
        
        total_errors = sum(self.analytics_data["error_counts"].values())
        
        stats = {
            "total_errors": total_errors,
            "most_common_errors": sorted(
                self.analytics_data["error_counts"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "resolution_rates": {
                error_type: data["successes"] / max(data["attempts"], 1)
                for error_type, data in self.analytics_data["resolution_success"].items()
            },
            "problematic_models": self._get_problematic_models()
        }
        
        return stats
    
    def _get_problematic_models(self) -> List[Dict[str, Any]]:
        """Get models with highest error rates"""
        
        model_errors = []
        for model, errors in self.analytics_data["common_contexts"].items():
            total_errors = sum(errors.values())
            model_errors.append({
                "model": model,
                "total_errors": total_errors,
                "error_types": errors
            })
        
        return sorted(model_errors, key=lambda x: x["total_errors"], reverse=True)[:5]
    
    def _save_analytics(self):
        """Save analytics data to file"""
        try:
            with open(self.analytics_file, 'w') as f:
                json.dump(self.analytics_data, f, indent=2)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to save analytics: {e}")


# Integration with existing error handling
class EnhancedErrorHandler:
    """Enhanced error handler with comprehensive messaging"""
    
    def __init__(self):
        self.disclosure = ProgressiveErrorDisclosure()
        self.guidance = ErrorGuidanceSystem()
        self.analytics = ErrorAnalytics()
        self.logger = logging.getLogger(__name__)
    
    def handle_compatibility_error(self, 
                                 error_type: str,
                                 context: ErrorContext,
                                 exception: Optional[Exception] = None,
                                 interactive: bool = True) -> Dict[str, Any]:
        """Handle compatibility error with full messaging system"""
        
        # Record the error
        self.analytics.record_error(error_type, context)
        
        # Generate error message
        error_message = self.disclosure.handle_error(
            error_type, context, exception, output_format="ui"
        )
        
        # Get guided resolution if interactive
        guided_resolution = None
        if interactive:
            guided_resolution = self.guidance.get_guided_resolution(error_type, context)
        
        # Prepare comprehensive response
        response = {
            "error_message": error_message,
            "guided_resolution": guided_resolution,
            "error_id": f"{error_type}_{hash(str(context))}",
            "support_info": {
                "documentation_url": "https://github.com/wan-ai/wan22/docs/troubleshooting",
                "community_forum": "https://github.com/wan-ai/wan22/discussions",
                "issue_tracker": "https://github.com/wan-ai/wan22/issues"
            }
        }
        
        return response
    
    def mark_error_resolved(self, error_type: str, context: ErrorContext):
        """Mark an error as resolved for analytics"""
        self.analytics.record_error(error_type, context, resolved=True)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health based on error patterns"""
        
        stats = self.analytics.get_error_statistics()
        
        # Calculate health score (0-100)
        total_errors = stats["total_errors"]
        avg_resolution_rate = sum(stats["resolution_rates"].values()) / max(len(stats["resolution_rates"]), 1)
        
        health_score = max(0, 100 - (total_errors * 2) + (avg_resolution_rate * 20))
        
        return {
            "health_score": min(100, health_score),
            "error_statistics": stats,
            "recommendations": self._get_health_recommendations(stats)
        }
    
    def _get_health_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Get recommendations based on error patterns"""
        
        recommendations = []
        
        # Check for common errors
        if stats["most_common_errors"]:
            most_common = stats["most_common_errors"][0]
            if most_common[1] > 5:  # More than 5 occurrences
                recommendations.append(
                    f"Consider addressing the most common error: {most_common[0]}"
                )
        
        # Check resolution rates
        low_resolution_errors = [
            error_type for error_type, rate in stats["resolution_rates"].items()
            if rate < 0.5
        ]
        
        if low_resolution_errors:
            recommendations.append(
                f"Improve documentation for errors with low resolution rates: {', '.join(low_resolution_errors)}"
            )
        
        # Check problematic models
        if stats.get("problematic_models"):
            problematic = stats["problematic_models"][0]
            if problematic["total_errors"] > 10:
                recommendations.append(
                    f"Review compatibility for model: {problematic['model']}"
                )
        
        return recommendations or ["System is operating normally"]