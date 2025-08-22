"""
UI Validation and Feedback System for Wan2.2 Video Generation

This module provides real-time validation feedback, error display, and progress
indicators for the Gradio UI components.
"""

import gradio as gr
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
import threading
import time

from input_validation import (
    PromptValidator, ImageValidator, ConfigValidator,
    ValidationResult, ValidationSeverity
)
from error_handler import (
    UserFriendlyError, ErrorCategory, ErrorSeverity as ErrorSev,
    RecoveryAction
)

logger = logging.getLogger(__name__)

@dataclass
class UIValidationState:
    """Tracks validation state for UI components"""
    is_valid: bool = True
    errors: List[str] = None
    warnings: List[str] = None
    suggestions: List[str] = None
    last_validated: Optional[datetime] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.suggestions is None:
            self.suggestions = []

class UIValidationManager:
    """Manages real-time validation and feedback for UI components"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize validators
        self.prompt_validator = PromptValidator(self.config)
        self.image_validator = ImageValidator(self.config)
        self.config_validator = ConfigValidator(self.config)
        
        # Validation state tracking
        self.validation_states: Dict[str, UIValidationState] = {}
        
        # UI component references
        self.ui_components: Dict[str, Any] = {}
        
        # Real-time validation settings
        self.enable_realtime = self.config.get("enable_realtime_validation", True)
        self.validation_delay = self.config.get("validation_delay_ms", 500)
        
        # Validation timers for debouncing
        self.validation_timers: Dict[str, threading.Timer] = {}
    
    def register_ui_components(self, components: Dict[str, Any]):
        """Register UI components for validation"""
        self.ui_components = components
        logger.info(f"Registered {len(components)} UI components for validation")
    
    def validate_prompt_realtime(self, prompt: str, model_type: str) -> Tuple[str, bool, str]:
        """
        Real-time prompt validation with immediate feedback
        Returns: (validation_html, is_valid, char_count)
        """
        try:
            # Update character count
            char_count = f"{len(prompt)}/500"
            
            if not prompt.strip():
                return self._create_empty_validation_html(), True, char_count
            
            # Validate prompt
            result = self.prompt_validator.validate_prompt(prompt, model_type)
            
            # Update validation state
            state = UIValidationState(
                is_valid=result.is_valid,
                errors=[issue.message for issue in result.get_errors()],
                warnings=[issue.message for issue in result.get_warnings()],
                suggestions=[issue.suggestion for issue in result.get_info() if issue.suggestion],
                last_validated=datetime.now()
            )
            self.validation_states['prompt'] = state
            
            # Create validation HTML
            validation_html = self._create_validation_html(result, 'prompt')
            
            return validation_html, result.is_valid, char_count
            
        except Exception as e:
            logger.error(f"Prompt validation error: {e}")
            error_html = self._create_error_html("Validation Error", str(e))
            return error_html, False, char_count
    
    def validate_image_realtime(self, image: Any, model_type: str) -> Tuple[str, bool]:
        """
        Real-time image validation with immediate feedback
        Returns: (validation_html, is_valid)
        """
        try:
            if image is None:
                return self._create_empty_validation_html(), True
            
            # Validate image
            result = self.image_validator.validate_image(image, model_type)
            
            # Update validation state
            state = UIValidationState(
                is_valid=result.is_valid,
                errors=[issue.message for issue in result.get_errors()],
                warnings=[issue.message for issue in result.get_warnings()],
                suggestions=[issue.suggestion for issue in result.get_info() if issue.suggestion],
                last_validated=datetime.now()
            )
            self.validation_states['image'] = state
            
            # Create validation HTML
            validation_html = self._create_validation_html(result, 'image')
            
            return validation_html, result.is_valid
            
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            error_html = self._create_error_html("Image Validation Error", str(e))
            return error_html, False
    
    def validate_generation_params(self, params: Dict[str, Any], model_type: str) -> Tuple[str, bool]:
        """
        Validate all generation parameters
        Returns: (validation_html, is_valid)
        """
        try:
            # Validate parameters
            result = self.config_validator.validate_generation_params(params, model_type)
            
            # Update validation state
            state = UIValidationState(
                is_valid=result.is_valid,
                errors=[issue.message for issue in result.get_errors()],
                warnings=[issue.message for issue in result.get_warnings()],
                suggestions=[issue.suggestion for issue in result.get_info() if issue.suggestion],
                last_validated=datetime.now()
            )
            self.validation_states['params'] = state
            
            # Create validation HTML
            validation_html = self._create_validation_html(result, 'parameters')
            
            return validation_html, result.is_valid
            
        except Exception as e:
            logger.error(f"Parameter validation error: {e}")
            error_html = self._create_error_html("Parameter Validation Error", str(e))
            return error_html, False
    
    def create_comprehensive_validation_summary(self) -> Tuple[str, bool]:
        """
        Create a comprehensive validation summary for all components
        Returns: (summary_html, all_valid)
        """
        try:
            all_valid = True
            total_errors = 0
            total_warnings = 0
            
            summary_parts = []
            
            for component_name, state in self.validation_states.items():
                if not state.is_valid:
                    all_valid = False
                
                total_errors += len(state.errors)
                total_warnings += len(state.warnings)
                
                if state.errors or state.warnings:
                    component_summary = self._create_component_summary(component_name, state)
                    summary_parts.append(component_summary)
            
            if all_valid and total_errors == 0 and total_warnings == 0:
                summary_html = self._create_success_summary()
            else:
                summary_html = self._create_validation_summary(
                    summary_parts, total_errors, total_warnings, all_valid
                )
            
            return summary_html, all_valid
            
        except Exception as e:
            logger.error(f"Validation summary error: {e}")
            error_html = self._create_error_html("Validation Summary Error", str(e))
            return error_html, False
    
    def create_progress_indicator(self, stage: str, progress: float, message: str) -> str:
        """
        Create HTML progress indicator with meaningful status
        """
        try:
            progress_percent = max(0, min(100, progress * 100))
            
            # Stage-specific styling
            stage_colors = {
                'validation': '#007bff',
                'model_loading': '#28a745',
                'generation': '#ffc107',
                'post_processing': '#17a2b8',
                'saving': '#6f42c1',
                'error': '#dc3545',
                'complete': '#28a745'
            }
            
            color = stage_colors.get(stage, '#6c757d')
            
            # Create animated progress bar
            progress_html = f"""
            <div class="progress-container" style="margin: 15px 0;">
                <div class="progress-header" style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-weight: bold; color: {color};">{stage.replace('_', ' ').title()}</span>
                    <span style="color: #666;">{progress_percent:.1f}%</span>
                </div>
                <div class="progress-bar-container" style="
                    width: 100%; 
                    height: 8px; 
                    background-color: #e9ecef; 
                    border-radius: 4px; 
                    overflow: hidden;
                ">
                    <div class="progress-bar" style="
                        width: {progress_percent}%; 
                        height: 100%; 
                        background: linear-gradient(90deg, {color}, {color}aa);
                        border-radius: 4px;
                        transition: width 0.3s ease;
                        {self._get_progress_animation(stage)}
                    "></div>
                </div>
                <div class="progress-message" style="
                    margin-top: 8px; 
                    font-size: 0.9em; 
                    color: #666;
                    font-style: italic;
                ">
                    {message}
                </div>
            </div>
            """
            
            return progress_html
            
        except Exception as e:
            logger.error(f"Progress indicator error: {e}")
            return f"<div>Progress: {progress_percent:.1f}% - {message}</div>"
    
    def create_error_display_with_recovery(self, error: Union[Exception, UserFriendlyError], 
                                         context: str = "") -> Tuple[str, bool]:
        """
        Create comprehensive error display with recovery suggestions
        Returns: (error_html, show_display)
        """
        try:
            if isinstance(error, UserFriendlyError):
                error_html = self._create_user_friendly_error_html(error)
            else:
                # Convert regular exception to user-friendly error
                user_error = self._convert_exception_to_user_error(error, context)
                error_html = self._create_user_friendly_error_html(user_error)
            
            return error_html, True
            
        except Exception as e:
            logger.error(f"Error display creation failed: {e}")
            fallback_html = f"""
            <div style="border: 2px solid #dc3545; border-radius: 8px; padding: 15px; margin: 10px 0; background: #f8d7da;">
                <strong style="color: #721c24;">⚠️ Error occurred: {str(error)}</strong>
            </div>
            """
            return fallback_html, True
    
    def _create_validation_html(self, result: ValidationResult, component: str) -> str:
        """Create HTML for validation results"""
        if not result.issues:
            return self._create_empty_validation_html()
        
        html_parts = []
        
        # Group issues by severity
        errors = result.get_errors()
        warnings = result.get_warnings()
        info = result.get_info()
        
        if errors:
            html_parts.append(self._create_issues_section("Errors", errors, "#dc3545", "❌"))
        
        if warnings:
            html_parts.append(self._create_issues_section("Warnings", warnings, "#ffc107", "⚠️"))
        
        if info:
            html_parts.append(self._create_issues_section("Suggestions", info, "#17a2b8", "💡"))
        
        container_style = "border-left: 4px solid #dee2e6; padding: 12px; margin: 8px 0; background: #f8f9fa; border-radius: 4px;"
        
        return f"""
        <div class="validation-feedback" style="{container_style}">
            {''.join(html_parts)}
        </div>
        """
    
    def _create_issues_section(self, title: str, issues: List, color: str, icon: str) -> str:
        """Create HTML section for validation issues"""
        if not issues:
            return ""
        
        issues_html = f"""
        <div class="validation-section" style="margin-bottom: 10px;">
            <h6 style="color: {color}; margin: 0 0 5px 0; font-size: 0.9em;">
                {icon} {title}
            </h6>
            <ul style="margin: 0; padding-left: 20px; font-size: 0.85em;">
        """
        
        for issue in issues:
            message = issue.message if hasattr(issue, 'message') else str(issue)
            suggestion = getattr(issue, 'suggestion', None)
            
            issues_html += f"<li style='margin-bottom: 3px;'>{message}"
            if suggestion:
                issues_html += f" <em style='color: #666;'>({suggestion})</em>"
            issues_html += "</li>"
        
        issues_html += "</ul></div>"
        return issues_html
    
    def _create_empty_validation_html(self) -> str:
        """Create empty validation HTML"""
        return ""
    
    def _create_error_html(self, title: str, message: str) -> str:
        """Create error HTML"""
        return f"""
        <div style="border: 2px solid #dc3545; border-radius: 8px; padding: 15px; margin: 10px 0; background: #f8d7da;">
            <strong style="color: #721c24;">{title}</strong><br>
            <span style="color: #721c24;">{message}</span>
        </div>
        """
    
    def _create_success_summary(self) -> str:
        """Create success validation summary"""
        return """
        <div style="border: 2px solid #28a745; border-radius: 8px; padding: 15px; margin: 10px 0; background: #d4edda;">
            <strong style="color: #155724;">✅ All validations passed</strong><br>
            <span style="color: #155724;">Ready to generate video</span>
        </div>
        """
    
    def _create_validation_summary(self, summary_parts: List[str], total_errors: int, 
                                 total_warnings: int, all_valid: bool) -> str:
        """Create comprehensive validation summary"""
        status_color = "#28a745" if all_valid else "#dc3545"
        status_text = "Ready" if all_valid else "Issues Found"
        status_icon = "✅" if all_valid else "❌"
        
        summary_html = f"""
        <div style="border: 2px solid {status_color}; border-radius: 8px; padding: 15px; margin: 10px 0; background: {status_color}15;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <strong style="color: {status_color};">{status_icon} Validation {status_text}</strong>
                <span style="color: #666; font-size: 0.9em;">
                    {total_errors} errors, {total_warnings} warnings
                </span>
            </div>
            {''.join(summary_parts)}
        </div>
        """
        
        return summary_html
    
    def _create_component_summary(self, component_name: str, state: UIValidationState) -> str:
        """Create summary for individual component validation"""
        component_display = component_name.replace('_', ' ').title()
        
        summary_html = f"""
        <div style="margin: 8px 0; padding: 8px; background: #fff; border-radius: 4px; border: 1px solid #dee2e6;">
            <strong style="font-size: 0.9em;">{component_display}</strong>
        """
        
        if state.errors:
            summary_html += f"""
            <div style="color: #dc3545; font-size: 0.8em; margin-top: 4px;">
                ❌ {len(state.errors)} error(s): {'; '.join(state.errors[:2])}
                {'...' if len(state.errors) > 2 else ''}
            </div>
            """
        
        if state.warnings:
            summary_html += f"""
            <div style="color: #ffc107; font-size: 0.8em; margin-top: 4px;">
                ⚠️ {len(state.warnings)} warning(s): {'; '.join(state.warnings[:2])}
                {'...' if len(state.warnings) > 2 else ''}
            </div>
            """
        
        summary_html += "</div>"
        return summary_html
    
    def _create_user_friendly_error_html(self, error: UserFriendlyError) -> str:
        """Create HTML for user-friendly error display"""
        severity_colors = {
            ErrorSev.LOW: "#28a745",
            ErrorSev.MEDIUM: "#ffc107",
            ErrorSev.HIGH: "#fd7e14", 
            ErrorSev.CRITICAL: "#dc3545"
        }
        
        severity_icons = {
            ErrorSev.LOW: "ℹ️",
            ErrorSev.MEDIUM: "⚠️",
            ErrorSev.HIGH: "🚨",
            ErrorSev.CRITICAL: "🔥"
        }
        
        color = severity_colors.get(error.severity, "#6c757d")
        icon = severity_icons.get(error.severity, "❓")
        
        html = f"""
        <div class="error-display" style="
            border: 2px solid {color}; 
            border-radius: 8px; 
            padding: 15px; 
            margin: 10px 0; 
            background: linear-gradient(135deg, {color}15, {color}05);
            animation: slideIn 0.3s ease-out;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 1.2em; margin-right: 8px;">{icon}</span>
                <strong style="color: {color}; font-size: 1.1em;">{error.title}</strong>
            </div>
            
            <p style="margin: 10px 0; color: #333;">{error.message}</p>
        """
        
        # Add recovery suggestions
        if error.recovery_suggestions:
            html += """
            <div style="margin-top: 15px;">
                <strong style="color: #007bff;">💡 Try these solutions:</strong>
                <ul style="margin: 8px 0; padding-left: 20px;">
            """
            for suggestion in error.recovery_suggestions[:3]:
                html += f"<li style='margin-bottom: 4px;'>{suggestion}</li>"
            html += "</ul></div>"
        
        # Add technical details if available
        if error.technical_details:
            html += f"""
            <details style="margin-top: 10px;">
                <summary style="cursor: pointer; color: #666; font-size: 0.9em;">
                    Technical Details
                </summary>
                <pre style="
                    background: #f8f9fa; 
                    padding: 10px; 
                    border-radius: 4px; 
                    font-size: 0.8em; 
                    overflow-x: auto;
                    margin-top: 5px;
                ">{error.technical_details}</pre>
            </details>
            """
        
        html += "</div>"
        return html
    
    def _convert_exception_to_user_error(self, error: Exception, context: str) -> UserFriendlyError:
        """Convert regular exception to user-friendly error"""
        error_str = str(error)
        error_type = type(error).__name__
        
        # Categorize error based on type and message
        category = ErrorCategory.UNKNOWN
        severity = ErrorSev.MEDIUM
        recovery_suggestions = []
        
        if "CUDA" in error_str or "GPU" in error_str or "VRAM" in error_str:
            category = ErrorCategory.VRAM_MEMORY
            severity = ErrorSev.HIGH
            recovery_suggestions = [
                "Close other GPU-intensive applications",
                "Reduce generation resolution or steps",
                "Enable VRAM optimization in settings"
            ]
        elif "model" in error_str.lower() or "loading" in error_str.lower():
            category = ErrorCategory.MODEL_LOADING
            severity = ErrorSev.HIGH
            recovery_suggestions = [
                "Check if model files exist and are not corrupted",
                "Restart the application",
                "Re-download the model if necessary"
            ]
        elif "validation" in error_str.lower() or "invalid" in error_str.lower():
            category = ErrorCategory.INPUT_VALIDATION
            severity = ErrorSev.MEDIUM
            recovery_suggestions = [
                "Check your input parameters",
                "Ensure all required fields are filled",
                "Use supported formats and values"
            ]
        
        return UserFriendlyError(
            category=category,
            severity=severity,
            title=f"{error_type} Error",
            message=error_str,
            recovery_suggestions=recovery_suggestions,
            recovery_actions=[],
            technical_details=f"Context: {context}\nError Type: {error_type}",
            error_code=error_type
        )
    
    def _get_progress_animation(self, stage: str) -> str:
        """Get CSS animation for progress bar based on stage"""
        if stage in ['generation', 'model_loading']:
            return """
            animation: pulse 2s infinite;
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.7; }
                100% { opacity: 1; }
            }
            """
        return ""

# Global validation manager instance
_validation_manager = None

def get_validation_manager(config: Optional[Dict[str, Any]] = None) -> UIValidationManager:
    """Get or create global validation manager instance"""
    global _validation_manager
    if _validation_manager is None:
        _validation_manager = UIValidationManager(config)
    return _validation_manager