#!/usr/bin/env python3
"""
UI Error Recovery and Fallback System for WAN22 UI
Provides comprehensive error recovery, component recreation, and fallback mechanisms
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import gradio as gr
import traceback
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ComponentRecoveryInfo:
    """Information about component recovery attempts"""
    component_name: str
    original_type: str
    recovery_attempts: int = 0
    max_attempts: int = 3
    last_error: Optional[str] = None
    recovery_timestamp: Optional[datetime] = None
    fallback_used: bool = False
    fallback_type: Optional[str] = None

@dataclass
class UIFallbackConfig:
    """Configuration for UI fallback mechanisms"""
    enable_component_recreation: bool = True
    enable_fallback_components: bool = True
    enable_simplified_ui: bool = True
    max_recovery_attempts: int = 3
    log_recovery_attempts: bool = True
    show_user_guidance: bool = True

@dataclass
class RecoveryGuidance:
    """User guidance for error recovery"""
    title: str
    message: str
    suggestions: List[str] = field(default_factory=list)
    severity: str = "warning"  # info, warning, error, critical
    show_technical_details: bool = False
    technical_details: Optional[str] = None

class UIErrorRecoveryManager:
    """Manages error recovery and fallback mechanisms for UI components"""
    
    def __init__(self, config: Optional[UIFallbackConfig] = None):
        self.config = config or UIFallbackConfig()
        self.recovery_info: Dict[str, ComponentRecoveryInfo] = {}
        self.fallback_components: Dict[str, Any] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize fallback component creators
        self._init_fallback_creators()
    
    def _init_fallback_creators(self):
        """Initialize fallback component creation functions"""
        self.fallback_creators = {
            'Textbox': self._create_fallback_textbox,
            'Button': self._create_fallback_button,
            'Dropdown': self._create_fallback_dropdown,
            'Slider': self._create_fallback_slider,
            'Image': self._create_fallback_image,
            'Video': self._create_fallback_video,
            'File': self._create_fallback_file,
            'HTML': self._create_fallback_html,
            'Markdown': self._create_fallback_markdown,
            'Gallery': self._create_fallback_gallery,
            'DataFrame': self._create_fallback_dataframe,
            'Plot': self._create_fallback_plot,
        } 
   
    def recreate_component(self, 
                          component_name: str, 
                          component_type: str, 
                          original_kwargs: Dict[str, Any],
                          fallback_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], RecoveryGuidance]:
        """
        Attempt to recreate a failed component with fallback options
        
        Args:
            component_name: Name of the component to recreate
            component_type: Type of the component (e.g., 'Textbox', 'Button')
            original_kwargs: Original arguments used to create the component
            fallback_kwargs: Alternative arguments to try if original fails
            
        Returns:
            Tuple of (recreated_component, recovery_guidance)
        """
        # Get or create recovery info
        if component_name not in self.recovery_info:
            self.recovery_info[component_name] = ComponentRecoveryInfo(
                component_name=component_name,
                original_type=component_type
            )
        
        recovery_info = self.recovery_info[component_name]
        
        # Check if we've exceeded max attempts
        if recovery_info.recovery_attempts >= self.config.max_recovery_attempts:
            guidance = RecoveryGuidance(
                title="Component Recovery Failed",
                message=f"Unable to recreate {component_name} after {recovery_info.recovery_attempts} attempts",
                suggestions=[
                    "Try restarting the application",
                    "Check system resources and memory",
                    "Contact support if the issue persists"
                ],
                severity="error",
                show_technical_details=True,
                technical_details=recovery_info.last_error
            )
            return None, guidance
        
        recovery_info.recovery_attempts += 1
        recovery_info.recovery_timestamp = datetime.now()
        
        try:
            # First, try to recreate with original arguments
            component = self._attempt_component_creation(component_type, original_kwargs)
            if component is not None:
                self.logger.info(f"Successfully recreated {component_name} with original arguments")
                guidance = RecoveryGuidance(
                    title="Component Recovered",
                    message=f"{component_name} has been successfully recreated",
                    severity="info"
                )
                return component, guidance
            
            # If original fails, try with fallback arguments
            if fallback_kwargs:
                component = self._attempt_component_creation(component_type, fallback_kwargs)
                if component is not None:
                    self.logger.info(f"Successfully recreated {component_name} with fallback arguments")
                    recovery_info.fallback_used = True
                    recovery_info.fallback_type = "modified_args"
                    guidance = RecoveryGuidance(
                        title="Component Recovered with Modifications",
                        message=f"{component_name} has been recreated with simplified settings",
                        suggestions=["Some advanced features may not be available"],
                        severity="warning"
                    )
                    return component, guidance
            
            # If both fail, try to create a fallback component
            if self.config.enable_fallback_components:
                fallback_component = self._create_fallback_component(component_type, component_name)
                if fallback_component is not None:
                    self.logger.info(f"Created fallback component for {component_name}")
                    recovery_info.fallback_used = True
                    recovery_info.fallback_type = "fallback_component"
                    guidance = RecoveryGuidance(
                        title="Fallback Component Created",
                        message=f"A simplified version of {component_name} has been created",
                        suggestions=[
                            "Full functionality may not be available",
                            "Try refreshing the page if issues persist"
                        ],
                        severity="warning"
                    )
                    return fallback_component, guidance
            
            # All attempts failed
            error_msg = f"All recreation attempts failed for {component_name}"
            recovery_info.last_error = error_msg
            self.logger.error(error_msg)
            
            guidance = RecoveryGuidance(
                title="Component Recreation Failed",
                message=f"Unable to recreate {component_name}",
                suggestions=[
                    "This feature will be temporarily unavailable",
                    "Try restarting the application",
                    "Check system resources"
                ],
                severity="error"
            )
            return None, guidance
            
        except Exception as e:
            error_msg = f"Exception during component recreation: {str(e)}"
            recovery_info.last_error = error_msg
            self.logger.error(f"Error recreating {component_name}: {e}")
            
            guidance = RecoveryGuidance(
                title="Component Recreation Error",
                message=f"An error occurred while recreating {component_name}",
                suggestions=[
                    "This may be a temporary issue",
                    "Try refreshing the page",
                    "Contact support if the problem persists"
                ],
                severity="error",
                show_technical_details=True,
                technical_details=str(e)
            )
            return None, guidance
    
    def _attempt_component_creation(self, component_type: str, kwargs: Dict[str, Any]) -> Optional[Any]:
        """
        Attempt to create a Gradio component with given arguments
        
        Args:
            component_type: Type of component to create
            kwargs: Arguments for component creation
            
        Returns:
            Created component or None if failed
        """
        try:
            # Get the Gradio component class
            component_class = getattr(gr, component_type, None)
            if component_class is None:
                self.logger.error(f"Unknown component type: {component_type}")
                return None
            
            # Create the component
            component = component_class(**kwargs)
            
            # Basic validation
            if hasattr(component, '_id') and component._id is not None:
                return component
            else:
                self.logger.warning(f"Created component lacks valid _id: {component_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create {component_type}: {e}")
            return None
    
    def _create_fallback_component(self, component_type: str, component_name: str) -> Optional[Any]:
        """
        Create a fallback component for the given type
        
        Args:
            component_type: Original component type
            component_name: Name of the component
            
        Returns:
            Fallback component or None
        """
        creator = self.fallback_creators.get(component_type)
        if creator:
            try:
                return creator(component_name)
            except Exception as e:
                self.logger.error(f"Failed to create fallback for {component_type}: {e}")
                return None
        else:
            # Generic fallback - create a simple HTML component
            try:
                return gr.HTML(
                    value=f"<div style='padding: 20px; border: 2px dashed #ccc; text-align: center;'>"
                          f"<p><strong>{component_name}</strong></p>"
                          f"<p>Component temporarily unavailable</p>"
                          f"<small>Original type: {component_type}</small></div>",
                    label=f"{component_name} (Fallback)"
                )
            except Exception as e:
                self.logger.error(f"Failed to create generic fallback: {e}")
                return None 
   
    # Fallback component creators
    def _create_fallback_textbox(self, component_name: str) -> gr.Textbox:
        """Create a fallback textbox component"""
        return gr.Textbox(
            label=f"{component_name} (Simplified)",
            placeholder="Enter text here...",
            info="Simplified version - some features may not be available"
        )
    
    def _create_fallback_button(self, component_name: str) -> gr.Button:
        """Create a fallback button component"""
        return gr.Button(
            value=f"{component_name}",
            variant="secondary",
            size="sm"
        )
    
    def _create_fallback_dropdown(self, component_name: str) -> gr.Dropdown:
        """Create a fallback dropdown component"""
        return gr.Dropdown(
            label=f"{component_name} (Limited Options)",
            choices=["Option 1", "Option 2", "Option 3"],
            value="Option 1",
            info="Limited options available"
        )
    
    def _create_fallback_slider(self, component_name: str) -> gr.Slider:
        """Create a fallback slider component"""
        return gr.Slider(
            label=f"{component_name} (Basic)",
            minimum=0,
            maximum=100,
            value=50,
            step=1
        )
    
    def _create_fallback_image(self, component_name: str) -> gr.Image:
        """Create a fallback image component"""
        return gr.Image(
            label=f"{component_name} (Basic)",
            type="pil",
            height=200
        )
    
    def _create_fallback_video(self, component_name: str) -> gr.Video:
        """Create a fallback video component"""
        return gr.Video(
            label=f"{component_name} (Basic)",
            height=200
        )
    
    def _create_fallback_file(self, component_name: str) -> gr.File:
        """Create a fallback file component"""
        return gr.File(
            label=f"{component_name} (Basic)",
            file_count="single"
        )
    
    def _create_fallback_html(self, component_name: str) -> gr.HTML:
        """Create a fallback HTML component"""
        return gr.HTML(
            value=f"<div style='padding: 10px; background: #f8f9fa; border-radius: 4px;'>"
                  f"<strong>{component_name}</strong><br>"
                  f"<small>Content will be displayed here</small></div>"
        )
    
    def _create_fallback_markdown(self, component_name: str) -> gr.Markdown:
        """Create a fallback markdown component"""
        return gr.Markdown(
            value=f"**{component_name}**\n\nContent will be displayed here.",
            height=100
        )
    
    def _create_fallback_gallery(self, component_name: str) -> gr.Gallery:
        """Create a fallback gallery component"""
        return gr.Gallery(
            label=f"{component_name} (Basic)",
            columns=2,
            rows=2,
            height=200
        )
    
    def _create_fallback_dataframe(self, component_name: str) -> gr.DataFrame:
        """Create a fallback dataframe component"""
        return gr.DataFrame(
            label=f"{component_name} (Basic)",
            headers=["Column 1", "Column 2"],
            datatype=["str", "str"],
            row_count=3,
            col_count=2
        )
    
    def _create_fallback_plot(self, component_name: str) -> gr.Plot:
        """Create a fallback plot component"""
        return gr.Plot(
            label=f"{component_name} (Basic)",
            value=None
        )
    
    def create_ui_fallback_mechanisms(self, failed_components: List[str]) -> Dict[str, Any]:
        """
        Create fallback mechanisms for failed UI components
        
        Args:
            failed_components: List of component names that failed
            
        Returns:
            Dictionary of fallback components and configurations
        """
        fallback_mechanisms = {
            'components': {},
            'simplified_ui': False,
            'guidance': []
        }
        
        # Create fallback components for each failed component
        for component_name in failed_components:
            if component_name in self.recovery_info:
                recovery_info = self.recovery_info[component_name]
                fallback_component = self._create_fallback_component(
                    recovery_info.original_type, 
                    component_name
                )
                if fallback_component:
                    fallback_mechanisms['components'][component_name] = fallback_component
        
        # Determine if simplified UI should be used
        critical_components_failed = len(failed_components) > 5
        if critical_components_failed and self.config.enable_simplified_ui:
            fallback_mechanisms['simplified_ui'] = True
            fallback_mechanisms['guidance'].append(
                RecoveryGuidance(
                    title="Simplified UI Mode",
                    message="Multiple components failed to load. Using simplified interface.",
                    suggestions=[
                        "Some advanced features may not be available",
                        "Try refreshing the page to restore full functionality",
                        "Check your internet connection and system resources"
                    ],
                    severity="warning"
                )
            )
        
        return fallback_mechanisms  
  
    def create_user_friendly_error_display(self, guidance: RecoveryGuidance) -> str:
        """
        Create user-friendly HTML error display
        
        Args:
            guidance: Recovery guidance information
            
        Returns:
            HTML string for error display
        """
        # Color scheme based on severity
        colors = {
            'info': {'bg': '#d1ecf1', 'border': '#bee5eb', 'text': '#0c5460', 'icon': '💡'},
            'warning': {'bg': '#fff3cd', 'border': '#ffeaa7', 'text': '#856404', 'icon': '⚠️'},
            'error': {'bg': '#f8d7da', 'border': '#f5c6cb', 'text': '#721c24', 'icon': '❌'},
            'critical': {'bg': '#d1ecf1', 'border': '#bee5eb', 'text': '#721c24', 'icon': '🚨'}
        }
        
        color_scheme = colors.get(guidance.severity, colors['info'])
        
        # Build suggestions HTML
        suggestions_html = ""
        if guidance.suggestions:
            suggestions_list = "".join([f"<li>{suggestion}</li>" for suggestion in guidance.suggestions])
            suggestions_html = f"""
            <div style="margin-top: 15px;">
                <strong>💡 Suggested Actions:</strong>
                <ul style="margin: 8px 0; padding-left: 20px; line-height: 1.4;">
                    {suggestions_list}
                </ul>
            </div>
            """
        
        # Technical details (collapsible)
        technical_html = ""
        if guidance.show_technical_details and guidance.technical_details:
            technical_html = f"""
            <details style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px;">
                <summary style="cursor: pointer; font-weight: bold; color: #495057;">
                    🔧 Technical Details (Click to expand)
                </summary>
                <pre style="margin: 10px 0; padding: 10px; background: #e9ecef; border-radius: 4px; font-size: 0.8em; overflow-x: auto;">
{guidance.technical_details}</pre>
            </details>
            """
        
        return f"""
        <div style="
            background: {color_scheme['bg']};
            border: 2px solid {color_scheme['border']};
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            color: {color_scheme['text']};
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            animation: slideIn 0.3s ease-out;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 1.2em; margin-right: 10px;">{color_scheme['icon']}</span>
                <h3 style="margin: 0; color: {color_scheme['text']}; font-size: 1.1em;">
                    {guidance.title}
                </h3>
            </div>
            
            <p style="margin: 10px 0; line-height: 1.5; font-size: 0.95em;">
                {guidance.message}
            </p>
            
            {suggestions_html}
            {technical_html}
            
            <div style="margin-top: 15px; font-size: 0.8em; color: #6c757d; border-top: 1px solid {color_scheme['border']}; padding-top: 10px;">
                <strong>Time:</strong> {datetime.now().strftime('%H:%M:%S')} | 
                <strong>Severity:</strong> {guidance.severity.title()}
            </div>
        </div>
        
        <style>
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        </style>
        """
    
    def generate_recovery_suggestions(self, error_type: str, component_name: str, error_details: str) -> List[str]:
        """
        Generate contextual recovery suggestions based on error type
        
        Args:
            error_type: Type of error encountered
            component_name: Name of the affected component
            error_details: Detailed error information
            
        Returns:
            List of recovery suggestions
        """
        suggestions = []
        
        # Common suggestions based on error type
        if "None" in error_details or "NoneType" in error_details:
            suggestions.extend([
                "Check if all required dependencies are properly loaded",
                "Verify that component initialization completed successfully",
                "Try refreshing the page to reinitialize components"
            ])
        
        if "memory" in error_details.lower() or "out of memory" in error_details.lower():
            suggestions.extend([
                "Close other applications to free up memory",
                "Try using a smaller model or reduced settings",
                "Restart the application to clear memory"
            ])
        
        if "connection" in error_details.lower() or "network" in error_details.lower():
            suggestions.extend([
                "Check your internet connection",
                "Try again in a few moments",
                "Verify that the server is running properly"
            ])
        
        if "file" in error_details.lower() or "path" in error_details.lower():
            suggestions.extend([
                "Check that all required files are present",
                "Verify file permissions and access rights",
                "Ensure the file path is correct and accessible"
            ])
        
        # Component-specific suggestions
        if "image" in component_name.lower():
            suggestions.extend([
                "Try uploading a different image format (PNG, JPG)",
                "Ensure the image file is not corrupted",
                "Check that the image size is within limits"
            ])
        
        if "video" in component_name.lower():
            suggestions.extend([
                "Try a different video format (MP4, AVI)",
                "Check that the video file is not too large",
                "Ensure sufficient disk space for video processing"
            ])
        
        if "model" in component_name.lower():
            suggestions.extend([
                "Check that the model files are properly downloaded",
                "Verify that you have sufficient VRAM/RAM for the model",
                "Try using a smaller or different model"
            ])
        
        # Generic fallback suggestions
        if not suggestions:
            suggestions.extend([
                "Try refreshing the page",
                "Check the browser console for additional error details",
                "Contact support if the issue persists"
            ])
        
        return suggestions[:5]  # Limit to 5 suggestions to avoid overwhelming the user
    
    def log_error_recovery_attempt(self, 
                                 component_name: str, 
                                 error_type: str, 
                                 recovery_action: str, 
                                 success: bool,
                                 details: Optional[str] = None):
        """
        Log error recovery attempts for analysis and debugging
        
        Args:
            component_name: Name of the component
            error_type: Type of error
            recovery_action: Action taken for recovery
            success: Whether recovery was successful
            details: Additional details
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'component_name': component_name,
            'error_type': error_type,
            'recovery_action': recovery_action,
            'success': success,
            'details': details
        }
        
        self.error_history.append(log_entry)
        
        # Log to file if configured
        if self.config.log_recovery_attempts:
            try:
                log_file = Path("logs/ui_error_recovery.log")
                log_file.parent.mkdir(exist_ok=True)
                
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry) + '\n')
                    
            except Exception as e:
                self.logger.error(f"Failed to write recovery log: {e}")
        
        # Also log to standard logger
        if success:
            self.logger.info(f"Recovery successful for {component_name}: {recovery_action}")
        else:
            self.logger.warning(f"Recovery failed for {component_name}: {recovery_action}")
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about error recovery attempts
        
        Returns:
            Dictionary with recovery statistics
        """
        total_attempts = len(self.error_history)
        successful_recoveries = sum(1 for entry in self.error_history if entry['success'])
        
        # Group by component
        component_stats = {}
        for entry in self.error_history:
            comp_name = entry['component_name']
            if comp_name not in component_stats:
                component_stats[comp_name] = {'attempts': 0, 'successes': 0}
            component_stats[comp_name]['attempts'] += 1
            if entry['success']:
                component_stats[comp_name]['successes'] += 1
        
        return {
            'total_recovery_attempts': total_attempts,
            'successful_recoveries': successful_recoveries,
            'recovery_success_rate': (successful_recoveries / total_attempts * 100) if total_attempts > 0 else 0,
            'components_with_issues': len(component_stats),
            'component_statistics': component_stats,
            'active_recovery_info': len(self.recovery_info)
        }
    
    def create_recovery_status_display(self) -> str:
        """
        Create HTML display showing current recovery status
        
        Returns:
            HTML string showing recovery status
        """
        stats = self.get_recovery_statistics()
        
        if stats['total_recovery_attempts'] == 0:
            return """
            <div style="background: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px; padding: 10px; margin: 10px 0;">
                <span style="color: #155724;">✅ All components loaded successfully</span>
            </div>
            """
        
        status_color = "#d4edda" if stats['recovery_success_rate'] > 80 else "#fff3cd" if stats['recovery_success_rate'] > 50 else "#f8d7da"
        border_color = "#c3e6cb" if stats['recovery_success_rate'] > 80 else "#ffeaa7" if stats['recovery_success_rate'] > 50 else "#f5c6cb"
        text_color = "#155724" if stats['recovery_success_rate'] > 80 else "#856404" if stats['recovery_success_rate'] > 50 else "#721c24"
        
        return f"""
        <div style="background: {status_color}; border: 1px solid {border_color}; border-radius: 4px; padding: 15px; margin: 10px 0;">
            <div style="color: {text_color}; font-weight: bold; margin-bottom: 10px;">
                🔧 Component Recovery Status
            </div>
            <div style="color: {text_color}; font-size: 0.9em; line-height: 1.4;">
                <strong>Recovery Attempts:</strong> {stats['total_recovery_attempts']}<br>
                <strong>Successful Recoveries:</strong> {stats['successful_recoveries']}<br>
                <strong>Success Rate:</strong> {stats['recovery_success_rate']:.1f}%<br>
                <strong>Components with Issues:</strong> {stats['components_with_issues']}
            </div>
        </div>
        """

# Global error recovery manager instance
global_recovery_manager = UIErrorRecoveryManager()

def get_error_recovery_manager(config: Optional[UIFallbackConfig] = None) -> UIErrorRecoveryManager:
    """
    Get the global error recovery manager instance
    
    Args:
        config: Optional configuration for the recovery manager
        
    Returns:
        UIErrorRecoveryManager instance
    """
    global global_recovery_manager
    if config and global_recovery_manager.config != config:
        global_recovery_manager = UIErrorRecoveryManager(config)
    return global_recovery_manager
