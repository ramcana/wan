#!/usr/bin/env python3
"""
Enhanced UI Creator with Error Recovery and Fallback Integration
Integrates error recovery mechanisms into the UI creation process
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Tuple
import gradio as gr
import time
from datetime import datetime

from ui_error_recovery import UIErrorRecoveryManager, UIFallbackConfig, RecoveryGuidance
from ui_creation_validator import UIComponentManager, UICreationReport
from infrastructure.hardware.safe_event_handler import SafeEventHandler
from utils_new.component_validator import ComponentValidator

logger = logging.getLogger(__name__)

class EnhancedUICreator:
    """Enhanced UI creator with integrated error recovery and fallback mechanisms"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize recovery and validation systems
        fallback_config = UIFallbackConfig(
            enable_component_recreation=True,
            enable_fallback_components=True,
            enable_simplified_ui=True,
            max_recovery_attempts=3,
            log_recovery_attempts=True,
            show_user_guidance=True
        )
        
        self.recovery_manager = UIErrorRecoveryManager(fallback_config)
        self.component_manager = UIComponentManager()
        self.safe_event_handler = SafeEventHandler()
        self.validator = ComponentValidator()
        
        # UI creation state
        self.creation_start_time = None
        self.created_components: Dict[str, Any] = {}
        self.failed_components: List[str] = []
        self.recovery_guidance: List[RecoveryGuidance] = []
        
        self.logger = logging.getLogger(__name__)
    
    def create_component_with_recovery(self, 
                                     component_type: str, 
                                     component_name: str,
                                     primary_kwargs: Dict[str, Any],
                                     fallback_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[RecoveryGuidance]]:
        """
        Create a component with automatic error recovery
        
        Args:
            component_type: Type of Gradio component to create
            component_name: Name/identifier for the component
            primary_kwargs: Primary arguments for component creation
            fallback_kwargs: Fallback arguments if primary fails
            
        Returns:
            Tuple of (component, recovery_guidance)
        """
        try:
            # First attempt with primary arguments
            component = self.component_manager.create_component_safely(
                getattr(gr, component_type), **primary_kwargs
            )
            
            if component is not None:
                # Validate the created component
                if self.validator.validate_component(component, component_name):
                    self.created_components[component_name] = component
                    self.logger.debug(f"Successfully created {component_name}")
                    return component, None
                else:
                    self.logger.warning(f"Created component failed validation: {component_name}")
            
            # Primary creation failed, attempt recovery
            self.logger.info(f"Attempting recovery for {component_name}")
            recovered_component, guidance = self.recovery_manager.recreate_component(
                component_name=component_name,
                component_type=component_type,
                original_kwargs=primary_kwargs,
                fallback_kwargs=fallback_kwargs
            )
            
            if recovered_component is not None:
                self.created_components[component_name] = recovered_component
                self.recovery_guidance.append(guidance)
                
                # Log recovery attempt
                self.recovery_manager.log_error_recovery_attempt(
                    component_name=component_name,
                    error_type="creation_failure",
                    recovery_action="component_recreation",
                    success=True,
                    details=f"Recovered with {'fallback_args' if fallback_kwargs else 'fallback_component'}"
                )
                
                return recovered_component, guidance
            else:
                # Recovery failed
                self.failed_components.append(component_name)
                self.recovery_guidance.append(guidance)
                
                # Log failed recovery
                self.recovery_manager.log_error_recovery_attempt(
                    component_name=component_name,
                    error_type="creation_failure",
                    recovery_action="component_recreation",
                    success=False,
                    details="All recovery attempts failed"
                )
                
                return None, guidance
                
        except Exception as e:
            self.logger.error(f"Exception creating {component_name}: {e}")
            
            # Generate error guidance
            suggestions = self.recovery_manager.generate_recovery_suggestions(
                error_type="creation_exception",
                component_name=component_name,
                error_details=str(e)
            )
            
            guidance = RecoveryGuidance(
                title="Component Creation Error",
                message=f"Failed to create {component_name} due to an unexpected error",
                suggestions=suggestions,
                severity="error",
                show_technical_details=True,
                technical_details=str(e)
            )
            
            self.failed_components.append(component_name)
            self.recovery_guidance.append(guidance)
            
            return None, guidance
    
    def setup_event_handlers_with_recovery(self, 
                                         component_configs: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Set up event handlers with error recovery
        
        Args:
            component_configs: List of event handler configurations
            
        Returns:
            Dictionary mapping event names to success status
        """
        results = {}
        
        for config in component_configs:
            try:
                component_name = config.get('component_name', 'unknown')
                event_type = config.get('event_type', 'click')
                handler_fn = config.get('handler_function')
                input_names = config.get('inputs', [])
                output_names = config.get('outputs', [])
                
                # Get components from registry
                component = self.created_components.get(component_name)
                if not component:
                    self.logger.warning(f"Component not found for event setup: {component_name}")
                    results[f"{component_name}_{event_type}"] = False
                    continue
                
                # Get input components
                input_components = []
                for input_name in input_names:
                    input_comp = self.created_components.get(input_name)
                    if input_comp:
                        input_components.append(input_comp)
                
                # Get output components
                output_components = []
                for output_name in output_names:
                    output_comp = self.created_components.get(output_name)
                    if output_comp:
                        output_components.append(output_comp)
                
                # Set up event handler safely
                success = self.safe_event_handler.setup_safe_event(
                    component=component,
                    event_type=event_type,
                    handler_fn=handler_fn,
                    inputs=input_components,
                    outputs=output_components,
                    component_name=component_name
                )
                
                results[f"{component_name}_{event_type}"] = success
                
                if success:
                    self.logger.debug(f"Successfully set up {event_type} event for {component_name}")
                else:
                    self.logger.warning(f"Failed to set up {event_type} event for {component_name}")
                    
                    # Log recovery attempt for event handler
                    self.recovery_manager.log_error_recovery_attempt(
                        component_name=component_name,
                        error_type="event_handler_failure",
                        recovery_action="safe_event_setup",
                        success=success,
                        details=f"Event type: {event_type}"
                    )
                
            except Exception as e:
                event_key = f"{config.get('component_name', 'unknown')}_{config.get('event_type', 'unknown')}"
                results[event_key] = False
                self.logger.error(f"Exception setting up event handler {event_key}: {e}")
        
        return results    

    def create_ui_with_fallbacks(self, ui_definition: Dict[str, Any]) -> Tuple[gr.Blocks, List[RecoveryGuidance]]:
        """
        Create UI with comprehensive fallback mechanisms
        
        Args:
            ui_definition: Dictionary defining the UI structure and components
            
        Returns:
            Tuple of (gradio_blocks, recovery_guidance_list)
        """
        self.creation_start_time = time.time()
        self.logger.info("Starting enhanced UI creation with fallback mechanisms")
        
        # Extract UI configuration
        css = ui_definition.get('css', '')
        title = ui_definition.get('title', 'WAN22 UI')
        theme = ui_definition.get('theme', gr.themes.Soft())
        components_def = ui_definition.get('components', {})
        event_handlers_def = ui_definition.get('event_handlers', [])
        
        try:
            # Create Gradio Blocks with error handling
            with gr.Blocks(css=css, title=title, theme=theme) as interface:
                
                # Create components with recovery
                self._create_components_with_recovery(components_def)
                
                # Set up event handlers with recovery
                if event_handlers_def:
                    self._setup_event_handlers_with_recovery(event_handlers_def)
                
                # Create fallback mechanisms for failed components
                if self.failed_components:
                    self._create_fallback_ui_elements()
                
                # Add recovery status display if there were issues
                if self.recovery_guidance:
                    self._add_recovery_status_display()
            
            # Log creation summary
            creation_time = time.time() - self.creation_start_time
            self._log_creation_summary(creation_time)
            
            return interface, self.recovery_guidance
            
        except Exception as e:
            self.logger.error(f"Critical error during UI creation: {e}")
            
            # Create emergency fallback UI
            emergency_interface = self._create_emergency_fallback_ui(str(e))
            
            # Create critical error guidance
            critical_guidance = RecoveryGuidance(
                title="Critical UI Creation Error",
                message="A critical error occurred during UI creation. Using emergency fallback interface.",
                suggestions=[
                    "Try restarting the application",
                    "Check system resources and dependencies",
                    "Contact support with the error details"
                ],
                severity="critical",
                show_technical_details=True,
                technical_details=str(e)
            )
            
            return emergency_interface, [critical_guidance]
    
    def _create_components_with_recovery(self, components_def: Dict[str, Any]):
        """Create components with recovery mechanisms"""
        for component_name, component_config in components_def.items():
            component_type = component_config.get('type', 'HTML')
            primary_kwargs = component_config.get('kwargs', {})
            fallback_kwargs = component_config.get('fallback_kwargs')
            
            component, guidance = self.create_component_with_recovery(
                component_type=component_type,
                component_name=component_name,
                primary_kwargs=primary_kwargs,
                fallback_kwargs=fallback_kwargs
            )
            
            if guidance:
                self.recovery_guidance.append(guidance)
    
    def _setup_event_handlers_with_recovery(self, event_handlers_def: List[Dict[str, Any]]):
        """Set up event handlers with recovery mechanisms"""
        results = self.setup_event_handlers_with_recovery(event_handlers_def)
        
        # Create guidance for failed event handlers
        failed_handlers = [name for name, success in results.items() if not success]
        if failed_handlers:
            guidance = RecoveryGuidance(
                title="Event Handler Setup Issues",
                message=f"Some event handlers could not be set up: {', '.join(failed_handlers)}",
                suggestions=[
                    "Some interactive features may not work as expected",
                    "Try refreshing the page to restore functionality",
                    "Check browser console for additional error details"
                ],
                severity="warning"
            )
            self.recovery_guidance.append(guidance)
    
    def _create_fallback_ui_elements(self):
        """Create fallback UI elements for failed components"""
        if not self.failed_components:
            return
        
        self.logger.info(f"Creating fallback elements for {len(self.failed_components)} failed components")
        
        # Create fallback mechanisms
        fallback_mechanisms = self.recovery_manager.create_ui_fallback_mechanisms(self.failed_components)
        
        # Add fallback components to the UI
        fallback_components = fallback_mechanisms.get('components', {})
        for component_name, fallback_component in fallback_components.items():
            if fallback_component:
                self.created_components[f"{component_name}_fallback"] = fallback_component
        
        # Add guidance from fallback mechanisms
        fallback_guidance = fallback_mechanisms.get('guidance', [])
        self.recovery_guidance.extend(fallback_guidance)
    
    def _add_recovery_status_display(self):
        """Add recovery status display to the UI"""
        try:
            # Create recovery status HTML
            status_html = self.recovery_manager.create_recovery_status_display()
            
            # Create guidance displays
            guidance_html = ""
            for guidance in self.recovery_guidance:
                guidance_html += self.recovery_manager.create_user_friendly_error_display(guidance)
            
            # Combine status and guidance
            combined_html = f"""
            <div style="margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                <h3 style="margin-top: 0; color: #495057;">🔧 System Status & Recovery Information</h3>
                {status_html}
                {guidance_html}
            </div>
            """
            
            # Add to UI
            recovery_display = gr.HTML(value=combined_html, visible=True)
            self.created_components['recovery_status_display'] = recovery_display
            
        except Exception as e:
            self.logger.error(f"Failed to create recovery status display: {e}")
    
    def _create_emergency_fallback_ui(self, error_details: str) -> gr.Blocks:
        """Create emergency fallback UI when main UI creation fails"""
        self.logger.info("Creating emergency fallback UI")
        
        try:
            with gr.Blocks(title="WAN22 UI - Emergency Mode") as emergency_interface:
                gr.Markdown("# 🚨 WAN22 UI - Emergency Mode")
                
                gr.HTML(f"""
                <div style="background: #f8d7da; border: 2px solid #f5c6cb; border-radius: 8px; padding: 20px; margin: 20px 0;">
                    <h3 style="color: #721c24; margin-top: 0;">⚠️ Critical Error</h3>
                    <p style="color: #721c24;">
                        The main UI could not be created due to a critical error. 
                        This emergency interface provides basic functionality.
                    </p>
                    <details style="margin-top: 15px;">
                        <summary style="cursor: pointer; font-weight: bold;">Technical Details</summary>
                        <pre style="background: #e9ecef; padding: 10px; border-radius: 4px; margin-top: 10px; overflow-x: auto;">
{error_details}</pre>
                    </details>
                </div>
                """)
                
                gr.Markdown("""
                ## 🔧 Recovery Actions
                
                1. **Restart the Application**: Close and restart the WAN22 UI
                2. **Check System Resources**: Ensure sufficient memory and disk space
                3. **Update Dependencies**: Make sure all required packages are installed
                4. **Contact Support**: If the issue persists, contact technical support
                
                ## 📋 Basic Information
                
                This emergency interface is displayed when the main UI cannot be created.
                Some features may not be available in this mode.
                """)
                
                # Add basic system info if possible
                try:
                    import psutil
                    import platform
                    
                    system_info = f"""
                    **System Information:**
                    - Platform: {platform.system()} {platform.release()}
                    - Python Version: {platform.python_version()}
                    - Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB
                    - CPU Usage: {psutil.cpu_percent()}%
                    """
                    
                    gr.Markdown(system_info)
                    
                except Exception:
                    gr.Markdown("*System information unavailable*")
            
            return emergency_interface
            
        except Exception as e:
            self.logger.error(f"Failed to create emergency fallback UI: {e}")
            # Return absolute minimal interface
            with gr.Blocks(title="WAN22 UI - Critical Error") as minimal_interface:
                gr.HTML(f"""
                <div style="padding: 20px; text-align: center;">
                    <h1>🚨 Critical Error</h1>
                    <p>Unable to create UI interface. Please restart the application.</p>
                    <p><strong>Error:</strong> {str(e)}</p>
                </div>
                """)
            return minimal_interface
    
    def _log_creation_summary(self, creation_time: float):
        """Log summary of UI creation process"""
        total_components = len(self.created_components)
        failed_count = len(self.failed_components)
        success_rate = ((total_components - failed_count) / total_components * 100) if total_components > 0 else 0
        
        self.logger.info("=== Enhanced UI Creation Summary ===")
        self.logger.info(f"Creation time: {creation_time:.2f} seconds")
        self.logger.info(f"Total components: {total_components}")
        self.logger.info(f"Failed components: {failed_count}")
        self.logger.info(f"Success rate: {success_rate:.1f}%")
        self.logger.info(f"Recovery guidance items: {len(self.recovery_guidance)}")
        
        if self.failed_components:
            self.logger.warning(f"Failed components: {', '.join(self.failed_components)}")
        
        # Log event handler statistics
        event_stats = self.safe_event_handler.get_setup_statistics()
        self.logger.info(f"Event handlers - Success: {event_stats['successful_setups']}, Failed: {event_stats['failed_setups']}")
        
        self.logger.info("=" * 40)
    
    def get_creation_report(self) -> Dict[str, Any]:
        """
        Get comprehensive report of UI creation process
        
        Returns:
            Dictionary with creation statistics and details
        """
        creation_time = time.time() - self.creation_start_time if self.creation_start_time else 0
        
        return {
            'creation_time': creation_time,
            'total_components': len(self.created_components),
            'failed_components': len(self.failed_components),
            'failed_component_names': self.failed_components.copy(),
            'recovery_guidance_count': len(self.recovery_guidance),
            'recovery_statistics': self.recovery_manager.get_recovery_statistics(),
            'event_handler_statistics': self.safe_event_handler.get_setup_statistics(),
            'validation_report': self.validator.get_validation_report()
        }

# Global enhanced UI creator instance
global_ui_creator = EnhancedUICreator()

def create_enhanced_ui(ui_definition: Dict[str, Any], 
                      config: Optional[Dict[str, Any]] = None) -> Tuple[gr.Blocks, List[RecoveryGuidance]]:
    """
    Create UI with enhanced error recovery and fallback mechanisms
    
    Args:
        ui_definition: Dictionary defining the UI structure
        config: Optional configuration
        
    Returns:
        Tuple of (gradio_blocks, recovery_guidance_list)
    """
    global global_ui_creator
    if config:
        global_ui_creator = EnhancedUICreator(config)
    
    return global_ui_creator.create_ui_with_fallbacks(ui_definition)

def get_ui_creation_report() -> Dict[str, Any]:
    """
    Get report of the last UI creation process
    
    Returns:
        Creation report dictionary
    """
    return global_ui_creator.get_creation_report()