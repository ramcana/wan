#!/usr/bin/env python3
"""
Safe Event Handler for WAN22 UI
Provides safe event handler setup for Gradio components to prevent None component errors
"""

import logging
from typing import List, Dict, Optional, Any, Callable, Union
from dataclasses import dataclass
import gradio as gr
from utils_new.component_validator import ComponentValidator, validate_gradio_component

logger = logging.getLogger(__name__)

@dataclass
class EventHandlerConfig:
    """Configuration for an event handler"""
    component_name: str
    event_type: str
    handler_function: Callable
    inputs: List[str]  # Component names
    outputs: List[str]  # Component names
    is_critical: bool = False

class SafeEventHandler:
    """Safely sets up Gradio event handlers with component validation"""
    
    def __init__(self, component_validator: Optional[ComponentValidator] = None):
        self.validator = component_validator or ComponentValidator()
        self.logger = logging.getLogger(__name__)
        self.event_handlers_setup = 0
        self.event_handlers_failed = 0
        self.failed_handlers: List[str] = []
        
    def setup_safe_event(self, 
                        component: Any, 
                        event_type: str, 
                        handler_fn: Callable, 
                        inputs: List[Any], 
                        outputs: List[Any],
                        component_name: str = "unknown") -> bool:
        """
        Safely set up an event handler with component validation
        
        Args:
            component: The Gradio component to attach the event to
            event_type: Type of event ('click', 'change', 'submit', etc.)
            handler_fn: The function to call when event occurs
            inputs: List of input components for the handler
            outputs: List of output components for the handler
            component_name: Name of the component for logging
            
        Returns:
            bool: True if event handler was set up successfully, False otherwise
        """
        try:
            # Validate the main component
            if not self.validator.validate_component(component, f"{component_name}_main"):
                self.logger.warning(f"Skipping event setup for invalid component: {component_name}")
                self.event_handlers_failed += 1
                self.failed_handlers.append(f"{component_name}_{event_type}")
                return False
            
            # Filter and validate input components
            valid_inputs = self.filter_valid_components(inputs, f"{component_name}_inputs")
            
            # Filter and validate output components
            valid_outputs = self.filter_valid_components(outputs, f"{component_name}_outputs")
            
            # Check if we have enough valid components to proceed
            if not valid_inputs and inputs:
                self.logger.warning(f"No valid input components for {component_name} event handler")
                self.event_handlers_failed += 1
                self.failed_handlers.append(f"{component_name}_{event_type}_no_inputs")
                return False
                
            if not valid_outputs and outputs:
                self.logger.warning(f"No valid output components for {component_name} event handler")
                self.event_handlers_failed += 1
                self.failed_handlers.append(f"{component_name}_{event_type}_no_outputs")
                return False
            
            # Set up the event handler based on type
            success = self._setup_event_by_type(component, event_type, handler_fn, valid_inputs, valid_outputs, component_name)
            
            if success:
                self.event_handlers_setup += 1
                self.logger.debug(f"Successfully set up {event_type} event for {component_name}")
                return True
            else:
                self.event_handlers_failed += 1
                self.failed_handlers.append(f"{component_name}_{event_type}_setup_failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error setting up {event_type} event for {component_name}: {e}")
            self.event_handlers_failed += 1
            self.failed_handlers.append(f"{component_name}_{event_type}_exception")
            return False
    
    def _setup_event_by_type(self, 
                           component: Any, 
                           event_type: str, 
                           handler_fn: Callable, 
                           inputs: List[Any], 
                           outputs: List[Any],
                           component_name: str) -> bool:
        """
        Set up event handler based on event type
        
        Args:
            component: The component to attach event to
            event_type: Type of event
            handler_fn: Handler function
            inputs: Valid input components
            outputs: Valid output components
            component_name: Component name for logging
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if event_type == 'click':
                component.click(fn=handler_fn, inputs=inputs, outputs=outputs)
            elif event_type == 'change':
                component.change(fn=handler_fn, inputs=inputs, outputs=outputs)
            elif event_type == 'submit':
                component.submit(fn=handler_fn, inputs=inputs, outputs=outputs)
            elif event_type == 'upload':
                component.upload(fn=handler_fn, inputs=inputs, outputs=outputs)
            elif event_type == 'select':
                component.select(fn=handler_fn, inputs=inputs, outputs=outputs)
            else:
                self.logger.error(f"Unsupported event type: {event_type} for {component_name}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set up {event_type} event for {component_name}: {e}")
            return False
    
    def filter_valid_components(self, components: List[Any], context: str) -> List[Any]:
        """
        Filter out invalid components from a list
        
        Args:
            components: List of components to filter
            context: Context for logging
            
        Returns:
            List of valid components
        """
        if not components:
            return []
        
        return self.validator.validate_component_list(components, context)
    
    def validate_event_setup(self, 
                           component: Any, 
                           inputs: List[Any], 
                           outputs: List[Any],
                           component_name: str = "unknown") -> bool:
        """
        Validate that an event setup would be successful without actually setting it up
        
        Args:
            component: The component to validate
            inputs: Input components to validate
            outputs: Output components to validate
            component_name: Component name for logging
            
        Returns:
            bool: True if event setup would be valid, False otherwise
        """
        # Validate main component
        if not self.validator.validate_component(component, f"{component_name}_validation"):
            return False
        
        # Validate inputs
        valid_inputs = self.filter_valid_components(inputs, f"{component_name}_inputs_validation")
        if not valid_inputs and inputs:
            return False
        
        # Validate outputs
        valid_outputs = self.filter_valid_components(outputs, f"{component_name}_outputs_validation")
        if not valid_outputs and outputs:
            return False
        
        return True
    
    def setup_safe_event_from_config(self, 
                                   config: EventHandlerConfig, 
                                   component_registry: Dict[str, Any]) -> bool:
        """
        Set up event handler from configuration using component registry
        
        Args:
            config: Event handler configuration
            component_registry: Registry of components by name
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get main component
        component = component_registry.get(config.component_name)
        if not component:
            self.logger.warning(f"Component not found in registry: {config.component_name}")
            return False
        
        # Get input components
        input_components = []
        for input_name in config.inputs:
            input_comp = component_registry.get(input_name)
            if input_comp:
                input_components.append(input_comp)
            else:
                self.logger.warning(f"Input component not found: {input_name}")
        
        # Get output components
        output_components = []
        for output_name in config.outputs:
            output_comp = component_registry.get(output_name)
            if output_comp:
                output_components.append(output_comp)
            else:
                self.logger.warning(f"Output component not found: {output_name}")
        
        # Set up the event
        return self.setup_safe_event(
            component=component,
            event_type=config.event_type,
            handler_fn=config.handler_function,
            inputs=input_components,
            outputs=output_components,
            component_name=config.component_name
        )
    
    def setup_multiple_events(self, 
                            event_configs: List[EventHandlerConfig], 
                            component_registry: Dict[str, Any]) -> Dict[str, bool]:
        """
        Set up multiple event handlers from configurations
        
        Args:
            event_configs: List of event handler configurations
            component_registry: Registry of components by name
            
        Returns:
            Dictionary mapping event names to success status
        """
        results = {}
        
        for config in event_configs:
            event_key = f"{config.component_name}_{config.event_type}"
            success = self.setup_safe_event_from_config(config, component_registry)
            results[event_key] = success
            
            if not success and config.is_critical:
                self.logger.error(f"Critical event handler failed: {event_key}")
        
        return results
    
    def get_setup_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about event handler setup
        
        Returns:
            Dictionary with setup statistics
        """
        total_attempts = self.event_handlers_setup + self.event_handlers_failed
        success_rate = (self.event_handlers_setup / total_attempts * 100) if total_attempts > 0 else 0
        
        return {
            'total_attempts': total_attempts,
            'successful_setups': self.event_handlers_setup,
            'failed_setups': self.event_handlers_failed,
            'success_rate': success_rate,
            'failed_handlers': self.failed_handlers.copy()
        }
    
    def log_setup_summary(self):
        """Log a summary of event handler setup results"""
        stats = self.get_setup_statistics()
        
        self.logger.info("=== Event Handler Setup Summary ===")
        self.logger.info(f"Total attempts: {stats['total_attempts']}")
        self.logger.info(f"Successful setups: {stats['successful_setups']}")
        self.logger.info(f"Failed setups: {stats['failed_setups']}")
        self.logger.info(f"Success rate: {stats['success_rate']:.1f}%")
        
        if stats['failed_handlers']:
            self.logger.warning(f"Failed handlers: {', '.join(stats['failed_handlers'])}")
        
        self.logger.info("=" * 40)
    
    def reset_statistics(self):
        """Reset setup statistics"""
        self.event_handlers_setup = 0
        self.event_handlers_failed = 0
        self.failed_handlers.clear()

# Global safe event handler instance
global_safe_handler = SafeEventHandler()

def setup_safe_click_event(component: Any, 
                          handler_fn: Callable, 
                          inputs: List[Any], 
                          outputs: List[Any],
                          component_name: str = "unknown") -> bool:
    """
    Convenience function to set up a safe click event
    
    Args:
        component: Component to attach click event to
        handler_fn: Click handler function
        inputs: Input components
        outputs: Output components
        component_name: Component name for logging
        
    Returns:
        bool: True if successful, False otherwise
    """
    return global_safe_handler.setup_safe_event(
        component=component,
        event_type='click',
        handler_fn=handler_fn,
        inputs=inputs,
        outputs=outputs,
        component_name=component_name
    )

def setup_safe_change_event(component: Any, 
                           handler_fn: Callable, 
                           inputs: List[Any], 
                           outputs: List[Any],
                           component_name: str = "unknown") -> bool:
    """
    Convenience function to set up a safe change event
    
    Args:
        component: Component to attach change event to
        handler_fn: Change handler function
        inputs: Input components
        outputs: Output components
        component_name: Component name for logging
        
    Returns:
        bool: True if successful, False otherwise
    """
    return global_safe_handler.setup_safe_event(
        component=component,
        event_type='change',
        handler_fn=handler_fn,
        inputs=inputs,
        outputs=outputs,
        component_name=component_name
    )

def filter_safe_components(components: List[Any], context: str = "unknown") -> List[Any]:
    """
    Convenience function to filter valid components
    
    Args:
        components: Components to filter
        context: Context for logging
        
    Returns:
        List of valid components
    """
    return global_safe_handler.filter_valid_components(components, context)

def get_event_handler_stats() -> Dict[str, Any]:
    """
    Get event handler setup statistics
    
    Returns:
        Statistics dictionary
    """
    return global_safe_handler.get_setup_statistics()