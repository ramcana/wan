#!/usr/bin/env python3
"""
Component Validator for WAN22 UI
Validates Gradio components to prevent None component errors during UI creation
"""

import logging
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
import gradio as gr

logger = logging.getLogger(__name__)

@dataclass
class ComponentValidationResult:
    """Result of component validation"""
    component_name: str
    is_valid: bool
    error_message: Optional[str]
    component_type: str
    validation_timestamp: datetime

@dataclass
class UICreationReport:
    """Report of UI creation process"""
    total_components: int
    valid_components: int
    failed_components: List[str]
    event_handlers_setup: int
    event_handlers_failed: int
    creation_time: float
    errors: List[str]

class ComponentValidator:
    """Validates Gradio components to prevent UI creation errors"""
    
    def __init__(self):
        self.validation_results: List[ComponentValidationResult] = []
        self.component_registry: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
    def validate_component(self, component: Any, component_name: str) -> bool:
        """
        Validate a single Gradio component
        
        Args:
            component: The Gradio component to validate
            component_name: Name/identifier for the component
            
        Returns:
            bool: True if component is valid, False otherwise
        """
        validation_time = datetime.now()
        
        try:
            # Check if component is None
            if component is None:
                result = ComponentValidationResult(
                    component_name=component_name,
                    is_valid=False,
                    error_message="Component is None",
                    component_type="None",
                    validation_timestamp=validation_time
                )
                self.validation_results.append(result)
                self.logger.warning(f"Component validation failed: {component_name} is None")
                return False
            
            # Check if component is a valid Gradio component first
            if not isinstance(component, (gr.components.Component, gr.blocks.Block)):
                result = ComponentValidationResult(
                    component_name=component_name,
                    is_valid=False,
                    error_message=f"Component is not a valid Gradio component: {type(component)}",
                    component_type=type(component).__name__,
                    validation_timestamp=validation_time
                )
                self.validation_results.append(result)
                self.logger.warning(f"Component validation failed: {component_name} is not a valid Gradio component")
                return False
            
            # Check if component has required Gradio attributes
            if not hasattr(component, '_id'):
                result = ComponentValidationResult(
                    component_name=component_name,
                    is_valid=False,
                    error_message="Component missing '_id' attribute",
                    component_type=type(component).__name__,
                    validation_timestamp=validation_time
                )
                self.validation_results.append(result)
                self.logger.warning(f"Component validation failed: {component_name} missing '_id' attribute")
                return False
            
            # Component is valid
            result = ComponentValidationResult(
                component_name=component_name,
                is_valid=True,
                error_message=None,
                component_type=type(component).__name__,
                validation_timestamp=validation_time
            )
            self.validation_results.append(result)
            self.logger.debug(f"Component validation passed: {component_name} ({type(component).__name__})")
            return True
            
        except Exception as e:
            result = ComponentValidationResult(
                component_name=component_name,
                is_valid=False,
                error_message=f"Validation error: {str(e)}",
                component_type="Unknown",
                validation_timestamp=validation_time
            )
            self.validation_results.append(result)
            self.logger.error(f"Component validation error for {component_name}: {e}")
            return False
    
    def validate_component_list(self, components: List[Any], context: str) -> List[Any]:
        """
        Validate a list of components and return only valid ones
        
        Args:
            components: List of components to validate
            context: Context description for logging
            
        Returns:
            List of valid components (None components filtered out)
        """
        if not components:
            self.logger.warning(f"Empty component list provided for context: {context}")
            return []
        
        valid_components = []
        
        for i, component in enumerate(components):
            component_name = f"{context}_component_{i}"
            
            if self.validate_component(component, component_name):
                valid_components.append(component)
            else:
                self.logger.warning(f"Filtered out invalid component {i} in context: {context}")
        
        self.logger.info(f"Validated {len(valid_components)}/{len(components)} components for context: {context}")
        return valid_components
    
    def register_component(self, name: str, component: Any) -> bool:
        """
        Register a component in the registry after validation
        
        Args:
            name: Component name/key
            component: The component to register
            
        Returns:
            bool: True if registered successfully, False otherwise
        """
        if self.validate_component(component, name):
            self.component_registry[name] = component
            self.logger.debug(f"Registered component: {name}")
            return True
        else:
            self.logger.warning(f"Failed to register invalid component: {name}")
            return False
    
    def get_component(self, name: str) -> Optional[Any]:
        """
        Get a registered component by name
        
        Args:
            name: Component name/key
            
        Returns:
            The component if found and valid, None otherwise
        """
        component = self.component_registry.get(name)
        if component is None:
            self.logger.warning(f"Component not found in registry: {name}")
            return None
        
        # Re-validate component to ensure it's still valid
        if self.validate_component(component, f"registry_{name}"):
            return component
        else:
            self.logger.warning(f"Component in registry is no longer valid: {name}")
            # Remove invalid component from registry
            del self.component_registry[name]
            return None
    
    def validate_component_dict(self, components_dict: Dict[str, Any], context: str) -> Dict[str, Any]:
        """
        Validate a dictionary of components and return only valid ones
        
        Args:
            components_dict: Dictionary of components to validate
            context: Context description for logging
            
        Returns:
            Dictionary with only valid components
        """
        if not components_dict:
            self.logger.warning(f"Empty component dictionary provided for context: {context}")
            return {}
        
        valid_components = {}
        
        for name, component in components_dict.items():
            component_name = f"{context}_{name}"
            
            if self.validate_component(component, component_name):
                valid_components[name] = component
            else:
                self.logger.warning(f"Filtered out invalid component '{name}' in context: {context}")
        
        self.logger.info(f"Validated {len(valid_components)}/{len(components_dict)} components for context: {context}")
        return valid_components
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive validation report
        
        Returns:
            Dictionary containing validation statistics and details
        """
        total_validations = len(self.validation_results)
        valid_count = sum(1 for result in self.validation_results if result.is_valid)
        invalid_count = total_validations - valid_count
        
        failed_components = [
            result.component_name for result in self.validation_results 
            if not result.is_valid
        ]
        
        error_messages = [
            f"{result.component_name}: {result.error_message}"
            for result in self.validation_results 
            if not result.is_valid and result.error_message
        ]
        
        return {
            'total_validations': total_validations,
            'valid_components': valid_count,
            'invalid_components': invalid_count,
            'success_rate': (valid_count / total_validations * 100) if total_validations > 0 else 0,
            'failed_components': failed_components,
            'error_messages': error_messages,
            'registered_components': len(self.component_registry),
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def clear_validation_history(self):
        """Clear validation history and results"""
        self.validation_results.clear()
        self.logger.info("Validation history cleared")
    
    def log_validation_summary(self):
        """Log a summary of validation results"""
        report = self.get_validation_report()
        
        self.logger.info("=== Component Validation Summary ===")
        self.logger.info(f"Total validations: {report['total_validations']}")
        self.logger.info(f"Valid components: {report['valid_components']}")
        self.logger.info(f"Invalid components: {report['invalid_components']}")
        self.logger.info(f"Success rate: {report['success_rate']:.1f}%")
        self.logger.info(f"Registered components: {report['registered_components']}")
        
        if report['failed_components']:
            self.logger.warning(f"Failed components: {', '.join(report['failed_components'])}")
        
        if report['error_messages']:
            self.logger.error("Validation errors:")
            for error in report['error_messages']:
                self.logger.error(f"  - {error}")
        
        self.logger.info("=" * 40)

# Global validator instance for use throughout the application
global_validator = ComponentValidator()

def validate_gradio_component(component: Any, name: str) -> bool:
    """
    Convenience function to validate a single component using the global validator
    
    Args:
        component: The component to validate
        name: Component name for logging
        
    Returns:
        bool: True if valid, False otherwise
    """
    return global_validator.validate_component(component, name)

def filter_valid_components(components: List[Any], context: str = "unknown") -> List[Any]:
    """
    Convenience function to filter valid components using the global validator
    
    Args:
        components: List of components to filter
        context: Context for logging
        
    Returns:
        List of valid components
    """
    return global_validator.validate_component_list(components, context)

def get_validation_summary() -> Dict[str, Any]:
    """
    Get validation summary from the global validator
    
    Returns:
        Validation report dictionary
    """
    return global_validator.get_validation_report()
