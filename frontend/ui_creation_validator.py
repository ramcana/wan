#!/usr/bin/env python3
"""
UI Creation Validator for WAN22 UI
Provides comprehensive validation during UI creation to prevent Gradio component errors
"""

import logging
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
import gradio as gr
from utils_new.component_validator import ComponentValidator

logger = logging.getLogger(__name__)

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

class UIComponentManager:
    """Manages UI components with validation and safe creation"""
    
    def __init__(self):
        self.validator = ComponentValidator()
        self.components_registry: Dict[str, Any] = {}
        self.creation_errors: List[str] = []
        self.logger = logging.getLogger(__name__)
        
    def create_component_safely(self, component_type: type, **kwargs) -> Optional[Any]:
        """
        Create a Gradio component safely with validation
        
        Args:
            component_type: The Gradio component class to create
            **kwargs: Arguments for component creation
            
        Returns:
            The created component or None if creation failed
        """
        try:
            component = component_type(**kwargs)
            
            # Validate the created component
            component_name = kwargs.get('label', kwargs.get('value', str(component_type.__name__)))
            if self.validator.validate_component(component, component_name):
                return component
            else:
                self.logger.warning(f"Created component failed validation: {component_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create component {component_type.__name__}: {e}")
            self.creation_errors.append(f"{component_type.__name__}: {str(e)}")
            return None
    
    def register_component(self, name: str, component: Any) -> bool:
        """
        Register a component with validation
        
        Args:
            name: Component name/key
            component: The component to register
            
        Returns:
            bool: True if registered successfully, False otherwise
        """
        if component is None:
            self.logger.warning(f"Attempted to register None component: {name}")
            return False
            
        if self.validator.validate_component(component, name):
            self.components_registry[name] = component
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
        return self.components_registry.get(name)
    
    def validate_all_components(self) -> bool:
        """
        Validate all registered components
        
        Returns:
            bool: True if all components are valid, False otherwise
        """
        all_valid = True
        
        for name, component in self.components_registry.items():
            if not self.validator.validate_component(component, f"registry_check_{name}"):
                self.logger.error(f"Component in registry is invalid: {name}")
                all_valid = False
        
        return all_valid
    
    def get_creation_report(self) -> UICreationReport:
        """
        Get a report of the UI creation process
        
        Returns:
            UICreationReport with statistics
        """
        validation_report = self.validator.get_validation_report()
        
        return UICreationReport(
            total_components=len(self.components_registry),
            valid_components=validation_report['valid_components'],
            failed_components=validation_report['failed_components'],
            event_handlers_setup=0,  # Will be updated by event handler system
            event_handlers_failed=0,  # Will be updated by event handler system
            creation_time=0.0,  # Will be updated by timing system
            errors=self.creation_errors.copy()
        )

class SafeGradioBlocks:
    """Safe wrapper around Gradio Blocks that validates components before finalization"""
    
    def __init__(self, **kwargs):
        self.blocks_kwargs = kwargs
        self.component_manager = UIComponentManager()
        self.blocks_instance = None
        self.logger = logging.getLogger(__name__)
        
    def __enter__(self):
        """Enter the Gradio Blocks context with validation"""
        try:
            self.blocks_instance = gr.Blocks(**self.blocks_kwargs)
            return self.blocks_instance.__enter__()
        except Exception as e:
            self.logger.error(f"Failed to create Gradio Blocks: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the Gradio Blocks context with component validation"""
        if self.blocks_instance is None:
            return False
            
        try:
            # Before calling the original __exit__, validate all components
            self._validate_blocks_components()
            
            # Call the original __exit__ method
            return self.blocks_instance.__exit__(exc_type, exc_val, exc_tb)
            
        except Exception as e:
            self.logger.error(f"Error during Gradio Blocks finalization: {e}")
            # Try to clean up any problematic components
            self._cleanup_invalid_components()
            
            # Try again with cleaned components
            try:
                return self.blocks_instance.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e2:
                self.logger.error(f"Failed even after cleanup: {e2}")
                raise e2
    
    def _validate_blocks_components(self):
        """Validate all components in the Gradio Blocks instance"""
        if not hasattr(self.blocks_instance, 'blocks') or not self.blocks_instance.blocks:
            self.logger.info("No blocks to validate")
            return
        
        self.logger.info(f"Validating {len(self.blocks_instance.blocks)} Gradio blocks...")
        
        invalid_blocks = []
        
        for i, block in enumerate(self.blocks_instance.blocks.values()):
            if not self.component_manager.validator.validate_component(block, f"gradio_block_{i}"):
                invalid_blocks.append((i, block))
        
        if invalid_blocks:
            self.logger.warning(f"Found {len(invalid_blocks)} invalid blocks")
            for i, block in invalid_blocks:
                self.logger.warning(f"Invalid block {i}: {type(block)} - {block}")
        else:
            self.logger.info("All Gradio blocks are valid")
    
    def _cleanup_invalid_components(self):
        """Clean up invalid components from the Gradio Blocks instance"""
        if not hasattr(self.blocks_instance, 'blocks') or not self.blocks_instance.blocks:
            return
        
        self.logger.info("Attempting to clean up invalid components...")
        
        # Get all event handlers (dependencies)
        if hasattr(self.blocks_instance, 'dependencies'):
            original_deps = self.blocks_instance.dependencies.copy()
            cleaned_deps = []
            
            for i, dep in enumerate(original_deps):
                try:
                    self.logger.debug(f"Processing dependency {i}: {type(dep)}")
                    
                    # Check if all inputs and outputs are valid
                    valid_inputs = []
                    valid_outputs = []
                    dependency_valid = True
                    
                    # Validate inputs
                    if hasattr(dep, 'inputs') and dep.inputs:
                        for j, inp in enumerate(dep.inputs):
                            if inp is None:
                                self.logger.warning(f"Dependency {i}: Removing None input at position {j}")
                                continue
                            if self.component_manager.validator.validate_component(inp, f"dep_{i}_input_{j}"):
                                valid_inputs.append(inp)
                            else:
                                self.logger.warning(f"Dependency {i}: Removing invalid input component at position {j}: {type(inp)}")
                    
                    # Validate outputs
                    if hasattr(dep, 'outputs') and dep.outputs:
                        for j, out in enumerate(dep.outputs):
                            if out is None:
                                self.logger.warning(f"Dependency {i}: Removing None output at position {j}")
                                continue
                            if self.component_manager.validator.validate_component(out, f"dep_{i}_output_{j}"):
                                valid_outputs.append(out)
                            else:
                                self.logger.warning(f"Dependency {i}: Removing invalid output component at position {j}: {type(out)}")
                    
                    # Update the dependency with valid components only
                    if hasattr(dep, 'inputs'):
                        original_input_count = len(dep.inputs) if dep.inputs else 0
                        dep.inputs = valid_inputs
                        if len(valid_inputs) != original_input_count:
                            self.logger.info(f"Dependency {i}: Cleaned inputs {original_input_count} -> {len(valid_inputs)}")
                    
                    if hasattr(dep, 'outputs'):
                        original_output_count = len(dep.outputs) if dep.outputs else 0
                        dep.outputs = valid_outputs
                        if len(valid_outputs) != original_output_count:
                            self.logger.info(f"Dependency {i}: Cleaned outputs {original_output_count} -> {len(valid_outputs)}")
                    
                    # Keep the dependency even if it has no inputs/outputs (some dependencies might be valid without them)
                    cleaned_deps.append(dep)
                    self.logger.debug(f"Dependency {i}: Kept after cleaning")
                        
                except Exception as e:
                    self.logger.error(f"Error cleaning dependency {i}: {e}")
                    # Skip this dependency entirely
                    continue
            
            # Update the dependencies list
            self.blocks_instance.dependencies = cleaned_deps
            self.logger.info(f"Cleaned dependencies: {len(original_deps)} -> {len(cleaned_deps)}")
            
        # Also clean up any None components in the blocks registry
        if hasattr(self.blocks_instance, 'blocks') and self.blocks_instance.blocks:
            original_blocks = dict(self.blocks_instance.blocks)
            cleaned_blocks = {}
            
            for block_id, block in original_blocks.items():
                if block is None:
                    self.logger.warning(f"Removing None block with ID: {block_id}")
                    continue
                if self.component_manager.validator.validate_component(block, f"block_{block_id}"):
                    cleaned_blocks[block_id] = block
                else:
                    self.logger.warning(f"Removing invalid block with ID: {block_id}, type: {type(block)}")
            
            self.blocks_instance.blocks = cleaned_blocks
            self.logger.info(f"Cleaned blocks registry: {len(original_blocks)} -> {len(cleaned_blocks)}")

def create_safe_gradio_blocks(**kwargs) -> SafeGradioBlocks:
    """
    Create a safe Gradio Blocks instance with component validation
    
    Args:
        **kwargs: Arguments for Gradio Blocks creation
        
    Returns:
        SafeGradioBlocks instance
    """
    return SafeGradioBlocks(**kwargs)

# Monkey patch to replace gr.Blocks with SafeGradioBlocks when needed
def patch_gradio_blocks():
    """Monkey patch Gradio Blocks to use safe validation"""
    original_blocks = gr.Blocks
    
    def safe_blocks_wrapper(*args, **kwargs):
        return SafeGradioBlocks(**kwargs)
    
    # Store original for restoration
    safe_blocks_wrapper._original_blocks = original_blocks
    
    return safe_blocks_wrapper

def restore_gradio_blocks(patched_blocks):
    """Restore original Gradio Blocks"""
    if hasattr(patched_blocks, '_original_blocks'):
        return patched_blocks._original_blocks
    return gr.Blocks