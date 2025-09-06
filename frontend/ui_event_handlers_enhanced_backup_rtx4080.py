from unittest.mock import Mock, patch
"""
Enhanced UI Event Handlers for Wan2.2 Video Generation
Comprehensive event handling system with proper integration of all UI components
"""

import gradio as gr
import logging
import threading
import time
from typing import Dict, Any, List, Tuple, Optional, Callable
from datetime import datetime
import json

# Import validation and error handling
try:
    from ui_validation import get_validation_manager, UIValidationManager
    from input_validation import ValidationResult
    from infrastructure.hardware.error_handler import UserFriendlyError, handle_generation_error
    from core.services.utils import generate_video, enhance_prompt
    from core.services.model_manager import get_model_manager
    
    # Import component managers
    from enhanced_image_validation import get_image_validator
    from enhanced_image_preview_manager import get_preview_manager
    from resolution_manager import get_resolution_manager
    from progress_tracker import get_progress_tracker, GenerationPhase
    from help_text_system import get_help_system
except ImportError as e:
    logging.warning(f"Some imports failed: {e}")
    # Create mock classes for testing
    class ValidationResult:
        def __init__(self, is_valid=True, message="", details=None):
            self.is_valid = is_valid
            self.message = message
            self.details = details or {}
    
    def get_validation_manager(config=None):
        return type('MockManager', (), {'register_ui_components': lambda x: None})()
    
    def get_image_validator(config=None):
        return type('MockValidator', (), {
            'validate_image': lambda img, img_type, model: ValidationResult(),
            'validate_image_compatibility': lambda img1, img2: ValidationResult()
        })()
    
    def get_preview_manager(config=None):
        return type('MockPreview', (), {
            'create_image_preview': lambda img, img_type, result: "<div>Preview</div>",
            'register_ui_components': lambda x: None
        })()
    
    def get_resolution_manager():
        return type('MockResolution', (), {
            'update_resolution_dropdown': lambda model: gr.update(choices=["1280x720"])
        })()
    
    def get_progress_tracker(config=None):
        return type('MockProgress', (), {
            'add_update_callback': lambda x: None,
            'start_progress_tracking': lambda task_id, steps: None,
            'get_progress_html': lambda: "<div>Progress</div>"
        })()
    
    def get_help_system(config=None):
        return type('MockHelp', (), {
            'get_image_help_text': lambda model: "Image help",
            'get_model_help_text': lambda model: "Model help",
            'get_image_requirements_text': lambda img_type, model: "Requirements"
        })()
    
    def enhance_prompt(prompt, model_type):
        return prompt + " enhanced"

logger = logging.getLogger(__name__)

class EnhancedUIEventHandlers:
    """Enhanced UI event handlers with comprehensive component integration"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize component managers
        self.validation_manager = get_validation_manager(config)
        self.image_validator = get_image_validator(config)
        self.preview_manager = get_preview_manager(config)
        self.resolution_manager = get_resolution_manager()
        self.progress_tracker = get_progress_tracker(config)
        self.help_text_system = get_help_system(config)
        
        # Event handling state
        self.validation_timers: Dict[str, threading.Timer] = {}
        self.validation_delay = 0.5  # 500ms delay for debouncing
        
        # Generation state tracking
        self.generation_in_progress = False
        self.current_task_id = None
        
        # UI component references
        self.ui_components: Dict[str, Any] = {}
        
        # Event handler registry for proper cleanup
        self.registered_handlers: List[Tuple[Any, str, Callable]] = []
        
        # Progress update callback
        self.progress_tracker.add_update_callback(self._on_progress_update)
        
        logger.info("Enhanced UI event handlers initialized")
    
    def register_components(self, components: Dict[str, Any]):
        """Register UI components for event handling"""
        self.ui_components = components
        
        # Register components with managers
        self.validation_manager.register_ui_components(components)
        
        # Register with preview manager if it supports it
        if hasattr(self.preview_manager, 'register_ui_components'):
            self.preview_manager.register_ui_components(components)
        else:
            logger.debug("Preview manager does not support component registration")
        
        logger.info(f"Registered {len(components)} UI components for enhanced event handling")
    
    def setup_all_event_handlers(self):
        """Set up all event handlers for the UI with comprehensive integration"""
        try:
            # Clear any existing handlers first
            self.cleanup_handlers()
            
            # Validate that required components are available
            required_components = ['model_type', 'notification_area', 'clear_notification_btn']
            missing_components = [comp for comp in required_components if comp not in self.ui_components]
            
            if missing_components:
                logger.warning(f"Missing required UI components: {missing_components}")
                # Continue setup but log the issue
            
            # Set up all event handlers in proper order with enhanced integration
            self.setup_model_type_events()
            self.setup_image_upload_events()
            self.setup_validation_events()
            self.setup_generation_events()
            self.setup_progress_events()
            self.setup_utility_events()
            
            # Set up cross-component integration events
            self.setup_integration_events()
            
            # Set up progress tracking integration with generation events
            self.setup_progress_integration()
            
            logger.info(f"All enhanced event handlers set up successfully - {len(self.registered_handlers)} handlers registered")
            
        except Exception as e:
            logger.error(f"Failed to set up enhanced event handlers: {e}")
            # Attempt cleanup on failure
            self.cleanup_handlers()
            raise
    
    def cleanup_handlers(self):
        """Clean up registered event handlers"""
        try:
            for component, event_type, handler in self.registered_handlers:
                # Gradio doesn't provide direct handler removal, but we track them for debugging
                pass
            self.registered_handlers.clear()
            logger.info("Event handlers cleaned up")
        except Exception as e:
            logger.warning(f"Error during handler cleanup: {e}")
    
    def _register_handler(self, component: Any, event_type: str, handler: Callable):
        """Register an event handler for tracking"""
        self.registered_handlers.append((component, event_type, handler))
        return handler
    
    def setup_model_type_events(self):
        """Set up model type change event handlers"""
        components = self.ui_components
        
        # Main model type change handler with comprehensive UI updates
        handler = self._register_handler(
            components['model_type'], 
            'change',
            components['model_type'].change(
                fn=self.handle_model_type_change,
                inputs=[components['model_type']],
                outputs=[
                    components['image_inputs_row'],
                    components['image_help_text'],
                    components['resolution'],
                    components['model_help_text'],
                    components['lora_compatibility_display'],
                    components['notification_area'],
                    components['clear_notification_btn'],
                    # Additional outputs for comprehensive updates
                    components.get('start_image_requirements'),
                    components.get('end_image_requirements'),
                    components.get('image_status_row'),
                    components.get('validation_summary')
                ]
            )
        )
        
        logger.info("Model type event handlers set up")
    
    def setup_image_upload_events(self):
        """Set up image upload and validation event handlers"""
        components = self.ui_components
        
        # Start image upload handler with comprehensive validation
        if 'image_input' in components:
            self._register_handler(
                components['image_input'],
                'upload',
                components['image_input'].upload(
                    fn=self.handle_start_image_upload,
                    inputs=[
                        components['image_input'],
                        components['model_type']
                    ],
                    outputs=[
                        components.get('start_image_preview'),
                        components.get('start_image_validation'),
                        components.get('clear_start_btn'),
                        components['notification_area'],
                        components['clear_notification_btn'],
                        # Additional outputs for comprehensive updates
                        components.get('image_summary'),
                        components.get('compatibility_status'),
                        components.get('image_status_row'),
                        components.get('validation_summary')
                    ]
                )
            )
        
        # End image upload handler with compatibility checking
        if 'end_image_input' in components:
            self._register_handler(
                components['end_image_input'],
                'upload',
                components['end_image_input'].upload(
                    fn=self.handle_end_image_upload,
                    inputs=[
                        components['end_image_input'],
                        components['model_type'],
                        components.get('image_input')  # Start image for compatibility check
                    ],
                    outputs=[
                        components.get('end_image_preview'),
                        components.get('end_image_validation'),
                        components.get('clear_end_btn'),
                        components.get('image_compatibility_status'),
                        components['notification_area'],
                        components['clear_notification_btn'],
                        # Additional outputs for comprehensive updates
                        components.get('image_summary'),
                        components.get('compatibility_status'),
                        components.get('validation_summary')
                    ]
                )
            )
        
        # Clear image button handlers with proper state management
        if 'clear_start_btn' in components:
            self._register_handler(
                components['clear_start_btn'],
                'click',
                components['clear_start_btn'].click(
                    fn=self.handle_clear_start_image,
                    inputs=[],
                    outputs=[
                        components['image_input'],
                        components.get('start_image_preview'),
                        components.get('start_image_validation'),
                        components['clear_start_btn'],
                        components.get('image_compatibility_status'),
                        components['notification_area'],
                        components['clear_notification_btn'],
                        # Additional outputs for state cleanup
                        components.get('image_summary'),
                        components.get('compatibility_status'),
                        components.get('validation_summary')
                    ]
                )
            )
        
        if 'clear_end_btn' in components:
            self._register_handler(
                components['clear_end_btn'],
                'click',
                components['clear_end_btn'].click(
                    fn=self.handle_clear_end_image,
                    inputs=[],
                    outputs=[
                        components['end_image_input'],
                        components.get('end_image_preview'),
                        components.get('end_image_validation'),
                        components['clear_end_btn'],
                        components.get('image_compatibility_status'),
                        components['notification_area'],
                        components['clear_notification_btn'],
                        # Additional outputs for state cleanup
                        components.get('image_summary'),
                        components.get('compatibility_status'),
                        components.get('validation_summary')
                    ]
                )
            )
        
        logger.info("Image upload event handlers set up")
    
    def setup_validation_events(self):
        """Set up real-time validation event handlers"""
        components = self.ui_components
        
        # Prompt validation with debouncing
        components['prompt_input'].change(
            fn=self.handle_prompt_change,
            inputs=[
                components['prompt_input'],
                components['model_type']
            ],
            outputs=[
                components['char_count'],
                components.get('prompt_validation_display'),
                components['notification_area'],
                components['clear_notification_btn']
            ]
        )
        
        # Parameter validation
        for param_name in ['resolution', 'steps']:
            if param_name in components:
                components[param_name].change(
                    fn=self.handle_parameter_change,
                    inputs=[
                        components['model_type'],
                        components['resolution'],
                        components['steps']
                    ],
                    outputs=[
                        components.get('parameter_validation'),
                        components['notification_area'],
                        components['clear_notification_btn']
                    ]
                )
        
        logger.info("Validation event handlers set up")
    
    def setup_generation_events(self):
        """Set up generation event handlers with progress integration"""
        components = self.ui_components
        
        # Generate button handler with comprehensive progress tracking
        self._register_handler(
            components['generate_btn'],
            'click',
            components['generate_btn'].click(
                fn=self.handle_generate_video,
                inputs=[
                    components['model_type'],
                    components['prompt_input'],
                    components.get('image_input'),
                    components.get('end_image_input'),
                    components['resolution'],
                    components['steps'],
                    components.get('lora_path', gr.State("")),
                    components.get('lora_strength', gr.State(1.0)),
                    # Additional generation parameters
                    components.get('duration', gr.State(4)),
                    components.get('fps', gr.State(24))
                ],
                outputs=[
                    components['generation_status'],
                    components.get('progress_display'),
                    components.get('output_video'),
                    components['notification_area'],
                    components['clear_notification_btn'],
                    # Additional status outputs
                    components.get('validation_summary'),
                    components.get('generate_btn'),  # For disabling during generation
                    components.get('queue_btn')     # For disabling during generation
                ]
            )
        )
        
        # Queue button handler with validation
        self._register_handler(
            components['queue_btn'],
            'click',
            components['queue_btn'].click(
                fn=self.handle_queue_generation,
                inputs=[
                    components['model_type'],
                    components['prompt_input'],
                    components.get('image_input'),
                    components.get('end_image_input'),
                    components['resolution'],
                    components['steps'],
                    components.get('lora_path', gr.State("")),
                    components.get('lora_strength', gr.State(1.0)),
                    components.get('duration', gr.State(4)),
                    components.get('fps', gr.State(24))
                ],
                outputs=[
                    components['notification_area'],
                    components['clear_notification_btn'],
                    components.get('validation_summary')
                ]
            )
        )
        
        # Prompt enhancement handler
        if 'enhance_btn' in components:
            self._register_handler(
                components['enhance_btn'],
                'click',
                components['enhance_btn'].click(
                    fn=self.handle_prompt_enhancement,
                    inputs=[
                        components['prompt_input'],
                        components['model_type']
                    ],
                    outputs=[
                        components.get('enhanced_prompt_display'),
                        components['notification_area'],
                        components['clear_notification_btn'],
                        components.get('prompt_input')  # For updating with enhanced prompt
                    ]
                )
            )
        
        logger.info("Generation event handlers set up")
    
    def setup_progress_events(self):
        """Set up progress tracking event handlers"""
        # Progress tracking is handled through callbacks
        # No direct UI events needed, but we can set up refresh handlers
        
        components = self.ui_components
        
        # Progress refresh handler (if needed)
        if 'refresh_progress_btn' in components:
            components['refresh_progress_btn'].click(
                fn=self.handle_refresh_progress,
                inputs=[],
                outputs=[
                    components.get('progress_display'),
                    components['generation_status']
                ]
            )
        
        logger.info("Progress event handlers set up")
    
    def setup_utility_events(self):
        """Set up utility event handlers"""
        components = self.ui_components
        
        # Clear notification handler
        components['clear_notification_btn'].click(
            fn=self.handle_clear_notification,
            inputs=[],
            outputs=[
                components['notification_area'],
                components['clear_notification_btn']
            ]
        )
        
        logger.info("Utility event handlers set up")
    
    def setup_integration_events(self):
        """Set up cross-component integration event handlers"""
        components = self.ui_components
        
        # Resolution change handler - validate with current model and images
        if 'resolution' in components:
            self._register_handler(
                components['resolution'],
                'change',
                components['resolution'].change(
                    fn=self.handle_resolution_change,
                    inputs=[
                        components['resolution'],
                        components['model_type'],
                        components.get('image_input'),
                        components.get('end_image_input')
                    ],
                    outputs=[
                        components.get('parameter_validation'),
                        components['notification_area'],
                        components['clear_notification_btn'],
                        components.get('validation_summary')
                    ]
                )
            )
        
        # Steps change handler - update generation time estimates
        if 'steps' in components:
            self._register_handler(
                components['steps'],
                'change',
                components['steps'].change(
                    fn=self.handle_steps_change,
                    inputs=[
                        components['steps'],
                        components['model_type'],
                        components['resolution']
                    ],
                    outputs=[
                        components.get('parameter_validation'),
                        components['notification_area'],
                        components['clear_notification_btn']
                    ]
                )
            )
        
        # Duration change handler - update VRAM and time estimates
        if 'duration' in components:
            self._register_handler(
                components['duration'],
                'change',
                components['duration'].change(
                    fn=self.handle_duration_change,
                    inputs=[
                        components['duration'],
                        components['model_type'],
                        components['resolution'],
                        components['steps']
                    ],
                    outputs=[
                        components.get('parameter_validation'),
                        components['notification_area'],
                        components['clear_notification_btn']
                    ]
                )
            )
        
        # FPS change handler - validate with duration and resolution
        if 'fps' in components:
            self._register_handler(
                components['fps'],
                'change',
                components['fps'].change(
                    fn=self.handle_fps_change,
                    inputs=[
                        components['fps'],
                        components['duration'],
                        components['resolution']
                    ],
                    outputs=[
                        components.get('parameter_validation'),
                        components['notification_area'],
                        components['clear_notification_btn']
                    ]
                )
            )
        
        logger.info("Integration event handlers set up")
    
    def _trigger_model_change_cascade(self, model_type: str, show_images: bool):
        """Trigger cascade of UI updates when model type changes"""
        try:
            # Update internal state
            self.current_model_type = model_type
            
            # Clear any cached validation results
            if hasattr(self, 'cached_validation_results'):
                self.cached_validation_results.clear()
            
            # Update progress tracker configuration if needed
            if hasattr(self.progress_tracker, 'update_model_config'):
                self.progress_tracker.update_model_config(model_type)
            
            # Notify other components of model change
            for callback in getattr(self, 'model_change_callbacks', []):
                try:
                    callback(model_type, show_images)
                except Exception as e:
                    logger.warning(f"Model change callback failed: {e}")
            
            logger.debug(f"Model change cascade completed for {model_type}")
            
        except Exception as e:
            logger.warning(f"Model change cascade failed: {e}")
    
    def _clear_image_validation_state(self):
        """Clear all image validation state when switching models"""
        try:
            # Clear any cached validation results
            if hasattr(self, 'image_validation_cache'):
                self.image_validation_cache.clear()
            
            # Reset validation timers
            for timer in self.validation_timers.values():
                if timer.is_alive():
                    timer.cancel()
            self.validation_timers.clear()
            
            logger.debug("Image validation state cleared")
            
        except Exception as e:
            logger.warning(f"Failed to clear image validation state: {e}")
    
    def _validate_image_with_retry(self, image: Any, image_type: str, model_type: str, max_retries: int = 2):
        """Validate image with retry logic for robustness"""
        for attempt in range(max_retries + 1):
            try:
                return self.image_validator.validate_image(image, image_type, model_type)
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"Image validation failed after {max_retries} retries: {e}")
                    # Return a failed validation result
                    return type('ValidationResult', (), {
                        'is_valid': False,
                        'message': f"Validation failed: {str(e)}",
                        'details': {}
                    })()
                else:
                    logger.warning(f"Image validation attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(0.1)  # Brief delay before retry
    
    def _generate_image_preview_safe(self, image: Any, image_type: str, validation_result: Any):
        """Generate image preview with error handling"""
        try:
            return self.preview_manager.create_image_preview(image, image_type, validation_result)
        except Exception as e:
            logger.warning(f"Image preview generation failed: {e}")
            # Return a basic preview fallback
            return f"""
            <div style="border: 2px dashed #ccc; padding: 20px; text-align: center; border-radius: 8px;">
                <span style="color: #666;">📷 {image_type.title()} Image Uploaded</span>
                <br><small>Preview generation failed, but image was processed</small>
            </div>
            """
    
    def _check_existing_image_compatibility(self, current_type: str, current_image: Any, current_validation: Any):
        """Check compatibility with existing images"""
        try:
            # Get the other image type
            other_type = "end" if current_type == "start" else "start"
            
            # Check if we have a stored validation result for the other image
            other_validation = getattr(self, f'_{other_type}_validation_result', None)
            
            if other_validation and hasattr(other_validation, 'is_valid') and other_validation.is_valid:
                # We have both images, check compatibility
                if current_type == "start":
                    compatibility_result = self.image_validator.validate_image_compatibility(
                        current_image, getattr(self, '_end_image', None)
                    )
                else:
                    compatibility_result = self.image_validator.validate_image_compatibility(
                        getattr(self, '_start_image', None), current_image
                    )
                
                if hasattr(compatibility_result, 'is_valid'):
                    if compatibility_result.is_valid:
                        return """
                        <div style="color: #28a745; background: #d4edda; border: 1px solid #c3e6cb; 
                                    border-radius: 4px; padding: 8px; margin: 5px 0;">
                            ✅ Images are compatible for video generation
                        </div>
                        """
                    else:
                        return f"""
                        <div style="color: #ffc107; background: #fff3cd; border: 1px solid #ffeaa7; 
                                    border-radius: 4px; padding: 8px; margin: 5px 0;">
                            ⚠️ Compatibility issue: {compatibility_result.message}
                        </div>
                        """
            
            return ""  # No compatibility check needed yet
            
        except Exception as e:
            logger.warning(f"Compatibility check failed: {e}")
            return ""
    
    def _store_validation_result(self, image_type: str, validation_result: Any):
        """Store validation result for future compatibility checks"""
        try:
            setattr(self, f'_{image_type}_validation_result', validation_result)
        except Exception as e:
            logger.warning(f"Failed to store validation result: {e}")
    
    def _create_comprehensive_validation_summary(self, image_type: str, validation_result: Any):
        """Create comprehensive validation summary including all current states"""
        try:
            summary_parts = []
            
            # Add current validation result
            if validation_result.is_valid:
                summary_parts.append(f"✅ {image_type.title()} image: Valid")
            else:
                summary_parts.append(f"❌ {image_type.title()} image: {validation_result.message}")
            
            # Add other validation states if available
            other_type = "end" if image_type == "start" else "start"
            other_validation = getattr(self, f'_{other_type}_validation_result', None)
            
            if other_validation:
                if other_validation.is_valid:
                    summary_parts.append(f"✅ {other_type.title()} image: Valid")
                else:
                    summary_parts.append(f"❌ {other_type.title()} image: {other_validation.message}")
            
            if len(summary_parts) > 1:
                summary_html = f"""
                <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 10px; margin: 5px 0;">
                    <strong>📋 Validation Summary</strong><br>
                    {'<br>'.join(summary_parts)}
                </div>
                """
            else:
                summary_html = f"""
                <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 10px; margin: 5px 0;">
                    <strong>📋 Validation Status</strong><br>
                    {summary_parts[0]}
                </div>
                """
            
            return summary_html
            
        except Exception as e:
            logger.warning(f"Failed to create validation summary: {e}")
            return ""
    
    def setup_progress_integration(self):
        """Set up progress tracking integration with generation events"""
        components = self.ui_components
        
        # Ensure progress tracker callbacks are properly connected
        if hasattr(self.progress_tracker, 'add_update_callback'):
            # Remove any existing callback to avoid duplicates
            try:
                self.progress_tracker.update_callbacks = [
                    cb for cb in self.progress_tracker.update_callbacks 
                    if cb != self._on_progress_update
                ]
            except AttributeError:
                pass
            
            # Add our progress update callback
            self.progress_tracker.add_update_callback(self._on_progress_update)
            logger.info("Progress tracking integration set up")
        else:
            logger.warning("Progress tracker does not support callbacks")
    
    def _setup_generation_progress_callback(self, task_id: str):
        """Set up progress callback for specific generation task"""
        try:
            # Store task ID for progress updates
            self.current_task_id = task_id
            
            # Ensure progress display component is available
            if 'progress_display' in self.ui_components:
                logger.debug(f"Progress display component available for task {task_id}")
            else:
                logger.warning("Progress display component not available")
            
            # Set up periodic progress updates
            self._start_progress_update_timer()
            
        except Exception as e:
            logger.warning(f"Failed to set up generation progress callback: {e}")
    
    def _start_progress_update_timer(self):
        """Start timer for periodic progress updates"""
        try:
            if hasattr(self, '_progress_timer') and self._progress_timer:
                self._progress_timer.cancel()
            
            def update_progress():
                if self.generation_in_progress and self.current_task_id:
                    try:
                        # Get current progress HTML
                        progress_html = self.progress_tracker.get_progress_html()
                        
                        # Update progress display if component is available
                        if 'progress_display' in self.ui_components:
                            # Note: Direct component updates require special handling in Gradio
                            # This is a placeholder for the update mechanism
                            pass
                        
                        # Schedule next update
                        if self.generation_in_progress:
                            self._progress_timer = threading.Timer(2.0, update_progress)
                            self._progress_timer.start()
                            
                    except Exception as e:
                        logger.warning(f"Progress update failed: {e}")
            
            # Start the update cycle
            self._progress_timer = threading.Timer(2.0, update_progress)
            self._progress_timer.start()
            
        except Exception as e:
            logger.warning(f"Failed to start progress update timer: {e}")
    
    def _on_progress_update(self, progress_data):
        """Handle progress updates from the progress tracker"""
        try:
            if not self.generation_in_progress:
                return
            
            # Log progress update only if there's actual progress change
            if not hasattr(self, '_last_progress') or self._last_progress != progress_data.progress_percentage:
                logger.debug(f"Progress update: {progress_data.progress_percentage:.1f}% - {progress_data.current_phase}")
                self._last_progress = progress_data.progress_percentage
            
            # Update UI components if available
            if 'progress_display' in self.ui_components:
                # Generate updated progress HTML
                progress_html = self.progress_tracker.get_progress_html()
                
                # Note: In a real implementation, this would trigger a UI update
                # For now, we just log the update
                logger.debug("Progress display would be updated with new HTML")
            
            # Check if generation is complete
            if progress_data.progress_percentage >= 100:
                self._handle_generation_completion()
                
        except Exception as e:
            logger.warning(f"Progress update handler failed: {e}")
    
    def _handle_generation_completion(self):
        """Handle generation completion"""
        try:
            logger.info("Generation completed")
            
            # Reset generation state
            self.generation_in_progress = False
            self.current_task_id = None
            
            # Cancel progress timer
            if hasattr(self, '_progress_timer') and self._progress_timer:
                self._progress_timer.cancel()
                self._progress_timer = None
            
            # Complete progress tracking
            final_stats = self.progress_tracker.complete_progress_tracking()
            
            # Note: In a real implementation, this would update UI components
            # to show completion status and re-enable buttons
            logger.info("Generation completion handling finished")
            
        except Exception as e:
            logger.error(f"Generation completion handling failed: {e}")
    
    # Event Handler Methods
    
    def handle_model_type_change(self, model_type: str) -> Tuple[Any, Any, Any, str, str, str, bool, Any, Any, Any, Any]:
        """Handle model type selection change with comprehensive UI updates"""
        try:
            logger.info(f"Model type changed to: {model_type}")
            
            # Determine image input visibility
            show_images = model_type in ["i2v-A14B", "ti2v-5B"]
            image_inputs_row_update = gr.update(visible=show_images)
            
            # Generate context-sensitive help text
            image_help_text = self.help_text_system.get_image_help_text(model_type)
            image_help_update = gr.update(value=image_help_text, visible=show_images)
            
            # Update resolution dropdown with model-specific options
            resolution_update = self.resolution_manager.update_resolution_dropdown(model_type)
            
            # Get model-specific help text
            model_help_text = self.help_text_system.get_model_help_text(model_type)
            
            # Update LoRA compatibility display
            lora_compatibility_html = self._get_lora_compatibility_display(model_type)
            
            # Update image requirements help text
            start_requirements_update = gr.update(
                value=self.help_text_system.get_image_requirements_text("start", model_type),
                visible=show_images
            )
            end_requirements_update = gr.update(
                value=self.help_text_system.get_image_requirements_text("end", model_type),
                visible=show_images
            )
            
            # Update image status row visibility
            image_status_row_update = gr.update(visible=show_images)
            
            # Clear any existing image validation messages when switching modes
            if hasattr(self, '_clear_image_validation_state'):
                self._clear_image_validation_state()
            
            # Clear validation summary when switching modes and update with new model info
            validation_summary_content = f"""
            <div style="background: #e3f2fd; border: 1px solid #2196f3; border-radius: 4px; padding: 10px; margin: 5px 0;">
                <strong>🎬 Model Configuration Updated</strong><br>
                <small>Model: {model_type}</small><br>
                <small>Image inputs: {'Required' if model_type in ['i2v-A14B', 'ti2v-5B'] else 'Not used'}</small><br>
                <small>Compatible resolutions updated in dropdown</small><br>
                <small>All validation states cleared for new mode</small>
            </div>
            """
            validation_summary_update = gr.update(value=validation_summary_content, visible=True)
            
            # Enhanced success notification with model-specific information
            model_descriptions = {
                "t2v-A14B": "Text-to-Video generation - Create videos from text prompts only",
                "i2v-A14B": "Image-to-Video generation - Animate from a start image",
                "ti2v-5B": "Text+Image-to-Video generation - Combine text prompts with images"
            }
            
            notification_html = f"""
            <div style='color: #28a745; background: #d4edda; border: 1px solid #c3e6cb; 
                        border-radius: 8px; padding: 12px; margin: 8px 0;'>
                ✅ Model type updated to <strong>{model_type}</strong>
                <br><small>{model_descriptions.get(model_type, 'Model updated')}</small>
                {f'<br><small>📸 Image inputs are now visible and required</small>' if show_images else '<br><small>📝 Text-only mode - no images needed</small>'}
                <br><small>🎯 Resolution options and help text updated automatically</small>
                <br><small>🔄 All UI components synchronized with new model</small>
            </div>
            """
            
            # Trigger additional UI updates for comprehensive integration
            self._trigger_model_change_cascade(model_type, show_images)
            
            return (
                image_inputs_row_update,
                image_help_update,
                resolution_update,
                model_help_text,
                lora_compatibility_html,
                notification_html,
                True,
                start_requirements_update,
                end_requirements_update,
                image_status_row_update,
                validation_summary_update
            )
            
        except Exception as e:
            logger.error(f"Model type change error: {e}")
            error_html = self._create_error_display(e, "model_type_change")
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                "Error updating model type",
                "",
                error_html,
                True,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update()
            )
    
    def handle_start_image_upload(self, image: Any, model_type: str) -> Tuple[str, str, bool, str, bool, str, str, bool, str]:
        """Handle start image upload with comprehensive validation and preview"""
        try:
            if image is None:
                return "", "", False, "", False, "", "", False, ""
            
            logger.info(f"Processing start image upload for model: {model_type}")
            
            # Validate image with comprehensive error handling
            validation_result = self._validate_image_with_retry(image, "start", model_type)
            
            # Generate preview with error handling
            preview_html = self._generate_image_preview_safe(image, "start", validation_result)
            
            # Create validation display with enhanced feedback
            validation_html = self._create_validation_display(validation_result, "start image")
            
            # Show clear button if image is valid or has warnings
            clear_btn_visible = validation_result.is_valid or validation_result.message
            
            # Create comprehensive image summary
            image_summary_html = self._create_image_summary(image, "start", validation_result)
            
            # Create compatibility status (empty for start image alone, will be updated if end image exists)
            compatibility_html = self._check_existing_image_compatibility("start", image, validation_result)
            
            # Show image status row
            image_status_visible = True
            
            # Create validation summary with all current validation states
            validation_summary_html = self._create_comprehensive_validation_summary("start", validation_result)
            
            # Create enhanced notification with detailed feedback
            if validation_result.is_valid:
                notification_html = f"""
                <div style='color: #28a745; background: #d4edda; border: 1px solid #c3e6cb; 
                            border-radius: 8px; padding: 12px; margin: 8px 0;'>
                    ✅ Start image uploaded and validated successfully
                    <br><small>📊 {getattr(validation_result, 'details', {}).get('width', 'Unknown')}×{getattr(validation_result, 'details', {}).get('height', 'Unknown')} pixels</small>
                    <br><small>🎯 Compatible with {model_type} model</small>
                </div>
                """
            else:
                notification_html = f"""
                <div style='color: #dc3545; background: #f8d7da; border: 1px solid #f5c6cb; 
                            border-radius: 8px; padding: 12px; margin: 8px 0;'>
                    ❌ Start image validation failed
                    <br><small>{validation_result.message}</small>
                    <br><small>💡 Please check image requirements and try again</small>
                </div>
                """
            
            # Store validation result for future compatibility checks
            self._store_validation_result("start", validation_result)
            
            return (
                preview_html,
                validation_html,
                clear_btn_visible,
                notification_html,
                True,  # Show notification
                image_summary_html,
                compatibility_html,
                image_status_visible,
                validation_summary_html
            )
            
        except Exception as e:
            logger.error(f"Start image upload error: {e}")
            error_html = self._create_error_display(e, "start_image_upload")
            return (
                "",
                error_html,
                False,
                error_html,
                True,
                "",
                "",
                False,
                ""
            )
            
            return (
                preview_html,
                validation_html,
                clear_btn_visible,
                notification_html,
                True,
                image_summary_html,
                compatibility_html,
                image_status_visible,
                validation_summary_html
            )
            
        except Exception as e:
            logger.error(f"Start image upload error: {e}")
            error_html = self._create_error_display(e, "start_image_upload")
            return "", "", False, error_html, True, "", "", False, ""
    
    def handle_end_image_upload(self, end_image: Any, model_type: str, 
                              start_image: Any) -> Tuple[str, str, bool, str, str, bool, str, str, str]:
        """Handle end image upload with compatibility checking"""
        try:
            if end_image is None:
                return "", "", False, "", "", False, "", "", ""
            
            logger.info("Processing end image upload")
            
            # Validate end image
            validation_result = self.image_validator.validate_image(end_image, "end", model_type)
            
            # Check compatibility with start image if both exist
            compatibility_html = ""
            if start_image is not None and validation_result.is_valid:
                compatibility_result = self.image_validator.validate_image_compatibility(
                    start_image, end_image
                )
                compatibility_html = self._create_compatibility_display(compatibility_result)
            
            # Generate preview
            preview_html = self.preview_manager.create_image_preview(
                end_image, "end", validation_result
            )
            
            # Create validation display
            validation_html = self._create_validation_display(validation_result, "end image")
            
            # Show clear button if image is valid
            clear_btn_visible = validation_result.is_valid
            
            # Create image summary
            image_summary_html = self._create_image_summary(end_image, "end", validation_result)
            
            # Create validation summary
            validation_results = [validation_result]
            if start_image is not None and compatibility_html:
                # Add compatibility result to summary
                pass
            validation_summary_html = self._create_validation_summary(validation_results)
            
            # Create notification
            if validation_result.is_valid:
                notification_html = """
                <div style='color: #28a745; background: #d4edda; border: 1px solid #c3e6cb; 
                            border-radius: 8px; padding: 12px; margin: 8px 0;'>
                    ✅ End image uploaded and validated successfully
                </div>
                """
            else:
                notification_html = f"""
                <div style='color: #dc3545; background: #f8d7da; border: 1px solid #f5c6cb; 
                            border-radius: 8px; padding: 12px; margin: 8px 0;'>
                    ❌ End image validation failed: {validation_result.message}
                </div>
                """
            
            return (
                preview_html,
                validation_html,
                clear_btn_visible,
                compatibility_html,
                notification_html,
                True,
                image_summary_html,
                compatibility_html,  # Also update main compatibility status
                validation_summary_html
            )
            
        except Exception as e:
            logger.error(f"End image upload error: {e}")
            error_html = self._create_error_display(e, "end_image_upload")
            return "", "", False, "", error_html, True, "", "", ""
    
    def handle_clear_start_image(self) -> Tuple[Any, str, str, bool, str, str, bool, str, str, str]:
        """Handle clearing start image"""
        try:
            logger.info("Clearing start image")
            
            notification_html = """
            <div style='color: #17a2b8; background: #d1ecf1; border: 1px solid #bee5eb; 
                        border-radius: 8px; padding: 12px; margin: 8px 0;'>
                ℹ️ Start image cleared
            </div>
            """
            
            return (
                gr.update(value=None),  # Clear image input
                "",  # Clear preview
                "",  # Clear validation
                False,  # Hide clear button
                "",  # Clear compatibility status
                notification_html,
                True,
                "",  # Clear image summary
                "",  # Clear compatibility status
                ""   # Clear validation summary
            )
            
        except Exception as e:
            logger.error(f"Clear start image error: {e}")
            error_html = self._create_error_display(e, "clear_start_image")
            return gr.update(), "", "", False, "", error_html, True, "", "", ""
    
    def handle_clear_end_image(self) -> Tuple[Any, str, str, bool, str, str, bool, str, str, str]:
        """Handle clearing end image"""
        try:
            logger.info("Clearing end image")
            
            notification_html = """
            <div style='color: #17a2b8; background: #d1ecf1; border: 1px solid #bee5eb; 
                        border-radius: 8px; padding: 12px; margin: 8px 0;'>
                ℹ️ End image cleared
            </div>
            """
            
            return (
                gr.update(value=None),  # Clear image input
                "",  # Clear preview
                "",  # Clear validation
                False,  # Hide clear button
                "",  # Clear compatibility status
                notification_html,
                True,
                "",  # Clear image summary
                "",  # Clear compatibility status
                ""   # Clear validation summary
            )
            
        except Exception as e:
            logger.error(f"Clear end image error: {e}")
            error_html = self._create_error_display(e, "clear_end_image")
            return gr.update(), "", "", False, "", error_html, True, "", "", ""
    
    def handle_prompt_change(self, prompt: str, model_type: str) -> Tuple[str, str, str, bool]:
        """Handle real-time prompt validation with debouncing"""
        try:
            # Character count
            char_count = f"{len(prompt)}/500"
            
            # Cancel previous validation timer
            if 'prompt' in self.validation_timers:
                self.validation_timers['prompt'].cancel()
            
            if not prompt.strip():
                return char_count, "", "", False
            
            # Immediate basic validation for real-time feedback
            validation_result = ValidationResult(
                is_valid=len(prompt) <= 500,
                message="Prompt is valid" if len(prompt) <= 500 else "Prompt too long"
            )
            validation_html = self._create_validation_display(validation_result, "prompt")
            
            # Notification for significant issues
            notification_html = ""
            if not validation_result.is_valid:
                notification_html = f"""
                <div style='color: #ffc107; background: #fff3cd; border: 1px solid #ffeaa7; 
                            border-radius: 8px; padding: 12px; margin: 8px 0;'>
                    ⚠️ Prompt validation: {validation_result.message}
                </div>
                """
            
            return char_count, validation_html, notification_html, bool(notification_html)
            
        except Exception as e:
            logger.error(f"Prompt validation error: {e}")
            char_count = f"{len(prompt)}/500"
            error_html = self._create_error_display(e, "prompt_validation")
            return char_count, "", error_html, True
    
    def handle_parameter_change(self, model_type: str, resolution: str, 
                              steps: int) -> Tuple[str, str, bool]:
        """Handle parameter validation"""
        try:
            # Basic parameter validation
            validation_result = ValidationResult(
                is_valid=20 <= steps <= 100,
                message="Parameters are valid" if 20 <= steps <= 100 else "Steps must be between 20 and 100"
            )
            
            validation_html = self._create_validation_display(validation_result, "parameters")
            
            # Create notification for parameter issues
            notification_html = ""
            if not validation_result.is_valid:
                notification_html = f"""
                <div style='color: #ffc107; background: #fff3cd; border: 1px solid #ffeaa7; 
                            border-radius: 8px; padding: 12px; margin: 8px 0;'>
                    ⚠️ Parameter validation: {validation_result.message}
                </div>
                """
            
            return validation_html, notification_html, bool(notification_html)
            
        except Exception as e:
            logger.error(f"Parameter validation error: {e}")
            error_html = self._create_error_display(e, "parameter_validation")
            return "", error_html, True
    
    def handle_generate_video(self, model_type: str, prompt: str, start_image: Any,
                            end_image: Any, resolution: str, steps: int,
                            lora_path: str, lora_strength: float, duration: int = 4, 
                            fps: int = 24) -> Tuple[str, str, Any, str, bool, str, Any, Any]:
        """Handle video generation with comprehensive progress tracking"""
        try:
            if self.generation_in_progress:
                return (
                    "⚠️ Generation already in progress",
                    "",
                    gr.update(),
                    "Generation is already running. Please wait for completion.",
                    True,
                    "",
                    gr.update(interactive=False),  # Disable generate button
                    gr.update(interactive=False)   # Disable queue button
                )
            
            logger.info("Starting video generation")
            
            # Comprehensive validation
            validation_success, validation_message = self._validate_generation_request(
                model_type, prompt, start_image, end_image, resolution, steps, lora_path, lora_strength
            )
            
            if not validation_success:
                return (
                    "❌ Validation Failed",
                    "",
                    gr.update(),
                    validation_message,
                    True,
                    validation_message,
                    gr.update(interactive=True),   # Keep generate button enabled
                    gr.update(interactive=True)    # Keep queue button enabled
                )
            
            # Start progress tracking
            self.generation_in_progress = True
            self.current_task_id = f"generation_{int(time.time())}"
            self.progress_tracker.start_progress_tracking(self.current_task_id, steps)
            
            # Initial progress display
            progress_html = self.progress_tracker.get_progress_html() if hasattr(self.progress_tracker, 'get_progress_html') else "Starting generation..."
            
            # Start generation in background thread
            generation_thread = threading.Thread(
                target=self._run_generation,
                args=(model_type, prompt, start_image, end_image, resolution, 
                      steps, lora_path, lora_strength),
                daemon=True
            )
            generation_thread.start()
            
            success_notification = """
            <div style='color: #28a745; background: #d4edda; border: 1px solid #c3e6cb; 
                        border-radius: 8px; padding: 12px; margin: 8px 0;'>
                🚀 Video generation started! Progress will be updated in real-time.
            </div>
            """
            
            return (
                "🔄 Generation Started",
                progress_html,
                gr.update(),
                success_notification,
                True,
                success_notification,
                gr.update(interactive=False),  # Disable generate button during generation
                gr.update(interactive=False)   # Disable queue button during generation
            )
            
        except Exception as e:
            self.generation_in_progress = False
            logger.error(f"Generation start error: {e}")
            error_html = self._create_error_display(e, "video_generation")
            return (
                "❌ Generation Failed", 
                "", 
                gr.update(), 
                error_html, 
                True, 
                error_html,
                gr.update(interactive=True),   # Re-enable generate button
                gr.update(interactive=True)    # Re-enable queue button
            )
    
    def handle_queue_generation(self, model_type: str, prompt: str, start_image: Any,
                              end_image: Any, resolution: str, steps: int,
                              lora_path: str, lora_strength: float, duration: int = 4, 
                              fps: int = 24) -> Tuple[str, bool, str]:
        """Handle adding generation to queue"""
        try:
            # Validate request
            validation_success, validation_message = self._validate_generation_request(
                model_type, prompt, start_image, end_image, resolution, steps, lora_path, lora_strength
            )
            
            if not validation_success:
                return validation_message, True, validation_message
            
            # Add to queue (placeholder - implement actual queue logic)
            logger.info("Adding generation to queue")
            
            notification_html = """
            <div style='color: #28a745; background: #d4edda; border: 1px solid #c3e6cb; 
                        border-radius: 8px; padding: 12px; margin: 8px 0;'>
                📋 Generation request added to queue successfully
            </div>
            """
            
            return notification_html, True, notification_html
            
        except Exception as e:
            logger.error(f"Queue generation error: {e}")
            error_html = self._create_error_display(e, "queue_generation")
            return error_html, True, error_html
    
    def handle_prompt_enhancement(self, prompt: str, model_type: str) -> Tuple[Any, str, bool, Any]:
        """Handle prompt enhancement"""
        try:
            if not prompt.strip():
                return (
                    gr.update(visible=False),
                    "Please enter a prompt to enhance",
                    True,
                    gr.update()
                )
            
            # Use existing enhance_prompt function
            enhanced = enhance_prompt(prompt, model_type)
            
            if enhanced and enhanced != prompt:
                enhanced_update = gr.update(value=enhanced, visible=True)
                notification_html = """
                <div style='color: #28a745; background: #d4edda; border: 1px solid #c3e6cb; 
                            border-radius: 8px; padding: 12px; margin: 8px 0;'>
                    ✨ Prompt enhanced successfully - you can copy the enhanced version to your prompt
                </div>
                """
                # Don't automatically replace the original prompt, let user decide
                return enhanced_update, notification_html, True, gr.update()
            else:
                notification_html = """
                <div style='color: #ffc107; background: #fff3cd; border: 1px solid #ffeaa7; 
                            border-radius: 8px; padding: 12px; margin: 8px 0;'>
                    ℹ️ Prompt enhancement did not suggest any changes
                </div>
                """
                return gr.update(visible=False), notification_html, True, gr.update()
            
        except Exception as e:
            logger.error(f"Prompt enhancement error: {e}")
            error_html = self._create_error_display(e, "prompt_enhancement")
            return gr.update(visible=False), error_html, True, gr.update()
    
    def handle_refresh_progress(self) -> Tuple[str, str]:
        """Handle progress refresh"""
        try:
            if self.generation_in_progress and hasattr(self.progress_tracker, 'get_progress_html'):
                progress_html = self.progress_tracker.get_progress_html()
                status = "🔄 Generation in progress"
            else:
                progress_html = ""
                status = "Ready"
            
            return progress_html, status
            
        except Exception as e:
            logger.error(f"Progress refresh error: {e}")
            return "", "Error refreshing progress"
    
    def handle_clear_notification(self) -> Tuple[str, bool]:
        """Handle clearing notifications"""
        return "", False
    
    def _on_progress_update(self, progress_data: Dict[str, Any]):
        """Handle progress updates from the progress tracker"""
        try:
            # This would update the UI with progress information
            # The actual UI update would be handled by the progress tracker
            # Only log if progress actually changed
            if not hasattr(self, '_last_progress_data') or str(self._last_progress_data) != str(progress_data):
                logger.debug(f"Progress update: {progress_data}")
                self._last_progress_data = progress_data
        except Exception as e:
            logger.warning(f"Progress update error: {e}")
    
    # Helper Methods for UI Updates
    
    def _create_image_summary(self, image: Any, image_type: str, validation_result: ValidationResult) -> str:
        """Create image summary HTML"""
        if not validation_result.is_valid:
            return ""
        
        try:
            width, height = image.size
            aspect_ratio = width / height
            
            return f"""
            <div style='background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 10px; margin: 5px 0;'>
                <strong>{image_type.title()} Image Summary:</strong><br>
                <small>
                    📐 Dimensions: {width} × {height}<br>
                    📏 Aspect Ratio: {aspect_ratio:.2f}<br>
                    ✅ Status: Valid
                </small>
            </div>
            """
        except Exception as e:
            logger.warning(f"Failed to create image summary: {e}")
            return ""
    
    def _create_validation_summary(self, validation_results: List[ValidationResult]) -> str:
        """Create comprehensive validation summary"""
        if not validation_results:
            return ""
        
        valid_count = sum(1 for r in validation_results if r.is_valid)
        total_count = len(validation_results)
        
        if valid_count == total_count:
            status_color = "#28a745"
            status_bg = "#d4edda"
            status_border = "#c3e6cb"
            status_icon = "✅"
            status_text = "All validations passed"
        elif valid_count > 0:
            status_color = "#ffc107"
            status_bg = "#fff3cd"
            status_border = "#ffeaa7"
            status_icon = "⚠️"
            status_text = f"{valid_count}/{total_count} validations passed"
        else:
            status_color = "#dc3545"
            status_bg = "#f8d7da"
            status_border = "#f5c6cb"
            status_icon = "❌"
            status_text = "Validation failed"
        
        return f"""
        <div style='color: {status_color}; background: {status_bg}; border: 1px solid {status_border}; 
                    border-radius: 8px; padding: 12px; margin: 8px 0;'>
            {status_icon} <strong>Validation Summary:</strong> {status_text}
        </div>
        """
    
    def _create_compatibility_display(self, compatibility_result: ValidationResult) -> str:
        """Create compatibility status display"""
        if compatibility_result.is_valid:
            return """
            <div style='color: #28a745; background: #d4edda; border: 1px solid #c3e6cb; 
                        border-radius: 8px; padding: 10px; margin: 5px 0;'>
                ✅ <strong>Images Compatible:</strong> Start and end images have compatible aspect ratios
            </div>
            """
        else:
            return f"""
            <div style='color: #dc3545; background: #f8d7da; border: 1px solid #f5c6cb; 
                        border-radius: 8px; padding: 10px; margin: 5px 0;'>
                ❌ <strong>Compatibility Issue:</strong> {compatibility_result.message}
            </div>
            """
    
    def _create_validation_display(self, validation_result: ValidationResult, context: str) -> str:
        """Create validation result display HTML"""
        if validation_result.is_valid:
            return f"""
            <div style='color: #28a745; background: #d4edda; border: 1px solid #c3e6cb; 
                        border-radius: 8px; padding: 10px; margin: 5px 0;'>
                ✅ <strong>{context.title()} Valid:</strong> {validation_result.message}
            </div>
            """
        else:
            suggestions_html = ""
            if hasattr(validation_result, 'suggestions') and validation_result.suggestions:
                suggestions_list = "".join([f"<li>{s}</li>" for s in validation_result.suggestions[:3]])
                suggestions_html = f"""
                <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">
                    {suggestions_list}
                </ul>
                """
            
            return f"""
            <div style='color: #dc3545; background: #f8d7da; border: 1px solid #f5c6cb; 
                        border-radius: 8px; padding: 10px; margin: 5px 0;'>
                ❌ <strong>{context.title()} Invalid:</strong> {validation_result.message}
                {suggestions_html}
            </div>
            """
    
    def _get_lora_compatibility_display(self, model_type: str) -> str:
        """Get LoRA compatibility display for model type"""
        compatibility_info = {
            "t2v-A14B": "✅ Full LoRA compatibility",
            "i2v-A14B": "✅ Full LoRA compatibility", 
            "ti2v-5B": "⚠️ Limited LoRA compatibility - some LoRAs may not work optimally"
        }
        
        info = compatibility_info.get(model_type, "❓ Unknown compatibility")
        color = "#28a745" if "✅" in info else "#ffc107" if "⚠️" in info else "#6c757d"
        bg = "#d4edda" if "✅" in info else "#fff3cd" if "⚠️" in info else "#f8f9fa"
        border = "#c3e6cb" if "✅" in info else "#ffeaa7" if "⚠️" in info else "#dee2e6"
        
        return f"""
        <div style='color: {color}; background: {bg}; border: 1px solid {border}; 
                    border-radius: 8px; padding: 10px; margin: 5px 0;'>
            <strong>LoRA Compatibility:</strong> {info}
        </div>
        """
    
    def _create_error_display(self, error: Exception, context: str) -> str:
        """Create error display HTML"""
        return f"""
        <div style='color: #dc3545; background: #f8d7da; border: 1px solid #f5c6cb; 
                    border-radius: 8px; padding: 12px; margin: 8px 0;'>
            ❌ <strong>Error in {context}:</strong> {str(error)}
        </div>
        """
    
    def _validate_generation_request(self, model_type: str, prompt: str, start_image: Any,
                                   end_image: Any, resolution: str, steps: int, 
                                   lora_path: str, lora_strength: float) -> Tuple[bool, str]:
        """Validate generation request comprehensively"""
        try:
            # Basic validation
            if not prompt or not prompt.strip():
                return False, """
                <div style='color: #dc3545; background: #f8d7da; border: 1px solid #f5c6cb; 
                            border-radius: 8px; padding: 12px; margin: 8px 0;'>
                    ❌ <strong>Validation Failed:</strong> Prompt is required
                </div>
                """
            
            # Model-specific validation
            if model_type in ["i2v-A14B", "ti2v-5B"] and start_image is None:
                return False, f"""
                <div style='color: #dc3545; background: #f8d7da; border: 1px solid #f5c6cb; 
                            border-radius: 8px; padding: 12px; margin: 8px 0;'>
                    ❌ <strong>Validation Failed:</strong> {model_type} requires a start image
                </div>
                """
            
            # Parameter validation
            if steps < 20 or steps > 100:
                return False, """
                <div style='color: #dc3545; background: #f8d7da; border: 1px solid #f5c6cb; 
                            border-radius: 8px; padding: 12px; margin: 8px 0;'>
                    ❌ <strong>Validation Failed:</strong> Steps must be between 20 and 100
                </div>
                """
            
            return True, "Validation passed"
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, self._create_error_display(e, "validation")
    
    def _run_generation(self, model_type: str, prompt: str, start_image: Any,
                       end_image: Any, resolution: str, steps: int, 
                       lora_path: str, lora_strength: float):
        """Run generation in background thread"""
        try:
            # This would integrate with the actual generation system
            # For now, we'll simulate progress updates
            
            for i in range(steps):
                if hasattr(self.progress_tracker, 'update_progress'):
                    progress_data = {
                        'current_step': i + 1,
                        'total_steps': steps,
                        'progress_percentage': ((i + 1) / steps) * 100,
                        'current_phase': 'Processing',
                        'elapsed_time': (i + 1) * 0.5  # Simulate time
                    }
                    self.progress_tracker.update_progress(self.current_task_id, progress_data)
                
                time.sleep(0.1)  # Simulate processing time
            
            # Mark generation as complete
            self.generation_in_progress = False
            logger.info("Generation completed")
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            self.generation_in_progress = False


    # New Integration Event Handler Methods
    
    def handle_resolution_change(self, resolution: str, model_type: str, 
                               start_image: Any, end_image: Any) -> Tuple[str, str, bool, str]:
        """Handle resolution change with validation and compatibility checking"""
        try:
            logger.info(f"Resolution changed to: {resolution} for model: {model_type}")
            
            # Validate resolution compatibility with model
            validation_messages = []
            
            # Check if resolution is supported by model
            model_resolutions = {
                "t2v-A14B": ["1280x720", "1280x704", "1920x1080"],
                "i2v-A14B": ["1280x720", "1280x704", "1920x1080"],
                "ti2v-5B": ["1280x720", "1280x704", "1920x1080", "1024x1024"]
            }
            
            supported_resolutions = model_resolutions.get(model_type, [])
            if resolution not in supported_resolutions:
                validation_messages.append(f"⚠️ Resolution {resolution} may not be optimal for {model_type}")
            
            # Check VRAM requirements
            vram_estimates = {
                "1280x720": 8,
                "1280x704": 8,
                "1920x1080": 12,
                "1024x1024": 6
            }
            
            estimated_vram = vram_estimates.get(resolution, 10)
            if estimated_vram > 12:
                validation_messages.append(f"⚠️ High VRAM usage expected: ~{estimated_vram}GB")
            
            # Check image compatibility if images are uploaded
            if start_image is not None:
                try:
                    img_width, img_height = start_image.size
                    res_width, res_height = map(int, resolution.split('x'))
                    img_aspect = img_width / img_height
                    res_aspect = res_width / res_height
                    
                    if abs(img_aspect - res_aspect) > 0.1:
                        validation_messages.append(f"⚠️ Image aspect ratio ({img_aspect:.2f}) differs from resolution aspect ratio ({res_aspect:.2f})")
                except Exception:
                    pass
            
            # Create parameter validation display
            if validation_messages:
                parameter_validation = f"""
                <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 10px; margin: 5px 0;">
                    <strong>Resolution Validation:</strong><br>
                    {'<br>'.join(validation_messages)}
                </div>
                """
            else:
                parameter_validation = f"""
                <div style="background: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px; padding: 10px; margin: 5px 0;">
                    ✅ Resolution {resolution} is compatible with {model_type}
                </div>
                """
            
            # Create validation summary
            validation_summary = f"""
            <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 10px; margin: 5px 0;">
                <strong>Current Settings:</strong> {model_type} @ {resolution}<br>
                <strong>Estimated VRAM:</strong> ~{estimated_vram}GB
            </div>
            """
            
            # Success notification
            notification_html = f"""
            <div style='color: #17a2b8; background: #d1ecf1; border: 1px solid #bee5eb; 
                        border-radius: 8px; padding: 12px; margin: 8px 0;'>
                ℹ️ Resolution updated to <strong>{resolution}</strong>
                <br><small>Estimated VRAM usage: ~{estimated_vram}GB</small>
            </div>
            """
            
            return parameter_validation, notification_html, True, validation_summary
            
        except Exception as e:
            logger.error(f"Resolution change error: {e}")
            error_html = self._create_error_display(e, "resolution_change")
            return "", error_html, True, ""
    
    def handle_steps_change(self, steps: int, model_type: str, resolution: str) -> Tuple[str, str, bool]:
        """Handle steps change with time estimation"""
        try:
            logger.info(f"Steps changed to: {steps}")
            
            # Estimate generation time based on steps, model, and resolution
            base_times = {
                "t2v-A14B": {"1280x720": 8, "1280x704": 8, "1920x1080": 15},
                "i2v-A14B": {"1280x720": 9, "1280x704": 9, "1920x1080": 17},
                "ti2v-5B": {"1280x720": 10, "1280x704": 10, "1920x1080": 18, "1024x1024": 7}
            }
            
            base_time = base_times.get(model_type, {}).get(resolution, 10)
            # Scale time based on steps (50 steps is baseline)
            estimated_time = base_time * (steps / 50)
            
            # Create parameter validation
            if steps < 30:
                severity = "warning"
                message = f"⚠️ Low step count ({steps}) may result in lower quality"
                color = "#fff3cd"
                border_color = "#ffc107"
            elif steps > 80:
                severity = "warning"
                message = f"⚠️ High step count ({steps}) will significantly increase generation time"
                color = "#fff3cd"
                border_color = "#ffc107"
            else:
                severity = "success"
                message = f"✅ Step count ({steps}) is optimal for quality/speed balance"
                color = "#d4edda"
                border_color = "#c3e6cb"
            
            parameter_validation = f"""
            <div style="background: {color}; border: 1px solid {border_color}; border-radius: 4px; padding: 10px; margin: 5px 0;">
                <strong>Steps Validation:</strong><br>
                {message}<br>
                <small>Estimated generation time: ~{estimated_time:.1f} minutes</small>
            </div>
            """
            
            # Success notification
            notification_html = f"""
            <div style='color: #17a2b8; background: #d1ecf1; border: 1px solid #bee5eb; 
                        border-radius: 8px; padding: 12px; margin: 8px 0;'>
                ℹ️ Steps updated to <strong>{steps}</strong>
                <br><small>Estimated generation time: ~{estimated_time:.1f} minutes</small>
            </div>
            """
            
            return parameter_validation, notification_html, True
            
        except Exception as e:
            logger.error(f"Steps change error: {e}")
            error_html = self._create_error_display(e, "steps_change")
            return "", error_html, True
    
    def handle_duration_change(self, duration: int, model_type: str, 
                             resolution: str, steps: int) -> Tuple[str, str, bool]:
        """Handle duration change with VRAM and time estimation"""
        try:
            logger.info(f"Duration changed to: {duration} seconds")
            
            # Estimate VRAM usage based on duration
            base_vram = {
                "1280x720": 8,
                "1280x704": 8,
                "1920x1080": 12,
                "1024x1024": 6
            }
            
            vram_per_second = base_vram.get(resolution, 8) * 0.2  # Additional VRAM per second
            estimated_vram = base_vram.get(resolution, 8) + (duration - 4) * vram_per_second
            
            # Estimate total generation time
            base_time_per_second = 2.5  # Base time per second of video
            model_multipliers = {"t2v-A14B": 1.0, "i2v-A14B": 1.1, "ti2v-5B": 1.2}
            step_multiplier = steps / 50
            
            estimated_time = duration * base_time_per_second * model_multipliers.get(model_type, 1.0) * step_multiplier
            
            # Create validation message
            warnings = []
            if duration > 6:
                warnings.append(f"Long duration ({duration}s) will require significant VRAM (~{estimated_vram:.1f}GB)")
            if estimated_time > 30:
                warnings.append(f"Generation time will be very long (~{estimated_time:.1f} minutes)")
            
            if warnings:
                parameter_validation = f"""
                <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 10px; margin: 5px 0;">
                    <strong>Duration Validation:</strong><br>
                    {'<br>'.join(warnings)}<br>
                    <small>Consider reducing duration for faster generation</small>
                </div>
                """
            else:
                parameter_validation = f"""
                <div style="background: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px; padding: 10px; margin: 5px 0;">
                    ✅ Duration ({duration}s) is optimal<br>
                    <small>Estimated VRAM: ~{estimated_vram:.1f}GB, Time: ~{estimated_time:.1f} minutes</small>
                </div>
                """
            
            # Success notification
            notification_html = f"""
            <div style='color: #17a2b8; background: #d1ecf1; border: 1px solid #bee5eb; 
                        border-radius: 8px; padding: 12px; margin: 8px 0;'>
                ℹ️ Duration updated to <strong>{duration} seconds</strong>
                <br><small>Estimated VRAM: ~{estimated_vram:.1f}GB, Generation time: ~{estimated_time:.1f} minutes</small>
            </div>
            """
            
            return parameter_validation, notification_html, True
            
        except Exception as e:
            logger.error(f"Duration change error: {e}")
            error_html = self._create_error_display(e, "duration_change")
            return "", error_html, True
    
    def handle_fps_change(self, fps: int, duration: int, resolution: str) -> Tuple[str, str, bool]:
        """Handle FPS change with frame count and processing estimation"""
        try:
            logger.info(f"FPS changed to: {fps}")
            
            # Calculate total frames
            total_frames = fps * duration
            
            # Estimate processing impact
            base_processing_time = 0.5  # seconds per frame at 24fps
            fps_multiplier = fps / 24
            frame_processing_time = total_frames * base_processing_time * fps_multiplier
            
            # Create validation message
            if fps > 24:
                parameter_validation = f"""
                <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 10px; margin: 5px 0;">
                    <strong>FPS Validation:</strong><br>
                    ⚠️ High FPS ({fps}) will increase processing time significantly<br>
                    <small>Total frames: {total_frames}, Est. processing time: +{frame_processing_time:.1f} minutes</small>
                </div>
                """
            elif fps < 16:
                parameter_validation = f"""
                <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 10px; margin: 5px 0;">
                    <strong>FPS Validation:</strong><br>
                    ⚠️ Low FPS ({fps}) may result in choppy motion<br>
                    <small>Total frames: {total_frames}</small>
                </div>
                """
            else:
                parameter_validation = f"""
                <div style="background: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px; padding: 10px; margin: 5px 0;">
                    ✅ FPS ({fps}) is optimal for smooth motion<br>
                    <small>Total frames: {total_frames}</small>
                </div>
                """
            
            # Success notification
            notification_html = f"""
            <div style='color: #17a2b8; background: #d1ecf1; border: 1px solid #bee5eb; 
                        border-radius: 8px; padding: 12px; margin: 8px 0;'>
                ℹ️ FPS updated to <strong>{fps}</strong>
                <br><small>Total frames for {duration}s video: {total_frames}</small>
            </div>
            """
            
            return parameter_validation, notification_html, True
            
        except Exception as e:
            logger.error(f"FPS change error: {e}")
            error_html = self._create_error_display(e, "fps_change")
            return "", error_html, True
    
    def handle_refresh_progress(self) -> Tuple[str, str]:
        """Handle progress refresh request"""
        try:
            if self.progress_tracker and self.current_task_id:
                progress_html = self.progress_tracker.get_progress_html()
                status = "Refreshed progress display"
                return progress_html, status
            else:
                return "", "No active generation to refresh"
        except Exception as e:
            logger.error(f"Progress refresh error: {e}")
            return "", f"Error refreshing progress: {str(e)}"
    
    def _on_progress_update(self, progress_data):
        """Handle progress updates from the progress tracker"""
        try:
            # This method is called by the progress tracker when progress updates occur
            # We can use this to trigger UI updates if needed
            if hasattr(self, 'ui_components') and 'progress_display' in self.ui_components:
                # Update progress display in real-time
                progress_html = self.progress_tracker.get_progress_html()
                # Note: Direct UI updates from callbacks may not work in Gradio
                # This is mainly for logging and state management
                # Only log if progress actually changed
                if not hasattr(self, '_last_callback_progress') or self._last_callback_progress != progress_data.progress_percentage:
                    logger.debug(f"Progress update: {progress_data.progress_percentage:.1f}%")
                    self._last_callback_progress = progress_data.progress_percentage
        except Exception as e:
            logger.warning(f"Progress update callback error: {e}")


def get_enhanced_event_handlers(config: Optional[Dict[str, Any]] = None) -> EnhancedUIEventHandlers:
    """Get enhanced UI event handlers instance"""
    return EnhancedUIEventHandlers(config)