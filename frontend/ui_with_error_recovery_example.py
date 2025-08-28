#!/usr/bin/env python3
"""
Example integration of Error Recovery System with WAN22 UI
Shows how to integrate the error recovery and fallback mechanisms into the existing UI
"""

import logging
import gradio as gr
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from ui_error_recovery_integration import create_ui_with_error_recovery

logger = logging.getLogger(__name__)

def create_wan22_ui_with_recovery(config: Optional[Dict[str, Any]] = None) -> gr.Blocks:
    """
    Create WAN22 UI with integrated error recovery and fallback mechanisms
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Gradio Blocks interface with error recovery
    """
    
    # Define the UI structure with error recovery support
    ui_definition = {
        'css': """
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .error-recovery-panel {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        
        .recovery-guidance {
            margin: 10px 0;
        }
        """,
        
        'title': 'WAN22 Video Generation UI - Enhanced with Error Recovery',
        'theme': gr.themes.Soft(),
        
        'components': {
            # Header components
            'header_markdown': {
                'type': 'Markdown',
                'kwargs': {
                    'value': """
                    # 🎬 WAN22 Video Generation UI
                    ## Enhanced with Error Recovery & Fallback Systems
                    
                    This interface includes comprehensive error recovery mechanisms to ensure stable operation.
                    """
                }
            },
            
            # Generation tab components
            'prompt_input': {
                'type': 'Textbox',
                'kwargs': {
                    'label': 'Generation Prompt',
                    'placeholder': 'Enter your video generation prompt...',
                    'lines': 3,
                    'max_lines': 5
                },
                'fallback_kwargs': {
                    'label': 'Prompt (Basic)',
                    'placeholder': 'Enter prompt...',
                    'lines': 2
                }
            },
            
            'model_dropdown': {
                'type': 'Dropdown',
                'kwargs': {
                    'label': 'Model Selection',
                    'choices': ['t2v-A14B', 'i2v-A14B', 'ti2v-A14B'],
                    'value': 't2v-A14B',
                    'info': 'Select the AI model for video generation'
                },
                'fallback_kwargs': {
                    'label': 'Model (Limited)',
                    'choices': ['Basic Model'],
                    'value': 'Basic Model'
                }
            },
            
            'steps_slider': {
                'type': 'Slider',
                'kwargs': {
                    'label': 'Generation Steps',
                    'minimum': 10,
                    'maximum': 100,
                    'value': 50,
                    'step': 5,
                    'info': 'Number of generation steps (higher = better quality, slower)'
                },
                'fallback_kwargs': {
                    'label': 'Steps (Basic)',
                    'minimum': 20,
                    'maximum': 60,
                    'value': 40,
                    'step': 10
                }
            },
            
            'generate_button': {
                'type': 'Button',
                'kwargs': {
                    'value': '🎬 Generate Video',
                    'variant': 'primary',
                    'size': 'lg'
                },
                'fallback_kwargs': {
                    'value': 'Generate',
                    'variant': 'secondary'
                }
            },
            
            # Output components
            'video_output': {
                'type': 'Video',
                'kwargs': {
                    'label': 'Generated Video',
                    'height': 400
                },
                'fallback_kwargs': {
                    'label': 'Video Output (Basic)',
                    'height': 200
                }
            },
            
            'status_display': {
                'type': 'HTML',
                'kwargs': {
                    'value': '<p>Ready for video generation...</p>',
                    'label': 'Status'
                }
            },
            
            # Image input components (for I2V/TI2V)
            'image_input': {
                'type': 'Image',
                'kwargs': {
                    'label': 'Input Image (Optional)',
                    'type': 'pil',
                    'height': 300
                },
                'fallback_kwargs': {
                    'label': 'Image (Basic)',
                    'height': 200
                }
            },
            
            # System monitoring components
            'system_status': {
                'type': 'HTML',
                'kwargs': {
                    'value': '<p>System status will be displayed here...</p>',
                    'label': 'System Status'
                }
            },
            
            'recovery_panel': {
                'type': 'HTML',
                'kwargs': {
                    'value': '',
                    'visible': False
                }
            }
        },
        
        'event_handlers': [
            {
                'component_name': 'generate_button',
                'event_type': 'click',
                'handler_function': generate_video_with_recovery,
                'inputs': ['prompt_input', 'model_dropdown', 'steps_slider', 'image_input'],
                'outputs': ['video_output', 'status_display']
            },
            {
                'component_name': 'model_dropdown',
                'event_type': 'change',
                'handler_function': update_model_info,
                'inputs': ['model_dropdown'],
                'outputs': ['status_display']
            }
        ]
    }
    
    # Create UI with error recovery
    interface, recovery_report = create_ui_with_error_recovery(ui_definition, config)
    
    # Log recovery report
    logger.info(f"UI created with recovery status: {recovery_report.get('status', 'unknown')}")
    
    if recovery_report.get('guidance_by_severity'):
        guidance = recovery_report['guidance_by_severity']
        total_issues = sum(len(items) for items in guidance.values())
        if total_issues > 0:
            logger.warning(f"UI created with {total_issues} recovery guidance items")
    
    return interface

def generate_video_with_recovery(prompt: str, model: str, steps: int, image=None):
    """
    Video generation function with error recovery
    
    Args:
        prompt: Generation prompt
        model: Selected model
        steps: Number of steps
        image: Optional input image
        
    Returns:
        Tuple of (video_output, status_message)
    """
    try:
        # Simulate video generation process
        if not prompt or len(prompt.strip()) < 5:
            return None, "❌ Please provide a valid prompt (at least 5 characters)"
        
        # Simulate processing
        status_msg = f"""
        <div style="background: #d1ecf1; padding: 15px; border-radius: 4px; margin: 10px 0;">
            <h4 style="color: #0c5460; margin-top: 0;">🎬 Video Generation Started</h4>
            <p style="color: #0c5460; margin: 5px 0;"><strong>Prompt:</strong> {prompt[:100]}...</p>
            <p style="color: #0c5460; margin: 5px 0;"><strong>Model:</strong> {model}</p>
            <p style="color: #0c5460; margin: 5px 0;"><strong>Steps:</strong> {steps}</p>
            <p style="color: #0c5460; margin: 5px 0;"><strong>Status:</strong> Processing... (This is a demo)</p>
        </div>
        """
        
        # In a real implementation, this would call the actual generation service
        # For demo purposes, we return a status message
        return None, status_msg
        
    except Exception as e:
        logger.error(f"Error in video generation: {e}")
        
        # Use recovery guidance system for error handling
        from recovery_guidance_system import generate_error_guidance
        
        title, message, suggestions, severity = generate_error_guidance(
            error_message=str(e),
            component_name="video_generation",
            error_type="generation_error"
        )
        
        error_html = f"""
        <div style="background: #f8d7da; border: 2px solid #f5c6cb; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <h4 style="color: #721c24; margin-top: 0;">⚠️ {title}</h4>
            <p style="color: #721c24; margin: 10px 0;">{message}</p>
            <div style="margin-top: 15px;">
                <strong style="color: #721c24;">💡 Suggestions:</strong>
                <ul style="margin: 8px 0; padding-left: 20px; color: #721c24;">
                    {''.join([f'<li>{suggestion}</li>' for suggestion in suggestions[:3]])}
                </ul>
            </div>
        </div>
        """
        
        return None, error_html

def update_model_info(model: str):
    """
    Update model information display
    
    Args:
        model: Selected model name
        
    Returns:
        Status message HTML
    """
    try:
        model_info = {
            't2v-A14B': 'Text-to-Video model - Generates videos from text prompts',
            'i2v-A14B': 'Image-to-Video model - Generates videos from input images',
            'ti2v-A14B': 'Text+Image-to-Video model - Uses both text and image inputs'
        }
        
        info = model_info.get(model, 'Unknown model selected')
        
        return f"""
        <div style="background: #e8f5e8; padding: 10px; border-radius: 4px; margin: 5px 0;">
            <p style="color: #155724; margin: 0;"><strong>Model:</strong> {model}</p>
            <p style="color: #155724; margin: 5px 0 0 0;"><strong>Description:</strong> {info}</p>
        </div>
        """
        
    except Exception as e:
        logger.error(f"Error updating model info: {e}")
        return f"<p style='color: #dc3545;'>Error loading model information: {str(e)}</p>"

def main():
    """Main function to launch the UI with error recovery"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Starting WAN22 UI with Error Recovery System")
    
    try:
        # Configuration for error recovery
        config = {
            'enable_system_monitoring': True,
            'max_recovery_attempts': 3,
            'show_recovery_guidance': True
        }
        
        # Create UI with error recovery
        interface = create_wan22_ui_with_recovery(config)
        
        # Launch the interface
        print("✅ UI created successfully with error recovery")
        print("🌐 Launching interface...")
        
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start UI: {e}")
        print(f"❌ Failed to start UI: {e}")
        
        # Try to create emergency interface
        try:
            print("🚨 Creating emergency fallback interface...")
            
            with gr.Blocks(title="WAN22 UI - Emergency Mode") as emergency_interface:
                gr.Markdown("# 🚨 WAN22 UI - Emergency Mode")
                gr.HTML(f"""
                <div style="background: #f8d7da; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <h3 style="color: #721c24;">Critical Error</h3>
                    <p style="color: #721c24;">The main UI could not start: {str(e)}</p>
                    <p style="color: #721c24;">Please restart the application or contact support.</p>
                </div>
                """)
            
            emergency_interface.launch(
                server_name="127.0.0.1",
                server_port=7861,
                share=False
            )
            
        except Exception as emergency_error:
            print(f"❌ Emergency interface also failed: {emergency_error}")

if __name__ == "__main__":
    main()