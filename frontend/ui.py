"""
Wan2.2 UI Variant - Gradio Web Interface
Main UI implementation with four tabs: Generation, Optimizations, Queue & Stats, and Outputs
"""

import gradio as gr
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import threading
import time
from datetime import datetime

# Configure logging first
logger = logging.getLogger(__name__)

# Apply emergency model loading fixes
try:
    import apply_model_fixes
    logger.info("Emergency model loading fixes applied")
except Exception as e:
    logger.warning(f"Could not apply model loading fixes: {e}")

# Import backend utilities
from core.services.utils import (
    GenerationTask,
    TaskStatus,
    get_system_stats,
    get_queue_manager,
    enhance_prompt,
    generate_video,
    get_output_manager,
    # New compatibility integration functions
    get_compatibility_status_for_ui,
    get_optimization_status_for_ui,
    apply_optimization_recommendations,
    check_model_compatibility_for_ui,
    get_model_loading_progress_info
)

# Import UI compatibility integration
from ui_compatibility_integration import (
    create_compatibility_ui_components,
    update_compatibility_ui,
    update_optimization_ui,
    apply_optimizations_ui
)

# Import performance profiler
from infrastructure.hardware.performance_profiler import (
    get_performance_profiler,
    start_performance_monitoring,
    stop_performance_monitoring,
    profile_operation
)

# Import error handling system
from infrastructure.hardware.error_handler import (
    handle_error_with_recovery,
    log_error_with_context,
    ErrorWithRecoveryInfo,
    get_error_recovery_manager,
    ErrorCategory
)

# Import progress tracking system
from progress_tracker import (
    get_progress_tracker,
    create_progress_callback,
    GenerationPhase
)

# Import session management system
from ui_session_integration import (
    get_ui_session_integration,
    cleanup_ui_session_integration,
    create_session_management_ui,
    setup_session_management_handlers
)

class Wan22UI:
    """Main UI class for Wan2.2 video generation interface"""
    
    def __init__(self, config_path: str = "config.json", system_optimizer=None):
        self.config = self._load_config(config_path)
        from core.services.model_manager import get_model_manager
        from core.services.optimization_service import get_vram_optimizer
        self.model_manager = get_model_manager()
        self.vram_optimizer = get_vram_optimizer(self.config)
        
        # Store system optimizer reference
        self.system_optimizer = system_optimizer
        
        # Initialize performance profiler
        self.performance_profiler = get_performance_profiler()
        
        # Initialize progress tracker
        self.progress_tracker = get_progress_tracker(self.config)
        
        # Add progress update callback
        self.progress_tracker.add_update_callback(self._on_progress_update)
        
        # Initialize session management
        self.ui_session_integration = get_ui_session_integration(self.config)
        logger.info("Session management initialized")
        
        # UI state variables
        self.current_model_type = "t2v-A14B"
        self.current_optimization_settings = {
            "quantization": "bf16",
            "enable_offload": True,
            "vae_tile_size": 256
        }
        self.auto_refresh_enabled = False  # Disabled during initialization to prevent loops
        self.selected_video_path = None
        
        # Initialize LoRA UI state management
        try:
            from lora_ui_state import LoRAUIState
            self.lora_ui_state = LoRAUIState(self.config)
            logger.info("LoRA UI state management initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize LoRA UI state: {e}")
            self.lora_ui_state = None
        
        # Initialize enhanced image preview manager
        try:
            from enhanced_image_preview_manager import get_preview_manager
            self.image_preview_manager = get_preview_manager(self.config)
            logger.info("Enhanced image preview manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize image preview manager: {e}")
            self.image_preview_manager = None
        
        # Real-time update system
        self.update_thread = None
        self.stop_updates = threading.Event()
        self.last_queue_update = datetime.now()
        self.last_stats_update = datetime.now()
        
        # Create main interface
        self.interface = self._create_interface()
        
        # Start auto-refresh by default
        if self.auto_refresh_enabled:
            self._start_auto_refresh()
        
        # Start performance monitoring
        start_performance_monitoring()
        
        # Check system requirements on startup
        self._perform_startup_checks()
        
        # Initialize system optimizer status if available
        if self.system_optimizer:
            self._initialize_optimizer_status()
    
    def _setup_enhanced_generation_events(self):
        """Set up enhanced event handlers with validation and feedback"""
        try:
            # Set up session management integration
            self._setup_session_management()
            
            if self.event_handlers:
                # Register components with event handlers
                self.event_handlers.register_components(self.generation_components)
                
                # Set up all enhanced event handlers
                self.event_handlers.setup_all_event_handlers()
                
                logger.info("Enhanced generation events set up successfully")
            else:
                # Fallback to basic event setup
                self._setup_generation_events()
                logger.warning("Using basic event handlers - enhanced features not available")
                
        except Exception as e:
            logger.error(f"Failed to set up enhanced generation events: {e}")
            # Fallback to basic event setup
            try:
                self._setup_generation_events()
                logger.info("Fallback to basic event handlers successful")
            except Exception as fallback_error:
                logger.error(f"Fallback event setup also failed: {fallback_error}")
                # At this point, we have a serious issue, but don't crash the UI
                pass
    
    def _setup_session_management(self):
        """Set up session management integration with UI components"""
        try:
            # Set up UI components with session integration
            self.ui_session_integration.setup_ui_components(
                start_image_input=self.generation_components['image_input'],
                end_image_input=self.generation_components['end_image_input'],
                model_type_dropdown=self.generation_components['model_type']
            )
            
            # Set up session management UI event handlers
            setup_session_management_handlers(
                ui_integration=self.ui_session_integration,
                session_info_display=self.generation_components['session_info_display'],
                clear_session_button=self.generation_components['clear_session_button'],
                refresh_info_button=self.generation_components['refresh_info_button'],
                start_image_input=self.generation_components['image_input'],
                end_image_input=self.generation_components['end_image_input']
            )
            
            logger.info("Session management integration set up successfully")
            
        except Exception as e:
            logger.error(f"Failed to set up session management: {e}")
    
    def __del__(self):
        """Cleanup when UI is destroyed"""
        try:
            stop_performance_monitoring()
            # Cleanup session management
            if hasattr(self, 'ui_session_integration'):
                cleanup_ui_session_integration()
        except Exception:
            pass
    
    def _create_error_display(self, error_info: 'ErrorInfo') -> str:
        """Create user-friendly error display HTML"""
        category_colors = {
            ErrorCategory.VRAM_ERROR: "#dc3545",  # Red
            ErrorCategory.MODEL_LOADING_ERROR: "#fd7e14",  # Orange
            ErrorCategory.GENERATION_ERROR: "#ffc107",  # Yellow
            ErrorCategory.NETWORK_ERROR: "#17a2b8",  # Cyan
            ErrorCategory.FILE_IO_ERROR: "#6f42c1",  # Purple
            ErrorCategory.VALIDATION_ERROR: "#e83e8c",  # Pink
            ErrorCategory.SYSTEM_ERROR: "#dc3545",  # Red
            ErrorCategory.UI_ERROR: "#28a745",  # Green
            ErrorCategory.UNKNOWN_ERROR: "#6c757d"  # Gray
        }
        
        color = category_colors.get(error_info.category, "#6c757d")
        
        suggestions_html = ""
        if error_info.recovery_suggestions:
            suggestions_list = "".join([f"<li>{suggestion}</li>" for suggestion in error_info.recovery_suggestions[:3]])
            suggestions_html = f"""
            <div style="margin-top: 10px;">
                <strong>💡 Try these solutions:</strong>
                <ul style="margin: 5px 0; padding-left: 20px;">
                    {suggestions_list}
                </ul>
            </div>
            """
        
        return f"""
        <div style="
            border: 2px solid {color}; 
            border-radius: 8px; 
            padding: 15px; 
            margin: 10px 0; 
            background: linear-gradient(135deg, {color}15, {color}05);
            animation: slideIn 0.3s ease-out;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="
                    background: {color}; 
                    color: white; 
                    padding: 4px 8px; 
                    border-radius: 12px; 
                    font-size: 0.8em; 
                    font-weight: bold; 
                    margin-right: 10px;
                ">
                    {error_info.category.value.replace('_', ' ').title()}
                </span>
                <span style="color: {color}; font-weight: bold;">
                    ⚠️ {error_info.user_message}
                </span>
            </div>
            {suggestions_html}
            <div style="margin-top: 10px; font-size: 0.8em; color: #666;">
                <strong>Time:</strong> {error_info.timestamp.strftime('%H:%M:%S')}
                {f" | <strong>Retries:</strong> {error_info.retry_count}/{error_info.max_retries}" if error_info.retry_count > 0 else ""}
            </div>
        </div>
        """
    
    def _handle_ui_error(self, error: Exception, context: str = "") -> Tuple[str, bool]:
        """Handle UI errors and return user-friendly message and visibility"""
        try:
            if isinstance(error, ErrorWithRecoveryInfo):
                error_html = self._create_error_display(error.error_info)
                return error_html, True
            else:
                # Create error info for regular exceptions
                from infrastructure.hardware.error_handler import create_error_info
                error_info = create_error_info(error, context)
                error_html = self._create_error_display(error_info)
                return error_html, True
        except Exception as e:
            # Fallback error display
            return f"""
            <div style="border: 2px solid #dc3545; border-radius: 8px; padding: 15px; margin: 10px 0; background: #dc354515;">
                <span style="color: #dc3545; font-weight: bold;">⚠️ An error occurred: {str(error)}</span>
            </div>
            """, True
    
    @handle_error_with_recovery
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            log_error_with_context(e, "ui_config_loading", {"config_path": config_path})
            print(f"Warning: Failed to load config: {e}")
            # Return default config
            return {
                "directories": {
                    "models_directory": "models",
                    "loras_directory": "loras", 
                    "outputs_directory": "outputs"
                },
                "optimization": {
                    "default_quantization": "bf16",
                    "enable_offload": True,
                    "vae_tile_size": 256,
                    ""max_vram_usage_gb": 14
                },
                # RTX 4080 Optimizations
                "rtx4080_optimizations": {
                    "enable_bf16": True,
                    "disable_cpu_offload": True,
                    "vae_tile_size": 512,
                    "max_concurrent_generations": 1,
                    "enable_attention_slicing": True
                },,
                "generation": {
                    "default_resolution": "1280x720",
                    "default_steps": 50,
                    "max_prompt_length": 500
                }
            }
    
    def _create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface with four tabs"""
        
        # Import responsive image interface
        try:
            from responsive_image_interface import get_responsive_image_interface
            responsive_interface = get_responsive_image_interface(self.config)
            responsive_css = responsive_interface.get_responsive_css()
        except ImportError:
            logger.warning("Responsive image interface not available")
            responsive_css = ""
        
        # Custom CSS for responsive design and animations
        css = responsive_css + """
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .tab-content {
            padding: 20px;
            min-height: 600px;
        }
        
        .stats-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .stat-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        .queue-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .queue-table th, .queue-table td {
            border: 1px solid #dee2e6;
            padding: 8px;
            text-align: left;
        }
        
        .video-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .video-card {
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            background: #f8f9fa;
        }
        
        .notification-area {
            margin: 10px 0;
        }
        
        .lora-help, .steps-help, .image-help, .image-requirements-help {
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            font-size: 0.9em;
            line-height: 1.4;
        }
        
        .image-help {
            background: #e8f5e8;
            border-left-color: #28a745;
        }
        
        .image-requirements-help {
            background: #fff3cd;
            border-left-color: #ffc107;
            font-size: 0.8em;
        }
        
        /* Enhanced help content styles */
        .help-content {
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            font-size: 0.9em;
            line-height: 1.4;
        }
        
        .help-title {
            color: #007bff;
            margin: 0 0 10px 0;
            font-size: 1.1em;
            font-weight: bold;
        }
        
        .help-text {
            color: #495057;
        }
        
        .help-content strong {
            color: #212529;
            font-weight: 600;
        }
        
        .help-content em {
            color: #6c757d;
            font-style: italic;
        }
        
        .error-help {
            background: #f8d7da;
            border-left-color: #dc3545;
        }
        
        .warning-help {
            background: #fff3cd;
            border-left-color: #ffc107;
        }
        
        /* Help tooltip styles */
        .help-tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        
        .help-tooltip .tooltip-text {
            visibility: hidden;
            width: 250px;
            background-color: #333;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            margin-left: -125px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.8em;
            line-height: 1.3;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        
        .help-tooltip .tooltip-text::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #333 transparent transparent transparent;
        }
        
        .help-tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
        
        .validation-success {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 8px 12px;
            border-radius: 4px;
            margin: 5px 0;
            font-size: 0.9em;
        }
        
        .validation-error {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 8px 12px;
            border-radius: 4px;
            margin: 5px 0;
            font-size: 0.9em;
        }
        
        .lora-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }
        
        .lora-card {
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            background: #f8f9fa;
            transition: all 0.3s ease;
        }
        
        .lora-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        .lora-card.selected {
            border-color: #007bff;
            background: #e3f2fd;
        }
        
        .lora-upload-area {
            border: 2px dashed #007bff;
            border-radius: 8px;
            padding: 30px;
            text-align: center;
            background: #f8f9fa;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .lora-upload-area:hover {
            background: #e3f2fd;
            border-color: #0056b3;
        }
        
        .lora-upload-area.dragover {
            background: #cce5ff;
            border-color: #004085;
        }
        
        .lora-selection-summary {
            background: #e8f5e8;
            border: 1px solid #28a745;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        
        .lora-memory-warning {
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
            color: #856404;
        }
        
        .lora-error {
            background: #f8d7da;
            border: 1px solid #dc3545;
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
            color: #721c24;
        }
        
        /* Clear image button styles */
        .clear-image-btn {
            transition: all 0.2s ease;
            margin: 5px 0;
        }
        
        .clear-image-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .clear-image-btn:active {
            transform: translateY(0);
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        .progress-indicator {
            background: linear-gradient(90deg, #007bff, #0056b3);
            height: 4px;
            border-radius: 2px;
            transition: width 0.3s ease;
            margin: 5px 0;
        }
        
        .status-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .status-processing {
            background: #fff3cd;
            color: #856404;
        }
        
        .status-completed {
            background: #d4edda;
            color: #155724;
        }
        
        .status-failed {
            background: #f8d7da;
            color: #721c24;
        }
        
        .status-pending {
            background: #d1ecf1;
            color: #0c5460;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .processing-indicator {
            animation: pulse 2s infinite;
        }
        
        @keyframes slideIn {
            from { 
                opacity: 0; 
                transform: translateY(-10px); 
            }
            to { 
                opacity: 1; 
                transform: translateY(0); 
            }
        }
        
        .notification-slide {
            animation: slideIn 0.3s ease-out;
        }
        
        /* Enhanced Image Preview Styles */
        .image-preview-display {
            margin: 10px 0;
        }
        
        .image-preview-container {
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .image-preview-container:hover {
            transform: scale(1.02);
            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        }
        
        .image-preview-empty {
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .image-preview-empty:hover {
            background: #e9ecef !important;
            border-color: #007bff !important;
        }
        
        .image-preview-error {
            animation: shake 0.5s ease-in-out;
        }
        
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-5px); }
            75% { transform: translateX(5px); }
        }
        
        .image-summary-display {
            margin: 10px 0;
        }
        
        .compatibility-status-display {
            margin: 10px 0;
        }
        
        /* Thumbnail hover effects */
        .image-preview-container img {
            transition: all 0.3s ease;
        }
        
        .image-preview-container img:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }
        
        /* Clear button hover effects */
        .image-preview-container button {
            transition: all 0.2s ease;
        }
        
        .image-preview-container button:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        @media (max-width: 768px) {
            .main-container {
                padding: 10px;
            }
            
            .stats-container {
                grid-template-columns: 1fr;
            }
            
            .video-gallery {
                grid-template-columns: 1fr;
            }
            
            /* Mobile responsive image previews */
            .image-preview-container {
                padding: 10px;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 10px;
            }
            
            .image-preview-container > div:first-child {
                flex-direction: column;
                gap: 10px;
                width: 100%;
                text-align: center;
            }
            
            .image-preview-container img {
                max-width: 120px;
                max-height: 120px;
                margin: 0 auto;
            }
            
            /* Mobile responsive help text */
            .lora-help, .steps-help, .image-help, .image-requirements-help, .help-content {
                padding: 8px;
                font-size: 0.8em;
                margin: 5px 0;
                line-height: 1.4;
            }
            
            .image-requirements-help {
                font-size: 0.75em;
                padding: 6px;
            }
            
            .help-title {
                font-size: 0.9em;
                margin-bottom: 6px;
            }
            
            .help-text br {
                display: none;
            }
            
            .help-text {
                line-height: 1.3;
            }
            
            /* Mobile tooltip adjustments */
            .help-tooltip .tooltip-text {
                width: 200px;
                margin-left: -100px;
                font-size: 0.7em;
                padding: 8px;
            }
            
            /* Stack image upload columns on mobile */
            .image-inputs-row {
                flex-direction: column !important;
                gap: 20px !important;
            }
            
            .image-inputs-row > div {
                width: 100% !important;
                margin-bottom: 15px;
            }
            
            /* Mobile image upload areas */
            .image-upload-area {
                min-height: 150px;
                padding: 15px;
            }
            
            /* Mobile validation messages */
            .validation-success, .validation-error {
                font-size: 0.8em;
                padding: 8px;
                text-align: center;
            }
            
            /* Mobile button adjustments */
            .responsive-button {
                padding: 8px 12px;
                font-size: 0.8em;
                margin: 2px;
            }
        }
        
        /* Tablet responsive adjustments */
        @media (min-width: 769px) and (max-width: 1024px) {
            .image-inputs-row {
                gap: 20px;
            }
            
            .image-preview-container img {
                max-width: 130px;
                max-height: 130px;
            }
            
            .image-requirements-help {
                font-size: 0.8em;
                padding: 10px;
            }
        }
        
        /* Desktop optimizations */
        @media (min-width: 1025px) {
            .image-inputs-row {
                gap: 30px;
            }
            
            .image-preview-container img {
                max-width: 150px;
                max-height: 150px;
            }
            
            .image-requirements-help {
                font-size: 0.85em;
                padding: 12px;
            }
        }
        
        /* Extra small mobile devices */
        @media (max-width: 480px) {
            .main-container {
                padding: 5px;
            }
            
            .image-upload-area {
                min-height: 120px;
                padding: 10px;
            }
            
            .image-preview-container img {
                max-width: 100px;
                max-height: 100px;
            }
            
            .image-requirements-help {
                font-size: 0.7em;
                padding: 5px;
            }
            
            .help-content {
                font-size: 0.75em;
                padding: 6px;
            }
            
            .responsive-button {
                padding: 6px 10px;
                font-size: 0.75em;
            }
        }
        """
        
        # Import safe UI creation validator
        from ui_creation_validator import create_safe_gradio_blocks
        
        with create_safe_gradio_blocks(css=css, title="Wan2.2 Video Generation UI", theme=gr.themes.Soft()) as interface:
            
            # Header
            gr.Markdown(
                """
                # 🎬 Wan2.2 Video Generation UI
                
                Advanced AI video generation with Text-to-Video (T2V), Image-to-Video (I2V), and Text-Image-to-Video (TI2V) capabilities.
                Optimized for NVIDIA RTX 4080 with comprehensive VRAM management.
                """,
                elem_classes=["main-container"]
            )
            
            # Main tabbed interface
            with gr.Tabs() as main_tabs:
                
                # Tab 1: Generation
                with gr.Tab("🎥 Generation", id="generation_tab"):
                    self._create_generation_tab()
                
                # Tab 2: LoRAs
                with gr.Tab("🎨 LoRAs", id="lora_tab"):
                    self._create_lora_tab()
                
                # Tab 3: Optimizations  
                with gr.Tab("⚙️ Optimizations", id="optimization_tab"):
                    self._create_optimization_tab()
                
                # Tab 4: Queue & Stats
                with gr.Tab("📊 Queue & Stats", id="queue_stats_tab"):
                    self._create_queue_stats_tab()
                
                # Tab 5: Outputs
                with gr.Tab("📁 Outputs", id="outputs_tab"):
                    self._create_outputs_tab()
            
            # Footer
            gr.Markdown(
                """
                ---
                **Wan2.2 UI Variant** - Powered by Gradio | Optimized for RTX 4080
                """,
                elem_classes=["main-container"]
            )
            
            # Enhanced Image Preview JavaScript
            gr.HTML("""
            <script>
            // Clear image function
            function clearImage(imageType) {
                // Trigger the appropriate clear button
                const clearBtn = document.getElementById('clear_' + imageType + '_image_btn');
                if (clearBtn) {
                    clearBtn.click();
                }
            }

            // Show large preview function
            function showLargePreview(imageType, thumbnailData) {
                // Create modal overlay
                const modal = document.createElement('div');
                modal.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0,0,0,0.8);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    z-index: 10000;
                    cursor: pointer;
                `;
                
                // Create large image
                const img = document.createElement('img');
                img.src = thumbnailData;
                img.style.cssText = `
                    max-width: 90%;
                    max-height: 90%;
                    border-radius: 8px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
                `;
                
                modal.appendChild(img);
                document.body.appendChild(modal);
                
                // Close on click
                modal.addEventListener('click', () => {
                    document.body.removeChild(modal);
                });
                
                // Close on escape key
                const escapeHandler = (e) => {
                    if (e.key === 'Escape') {
                        document.body.removeChild(modal);
                        document.removeEventListener('keydown', escapeHandler);
                    }
                };
                document.addEventListener('keydown', escapeHandler);
            }

            // Enhanced tooltip functionality
            let tooltipElement = null;

            function showTooltip(event, data) {
                hideTooltip(); // Remove any existing tooltip
                
                tooltipElement = document.createElement('div');
                tooltipElement.style.cssText = `
                    position: absolute;
                    background: #333;
                    color: white;
                    padding: 10px;
                    border-radius: 6px;
                    font-size: 0.8em;
                    z-index: 1000;
                    max-width: 250px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                    pointer-events: none;
                `;
                
                const tooltipContent = `
                    <div><strong>File:</strong> ${data.filename}</div>
                    <div><strong>Size:</strong> ${data.dimensions}</div>
                    <div><strong>Format:</strong> ${data.format}</div>
                    <div><strong>File Size:</strong> ${data.file_size}</div>
                    <div><strong>Aspect:</strong> ${data.aspect_ratio}</div>
                    <div><strong>Uploaded:</strong> ${data.upload_time}</div>
                    <div><strong>Status:</strong> ${data.validation_status}</div>
                `;
                
                tooltipElement.innerHTML = tooltipContent;
                document.body.appendChild(tooltipElement);
                
                // Position tooltip
                const rect = event.target.getBoundingClientRect();
                tooltipElement.style.left = (rect.right + 10) + 'px';
                tooltipElement.style.top = rect.top + 'px';
                
                // Adjust if tooltip goes off screen
                const tooltipRect = tooltipElement.getBoundingClientRect();
                if (tooltipRect.right > window.innerWidth) {
                    tooltipElement.style.left = (rect.left - tooltipRect.width - 10) + 'px';
                }
                if (tooltipRect.bottom > window.innerHeight) {
                    tooltipElement.style.top = (rect.bottom - tooltipRect.height) + 'px';
                }
            }

            function hideTooltip() {
                if (tooltipElement) {
                    document.body.removeChild(tooltipElement);
                    tooltipElement = null;
                }
            }

            // Initialize tooltip functionality when DOM is ready
            document.addEventListener('DOMContentLoaded', function() {
                // Add hover tooltips to image previews
                const observer = new MutationObserver(function(mutations) {
                    mutations.forEach(function(mutation) {
                        mutation.addedNodes.forEach(function(node) {
                            if (node.nodeType === 1) { // Element node
                                const previewContainers = node.querySelectorAll ? 
                                    node.querySelectorAll('.image-preview-container') : [];
                                
                                previewContainers.forEach(container => {
                                    const tooltipData = container.querySelector('.tooltip-data');
                                    if (tooltipData && tooltipData.dataset.tooltip) {
                                        try {
                                            const data = JSON.parse(tooltipData.dataset.tooltip);
                                            
                                            container.addEventListener('mouseenter', function(e) {
                                                showTooltip(e, data);
                                            });
                                            
                                            container.addEventListener('mouseleave', function() {
                                                hideTooltip();
                                            });
                                        } catch (e) {
                                            console.warn('Failed to parse tooltip data:', e);
                                        }
                                    }
                                });
                            }
                        });
                    });
                });
                
                // Start observing
                observer.observe(document.body, {
                    childList: true,
                    subtree: true
                });
            });
            </script>
            """)
            
            # Add responsive JavaScript functionality
            try:
                responsive_js = responsive_interface.get_responsive_javascript()
                gr.HTML(responsive_js)
            except (NameError, AttributeError):
                # Fallback responsive JavaScript
                gr.HTML("""
                <script>
                // Basic responsive functionality fallback
                function updateResponsiveLayout() {
                    const isMobile = window.innerWidth <= 768;
                    const imageRows = document.querySelectorAll('.image-inputs-row');
                    
                    imageRows.forEach(row => {
                        if (isMobile) {
                            row.style.flexDirection = 'column';
                            row.style.gap = '20px';
                        } else {
                            row.style.flexDirection = 'row';
                            row.style.gap = '30px';
                        }
                    });
                    
                    // Update help text visibility
                    const desktopHelp = document.querySelectorAll('.desktop-requirements, .desktop-help');
                    const mobileHelp = document.querySelectorAll('.mobile-requirements, .mobile-help');
                    
                    if (isMobile) {
                        desktopHelp.forEach(el => el.style.display = 'none');
                        mobileHelp.forEach(el => el.style.display = 'block');
                    } else {
                        desktopHelp.forEach(el => el.style.display = 'block');
                        mobileHelp.forEach(el => el.style.display = 'none');
                    }
                }
                
                // Initialize and listen for resize events
                document.addEventListener('DOMContentLoaded', updateResponsiveLayout);
                window.addEventListener('resize', updateResponsiveLayout);
                </script>
                """)
        
        return interface
    
    def _create_generation_tab(self):
        """Create the Generation tab interface with enhanced validation and feedback"""
        with gr.Column(elem_classes=["tab-content"]):
            
            gr.Markdown("### 🎬 Video Generation")
            
            # Import enhanced UI components
            try:
                from ui_event_handlers import get_event_handlers
                self.event_handlers = get_event_handlers(self.config)
                logger.info("Enhanced UI event handlers initialized")
            except ImportError as e:
                logger.warning(f"Enhanced UI handlers not available: {e}")
                self.event_handlers = None
            except Exception as e:
                logger.error(f"Failed to initialize enhanced UI handlers: {e}")
                self.event_handlers = None
            
            with gr.Row():
                with gr.Column(scale=2):
                    
                    # Model type selection with enhanced feedback
                    model_type = gr.Dropdown(
                        choices=["t2v-A14B", "i2v-A14B", "ti2v-5B"],
                        value="t2v-A14B",
                        label="Model Type - Select the generation mode: T2V (text-only), I2V (image-only), TI2V (text+image)"
                    )
                    
                    # Context-sensitive help text
                    model_help_text = gr.Markdown(
                        value=self._get_model_help_text("t2v-A14B"),
                        visible=True
                    )
                    
                    # Prompt input with real-time validation
                    prompt_input = gr.Textbox(
                        label="Prompt - Describe the video you want to generate (max 500 characters)",
                        placeholder="Enter your video generation prompt...",
                        lines=3,
                        max_lines=5
                    )
                    
                    # Character counter with validation status
                    char_count = gr.Textbox(
                        value="0/500",
                        label="Character Count",
                        interactive=False,
                        max_lines=1
                    )
                    
                    # Real-time prompt validation feedback
                    prompt_validation_display = gr.HTML(
                        value="",
                        visible=False,
                        elem_classes=["validation-feedback"]
                    )
                    
                    # Responsive image uploads (hidden by default) with comprehensive tooltips
                    with gr.Row(visible=False, elem_classes=["image-inputs-row", "responsive-grid"]) as image_inputs_row:
                        with gr.Column(elem_classes=["image-column"]):
                            # Start image upload with enhanced tooltip
                            start_image_tooltip = self._get_tooltip_text("start_image")
                            image_input = gr.Image(
                                label="📸 Start Frame Image (Required for I2V/TI2V generation)",
                                type="pil",
                                interactive=True,
                                elem_id="start_image_upload",
                                elem_classes=["image-upload-area"]
                            )
                            
                            # Enhanced start image preview
                            start_image_preview = gr.HTML(
                                value="",
                                visible=False,
                                elem_classes=["image-preview-display", "image-preview-fade-in"]
                            )
                            
                            # Start image requirements display
                            start_image_requirements = gr.Markdown(
                                value=self._get_responsive_image_requirements_text("start_image"),
                                visible=False,
                                elem_classes=["image-requirements-help", "responsive-help"]
                            )
                        
                        with gr.Column(elem_classes=["image-column"]):
                            # End image upload with enhanced tooltip
                            end_image_tooltip = self._get_tooltip_text("end_image")
                            end_image_input = gr.Image(
                                label="🎯 End Frame Image (Optional - defines video ending)",
                                type="pil",
                                interactive=True,
                                elem_id="end_image_upload",
                                elem_classes=["image-upload-area"]
                            )
                            
                            # Enhanced end image preview
                            end_image_preview = gr.HTML(
                                value="",
                                visible=False,
                                elem_classes=["image-preview-display", "image-preview-fade-in"]
                            )
                            
                            # End image requirements display
                            end_image_requirements = gr.Markdown(
                                value=self._get_responsive_image_requirements_text("end_image"),
                                visible=False,
                                elem_classes=["image-requirements-help", "responsive-help"]
                            )
                    
                    # Image summary and compatibility status
                    with gr.Row(visible=False) as image_status_row:
                        with gr.Column():
                            image_summary = gr.HTML(
                                value="",
                                visible=False,
                                elem_classes=["image-summary-display"]
                            )
                        
                        with gr.Column():
                            compatibility_status = gr.HTML(
                                value="",
                                visible=False,
                                elem_classes=["compatibility-status-display"]
                            )
                    
                    # Clear buttons for image management (visible when images are uploaded)
                    clear_start_btn = gr.Button(
                        "🗑️ Clear Start Image",
                        visible=False,
                        elem_id="clear_start_image_btn",
                        variant="secondary",
                        size="sm",
                        elem_classes=["clear-image-btn"]
                    )
                    
                    clear_end_btn = gr.Button(
                        "🗑️ Clear End Image", 
                        visible=False,
                        elem_id="clear_end_image_btn",
                        variant="secondary", 
                        size="sm",
                        elem_classes=["clear-image-btn"]
                    )
                    
                    # Image upload help text - context-sensitive
                    image_help_text = gr.Markdown(
                        value=self._get_image_help_text("t2v-A14B"),
                        visible=False,
                        elem_classes=["image-help"]
                    )
                    
                    # Image validation feedback (legacy - kept for compatibility)
                    image_validation_display = gr.HTML(
                        value="",
                        visible=False,
                        elem_classes=["validation-feedback"]
                    )
                    
                    # End image validation feedback (legacy - kept for compatibility)
                    end_image_validation_display = gr.HTML(
                        value="",
                        visible=False,
                        elem_classes=["validation-feedback"]
                    )
                    
                    # Resolution selection with model-specific options
                    try:
                        from resolution_manager import get_resolution_manager
                        resolution_manager = get_resolution_manager()
                        default_choices = resolution_manager.get_resolution_options("t2v-A14B")
                        default_value = resolution_manager.get_default_resolution("t2v-A14B")
                        default_info = resolution_manager.get_resolution_info("t2v-A14B")
                    except Exception as e:
                        logger.warning(f"Failed to load resolution manager: {e}")
                        default_choices = ["1280x720", "1280x704", "1920x1080"]
                        default_value = "1280x720"
                        default_info = "Higher resolutions require more VRAM and processing time"
                    
                    resolution = gr.Dropdown(
                        choices=default_choices,
                        value=default_value,
                        label="Resolution",
                        info=default_info
                    )
                    
                    # Enhanced LoRA settings with multi-selection support
                    with gr.Accordion("LoRA Settings", open=False):
                        
                        # LoRA selection status display
                        lora_status_display = gr.HTML(
                            value=self._get_lora_selection_status_html(),
                            elem_classes=["lora-selection-summary"]
                        )
                        
                        # LoRA selection dropdown with multi-select capability
                        with gr.Row():
                            with gr.Column(scale=2):
                                lora_dropdown = gr.Dropdown(
                                    choices=self._get_available_lora_choices(),
                                    value=None,
                                    label="Select LoRA - Choose a LoRA to add to your selection (max 5)",
                                    allow_custom_value=False,
                                    interactive=True
                                )
                            
                            with gr.Column(scale=1):
                                add_lora_btn = gr.Button(
                                    "➕ Add LoRA",
                                    variant="primary",
                                    size="sm"
                                )
                                
                                refresh_lora_btn = gr.Button(
                                    "🔄 Refresh",
                                    variant="secondary",
                                    size="sm"
                                )
                        
                        # Quick selection for recently used LoRAs
                        with gr.Row():
                            gr.Markdown("**Quick Selection:**")
                        
                        with gr.Row():
                            recent_lora_1_btn = gr.Button(
                                "📌 Recent 1",
                                variant="secondary",
                                size="sm",
                                scale=1
                            )
                            recent_lora_2_btn = gr.Button(
                                "📌 Recent 2", 
                                variant="secondary",
                                size="sm",
                                scale=1
                            )
                            recent_lora_3_btn = gr.Button(
                                "📌 Recent 3",
                                variant="secondary", 
                                size="sm",
                                scale=1
                            )
                            clear_all_loras_btn = gr.Button(
                                "🗑️ Clear All",
                                variant="secondary",
                                size="sm",
                                scale=1
                            )
                        
                        # Individual strength sliders for selected LoRAs (dynamic)
                        selected_loras_container = gr.HTML(
                            value=self._get_selected_loras_controls_html(),
                            elem_classes=["selected-loras-container"]
                        )
                        
                        # LoRA memory usage display and warnings
                        lora_memory_display = gr.HTML(
                            value=self._get_lora_memory_display_html(),
                            elem_classes=["lora-memory-info"]
                        )
                        
                        # LoRA compatibility validation display
                        lora_compatibility_display = gr.HTML(
                            value=self._get_lora_compatibility_display_html("t2v-A14B"),
                            elem_classes=["lora-compatibility-info"]
                        )
                        
                        # Legacy single LoRA path support (hidden by default)
                        with gr.Accordion("Legacy LoRA Path", open=False):
                            lora_path = gr.Textbox(
                                label="LoRA Path - For backward compatibility. Use dropdown selection above for better experience.",
                                placeholder="Path to LoRA weights file (optional)",
                                scale=3
                            )
                            
                            lora_strength = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="LoRA Strength - 0.0 = No effect, 1.0 = Full strength, 2.0 = Enhanced effect (may cause artifacts)"
                            )
                        
                        # LoRA help text
                        gr.Markdown("""
                        **Enhanced LoRA Tips:**
                        - Select up to 5 LoRAs simultaneously for complex styles
                        - Each LoRA has individual strength control (0.0-2.0)
                        - Memory usage is estimated and displayed in real-time
                        - Compatibility with current model is automatically validated
                        - Use Quick Selection buttons for recently used LoRAs
                        - Higher strength values increase style influence but may reduce prompt adherence
                        """, elem_classes=["lora-help"])
                    
                    # Generation steps
                    steps = gr.Slider(
                        minimum=20,
                        maximum=100,
                        value=50,
                        step=5,
                        label="Generation Steps - 20-30: Fast (lower quality), 50: Balanced, 70+: High quality (slower)"
                    )
                    
                    # Video duration
                    duration = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=4,
                        step=1,
                        label="Video Duration (seconds) - Longer videos require more VRAM and processing time"
                    )
                    
                    # Frames per second
                    fps = gr.Slider(
                        minimum=8,
                        maximum=30,
                        value=24,
                        step=2,
                        label="Frames Per Second (FPS) - Higher FPS = smoother motion but longer generation time"
                    )
                    
                    # Steps help text
                    steps_help = gr.Markdown(
                        value="**Recommended:** 50 steps for balanced quality/speed. Use 30 for quick previews, 70+ for final renders.",
                        visible=True,
                        elem_classes=["steps-help"]
                    )
                    
                    # Duration help text
                    duration_help = gr.Markdown(
                        value="**Duration Tips:** 4 seconds is optimal for most content. Longer videos may require more VRAM and significantly increase generation time.",
                        visible=True,
                        elem_classes=["steps-help"]
                    )
                
                with gr.Column(scale=1):
                    
                    # Prompt enhancement
                    enhance_btn = gr.Button(
                        "✨ Enhance Prompt",
                        variant="secondary",
                        size="sm"
                    )
                    
                    enhanced_prompt_display = gr.Textbox(
                        label="Enhanced Prompt",
                        lines=3,
                        interactive=False,
                        visible=False
                    )
                    
                    # Generation buttons
                    generate_btn = gr.Button(
                        "🎬 Generate Now",
                        variant="primary",
                        size="lg"
                    )
                    
                    queue_btn = gr.Button(
                        "📋 Add to Queue",
                        variant="secondary",
                        size="lg"
                    )
                    
                    # Generation status
                    generation_status = gr.Textbox(
                        label="Status",
                        value="Ready",
                        interactive=False
                    )
                    
                    # Progress bar
                    progress_bar = gr.Progress()
            
            # Session Management UI
            with gr.Accordion("📁 Session Management", open=False):
                session_info_display, clear_session_button, refresh_info_button = create_session_management_ui()
            
            # Enhanced notification area with comprehensive feedback
            with gr.Row():
                with gr.Column(scale=4):
                    notification_area = gr.HTML(
                        value="",
                        visible=False,
                        elem_classes=["notification-area"]
                    )
                
                with gr.Column(scale=1):
                    clear_notification_btn = gr.Button(
                        "✖️ Clear",
                        variant="secondary",
                        size="sm",
                        visible=False,
                        elem_classes=["clear-notification-btn"]
                    )
            
            # Comprehensive validation summary
            with gr.Row():
                validation_summary = gr.HTML(
                    value="",
                    visible=False,
                    elem_classes=["validation-summary"]
                )
            
            # Enhanced progress bar with generation statistics
            with gr.Row():
                progress_display = gr.HTML(
                    value="",
                    visible=False,
                    elem_classes=["progress-display"]
                )
            
            # Output video display
            with gr.Row():
                output_video = gr.Video(
                    label="Generated Video",
                    visible=False
                )
            
            # Store UI components for event handling
            self.generation_components = {
                'model_type': model_type,
                'prompt_input': prompt_input,
                'char_count': char_count,
                'image_input': image_input,
                'end_image_input': end_image_input,
                'image_inputs_row': image_inputs_row,
                'image_status_row': image_status_row,
                'image_help_text': image_help_text,
                'resolution': resolution,
                'model_help_text': model_help_text,
                # Enhanced image preview components
                'start_image_preview': start_image_preview,
                # Session management components
                'session_info_display': session_info_display,
                'clear_session_button': clear_session_button,
                'refresh_info_button': refresh_info_button,
                'end_image_preview': end_image_preview,
                'image_summary': image_summary,
                'compatibility_status': compatibility_status,
                'clear_start_btn': clear_start_btn,
                'clear_end_btn': clear_end_btn,
                # Image requirements help components
                'start_image_requirements': start_image_requirements,
                'end_image_requirements': end_image_requirements,
                # Enhanced validation displays
                'prompt_validation_display': prompt_validation_display,
                'image_validation_display': image_validation_display,
                'end_image_validation_display': end_image_validation_display,
                'validation_summary': validation_summary,
                'progress_display': progress_display,
                # Enhanced LoRA controls
                'lora_dropdown': lora_dropdown,
                'add_lora_btn': add_lora_btn,
                'refresh_lora_btn': refresh_lora_btn,
                'recent_lora_1_btn': recent_lora_1_btn,
                'recent_lora_2_btn': recent_lora_2_btn,
                'recent_lora_3_btn': recent_lora_3_btn,
                'clear_all_loras_btn': clear_all_loras_btn,
                'selected_loras_container': selected_loras_container,
                'lora_status_display': lora_status_display,
                'lora_memory_display': lora_memory_display,
                'lora_compatibility_display': lora_compatibility_display,
                # Legacy LoRA controls
                'lora_path': lora_path,
                'lora_strength': lora_strength,
                # Other components
                'steps': steps,
                'duration': duration,
                'fps': fps,
                'enhance_btn': enhance_btn,
                'enhanced_prompt_display': enhanced_prompt_display,
                'generate_btn': generate_btn,
                'queue_btn': queue_btn,
                'generation_status': generation_status,
                'progress_bar': progress_bar,
                'notification_area': notification_area,
                'clear_notification_btn': clear_notification_btn,
                'output_video': output_video
            }
            
            # Set up enhanced event handlers for Generation tab
            self._setup_enhanced_generation_events()
    
    def _create_lora_tab(self):
        """Create the LoRA management tab interface"""
        with gr.Column(elem_classes=["tab-content"]):
            
            gr.Markdown("### 🎨 LoRA Management & Selection")
            
            with gr.Row():
                with gr.Column(scale=2):
                    
                    # File upload section
                    gr.Markdown("#### 📤 Upload LoRA Files")
                    
                    with gr.Row():
                        lora_file_upload = gr.File(
                            label="Upload LoRA File",
                            file_types=[".safetensors", ".ckpt", ".pt", ".pth"],
                            file_count="single",
                            interactive=True
                        )
                        
                        upload_btn = gr.Button(
                            "📤 Upload",
                            variant="primary",
                            size="sm"
                        )
                    
                    # Upload status
                    upload_status = gr.HTML(
                        value="",
                        visible=False,
                        elem_classes=["notification-area"]
                    )
                    
                    # LoRA library section
                    gr.Markdown("#### 📚 LoRA Library")
                    
                    with gr.Row():
                        refresh_loras_btn = gr.Button(
                            "🔄 Refresh",
                            variant="secondary",
                            size="sm"
                        )
                        
                        sort_loras = gr.Dropdown(
                            choices=["Name (A-Z)", "Name (Z-A)", "Size (Large-Small)", "Size (Small-Large)", "Date (Newest)", "Date (Oldest)"],
                            value="Name (A-Z)",
                            label="Sort By",
                            scale=2
                        )
                        
                        auto_refresh_loras = gr.Checkbox(
                            value=True,
                            label="Auto-refresh",
                            info="Automatically refresh library every 10 seconds"
                        )
                    
                    # LoRA grid display
                    lora_library_display = gr.HTML(
                        value=self._get_lora_library_html(),
                        elem_classes=["lora-grid"]
                    )
                
                with gr.Column(scale=1):
                    
                    # Selection summary
                    gr.Markdown("#### 🎯 Current Selection")
                    
                    selection_summary = gr.HTML(
                        value=self._get_selection_summary_html(),
                        elem_classes=["lora-selection-summary"]
                    )
                    
                    # Selection controls
                    with gr.Accordion("Selection Controls", open=True):
                        
                        clear_selection_btn = gr.Button(
                            "🗑️ Clear All",
                            variant="secondary",
                            size="sm"
                        )
                        
                        # Quick selection presets
                        gr.Markdown("**Quick Presets:**")
                        
                        preset_cinematic_btn = gr.Button(
                            "🎬 Cinematic",
                            variant="secondary",
                            size="sm"
                        )
                        
                        preset_anime_btn = gr.Button(
                            "🎌 Anime Style",
                            variant="secondary",
                            size="sm"
                        )
                        
                        preset_realistic_btn = gr.Button(
                            "📸 Realistic",
                            variant="secondary",
                            size="sm"
                        )
                    
                    # Memory usage display
                    gr.Markdown("#### 💾 Memory Impact")
                    
                    memory_usage_display = gr.HTML(
                        value=self._get_memory_usage_html(),
                        elem_classes=["lora-memory-info"]
                    )
                    
                    # Individual LoRA controls (dynamic)
                    gr.Markdown("#### 🎚️ Strength Controls")
                    
                    strength_controls_container = gr.HTML(
                        value=self._get_strength_controls_html(),
                        elem_classes=["strength-controls"]
                    )
            
            # File management section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🛠️ File Management")
                    
                    with gr.Row():
                        selected_lora_for_action = gr.Dropdown(
                            choices=self._get_available_lora_names(),
                            label="Select LoRA for Action",
                            interactive=True
                        )
                        
                        delete_lora_btn = gr.Button(
                            "🗑️ Delete",
                            variant="stop",
                            size="sm"
                        )
                        
                        rename_lora_btn = gr.Button(
                            "✏️ Rename",
                            variant="secondary",
                            size="sm"
                        )
                    
                    # Rename dialog (initially hidden)
                    with gr.Row(visible=False) as rename_dialog:
                        new_lora_name = gr.Textbox(
                            label="New Name",
                            placeholder="Enter new LoRA name...",
                            scale=3
                        )
                        
                        confirm_rename_btn = gr.Button(
                            "✅ Confirm",
                            variant="primary",
                            size="sm",
                            scale=1
                        )
                        
                        cancel_rename_btn = gr.Button(
                            "❌ Cancel",
                            variant="secondary",
                            size="sm",
                            scale=1
                        )
                    
                    # Action status
                    action_status = gr.HTML(
                        value="",
                        visible=False,
                        elem_classes=["notification-area"]
                    )
            
            # Store UI components for event handling
            self.lora_components = {
                'lora_file_upload': lora_file_upload,
                'upload_btn': upload_btn,
                'upload_status': upload_status,
                'refresh_loras_btn': refresh_loras_btn,
                'sort_loras': sort_loras,
                'auto_refresh_loras': auto_refresh_loras,
                'lora_library_display': lora_library_display,
                'selection_summary': selection_summary,
                'clear_selection_btn': clear_selection_btn,
                'preset_cinematic_btn': preset_cinematic_btn,
                'preset_anime_btn': preset_anime_btn,
                'preset_realistic_btn': preset_realistic_btn,
                'memory_usage_display': memory_usage_display,
                'strength_controls_container': strength_controls_container,
                'selected_lora_for_action': selected_lora_for_action,
                'delete_lora_btn': delete_lora_btn,
                'rename_lora_btn': rename_lora_btn,
                'rename_dialog': rename_dialog,
                'new_lora_name': new_lora_name,
                'confirm_rename_btn': confirm_rename_btn,
                'cancel_rename_btn': cancel_rename_btn,
                'action_status': action_status
            }
            
            # Set up event handlers for LoRA tab
            self._setup_lora_events()
    
    def _create_optimization_tab(self):
        """Create the Optimizations tab interface with enhanced compatibility system"""
        with gr.Column(elem_classes=["tab-content"]):
            
            gr.Markdown("### ⚙️ VRAM & Performance Optimizations")
            
            # System Optimizer Status Section
            if self.system_optimizer:
                gr.Markdown("#### 🔧 System Optimizer Status")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        system_optimizer_status = gr.Textbox(
                            label="Optimizer Status",
                            value="Loading...",
                            interactive=False
                        )
                        
                        hardware_profile_display = gr.Textbox(
                            label="Hardware Profile",
                            value="Loading...",
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        refresh_optimizer_btn = gr.Button(
                            "🔄 Refresh Status",
                            variant="secondary"
                        )
                        
                        run_optimization_btn = gr.Button(
                            "🚀 Run Optimization",
                            variant="primary"
                        )
                
                # Health Metrics Display
                with gr.Row():
                    gpu_temp_display = gr.Textbox(
                        label="GPU Temperature",
                        value="N/A",
                        interactive=False
                    )
                    
                    vram_usage_optimizer = gr.Textbox(
                        label="VRAM Usage (Optimizer)",
                        value="N/A",
                        interactive=False
                    )
                    
                    cpu_usage_display = gr.Textbox(
                        label="CPU Usage",
                        value="N/A",
                        interactive=False
                    )
                
                gr.Markdown("---")
            
            # Model Compatibility Section
            gr.Markdown("#### 🔍 Model Compatibility Status")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Model selection for compatibility check
                    compatibility_model_select = gr.Dropdown(
                        choices=["t2v-A14B", "i2v-A14B", "ti2v-5B"],
                        value="t2v-A14B",
                        label="Select Model to Check Compatibility"
                    )
                    
                    check_compatibility_btn = gr.Button(
                        "🔍 Check Compatibility",
                        variant="primary"
                    )
                
                with gr.Column(scale=1):
                    show_technical_details = gr.Checkbox(
                        value=False,
                        label="Show Technical Details"
                    )
            
            # Create compatibility UI components
            compatibility_components = create_compatibility_ui_components()
            
            # Store compatibility components
            self.compatibility_components = compatibility_components["compatibility"]
            self.optimization_control_components = compatibility_components["optimization"]
            
            # Traditional optimization settings
            gr.Markdown("#### ⚙️ Manual Optimization Settings")
            
            with gr.Row():
                with gr.Column():
                    
                    # Quantization settings
                    quantization_level = gr.Dropdown(
                        choices=["fp16", "bf16", "int8"],
                        value="bf16",
                        label="Quantization Level - Lower precision = less VRAM usage but potentially lower quality"
                    )
                    
                    # Model offloading
                    enable_offload = gr.Checkbox(
                        value=True,
                        label="Enable Model Offloading - Move model components between GPU and CPU memory to reduce VRAM usage"
                    )
                    
                    # VAE tile size
                    vae_tile_size = gr.Slider(
                        minimum=128,
                        maximum=512,
                        value=256,
                        step=32,
                        label="VAE Tile Size - Smaller tiles = less VRAM usage but slower processing"
                    )
                
                with gr.Column():
                    
                    # Optimization presets
                    gr.Markdown("#### Quick Presets")
                    
                    preset_low_vram = gr.Button(
                        "🔋 Low VRAM (8GB)",
                        variant="secondary"
                    )
                    
                    preset_balanced = gr.Button(
                        "⚖️ Balanced (12GB)",
                        variant="secondary"
                    )
                    
                    preset_high_quality = gr.Button(
                        "🎯 High Quality (16GB+)",
                        variant="secondary"
                    )
                    
                    # Current VRAM usage
                    gr.Markdown("#### Current VRAM Usage")
                    
                    vram_usage_display = gr.Textbox(
                        label="VRAM Status",
                        value="Loading...",
                        interactive=False
                    )
                    
                    refresh_vram_btn = gr.Button(
                        "🔄 Refresh VRAM",
                        variant="secondary",
                        size="sm"
                    )
            
            # Optimization status
            optimization_status = gr.Textbox(
                label="Optimization Status",
                value="Default settings loaded",
                interactive=False
            )
            
            # Store components for event handling
            self.optimization_components = {
                'quantization_level': quantization_level,
                'enable_offload': enable_offload,
                'vae_tile_size': vae_tile_size,
                'preset_low_vram': preset_low_vram,
                'preset_balanced': preset_balanced,
                'preset_high_quality': preset_high_quality,
                'vram_usage_display': vram_usage_display,
                'refresh_vram_btn': refresh_vram_btn,
                'optimization_status': optimization_status,
                # New compatibility components
                'compatibility_model_select': compatibility_model_select,
                'check_compatibility_btn': check_compatibility_btn,
                'show_technical_details': show_technical_details
            }
            
            # Add system optimizer components if available
            if self.system_optimizer:
                self.optimization_components.update({
                    'system_optimizer_status': system_optimizer_status,
                    'hardware_profile_display': hardware_profile_display,
                    'refresh_optimizer_btn': refresh_optimizer_btn,
                    'run_optimization_btn': run_optimization_btn,
                    'gpu_temp_display': gpu_temp_display,
                    'vram_usage_optimizer': vram_usage_optimizer,
                    'cpu_usage_display': cpu_usage_display
                })
            
            # Set up event handlers for Optimizations tab
            self._setup_optimization_events()
    
    def _create_queue_stats_tab(self):
        """Create the Queue & Stats tab interface"""
        with gr.Column(elem_classes=["tab-content"]):
            
            gr.Markdown("### 📊 Queue Management & System Statistics")
            
            with gr.Row():
                with gr.Column(scale=2):
                    
                    # Queue management
                    gr.Markdown("#### 📋 Generation Queue")
                    
                    queue_table = gr.Dataframe(
                        headers=["ID", "Model", "Prompt", "Status", "Progress", "Created"],
                        datatype=["str", "str", "str", "str", "str", "str"],
                        label="Queue Status",
                        interactive=False,
                        wrap=True
                    )
                    
                    with gr.Row():
                        clear_queue_btn = gr.Button(
                            "🗑️ Clear Queue",
                            variant="secondary"
                        )
                        
                        pause_queue_btn = gr.Button(
                            "⏸️ Pause Queue",
                            variant="secondary"
                        )
                        
                        resume_queue_btn = gr.Button(
                            "▶️ Resume Queue",
                            variant="secondary"
                        )
                
                with gr.Column(scale=1):
                    
                    # System statistics
                    gr.Markdown("#### 💻 System Statistics")
                    
                    cpu_usage = gr.Textbox(
                        label="CPU Usage",
                        value="Loading...",
                        interactive=False
                    )
                    
                    ram_usage = gr.Textbox(
                        label="RAM Usage",
                        value="Loading...",
                        interactive=False
                    )
                    
                    gpu_usage = gr.Textbox(
                        label="GPU Usage",
                        value="Loading...",
                        interactive=False
                    )
                    
                    vram_usage = gr.Textbox(
                        label="VRAM Usage",
                        value="Loading...",
                        interactive=False
                    )
                    
                    refresh_stats_btn = gr.Button(
                        "🔄 Refresh Stats",
                        variant="primary",
                        size="sm"
                    )
                    
                    # Auto-refresh toggle
                    auto_refresh = gr.Checkbox(
                        value=True,
                        label="Auto-refresh (5s) - Automatically update stats every 5 seconds"
                    )
                    
                    # Error statistics
                    gr.Markdown("#### 🚨 Error Statistics")
                    
                    error_stats_display = gr.JSON(
                        label="Recent Errors",
                        value={},
                        visible=True
                    )
                    
                    clear_errors_btn = gr.Button(
                        "🗑️ Clear Error History",
                        variant="secondary",
                        size="sm"
                    )
            
            # Queue status summary
            queue_summary = gr.Textbox(
                label="Queue Summary",
                value="No pending tasks",
                interactive=False
            )
            
            # Store components for event handling
            self.queue_stats_components = {
                'queue_table': queue_table,
                'clear_queue_btn': clear_queue_btn,
                'pause_queue_btn': pause_queue_btn,
                'resume_queue_btn': resume_queue_btn,
                'cpu_usage': cpu_usage,
                'ram_usage': ram_usage,
                'gpu_usage': gpu_usage,
                'vram_usage': vram_usage,
                'refresh_stats_btn': refresh_stats_btn,
                'auto_refresh': auto_refresh,
                'queue_summary': queue_summary,
                'error_stats_display': error_stats_display,
                'clear_errors_btn': clear_errors_btn
            }
            
            # Set up event handlers for Queue & Stats tab
            self._setup_queue_stats_events()
    
    def _create_outputs_tab(self):
        """Create the Outputs tab interface"""
        with gr.Column(elem_classes=["tab-content"]):
            
            gr.Markdown("### 📁 Generated Videos & Output Management")
            
            with gr.Row():
                with gr.Column(scale=2):
                    
                    # Video gallery
                    gr.Markdown("#### 🎬 Video Gallery")
                    
                    video_gallery = gr.Gallery(
                        label="Generated Videos",
                        show_label=False,
                        elem_id="video_gallery",
                        columns=3,
                        rows=2,
                        height="400px",
                        allow_preview=True
                    )
                    
                    # Gallery controls
                    with gr.Row():
                        refresh_gallery_btn = gr.Button(
                            "🔄 Refresh Gallery",
                            variant="secondary"
                        )
                        
                        sort_by = gr.Dropdown(
                            choices=["Date (Newest)", "Date (Oldest)", "Name (A-Z)", "Name (Z-A)"],
                            value="Date (Newest)",
                            label="Sort By"
                        )
                
                with gr.Column(scale=1):
                    
                    # Video player and details
                    gr.Markdown("#### 📺 Video Player")
                    
                    selected_video = gr.Video(
                        label="Selected Video",
                        interactive=False
                    )
                    
                    # Video metadata
                    gr.Markdown("#### 📋 Video Details")
                    
                    video_metadata = gr.JSON(
                        label="Metadata",
                        visible=True
                    )
                    
                    # File management
                    gr.Markdown("#### 🛠️ File Management")
                    
                    with gr.Row():
                        delete_video_btn = gr.Button(
                            "🗑️ Delete",
                            variant="stop",
                            size="sm"
                        )
                        
                        rename_video_btn = gr.Button(
                            "✏️ Rename",
                            variant="secondary",
                            size="sm"
                        )
                        
                        export_video_btn = gr.Button(
                            "📤 Export",
                            variant="secondary",
                            size="sm"
                        )
            
            # Output directory info
            output_info = gr.Textbox(
                label="Output Directory",
                value=f"Videos saved to: {self.config['directories']['output_directory']}",
                interactive=False
            )
            
            # Store components for event handling
            self.outputs_components = {
                'video_gallery': video_gallery,
                'refresh_gallery_btn': refresh_gallery_btn,
                'sort_by': sort_by,
                'selected_video': selected_video,
                'video_metadata': video_metadata,
                'delete_video_btn': delete_video_btn,
                'rename_video_btn': rename_video_btn,
                'export_video_btn': export_video_btn,
                'output_info': output_info
            }
            
            # Set up event handlers for Outputs tab
            self._setup_outputs_events()
    
    def _setup_generation_events(self):
        """Set up event handlers for the Generation tab with None component safety"""
        
        try:
            # Import safe event handler
            from safe_event_handler import SafeEventHandler
            safe_handler = SafeEventHandler()
            
            logger.info("Setting up generation event handlers with safe validation...")
            
            # Model type change - show/hide image input and update resolution options
            model_type_outputs = [
                self.generation_components.get('image_inputs_row'),
                self.generation_components.get('image_status_row'),
                self.generation_components.get('image_help_text'),
                self.generation_components.get('resolution'),
                self.generation_components.get('model_help_text'),
                self.generation_components.get('start_image_requirements'),
                self.generation_components.get('end_image_requirements'),
                self.generation_components.get('image_validation_display'),
                self.generation_components.get('end_image_validation_display'),
                self.generation_components.get('notification_area')
            ]
            
            safe_handler.setup_safe_event(
                component=self.generation_components.get('model_type'),
                event_type='change',
                handler_fn=lambda model_type: self._on_model_type_change_safe(model_type, len([o for o in model_type_outputs if o is not None])),
                inputs=[self.generation_components.get('model_type')],
                outputs=model_type_outputs,
                component_name='generation_model_type'
            )
            
            # Prompt input change - update character count
            safe_handler.setup_safe_event(
                component=self.generation_components.get('prompt_input'),
                event_type='change',
                handler_fn=self._update_char_count,
                inputs=[self.generation_components.get('prompt_input')],
                outputs=[self.generation_components.get('char_count')],
                component_name='generation_prompt_input'
            )
            
            # Enhance prompt button
            safe_handler.setup_safe_event(
                component=self.generation_components.get('enhance_btn'),
                event_type='click',
                handler_fn=lambda prompt: self._enhance_prompt_safe(prompt, 2),  # Fixed output count
                inputs=[self.generation_components.get('prompt_input')],
                outputs=[
                    self.generation_components.get('enhanced_prompt_display'),
                    self.generation_components.get('prompt_input')
                ],
                component_name='generation_enhance_btn'
            )
            
            # Generate button
            generate_inputs = [
                self.generation_components.get('model_type'),
                self.generation_components.get('prompt_input'),
                self.generation_components.get('image_input'),
                self.generation_components.get('end_image_input'),
                self.generation_components.get('resolution'),
                self.generation_components.get('steps'),
                self.generation_components.get('duration'),
                self.generation_components.get('fps'),
                self.generation_components.get('lora_path'),
                self.generation_components.get('lora_strength')
            ]
            
            generate_outputs = [
                self.generation_components.get('generation_status'),
                self.generation_components.get('notification_area'),
                self.generation_components.get('clear_notification_btn'),
                self.generation_components.get('output_video'),
                self.generation_components.get('progress_display'),
                self.generation_components.get('lora_status_display'),
                self.generation_components.get('selected_loras_container')
            ]
            
            safe_handler.setup_safe_event(
                component=self.generation_components.get('generate_btn'),
                event_type='click',
                handler_fn=self._generate_video_enhanced,
                inputs=generate_inputs,
                outputs=generate_outputs,
                component_name='generation_generate_btn'
            )
            
            # Log setup statistics
            safe_handler.log_setup_summary()
            stats = safe_handler.get_setup_statistics()
            logger.info(f"Generation event handlers setup completed: {stats['successful_setups']}/{stats['total_attempts']} successful")
            
            if stats['failed_setups'] > 0:
                logger.warning(f"Some generation event handlers failed to set up: {stats['failed_handlers']}")
                logger.info("Generation functionality may be limited but the UI should still work")
            
        except Exception as e:
            logger.error(f"Failed to set up generation event handlers: {e}")
            logger.info("Generation functionality will be disabled but the UI should still work")
            # Don't re-raise the exception to prevent UI creation failure
    
    def _setup_lora_events(self):
        """Set up event handlers for the LoRA tab with safe component validation"""
        try:
            # Import safe event handler
            from safe_event_handler import SafeEventHandler
            safe_handler = SafeEventHandler()
            
            logger.info("Setting up LoRA event handlers with safe validation...")
            
            # File upload handler
            safe_handler.setup_safe_event(
                component=self.lora_components.get('upload_btn'),
                event_type='click',
                handler_fn=self._handle_lora_upload,
                inputs=[self.lora_components.get('lora_file_upload')],
                outputs=[
                    self.lora_components.get('upload_status'),
                    self.lora_components.get('lora_library_display'),
                    self.lora_components.get('selected_lora_for_action')
                ],
                component_name='lora_upload_btn'
            )
            
            # Refresh LoRAs button
            safe_handler.setup_safe_event(
                component=self.lora_components.get('refresh_loras_btn'),
                event_type='click',
                handler_fn=self._refresh_lora_library,
                inputs=[],
                outputs=[
                    self.lora_components.get('lora_library_display'),
                    self.lora_components.get('selection_summary'),
                    self.lora_components.get('memory_usage_display'),
                    self.lora_components.get('strength_controls_container'),
                    self.lora_components.get('selected_lora_for_action')
                ],
                component_name='lora_refresh_btn'
            )
            
            # Sort change handler
            safe_handler.setup_safe_event(
                component=self.lora_components.get('sort_loras'),
                event_type='change',
                handler_fn=self._sort_lora_library,
                inputs=[self.lora_components.get('sort_loras')],
                outputs=[self.lora_components.get('lora_library_display')],
                component_name='lora_sort_dropdown'
            )
            
            # Clear selection button
            safe_handler.setup_safe_event(
                component=self.lora_components.get('clear_selection_btn'),
                event_type='click',
                handler_fn=self._clear_lora_selection,
                inputs=[],
                outputs=[
                    self.lora_components.get('selection_summary'),
                    self.lora_components.get('memory_usage_display'),
                    self.lora_components.get('strength_controls_container'),
                    self.lora_components.get('lora_library_display')
                ],
                component_name='lora_clear_selection_btn'
            )
            
            # Preset buttons
            safe_handler.setup_safe_event(
                component=self.lora_components.get('preset_cinematic_btn'),
                event_type='click',
                handler_fn=lambda: self._apply_lora_preset("cinematic"),
                inputs=[],
                outputs=[
                    self.lora_components.get('selection_summary'),
                    self.lora_components.get('memory_usage_display'),
                    self.lora_components.get('strength_controls_container'),
                    self.lora_components.get('lora_library_display')
                ],
                component_name='lora_preset_cinematic_btn'
            )
            
            safe_handler.setup_safe_event(
                component=self.lora_components.get('preset_anime_btn'),
                event_type='click',
                handler_fn=lambda: self._apply_lora_preset("anime"),
                inputs=[],
                outputs=[
                    self.lora_components.get('selection_summary'),
                    self.lora_components.get('memory_usage_display'),
                    self.lora_components.get('strength_controls_container'),
                    self.lora_components.get('lora_library_display')
                ],
                component_name='lora_preset_anime_btn'
            )
            
            safe_handler.setup_safe_event(
                component=self.lora_components.get('preset_realistic_btn'),
                event_type='click',
                handler_fn=lambda: self._apply_lora_preset("realistic"),
                inputs=[],
                outputs=[
                    self.lora_components.get('selection_summary'),
                    self.lora_components.get('memory_usage_display'),
                    self.lora_components.get('strength_controls_container'),
                    self.lora_components.get('lora_library_display')
                ],
                component_name='lora_preset_realistic_btn'
            )
            
            # Delete LoRA button
            safe_handler.setup_safe_event(
                component=self.lora_components.get('delete_lora_btn'),
                event_type='click',
                handler_fn=self._delete_lora_file,
                inputs=[self.lora_components.get('selected_lora_for_action')],
                outputs=[
                    self.lora_components.get('action_status'),
                    self.lora_components.get('lora_library_display'),
                    self.lora_components.get('selected_lora_for_action')
                ],
                component_name='lora_delete_btn'
            )
            
            # Rename LoRA button
            safe_handler.setup_safe_event(
                component=self.lora_components.get('rename_lora_btn'),
                event_type='click',
                handler_fn=self._show_rename_dialog,
                inputs=[self.lora_components.get('selected_lora_for_action')],
                outputs=[
                    self.lora_components.get('rename_dialog'),
                    self.lora_components.get('new_lora_name')
                ],
                component_name='lora_rename_btn'
            )
            
            # Confirm rename button
            safe_handler.setup_safe_event(
                component=self.lora_components.get('confirm_rename_btn'),
                event_type='click',
                handler_fn=self._confirm_rename_lora,
                inputs=[
                    self.lora_components.get('selected_lora_for_action'),
                    self.lora_components.get('new_lora_name')
                ],
                outputs=[
                    self.lora_components.get('action_status'),
                    self.lora_components.get('rename_dialog'),
                    self.lora_components.get('lora_library_display'),
                    self.lora_components.get('selected_lora_for_action')
                ],
                component_name='lora_confirm_rename_btn'
            )
            
            # Cancel rename button
            safe_handler.setup_safe_event(
                component=self.lora_components.get('cancel_rename_btn'),
                event_type='click',
                handler_fn=self._cancel_rename_dialog,
                inputs=[],
                outputs=[
                    self.lora_components.get('rename_dialog'),
                    self.lora_components.get('new_lora_name')
                ],
                component_name='lora_cancel_rename_btn'
            )
            
            # Log setup statistics
            safe_handler.log_setup_summary()
            stats = safe_handler.get_setup_statistics()
            logger.info(f"LoRA event handlers setup completed: {stats['successful_setups']}/{stats['total_attempts']} successful")
            
            if stats['failed_setups'] > 0:
                logger.warning(f"Some LoRA event handlers failed to set up: {stats['failed_handlers']}")
                logger.info("LoRA functionality may be limited but the UI should still work")
            
        except Exception as e:
            logger.error(f"Failed to set up LoRA event handlers: {e}")
            logger.info("LoRA functionality will be disabled but the UI should still work")
            # Don't re-raise the exception to prevent UI creation failure
    
    def _on_model_type_change_safe(self, model_type: str, num_outputs: int):
        """Safe version of model type change handler that returns the correct number of outputs"""
        try:
            # Update session with model type change
            if hasattr(self, 'ui_session_integration'):
                self.ui_session_integration._on_model_type_change(model_type)
            
            # Import resolution manager
            from resolution_manager import get_resolution_manager
            resolution_manager = get_resolution_manager()
            
            # Show image inputs for I2V and TI2V modes
            show_images = model_type in ["i2v-A14B", "ti2v-5B"]
            
            # Update resolution dropdown using the resolution manager
            resolution_update = resolution_manager.update_resolution_dropdown(model_type)
            
            # Generate comprehensive image help text using the help system
            image_help = self._get_image_help_text(model_type)
            
            # Get comprehensive model help text
            model_help = self._get_model_help_text(model_type)
            
            # Update current model type for VRAM estimation
            self.current_model_type = model_type
            
            # Clear validation messages when switching model types (requirement 8.4)
            clear_validation = "" if not show_images else None
            
            # Log the resolution update for debugging
            logger.info(f"Model type changed to {model_type}, resolution options updated, help text refreshed, validation messages cleared")
            
            # Create the full list of possible outputs
            all_outputs = [
                gr.update(visible=show_images),  # image_inputs_row visibility
                gr.update(visible=show_images),  # image_status_row visibility
                gr.update(value=image_help, visible=show_images),  # image_help_text
                resolution_update,  # resolution dropdown with proper options
                model_help,  # comprehensive model help text
                gr.update(visible=show_images),  # start_image_requirements visibility
                gr.update(visible=show_images),  # end_image_requirements visibility
                gr.update(value=clear_validation, visible=False) if clear_validation is not None else gr.update(),  # image_validation_display
                gr.update(value=clear_validation, visible=False) if clear_validation is not None else gr.update(),  # end_image_validation_display
                gr.update(value="", visible=False)  # notification_area (clear notifications)
            ]
            
            # Return only the number of outputs expected
            return tuple(all_outputs[:num_outputs])
            
        except Exception as e:
            logger.error(f"Error in model type change handler: {e}")
            # Return safe defaults for the expected number of outputs
            return tuple([gr.update() for _ in range(num_outputs)])

    def _on_model_type_change(self, model_type: str):
        """Handle model type change - show/hide image inputs and update resolution options with comprehensive help"""
        return self._on_model_type_change_safe(model_type, 10)
    
    def _get_model_help_text(self, model_type: str, mobile: bool = False) -> str:
        """Get context-sensitive help text for the selected model"""
        try:
            from help_text_system import get_model_help_text
            return get_model_help_text(model_type, mobile)
        except ImportError:
            # Fallback to basic help text
            help_texts = {
                "t2v-A14B": """
**📝 Text-to-Video (T2V-A14B)**
- **Input**: Text prompt only
- **Best for**: Creating videos from detailed text descriptions
- **Resolution**: Optimized for 1280x720 (9 min generation)
- **VRAM**: ~14GB base model size
- **Tips**: Use descriptive prompts with camera movements, lighting, and style details
                """,
                "i2v-A14B": """
**🖼️ Image-to-Video (I2V-A14B)**
- **Input**: Image + optional text prompt
- **Best for**: Animating static images or extending image content
- **Resolution**: Optimized for 1280x720 (9 min generation)
- **VRAM**: ~14GB base model size
- **Tips**: Upload high-quality images (PNG/JPG), use prompts to guide animation style
                """,
                "ti2v-5B": """
**🎬 Text-Image-to-Video (TI2V-5B)**
- **Input**: Text prompt + reference image
- **Best for**: Precise video generation combining text and visual guidance
- **Resolution**: Supports up to 1920x1080 (17 min for 1080p)
- **VRAM**: ~5GB base model size (more efficient)
- **Tips**: Use image as style/composition reference, text for motion and details
                """
            }
            return help_texts.get(model_type, "")
    
    def _get_image_help_text(self, model_type: str, mobile: bool = False) -> str:
        """Get context-sensitive image upload help text"""
        try:
            from help_text_system import get_image_help_text
            return get_image_help_text(model_type, mobile)
        except ImportError:
            # Fallback help text
            if model_type == "t2v-A14B":
                return ""  # No images needed for T2V
            elif model_type == "i2v-A14B":
                return """
**📸 Image Upload for I2V Generation**
- **Start image:** Required - will be animated into video
- **End image:** Optional - defines animation target
- **Formats:** PNG, JPG, JPEG, WebP
- **Minimum size:** 256x256 pixels
- **Tips:** Use clear, well-lit photos for best results
                """
            elif model_type == "ti2v-5B":
                return """
**📸 Image Upload for TI2V Generation**
- **Start image:** Required - provides visual reference
- **End image:** Optional - creates smooth transitions
- **Formats:** PNG, JPG, JPEG, WebP
- **Minimum size:** 256x256 pixels
- **Tips:** Combine with detailed text prompts for precise control
                """
            return ""
    
    def _get_tooltip_text(self, element: str) -> str:
        """Get tooltip text for UI elements"""
        try:
            from help_text_system import get_tooltip_text
            return get_tooltip_text(element)
        except ImportError:
            # Fallback tooltips
            tooltips = {
                "model_type": "Choose generation mode: T2V (text only), I2V (image animation), TI2V (text + image control)",
                "prompt": "Describe your video in detail. Include camera movements, lighting, style, and specific actions.",
                "resolution": "Higher resolutions look better but take longer to generate and use more VRAM.",
                "steps": "More steps = higher quality but longer generation time. 50 is balanced.",
                "duration": "Video length in seconds. Longer videos require significantly more processing time.",
                "fps": "Frames per second. Higher FPS = smoother motion but longer generation time.",
                "image_upload_area": "Click to browse or drag & drop images. Supports PNG, JPG, JPEG, WebP formats.",
                "start_image": "Start image defines the first frame. Required for I2V/TI2V modes. PNG/JPG, min 256x256px.",
                "end_image": "End image defines the final frame. Optional but provides better control. Should match start image aspect ratio.",
                "clear_image": "Remove the uploaded image and reset the upload area.",
                "image_preview": "Click to view full-size image. Hover for detailed information.",
                "validation_status": "Shows if your uploaded image meets technical requirements.",
                "compatibility_check": "Verifies that start and end images work well together."
            }
            return tooltips.get(element, "")
    
    def _get_context_sensitive_help(self, model_type: str, has_start_image: bool = False, 
                                  has_end_image: bool = False, mobile: bool = False) -> str:
        """Get comprehensive context-sensitive help"""
        try:
            from help_text_system import get_context_sensitive_help
            return get_context_sensitive_help(model_type, has_start_image, has_end_image, mobile)
        except ImportError:
            # Fallback to basic help
            return self._get_model_help_text(model_type, mobile)
    
    def _get_image_requirements_text(self, image_type: str) -> str:
        """Get requirements text for image upload areas"""
        try:
            from help_text_system import get_help_system
            help_system = get_help_system()
            if image_type in help_system.image_help:
                content = help_system.image_help[image_type].content
                # Extract just the requirements section
                lines = content.split('\n')
                requirements_section = []
                in_requirements = False
                for line in lines:
                    if '**📋 Technical Requirements:**' in line or '**📋 Technical Requirements:**' in line:
                        in_requirements = True
                        continue
                    elif in_requirements and line.strip().startswith('**') and 'Requirements' not in line:
                        break
                    elif in_requirements:
                        requirements_section.append(line)
                
                if requirements_section:
                    return '\n'.join(requirements_section)
            
            # Fallback requirements
            if image_type == "start_image":
                return """
**📋 Start Image Requirements:**
• **Formats:** PNG, JPG, JPEG, WebP
• **Minimum size:** 256x256 pixels  
• **Quality:** High resolution recommended
• **Content:** Clear, well-lit subjects work best
                """
            else:
                return """
**🎯 End Image Guidelines:**
• **Purpose:** Defines final frame target
• **Compatibility:** Should match start image aspect ratio
• **Content:** Related to start image for smooth transitions
• **Optional:** Provides better control when used
                """
        except ImportError:
            # Fallback requirements
            if image_type == "start_image":
                return """
**📋 Start Image Requirements:**
• **Formats:** PNG, JPG, JPEG, WebP
• **Minimum size:** 256x256 pixels  
• **Quality:** High resolution recommended
• **Content:** Clear, well-lit subjects work best
                """
            else:
                return """
**🎯 End Image Guidelines:**
• **Purpose:** Defines final frame target
• **Compatibility:** Should match start image aspect ratio
                """
    
    def _get_responsive_image_requirements_text(self, image_type: str) -> str:
        """Get responsive requirements text that adapts to screen size"""
        
        if image_type == "start_image":
            desktop_text = """
**📸 Start Image Requirements:**

• **Format:** PNG, JPG, JPEG, WebP
• **Min Size:** 256×256 pixels  
• **Recommended:** 1280×720 or higher
• **Purpose:** Defines the first frame of your video
• **Quality:** Higher resolution images produce better results
• **Content:** Clear, well-lit subjects work best

*Tip: The start image becomes the opening frame of your generated video*
            """
            
            mobile_text = """
**📸 Start Image:**

• Format: PNG, JPG, JPEG, WebP
• Min: 256×256 pixels
• Purpose: First video frame
• Tip: Higher resolution = better results
            """
        else:  # end image
            desktop_text = """
**🎯 End Image Requirements:**

• **Format:** PNG, JPG, JPEG, WebP
• **Min Size:** 256×256 pixels
• **Aspect:** Should match start image
• **Purpose:** Defines the final frame (optional)
• **Compatibility:** Best results when aspect ratio matches start image
• **Content:** Should relate to start image for smooth transitions

*Note: End image helps guide the video's conclusion and improves coherence*
            """
            
            mobile_text = """
**🎯 End Image:**

• Format: PNG, JPG, JPEG, WebP
• Min: 256×256 pixels
• Purpose: Final frame (optional)
• Tip: Match start image aspect ratio
            """
        
        return f"""
        <div class="responsive-requirements">
            <div class="desktop-requirements" style="display: block;">
                {desktop_text}
            </div>
            <div class="mobile-requirements" style="display: none;">
                {mobile_text}
            </div>
        </div>
        
        <script>
        // Show appropriate requirements text based on screen size
        function updateRequirementsText() {{
            const desktopReqs = document.querySelectorAll('.desktop-requirements');
            const mobileReqs = document.querySelectorAll('.mobile-requirements');
            
            if (window.innerWidth <= 768) {{
                desktopReqs.forEach(el => el.style.display = 'none');
                mobileReqs.forEach(el => el.style.display = 'block');
            }} else {{
                desktopReqs.forEach(el => el.style.display = 'block');
                mobileReqs.forEach(el => el.style.display = 'none');
            }}
        }}
        
        // Update on load and resize
        updateRequirementsText();
        window.addEventListener('resize', updateRequirementsText);
        </script>
• **Content:** Related to start image for smooth transitions
• **Optional:** Provides better control when used
                """
    
    def _get_error_help_text(self, error_type: str) -> str:
        """Get error help text with recovery suggestions"""
        try:
            from help_text_system import get_help_system
            help_system = get_help_system()
            error_help = help_system.get_error_help(error_type)
            
            suggestions_text = ""
            if error_help.get("suggestions"):
                suggestions_text = "\n\n**💡 How to fix:**\n" + "\n".join([f"• {s}" for s in error_help["suggestions"]])
            
            return f"**❌ {error_help.get('message', 'An error occurred')}**{suggestions_text}"
            
        except ImportError:
            # Fallback error help
            error_messages = {
                "invalid_format": "**❌ Unsupported image format**\n\n**💡 How to fix:**\n• Convert to PNG, JPG, JPEG, or WebP\n• Check file extension matches format",
                "too_small": "**❌ Image too small**\n\n**💡 How to fix:**\n• Use images at least 256x256 pixels\n• Upscale using image editing software",
                "aspect_mismatch": "**❌ Image aspect ratios don't match**\n\n**💡 How to fix:**\n• Crop images to same aspect ratio\n• Use consistent dimensions",
                "file_too_large": "**❌ File size too large**\n\n**💡 How to fix:**\n• Compress image to under 50MB\n• Reduce resolution if needed"
            }
            return error_messages.get(error_type, "**❌ An error occurred**\n\n**💡 How to fix:**\n• Please try again or contact support")
    
    def _update_char_count(self, prompt: str):
        """Update character count display"""
        if prompt is None:
            prompt = ""
        char_count = len(prompt)
        max_chars = self.config.get("generation", {}).get("max_prompt_length", 500)
        
        # Color coding for character count
        if char_count > max_chars:
            return f"⚠️ {char_count}/{max_chars} (too long)"
        elif char_count >= max_chars * 0.9:
            return f"⚡ {char_count}/{max_chars} (almost full)"
        else:
            return f"✅ {char_count}/{max_chars}"
    
    def _enhance_prompt(self, prompt: str):
        """Enhance the input prompt"""
        if not prompt or not prompt.strip():
            return gr.update(visible=False), prompt
        
        try:
            # Import the enhance_prompt function
            from core.services.utils import enhance_prompt
            
            enhanced = enhance_prompt(prompt)
            
            # Show the enhanced prompt display
            return gr.update(value=enhanced, visible=True), enhanced
            
        except Exception as e:
            print(f"Error enhancing prompt: {e}")
            return gr.update(visible=False), prompt
    
    def _generate_video(self, model_type: str, prompt: str, image, end_image, resolution: str, 
                       steps: int, duration: int, fps: int, lora_path: str, lora_strength: float):
        """Generate video with current settings, real-time progress, and end frame support"""
        if not prompt or not prompt.strip():
            error_notification = self._show_notification("Please enter a prompt", "error")
            return "❌ Error: Please enter a prompt", error_notification, gr.update(visible=True), None, ""
        
        # Validate prompt length
        max_chars = self.config.get("generation", {}).get("max_prompt_length", 500)
        if len(prompt) > max_chars:
            error_msg = f"Prompt too long ({len(prompt)}/{max_chars} characters)"
            error_notification = self._show_notification(error_msg, "error")
            return f"❌ Error: {error_msg}", error_notification, gr.update(visible=True), None, ""
        
        # Validate image input for I2V/TI2V modes
        if model_type in ["i2v-A14B", "ti2v-5B"] and image is None:
            error_msg = f"{model_type} mode requires an image input"
            error_notification = self._show_notification(error_msg, "error")
            return f"❌ Error: {error_msg}", error_notification, gr.update(visible=True), None, ""
        
        try:
            # Generate unique task ID
            import time
            task_id = f"gen_{int(time.time())}"
            
            # Start progress tracking
            self.progress_tracker.start_progress_tracking(task_id, steps)
            
            # Show start notification
            start_notification = self._show_notification(
                f"Starting {resolution} video generation with {model_type}...", 
                "info"
            )
            
            # Update progress display
            progress_html = self.progress_tracker.get_progress_html()
            
            # Import generation function
            from core.services.utils import generate_video
            
            # Prepare generation parameters
            gen_params = {
                "model_type": model_type,
                "prompt": prompt,
                "image": image,
                "resolution": resolution,
                "num_inference_steps": steps,
                "duration": duration,
                "fps": fps
            }
            
            # Add LoRA selection from enhanced state management
            lora_selection = self._get_lora_selection_for_generation()
            if lora_selection:
                gen_params["selected_loras"] = lora_selection
                logger.info(f"Using enhanced LoRA selection: {list(lora_selection.keys())}")
            elif lora_path and lora_path.strip():
                # Fallback to single LoRA path for backward compatibility
                gen_params["lora_path"] = lora_path.strip()
                gen_params["lora_strength"] = lora_strength
                logger.info(f"Using legacy LoRA path: {lora_path}")
            else:
                logger.info("No LoRAs selected for generation")
            
            # Create progress callback that updates the tracker
            def progress_callback(step, total_steps, **kwargs):
                # Update progress tracker
                phase = kwargs.get('phase', GenerationPhase.GENERATION.value)
                frames_processed = kwargs.get('frames_processed', step * 2)  # Estimate frames
                
                self.progress_tracker.update_progress(
                    step=step,
                    phase=phase,
                    frames_processed=frames_processed,
                    additional_data=kwargs
                )
                
                # Return progress message for status display
                progress_percent = (step / total_steps) * 100
                return f"🎬 {phase.replace('_', ' ').title()}: Step {step}/{total_steps} ({progress_percent:.1f}%)"
            
            gen_params["progress_callback"] = progress_callback
            
            # Update status with progress indication
            status = f"🎬 Generating {resolution} video with {model_type}..."
            
            # Update progress phase to model loading
            self.progress_tracker.update_phase(GenerationPhase.MODEL_LOADING.value)
            
            # Generate video with progress tracking
            result = generate_video(**gen_params)
            
            # Complete progress tracking
            final_stats = self.progress_tracker.complete_progress_tracking(result)
            
            if result.get("success"):
                output_path = result.get("output_path")
                success_notification = self._show_notification(
                    f"Video generation completed! Saved to: {os.path.basename(output_path)}", 
                    "success"
                )
                
                # Get final progress HTML
                final_progress_html = self.progress_tracker.get_progress_html()
                
                return ("✅ Generation completed successfully!", 
                       success_notification, 
                       gr.update(visible=True), 
                       output_path,
                       final_progress_html)
            else:
                error_msg = result.get("error", "Unknown error occurred")
                error_notification = self._show_notification(f"Generation failed: {error_msg}", "error")
                return (f"❌ Generation failed: {error_msg}", 
                       error_notification, 
                       gr.update(visible=True), 
                       None,
                       "")
                
        except Exception as e:
            # Complete progress tracking on error
            if self.progress_tracker.is_tracking:
                self.progress_tracker.complete_progress_tracking()
            
            status, notification, btn_update = self._handle_generation_error(e, "generation")
            return status, notification, btn_update, None, ""
    
    def _generate_video_enhanced(self, model_type: str, prompt: str, image, end_image, resolution: str, 
                                steps: int, duration: int, fps: int, lora_path: str, lora_strength: float):
        """Enhanced video generation with LoRA usage tracking, session state, and end frame support"""
        
        # Get images from session state if not provided directly
        try:
            ui_state = self.ui_session_integration.get_ui_state_for_generation()
            session_start_image = ui_state.get('start_image')
            session_end_image = ui_state.get('end_image')
            
            # Use session images if UI images are None
            if image is None and session_start_image is not None:
                image = session_start_image
                logger.info("Using start image from session state")
            
            if end_image is None and session_end_image is not None:
                end_image = session_end_image
                logger.info("Using end image from session state")
                
        except Exception as e:
            logger.warning(f"Failed to get images from session state: {e}")
        
        # Call the original generation method with end image support
        status, notification, btn_update, output_video, progress_html = self._generate_video(
            model_type, prompt, image, end_image, resolution, steps, duration, fps, lora_path, lora_strength
        )
        
        # Update LoRA usage tracking if generation was successful
        if "✅" in status and self.lora_ui_state:
            try:
                # Track LoRA usage for recently used functionality
                lora_selection = self._get_lora_selection_for_generation()
                if lora_selection:
                    self._track_lora_usage(lora_selection)
            except Exception as e:
                logger.error(f"Error tracking LoRA usage: {str(e)}")
        
        # Return updated displays
        return (
            status,
            notification,
            btn_update,
            output_video,
            progress_html,  # Progress display HTML
            self._get_lora_selection_status_html(),  # Updated LoRA status
            self._get_selected_loras_controls_html()  # Updated LoRA controls
        )
    
    def _add_to_queue(self, model_type: str, prompt: str, image, resolution: str,
                     steps: int, lora_path: str, lora_strength: float):
        """Add generation task to queue with session state support"""
        if not prompt or not prompt.strip():
            return "❌ Error: Please enter a prompt"
        
        # Validate prompt length
        max_chars = self.config.get("generation", {}).get("max_prompt_length", 500)
        if len(prompt) > max_chars:
            return f"❌ Error: Prompt too long ({len(prompt)}/{max_chars} characters)"
        
        # Get images from session state if not provided directly
        end_image = None
        try:
            ui_state = self.ui_session_integration.get_ui_state_for_generation()
            session_start_image = ui_state.get('start_image')
            session_end_image = ui_state.get('end_image')
            
            # Use session images if UI images are None
            if image is None and session_start_image is not None:
                image = session_start_image
                logger.info("Using start image from session state for queue")
            
            if session_end_image is not None:
                end_image = session_end_image
                logger.info("Using end image from session state for queue")
                
        except Exception as e:
            logger.warning(f"Failed to get images from session state for queue: {e}")
        
        # Validate image input for I2V/TI2V modes
        if model_type in ["i2v-A14B", "ti2v-5B"] and image is None:
            return f"❌ Error: {model_type} mode requires an image input"
        
        try:
            # Import queue manager
            from core.services.utils import get_queue_manager, GenerationTask
            
            queue_manager = get_queue_manager()
            
            # Get LoRA selection from enhanced state management
            lora_selection = self._get_lora_selection_for_generation()
            
            # Create generation task with session state support
            task = GenerationTask(
                model_type=model_type,
                prompt=prompt,
                image=image,
                end_image=end_image,
                resolution=resolution,
                steps=steps,
                duration=duration,
                fps=fps,
                lora_path=lora_path.strip() if lora_path and lora_path.strip() else None,
                lora_strength=lora_strength
            )
            
            # Store image data with enhanced metadata
            if image is not None or end_image is not None:
                task.store_image_data(start_image=image, end_image=end_image)
            
            # Add enhanced LoRA selection to task if available
            if lora_selection:
                if hasattr(task, 'selected_loras'):
                    task.selected_loras = lora_selection
                    logger.info(f"Added LoRA selection to queue task: {list(lora_selection.keys())}")
                else:
                    logger.warning("GenerationTask does not support selected_loras field - using legacy LoRA support")
            else:
                logger.info("No LoRAs selected for queue task")
            
            # Add to queue
            queue_manager.add_task(task)
            
            return f"✅ Task added to queue (ID: {task.id[:8]}...)"
            
        except Exception as e:
            return f"❌ Error adding to queue: {str(e)}"
    
    def _add_to_queue_enhanced(self, model_type: str, prompt: str, image, end_image, resolution: str,
                              steps: int, duration: int, fps: int, lora_path: str, lora_strength: float):
        """Enhanced queue addition with LoRA usage tracking and end frame support"""
        # Call the original queue method
        status = self._add_to_queue(model_type, prompt, image, resolution, steps, lora_path, lora_strength)
        
        # Update LoRA usage tracking if task was added successfully
        if "✅" in status and self.lora_ui_state:
            try:
                # Track LoRA usage for recently used functionality
                lora_selection = self._get_lora_selection_for_generation()
                if lora_selection:
                    self._track_lora_usage(lora_selection)
            except Exception as e:
                logger.error(f"Error tracking LoRA usage: {str(e)}")
        
        # Return updated displays
        return (
            status,
            self._get_lora_selection_status_html(),  # Updated LoRA status
            self._get_selected_loras_controls_html()  # Updated LoRA controls
        )
    
    def _on_progress_update(self, progress_data):
        """Callback for progress tracker updates"""
        try:
            # Update the progress display component if it exists
            if hasattr(self, 'generation_components') and 'progress_display' in self.generation_components:
                progress_html = self.progress_tracker.get_progress_html()
                # Note: In a real implementation, this would need to be handled differently
                # as Gradio doesn't support direct component updates from callbacks
                # Only log if progress actually changed
                if not hasattr(self, '_last_ui_progress') or self._last_ui_progress != progress_data.progress_percentage:
                    logger.debug(f"Progress update: {progress_data.progress_percentage:.1f}%")
                    self._last_ui_progress = progress_data.progress_percentage
        except Exception as e:
            logger.warning(f"Progress update callback failed: {e}")
    
    def _track_lora_usage(self, lora_selection: Dict[str, float]):
        """Track LoRA usage for recently used functionality"""
        try:
            # This would update the recently used LoRAs list
            # For now, just log the usage
            logger.info(f"Tracking LoRA usage: {list(lora_selection.keys())}")
            
            # In a full implementation, this would:
            # 1. Update usage timestamps
            # 2. Update usage counts
            # 3. Maintain a recently used list
            # 4. Save to persistent storage
            
        except Exception as e:
            logger.error(f"Error tracking LoRA usage: {str(e)}")
    
    def _validate_image_upload(self, image):
        """Validate uploaded image file"""
        if image is None:
            return "", gr.update(visible=False)
        
        try:
            # Check image format and size
            if hasattr(image, 'size'):
                width, height = image.size
                
                # Check image dimensions (reasonable limits)
                if width > 4096 or height > 4096:
                    notification = self._show_notification(
                        "⚠️ Image is very large. Consider resizing for better performance.", 
                        "warning"
                    )
                    return notification, gr.update(visible=True)
                
                if width < 64 or height < 64:
                    notification = self._show_notification(
                        "⚠️ Image is very small. This may affect generation quality.", 
                        "warning"
                    )
                    return notification, gr.update(visible=True)
                
                # Check aspect ratio
                aspect_ratio = width / height
                if aspect_ratio > 3 or aspect_ratio < 0.33:
                    notification = self._show_notification(
                        "⚠️ Unusual aspect ratio detected. Consider using a more standard ratio.", 
                        "warning"
                    )
                    return notification, gr.update(visible=True)
                
                notification = self._show_notification(
                    f"✅ Image uploaded successfully ({width}x{height})", 
                    "success"
                )
                return notification, gr.update(visible=True)
            
            return "", gr.update(visible=False)
            
        except Exception as e:
            notification = self._show_notification(
                f"❌ Error validating image: {str(e)}", 
                "error"
            )
            return notification, gr.update(visible=True)
    
    def _validate_start_image_upload(self, image, model_type="i2v-A14B"):
        """Validate uploaded start frame image file using enhanced validation"""
        try:
            from enhanced_image_validation import validate_start_image
            
            # Use enhanced validation system
            feedback = validate_start_image(image, model_type)
            
            # Generate HTML feedback
            validation_html = feedback.to_html()
            
            # Create notification based on severity
            if feedback.severity == "success":
                notification = self._show_notification(feedback.title, "success")
            elif feedback.severity == "warning":
                notification = self._show_notification(feedback.title, "warning")
            else:
                notification = self._show_notification(feedback.title, "error")
            
            return validation_html, notification, gr.update(visible=True)
            
        except ImportError:
            # Fallback to basic validation if enhanced system not available
            return self._validate_start_image_upload_basic(image)
        except Exception as e:
            error_msg = f"Failed to validate start frame: {str(e)}"
            validation_html = f'<div class="validation-error">⚠️ {error_msg}</div>'
            notification = self._show_notification(error_msg, "error")
            return validation_html, notification, gr.update(visible=True)
    
    def _validate_start_image_upload_basic(self, image):
        """Basic fallback validation for start image"""
        if image is None:
            return "", "", gr.update(visible=False)
        
        try:
            # Get image dimensions
            width, height = image.size
            
            # Validate dimensions
            if width < 256 or height < 256:
                error_msg = f"Start frame too small ({width}x{height}). Minimum size: 256x256"
                validation_html = f'<div class="validation-error">⚠️ {error_msg}</div>'
                notification = self._show_notification(error_msg, "error")
                return validation_html, notification, gr.update(visible=True)
            
            # Success
            aspect_info = f" (aspect ratio: {width}:{height})" if width != height else " (square)"
            success_msg = f"✅ Start frame uploaded successfully ({width}x{height}){aspect_info}"
            validation_html = f'<div class="validation-success">{success_msg}</div>'
            notification = self._show_notification(success_msg, "success")
            
            return validation_html, notification, gr.update(visible=True)
            
        except Exception as e:
            error_msg = f"Failed to process start frame: {str(e)}"
            validation_html = f'<div class="validation-error">⚠️ {error_msg}</div>'
            notification = self._show_notification(error_msg, "error")
            return validation_html, notification, gr.update(visible=True)
    
    def _validate_end_image_upload(self, end_image, model_type="i2v-A14B"):
        """Validate uploaded end frame image file using enhanced validation"""
        try:
            from enhanced_image_validation import validate_end_image
            
            # Use enhanced validation system
            feedback = validate_end_image(end_image, model_type)
            
            # Generate HTML feedback
            validation_html = feedback.to_html()
            
            # Create notification based on severity
            if feedback.severity == "success":
                notification = self._show_notification(feedback.title, "success")
            elif feedback.severity == "warning":
                notification = self._show_notification(feedback.title, "warning")
            else:
                notification = self._show_notification(feedback.title, "error")
            
            return validation_html, notification, gr.update(visible=True)
            
        except ImportError:
            # Fallback to basic validation if enhanced system not available
            return self._validate_end_image_upload_basic(end_image)
        except Exception as e:
            error_msg = f"Failed to validate end frame: {str(e)}"
            validation_html = f'<div class="validation-error">⚠️ {error_msg}</div>'
            notification = self._show_notification(error_msg, "error")
            return validation_html, notification, gr.update(visible=True)
    
    def _validate_end_image_upload_basic(self, end_image):
        """Basic fallback validation for end image"""
        if end_image is None:
            return "", "", gr.update(visible=False)
        
        try:
            # Get image dimensions
            width, height = end_image.size
            
            # Validate dimensions
            if width < 256 or height < 256:
                error_msg = f"End frame too small ({width}x{height}). Minimum size: 256x256"
                validation_html = f'<div class="validation-error">⚠️ {error_msg}</div>'
                notification = self._show_notification(error_msg, "error")
                return validation_html, notification, gr.update(visible=True)
            
            # Success
            aspect_info = f" (aspect ratio: {width}:{height})" if width != height else " (square)"
            success_msg = f"✅ End frame uploaded successfully ({width}x{height}){aspect_info}"
            validation_html = f'<div class="validation-success">{success_msg}</div>'
            notification = self._show_notification(success_msg, "success")
            
            return validation_html, notification, gr.update(visible=True)
            
        except Exception as e:
            error_msg = f"Failed to process end frame: {str(e)}"
            validation_html = f'<div class="validation-error">⚠️ {error_msg}</div>'
            notification = self._show_notification(error_msg, "error")
            return validation_html, notification, gr.update(visible=True)
    
    def _validate_image_compatibility(self, start_image, end_image):
        """Validate compatibility between start and end images"""
        if start_image is None or end_image is None:
            return "", gr.update(visible=False)
        
        try:
            from enhanced_image_validation import validate_image_pair
            
            # Use enhanced validation system
            feedback = validate_image_pair(start_image, end_image)
            
            # Generate HTML feedback
            validation_html = feedback.to_html()
            
            return validation_html, gr.update(visible=True)
            
        except ImportError:
            # Fallback to basic compatibility check
            return self._validate_image_compatibility_basic(start_image, end_image)
        except Exception as e:
            error_msg = f"Failed to validate image compatibility: {str(e)}"
            validation_html = f'<div class="validation-error">⚠️ {error_msg}</div>'
            return validation_html, gr.update(visible=True)
    
    def _validate_image_compatibility_basic(self, start_image, end_image):
        """Basic fallback compatibility validation"""
        try:
            start_w, start_h = start_image.size
            end_w, end_h = end_image.size
            
            if start_w != end_w or start_h != end_h:
                warning_msg = f"⚠️ Dimension mismatch: Start {start_w}x{start_h} vs End {end_w}x{end_h}"
                validation_html = f'<div class="validation-error">{warning_msg}</div>'
                return validation_html, gr.update(visible=True)
            else:
                success_msg = f"✅ Images are compatible ({start_w}x{start_h})"
                validation_html = f'<div class="validation-success">{success_msg}</div>'
                return validation_html, gr.update(visible=True)
                
        except Exception as e:
            error_msg = f"Failed to check compatibility: {str(e)}"
            validation_html = f'<div class="validation-error">⚠️ {error_msg}</div>'
            return validation_html, gr.update(visible=True)
    
    # Enhanced Image Preview and Management Methods
    
    def _handle_start_image_upload(self, image, model_type="i2v-A14B"):
        """Handle start image upload with enhanced preview and validation"""
        try:
            # Store image in session for persistence
            if image is not None:
                self.ui_session_integration._on_start_image_change(image)
            else:
                self.ui_session_integration._on_start_image_change(None)
            
            if self.image_preview_manager:
                # Process with enhanced preview manager
                preview_html, tooltip_data, preview_visible = self.image_preview_manager.process_image_upload(image, "start")
                
                # Get image summary and compatibility status
                summary_html = self.image_preview_manager.get_image_summary()
                compatibility_html = self.image_preview_manager.get_compatibility_status()
                
                # Also run validation for legacy compatibility
                validation_html, notification, clear_visible = self._validate_start_image_upload(image, model_type)
                
                # Show clear button when image is uploaded
                show_clear_btn = image is not None
                
                return (
                    gr.update(value=preview_html, visible=preview_visible),  # start_image_preview
                    gr.update(value=summary_html, visible=bool(summary_html.strip())),  # image_summary
                    gr.update(value=compatibility_html, visible=bool(compatibility_html.strip())),  # compatibility_status
                    validation_html,  # image_validation_display (legacy)
                    notification,  # notification_area
                    clear_visible,  # clear_notification_btn
                    gr.update(visible=show_clear_btn)  # clear_start_btn
                )
            else:
                # Fallback to basic validation
                validation_html, notification, clear_visible = self._validate_start_image_upload(image, model_type)
                show_clear_btn = image is not None
                
                return (
                    gr.update(value="", visible=False),  # start_image_preview
                    gr.update(value="", visible=False),  # image_summary
                    gr.update(value="", visible=False),  # compatibility_status
                    validation_html,  # image_validation_display
                    notification,  # notification_area
                    clear_visible,  # clear_notification_btn
                    gr.update(visible=show_clear_btn)  # clear_start_btn
                )
                
        except Exception as e:
            logger.error(f"Failed to handle start image upload: {e}")
            error_msg = f"Failed to process start image: {str(e)}"
            validation_html = f'<div class="validation-error">⚠️ {error_msg}</div>'
            notification = self._show_notification(error_msg, "error")
            
            return (
                gr.update(value="", visible=False),  # start_image_preview
                gr.update(value="", visible=False),  # image_summary
                gr.update(value="", visible=False),  # compatibility_status
                validation_html,  # image_validation_display
                notification,  # notification_area
                gr.update(visible=True),  # clear_notification_btn
                gr.update(visible=False)  # clear_start_btn
            )
    
    def _handle_end_image_upload(self, image, model_type="i2v-A14B"):
        """Handle end image upload with enhanced preview and validation"""
        try:
            # Store image in session for persistence
            if image is not None:
                self.ui_session_integration._on_end_image_change(image)
            else:
                self.ui_session_integration._on_end_image_change(None)
            
            if self.image_preview_manager:
                # Process with enhanced preview manager
                preview_html, tooltip_data, preview_visible = self.image_preview_manager.process_image_upload(image, "end")
                
                # Get image summary and compatibility status
                summary_html = self.image_preview_manager.get_image_summary()
                compatibility_html = self.image_preview_manager.get_compatibility_status()
                
                # Also run validation for legacy compatibility
                validation_html, notification, clear_visible = self._validate_end_image_upload(image, model_type)
                
                # Show clear button when image is uploaded
                show_clear_btn = image is not None
                
                return (
                    gr.update(value=preview_html, visible=preview_visible),  # end_image_preview
                    gr.update(value=summary_html, visible=bool(summary_html.strip())),  # image_summary
                    gr.update(value=compatibility_html, visible=bool(compatibility_html.strip())),  # compatibility_status
                    validation_html,  # end_image_validation_display (legacy)
                    notification,  # notification_area
                    clear_visible,  # clear_notification_btn
                    gr.update(visible=show_clear_btn)  # clear_end_btn
                )
            else:
                # Fallback to basic validation
                validation_html, notification, clear_visible = self._validate_end_image_upload(image, model_type)
                show_clear_btn = image is not None
                
                return (
                    gr.update(value="", visible=False),  # end_image_preview
                    gr.update(value="", visible=False),  # image_summary
                    gr.update(value="", visible=False),  # compatibility_status
                    validation_html,  # end_image_validation_display
                    notification,  # notification_area
                    clear_visible,  # clear_notification_btn
                    gr.update(visible=show_clear_btn)  # clear_end_btn
                )
                
        except Exception as e:
            logger.error(f"Failed to handle end image upload: {e}")
            error_msg = f"Failed to process end image: {str(e)}"
            validation_html = f'<div class="validation-error">⚠️ {error_msg}</div>'
            notification = self._show_notification(error_msg, "error")
            
            return (
                gr.update(value="", visible=False),  # end_image_preview
                gr.update(value="", visible=False),  # image_summary
                gr.update(value="", visible=False),  # compatibility_status
                validation_html,  # end_image_validation_display
                notification,  # notification_area
                gr.update(visible=True)  # clear_notification_btn
            )
    
    def _clear_start_image(self):
        """Clear start image and reset preview"""
        try:
            # Clear from session
            self.ui_session_integration._on_start_image_change(None)
            
            if self.image_preview_manager:
                preview_html, tooltip_data, preview_visible = self.image_preview_manager.clear_image("start")
                summary_html = self.image_preview_manager.get_image_summary()
                compatibility_html = self.image_preview_manager.get_compatibility_status()
                
                return (
                    gr.update(value=None),  # image_input
                    gr.update(value=preview_html, visible=preview_visible),  # start_image_preview
                    gr.update(value=summary_html, visible=bool(summary_html.strip())),  # image_summary
                    gr.update(value=compatibility_html, visible=bool(compatibility_html.strip())),  # compatibility_status
                    gr.update(value="", visible=False),  # image_validation_display
                    gr.update(visible=False),  # clear_start_btn (hide when no image)
                    gr.update(value="", visible=False)  # notification_area (clear any notifications)
                )
            else:
                return (
                    gr.update(value=None),  # image_input
                    gr.update(value="", visible=False),  # start_image_preview
                    gr.update(value="", visible=False),  # image_summary
                    gr.update(value="", visible=False),  # compatibility_status
                    gr.update(value="", visible=False),  # image_validation_display
                    gr.update(visible=False),  # clear_start_btn
                    gr.update(value="", visible=False)  # notification_area
                )
                
        except Exception as e:
            logger.error(f"Failed to clear start image: {e}")
            return (
                gr.update(value=None),  # image_input
                gr.update(value="", visible=False),  # start_image_preview
                gr.update(value="", visible=False),  # image_summary
                gr.update(value="", visible=False),  # compatibility_status
                gr.update(value="", visible=False),  # image_validation_display
                gr.update(visible=False),  # clear_start_btn
                gr.update(value="", visible=False)  # notification_area
            )
    
    def _clear_end_image(self):
        """Clear end image and reset preview"""
        try:
            # Clear from session
            self.ui_session_integration._on_end_image_change(None)
            
            if self.image_preview_manager:
                preview_html, tooltip_data, preview_visible = self.image_preview_manager.clear_image("end")
                summary_html = self.image_preview_manager.get_image_summary()
                compatibility_html = self.image_preview_manager.get_compatibility_status()
                
                return (
                    gr.update(value=None),  # end_image_input
                    gr.update(value=preview_html, visible=preview_visible),  # end_image_preview
                    gr.update(value=summary_html, visible=bool(summary_html.strip())),  # image_summary
                    gr.update(value=compatibility_html, visible=bool(compatibility_html.strip())),  # compatibility_status
                    gr.update(value="", visible=False),  # end_image_validation_display
                    gr.update(visible=False),  # clear_end_btn (hide when no image)
                    gr.update(value="", visible=False)  # notification_area (clear any notifications)
                )
            else:
                return (
                    gr.update(value=None),  # end_image_input
                    gr.update(value="", visible=False),  # end_image_preview
                    gr.update(value="", visible=False),  # image_summary
                    gr.update(value="", visible=False),  # compatibility_status
                    gr.update(value="", visible=False),  # end_image_validation_display
                    gr.update(visible=False),  # clear_end_btn
                    gr.update(value="", visible=False)  # notification_area
                )
                
        except Exception as e:
            logger.error(f"Failed to clear end image: {e}")
            return (
                gr.update(value=None),  # end_image_input
                gr.update(value="", visible=False),  # end_image_preview
                gr.update(value="", visible=False),  # image_summary
                gr.update(value="", visible=False),  # compatibility_status
                gr.update(value="", visible=False),  # end_image_validation_display
                gr.update(visible=False),  # clear_end_btn
                gr.update(value="", visible=False)  # notification_area
            )
    
    def _validate_lora_path(self, lora_path):
        """Validate LoRA file path"""
        if not lora_path or not lora_path.strip():
            return "", gr.update(visible=False)
        
        try:
            from pathlib import Path
            
            path = Path(lora_path.strip())
            
            # Check if file exists
            if not path.exists():
                notification = self._show_notification(
                    "⚠️ LoRA file not found. Please check the path.", 
                    "warning"
                )
                return notification, gr.update(visible=True)
            
            # Check file extension
            valid_extensions = ['.safetensors', '.pt', '.bin', '.ckpt']
            if path.suffix.lower() not in valid_extensions:
                notification = self._show_notification(
                    f"⚠️ Unsupported LoRA format. Use: {', '.join(valid_extensions)}", 
                    "warning"
                )
                return notification, gr.update(visible=True)
            
            # Check file size (reasonable limits)
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > 1000:  # 1GB limit
                notification = self._show_notification(
                    f"⚠️ LoRA file is very large ({file_size_mb:.1f}MB). This may cause memory issues.", 
                    "warning"
                )
                return notification, gr.update(visible=True)
            
            notification = self._show_notification(
                f"✅ LoRA file validated ({file_size_mb:.1f}MB)", 
                "success"
            )
            return notification, gr.update(visible=True)
            
        except Exception as e:
            notification = self._show_notification(
                f"❌ Error validating LoRA path: {str(e)}", 
                "error"
            )
            return notification, gr.update(visible=True)
    
    def _get_lora_selection_status(self):
        """Get current LoRA selection status for display"""
        if not self.lora_ui_state:
            return "LoRA state management not available"
        
        try:
            summary = self.lora_ui_state.get_selection_summary()
            
            if summary["count"] == 0:
                return "No LoRAs selected"
            
            status_parts = [
                f"{summary['count']}/{summary['max_count']} LoRAs selected",
                f"{summary['total_memory_mb']:.1f}MB total"
            ]
            
            if not summary["is_valid"]:
                status_parts.append("⚠️ Validation errors")
            
            return " | ".join(status_parts)
            
        except Exception as e:
            return f"Error getting LoRA status: {str(e)}"
    
    def _get_lora_selection_status_html(self):
        """Get current LoRA selection status as formatted HTML"""
        if not self.lora_ui_state:
            return """
            <div style="background: #f8d7da; border: 1px solid #dc3545; border-radius: 8px; padding: 10px; color: #721c24;">
                ⚠️ LoRA state management not available
            </div>
            """
        
        try:
            summary = self.lora_ui_state.get_selection_summary()
            
            if summary["count"] == 0:
                return """
                <div style="background: #e2e3e5; border: 1px solid #6c757d; border-radius: 8px; padding: 10px; color: #495057;">
                    📝 No LoRAs selected - Choose from dropdown above
                </div>
                """
            
            # Determine status color based on validation and count
            if not summary["is_valid"]:
                bg_color = "#f8d7da"
                border_color = "#dc3545"
                text_color = "#721c24"
                icon = "⚠️"
            elif summary["count"] >= summary["max_count"]:
                bg_color = "#fff3cd"
                border_color = "#ffc107"
                text_color = "#856404"
                icon = "⚡"
            else:
                bg_color = "#d4edda"
                border_color = "#28a745"
                text_color = "#155724"
                icon = "✅"
            
            status_html = f"""
            <div style="background: {bg_color}; border: 1px solid {border_color}; border-radius: 8px; padding: 12px; color: {text_color};">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 1.2em;">{icon}</span>
                    <strong>{summary['count']}/{summary['max_count']} LoRAs Selected</strong>
                </div>
                <div style="font-size: 0.9em;">
                    💾 Total Memory: {summary['total_memory_mb']:.1f} MB ({summary['total_memory_mb']/1024:.2f} GB)
                </div>
            """
            
            if summary["validation_errors"]:
                status_html += f"""
                <div style="margin-top: 8px; font-size: 0.85em;">
                    <strong>Issues:</strong> {'; '.join(summary['validation_errors'][:2])}
                </div>
                """
            
            status_html += "</div>"
            return status_html
            
        except Exception as e:
            return f"""
            <div style="background: #f8d7da; border: 1px solid #dc3545; border-radius: 8px; padding: 10px; color: #721c24;">
                ❌ Error getting LoRA status: {str(e)}
            </div>
            """
    
    def _get_available_lora_choices(self):
        """Get list of available LoRA choices for dropdown"""
        if not self.lora_ui_state:
            return []
        
        try:
            display_data = self.lora_ui_state.get_display_data()
            choices = []
            
            # Add available LoRAs (not currently selected)
            for lora in display_data["available_loras"]:
                if lora["can_select"]:
                    choices.append((f"{lora['name']} ({lora['size_formatted']})", lora['name']))
            
            return choices
            
        except Exception as e:
            logger.error(f"Error getting LoRA choices: {str(e)}")
            return []
    
    def _get_selected_loras_controls_html(self):
        """Get HTML for individual LoRA strength controls"""
        if not self.lora_ui_state:
            return "<div>LoRA state management not available</div>"
        
        try:
            display_data = self.lora_ui_state.get_display_data()
            
            if not display_data["selected_loras"]:
                return """
                <div style="text-align: center; padding: 20px; color: #6c757d; font-style: italic;">
                    No LoRAs selected. Use the dropdown above to add LoRAs.
                </div>
                """
            
            controls_html = ""
            for lora in display_data["selected_loras"]:
                # Color coding based on strength
                if lora["strength"] < 0.5:
                    strength_color = "#28a745"  # Green for low
                elif lora["strength"] <= 1.5:
                    strength_color = "#ffc107"  # Yellow for medium
                else:
                    strength_color = "#dc3545"  # Red for high
                
                controls_html += f"""
                <div style="border: 1px solid #dee2e6; border-radius: 8px; padding: 12px; margin: 8px 0; background: #f8f9fa;">
                    <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 8px;">
                        <div style="flex: 1;">
                            <strong>{lora['name']}</strong>
                            <span style="color: #6c757d; font-size: 0.85em; margin-left: 8px;">
                                {lora['size_formatted']}
                            </span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="color: {strength_color}; font-weight: bold; font-size: 0.9em;">
                                {lora['strength']:.1f}
                            </span>
                            <button onclick="removeLora('{lora['name']}')" 
                                    style="background: #dc3545; color: white; border: none; border-radius: 4px; padding: 2px 6px; cursor: pointer; font-size: 0.8em;">
                                ✖️
                            </button>
                        </div>
                    </div>
                    <div style="margin-top: 8px;">
                        <input type="range" 
                               min="0" max="2" step="0.1" 
                               value="{lora['strength']}" 
                               onchange="updateLoraStrength('{lora['name']}', this.value)"
                               style="width: 100%; margin: 4px 0;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.75em; color: #6c757d;">
                            <span>0.0 (Off)</span>
                            <span>1.0 (Normal)</span>
                            <span>2.0 (Max)</span>
                        </div>
                    </div>
                </div>
                """
            
            # Add JavaScript for interactive controls
            controls_html += """
            <script>
            function updateLoraStrength(loraName, strength) {
                // This would trigger a Gradio event to update the LoRA strength
                console.log('Update LoRA strength:', loraName, strength);
                // Implementation would depend on Gradio's JavaScript API
            }
            
            function removeLora(loraName) {
                // This would trigger a Gradio event to remove the LoRA
                console.log('Remove LoRA:', loraName);
                // Implementation would depend on Gradio's JavaScript API
            }
            </script>
            """
            
            return controls_html
            
        except Exception as e:
            return f"<div>Error generating LoRA controls: {str(e)}</div>"
    
    def _get_lora_memory_display_html(self):
        """Get HTML for LoRA memory usage display and warnings"""
        if not self.lora_ui_state:
            return "<div>LoRA state management not available</div>"
        
        try:
            memory_estimate = self.lora_ui_state.estimate_memory_impact()
            
            if memory_estimate["total_mb"] == 0:
                return """
                <div style="background: #e2e3e5; border: 1px solid #6c757d; border-radius: 8px; padding: 10px; color: #495057;">
                    💾 No memory impact - No LoRAs selected
                </div>
                """
            
            # Determine warning level based on memory usage
            total_gb = memory_estimate["total_gb"]
            if total_gb < 2.0:
                bg_color = "#d4edda"
                border_color = "#28a745"
                text_color = "#155724"
                icon = "✅"
                warning = ""
            elif total_gb < 4.0:
                bg_color = "#fff3cd"
                border_color = "#ffc107"
                text_color = "#856404"
                icon = "⚠️"
                warning = "<div style='margin-top: 8px; font-size: 0.85em;'>⚠️ Moderate memory usage - Monitor VRAM</div>"
            else:
                bg_color = "#f8d7da"
                border_color = "#dc3545"
                text_color = "#721c24"
                icon = "🚨"
                warning = "<div style='margin-top: 8px; font-size: 0.85em;'>🚨 High memory usage - May cause VRAM issues</div>"
            
            memory_html = f"""
            <div style="background: {bg_color}; border: 1px solid {border_color}; border-radius: 8px; padding: 12px; color: {text_color};">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 1.2em;">{icon}</span>
                    <strong>Memory Impact</strong>
                </div>
                <div style="font-size: 0.9em;">
                    💾 Total: {memory_estimate['total_mb']:.1f} MB ({total_gb:.2f} GB)<br>
                    ⏱️ Est. Load Time: {memory_estimate['estimated_load_time_seconds']:.1f}s
                </div>
                {warning}
            </div>
            """
            
            return memory_html
            
        except Exception as e:
            return f"<div>Error calculating memory impact: {str(e)}</div>"
    
    def _get_lora_compatibility_display_html(self, model_type: str):
        """Get HTML for LoRA compatibility validation display"""
        if not self.lora_ui_state:
            return "<div>LoRA state management not available</div>"
        
        try:
            display_data = self.lora_ui_state.get_display_data()
            
            if not display_data["selected_loras"]:
                return """
                <div style="background: #e2e3e5; border: 1px solid #6c757d; border-radius: 8px; padding: 10px; color: #495057;">
                    🔍 No compatibility checks needed - No LoRAs selected
                </div>
                """
            
            # Check compatibility for each selected LoRA
            compatible_count = 0
            incompatible_loras = []
            
            for lora in display_data["selected_loras"]:
                # This is a simplified compatibility check
                # In a real implementation, you'd check against actual model compatibility data
                if self._check_lora_model_compatibility(lora["name"], model_type):
                    compatible_count += 1
                else:
                    incompatible_loras.append(lora["name"])
            
            total_loras = len(display_data["selected_loras"])
            
            if len(incompatible_loras) == 0:
                # All compatible
                return f"""
                <div style="background: #d4edda; border: 1px solid #28a745; border-radius: 8px; padding: 12px; color: #155724;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.2em;">✅</span>
                        <strong>All LoRAs Compatible</strong>
                    </div>
                    <div style="font-size: 0.9em; margin-top: 4px;">
                        {total_loras}/{total_loras} LoRAs are compatible with {model_type}
                    </div>
                </div>
                """
            else:
                # Some incompatible
                return f"""
                <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 12px; color: #856404;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.2em;">⚠️</span>
                        <strong>Compatibility Issues</strong>
                    </div>
                    <div style="font-size: 0.9em; margin-top: 4px;">
                        {compatible_count}/{total_loras} LoRAs compatible with {model_type}
                    </div>
                    <div style="font-size: 0.85em; margin-top: 8px;">
                        <strong>Incompatible:</strong> {', '.join(incompatible_loras[:3])}
                        {' and others...' if len(incompatible_loras) > 3 else ''}
                    </div>
                </div>
                """
            
        except Exception as e:
            return f"<div>Error checking compatibility: {str(e)}</div>"
    
    def _check_lora_model_compatibility(self, lora_name: str, model_type: str) -> bool:
        """Check if a LoRA is compatible with the given model type"""
        try:
            # This is a simplified compatibility check
            # In a real implementation, you'd have a database of LoRA-model compatibility
            
            # For now, assume most LoRAs are compatible with t2v and i2v models
            # but may have issues with ti2v models
            if model_type in ["t2v-A14B", "i2v-A14B"]:
                return True
            elif model_type == "ti2v-5B":
                # Some LoRAs might not work well with ti2v models
                # This would be based on actual compatibility data
                return not any(keyword in lora_name.lower() for keyword in ["old", "legacy", "v1"])
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking LoRA compatibility: {str(e)}")
            return True  # Default to compatible if check fails
    
    def _update_lora_selection_from_ui(self, lora_path: str, lora_strength: float):
        """Update LoRA selection based on UI inputs"""
        if not self.lora_ui_state:
            return "LoRA state management not available"
        
        try:
            if not lora_path or not lora_path.strip():
                # Clear selection if no path provided
                success, message = self.lora_ui_state.clear_selection()
                return message
            
            # Extract LoRA name from path
            from pathlib import Path
            lora_name = Path(lora_path.strip()).stem
            
            # Update selection
            success, message = self.lora_ui_state.update_selection(lora_name, lora_strength)
            return message
            
        except Exception as e:
            return f"Error updating LoRA selection: {str(e)}"
    
    def _get_lora_display_data(self):
        """Get LoRA data formatted for UI display"""
        if not self.lora_ui_state:
            return {
                "selected_loras": [],
                "available_loras": [],
                "selection_status": {"count": 0, "max_count": 5, "is_valid": True},
                "memory_info": {"formatted": "0 MB"}
            }
        
        try:
            return self.lora_ui_state.get_display_data()
        except Exception as e:
            logger.error(f"Error getting LoRA display data: {str(e)}")
            return {
                "selected_loras": [],
                "available_loras": [],
                "selection_status": {"count": 0, "max_count": 5, "is_valid": False},
                "memory_info": {"formatted": "Error"}
            }
    
    def _refresh_lora_state(self):
        """Refresh LoRA state and return updated status"""
        if not self.lora_ui_state:
            return "LoRA state management not available"
        
        try:
            success, message = self.lora_ui_state.refresh_state()
            return message
        except Exception as e:
            return f"Error refreshing LoRA state: {str(e)}"
    
    def _get_lora_selection_for_generation(self):
        """Get LoRA selection formatted for generation pipeline"""
        if not self.lora_ui_state:
            return {}
        
        try:
            return self.lora_ui_state.get_selection_for_generation()
        except Exception as e:
            logger.error(f"Error getting LoRA selection for generation: {str(e)}")
            return {}
    
    def _update_lora_ui_state(self, lora_path: str, lora_strength: float):
        """Update LoRA UI state based on current inputs"""
        try:
            # Update the LoRA selection in state management
            status_message = self._update_lora_selection_from_ui(lora_path, lora_strength)
            
            # Get updated status and memory info
            status_display = self._get_lora_selection_status()
            
            # Get memory impact
            if self.lora_ui_state:
                memory_estimate = self.lora_ui_state.estimate_memory_impact()
                if memory_estimate["total_mb"] > 0:
                    memory_display = f"{memory_estimate['total_mb']:.1f}MB ({memory_estimate['total_gb']:.2f}GB) - Est. load time: {memory_estimate['estimated_load_time_seconds']:.1f}s"
                else:
                    memory_display = "No LoRAs selected"
            else:
                memory_display = "LoRA state management not available"
            
            return status_display, memory_display
            
        except Exception as e:
            error_msg = f"Error updating LoRA UI state: {str(e)}"
            logger.error(error_msg)
            return error_msg, "Error calculating memory impact"
    
    def _add_lora_to_selection(self, selected_lora_name: str):
        """Add a LoRA to the current selection"""
        try:
            if not selected_lora_name or not self.lora_ui_state:
                notification = self._show_notification("Please select a LoRA from the dropdown", "warning")
                return (
                    gr.update(),  # lora_dropdown
                    self._get_lora_selection_status_html(),  # lora_status_display
                    self._get_selected_loras_controls_html(),  # selected_loras_container
                    self._get_lora_memory_display_html(),  # lora_memory_display
                    self._get_lora_compatibility_display_html(self.current_model_type),  # lora_compatibility_display
                    notification,  # notification_area
                    gr.update(visible=True)  # clear_notification_btn
                )
            
            # Add LoRA with default strength of 1.0
            success, message = self.lora_ui_state.update_selection(selected_lora_name, 1.0)
            
            if success:
                notification = self._show_notification(f"✅ {message}", "success")
                # Update dropdown choices to remove the selected LoRA
                new_choices = self._get_available_lora_choices()
                dropdown_update = gr.update(choices=new_choices, value=None)
            else:
                notification = self._show_notification(f"❌ {message}", "error")
                dropdown_update = gr.update()
            
            return (
                dropdown_update,  # lora_dropdown
                self._get_lora_selection_status_html(),  # lora_status_display
                self._get_selected_loras_controls_html(),  # selected_loras_container
                self._get_lora_memory_display_html(),  # lora_memory_display
                self._get_lora_compatibility_display_html(self.current_model_type),  # lora_compatibility_display
                notification,  # notification_area
                gr.update(visible=True)  # clear_notification_btn
            )
            
        except Exception as e:
            error_msg = f"Error adding LoRA to selection: {str(e)}"
            logger.error(error_msg)
            notification = self._show_notification(error_msg, "error")
            return (
                gr.update(),  # lora_dropdown
                self._get_lora_selection_status_html(),  # lora_status_display
                self._get_selected_loras_controls_html(),  # selected_loras_container
                self._get_lora_memory_display_html(),  # lora_memory_display
                self._get_lora_compatibility_display_html(self.current_model_type),  # lora_compatibility_display
                notification,  # notification_area
                gr.update(visible=True)  # clear_notification_btn
            )
    
    def _refresh_lora_dropdown(self):
        """Refresh the LoRA dropdown and all related displays"""
        try:
            if self.lora_ui_state:
                success, message = self.lora_ui_state.refresh_state()
                notification = self._show_notification(f"🔄 {message}", "info")
            else:
                notification = self._show_notification("LoRA state management not available", "warning")
            
            # Update dropdown choices
            new_choices = self._get_available_lora_choices()
            dropdown_update = gr.update(choices=new_choices, value=None)
            
            return (
                dropdown_update,  # lora_dropdown
                self._get_lora_selection_status_html(),  # lora_status_display
                self._get_selected_loras_controls_html(),  # selected_loras_container
                self._get_lora_memory_display_html(),  # lora_memory_display
                notification,  # notification_area
                gr.update(visible=True)  # clear_notification_btn
            )
            
        except Exception as e:
            error_msg = f"Error refreshing LoRA dropdown: {str(e)}"
            logger.error(error_msg)
            notification = self._show_notification(error_msg, "error")
            return (
                gr.update(),  # lora_dropdown
                self._get_lora_selection_status_html(),  # lora_status_display
                self._get_selected_loras_controls_html(),  # selected_loras_container
                self._get_lora_memory_display_html(),  # lora_memory_display
                notification,  # notification_area
                gr.update(visible=True)  # clear_notification_btn
            )
    
    def _select_recent_lora(self, index: int):
        """Select a recently used LoRA by index"""
        try:
            if not self.lora_ui_state:
                notification = self._show_notification("LoRA state management not available", "warning")
                return self._get_empty_lora_update_tuple(notification)
            
            # Get recent LoRAs (this would be implemented in LoRAUIState)
            recent_loras = self._get_recent_loras()
            
            if index >= len(recent_loras):
                notification = self._show_notification(f"No recent LoRA at position {index + 1}", "warning")
                return self._get_empty_lora_update_tuple(notification)
            
            lora_name = recent_loras[index]["name"]
            default_strength = recent_loras[index].get("last_strength", 1.0)
            
            # Add to selection
            success, message = self.lora_ui_state.update_selection(lora_name, default_strength)
            
            if success:
                notification = self._show_notification(f"✅ Added recent LoRA: {lora_name}", "success")
            else:
                notification = self._show_notification(f"❌ {message}", "error")
            
            return (
                self._get_lora_selection_status_html(),  # lora_status_display
                self._get_selected_loras_controls_html(),  # selected_loras_container
                self._get_lora_memory_display_html(),  # lora_memory_display
                self._get_lora_compatibility_display_html(self.current_model_type),  # lora_compatibility_display
                notification,  # notification_area
                gr.update(visible=True)  # clear_notification_btn
            )
            
        except Exception as e:
            error_msg = f"Error selecting recent LoRA: {str(e)}"
            logger.error(error_msg)
            notification = self._show_notification(error_msg, "error")
            return self._get_empty_lora_update_tuple(notification)
    
    def _clear_all_lora_selections(self):
        """Clear all LoRA selections"""
        try:
            if not self.lora_ui_state:
                notification = self._show_notification("LoRA state management not available", "warning")
            else:
                success, message = self.lora_ui_state.clear_selection()
                if success:
                    notification = self._show_notification(f"🗑️ {message}", "success")
                else:
                    notification = self._show_notification(f"❌ {message}", "error")
            
            # Update dropdown choices to include all available LoRAs
            new_choices = self._get_available_lora_choices()
            dropdown_update = gr.update(choices=new_choices, value=None)
            
            return (
                dropdown_update,  # lora_dropdown
                self._get_lora_selection_status_html(),  # lora_status_display
                self._get_selected_loras_controls_html(),  # selected_loras_container
                self._get_lora_memory_display_html(),  # lora_memory_display
                self._get_lora_compatibility_display_html(self.current_model_type),  # lora_compatibility_display
                notification,  # notification_area
                gr.update(visible=True)  # clear_notification_btn
            )
            
        except Exception as e:
            error_msg = f"Error clearing LoRA selections: {str(e)}"
            logger.error(error_msg)
            notification = self._show_notification(error_msg, "error")
            return (
                gr.update(),  # lora_dropdown
                self._get_lora_selection_status_html(),  # lora_status_display
                self._get_selected_loras_controls_html(),  # selected_loras_container
                self._get_lora_memory_display_html(),  # lora_memory_display
                self._get_lora_compatibility_display_html(self.current_model_type),  # lora_compatibility_display
                notification,  # notification_area
                gr.update(visible=True)  # clear_notification_btn
            )
    
    def _update_lora_compatibility_on_model_change(self, model_type: str):
        """Update LoRA compatibility display when model type changes"""
        try:
            self.current_model_type = model_type
            return self._get_lora_compatibility_display_html(model_type)
        except Exception as e:
            logger.error(f"Error updating LoRA compatibility: {str(e)}")
            return f"<div>Error updating compatibility: {str(e)}</div>"
    
    def _get_recent_loras(self):
        """Get list of recently used LoRAs"""
        try:
            # This would be implemented to track recently used LoRAs
            # For now, return a mock list
            return [
                {"name": "cinematic_v2", "last_strength": 0.8},
                {"name": "anime_style", "last_strength": 1.2},
                {"name": "realistic_enhance", "last_strength": 0.6}
            ]
        except Exception as e:
            logger.error(f"Error getting recent LoRAs: {str(e)}")
            return []
    
    def _get_empty_lora_update_tuple(self, notification):
        """Get empty update tuple for LoRA controls"""
        return (
            self._get_lora_selection_status_html(),  # lora_status_display
            self._get_selected_loras_controls_html(),  # selected_loras_container
            self._get_lora_memory_display_html(),  # lora_memory_display
            self._get_lora_compatibility_display_html(self.current_model_type),  # lora_compatibility_display
            notification,  # notification_area
            gr.update(visible=True)  # clear_notification_btn
        )
    
    def _refresh_lora_ui_state(self):
        """Refresh LoRA UI state and update displays"""
        try:
            # Refresh the LoRA state
            refresh_message = self._refresh_lora_state()
            
            # Get updated displays
            status_display = self._get_lora_selection_status()
            
            # Get memory impact
            if self.lora_ui_state:
                memory_estimate = self.lora_ui_state.estimate_memory_impact()
                if memory_estimate["total_mb"] > 0:
                    memory_display = f"{memory_estimate['total_mb']:.1f}MB ({memory_estimate['total_gb']:.2f}GB) - Est. load time: {memory_estimate['estimated_load_time_seconds']:.1f}s"
                else:
                    memory_display = "No LoRAs selected"
            else:
                memory_display = "LoRA state management not available"
            
            # Show refresh notification
            notification = self._show_notification(refresh_message, "info")
            
            return status_display, memory_display, notification, gr.update(visible=True)
            
        except Exception as e:
            error_msg = f"Error refreshing LoRA UI state: {str(e)}"
            logger.error(error_msg)
            notification = self._show_notification(error_msg, "error")
            return "Error refreshing LoRA state", "Error calculating memory impact", notification, gr.update(visible=True)
    
    def _show_notification(self, message: str, notification_type: str = "info") -> str:
        """Show a notification message with appropriate styling"""
        icons = {
            "success": "✅",
            "error": "❌", 
            "warning": "⚠️",
            "info": "ℹ️"
        }
        
        colors = {
            "success": "#d4edda",
            "error": "#f8d7da",
            "warning": "#fff3cd", 
            "info": "#d1ecf1"
        }
        
        icon = icons.get(notification_type, "ℹ️")
        color = colors.get(notification_type, "#d1ecf1")
        
        notification_html = f"""
        <div style="
            background-color: {color};
            border: 1px solid rgba(0,0,0,0.1);
            border-radius: 8px;
            padding: 12px 16px;
            margin: 8px 0;
            display: flex;
            align-items: center;
            gap: 8px;
            animation: slideIn 0.3s ease-out;
        ">
            <span style="font-size: 16px;">{icon}</span>
            <span>{message}</span>
        </div>
        <style>
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        </style>
        """
        
        return notification_html
    
    def _clear_notification(self) -> Tuple[str, gr.update]:
        """Clear notification area"""
        return "", gr.update(visible=False)
    
    def _handle_generation_error(self, error: Exception, context: str = "generation") -> Tuple[str, str, gr.update]:
        """Handle generation errors with recovery suggestions"""
        error_msg = str(error)
        
        # Provide specific error handling and recovery suggestions
        if "CUDA out of memory" in error_msg or "OutOfMemoryError" in error_msg:
            recovery_msg = """
            🔧 **VRAM Out of Memory Solutions:**
            1. Try lower resolution (720p instead of 1080p)
            2. Enable model offloading in Optimizations tab
            3. Use int8 quantization for maximum memory savings
            4. Reduce VAE tile size to 128-192
            5. Close other GPU applications
            """
            notification = self._show_notification(
                f"❌ VRAM Error: {recovery_msg}", 
                "error"
            )
        elif "Model not found" in error_msg or "404" in error_msg:
            recovery_msg = """
            🔧 **Model Loading Solutions:**
            1. Check your internet connection
            2. Verify model name is correct
            3. Try clearing model cache and re-downloading
            4. Check Hugging Face Hub status
            """
            notification = self._show_notification(
                f"❌ Model Error: {recovery_msg}", 
                "error"
            )
        elif "Invalid image" in error_msg or "PIL" in error_msg:
            recovery_msg = """
            🔧 **Image Input Solutions:**
            1. Use supported formats: PNG, JPG, JPEG, WebP
            2. Ensure image is not corrupted
            3. Try resizing image to standard dimensions
            4. Check image file size (max 10MB recommended)
            """
            notification = self._show_notification(
                f"❌ Image Error: {recovery_msg}", 
                "error"
            )
        else:
            # Generic error handling
            notification = self._show_notification(
                f"❌ {context.title()} Error: {error_msg}", 
                "error"
            )
        
        status = f"❌ {context.title()} failed: {error_msg}"
        return status, notification, gr.update(visible=True)
    
    def _get_lora_library_html(self) -> str:
        """Generate HTML for LoRA library grid display"""
        try:
            if not self.lora_ui_state:
                return "<p>LoRA state management not available</p>"
            
            display_data = self.lora_ui_state.get_display_data()
            
            # Combine selected and available LoRAs for display
            all_loras = []
            
            # Add selected LoRAs first
            for lora in display_data["selected_loras"]:
                all_loras.append({
                    "name": lora["name"],
                    "size_formatted": lora["size_formatted"],
                    "strength": lora["strength"],
                    "is_selected": True,
                    "is_valid": lora["is_valid"]
                })
            
            # Add available LoRAs
            for lora in display_data["available_loras"]:
                all_loras.append({
                    "name": lora["name"],
                    "size_formatted": lora["size_formatted"],
                    "strength": 1.0,
                    "is_selected": False,
                    "is_valid": True
                })
            
            if not all_loras:
                return """
                <div style="text-align: center; padding: 40px; color: #666;">
                    <p>📁 No LoRA files found</p>
                    <p>Upload LoRA files using the upload section above</p>
                </div>
                """
            
            # Generate HTML cards
            cards_html = ""
            for lora in all_loras:
                selected_class = "selected" if lora["is_selected"] else ""
                valid_indicator = "✅" if lora["is_valid"] else "❌"
                
                strength_display = ""
                if lora["is_selected"]:
                    strength_display = f"""
                    <div style="margin-top: 10px;">
                        <strong>Strength:</strong> {lora['strength']:.1f} ({int(lora['strength'] * 100)}%)
                    </div>
                    """
                
                cards_html += f"""
                <div class="lora-card {selected_class}" onclick="toggleLoRASelection('{lora['name']}')">
                    <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 8px;">
                        <strong>{lora['name']}</strong>
                        <span>{valid_indicator}</span>
                    </div>
                    <div style="font-size: 0.9em; color: #666;">
                        Size: {lora['size_formatted']}
                    </div>
                    {strength_display}
                    <div style="margin-top: 10px;">
                        <button onclick="selectLoRA('{lora['name']}')" 
                                style="background: #007bff; color: white; border: none; padding: 4px 8px; border-radius: 4px; font-size: 0.8em;">
                            {'Update' if lora['is_selected'] else 'Select'}
                        </button>
                        <button onclick="removeLoRA('{lora['name']}')" 
                                style="background: #dc3545; color: white; border: none; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; margin-left: 5px;">
                            Remove
                        </button>
                    </div>
                </div>
                """
            
            return f'<div class="lora-grid">{cards_html}</div>'
            
        except Exception as e:
            logger.error(f"Error generating LoRA library HTML: {str(e)}")
            return f"<p>Error loading LoRA library: {str(e)}</p>"
    
    def _get_selection_summary_html(self) -> str:
        """Generate HTML for LoRA selection summary"""
        try:
            if not self.lora_ui_state:
                return "<p>LoRA state management not available</p>"
            
            display_data = self.lora_ui_state.get_display_data()
            status = display_data["selection_status"]
            
            if status["count"] == 0:
                return """
                <div style="text-align: center; padding: 20px; color: #666;">
                    <p>🎨 No LoRAs selected</p>
                    <p>Select LoRAs from the library below</p>
                </div>
                """
            
            # Generate selected LoRAs list
            selected_html = ""
            for lora in display_data["selected_loras"]:
                valid_indicator = "✅" if lora["is_valid"] else "❌"
                selected_html += f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; border-bottom: 1px solid #eee;">
                    <div>
                        <strong>{lora['name']}</strong> {valid_indicator}
                        <br><small>{lora['size_formatted']}</small>
                    </div>
                    <div style="text-align: right;">
                        <strong>{lora['strength']:.1f}</strong>
                        <br><small>{lora['strength_percent']}%</small>
                    </div>
                </div>
                """
            
            status_color = "#28a745" if status["is_valid"] else "#dc3545"
            status_text = "Valid" if status["is_valid"] else "Invalid"
            
            return f"""
            <div style="border: 2px solid {status_color}; border-radius: 8px; padding: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <strong>Selected: {status['count']}/{status['max_count']}</strong>
                    <span style="color: {status_color};">{status_text}</span>
                </div>
                {selected_html}
            </div>
            """
            
        except Exception as e:
            logger.error(f"Error generating selection summary HTML: {str(e)}")
            return f"<p>Error loading selection summary: {str(e)}</p>"
    
    def _get_memory_usage_html(self) -> str:
        """Generate HTML for memory usage display"""
        try:
            if not self.lora_ui_state:
                return "<p>LoRA state management not available</p>"
            
            memory_estimate = self.lora_ui_state.estimate_memory_impact()
            
            if memory_estimate["total_mb"] == 0:
                return """
                <div style="text-align: center; padding: 20px; color: #666;">
                    <p>💾 No memory impact</p>
                    <p>Select LoRAs to see memory usage</p>
                </div>
                """
            
            # Determine warning level
            total_gb = memory_estimate["total_gb"]
            if total_gb > 8:
                warning_class = "lora-error"
                warning_text = "⚠️ High memory usage - may cause VRAM issues"
            elif total_gb > 4:
                warning_class = "lora-memory-warning"
                warning_text = "⚡ Moderate memory usage"
            else:
                warning_class = "lora-selection-summary"
                warning_text = "✅ Low memory usage"
            
            # Individual LoRA breakdown
            individual_html = ""
            for lora_name, size_mb in memory_estimate["individual_mb"].items():
                individual_html += f"""
                <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                    <span>{lora_name}</span>
                    <span>{size_mb:.1f}MB</span>
                </div>
                """
            
            return f"""
            <div class="{warning_class}">
                <div style="margin-bottom: 10px;">
                    <strong>Total Memory Impact:</strong>
                    <br>{memory_estimate['total_mb']:.1f}MB ({total_gb:.2f}GB)
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>Estimated Load Time:</strong>
                    <br>{memory_estimate['estimated_load_time_seconds']:.1f} seconds
                </div>
                <div style="font-size: 0.9em; border-top: 1px solid #ddd; padding-top: 10px;">
                    <strong>Individual LoRAs:</strong>
                    {individual_html}
                </div>
                <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
                    {warning_text}
                </div>
            </div>
            """
            
        except Exception as e:
            logger.error(f"Error generating memory usage HTML: {str(e)}")
            return f"<p>Error calculating memory usage: {str(e)}</p>"
    
    def _get_strength_controls_html(self) -> str:
        """Generate HTML for individual LoRA strength controls"""
        try:
            if not self.lora_ui_state:
                return "<p>LoRA state management not available</p>"
            
            display_data = self.lora_ui_state.get_display_data()
            
            if not display_data["selected_loras"]:
                return """
                <div style="text-align: center; padding: 20px; color: #666;">
                    <p>🎚️ No LoRAs selected</p>
                    <p>Select LoRAs to adjust their strength</p>
                </div>
                """
            
            controls_html = ""
            for lora in display_data["selected_loras"]:
                controls_html += f"""
                <div style="margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 6px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <strong>{lora['name']}</strong>
                        <span>{lora['strength']:.1f}</span>
                    </div>
                    <input type="range" 
                           min="0" max="2" step="0.1" 
                           value="{lora['strength']}" 
                           onchange="updateLoRAStrength('{lora['name']}', this.value)"
                           style="width: 100%;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.8em; color: #666; margin-top: 4px;">
                        <span>0.0</span>
                        <span>1.0</span>
                        <span>2.0</span>
                    </div>
                </div>
                """
            
            return controls_html
            
        except Exception as e:
            logger.error(f"Error generating strength controls HTML: {str(e)}")
            return f"<p>Error loading strength controls: {str(e)}</p>"
    
    def _get_available_lora_names(self) -> List[str]:
        """Get list of available LoRA names for dropdown"""
        try:
            if not self.lora_ui_state:
                return []
            
            display_data = self.lora_ui_state.get_display_data()
            
            # Combine selected and available LoRAs
            all_names = []
            for lora in display_data["selected_loras"]:
                all_names.append(lora["name"])
            for lora in display_data["available_loras"]:
                all_names.append(lora["name"])
            
            return sorted(all_names)
            
        except Exception as e:
            logger.error(f"Error getting available LoRA names: {str(e)}")
            return []
    
    def _handle_lora_upload(self, uploaded_file):
        """Handle LoRA file upload"""
        try:
            if uploaded_file is None:
                return self._show_notification("Please select a file to upload", "error"), self._get_lora_library_html(), gr.update()
            
            # Import upload handler
            from lora_upload_handler import LoRAUploadHandler
            
            # Initialize upload handler
            loras_dir = self.config["directories"]["loras_directory"]
            upload_handler = LoRAUploadHandler(loras_dir, self.config)
            
            # Read file data
            with open(uploaded_file.name, 'rb') as f:
                file_data = f.read()
            
            # Get filename
            filename = os.path.basename(uploaded_file.name)
            
            # Process upload
            result = upload_handler.process_upload(file_data, filename, overwrite=False)
            
            if result["success"]:
                # Refresh LoRA state
                if self.lora_ui_state:
                    self.lora_ui_state.refresh_state()
                
                success_msg = f"✅ Successfully uploaded: {result['filename']} ({result['size_mb']:.1f}MB)"
                notification = self._show_notification(success_msg, "success")
                
                # Update displays
                library_html = self._get_lora_library_html()
                dropdown_choices = gr.update(choices=self._get_available_lora_names())
                
                return notification, library_html, dropdown_choices
            else:
                error_msg = result.get("error", "Unknown upload error")
                if "suggested_name" in result:
                    error_msg += f" Suggested name: {result['suggested_name']}"
                
                notification = self._show_notification(f"❌ Upload failed: {error_msg}", "error")
                return notification, self._get_lora_library_html(), gr.update()
                
        except Exception as e:
            error_msg = f"Error uploading LoRA file: {str(e)}"
            logger.error(error_msg)
            notification = self._show_notification(error_msg, "error")
            return notification, self._get_lora_library_html(), gr.update()
    
    def _refresh_lora_library(self):
        """Refresh LoRA library display"""
        try:
            # Refresh LoRA state
            if self.lora_ui_state:
                self.lora_ui_state.refresh_state()
            
            # Update all displays
            library_html = self._get_lora_library_html()
            selection_html = self._get_selection_summary_html()
            memory_html = self._get_memory_usage_html()
            strength_html = self._get_strength_controls_html()
            dropdown_choices = gr.update(choices=self._get_available_lora_names())
            
            return library_html, selection_html, memory_html, strength_html, dropdown_choices
            
        except Exception as e:
            error_msg = f"Error refreshing LoRA library: {str(e)}"
            logger.error(error_msg)
            error_html = f"<p>Error: {error_msg}</p>"
            return error_html, error_html, error_html, error_html, gr.update()
    
    def _sort_lora_library(self, sort_option):
        """Sort LoRA library based on selected option"""
        try:
            # For now, just refresh the library
            # In a full implementation, this would sort the display data
            return self._get_lora_library_html()
            
        except Exception as e:
            error_msg = f"Error sorting LoRA library: {str(e)}"
            logger.error(error_msg)
            return f"<p>Error: {error_msg}</p>"
    
    def _clear_lora_selection(self):
        """Clear all LoRA selections"""
        try:
            if not self.lora_ui_state:
                error_msg = "LoRA state management not available"
                return error_msg, error_msg, error_msg, error_msg
            
            # Clear selection
            success, message = self.lora_ui_state.clear_selection()
            
            if success:
                # Update all displays
                selection_html = self._get_selection_summary_html()
                memory_html = self._get_memory_usage_html()
                strength_html = self._get_strength_controls_html()
                library_html = self._get_lora_library_html()
                
                return selection_html, memory_html, strength_html, library_html
            else:
                error_html = f"<p>Error: {message}</p>"
                return error_html, error_html, error_html, error_html
                
        except Exception as e:
            error_msg = f"Error clearing LoRA selection: {str(e)}"
            logger.error(error_msg)
            error_html = f"<p>Error: {error_msg}</p>"
            return error_html, error_html, error_html, error_html
    
    def _apply_lora_preset(self, preset_name):
        """Apply a LoRA preset selection"""
        try:
            if not self.lora_ui_state:
                error_msg = "LoRA state management not available"
                return error_msg, error_msg, error_msg, error_msg
            
            # Define presets (this would be configurable in a full implementation)
            presets = {
                "cinematic": [
                    ("cinematic_lora", 0.8),
                    ("film_grain", 0.6),
                    ("lighting_enhance", 0.7)
                ],
                "anime": [
                    ("anime_style", 1.0),
                    ("cel_shading", 0.9),
                    ("vibrant_colors", 0.8)
                ],
                "realistic": [
                    ("photorealistic", 1.2),
                    ("detail_enhance", 0.9),
                    ("skin_texture", 0.7)
                ]
            }
            
            if preset_name not in presets:
                error_msg = f"Unknown preset: {preset_name}"
                error_html = f"<p>Error: {error_msg}</p>"
                return error_html, error_html, error_html, error_html
            
            # Clear current selection
            self.lora_ui_state.clear_selection()
            
            # Apply preset LoRAs (only if they exist)
            available_loras = self.lora_ui_state.lora_manager.list_available_loras()
            applied_count = 0
            
            for lora_name, strength in presets[preset_name]:
                if lora_name in available_loras:
                    success, _ = self.lora_ui_state.update_selection(lora_name, strength)
                    if success:
                        applied_count += 1
            
            # Update displays
            selection_html = self._get_selection_summary_html()
            memory_html = self._get_memory_usage_html()
            strength_html = self._get_strength_controls_html()
            library_html = self._get_lora_library_html()
            
            return selection_html, memory_html, strength_html, library_html
            
        except Exception as e:
            error_msg = f"Error applying preset {preset_name}: {str(e)}"
            logger.error(error_msg)
            error_html = f"<p>Error: {error_msg}</p>"
            return error_html, error_html, error_html, error_html
    
    def _delete_lora_file(self, lora_name):
        """Delete a LoRA file"""
        try:
            if not lora_name:
                return self._show_notification("Please select a LoRA to delete", "error"), self._get_lora_library_html(), gr.update()
            
            # Get LoRA file path
            loras_dir = Path(self.config["directories"]["loras_directory"])
            
            # Find the file (check multiple extensions)
            lora_file = None
            for ext in ['.safetensors', '.pt', '.pth', '.ckpt']:
                potential_file = loras_dir / f"{lora_name}{ext}"
                if potential_file.exists():
                    lora_file = potential_file
                    break
            
            if not lora_file:
                return self._show_notification(f"LoRA file not found: {lora_name}", "error"), self._get_lora_library_html(), gr.update()
            
            # Delete the file
            lora_file.unlink()
            
            # Remove from selection if selected
            if self.lora_ui_state:
                self.lora_ui_state.remove_selection(lora_name)
                self.lora_ui_state.refresh_state()
            
            success_msg = f"✅ Successfully deleted: {lora_name}"
            notification = self._show_notification(success_msg, "success")
            
            # Update displays
            library_html = self._get_lora_library_html()
            dropdown_choices = gr.update(choices=self._get_available_lora_names())
            
            return notification, library_html, dropdown_choices
            
        except Exception as e:
            error_msg = f"Error deleting LoRA file: {str(e)}"
            logger.error(error_msg)
            notification = self._show_notification(error_msg, "error")
            return notification, self._get_lora_library_html(), gr.update()
    
    def _show_rename_dialog(self, lora_name):
        """Show rename dialog for selected LoRA"""
        try:
            if not lora_name:
                return gr.update(visible=False), ""
            
            return gr.update(visible=True), lora_name
            
        except Exception as e:
            logger.error(f"Error showing rename dialog: {str(e)}")
            return gr.update(visible=False), ""
    
    def _confirm_rename_lora(self, old_name, new_name):
        """Confirm LoRA file rename"""
        try:
            if not old_name or not new_name:
                return self._show_notification("Please provide both old and new names", "error"), gr.update(visible=False), self._get_lora_library_html(), gr.update()
            
            if old_name == new_name:
                return self._show_notification("New name must be different from old name", "error"), gr.update(visible=False), self._get_lora_library_html(), gr.update()
            
            # Get LoRA directory
            loras_dir = Path(self.config["directories"]["loras_directory"])
            
            # Find the old file
            old_file = None
            for ext in ['.safetensors', '.pt', '.pth', '.ckpt']:
                potential_file = loras_dir / f"{old_name}{ext}"
                if potential_file.exists():
                    old_file = potential_file
                    break
            
            if not old_file:
                return self._show_notification(f"LoRA file not found: {old_name}", "error"), gr.update(visible=False), self._get_lora_library_html(), gr.update()
            
            # Create new file path
            new_file = loras_dir / f"{new_name}{old_file.suffix}"
            
            # Check if new name already exists
            if new_file.exists():
                return self._show_notification(f"File already exists: {new_name}", "error"), gr.update(visible=False), self._get_lora_library_html(), gr.update()
            
            # Rename the file
            old_file.rename(new_file)
            
            # Update selection if the LoRA was selected
            if self.lora_ui_state and old_name in self.lora_ui_state.selected_loras:
                old_selection = self.lora_ui_state.selected_loras[old_name]
                self.lora_ui_state.remove_selection(old_name)
                self.lora_ui_state.update_selection(new_name, old_selection.strength)
                self.lora_ui_state.refresh_state()
            
            success_msg = f"✅ Successfully renamed: {old_name} → {new_name}"
            notification = self._show_notification(success_msg, "success")
            
            # Update displays
            library_html = self._get_lora_library_html()
            dropdown_choices = gr.update(choices=self._get_available_lora_names())
            
            return notification, gr.update(visible=False), library_html, dropdown_choices
            
        except Exception as e:
            error_msg = f"Error renaming LoRA file: {str(e)}"
            logger.error(error_msg)
            notification = self._show_notification(error_msg, "error")
            return notification, gr.update(visible=False), self._get_lora_library_html(), gr.update()
    
    def _cancel_rename_dialog(self):
        """Cancel rename dialog"""
        return gr.update(visible=False), ""
    
    def _check_system_requirements(self) -> Tuple[bool, str]:
        """Check if system meets minimum requirements for video generation"""
        try:
            import torch
            from core.services.utils import get_system_stats
            
            warnings = []
            
            # Check CUDA availability
            if not torch.cuda.is_available():
                warnings.append("⚠️ CUDA not available - CPU generation will be very slow")
            
            # Check VRAM
            try:
                stats = get_system_stats()
                vram_gb = stats.get("vram_total_mb", 0) / 1024
                
                if vram_gb < 8:
                    warnings.append(f"⚠️ Low VRAM ({vram_gb:.1f}GB) - Use int8 quantization and enable offloading")
                elif vram_gb < 12:
                    warnings.append(f"⚠️ Limited VRAM ({vram_gb:.1f}GB) - Consider using 720p resolution")
            except:
                warnings.append("⚠️ Could not detect VRAM - Monitor memory usage carefully")
            
            # Check RAM
            try:
                import psutil
                ram_gb = psutil.virtual_memory().total / (1024**3)
                
                if ram_gb < 16:
                    warnings.append(f"⚠️ Low RAM ({ram_gb:.1f}GB) - May cause system slowdown")
            except:
                pass
            
            if warnings:
                warning_msg = "\n".join(warnings)
                return False, warning_msg
            else:
                return True, "✅ System requirements check passed"
                
        except Exception as e:
            return False, f"⚠️ Could not check system requirements: {str(e)}"
    
    def _perform_startup_checks(self):
        """Perform system checks on startup"""
        try:
            requirements_ok, message = self._check_system_requirements()
            
            if not requirements_ok:
                print("🔧 System Requirements Check:")
                print(message)
                print("\n💡 Consider adjusting settings in the Optimizations tab for better performance.")
            else:
                print("✅ System requirements check passed")
                
        except Exception as e:
            print(f"⚠️ Could not perform startup checks: {e}")
    
    def _update_generation_progress(self, task_id: str, progress: float, status: str) -> Tuple[str, str]:
        """Update generation progress and status"""
        # Format progress display
        progress_text = f"🔄 {status} - {progress:.1f}% complete"
        
        # Create progress bar HTML
        progress_html = f"""
        <div style="
            width: 100%;
            background-color: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            margin: 8px 0;
        ">
            <div style="
                width: {progress}%;
                height: 20px;
                background: linear-gradient(90deg, #007bff, #0056b3);
                transition: width 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 12px;
                font-weight: bold;
            ">
                {progress:.1f}%
            </div>
        </div>
        """
        
        return progress_text, progress_html
    
    def _setup_optimization_events(self):
        """Set up event handlers for the Optimizations tab with safe component validation"""
        try:
            # Import safe event handler
            from safe_event_handler import SafeEventHandler
            safe_handler = SafeEventHandler()
            
            logger.info("Setting up optimization event handlers with safe validation...")
            
            # Optimization setting changes
            optimization_inputs = [
                self.optimization_components.get('quantization_level'),
                self.optimization_components.get('enable_offload'),
                self.optimization_components.get('vae_tile_size')
            ]
            optimization_outputs = [self.optimization_components.get('optimization_status')]
            
            safe_handler.setup_safe_event(
                component=self.optimization_components.get('quantization_level'),
                event_type='change',
                handler_fn=self._on_optimization_change,
                inputs=optimization_inputs,
                outputs=optimization_outputs,
                component_name='optimization_quantization_level'
            )
            
            safe_handler.setup_safe_event(
                component=self.optimization_components.get('enable_offload'),
                event_type='change',
                handler_fn=self._on_optimization_change,
                inputs=optimization_inputs,
                outputs=optimization_outputs,
                component_name='optimization_enable_offload'
            )
            
            safe_handler.setup_safe_event(
                component=self.optimization_components.get('vae_tile_size'),
                event_type='change',
                handler_fn=self._on_optimization_change,
                inputs=optimization_inputs,
                outputs=optimization_outputs,
                component_name='optimization_vae_tile_size'
            )
            
            # Preset buttons
            preset_outputs = [
                self.optimization_components.get('quantization_level'),
                self.optimization_components.get('enable_offload'),
                self.optimization_components.get('vae_tile_size'),
                self.optimization_components.get('optimization_status')
            ]
            
            safe_handler.setup_safe_event(
                component=self.optimization_components.get('preset_low_vram'),
                event_type='click',
                handler_fn=self._apply_low_vram_preset,
                inputs=[],
                outputs=preset_outputs,
                component_name='optimization_preset_low_vram'
            )
            
            safe_handler.setup_safe_event(
                component=self.optimization_components.get('preset_balanced'),
                event_type='click',
                handler_fn=self._apply_balanced_preset,
                inputs=[],
                outputs=preset_outputs,
                component_name='optimization_preset_balanced'
            )
            
            safe_handler.setup_safe_event(
                component=self.optimization_components.get('preset_high_quality'),
                event_type='click',
                handler_fn=self._apply_high_quality_preset,
                inputs=[],
                outputs=preset_outputs,
                component_name='optimization_preset_high_quality'
            )
            
            # VRAM refresh button
            safe_handler.setup_safe_event(
                component=self.optimization_components.get('refresh_vram_btn'),
                event_type='click',
                handler_fn=self._refresh_vram_usage,
                inputs=[],
                outputs=[self.optimization_components.get('vram_usage_display')],
                component_name='optimization_refresh_vram_btn'
            )
            
            # System optimizer events (if available)
            if self.system_optimizer and self.optimization_components.get('refresh_optimizer_btn'):
                safe_handler.setup_safe_event(
                    component=self.optimization_components.get('refresh_optimizer_btn'),
                    event_type='click',
                    handler_fn=self._refresh_system_optimizer_status,
                    inputs=[],
                    outputs=[
                        self.optimization_components.get('system_optimizer_status'),
                        self.optimization_components.get('hardware_profile_display'),
                        self.optimization_components.get('gpu_temp_display'),
                        self.optimization_components.get('vram_usage_optimizer'),
                        self.optimization_components.get('cpu_usage_display')
                    ],
                    component_name='optimization_refresh_optimizer_btn'
                )
                
                safe_handler.setup_safe_event(
                    component=self.optimization_components.get('run_optimization_btn'),
                    event_type='click',
                    handler_fn=self._run_system_optimization,
                    inputs=[],
                    outputs=[
                        self.optimization_components.get('system_optimizer_status'),
                        self.optimization_components.get('optimization_status')
                    ],
                    component_name='optimization_run_optimization_btn'
                )
            else:
                logger.info("System optimizer not available or components missing - skipping system optimizer events")
            
            # Compatibility check events
            safe_handler.setup_safe_event(
                component=self.optimization_components.get('check_compatibility_btn'),
                event_type='click',
                handler_fn=self._check_model_compatibility,
                inputs=[
                    self.optimization_components.get('compatibility_model_select'),
                    self.optimization_components.get('show_technical_details')
                ],
                outputs=[
                    self.compatibility_components.get('status') if hasattr(self, 'compatibility_components') else None,
                    self.compatibility_components.get('actions') if hasattr(self, 'compatibility_components') else None,
                    self.compatibility_components.get('details') if hasattr(self, 'compatibility_components') else None,
                    self.compatibility_components.get('progress') if hasattr(self, 'compatibility_components') else None,
                    self.compatibility_components.get('panel') if hasattr(self, 'compatibility_components') else None
                ],
                component_name='optimization_check_compatibility_btn'
            )
            
            # Model selection change for compatibility
            safe_handler.setup_safe_event(
                component=self.optimization_components.get('compatibility_model_select'),
                event_type='change',
                handler_fn=self._on_compatibility_model_change,
                inputs=[self.optimization_components.get('compatibility_model_select')],
                outputs=[
                    self.optimization_control_components.get('status') if hasattr(self, 'optimization_control_components') else None,
                    self.optimization_control_components.get('available') if hasattr(self, 'optimization_control_components') else None,
                    self.optimization_control_components.get('apply_btn') if hasattr(self, 'optimization_control_components') else None,
                    self.optimization_control_components.get('panel') if hasattr(self, 'optimization_control_components') else None
                ],
                component_name='optimization_compatibility_model_select'
            )
            
            # Apply optimizations button
            if hasattr(self, 'optimization_control_components'):
                safe_handler.setup_safe_event(
                    component=self.optimization_control_components.get('apply_btn'),
                    event_type='click',
                    handler_fn=self._apply_compatibility_optimizations,
                    inputs=[
                        self.optimization_components.get('compatibility_model_select'),
                        self.optimization_control_components.get('available')
                    ],
                    outputs=[
                        self.optimization_control_components.get('status'),
                        self.optimization_control_components.get('progress')
                    ],
                    component_name='optimization_apply_btn'
                )
            else:
                logger.info("Optimization control components not available - skipping apply button event")
            
            # Log setup statistics
            safe_handler.log_setup_summary()
            stats = safe_handler.get_setup_statistics()
            logger.info(f"Optimization event handlers setup completed: {stats['successful_setups']}/{stats['total_attempts']} successful")
            
            if stats['failed_setups'] > 0:
                logger.warning(f"Some optimization event handlers failed to set up: {stats['failed_handlers']}")
                logger.info("Optimization functionality may be limited but the UI should still work")
            
        except Exception as e:
            logger.error(f"Failed to set up optimization event handlers: {e}")
            logger.info("Optimization functionality will be disabled but the UI should still work")
            # Don't re-raise the exception to prevent UI creation failure
    
    def _on_optimization_change(self, quantization: str, enable_offload: bool, vae_tile_size: int):
        """Handle optimization setting changes"""
        # Update current optimization settings
        self.current_optimization_settings = {
            "quantization": quantization,
            "enable_offload": enable_offload,
            "vae_tile_size": vae_tile_size
        }
        
        # Estimate VRAM usage
        estimated_vram = self._estimate_vram_usage(quantization, enable_offload, vae_tile_size)
        
        return f"⚙️ Settings updated - Estimated VRAM usage: {estimated_vram:.1f}GB"
    
    def _apply_low_vram_preset(self):
        """Apply low VRAM preset (8GB target)"""
        return (
            gr.update(value="int8"),      # quantization_level
            gr.update(value=True),        # enable_offload
            gr.update(value=128),         # vae_tile_size
            "🔋 Low VRAM preset applied - Optimized for 8GB VRAM"
        )
    
    def _apply_balanced_preset(self):
        """Apply balanced preset (12GB target)"""
        return (
            gr.update(value="bf16"),      # quantization_level
            gr.update(value=True),        # enable_offload
            gr.update(value=256),         # vae_tile_size
            "⚖️ Balanced preset applied - Optimized for 12GB VRAM"
        )
    
    def _apply_high_quality_preset(self):
        """Apply high quality preset (16GB+ target)"""
        return (
            gr.update(value="fp16"),      # quantization_level
            gr.update(value=False),       # enable_offload
            gr.update(value=512),         # vae_tile_size
            "🎯 High Quality preset applied - Requires 16GB+ VRAM"
        )
    
    def _estimate_vram_usage(self, quantization: str, enable_offload: bool, vae_tile_size: int) -> float:
        """Estimate VRAM usage based on optimization settings"""
        # Base model size estimates (in GB)
        base_sizes = {
            "t2v-A14B": 14.0,
            "i2v-A14B": 14.0,
            "ti2v-5B": 5.0
        }
        
        base_size = base_sizes.get(self.current_model_type, 10.0)
        
        # Quantization multipliers
        quant_multipliers = {
            "fp16": 1.0,
            "bf16": 1.0,
            "int8": 0.5
        }
        
        # Apply quantization
        estimated = base_size * quant_multipliers.get(quantization, 1.0)
        
        # CPU offloading reduces VRAM usage
        if enable_offload:
            estimated *= 0.6  # Roughly 40% reduction
        
        # VAE tiling has minimal impact on base usage but helps with peaks
        # Smaller tiles = lower peak usage
        tile_factor = 1.0 - (512 - vae_tile_size) / 512 * 0.2
        estimated *= tile_factor
        
        return estimated
    
    def _refresh_vram_usage(self):
        """Refresh VRAM usage display"""
        try:
            vram_info = self.vram_optimizer.get_vram_usage()
            
            return f"🖥️ VRAM: {vram_info['used']:.1f}GB / {vram_info['total']:.1f}GB ({vram_info['free']:.1f}GB free)"
            
        except Exception as e:
            logger.error(f"Failed to refresh VRAM usage: {e}")
            return f"❌ Error getting VRAM info: {str(e)}"
    
    def _check_model_compatibility(self, model_id: str, show_details: bool):
        """Check model compatibility and update display"""
        try:
            return update_compatibility_ui(model_id, self.compatibility_components, show_details)
        except Exception as e:
            logger.error(f"Failed to check compatibility: {e}")
            error_html = f"""
            <div style="color: red; padding: 10px; border: 1px solid red; border-radius: 5px;">
                <strong>Error:</strong> Failed to check compatibility: {str(e)}
            </div>
            """
            return error_html, "", {}, "", True
    
    def _on_compatibility_model_change(self, model_id: str):
        """Handle compatibility model selection change"""
        try:
            return update_optimization_ui(model_id, self.optimization_control_components)
        except Exception as e:
            logger.error(f"Failed to update optimization UI: {e}")
            error_html = f"""
            <div style="color: red; padding: 10px;">
                Error updating optimization controls: {str(e)}
            </div>
            """
            return error_html, [], False, True
    
    def _refresh_system_optimizer_status(self):
        """Refresh system optimizer status and health metrics"""
        if not self.system_optimizer:
            return "❌ System optimizer not available", "N/A", "N/A", "N/A", "N/A"
        
        try:
            # Get hardware profile
            hardware_profile = self.system_optimizer.get_hardware_profile()
            if hardware_profile:
                profile_text = f"🖥️ {hardware_profile.cpu_model[:30]}... | 🎮 {hardware_profile.gpu_model} ({hardware_profile.vram_gb}GB)"
            else:
                profile_text = "❌ Hardware profile not available"
            
            # Get system health metrics
            health_metrics = self.system_optimizer.monitor_system_health()
            
            # Format status
            status_text = "✅ System optimizer active"
            if self.system_optimizer.is_initialized:
                status_text += " | Initialized"
            
            # Format health metrics
            gpu_temp = f"{health_metrics.gpu_temperature:.1f}°C" if health_metrics.gpu_temperature > 0 else "N/A"
            vram_usage = f"{health_metrics.vram_usage_mb}MB / {health_metrics.vram_total_mb}MB" if health_metrics.vram_total_mb > 0 else "N/A"
            cpu_usage = f"{health_metrics.cpu_usage_percent:.1f}%" if health_metrics.cpu_usage_percent > 0 else "N/A"
            
            return status_text, profile_text, gpu_temp, vram_usage, cpu_usage
            
        except Exception as e:
            logger.error(f"Failed to refresh system optimizer status: {e}")
            return f"❌ Error: {str(e)}", "N/A", "N/A", "N/A", "N/A"
    
    def _run_system_optimization(self):
        """Run system optimization and return status"""
        if not self.system_optimizer:
            return "❌ System optimizer not available", "System optimizer not available"
        
        try:
            # Run hardware optimizations
            opt_result = self.system_optimizer.apply_hardware_optimizations()
            
            if opt_result.success:
                status_text = f"✅ Optimization completed: {len(opt_result.optimizations_applied)} optimizations applied"
                optimization_status = f"Applied: {', '.join(opt_result.optimizations_applied[:3])}"
                if len(opt_result.optimizations_applied) > 3:
                    optimization_status += f" and {len(opt_result.optimizations_applied) - 3} more..."
            else:
                status_text = "❌ Optimization failed"
                optimization_status = f"Errors: {', '.join(opt_result.errors[:2])}"
                if len(opt_result.errors) > 2:
                    optimization_status += "..."
            
            # Add warnings if any
            if opt_result.warnings:
                status_text += f" | {len(opt_result.warnings)} warnings"
            
            return status_text, optimization_status
            
        except Exception as e:
            logger.error(f"Failed to run system optimization: {e}")
            return f"❌ Error: {str(e)}", f"Error: {str(e)}"
    
    def _initialize_optimizer_status(self):
        """Initialize system optimizer status display"""
        try:
            if hasattr(self, 'optimization_components') and 'system_optimizer_status' in self.optimization_components:
                # Get initial status
                status_data = self._refresh_system_optimizer_status()
                
                # Update the components with initial values
                # Note: This is for initialization only, actual updates happen through events
                logger.info("System optimizer status initialized")
            else:
                logger.debug("System optimizer components not yet available for initialization")
        except Exception as e:
            logger.error(f"Failed to initialize optimizer status: {e}")
    
    def _apply_compatibility_optimizations(self, model_id: str, selected_optimizations: List[str]):
        """Apply selected compatibility optimizations"""
        try:
            result = apply_optimizations_ui(model_id, selected_optimizations, self.optimization_control_components['progress'])
            
            if result.get("success", False):
                applied = result.get("applied_optimizations", [])
                status_html = f"""
                <div style="color: green; padding: 10px; border: 1px solid green; border-radius: 5px;">
                    <strong>✅ Success:</strong> Applied {len(applied)} optimizations: {', '.join(applied)}
                </div>
                """
                progress_html = """
                <div style="color: green; font-weight: bold; text-align: center; padding: 10px;">
                    ✅ Optimizations applied successfully!
                </div>
                """
            else:
                error = result.get("error", "Unknown error")
                status_html = f"""
                <div style="color: red; padding: 10px; border: 1px solid red; border-radius: 5px;">
                    <strong>❌ Error:</strong> {error}
                </div>
                """
                progress_html = f"""
                <div style="color: red; font-weight: bold; text-align: center; padding: 10px;">
                    ❌ Failed to apply optimizations: {error}
                </div>
                """
            
            return status_html, progress_html
            
        except Exception as e:
            logger.error(f"Failed to apply optimizations: {e}")
            error_html = f"""
            <div style="color: red; padding: 10px; border: 1px solid red; border-radius: 5px;">
                <strong>Error:</strong> Failed to apply optimizations: {str(e)}
            </div>
            """
            return error_html, error_html
            
            used_gb = vram_info["used_mb"] / 1024
            total_gb = vram_info["total_mb"] / 1024
            usage_percent = vram_info["usage_percent"]
            
            # Format the display
            status_emoji = "🟢" if usage_percent < 70 else "🟡" if usage_percent < 90 else "🔴"
            
            return f"{status_emoji} {used_gb:.1f}GB / {total_gb:.1f}GB ({usage_percent:.1f}%)"
            
        except Exception as e:
            return f"❌ Error reading VRAM: {str(e)}"
    
    def _setup_queue_stats_events(self):
        """Set up event handlers for the Queue & Stats tab with safe component validation"""
        try:
            # Import safe event handler
            from safe_event_handler import SafeEventHandler
            safe_handler = SafeEventHandler()
            
            logger.info("Setting up queue stats event handlers with safe validation...")
            
            # Queue management buttons
            safe_handler.setup_safe_event(
                component=self.queue_stats_components.get('clear_queue_btn'),
                event_type='click',
                handler_fn=self._clear_queue,
                inputs=[],
                outputs=[
                    self.queue_stats_components.get('queue_table'),
                    self.queue_stats_components.get('queue_summary')
                ],
                component_name='queue_stats_clear_queue_btn'
            )
            
            safe_handler.setup_safe_event(
                component=self.queue_stats_components.get('pause_queue_btn'),
                event_type='click',
                handler_fn=self._pause_queue,
                inputs=[],
                outputs=[self.queue_stats_components.get('queue_summary')],
                component_name='queue_stats_pause_queue_btn'
            )
            
            safe_handler.setup_safe_event(
                component=self.queue_stats_components.get('resume_queue_btn'),
                event_type='click',
                handler_fn=self._resume_queue,
                inputs=[],
                outputs=[self.queue_stats_components.get('queue_summary')],
                component_name='queue_stats_resume_queue_btn'
            )
            
            # Stats refresh button
            safe_handler.setup_safe_event(
                component=self.queue_stats_components.get('refresh_stats_btn'),
                event_type='click',
                handler_fn=self._refresh_system_stats,
                inputs=[],
                outputs=[
                    self.queue_stats_components.get('cpu_usage'),
                    self.queue_stats_components.get('ram_usage'),
                    self.queue_stats_components.get('gpu_usage'),
                    self.queue_stats_components.get('vram_usage'),
                    self.queue_stats_components.get('queue_table'),
                    self.queue_stats_components.get('queue_summary'),
                    self.queue_stats_components.get('error_stats_display')
                ],
                component_name='queue_stats_refresh_stats_btn'
            )
            
            # Auto-refresh toggle
            safe_handler.setup_safe_event(
                component=self.queue_stats_components.get('auto_refresh'),
                event_type='change',
                handler_fn=self._toggle_auto_refresh,
                inputs=[self.queue_stats_components.get('auto_refresh')],
                outputs=[],
                component_name='queue_stats_auto_refresh'
            )
            
            # Error statistics clear button
            safe_handler.setup_safe_event(
                component=self.queue_stats_components.get('clear_errors_btn'),
                event_type='click',
                handler_fn=self._clear_error_history,
                inputs=[],
                outputs=[self.queue_stats_components.get('error_stats_display')],
                component_name='queue_stats_clear_errors_btn'
            )
            
            # Log setup statistics
            safe_handler.log_setup_summary()
            stats = safe_handler.get_setup_statistics()
            logger.info(f"Queue stats event handlers setup completed: {stats['successful_setups']}/{stats['total_attempts']} successful")
            
            if stats['failed_setups'] > 0:
                logger.warning(f"Some queue stats event handlers failed to set up: {stats['failed_handlers']}")
                logger.info("Queue stats functionality may be limited but the UI should still work")
            
        except Exception as e:
            logger.error(f"Failed to set up queue stats event handlers: {e}")
            logger.info("Queue stats functionality will be disabled but the UI should still work")
            # Don't re-raise the exception to prevent UI creation failure
    
    @handle_error_with_recovery
    def _get_error_statistics(self) -> Dict[str, Any]:
        """Get current error statistics"""
        try:
            recovery_manager = get_error_recovery_manager()
            stats = recovery_manager.get_error_statistics()
            
            # Format for display
            if stats["total_errors"] == 0:
                return {"message": "No errors recorded", "total_errors": 0}
            
            return {
                "total_errors": stats["total_errors"],
                "by_category": stats["by_category"],
                "recent_errors": stats["recent_errors"][-5:],  # Show last 5 errors
                "last_updated": datetime.now().strftime("%H:%M:%S")
            }
            
        except Exception as e:
            log_error_with_context(e, "error_statistics_retrieval")
            return {"error": f"Failed to get error statistics: {str(e)}"}
    
    @handle_error_with_recovery
    def _clear_error_history(self) -> Dict[str, Any]:
        """Clear error history"""
        try:
            recovery_manager = get_error_recovery_manager()
            recovery_manager.error_history.clear()
            
            return {
                "message": "Error history cleared successfully",
                "total_errors": 0,
                "cleared_at": datetime.now().strftime("%H:%M:%S")
            }
            
        except Exception as e:
            log_error_with_context(e, "error_history_clearing")
            return {"error": f"Failed to clear error history: {str(e)}"}
    
    def _clear_queue(self):
        """Clear all tasks from the queue"""
        try:
            from core.services.utils import get_queue_manager
            
            queue_manager = get_queue_manager()
            
            # Get current queue size before clearing
            current_tasks = queue_manager.get_queue_status()
            task_count = len(current_tasks)
            
            if task_count == 0:
                return [], "ℹ️ Queue is already empty"
            
            # Clear all tasks
            queue_manager.clear_queue()
            
            # Return empty table and updated summary
            empty_data = []
            summary = f"✅ Cleared {task_count} task{'s' if task_count != 1 else ''} from queue"
            
            return empty_data, summary
            
        except Exception as e:
            error_msg = f"❌ Error clearing queue: {str(e)}"
            return [], error_msg
    
    def _pause_queue(self):
        """Pause queue processing"""
        try:
            from core.services.utils import get_queue_manager
            
            queue_manager = get_queue_manager()
            
            # Check if queue is already paused
            if hasattr(queue_manager, 'is_paused') and queue_manager.is_paused():
                return "ℹ️ Queue processing is already paused"
            
            queue_manager.pause_processing()
            
            return "⏸️ Queue processing paused"
            
        except Exception as e:
            return f"❌ Error pausing queue: {str(e)}"
    
    def _resume_queue(self):
        """Resume queue processing"""
        try:
            from core.services.utils import get_queue_manager
            
            queue_manager = get_queue_manager()
            
            # Check if queue is already running
            if hasattr(queue_manager, 'is_paused') and not queue_manager.is_paused():
                return "ℹ️ Queue processing is already running"
            
            queue_manager.resume_processing()
            
            return "▶️ Queue processing resumed"
            
        except Exception as e:
            return f"❌ Error resuming queue: {str(e)}"
    
    def _refresh_system_stats(self):
        """Refresh all system statistics and queue status"""
        try:
            # Get system stats
            from core.services.utils import get_system_stats, get_queue_manager
            
            stats = get_system_stats()
            queue_manager = get_queue_manager()
            
            # Format CPU usage
            cpu_text = f"💻 {stats.cpu_percent:.1f}%"
            
            # Format RAM usage
            ram_text = f"🧠 {stats.ram_used_gb:.1f}GB / {stats.ram_total_gb:.1f}GB ({stats.ram_percent:.1f}%)"
            
            # Format GPU usage
            gpu_text = f"🎮 {stats.gpu_percent:.1f}%"
            
            # Format VRAM usage
            vram_gb_used = stats.vram_used_mb / 1024
            vram_gb_total = stats.vram_total_mb / 1024
            vram_percent = (stats.vram_used_mb / stats.vram_total_mb) * 100 if stats.vram_total_mb > 0 else 0
            vram_text = f"🎯 {vram_gb_used:.1f}GB / {vram_gb_total:.1f}GB ({vram_percent:.1f}%)"
            
            # Get queue status
            queue_data, queue_summary = self._get_queue_status()
            
            # Get error statistics
            error_stats = self._get_error_statistics()
            
            return cpu_text, ram_text, gpu_text, vram_text, queue_data, queue_summary, error_stats
            
        except Exception as e:
            error_msg = f"❌ Error refreshing stats: {str(e)}"
            error_stats = {"error": "Failed to load error statistics"}
            return error_msg, error_msg, error_msg, error_msg, [], error_msg, error_stats
    
    def _get_queue_status(self):
        """Get current queue status for display with enhanced real-time info"""
        try:
            from core.services.utils import get_queue_manager
            
            queue_manager = get_queue_manager()
            tasks = queue_manager.get_all_tasks()
            
            # Format tasks for table display with enhanced status
            queue_data = []
            for task in tasks:
                # Enhanced status with progress indicators
                status_display = task.status.value
                if task.status.value == "processing":
                    status_display = f"🔄 Processing ({task.progress:.1f}%)"
                elif task.status.value == "completed":
                    status_display = "✅ Completed"
                elif task.status.value == "failed":
                    status_display = "❌ Failed"
                elif task.status.value == "pending":
                    status_display = "⏳ Pending"
                
                # Time information
                time_info = task.created_at.strftime("%H:%M:%S")
                if task.completed_at:
                    duration = (task.completed_at - task.created_at).total_seconds()
                    time_info += f" ({duration:.0f}s)"
                
                queue_data.append([
                    task.id[:8] + "...",  # Shortened ID
                    task.model_type,
                    task.prompt[:40] + "..." if len(task.prompt) > 40 else task.prompt,
                    status_display,
                    f"{task.progress:.1f}%",
                    time_info
                ])
            
            # Generate enhanced summary with ETA
            total_tasks = len(tasks)
            pending_tasks = len([t for t in tasks if t.status.value == "pending"])
            processing_tasks = len([t for t in tasks if t.status.value == "processing"])
            completed_tasks = len([t for t in tasks if t.status.value == "completed"])
            failed_tasks = len([t for t in tasks if t.status.value == "failed"])
            
            if total_tasks == 0:
                summary = "📋 No pending tasks"
            else:
                # Calculate estimated time remaining
                eta_text = ""
                if pending_tasks > 0:
                    # Rough estimate: 9 minutes per 720p task, 17 minutes per 1080p task
                    avg_time_per_task = 9  # minutes (simplified)
                    eta_minutes = pending_tasks * avg_time_per_task
                    eta_text = f" | ⏱️ ETA: ~{eta_minutes}min"
                
                summary = f"📋 Total: {total_tasks} | ⏳ Pending: {pending_tasks} | 🔄 Processing: {processing_tasks} | ✅ Completed: {completed_tasks} | ❌ Failed: {failed_tasks}{eta_text}"
            
            return queue_data, summary
            
        except Exception as e:
            return [], f"❌ Error getting queue status: {str(e)}"
    
    def _toggle_auto_refresh(self, auto_refresh_enabled: bool):
        """Toggle auto-refresh functionality"""
        self.auto_refresh_enabled = auto_refresh_enabled
        
        if auto_refresh_enabled:
            self._start_auto_refresh()
            print("🔄 Auto-refresh enabled (5 second intervals)")
        else:
            self._stop_auto_refresh()
            print("⏸️ Auto-refresh disabled")
        
        return []
    
    def _start_auto_refresh(self):
        """Start the auto-refresh background thread"""
        if self.update_thread is None or not self.update_thread.is_alive():
            self.stop_updates.clear()
            self.update_thread = threading.Thread(target=self._auto_refresh_worker, daemon=True)
            self.update_thread.start()
    
    def _stop_auto_refresh(self):
        """Stop the auto-refresh background thread"""
        if self.update_thread and self.update_thread.is_alive():
            self.stop_updates.set()
    
    def _auto_refresh_worker(self):
        """Background worker for auto-refresh functionality"""
        import time
        
        while not self.stop_updates.is_set():
            try:
                if self.auto_refresh_enabled:
                    # Update stats every 30 seconds (reduced frequency)
                    current_time = datetime.now()
                    if (current_time - self.last_stats_update).seconds >= 30:
                        self.last_stats_update = current_time
                        # Trigger stats update (this would need to be connected to the UI)
                        self._background_stats_update()
                    
                    # Update queue status every 15 seconds (reduced frequency)
                    if (current_time - self.last_queue_update).seconds >= 15:
                        self.last_queue_update = current_time
                        # Trigger queue update
                        self._background_queue_update()
                
                # Sleep for 5 seconds before next check (reduced frequency)
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in auto-refresh worker: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _background_stats_update(self):
        """Background stats update (called by auto-refresh worker)"""
        try:
            # This would trigger a UI update in a real implementation
            # For now, we'll just log the update
            logger.info("Background stats update triggered")
        except Exception as e:
            logger.error(f"Error in background stats update: {e}")
    
    def _background_queue_update(self):
        """Background queue update (called by auto-refresh worker)"""
        try:
            # This would trigger a UI update in a real implementation
            # For now, we'll just log the update
            logger.info("Background queue update triggered")
        except Exception as e:
            logger.error(f"Error in background queue update: {e}")
    
    def _setup_outputs_events(self):
        """Set up event handlers for the Outputs tab with safe component validation"""
        try:
            # Import safe event handler
            from safe_event_handler import SafeEventHandler
            safe_handler = SafeEventHandler()
            
            logger.info("Setting up output event handlers with safe validation...")
            
            # Gallery refresh and sorting
            safe_handler.setup_safe_event(
                component=self.outputs_components.get('refresh_gallery_btn'),
                event_type='click',
                handler_fn=self._refresh_video_gallery,
                inputs=[self.outputs_components.get('sort_by')],
                outputs=[self.outputs_components.get('video_gallery')],
                component_name='outputs_refresh_gallery_btn'
            )
            
            safe_handler.setup_safe_event(
                component=self.outputs_components.get('sort_by'),
                event_type='change',
                handler_fn=self._refresh_video_gallery,
                inputs=[self.outputs_components.get('sort_by')],
                outputs=[self.outputs_components.get('video_gallery')],
                component_name='outputs_sort_by'
            )
            
            # Gallery selection
            safe_handler.setup_safe_event(
                component=self.outputs_components.get('video_gallery'),
                event_type='select',
                handler_fn=self._on_video_select,
                inputs=[],
                outputs=[
                    self.outputs_components.get('selected_video'),
                    self.outputs_components.get('video_metadata')
                ],
                component_name='outputs_video_gallery'
            )
            
            # File management buttons
            safe_handler.setup_safe_event(
                component=self.outputs_components.get('delete_video_btn'),
                event_type='click',
                handler_fn=self._delete_selected_video,
                inputs=[],
                outputs=[
                    self.outputs_components.get('video_gallery'),
                    self.outputs_components.get('selected_video'),
                    self.outputs_components.get('video_metadata')
                ],
                component_name='outputs_delete_video_btn'
            )
            
            safe_handler.setup_safe_event(
                component=self.outputs_components.get('rename_video_btn'),
                event_type='click',
                handler_fn=self._rename_selected_video,
                inputs=[],
                outputs=[
                    self.outputs_components.get('video_gallery'),
                    self.outputs_components.get('video_metadata')
                ],
                component_name='outputs_rename_video_btn'
            )
            
            safe_handler.setup_safe_event(
                component=self.outputs_components.get('export_video_btn'),
                event_type='click',
                handler_fn=self._export_selected_video,
                inputs=[],
                outputs=[],
                component_name='outputs_export_video_btn'
            )
            
            # Log setup statistics
            safe_handler.log_setup_summary()
            stats = safe_handler.get_setup_statistics()
            logger.info(f"Output event handlers setup completed: {stats['successful_setups']}/{stats['total_attempts']} successful")
            
            if stats['failed_setups'] > 0:
                logger.warning(f"Some output event handlers failed to set up: {stats['failed_handlers']}")
                logger.info("Output functionality may be limited but the UI should still work")
            
        except Exception as e:
            logger.error(f"Failed to set up output event handlers: {e}")
            logger.info("Output functionality will be disabled but the UI should still work")
            # Don't re-raise the exception to prevent UI creation failure
    
    def _refresh_video_gallery(self, sort_by: str):
        """Refresh the video gallery with current sorting"""
        try:
            from core.services.utils import get_output_manager
            
            output_manager = get_output_manager()
            videos = output_manager.list_videos()
            
            # Sort videos based on selection
            if sort_by == "Date (Newest)":
                videos.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            elif sort_by == "Date (Oldest)":
                videos.sort(key=lambda x: x.get("created_at", ""))
            elif sort_by == "Name (A-Z)":
                videos.sort(key=lambda x: x.get("filename", ""))
            elif sort_by == "Name (Z-A)":
                videos.sort(key=lambda x: x.get("filename", ""), reverse=True)
            
            # Prepare gallery data - use thumbnails if available, otherwise video files
            gallery_data = []
            for video in videos:
                thumbnail_path = video.get("thumbnail_path")
                video_path = video.get("file_path")
                
                # Use thumbnail if available, otherwise the video file itself
                display_path = thumbnail_path if thumbnail_path and os.path.exists(thumbnail_path) else video_path
                
                if display_path and os.path.exists(display_path):
                    gallery_data.append((display_path, video.get("filename", "Unknown")))
            
            return gallery_data
            
        except Exception as e:
            print(f"Error refreshing gallery: {e}")
            return []
    
    def _on_video_select(self, evt: gr.SelectData):
        """Handle video selection from gallery"""
        try:
            from core.services.utils import get_output_manager
            
            output_manager = get_output_manager()
            videos = output_manager.list_videos()
            
            if evt.index < len(videos):
                selected_video = videos[evt.index]
                video_path = selected_video.get("file_path")
                
                # Load video metadata
                metadata = {
                    "filename": selected_video.get("filename", "Unknown"),
                    "file_path": video_path,
                    "created_at": selected_video.get("created_at", "Unknown"),
                    "file_size": selected_video.get("file_size_mb", 0),
                    "duration": selected_video.get("duration_seconds", 0),
                    "resolution": selected_video.get("resolution", "Unknown"),
                    "prompt": selected_video.get("generation_params", {}).get("prompt", "Not available"),
                    "model_type": selected_video.get("generation_params", {}).get("model_type", "Unknown"),
                    "generation_settings": selected_video.get("generation_params", {})
                }
                
                # Store selected video for file operations
                self.selected_video_path = video_path
                
                return video_path, metadata
            
            return None, {}
            
        except Exception as e:
            print(f"Error selecting video: {e}")
            return None, {"error": str(e)}
    
    def _delete_selected_video(self):
        """Delete the currently selected video"""
        if not hasattr(self, 'selected_video_path') or not self.selected_video_path:
            return [], None, {"error": "No video selected"}
        
        try:
            from core.services.utils import get_output_manager
            
            output_manager = get_output_manager()
            success = output_manager.delete_video(self.selected_video_path)
            
            if success:
                # Refresh gallery and clear selection
                updated_gallery = self._refresh_video_gallery("Date (Newest)")
                self.selected_video_path = None
                
                return updated_gallery, None, {"message": "Video deleted successfully"}
            else:
                return [], None, {"error": "Failed to delete video"}
                
        except Exception as e:
            return [], None, {"error": f"Error deleting video: {str(e)}"}
    
    def _rename_selected_video(self):
        """Rename the currently selected video"""
        if not hasattr(self, 'selected_video_path') or not self.selected_video_path:
            return [], {"error": "No video selected"}
        
        try:
            # In a real implementation, this would open a dialog for the new name
            # For now, we'll just add a timestamp suffix
            from datetime import datetime
            import os
            
            old_path = self.selected_video_path
            directory = os.path.dirname(old_path)
            filename = os.path.basename(old_path)
            name, ext = os.path.splitext(filename)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{name}_renamed_{timestamp}{ext}"
            new_path = os.path.join(directory, new_filename)
            
            os.rename(old_path, new_path)
            self.selected_video_path = new_path
            
            # Refresh gallery
            updated_gallery = self._refresh_video_gallery("Date (Newest)")
            
            return updated_gallery, {"message": f"Video renamed to: {new_filename}"}
            
        except Exception as e:
            return [], {"error": f"Error renaming video: {str(e)}"}
    
    def _export_selected_video(self):
        """Export the currently selected video"""
        if not hasattr(self, 'selected_video_path') or not self.selected_video_path:
            print("❌ No video selected for export")
            return
        
        try:
            # In a real implementation, this would open a file dialog or copy to a specific location
            # For now, we'll just log the export action
            print(f"📤 Exporting video: {self.selected_video_path}")
            print("✅ Export functionality would be implemented here")
            
        except Exception as e:
            print(f"❌ Error exporting video: {str(e)}")

    def launch(self, **kwargs):
        """Launch the Gradio interface with cleanup handling"""
        default_kwargs = {
            'server_name': '0.0.0.0',
            'server_port': 7860,
            'share': False,
            'debug': False,
            'show_error': True,
            'quiet': False
        }
        
        # Merge with user-provided kwargs
        launch_kwargs = {**default_kwargs, **kwargs}
        
        print("🚀 Launching Wan2.2 Video Generation UI...")
        print(f"📍 Server: http://{launch_kwargs['server_name']}:{launch_kwargs['server_port']}")
        
        try:
            return self.interface.launch(**launch_kwargs)
        finally:
            # Cleanup on exit
            self._stop_auto_refresh()
    
    def close(self):
        """Clean up resources when closing the interface"""
        self._stop_auto_refresh()
        if hasattr(self.interface, 'close'):
            self.interface.close()


def create_ui(config_path: str = "config.json") -> Wan22UI:
    """Factory function to create the UI instance"""
    return Wan22UI(config_path)


if __name__ == "__main__":
    # Create and launch the UI
    ui = create_ui()
    ui.launch()