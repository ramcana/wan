#!/usr/bin/env python3
"""
Wan2.2 UI Variant - Main Application Entry Point
Handles configuration loading, command-line arguments, and application lifecycle
"""

import argparse
import json
import logging
import os
import sys
import signal
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional

import gradio as gr

# Apply emergency model loading fixes
try:
    import apply_model_fixes
except ImportError:
    try:
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        from backend.core.services import apply_model_fixes
    except ImportError:
        print("Warning: Could not import apply_model_fixes module")

# Apply Wan2.2 pipeline fixes
try:
    from fix_wan22_pipeline import fix_wan22_pipeline_loading
    fix_wan22_pipeline_loading()
except Exception as e:
    print(f"Warning: Could not apply Wan2.2 pipeline fixes: {e}")

# Apply Wan2.2 compatibility layer
try:
    import wan22_compatibility_clean
    print("✅ Wan2.2 compatibility layer loaded")
except Exception as e:
    print(f"Warning: Could not load Wan2.2 compatibility layer: {e}")

# Import error handling system (skip if not available)
try:
    from infrastructure.hardware.error_handler import (
        GenerationErrorHandler,
        handle_validation_error,
        handle_model_loading_error,
        handle_vram_error,
        handle_generation_error
    )
    ERROR_HANDLER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import error handler: {e}")
    ERROR_HANDLER_AVAILABLE = False
    # Create dummy classes for missing imports
    class GenerationErrorHandler:
        def handle_error(self, error, context):
            return type('UserError', (), {'message': str(error), 'recovery_suggestions': []})()
        def set_system_optimizer(self, optimizer):
            pass

# Import WAN22 System Optimizer (skip if not available)
try:
    from wan22_system_optimizer import WAN22SystemOptimizer
    SYSTEM_OPTIMIZER_AVAILABLE = True
except ImportError:
    try:
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        from backend.core.services.wan22_system_optimizer import WAN22SystemOptimizer
        SYSTEM_OPTIMIZER_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Could not import system optimizer: {e}")
        SYSTEM_OPTIMIZER_AVAILABLE = False
        # Create a dummy class to prevent NameError
        class WAN22SystemOptimizer:
            def __init__(self, *args, **kwargs):
                pass
            def get_optimization_history(self):
                return []

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('wan22_ui.log')
    ]
)
logger = logging.getLogger(__name__)

class ApplicationConfig:
    """Manages application configuration loading and validation"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        self._ensure_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file with fallback to defaults"""
        if not self.config_path.exists():
            logger.warning(f"Config file {self.config_path} not found, creating default config")
            default_config = self._get_default_config()
            self._save_config(default_config)
            return default_config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            logger.info("Using default configuration")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "system": {
                "default_quantization": "bf16",
                "enable_offload": True,
                "vae_tile_size": 256,
                "max_queue_size": 10,
                "stats_refresh_interval": 5
            },
            "directories": {
                "output_directory": "outputs",
                "models_directory": "models",
                "loras_directory": "loras"
            },
            "generation": {
                "default_resolution": "1280x720",
                "default_steps": 50,
                "max_prompt_length": 500,
                "supported_resolutions": [
                    "1280x720",
                    "1280x704", 
                    "1920x1080"
                ]
            },
            "models": {
                "t2v_model": "Wan2.2-T2V-A14B",
                "i2v_model": "Wan2.2-I2V-A14B", 
                "ti2v_model": "Wan2.2-TI2V-5B"
            },
            "optimization": {
                "quantization_levels": ["fp16", "bf16", "int8"],
                "vae_tile_size_range": [128, 512],
                "max_vram_usage_gb": 12
            },
            "ui": {
                "max_file_size_mb": 10,
                "supported_image_formats": ["PNG", "JPG", "JPEG", "WebP"],
                "gallery_thumbnail_size": 256
            },
            "performance": {
                "target_720p_time_minutes": 9,
                "target_1080p_time_minutes": 17,
                "vram_warning_threshold": 0.9
            },
            "prompt_enhancement": {
                "max_prompt_length": 500,
                "enable_basic_quality": True,
                "enable_vace_detection": True,
                "enable_cinematic_enhancement": True,
                "enable_style_detection": True,
                "max_quality_keywords": 3,
                "max_cinematic_keywords": 3,
                "max_style_keywords": 2
            }
        }
    
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to JSON file"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved configuration to {self.config_path}")
        except IOError as e:
            logger.error(f"Failed to save config to {self.config_path}: {e}")
    
    def _validate_config(self):
        """Validate configuration values and fix any issues"""
        # Validate quantization levels
        valid_quant_levels = ["fp16", "bf16", "int8"]
        if self.config["system"]["default_quantization"] not in valid_quant_levels:
            logger.warning(f"Invalid quantization level, using 'bf16'")
            self.config["system"]["default_quantization"] = "bf16"
        
        # Validate VAE tile size
        tile_size = self.config["system"]["vae_tile_size"]
        if not (128 <= tile_size <= 512):
            logger.warning(f"Invalid VAE tile size {tile_size}, using 256")
            self.config["system"]["vae_tile_size"] = 256
        
        # Validate max queue size
        if self.config["system"]["max_queue_size"] < 1:
            logger.warning("Invalid max queue size, using 10")
            self.config["system"]["max_queue_size"] = 10
        
        # Validate refresh interval
        if self.config["system"]["stats_refresh_interval"] < 1:
            logger.warning("Invalid stats refresh interval, using 5")
            self.config["system"]["stats_refresh_interval"] = 5
        
        logger.info("Configuration validation completed")
    
    def _ensure_directories(self):
        """Create required directories if they don't exist"""
        directories = [
            self.config["directories"]["output_directory"],
            self.config["directories"]["models_directory"],
            self.config["directories"]["loras_directory"]
        ]
        
        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory: {dir_path}")
                except OSError as e:
                    logger.error(f"Failed to create directory {dir_path}: {e}")
                    raise
    
    def get_config(self) -> Dict[str, Any]:
        """Get the loaded configuration"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
        self._validate_config()
        self._save_config(self.config)
        logger.info("Configuration updated")


class ApplicationManager:
    """Manages the application lifecycle and cleanup"""
    
    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.ui_app: Optional[Wan22UI] = None
        self.gradio_app: Optional[gr.Blocks] = None
        self.shutdown_event = threading.Event()
        self.cleanup_handlers = []
        
        # Initialize WAN22 System Optimizer
        self.system_optimizer: Optional[WAN22SystemOptimizer] = None
        self.optimization_status = {
            'initialized': False,
            'hardware_profile': None,
            'last_health_check': None,
            'optimization_history': []
        }
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
        self.cleanup()
        sys.exit(0)
    
    def initialize(self):
        """Initialize the application components"""
        logger.info("Initializing Wan2.2 UI application...")
        
        try:
            # Initialize error handler
            self.error_handler = GenerationErrorHandler()
            
            # Initialize WAN22 System Optimizer
            logger.info("Initializing WAN22 System Optimizer...")
            self.system_optimizer = WAN22SystemOptimizer(
                config_path=str(self.config.config_path),
                log_level="INFO"
            )
            
            # Initialize the optimization system
            init_result = self.system_optimizer.initialize_system()
            self.optimization_status['initialized'] = init_result.success
            self.optimization_status['hardware_profile'] = self.system_optimizer.get_hardware_profile()
            self.optimization_status['optimization_history'] = [init_result]
            
            if init_result.success:
                logger.info("System optimizer initialized successfully")
                
                # Apply hardware optimizations
                opt_result = self.system_optimizer.apply_hardware_optimizations()
                self.optimization_status['optimization_history'].append(opt_result)
                
                if opt_result.success:
                    logger.info(f"Applied {len(opt_result.optimizations_applied)} hardware optimizations")
                
                # Validate and repair system
                validation_result = self.system_optimizer.validate_and_repair_system()
                self.optimization_status['optimization_history'].append(validation_result)
                
                if validation_result.success:
                    logger.info("System validation completed successfully")
                else:
                    logger.warning("System validation completed with warnings")
                    
            else:
                logger.warning("System optimizer initialization failed, continuing with basic functionality")
                for error in init_result.errors:
                    logger.error(f"Optimizer error: {error}")
            
            # Lazy import to avoid heavy dependencies during startup
            from ui import Wan22UI
            
            # Initialize the UI with configuration and optimizer
            self.ui_app = Wan22UI(
                config_path=str(self.config.config_path),
                system_optimizer=self.system_optimizer
            )
            self.gradio_app = self.ui_app.interface
            
            # Integrate system optimizer with error handler
            if self.system_optimizer and hasattr(self, 'error_handler'):
                self.error_handler.set_system_optimizer(self.system_optimizer)
                logger.info("System optimizer integrated with error handler")
            
            # Integrate system optimizer with performance monitor
            try:
                from performance_monitor import get_performance_monitor
                performance_monitor = get_performance_monitor()
                if self.system_optimizer:
                    performance_monitor.integrate_system_optimizer(self.system_optimizer)
                    logger.info("System optimizer integrated with performance monitor")
            except Exception as e:
                logger.warning(f"Failed to integrate system optimizer with performance monitor: {e}")
            
            # Register cleanup handlers
            self.register_cleanup_handler(self._cleanup_models)
            self.register_cleanup_handler(self._cleanup_temp_files)
            self.register_cleanup_handler(self._cleanup_optimizer)
            
            logger.info("Application initialization completed successfully")
            
        except Exception as e:
            if hasattr(self, 'error_handler'):
                user_error = self.error_handler.handle_error(e, {"config_path": str(self.config.config_path)})
                logger.error(f"Failed to initialize application: {user_error.message}")
            else:
                logger.error(f"Failed to initialize application: {e}")
            raise
    
    def register_cleanup_handler(self, handler):
        """Register a cleanup handler to be called on shutdown"""
        self.cleanup_handlers.append(handler)
    
    def cleanup(self):
        """Perform application cleanup"""
        logger.info("Starting application cleanup...")
        
        # Stop performance monitoring
        try:
            from infrastructure.hardware.performance_profiler import stop_performance_monitoring
            stop_performance_monitoring()
            logger.info("Performance monitoring stopped")
        except Exception as e:
            logger.debug(f"Error stopping performance monitoring: {e}")
        
        # Stop UI auto-refresh
        if self.ui_app:
            self.ui_app.stop_updates.set()
            if self.ui_app.update_thread and self.ui_app.update_thread.is_alive():
                self.ui_app.update_thread.join(timeout=5)
        
        # Run registered cleanup handlers
        for handler in self.cleanup_handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"Error in cleanup handler: {e}")
        
        logger.info("Application cleanup completed")
    
    def _cleanup_models(self):
        """Cleanup loaded models and free GPU memory"""
        try:
            # Lazy import to avoid heavy dependencies during cleanup
            project_root = Path(__file__).parent.parent
            sys.path.insert(0, str(project_root))
            from backend.core.services.model_manager import get_model_manager
            
            model_manager = get_model_manager()
            # Unload all models
            for model_id in list(model_manager.loaded_models.keys()):
                model_manager.unload_model(model_id)
            logger.info("Cleaned up loaded models")
        except Exception as e:
            logger.error(f"Error cleaning up models: {e}")
    
    def _cleanup_temp_files(self):
        """Cleanup temporary files"""
        try:
            # Clean up any temporary files that might have been created
            temp_patterns = ["*.tmp", "*.temp", ".gradio_temp*"]
            for pattern in temp_patterns:
                for temp_file in Path(".").glob(pattern):
                    try:
                        temp_file.unlink()
                        logger.debug(f"Removed temp file: {temp_file}")
                    except OSError:
                        pass
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
    
    def _cleanup_optimizer(self):
        """Cleanup system optimizer"""
        try:
            if self.system_optimizer:
                # Save final hardware profile
                self.system_optimizer.save_profile_to_file("hardware_profile_final.json")
                logger.info("System optimizer cleanup completed")
        except Exception as e:
            logger.error(f"Error cleaning up system optimizer: {e}")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status for UI display"""
        if not self.system_optimizer:
            return {
                'initialized': False,
                'hardware_profile': None,
                'health_metrics': None,
                'optimization_history': []
            }
        
        # Update health metrics
        health_metrics = self.system_optimizer.monitor_system_health()
        self.optimization_status['last_health_check'] = health_metrics
        
        return {
            'initialized': self.optimization_status['initialized'],
            'hardware_profile': self.optimization_status['hardware_profile'],
            'health_metrics': health_metrics,
            'optimization_history': self.system_optimizer.get_optimization_history()
        }
    
    def get_system_optimizer(self) -> Optional[WAN22SystemOptimizer]:
        """Get the system optimizer instance"""
        return self.system_optimizer
    
    def launch(self, **launch_kwargs):
        """Launch the Gradio application"""
        if not self.gradio_app:
            raise RuntimeError("Application not initialized. Call initialize() first.")
        
        logger.info("Launching Wan2.2 UI application...")
        
        try:
            # Launch the Gradio app
            self.gradio_app.launch(**launch_kwargs)
            
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
            self.cleanup()
        except Exception as e:
            if hasattr(self, 'error_handler'):
                user_error = self.error_handler.handle_error(e, {"launch_kwargs": launch_kwargs})
                logger.error(f"Error launching application: {user_error.message}")
            else:
                logger.error(f"Error launching application: {e}")
            self.cleanup()
            raise


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Wan2.2 Video Generation UI - Advanced AI video generation interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Launch with default settings
  python main.py --port 7860 --share     # Launch on port 7860 with sharing enabled
  python main.py --config custom.json    # Use custom configuration file
  python main.py --debug                 # Enable debug logging
        """
    )
    
    # Configuration options
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.json",
        help="Path to configuration file (default: config.json)"
    )
    
    # Gradio launch options
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    
    parser.add_argument(
        "--auth",
        type=str,
        nargs=2,
        metavar=("USERNAME", "PASSWORD"),
        help="Enable authentication with username and password"
    )
    
    parser.add_argument(
        "--ssl-keyfile",
        type=str,
        help="Path to SSL key file for HTTPS"
    )
    
    parser.add_argument(
        "--ssl-certfile",
        type=str,
        help="Path to SSL certificate file for HTTPS"
    )
    
    # Application options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser"
    )
    
    parser.add_argument(
        "--queue-max-size",
        type=int,
        help="Override maximum queue size from config"
    )
    
    parser.add_argument(
        "--models-dir",
        type=str,
        help="Override models directory from config"
    )
    
    parser.add_argument(
        "--outputs-dir",
        type=str,
        help="Override outputs directory from config"
    )
    
    return parser.parse_args()


def setup_logging(debug: bool = False):
    """Setup application logging"""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Update root logger level
    logging.getLogger().setLevel(log_level)
    
    # Update specific loggers
    logging.getLogger("wan22_ui").setLevel(log_level)
    logging.getLogger("utils").setLevel(log_level)
    
    if debug:
        logger.info("Debug logging enabled")


def main():
    """Main application entry point"""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.debug)
        
        logger.info("Starting Wan2.2 UI Variant application")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Load configuration
        config = ApplicationConfig(args.config)
        
        # Apply command-line overrides to config
        config_updates = {}
        if args.queue_max_size:
            config_updates["system"] = {"max_queue_size": args.queue_max_size}
        if args.models_dir:
            config_updates["directories"] = {"models_directory": args.models_dir}
        if args.outputs_dir:
            if "directories" not in config_updates:
                config_updates["directories"] = {}
            config_updates["directories"]["outputs_directory"] = args.outputs_dir
        
        if config_updates:
            config.update_config(config_updates)
            logger.info("Applied command-line configuration overrides")
        
        # Initialize application manager
        app_manager = ApplicationManager(config)
        app_manager.initialize()
        
        # Check for environment variable override (set by port_manager.py)
        port = int(os.environ.get('GRADIO_SERVER_PORT', args.port))
        if port != args.port:
            logger.info(f"Using port {port} from environment variable (original: {args.port})")
        
        # Prepare Gradio launch arguments
        launch_kwargs = {
            "server_name": args.host,
            "server_port": port,
            "share": args.share,
            "inbrowser": not args.no_browser,
            "show_error": True,
            "quiet": not args.debug
        }
        
        # Add authentication if provided
        if args.auth:
            launch_kwargs["auth"] = tuple(args.auth)
            logger.info("Authentication enabled")
        
        # Add SSL configuration if provided
        if args.ssl_keyfile and args.ssl_certfile:
            launch_kwargs["ssl_keyfile"] = args.ssl_keyfile
            launch_kwargs["ssl_certfile"] = args.ssl_certfile
            logger.info("SSL/HTTPS enabled")
        
        # Launch the application
        logger.info(f"Launching application on {args.host}:{args.port}")
        if args.share:
            logger.info("Public sharing enabled - link will be displayed after launch")
        
        app_manager.launch(**launch_kwargs)
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        # Handle any application-level errors
        try:
            error_handler = GenerationErrorHandler()
            user_error = error_handler.handle_error(e, {"context": "main_application", "args": vars(args) if 'args' in locals() else None})
            logger.error(f"Application error: {user_error.message}")
            if user_error.recovery_suggestions:
                logger.error(f"Recovery suggestions: {', '.join(user_error.recovery_suggestions)}")
        except Exception:
            logger.error(f"Fatal error: {e}", exc_info=True)
        
        sys.exit(1)


if __name__ == "__main__":
    main()
