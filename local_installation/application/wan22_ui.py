"""
WAN2.2 Local UI Application
A comprehensive user interface for the WAN2.2 video generation system.
"""

import sys
import os
import json
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    from PIL import Image, ImageTk
    import cv2
    import numpy as np
except ImportError as e:
    print(f"Required UI dependencies not installed: {e}")
    print("Please run the installation again to install UI dependencies.")
    sys.exit(1)

# Add the scripts directory to path for accessing installation components
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

try:
    from config_manager import ConfigurationManager
    from model_configuration import ModelConfigurationManager
    from logging_system import setup_installation_logging
except ImportError:
    # Fallback if installation components not available
    ConfigurationManager = None
    ModelConfigurationManager = None
    setup_installation_logging = None


class WAN22UI:
    """Main UI application for WAN2.2 video generation."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("WAN2.2 - Video Generation System")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Application state
        self.config = {}
        self.models_loaded = False
        self.generation_in_progress = False
        self.current_project = None
        
        # Initialize components
        self.setup_logging()
        self.load_configuration()
        self.setup_ui()
        self.check_system_status()
        
    def setup_logging(self):
        """Setup logging for the UI application."""
        try:
            if setup_installation_logging:
                self.logging_system = setup_installation_logging(
                    installation_path=str(Path(__file__).parent.parent),
                    log_level="INFO",
                    enable_console=False
                )
                self.logger = self.logging_system.get_logger(__name__)
            else:
                import logging
                logging.basicConfig(level=logging.INFO)
                self.logger = logging.getLogger(__name__)
        except Exception as e:
            import logging
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            self.logger.warning(f"Could not setup advanced logging: {e}")
    
    def load_configuration(self):
        """Load system configuration."""
        try:
            config_path = Path(__file__).parent.parent / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                self.logger.info("Configuration loaded successfully")
            else:
                self.config = self.get_default_config()
                self.logger.warning("No configuration found, using defaults")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.config = self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "system": {
                "threads": 4,
                "memory_limit_gb": 8,
                "default_quantization": "fp16",
                "enable_offload": True,
                "vae_tile_size": 256,
                "max_queue_size": 10,
                "worker_threads": 4
            },
            "models": {
                "cache_dir": "models",
                "download_timeout": 300,
                "verify_checksums": True
            },
            "ui": {
                "theme": "default",
                "auto_save": True,
                "preview_quality": "medium"
            }
        }
    
    def setup_ui(self):
        """Setup the main user interface."""
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create main menu
        self.create_menu()
        
        # Create main layout
        self.create_main_layout()
        
        # Create status bar
        self.create_status_bar()
        
        # Bind events
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_menu(self):
        """Create the main menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Project", command=self.new_project)
        file_menu.add_command(label="Open Project", command=self.open_project)
        file_menu.add_command(label="Save Project", command=self.save_project)
        file_menu.add_separator()
        file_menu.add_command(label="Import Video", command=self.import_video)
        file_menu.add_command(label="Import Image", command=self.import_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Models menu
        models_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Models", menu=models_menu)
        models_menu.add_command(label="Check Models", command=self.check_models)
        models_menu.add_command(label="Download Models", command=self.download_models)
        models_menu.add_command(label="Model Settings", command=self.model_settings)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="System Information", command=self.show_system_info)
        tools_menu.add_command(label="Performance Monitor", command=self.show_performance_monitor)
        tools_menu.add_command(label="Settings", command=self.show_settings)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        help_menu.add_command(label="Troubleshooting", command=self.show_troubleshooting)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_main_layout(self):
        """Create the main application layout."""
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Controls
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Right panel - Preview and Output
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        
        # Setup left panel
        self.setup_control_panel(left_frame)
        
        # Setup right panel
        self.setup_preview_panel(right_frame)
    
    def setup_control_panel(self, parent):
        """Setup the control panel on the left side."""
        # Create notebook for different modes
        self.control_notebook = ttk.Notebook(parent)
        self.control_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Text-to-Video tab
        self.t2v_frame = ttk.Frame(self.control_notebook)
        self.control_notebook.add(self.t2v_frame, text="Text-to-Video")
        self.setup_t2v_controls(self.t2v_frame)
        
        # Image-to-Video tab
        self.i2v_frame = ttk.Frame(self.control_notebook)
        self.control_notebook.add(self.i2v_frame, text="Image-to-Video")
        self.setup_i2v_controls(self.i2v_frame)
        
        # Text+Image-to-Video tab
        self.ti2v_frame = ttk.Frame(self.control_notebook)
        self.control_notebook.add(self.ti2v_frame, text="Text+Image-to-Video")
        self.setup_ti2v_controls(self.ti2v_frame)
    
    def setup_t2v_controls(self, parent):
        """Setup Text-to-Video controls."""
        # Prompt input
        ttk.Label(parent, text="Text Prompt:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.t2v_prompt = scrolledtext.ScrolledText(parent, height=4, wrap=tk.WORD)
        self.t2v_prompt.pack(fill=tk.X, padx=5, pady=5)
        
        # Negative prompt
        ttk.Label(parent, text="Negative Prompt (Optional):").pack(anchor=tk.W, padx=5)
        self.t2v_negative = scrolledtext.ScrolledText(parent, height=2, wrap=tk.WORD)
        self.t2v_negative.pack(fill=tk.X, padx=5, pady=5)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(parent, text="Generation Parameters")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Duration
        ttk.Label(params_frame, text="Duration (seconds):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.t2v_duration = ttk.Scale(params_frame, from_=1, to=10, orient=tk.HORIZONTAL)
        self.t2v_duration.set(4)
        self.t2v_duration.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        self.t2v_duration_label = ttk.Label(params_frame, text="4s")
        self.t2v_duration_label.grid(row=0, column=2, padx=5, pady=2)
        self.t2v_duration.configure(command=lambda v: self.t2v_duration_label.config(text=f"{int(float(v))}s"))
        
        # Resolution
        ttk.Label(params_frame, text="Resolution:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.t2v_resolution = ttk.Combobox(params_frame, values=["512x512", "768x768", "1024x1024", "1280x720", "1920x1080"])
        self.t2v_resolution.set("1280x720")
        self.t2v_resolution.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=2)
        
        # FPS
        ttk.Label(params_frame, text="Frame Rate:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.t2v_fps = ttk.Combobox(params_frame, values=["24", "30", "60"])
        self.t2v_fps.set("30")
        self.t2v_fps.grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=2)
        
        # Steps
        ttk.Label(params_frame, text="Inference Steps:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.t2v_steps = ttk.Scale(params_frame, from_=10, to=100, orient=tk.HORIZONTAL)
        self.t2v_steps.set(50)
        self.t2v_steps.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=2)
        self.t2v_steps_label = ttk.Label(params_frame, text="50")
        self.t2v_steps_label.grid(row=3, column=2, padx=5, pady=2)
        self.t2v_steps.configure(command=lambda v: self.t2v_steps_label.config(text=str(int(float(v)))))
        
        # Guidance Scale
        ttk.Label(params_frame, text="Guidance Scale:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.t2v_guidance = ttk.Scale(params_frame, from_=1, to=20, orient=tk.HORIZONTAL)
        self.t2v_guidance.set(7.5)
        self.t2v_guidance.grid(row=4, column=1, sticky=tk.EW, padx=5, pady=2)
        self.t2v_guidance_label = ttk.Label(params_frame, text="7.5")
        self.t2v_guidance_label.grid(row=4, column=2, padx=5, pady=2)
        self.t2v_guidance.configure(command=lambda v: self.t2v_guidance_label.config(text=f"{float(v):.1f}"))
        
        params_frame.columnconfigure(1, weight=1)
        
        # Generate button
        self.t2v_generate_btn = ttk.Button(parent, text="Generate Video", command=self.generate_t2v)
        self.t2v_generate_btn.pack(fill=tk.X, padx=5, pady=10)
    
    def setup_i2v_controls(self, parent):
        """Setup Image-to-Video controls."""
        # Image input
        image_frame = ttk.LabelFrame(parent, text="Input Image")
        image_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.i2v_image_path = tk.StringVar()
        ttk.Entry(image_frame, textvariable=self.i2v_image_path, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        ttk.Button(image_frame, text="Browse", command=self.browse_i2v_image).pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Prompt input
        ttk.Label(parent, text="Motion Description:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.i2v_prompt = scrolledtext.ScrolledText(parent, height=3, wrap=tk.WORD)
        self.i2v_prompt.pack(fill=tk.X, padx=5, pady=5)
        
        # Parameters (similar to T2V but fewer options)
        params_frame = ttk.LabelFrame(parent, text="Generation Parameters")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Duration
        ttk.Label(params_frame, text="Duration (seconds):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.i2v_duration = ttk.Scale(params_frame, from_=1, to=8, orient=tk.HORIZONTAL)
        self.i2v_duration.set(3)
        self.i2v_duration.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        self.i2v_duration_label = ttk.Label(params_frame, text="3s")
        self.i2v_duration_label.grid(row=0, column=2, padx=5, pady=2)
        self.i2v_duration.configure(command=lambda v: self.i2v_duration_label.config(text=f"{int(float(v))}s"))
        
        # Motion strength
        ttk.Label(params_frame, text="Motion Strength:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.i2v_motion = ttk.Scale(params_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL)
        self.i2v_motion.set(1.0)
        self.i2v_motion.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        self.i2v_motion_label = ttk.Label(params_frame, text="1.0")
        self.i2v_motion_label.grid(row=1, column=2, padx=5, pady=2)
        self.i2v_motion.configure(command=lambda v: self.i2v_motion_label.config(text=f"{float(v):.1f}"))
        
        params_frame.columnconfigure(1, weight=1)
        
        # Generate button
        self.i2v_generate_btn = ttk.Button(parent, text="Generate Video", command=self.generate_i2v)
        self.i2v_generate_btn.pack(fill=tk.X, padx=5, pady=10)
    
    def setup_ti2v_controls(self, parent):
        """Setup Text+Image-to-Video controls."""
        # Image input
        image_frame = ttk.LabelFrame(parent, text="Input Image")
        image_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.ti2v_image_path = tk.StringVar()
        ttk.Entry(image_frame, textvariable=self.ti2v_image_path, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        ttk.Button(image_frame, text="Browse", command=self.browse_ti2v_image).pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Text prompt
        ttk.Label(parent, text="Text Prompt:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.ti2v_prompt = scrolledtext.ScrolledText(parent, height=3, wrap=tk.WORD)
        self.ti2v_prompt.pack(fill=tk.X, padx=5, pady=5)
        
        # Parameters
        params_frame = ttk.LabelFrame(parent, text="Generation Parameters")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Duration
        ttk.Label(params_frame, text="Duration (seconds):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.ti2v_duration = ttk.Scale(params_frame, from_=1, to=6, orient=tk.HORIZONTAL)
        self.ti2v_duration.set(3)
        self.ti2v_duration.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        self.ti2v_duration_label = ttk.Label(params_frame, text="3s")
        self.ti2v_duration_label.grid(row=0, column=2, padx=5, pady=2)
        self.ti2v_duration.configure(command=lambda v: self.ti2v_duration_label.config(text=f"{int(float(v))}s"))
        
        # Text influence
        ttk.Label(params_frame, text="Text Influence:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.ti2v_text_influence = ttk.Scale(params_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL)
        self.ti2v_text_influence.set(1.0)
        self.ti2v_text_influence.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        self.ti2v_text_label = ttk.Label(params_frame, text="1.0")
        self.ti2v_text_label.grid(row=1, column=2, padx=5, pady=2)
        self.ti2v_text_influence.configure(command=lambda v: self.ti2v_text_label.config(text=f"{float(v):.1f}"))
        
        params_frame.columnconfigure(1, weight=1)
        
        # Generate button
        self.ti2v_generate_btn = ttk.Button(parent, text="Generate Video", command=self.generate_ti2v)
        self.ti2v_generate_btn.pack(fill=tk.X, padx=5, pady=10)
    
    def setup_preview_panel(self, parent):
        """Setup the preview panel on the right side."""
        # Create notebook for preview and output
        self.preview_notebook = ttk.Notebook(parent)
        self.preview_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Preview tab
        self.preview_frame = ttk.Frame(self.preview_notebook)
        self.preview_notebook.add(self.preview_frame, text="Preview")
        self.setup_preview_tab(self.preview_frame)
        
        # Output tab
        self.output_frame = ttk.Frame(self.preview_notebook)
        self.preview_notebook.add(self.output_frame, text="Output")
        self.setup_output_tab(self.output_frame)
        
        # Queue tab
        self.queue_frame = ttk.Frame(self.preview_notebook)
        self.preview_notebook.add(self.queue_frame, text="Queue")
        self.setup_queue_tab(self.queue_frame)
    
    def setup_preview_tab(self, parent):
        """Setup the preview tab."""
        # Preview canvas
        self.preview_canvas = tk.Canvas(parent, bg='black', width=640, height=360)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Preview controls
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.preview_play_btn = ttk.Button(controls_frame, text="▶", command=self.toggle_preview_playback)
        self.preview_play_btn.pack(side=tk.LEFT, padx=2)
        
        self.preview_progress = ttk.Scale(controls_frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.preview_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.preview_time_label = ttk.Label(controls_frame, text="00:00 / 00:00")
        self.preview_time_label.pack(side=tk.RIGHT, padx=5)
    
    def setup_output_tab(self, parent):
        """Setup the output tab."""
        # Output list
        self.output_tree = ttk.Treeview(parent, columns=("Type", "Status", "Duration", "Created"), show="tree headings")
        self.output_tree.heading("#0", text="Name")
        self.output_tree.heading("Type", text="Type")
        self.output_tree.heading("Status", text="Status")
        self.output_tree.heading("Duration", text="Duration")
        self.output_tree.heading("Created", text="Created")
        
        # Configure column widths
        self.output_tree.column("#0", width=200)
        self.output_tree.column("Type", width=100)
        self.output_tree.column("Status", width=100)
        self.output_tree.column("Duration", width=80)
        self.output_tree.column("Created", width=120)
        
        self.output_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Output controls
        output_controls = ttk.Frame(parent)
        output_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(output_controls, text="Open Folder", command=self.open_output_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(output_controls, text="Preview", command=self.preview_selected_output).pack(side=tk.LEFT, padx=2)
        ttk.Button(output_controls, text="Delete", command=self.delete_selected_output).pack(side=tk.LEFT, padx=2)
        ttk.Button(output_controls, text="Refresh", command=self.refresh_output_list).pack(side=tk.RIGHT, padx=2)
    
    def setup_queue_tab(self, parent):
        """Setup the generation queue tab."""
        # Queue list
        self.queue_tree = ttk.Treeview(parent, columns=("Type", "Status", "Progress"), show="tree headings")
        self.queue_tree.heading("#0", text="Task")
        self.queue_tree.heading("Type", text="Type")
        self.queue_tree.heading("Status", text="Status")
        self.queue_tree.heading("Progress", text="Progress")
        
        self.queue_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Queue controls
        queue_controls = ttk.Frame(parent)
        queue_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(queue_controls, text="Pause", command=self.pause_queue).pack(side=tk.LEFT, padx=2)
        ttk.Button(queue_controls, text="Resume", command=self.resume_queue).pack(side=tk.LEFT, padx=2)
        ttk.Button(queue_controls, text="Clear", command=self.clear_queue).pack(side=tk.LEFT, padx=2)
        
        # Progress info
        self.queue_progress_label = ttk.Label(queue_controls, text="Queue: 0 pending")
        self.queue_progress_label.pack(side=tk.RIGHT, padx=5)
    
    def create_status_bar(self):
        """Create the status bar at the bottom."""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Status label
        self.status_label = ttk.Label(self.status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(self.status_frame, mode='indeterminate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5, pady=2, fill=tk.X, expand=True)
        
        # System status indicators
        self.gpu_status = ttk.Label(self.status_frame, text="GPU: Unknown")
        self.gpu_status.pack(side=tk.RIGHT, padx=5, pady=2)
        
        self.models_status = ttk.Label(self.status_frame, text="Models: Checking...")
        self.models_status.pack(side=tk.RIGHT, padx=5, pady=2)
    
    def check_system_status(self):
        """Check and update system status."""
        def check_status():
            try:
                # Check models
                models_dir = Path(__file__).parent.parent / "models"
                if models_dir.exists():
                    model_files = list(models_dir.glob("**/*.bin")) + list(models_dir.glob("**/*.safetensors"))
                    if len(model_files) > 0:
                        self.models_status.config(text=f"Models: {len(model_files)} loaded")
                        self.models_loaded = True
                    else:
                        self.models_status.config(text="Models: Not found")
                        self.models_loaded = False
                else:
                    self.models_status.config(text="Models: Not found")
                    self.models_loaded = False
                
                # Check GPU (simplified)
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0)
                        self.gpu_status.config(text=f"GPU: {gpu_name[:20]}...")
                    else:
                        self.gpu_status.config(text="GPU: CPU only")
                except ImportError:
                    self.gpu_status.config(text="GPU: Unknown")
                
                # Update status
                if self.models_loaded:
                    self.status_label.config(text="Ready for generation")
                else:
                    self.status_label.config(text="Models not loaded - check Models menu")
                    
            except Exception as e:
                self.logger.error(f"Error checking system status: {e}")
                self.status_label.config(text="Error checking system status")
        
        # Run in background thread
        threading.Thread(target=check_status, daemon=True).start()
    
    # Event handlers
    def new_project(self):
        """Create a new project."""
        messagebox.showinfo("New Project", "New project functionality coming soon!")
    
    def open_project(self):
        """Open an existing project."""
        messagebox.showinfo("Open Project", "Open project functionality coming soon!")
    
    def save_project(self):
        """Save the current project."""
        messagebox.showinfo("Save Project", "Save project functionality coming soon!")
    
    def import_video(self):
        """Import a video file."""
        file_path = filedialog.askopenfilename(
            title="Import Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            messagebox.showinfo("Import Video", f"Video import functionality coming soon!\nSelected: {file_path}")
    
    def import_image(self):
        """Import an image file."""
        file_path = filedialog.askopenfilename(
            title="Import Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if file_path:
            messagebox.showinfo("Import Image", f"Image import functionality coming soon!\nSelected: {file_path}")
    
    def browse_i2v_image(self):
        """Browse for I2V input image."""
        file_path = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.i2v_image_path.set(file_path)
    
    def browse_ti2v_image(self):
        """Browse for TI2V input image."""
        file_path = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.ti2v_image_path.set(file_path)
    
    def generate_t2v(self):
        """Generate Text-to-Video."""
        if not self.models_loaded:
            messagebox.showerror("Error", "Models not loaded. Please check the Models menu.")
            return
        
        prompt = self.t2v_prompt.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showerror("Error", "Please enter a text prompt.")
            return
        
        # Get parameters
        params = {
            "prompt": prompt,
            "negative_prompt": self.t2v_negative.get("1.0", tk.END).strip(),
            "duration": int(self.t2v_duration.get()),
            "resolution": self.t2v_resolution.get(),
            "fps": int(self.t2v_fps.get()),
            "steps": int(self.t2v_steps.get()),
            "guidance_scale": float(self.t2v_guidance.get())
        }
        
        self.start_generation("Text-to-Video", params)
    
    def generate_i2v(self):
        """Generate Image-to-Video."""
        if not self.models_loaded:
            messagebox.showerror("Error", "Models not loaded. Please check the Models menu.")
            return
        
        if not self.i2v_image_path.get():
            messagebox.showerror("Error", "Please select an input image.")
            return
        
        params = {
            "image_path": self.i2v_image_path.get(),
            "prompt": self.i2v_prompt.get("1.0", tk.END).strip(),
            "duration": int(self.i2v_duration.get()),
            "motion_strength": float(self.i2v_motion.get())
        }
        
        self.start_generation("Image-to-Video", params)
    
    def generate_ti2v(self):
        """Generate Text+Image-to-Video."""
        if not self.models_loaded:
            messagebox.showerror("Error", "Models not loaded. Please check the Models menu.")
            return
        
        if not self.ti2v_image_path.get():
            messagebox.showerror("Error", "Please select an input image.")
            return
        
        prompt = self.ti2v_prompt.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showerror("Error", "Please enter a text prompt.")
            return
        
        params = {
            "image_path": self.ti2v_image_path.get(),
            "prompt": prompt,
            "duration": int(self.ti2v_duration.get()),
            "text_influence": float(self.ti2v_text_influence.get())
        }
        
        self.start_generation("Text+Image-to-Video", params)
    
    def start_generation(self, generation_type: str, params: Dict[str, Any]):
        """Start a video generation task."""
        if self.generation_in_progress:
            messagebox.showwarning("Warning", "Generation already in progress. Please wait.")
            return
        
        # Add to queue
        task_id = f"{generation_type}_{int(time.time())}"
        self.queue_tree.insert("", tk.END, text=task_id, values=(generation_type, "Queued", "0%"))
        
        # Start generation in background
        def generate():
            try:
                self.generation_in_progress = True
                self.status_label.config(text=f"Generating {generation_type}...")
                self.progress_bar.start()
                
                # Update queue status
                for item in self.queue_tree.get_children():
                    if self.queue_tree.item(item, "text") == task_id:
                        self.queue_tree.set(item, "Status", "Processing")
                        break
                
                # Simulate generation (replace with actual generation code)
                for i in range(101):
                    time.sleep(0.1)  # Simulate work
                    progress = f"{i}%"
                    
                    # Update queue progress
                    for item in self.queue_tree.get_children():
                        if self.queue_tree.item(item, "text") == task_id:
                            self.queue_tree.set(item, "Progress", progress)
                            break
                
                # Complete generation
                self.progress_bar.stop()
                self.generation_in_progress = False
                self.status_label.config(text="Generation completed")
                
                # Update queue status
                for item in self.queue_tree.get_children():
                    if self.queue_tree.item(item, "text") == task_id:
                        self.queue_tree.set(item, "Status", "Completed")
                        self.queue_tree.set(item, "Progress", "100%")
                        break
                
                # Add to output list
                self.output_tree.insert("", tk.END, text=task_id, 
                                      values=(generation_type, "Completed", f"{params.get('duration', 'N/A')}s", 
                                            datetime.now().strftime("%H:%M:%S")))
                
                messagebox.showinfo("Success", f"{generation_type} generation completed!")
                
            except Exception as e:
                self.logger.error(f"Generation error: {e}")
                self.progress_bar.stop()
                self.generation_in_progress = False
                self.status_label.config(text="Generation failed")
                messagebox.showerror("Error", f"Generation failed: {str(e)}")
        
        threading.Thread(target=generate, daemon=True).start()
    
    def check_models(self):
        """Check model status."""
        def check():
            models_dir = Path(__file__).parent.parent / "models"
            if not models_dir.exists():
                messagebox.showwarning("Models", "Models directory not found.")
                return
            
            model_info = []
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    model_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("*.safetensors"))
                    if model_files:
                        total_size = sum(f.stat().st_size for f in model_files) / (1024**3)  # GB
                        model_info.append(f"✓ {model_dir.name} ({total_size:.1f} GB)")
                    else:
                        model_info.append(f"✗ {model_dir.name} (incomplete)")
            
            if model_info:
                messagebox.showinfo("Models Status", "\n".join(model_info))
            else:
                messagebox.showwarning("Models", "No models found. Use 'Download Models' to get them.")
        
        threading.Thread(target=check, daemon=True).start()
    
    def download_models(self):
        """Download missing models."""
        messagebox.showinfo("Download Models", 
                          "Model download functionality will be implemented.\n"
                          "For now, please use the installation script to download models.")
    
    def model_settings(self):
        """Show model settings dialog."""
        messagebox.showinfo("Model Settings", "Model settings dialog coming soon!")
    
    def show_system_info(self):
        """Show system information dialog."""
        info_window = tk.Toplevel(self.root)
        info_window.title("System Information")
        info_window.geometry("600x400")
        
        info_text = scrolledtext.ScrolledText(info_window, wrap=tk.WORD)
        info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Get system info
        info_content = f"""WAN2.2 System Information
{'='*50}

Installation Path: {Path(__file__).parent.parent}
Configuration: {json.dumps(self.config, indent=2)}

Hardware Status:
- Models Loaded: {self.models_loaded}
- Generation in Progress: {self.generation_in_progress}

System Configuration:
{json.dumps(self.config.get('system', {}), indent=2)}
"""
        
        info_text.insert(tk.END, info_content)
        info_text.config(state=tk.DISABLED)
    
    def show_performance_monitor(self):
        """Show performance monitoring dialog."""
        messagebox.showinfo("Performance Monitor", "Performance monitoring coming soon!")
    
    def show_settings(self):
        """Show settings dialog."""
        messagebox.showinfo("Settings", "Settings dialog coming soon!")
    
    def show_user_guide(self):
        """Show user guide."""
        messagebox.showinfo("User Guide", "User guide will open the documentation.")
    
    def show_troubleshooting(self):
        """Show troubleshooting guide."""
        messagebox.showinfo("Troubleshooting", "Troubleshooting guide coming soon!")
    
    def show_about(self):
        """Show about dialog."""
        messagebox.showinfo("About WAN2.2", 
                          "WAN2.2 Video Generation System\n"
                          "Version 1.0.0\n\n"
                          "A local installation system for WAN2.2 video generation models.\n"
                          "Supports Text-to-Video, Image-to-Video, and Text+Image-to-Video generation.")
    
    def toggle_preview_playback(self):
        """Toggle preview playback."""
        if self.preview_play_btn.cget("text") == "▶":
            self.preview_play_btn.config(text="⏸")
        else:
            self.preview_play_btn.config(text="▶")
    
    def open_output_folder(self):
        """Open the output folder."""
        output_dir = Path(__file__).parent.parent / "outputs"
        if output_dir.exists():
            os.startfile(str(output_dir))
        else:
            messagebox.showwarning("Output Folder", "Output folder not found.")
    
    def preview_selected_output(self):
        """Preview selected output."""
        selection = self.output_tree.selection()
        if selection:
            messagebox.showinfo("Preview", "Preview functionality coming soon!")
        else:
            messagebox.showwarning("Preview", "Please select an output to preview.")
    
    def delete_selected_output(self):
        """Delete selected output."""
        selection = self.output_tree.selection()
        if selection:
            if messagebox.askyesno("Delete", "Are you sure you want to delete the selected output?"):
                for item in selection:
                    self.output_tree.delete(item)
        else:
            messagebox.showwarning("Delete", "Please select an output to delete.")
    
    def refresh_output_list(self):
        """Refresh the output list."""
        # Clear current list
        for item in self.output_tree.get_children():
            self.output_tree.delete(item)
        
        # Scan output directory
        output_dir = Path(__file__).parent.parent / "outputs"
        if output_dir.exists():
            for video_file in output_dir.glob("*.mp4"):
                self.output_tree.insert("", tk.END, text=video_file.stem,
                                      values=("Video", "Completed", "N/A", 
                                            datetime.fromtimestamp(video_file.stat().st_mtime).strftime("%H:%M:%S")))
    
    def pause_queue(self):
        """Pause the generation queue."""
        messagebox.showinfo("Queue", "Queue pause functionality coming soon!")
    
    def resume_queue(self):
        """Resume the generation queue."""
        messagebox.showinfo("Queue", "Queue resume functionality coming soon!")
    
    def clear_queue(self):
        """Clear the generation queue."""
        if messagebox.askyesno("Clear Queue", "Are you sure you want to clear the queue?"):
            for item in self.queue_tree.get_children():
                self.queue_tree.delete(item)
    
    def on_closing(self):
        """Handle application closing."""
        if self.generation_in_progress:
            if not messagebox.askyesno("Exit", "Generation is in progress. Are you sure you want to exit?"):
                return
        
        self.logger.info("WAN2.2 UI application closing")
        self.root.destroy()
    
    def run(self):
        """Run the application."""
        self.logger.info("Starting WAN2.2 UI application")
        self.root.mainloop()


def main():
    """Main entry point for the UI application."""
    try:
        app = WAN22UI()
        app.run()
    except Exception as e:
        print(f"Error starting WAN2.2 UI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()