"""
VRAM Monitor UI Component for WAN22 System

Provides real-time VRAM usage display and GPU load balancing interface.
"""

import gradio as gr
import json
import time
import threading
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from vram_manager import VRAMManager, GPUInfo, VRAMUsage


class VRAMMonitorUI:
    """
    UI component for VRAM monitoring and GPU management
    
    Provides:
    - Real-time VRAM usage display
    - GPU selection interface
    - Load balancing controls
    - Memory optimization triggers
    """
    
    def __init__(self, vram_manager: Optional[VRAMManager] = None):
        self.logger = logging.getLogger(__name__)
        self.vram_manager = vram_manager or VRAMManager()
        self.update_interval = 2.0  # seconds
        self.ui_update_thread: Optional[threading.Thread] = None
        self.ui_active = False
        
        # UI state
        self.current_usage_data: Dict[int, VRAMUsage] = {}
        self.gpu_selection_state: Dict[str, Any] = {}
        
    def create_ui_components(self) -> Dict[str, Any]:
        """Create Gradio UI components for VRAM monitoring"""
        
        with gr.Row():
            with gr.Column(scale=2):
                # GPU Detection and Selection
                gpu_info_display = gr.HTML(
                    value=self._get_gpu_info_html(),
                    label="Detected GPUs"
                )
                
                with gr.Row():
                    refresh_gpus_btn = gr.Button("🔄 Refresh GPU Detection", variant="secondary")
                    detect_method_info = gr.Textbox(
                        value="Detection method will be shown here",
                        label="Detection Method",
                        interactive=False
                    )
                
                # GPU Selection
                with gr.Group():
                    gr.Markdown("### GPU Selection")
                    
                    available_gpus = self._get_available_gpu_choices()
                    preferred_gpu_dropdown = gr.Dropdown(
                        choices=available_gpus,
                        value=self._get_current_preferred_gpu(),
                        label="Preferred GPU",
                        info="Select primary GPU for processing"
                    )
                    
                    multi_gpu_checkbox = gr.Checkbox(
                        value=self.vram_manager.config.enable_multi_gpu,
                        label="Enable Multi-GPU Support",
                        info="Use multiple GPUs for load balancing"
                    )
                    
                    apply_gpu_settings_btn = gr.Button("Apply GPU Settings", variant="primary")
            
            with gr.Column(scale=3):
                # Real-time VRAM Usage
                with gr.Group():
                    gr.Markdown("### Real-time VRAM Usage")
                    
                    vram_usage_html = gr.HTML(
                        value=self._get_vram_usage_html(),
                        label="VRAM Usage"
                    )
                    
                    with gr.Row():
                        start_monitoring_btn = gr.Button("▶️ Start Monitoring", variant="primary")
                        stop_monitoring_btn = gr.Button("⏹️ Stop Monitoring", variant="secondary")
                        optimize_memory_btn = gr.Button("🧹 Optimize Memory", variant="secondary")
        
        with gr.Row():
            # Memory Optimization Settings
            with gr.Column():
                with gr.Group():
                    gr.Markdown("### Memory Optimization Settings")
                    
                    memory_fraction_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=self.vram_manager.config.memory_fraction,
                        step=0.1,
                        label="Memory Fraction",
                        info="Fraction of VRAM to use (0.1-1.0)"
                    )
                    
                    optimization_threshold_slider = gr.Slider(
                        minimum=70,
                        maximum=95,
                        value=90,
                        step=5,
                        label="Optimization Threshold (%)",
                        info="Trigger optimization when usage exceeds this percentage"
                    )
                    
                    enable_auto_optimization = gr.Checkbox(
                        value=True,
                        label="Enable Automatic Memory Optimization",
                        info="Automatically optimize memory when threshold is exceeded"
                    )
            
            with gr.Column():
                # Manual VRAM Configuration
                with gr.Group():
                    gr.Markdown("### Manual VRAM Configuration")
                    
                    manual_config_json = gr.Textbox(
                        value=self._get_manual_config_json(),
                        label="Manual VRAM Config (JSON)",
                        info='Format: {"0": 16, "1": 8} for GPU indices and VRAM in GB',
                        lines=3
                    )
                    
                    validate_config_btn = gr.Button("Validate Config", variant="secondary")
                    apply_manual_config_btn = gr.Button("Apply Manual Config", variant="primary")
                    
                    config_status = gr.Textbox(
                        value="Ready",
                        label="Configuration Status",
                        interactive=False
                    )
        
        # Status and Logs
        with gr.Row():
            with gr.Column():
                system_status = gr.Textbox(
                    value="System ready",
                    label="System Status",
                    interactive=False,
                    lines=2
                )
                
                vram_logs = gr.Textbox(
                    value="VRAM monitoring logs will appear here...",
                    label="VRAM Monitor Logs",
                    interactive=False,
                    lines=5,
                    max_lines=10
                )
        
        # Event handlers
        components = {
            'gpu_info_display': gpu_info_display,
            'refresh_gpus_btn': refresh_gpus_btn,
            'detect_method_info': detect_method_info,
            'preferred_gpu_dropdown': preferred_gpu_dropdown,
            'multi_gpu_checkbox': multi_gpu_checkbox,
            'apply_gpu_settings_btn': apply_gpu_settings_btn,
            'vram_usage_html': vram_usage_html,
            'start_monitoring_btn': start_monitoring_btn,
            'stop_monitoring_btn': stop_monitoring_btn,
            'optimize_memory_btn': optimize_memory_btn,
            'memory_fraction_slider': memory_fraction_slider,
            'optimization_threshold_slider': optimization_threshold_slider,
            'enable_auto_optimization': enable_auto_optimization,
            'manual_config_json': manual_config_json,
            'validate_config_btn': validate_config_btn,
            'apply_manual_config_btn': apply_manual_config_btn,
            'config_status': config_status,
            'system_status': system_status,
            'vram_logs': vram_logs
        }
        
        self._setup_event_handlers(components)
        
        return components
    
    def _setup_event_handlers(self, components: Dict[str, Any]) -> None:
        """Setup event handlers for UI components"""
        
        # GPU Detection and Refresh
        components['refresh_gpus_btn'].click(
            fn=self._refresh_gpu_detection,
            outputs=[
                components['gpu_info_display'],
                components['detect_method_info'],
                components['preferred_gpu_dropdown'],
                components['system_status']
            ]
        )
        
        # GPU Settings Application
        components['apply_gpu_settings_btn'].click(
            fn=self._apply_gpu_settings,
            inputs=[
                components['preferred_gpu_dropdown'],
                components['multi_gpu_checkbox'],
                components['memory_fraction_slider']
            ],
            outputs=[components['system_status']]
        )
        
        # VRAM Monitoring Controls
        components['start_monitoring_btn'].click(
            fn=self._start_monitoring,
            outputs=[components['system_status']]
        )
        
        components['stop_monitoring_btn'].click(
            fn=self._stop_monitoring,
            outputs=[components['system_status']]
        )
        
        components['optimize_memory_btn'].click(
            fn=self._trigger_memory_optimization,
            outputs=[components['system_status']]
        )
        
        # Manual Configuration
        components['validate_config_btn'].click(
            fn=self._validate_manual_config,
            inputs=[components['manual_config_json']],
            outputs=[components['config_status']]
        )
        
        components['apply_manual_config_btn'].click(
            fn=self._apply_manual_config,
            inputs=[components['manual_config_json']],
            outputs=[
                components['config_status'],
                components['gpu_info_display'],
                components['preferred_gpu_dropdown']
            ]
        )
    
    def _get_gpu_info_html(self) -> str:
        """Generate HTML display for GPU information"""
        try:
            gpus = self.vram_manager.get_available_gpus()
            if not gpus:
                return "<div style='color: orange;'>⚠️ No GPUs detected. Click 'Refresh GPU Detection' to try again.</div>"
            
            html = "<div style='font-family: monospace;'>"
            html += "<h4>🖥️ Detected GPUs:</h4>"
            
            for gpu in gpus:
                status_color = "green" if gpu.is_available else "red"
                status_icon = "✅" if gpu.is_available else "❌"
                
                html += f"<div style='margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>"
                html += f"<div><strong>{status_icon} GPU {gpu.index}: {gpu.name}</strong></div>"
                html += f"<div>💾 VRAM: {gpu.total_memory_mb:,} MB ({gpu.total_memory_mb/1024:.1f} GB)</div>"
                html += f"<div>🔧 Driver: {gpu.driver_version}</div>"
                
                if gpu.cuda_version:
                    html += f"<div>🚀 CUDA: {gpu.cuda_version}</div>"
                
                if gpu.temperature is not None:
                    temp_color = "red" if gpu.temperature > 80 else "orange" if gpu.temperature > 70 else "green"
                    html += f"<div>🌡️ Temperature: <span style='color: {temp_color};'>{gpu.temperature}°C</span></div>"
                
                if gpu.utilization is not None:
                    util_color = "red" if gpu.utilization > 90 else "orange" if gpu.utilization > 70 else "green"
                    html += f"<div>⚡ Utilization: <span style='color: {util_color};'>{gpu.utilization}%</span></div>"
                
                html += "</div>"
            
            html += "</div>"
            return html
            
        except Exception as e:
            return f"<div style='color: red;'>❌ Error getting GPU info: {str(e)}</div>"
    
    def _get_vram_usage_html(self) -> str:
        """Generate HTML display for VRAM usage"""
        try:
            usage_list = self.vram_manager.get_current_vram_usage()
            if not usage_list:
                return "<div style='color: orange;'>⚠️ No VRAM usage data available. Start monitoring to see real-time usage.</div>"
            
            html = "<div style='font-family: monospace;'>"
            html += "<h4>📊 Current VRAM Usage:</h4>"
            
            for usage in usage_list:
                usage_color = "red" if usage.usage_percent > 90 else "orange" if usage.usage_percent > 70 else "green"
                
                html += f"<div style='margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>"
                html += f"<div><strong>🖥️ GPU {usage.gpu_index}</strong></div>"
                html += f"<div>📈 Usage: <span style='color: {usage_color}; font-weight: bold;'>{usage.usage_percent:.1f}%</span></div>"
                html += f"<div>💾 Used: {usage.used_mb:,} MB ({usage.used_mb/1024:.1f} GB)</div>"
                html += f"<div>🆓 Free: {usage.free_mb:,} MB ({usage.free_mb/1024:.1f} GB)</div>"
                html += f"<div>📦 Total: {usage.total_mb:,} MB ({usage.total_mb/1024:.1f} GB)</div>"
                html += f"<div>🕒 Updated: {usage.timestamp.strftime('%H:%M:%S')}</div>"
                
                # Progress bar
                html += f"<div style='margin-top: 5px;'>"
                html += f"<div style='background-color: #f0f0f0; border-radius: 10px; overflow: hidden;'>"
                html += f"<div style='width: {usage.usage_percent}%; height: 20px; background-color: {usage_color}; transition: width 0.3s;'></div>"
                html += f"</div></div>"
                
                html += "</div>"
            
            html += "</div>"
            return html
            
        except Exception as e:
            return f"<div style='color: red;'>❌ Error getting VRAM usage: {str(e)}</div>"
    
    def _get_available_gpu_choices(self) -> List[Tuple[str, int]]:
        """Get available GPU choices for dropdown"""
        try:
            gpus = self.vram_manager.get_available_gpus()
            choices = []
            for gpu in gpus:
                label = f"GPU {gpu.index}: {gpu.name} ({gpu.total_memory_mb/1024:.1f}GB)"
                choices.append((label, gpu.index))
            return choices
        except:
            return [("No GPUs available", -1)]
    
    def _get_current_preferred_gpu(self) -> Optional[int]:
        """Get currently preferred GPU"""
        return self.vram_manager.config.preferred_gpu
    
    def _get_manual_config_json(self) -> str:
        """Get manual configuration as JSON string"""
        if self.vram_manager.config.manual_vram_gb:
            return json.dumps(self.vram_manager.config.manual_vram_gb, indent=2)
        return '{\n  "0": 16,\n  "1": 8\n}'
    
    def _refresh_gpu_detection(self) -> Tuple[str, str, List[Tuple[str, int]], str]:
        """Refresh GPU detection"""
        try:
            gpus = self.vram_manager.detect_vram_capacity()
            detection_summary = self.vram_manager.get_detection_summary()
            
            # Determine detection method used
            method_info = "Detection successful"
            if detection_summary.get('nvml_available'):
                method_info = "✅ NVIDIA ML (NVML) - Primary method"
            elif detection_summary.get('torch_available'):
                method_info = "⚠️ PyTorch CUDA - Secondary method"
            else:
                method_info = "⚠️ Fallback method used"
            
            gpu_info_html = self._get_gpu_info_html()
            gpu_choices = self._get_available_gpu_choices()
            status = f"✅ Detected {len(gpus)} GPU(s) successfully"
            
            return gpu_info_html, method_info, gpu_choices, status
            
        except Exception as e:
            error_msg = f"❌ GPU detection failed: {str(e)}"
            return error_msg, "❌ Detection failed", [], error_msg
    
    def _apply_gpu_settings(self, preferred_gpu: int, enable_multi_gpu: bool, memory_fraction: float) -> str:
        """Apply GPU settings"""
        try:
            if preferred_gpu >= 0:
                self.vram_manager.set_preferred_gpu(preferred_gpu)
            
            self.vram_manager.enable_multi_gpu(enable_multi_gpu)
            self.vram_manager.config.memory_fraction = memory_fraction
            self.vram_manager._save_config()
            
            return f"✅ GPU settings applied: Preferred GPU {preferred_gpu}, Multi-GPU {'enabled' if enable_multi_gpu else 'disabled'}, Memory fraction {memory_fraction}"
            
        except Exception as e:
            return f"❌ Failed to apply GPU settings: {str(e)}"
    
    def _start_monitoring(self) -> str:
        """Start VRAM monitoring"""
        try:
            if not self.vram_manager.monitoring_active:
                self.vram_manager.start_monitoring(interval_seconds=self.update_interval)
                return "✅ VRAM monitoring started"
            else:
                return "⚠️ VRAM monitoring already active"
        except Exception as e:
            return f"❌ Failed to start monitoring: {str(e)}"
    
    def _stop_monitoring(self) -> str:
        """Stop VRAM monitoring"""
        try:
            if self.vram_manager.monitoring_active:
                self.vram_manager.stop_monitoring()
                return "✅ VRAM monitoring stopped"
            else:
                return "⚠️ VRAM monitoring not active"
        except Exception as e:
            return f"❌ Failed to stop monitoring: {str(e)}"
    
    def _trigger_memory_optimization(self) -> str:
        """Trigger memory optimization for all GPUs"""
        try:
            gpus = self.vram_manager.get_available_gpus()
            optimized_count = 0
            
            for gpu in gpus:
                self.vram_manager._trigger_memory_optimization(gpu.index)
                optimized_count += 1
            
            return f"✅ Memory optimization triggered for {optimized_count} GPU(s)"
            
        except Exception as e:
            return f"❌ Memory optimization failed: {str(e)}"
    
    def _validate_manual_config(self, config_json: str) -> str:
        """Validate manual VRAM configuration"""
        try:
            config_dict = json.loads(config_json)
            
            # Convert string keys to integers
            gpu_vram_mapping = {}
            for key, value in config_dict.items():
                gpu_vram_mapping[int(key)] = int(value)
            
            is_valid, errors = self.vram_manager.validate_manual_config(gpu_vram_mapping)
            
            if is_valid:
                return "✅ Configuration is valid"
            else:
                return f"❌ Configuration errors:\n" + "\n".join(errors)
                
        except json.JSONDecodeError as e:
            return f"❌ Invalid JSON format: {str(e)}"
        except ValueError as e:
            return f"❌ Invalid values: {str(e)}"
        except Exception as e:
            return f"❌ Validation error: {str(e)}"
    
    def _apply_manual_config(self, config_json: str) -> Tuple[str, str, List[Tuple[str, int]]]:
        """Apply manual VRAM configuration"""
        try:
            config_dict = json.loads(config_json)
            
            # Convert string keys to integers
            gpu_vram_mapping = {}
            for key, value in config_dict.items():
                gpu_vram_mapping[int(key)] = int(value)
            
            is_valid, errors = self.vram_manager.validate_manual_config(gpu_vram_mapping)
            
            if not is_valid:
                error_msg = f"❌ Configuration errors:\n" + "\n".join(errors)
                return error_msg, self._get_gpu_info_html(), self._get_available_gpu_choices()
            
            self.vram_manager.set_manual_vram_config(gpu_vram_mapping)
            
            # Refresh GPU detection with new manual config
            self.vram_manager.detect_vram_capacity()
            
            status = f"✅ Manual configuration applied: {gpu_vram_mapping}"
            gpu_info_html = self._get_gpu_info_html()
            gpu_choices = self._get_available_gpu_choices()
            
            return status, gpu_info_html, gpu_choices
            
        except Exception as e:
            error_msg = f"❌ Failed to apply configuration: {str(e)}"
            return error_msg, self._get_gpu_info_html(), self._get_available_gpu_choices()
    
    def cleanup(self) -> None:
        """Cleanup UI resources"""
        self.ui_active = False
        if self.ui_update_thread:
            self.ui_update_thread.join(timeout=2.0)


def create_vram_monitor_tab(vram_manager: Optional[VRAMManager] = None) -> Dict[str, Any]:
    """
    Create a complete VRAM monitor tab for Gradio interface
    
    Args:
        vram_manager: Optional VRAMManager instance
        
    Returns:
        Dictionary of UI components
    """
    monitor_ui = VRAMMonitorUI(vram_manager)
    
    with gr.Tab("🖥️ VRAM Monitor"):
        gr.Markdown("""
        # VRAM Monitor & GPU Management
        
        Monitor VRAM usage in real-time, configure GPU settings, and optimize memory usage.
        """)
        
        components = monitor_ui.create_ui_components()
    
    return components


if __name__ == "__main__":
    # Demo the VRAM monitor UI
    import gradio as gr
    
    def demo_app():
        with gr.Blocks(title="VRAM Monitor Demo") as demo:
            create_vram_monitor_tab()
        
        return demo
    
    if __name__ == "__main__":
        demo = demo_app()
        demo.launch()
