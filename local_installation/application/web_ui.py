"""
WAN2.2 Web UI Application
A web-based user interface for the WAN2.2 video generation system.
Alternative to the desktop UI for users who prefer browser-based interfaces.
"""

import os
import sys
import json
import threading
import time
import webbrowser
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
    from werkzeug.utils import secure_filename
except ImportError:
    print("Flask not installed. Please install with: pip install flask")
    sys.exit(1)

# Add the scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

try:
    from config_manager import ConfigurationManager
    from logging_system import setup_installation_logging
except ImportError:
    ConfigurationManager = None
    setup_installation_logging = None


class WAN22WebUI:
    """Web-based UI for WAN2.2 video generation."""
    
    def __init__(self, host='127.0.0.1', port=7860):
        self.host = host
        self.port = port
        self.app = Flask(__name__, 
                        template_folder=str(Path(__file__).parent / 'templates'),
                        static_folder=str(Path(__file__).parent / 'static'))
        
        # Configuration
        self.installation_path = Path(__file__).parent.parent
        self.config = self.load_configuration()
        self.generation_queue = []
        self.generation_in_progress = False
        
        # Setup logging
        self.setup_logging()
        
        # Setup Flask routes
        self.setup_routes()
        
        # Create necessary directories
        self.setup_directories()
    
    def setup_logging(self):
        """Setup logging for the web UI."""
        try:
            if setup_installation_logging:
                self.logging_system = setup_installation_logging(
                    installation_path=str(self.installation_path),
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
    
    def load_configuration(self) -> Dict[str, Any]:
        """Load system configuration."""
        try:
            config_path = self.installation_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                return self.get_default_config()
        except Exception as e:
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "system": {
                "threads": 4,
                "memory_limit_gb": 8,
                "default_quantization": "fp16",
                "enable_offload": True
            },
            "models": {
                "cache_dir": "models"
            },
            "ui": {
                "theme": "default",
                "auto_save": True
            }
        }
    
    def setup_directories(self):
        """Setup necessary directories."""
        directories = [
            self.installation_path / "outputs",
            self.installation_path / "uploads",
            self.installation_path / "application" / "templates",
            self.installation_path / "application" / "static"
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main page."""
            return self.render_index()
        
        @self.app.route('/api/status')
        def api_status():
            """Get system status."""
            return jsonify(self.get_system_status())
        
        @self.app.route('/api/generate', methods=['POST'])
        def api_generate():
            """Start video generation."""
            return jsonify(self.start_generation(request.json))
        
        @self.app.route('/api/queue')
        def api_queue():
            """Get generation queue status."""
            return jsonify(self.get_queue_status())
        
        @self.app.route('/api/outputs')
        def api_outputs():
            """Get list of generated outputs."""
            return jsonify(self.get_outputs())
        
        @self.app.route('/upload', methods=['POST'])
        def upload_file():
            """Handle file uploads."""
            return jsonify(self.handle_upload(request))
        
        @self.app.route('/download/<filename>')
        def download_file(filename):
            """Download generated files."""
            return self.handle_download(filename)
    
    def render_index(self):
        """Render the main index page."""
        # Create a simple HTML template if templates don't exist
        template_path = self.installation_path / "application" / "templates" / "index.html"
        if not template_path.exists():
            self.create_default_template()
        
        try:
            return render_template('index.html', 
                                 config=self.config,
                                 system_status=self.get_system_status())
        except Exception as e:
            # Fallback to simple HTML
            return self.get_simple_html()
    
    def create_default_template(self):
        """Create default HTML template."""
        template_dir = self.installation_path / "application" / "templates"
        template_dir.mkdir(exist_ok=True)
        
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WAN2.2 - Video Generation System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .tabs { display: flex; border-bottom: 2px solid #ddd; margin-bottom: 20px; }
        .tab { padding: 10px 20px; cursor: pointer; border: none; background: none; font-size: 16px; }
        .tab.active { background-color: #007bff; color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        .form-group input, .form-group textarea, .form-group select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        .form-group textarea { height: 80px; resize: vertical; }
        .btn { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        .btn:hover { background-color: #0056b3; }
        .btn:disabled { background-color: #ccc; cursor: not-allowed; }
        .status-bar { background-color: #f8f9fa; padding: 10px; border-radius: 4px; margin-bottom: 20px; }
        .progress-bar { width: 100%; height: 20px; background-color: #e9ecef; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background-color: #007bff; transition: width 0.3s ease; }
        .output-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }
        .output-item { border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: white; }
        .queue-item { padding: 10px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 WAN2.2 Video Generation System</h1>
            <div class="status-bar">
                <strong>Status:</strong> <span id="system-status">Loading...</span> |
                <strong>Models:</strong> <span id="models-status">Checking...</span> |
                <strong>GPU:</strong> <span id="gpu-status">Unknown</span>
            </div>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="showTab('t2v')">Text-to-Video</button>
            <button class="tab" onclick="showTab('i2v')">Image-to-Video</button>
            <button class="tab" onclick="showTab('ti2v')">Text+Image-to-Video</button>
            <button class="tab" onclick="showTab('queue')">Queue</button>
            <button class="tab" onclick="showTab('outputs')">Outputs</button>
        </div>

        <!-- Text-to-Video Tab -->
        <div id="t2v" class="tab-content active">
            <h3>Text-to-Video Generation</h3>
            <form id="t2v-form">
                <div class="form-group">
                    <label for="t2v-prompt">Text Prompt:</label>
                    <textarea id="t2v-prompt" placeholder="Describe the video you want to generate..." required></textarea>
                </div>
                <div class="form-group">
                    <label for="t2v-negative">Negative Prompt (Optional):</label>
                    <textarea id="t2v-negative" placeholder="What you don't want in the video..."></textarea>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div class="form-group">
                        <label for="t2v-duration">Duration (seconds):</label>
                        <input type="range" id="t2v-duration" min="1" max="10" value="4" oninput="updateValue('t2v-duration', 't2v-duration-value')">
                        <span id="t2v-duration-value">4</span>s
                    </div>
                    <div class="form-group">
                        <label for="t2v-resolution">Resolution:</label>
                        <select id="t2v-resolution">
                            <option value="512x512">512x512</option>
                            <option value="768x768">768x768</option>
                            <option value="1280x720" selected>1280x720 (HD)</option>
                            <option value="1920x1080">1920x1080 (Full HD)</option>
                        </select>
                    </div>
                </div>
                <button type="submit" class="btn" id="t2v-generate">Generate Video</button>
            </form>
        </div>

        <!-- Image-to-Video Tab -->
        <div id="i2v" class="tab-content">
            <h3>Image-to-Video Generation</h3>
            <form id="i2v-form">
                <div class="form-group">
                    <label for="i2v-image">Input Image:</label>
                    <input type="file" id="i2v-image" accept="image/*" required>
                </div>
                <div class="form-group">
                    <label for="i2v-prompt">Motion Description:</label>
                    <textarea id="i2v-prompt" placeholder="Describe how the image should move or animate..."></textarea>
                </div>
                <div class="form-group">
                    <label for="i2v-duration">Duration (seconds):</label>
                    <input type="range" id="i2v-duration" min="1" max="8" value="3" oninput="updateValue('i2v-duration', 'i2v-duration-value')">
                    <span id="i2v-duration-value">3</span>s
                </div>
                <button type="submit" class="btn" id="i2v-generate">Generate Video</button>
            </form>
        </div>

        <!-- Text+Image-to-Video Tab -->
        <div id="ti2v" class="tab-content">
            <h3>Text+Image-to-Video Generation</h3>
            <form id="ti2v-form">
                <div class="form-group">
                    <label for="ti2v-image">Input Image:</label>
                    <input type="file" id="ti2v-image" accept="image/*" required>
                </div>
                <div class="form-group">
                    <label for="ti2v-prompt">Text Prompt:</label>
                    <textarea id="ti2v-prompt" placeholder="Describe the video based on the image..." required></textarea>
                </div>
                <div class="form-group">
                    <label for="ti2v-duration">Duration (seconds):</label>
                    <input type="range" id="ti2v-duration" min="1" max="6" value="3" oninput="updateValue('ti2v-duration', 'ti2v-duration-value')">
                    <span id="ti2v-duration-value">3</span>s
                </div>
                <button type="submit" class="btn" id="ti2v-generate">Generate Video</button>
            </form>
        </div>

        <!-- Queue Tab -->
        <div id="queue" class="tab-content">
            <h3>Generation Queue</h3>
            <div id="queue-list">
                <p>No items in queue</p>
            </div>
        </div>

        <!-- Outputs Tab -->
        <div id="outputs" class="tab-content">
            <h3>Generated Videos</h3>
            <div id="outputs-grid" class="output-grid">
                <p>No outputs yet</p>
            </div>
        </div>
    </div>

    <script>
        // Tab switching
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }

        // Update range input values
        function updateValue(inputId, outputId) {
            const value = document.getElementById(inputId).value;
            document.getElementById(outputId).textContent = value;
        }

        // Update system status
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('system-status').textContent = data.status;
                    document.getElementById('models-status').textContent = data.models_status;
                    document.getElementById('gpu-status').textContent = data.gpu_status;
                })
                .catch(error => console.error('Error updating status:', error));
        }

        // Form submissions
        document.getElementById('t2v-form').addEventListener('submit', function(e) {
            e.preventDefault();
            generateVideo('t2v');
        });

        document.getElementById('i2v-form').addEventListener('submit', function(e) {
            e.preventDefault();
            generateVideo('i2v');
        });

        document.getElementById('ti2v-form').addEventListener('submit', function(e) {
            e.preventDefault();
            generateVideo('ti2v');
        });

        function generateVideo(type) {
            const formData = new FormData();
            formData.append('type', type);
            
            if (type === 't2v') {
                formData.append('prompt', document.getElementById('t2v-prompt').value);
                formData.append('negative_prompt', document.getElementById('t2v-negative').value);
                formData.append('duration', document.getElementById('t2v-duration').value);
                formData.append('resolution', document.getElementById('t2v-resolution').value);
            } else if (type === 'i2v') {
                const imageFile = document.getElementById('i2v-image').files[0];
                if (imageFile) formData.append('image', imageFile);
                formData.append('prompt', document.getElementById('i2v-prompt').value);
                formData.append('duration', document.getElementById('i2v-duration').value);
            } else if (type === 'ti2v') {
                const imageFile = document.getElementById('ti2v-image').files[0];
                if (imageFile) formData.append('image', imageFile);
                formData.append('prompt', document.getElementById('ti2v-prompt').value);
                formData.append('duration', document.getElementById('ti2v-duration').value);
            }

            fetch('/api/generate', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Generation started! Check the Queue tab for progress.');
                    showTab('queue');
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error starting generation');
            });
        }

        // Initialize
        updateStatus();
        setInterval(updateStatus, 5000); // Update every 5 seconds
    </script>
</body>
</html>'''
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def get_simple_html(self):
        """Get simple HTML fallback."""
        return '''
        <html>
        <head><title>WAN2.2 - Video Generation System</title></head>
        <body>
            <h1>WAN2.2 Video Generation System</h1>
            <p>Web UI is starting up...</p>
            <p>If this page doesn't load properly, please check the installation.</p>
        </body>
        </html>
        '''
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            # Check models
            models_dir = self.installation_path / "models"
            models_count = 0
            if models_dir.exists():
                model_files = list(models_dir.glob("**/*.bin")) + list(models_dir.glob("**/*.safetensors"))
                models_count = len(model_files)
            
            # Check GPU
            gpu_status = "CPU only"
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_status = f"CUDA ({torch.cuda.get_device_name(0)[:20]}...)"
            except ImportError:
                pass
            
            return {
                "status": "Ready" if models_count > 0 else "Models not loaded",
                "models_status": f"{models_count} models loaded" if models_count > 0 else "No models found",
                "gpu_status": gpu_status,
                "generation_in_progress": self.generation_in_progress,
                "queue_length": len(self.generation_queue)
            }
        except Exception as e:
            return {
                "status": "Error",
                "models_status": "Unknown",
                "gpu_status": "Unknown",
                "generation_in_progress": False,
                "queue_length": 0
            }
    
    def start_generation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Start video generation."""
        try:
            generation_type = data.get('type', 'unknown')
            task_id = f"{generation_type}_{int(time.time())}"
            
            # Add to queue
            task = {
                "id": task_id,
                "type": generation_type,
                "status": "queued",
                "progress": 0,
                "created": datetime.now().isoformat(),
                "params": data
            }
            
            self.generation_queue.append(task)
            
            # Start processing if not already running
            if not self.generation_in_progress:
                threading.Thread(target=self.process_queue, daemon=True).start()
            
            return {"success": True, "task_id": task_id}
            
        except Exception as e:
            self.logger.error(f"Error starting generation: {e}")
            return {"success": False, "error": str(e)}
    
    def process_queue(self):
        """Process the generation queue."""
        while self.generation_queue:
            if self.generation_in_progress:
                time.sleep(1)
                continue
            
            self.generation_in_progress = True
            task = self.generation_queue[0]
            
            try:
                # Update task status
                task["status"] = "processing"
                
                # Simulate generation (replace with actual generation code)
                for i in range(101):
                    task["progress"] = i
                    time.sleep(0.1)  # Simulate work
                
                # Complete task
                task["status"] = "completed"
                task["progress"] = 100
                
                # Move to completed (could save to outputs directory)
                self.generation_queue.pop(0)
                
            except Exception as e:
                self.logger.error(f"Generation error: {e}")
                task["status"] = "failed"
                task["error"] = str(e)
                self.generation_queue.pop(0)
            
            finally:
                self.generation_in_progress = False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get generation queue status."""
        return {
            "queue": self.generation_queue,
            "processing": self.generation_in_progress
        }
    
    def get_outputs(self) -> Dict[str, Any]:
        """Get list of generated outputs."""
        try:
            outputs_dir = self.installation_path / "outputs"
            outputs = []
            
            if outputs_dir.exists():
                for video_file in outputs_dir.glob("*.mp4"):
                    stat = video_file.stat()
                    outputs.append({
                        "name": video_file.name,
                        "size": stat.st_size,
                        "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "path": str(video_file.relative_to(self.installation_path))
                    })
            
            return {"outputs": outputs}
            
        except Exception as e:
            self.logger.error(f"Error getting outputs: {e}")
            return {"outputs": []}
    
    def handle_upload(self, request) -> Dict[str, Any]:
        """Handle file uploads."""
        try:
            if 'image' not in request.files:
                return {"success": False, "error": "No file uploaded"}
            
            file = request.files['image']
            if file.filename == '':
                return {"success": False, "error": "No file selected"}
            
            if file:
                filename = secure_filename(file.filename)
                upload_path = self.installation_path / "uploads" / filename
                file.save(str(upload_path))
                
                return {"success": True, "filename": filename, "path": str(upload_path)}
                
        except Exception as e:
            self.logger.error(f"Upload error: {e}")
            return {"success": False, "error": str(e)}
    
    def handle_download(self, filename):
        """Handle file downloads."""
        try:
            file_path = self.installation_path / "outputs" / filename
            if file_path.exists():
                return send_file(str(file_path), as_attachment=True)
            else:
                return "File not found", 404
        except Exception as e:
            self.logger.error(f"Download error: {e}")
            return "Error downloading file", 500
    
    def run(self, debug=False, open_browser=True):
        """Run the web UI."""
        self.logger.info(f"Starting WAN2.2 Web UI on http://{self.host}:{self.port}")
        
        if open_browser:
            # Open browser after a short delay
            def open_browser_delayed():
                time.sleep(1.5)
                webbrowser.open(f"http://{self.host}:{self.port}")
            
            threading.Thread(target=open_browser_delayed, daemon=True).start()
        
        try:
            self.app.run(host=self.host, port=self.port, debug=debug, use_reloader=False)
        except Exception as e:
            self.logger.error(f"Error running web UI: {e}")
            print(f"Error starting web UI: {e}")


def main():
    """Main entry point for the web UI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="WAN2.2 Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    
    args = parser.parse_args()
    
    try:
        web_ui = WAN22WebUI(host=args.host, port=args.port)
        web_ui.run(debug=args.debug, open_browser=not args.no_browser)
    except Exception as e:
        print(f"Error starting WAN2.2 Web UI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
