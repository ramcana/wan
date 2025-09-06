#!/usr/bin/env python3
"""
WAN2.2 Main Application Entry Point
Handles application startup and coordination between frontend and backend
"""

import argparse
import asyncio
import logging
import os
import sys
import subprocess
import signal
import time
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_python_path():
    """Setup Python path for imports"""
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

def start_backend():
    """Start the FastAPI backend server"""
    try:
        logger.info("Starting FastAPI backend...")
        backend_dir = Path(__file__).parent / "backend"
        
        # Change to backend directory and start uvicorn
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ]
        
        backend_process = subprocess.Popen(
            cmd,
            cwd=str(backend_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        logger.info("Backend started on http://localhost:8000")
        return backend_process
        
    except Exception as e:
        logger.error(f"Failed to start backend: {e}")
        return None

def start_frontend():
    """Start the React frontend development server"""
    try:
        logger.info("Starting React frontend...")
        frontend_dir = Path(__file__).parent / "frontend"
        
        # Check if node_modules exists
        if not (frontend_dir / "node_modules").exists():
            logger.info("Installing frontend dependencies...")
            subprocess.run(["npm", "install"], cwd=str(frontend_dir), check=True)
        
        # Start the development server
        cmd = ["npm", "run", "dev"]
        
        frontend_process = subprocess.Popen(
            cmd,
            cwd=str(frontend_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        logger.info("Frontend started on http://localhost:3000")
        return frontend_process
        
    except Exception as e:
        logger.error(f"Failed to start frontend: {e}")
        return None

def start_gradio_ui():
    """Start the Gradio UI (legacy mode)"""
    try:
        logger.info("Starting Gradio UI...")
        
        # Import and start the Gradio UI
        from frontend.ui import create_interface
        
        interface = create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False
        )
        
    except Exception as e:
        logger.error(f"Failed to start Gradio UI: {e}")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="WAN2.2 Video Generation System")
    parser.add_argument(
        "--mode", 
        choices=["full", "backend", "frontend", "gradio"], 
        default="full",
        help="Application mode to run"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Backend port (default: 8000)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Setup Python path
    setup_python_path()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting WAN2.2 in {args.mode} mode...")
    
    processes = []
    
    try:
        if args.mode in ["full", "backend"]:
            backend_process = start_backend()
            if backend_process:
                processes.append(backend_process)
        
        if args.mode in ["full", "frontend"]:
            # Wait a moment for backend to start
            if args.mode == "full":
                time.sleep(3)
            
            frontend_process = start_frontend()
            if frontend_process:
                processes.append(frontend_process)
        
        if args.mode == "gradio":
            start_gradio_ui()
            return
        
        if not processes:
            logger.error("No processes started successfully")
            return
        
        logger.info("All services started successfully!")
        logger.info("Access the application at:")
        
        if args.mode in ["full", "frontend"]:
            logger.info("  Frontend: http://localhost:3000")
        
        if args.mode in ["full", "backend"]:
            logger.info("  Backend API: http://localhost:8000")
            logger.info("  API Documentation: http://localhost:8000/docs")
        
        # Wait for processes
        def signal_handler(signum, frame):
            logger.info("Shutting down...")
            for process in processes:
                process.terminate()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Keep the main process alive
        while True:
            time.sleep(1)
            
            # Check if any process has died
            for process in processes[:]:
                if process.poll() is not None:
                    logger.error(f"Process {process.pid} has died")
                    processes.remove(process)
            
            if not processes:
                logger.error("All processes have died, exiting")
                break
                
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        # Clean up processes
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()

if __name__ == "__main__":
    main()
