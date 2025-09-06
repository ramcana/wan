#!/usr/bin/env python3
"""
Full Stack Startup Script for Wan2.2 React + FastAPI
Starts both backend and frontend services for development testing
"""

import os
import sys
import subprocess
import time
import threading
import signal
import requests
from pathlib import Path

class FullStackRunner:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = True
        
    def check_port(self, port, service_name):
        """Check if a port is available"""
        try:
            response = requests.get(f"http://localhost:{port}", timeout=2)
            print(f"✓ {service_name} is already running on port {port}")
            return True
        except:
            return False
    
    def start_backend(self):
        """Start the FastAPI backend"""
        print("🚀 Starting FastAPI backend...")
        
        # Change to backend directory
        backend_dir = Path(__file__).parent / "backend"
        
        # Check if we're in a virtual environment
        if not os.environ.get('VIRTUAL_ENV'):
            print("⚠️  Warning: No virtual environment detected. Consider using venv or conda.")
        
        try:
            # Start backend with uvicorn
            cmd = [
                sys.executable, "-m", "uvicorn", 
                "main:app", 
                "--host", "0.0.0.0", 
                "--port", "8000", 
                "--reload",
                "--log-level", "info"
            ]
            
            self.backend_process = subprocess.Popen(
                cmd,
                cwd=backend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Monitor backend output in a separate thread
            def monitor_backend():
                for line in iter(self.backend_process.stdout.readline, ''):
                    if self.running:
                        print(f"[BACKEND] {line.strip()}")
                    else:
                        break
            
            backend_thread = threading.Thread(target=monitor_backend, daemon=True)
            backend_thread.start()
            
            # Wait for backend to start
            print("⏳ Waiting for backend to start...")
            for i in range(30):  # Wait up to 30 seconds
                if self.check_port(8000, "Backend"):
                    break
                time.sleep(1)
            else:
                print("❌ Backend failed to start within 30 seconds")
                return False
                
            print("✅ Backend started successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the React frontend"""
        print("🚀 Starting React frontend...")
        
        frontend_dir = Path(__file__).parent / "frontend"
        
        # Check if node_modules exists
        if not (frontend_dir / "node_modules").exists():
            print("📦 Installing frontend dependencies...")
            try:
                subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install dependencies: {e}")
                return False
        
        try:
            # Start frontend with npm dev
            cmd = ["npm", "run", "dev"]
            
            self.frontend_process = subprocess.Popen(
                cmd,
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Monitor frontend output in a separate thread
            def monitor_frontend():
                for line in iter(self.frontend_process.stdout.readline, ''):
                    if self.running:
                        print(f"[FRONTEND] {line.strip()}")
                    else:
                        break
            
            frontend_thread = threading.Thread(target=monitor_frontend, daemon=True)
            frontend_thread.start()
            
            # Wait for frontend to start
            print("⏳ Waiting for frontend to start...")
            for i in range(60):  # Wait up to 60 seconds for Vite
                if self.check_port(3000, "Frontend"):
                    break
                time.sleep(1)
            else:
                print("❌ Frontend failed to start within 60 seconds")
                return False
                
            print("✅ Frontend started successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return False
    
    def test_integration(self):
        """Test the integration between frontend and backend"""
        print("\n🧪 Testing integration...")
        
        # Test backend health
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Backend health check passed")
            else:
                print(f"⚠️  Backend health check returned {response.status_code}")
        except Exception as e:
            print(f"❌ Backend health check failed: {e}")
        
        # Test enhanced health check
        try:
            response = requests.get("http://localhost:8000/api/v1/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print("✅ Enhanced health check passed")
                print(f"   System status: {health_data.get('status', 'unknown')}")
                
                # Show check results
                checks = health_data.get('checks', {})
                for check_name, check_data in checks.items():
                    status = check_data.get('status', 'unknown')
                    emoji = "✅" if status == "pass" else "❌"
                    print(f"   {emoji} {check_name}: {status}")
                    
            else:
                print(f"⚠️  Enhanced health check returned {response.status_code}")
        except Exception as e:
            print(f"❌ Enhanced health check failed: {e}")
        
        # Test CORS and API endpoints
        try:
            response = requests.get("http://localhost:8000/api/v1/system/info", timeout=5)
            if response.status_code == 200:
                print("✅ System info API accessible")
            else:
                print(f"⚠️  System info API returned {response.status_code}")
        except Exception as e:
            print(f"❌ System info API failed: {e}")
        
        print("\n🌐 Services are running:")
        print("   Frontend: http://localhost:3000")
        print("   Backend API: http://localhost:8000")
        print("   API Docs: http://localhost:8000/docs")
        print("   Health Check: http://localhost:8000/api/v1/health")
    
    def cleanup(self):
        """Clean up processes"""
        print("\n🛑 Shutting down services...")
        self.running = False
        
        if self.frontend_process:
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=5)
                print("✅ Frontend stopped")
            except:
                self.frontend_process.kill()
                print("⚠️  Frontend force killed")
        
        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
                print("✅ Backend stopped")
            except:
                self.backend_process.kill()
                print("⚠️  Backend force killed")
    
    def run(self):
        """Main run method"""
        print("🎬 Starting Wan2.2 Full Stack Development Environment")
        print("=" * 60)
        
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print(f"\n📡 Received signal {signum}")
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Start backend first
            if not self.start_backend():
                print("❌ Failed to start backend. Exiting.")
                return False
            
            # Start frontend
            if not self.start_frontend():
                print("❌ Failed to start frontend. Exiting.")
                self.cleanup()
                return False
            
            # Test integration
            self.test_integration()
            
            print("\n🎉 Full stack is running! Press Ctrl+C to stop.")
            print("=" * 60)
            
            # Keep running until interrupted
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            self.cleanup()
        
        return True

def main():
    """Main entry point"""
    runner = FullStackRunner()
    success = runner.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()