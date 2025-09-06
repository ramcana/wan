#!/usr/bin/env python3
"""
WAN22 Simple Startup Script
Single entry point for all users - no confusion, just works.
"""

import os
import sys
import subprocess
import time
import webbrowser
import argparse
from pathlib import Path

def print_banner():
    """Print a simple, friendly banner"""
    print("\n" + "="*50)
    print("🚀 WAN22 Video Generation System")
    print("="*50)

def check_requirements():
    """Check comprehensive system requirements and provide helpful messages"""
    print("📋 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"❌ Python {python_version.major}.{python_version.minor} found, but 3.8+ required")
        print("   Download Python 3.8+ from https://python.org")
        print("   Make sure to check 'Add Python to PATH' during installation")
        return False
    elif python_version >= (3, 12):
        print(f"⚠️  Python {python_version.major}.{python_version.minor} found - some dependencies may have issues")
        print("   Python 3.8-3.11 recommended for best compatibility")
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor} OK")
    
    # Check if we're in the right directory
    if not Path("backend").exists() or not Path("frontend").exists():
        print("❌ Please run this script from the project root directory")
        print("   (The directory containing 'backend' and 'frontend' folders)")
        print(f"   Current directory: {Path.cwd()}")
        return False
    print("✅ Project structure OK")
    
    # Check Node.js version
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, "node")
        
        node_version = result.stdout.strip()
        version_num = int(node_version.replace('v', '').split('.')[0])
        
        if version_num < 16:
            print(f"❌ Node.js {node_version} found, but version 16+ required")
            print("   Download Node.js LTS from https://nodejs.org")
            return False
        elif version_num >= 21:
            print(f"⚠️  Node.js {node_version} found - may have compatibility issues")
            print("   Node.js 16-20 LTS recommended")
        else:
            print(f"✅ Node.js {node_version} OK")
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Node.js not found")
        print("   Download and install Node.js 16+ LTS from https://nodejs.org")
        print("   Make sure to restart your terminal after installation")
        return False
    
    # Check available disk space
    try:
        import shutil
        free_space = shutil.disk_usage('.').free / (1024**3)  # GB
        if free_space < 10:
            print(f"⚠️  Low disk space: {free_space:.1f} GB free")
            print("   At least 10 GB recommended, 50+ GB for models")
        else:
            print(f"✅ Disk space OK ({free_space:.1f} GB free)")
    except:
        print("⚠️  Could not check disk space")
    
    # Check memory (basic check)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 8:
            print(f"⚠️  Low system memory: {memory_gb:.1f} GB")
            print("   8+ GB RAM recommended for optimal performance")
        else:
            print(f"✅ System memory OK ({memory_gb:.1f} GB)")
    except ImportError:
        # psutil not available, skip memory check
        pass
    
    # Check for GPU (optional)
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected (will enable GPU acceleration)")
        else:
            print("ℹ️  No NVIDIA GPU detected (will use CPU mode)")
    except FileNotFoundError:
        print("ℹ️  No NVIDIA GPU detected (will use CPU mode)")
    
    return True

def install_dependencies():
    """Install dependencies if needed"""
    print("\n📦 Checking dependencies...")
    
    # Check backend dependencies
    try:
        import fastapi
        import uvicorn
        print("✅ Backend dependencies OK")
    except ImportError:
        print("📥 Installing backend dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"], check=True)
        print("✅ Backend dependencies installed")
    
    # Check frontend dependencies
    if not Path("frontend/node_modules").exists():
        print("📥 Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd="frontend", check=True)
        print("✅ Frontend dependencies installed")
    else:
        print("✅ Frontend dependencies OK")

def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from start_port"""
    import socket
    
    for i in range(max_attempts):
        port = start_port + i
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except OSError:
                continue
    return None

def start_backend(port=8000):
    """Start the backend server"""
    print(f"\n🔧 Starting backend server on port {port}...")
    
    # Find available port if default is taken
    available_port = find_available_port(port)
    if available_port != port:
        print(f"   Port {port} is busy, using port {available_port}")
        port = available_port
    
    if available_port is None:
        print(f"❌ Could not find available port starting from {port}")
        return None, None
    
    # Start backend
    backend_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "app:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload"
    ], cwd="backend")
    
    # Wait a moment and check if it started
    time.sleep(2)
    if backend_process.poll() is not None:
        print("❌ Backend failed to start")
        return None, None
    
    print(f"✅ Backend running at http://localhost:{port}")
    return backend_process, port

def start_frontend(backend_port, port=3000):
    """Start the frontend server"""
    print(f"\n⚛️  Starting frontend server on port {port}...")
    
    # Find available port if default is taken
    available_port = find_available_port(port)
    if available_port != port:
        print(f"   Port {port} is busy, using port {available_port}")
        port = available_port
    
    if available_port is None:
        print(f"❌ Could not find available port starting from {port}")
        return None, None
    
    # Set environment variable for backend URL
    env = os.environ.copy()
    env['VITE_API_URL'] = f'http://localhost:{backend_port}'
    
    # Start frontend
    frontend_process = subprocess.Popen([
        "npm", "run", "dev", "--", "--port", str(port)
    ], cwd="frontend", env=env)
    
    # Wait a moment and check if it started
    time.sleep(3)
    if frontend_process.poll() is not None:
        print("❌ Frontend failed to start")
        return None, None
    
    print(f"✅ Frontend running at http://localhost:{port}")
    return frontend_process, port

def main():
    """Main startup function with comprehensive error handling"""
    print_banner()
    
    backend_process = None
    frontend_process = None
    
    try:
        # Check requirements
        if not check_requirements():
            print("\n🔧 Troubleshooting:")
            print("   • See SYSTEM_REQUIREMENTS.md for detailed requirements")
            print("   • See COMPREHENSIVE_TROUBLESHOOTING.md for solutions")
            input("\nPress Enter to exit...")
            return
        
        # Install dependencies
        try:
            install_dependencies()
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Dependency installation failed: {e}")
            print("\n🔧 Try these solutions:")
            print("   • Run as administrator")
            print("   • Check internet connection")
            print("   • Clear package caches:")
            print("     - pip cache purge")
            print("     - npm cache clean --force")
            input("\nPress Enter to exit...")
            return
        except Exception as e:
            print(f"\n❌ Unexpected error during dependency installation: {e}")
            input("\nPress Enter to exit...")
            return
        
        # Start backend
        try:
            backend_process, backend_port = start_backend()
            if not backend_process:
                print("\n🔧 Backend startup failed. Try these solutions:")
                print("   • Check if port 8000 is available")
                print("   • Run: netstat -ano | findstr :8000")
                print("   • Kill any processes using port 8000")
                print("   • Try running as administrator")
                input("\nPress Enter to exit...")
                return
        except Exception as e:
            print(f"\n❌ Backend startup error: {e}")
            print("\n🔧 Troubleshooting steps:")
            print("   • Check backend/requirements.txt is installed")
            print("   • Verify Python path and imports")
            print("   • See COMPREHENSIVE_TROUBLESHOOTING.md")
            input("\nPress Enter to exit...")
            return
        
        # Start frontend
        try:
            frontend_process, frontend_port = start_frontend(backend_port)
            if not frontend_process:
                print("\n🔧 Frontend startup failed. Try these solutions:")
                print("   • Check if Node.js and npm are installed")
                print("   • Run: cd frontend && npm install")
                print("   • Check if port 3000 is available")
                backend_process.terminate()
                input("\nPress Enter to exit...")
                return
        except Exception as e:
            print(f"\n❌ Frontend startup error: {e}")
            print("\n🔧 Troubleshooting steps:")
            print("   • Check Node.js version (16+ required)")
            print("   • Clear npm cache: npm cache clean --force")
            print("   • Delete node_modules and reinstall")
            if backend_process:
                backend_process.terminate()
            input("\nPress Enter to exit...")
            return
        
        # Success message
        print("\n" + "="*50)
        print("🎉 WAN22 is now running!")
        print("="*50)
        print(f"🌐 Frontend: http://localhost:{frontend_port}")
        print(f"🔧 Backend API: http://localhost:{backend_port}")
        print(f"📚 API docs: http://localhost:{backend_port}/docs")
        print("\n💡 Tips:")
        print("   • Keep this window open while using WAN22")
        print("   • Press Ctrl+C to stop both servers")
        print("   • Check browser console for any errors")
        print("   • See QUICK_START.md for usage guide")
        print("="*50)
        
        # Open browser automatically
        try:
            webbrowser.open(f"http://localhost:{frontend_port}")
            print("🌐 Opening browser automatically...")
        except Exception as e:
            print(f"ℹ️  Could not open browser automatically: {e}")
            print(f"   Please open http://localhost:{frontend_port} manually")
        
        # Wait for user to stop with better monitoring
        try:
            print("\n⏳ Monitoring servers... (Press Ctrl+C to stop)")
            consecutive_failures = 0
            
            while True:
                time.sleep(2)
                
                # Check if processes are still running
                backend_alive = backend_process.poll() is None
                frontend_alive = frontend_process.poll() is None
                
                if not backend_alive:
                    print("\n❌ Backend process stopped unexpectedly")
                    print("🔧 Check the backend terminal window for error messages")
                    consecutive_failures += 1
                    
                if not frontend_alive:
                    print("\n❌ Frontend process stopped unexpectedly")
                    print("🔧 Check the frontend terminal window for error messages")
                    consecutive_failures += 1
                
                if consecutive_failures >= 2:
                    print("❌ Multiple server failures detected, exiting...")
                    break
                    
                if not backend_alive or not frontend_alive:
                    break
                    
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down WAN22...")
        
        # Clean shutdown
        print("🛑 Stopping servers...")
        
        if backend_process and backend_process.poll() is None:
            backend_process.terminate()
            print("   • Backend server stopped")
            
        if frontend_process and frontend_process.poll() is None:
            frontend_process.terminate()
            print("   • Frontend server stopped")
        
        # Wait for graceful shutdown
        try:
            if backend_process:
                backend_process.wait(timeout=5)
            if frontend_process:
                frontend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("⚠️  Force killing unresponsive processes...")
            if backend_process:
                backend_process.kill()
            if frontend_process:
                frontend_process.kill()
        
        print("✅ Shutdown complete")
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        
    except PermissionError as e:
        print(f"\n❌ Permission error: {e}")
        print("\n🔧 Solutions:")
        print("   • Run as administrator (right-click → 'Run as administrator')")
        print("   • Check Windows Firewall settings")
        print("   • Add Python and Node.js to firewall exceptions")
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("\n🔧 Solutions:")
        print("   • Verify you're in the correct project directory")
        print("   • Check that all required files exist")
        print("   • Re-download the project if files are missing")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(f"   Error type: {type(e).__name__}")
        print("\n🔧 Troubleshooting:")
        print("   • See COMPREHENSIVE_TROUBLESHOOTING.md for detailed help")
        print("   • Check system requirements in SYSTEM_REQUIREMENTS.md")
        print("   • Try running with administrator privileges")
        print("   • Restart your computer and try again")
        
    finally:
        # Ensure cleanup happens
        if backend_process and backend_process.poll() is None:
            try:
                backend_process.terminate()
                backend_process.wait(timeout=3)
            except:
                try:
                    backend_process.kill()
                except:
                    pass
                    
        if frontend_process and frontend_process.poll() is None:
            try:
                frontend_process.terminate()
                frontend_process.wait(timeout=3)
            except:
                try:
                    frontend_process.kill()
                except:
                    pass
        
        input("\nPress Enter to exit...")

def run_diagnostics():
    """Run comprehensive system diagnostics"""
    print("🔍 WAN22 System Diagnostics")
    print("="*50)
    
    # System info
    import platform
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version}")
    
    # Check Python packages
    print("\n📦 Python Environment:")
    try:
        import pip
        installed_packages = subprocess.run([sys.executable, "-m", "pip", "list"], 
                                          capture_output=True, text=True)
        key_packages = ['fastapi', 'uvicorn', 'torch', 'transformers']
        for package in key_packages:
            if package in installed_packages.stdout:
                print(f"✅ {package} installed")
            else:
                print(f"❌ {package} missing")
    except Exception as e:
        print(f"⚠️  Could not check Python packages: {e}")
    
    # Check Node.js
    print("\n🟢 Node.js Environment:")
    try:
        node_result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        npm_result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        print(f"Node.js: {node_result.stdout.strip()}")
        print(f"npm: {npm_result.stdout.strip()}")
    except Exception as e:
        print(f"❌ Node.js check failed: {e}")
    
    # Check GPU
    print("\n🎮 GPU Information:")
    try:
        nvidia_result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", 
                                      "--format=csv,noheader,nounits"], 
                                     capture_output=True, text=True)
        if nvidia_result.returncode == 0:
            for line in nvidia_result.stdout.strip().split('\n'):
                if line.strip():
                    name, memory, driver = line.split(', ')
                    print(f"✅ {name} ({memory} MB VRAM, Driver: {driver})")
        else:
            print("ℹ️  No NVIDIA GPU detected")
    except Exception:
        print("ℹ️  No NVIDIA GPU detected")
    
    # Check disk space
    print("\n💾 Storage Information:")
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        print(f"Total: {total // (1024**3)} GB")
        print(f"Used: {used // (1024**3)} GB") 
        print(f"Free: {free // (1024**3)} GB")
        
        if free < 10 * (1024**3):  # Less than 10 GB
            print("⚠️  Low disk space - consider freeing up space")
        else:
            print("✅ Adequate disk space available")
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
    
    # Check memory
    print("\n🧠 Memory Information:")
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Total RAM: {memory.total // (1024**3)} GB")
        print(f"Available: {memory.available // (1024**3)} GB")
        print(f"Usage: {memory.percent}%")
        
        if memory.total < 8 * (1024**3):  # Less than 8 GB
            print("⚠️  Low system memory - 8+ GB recommended")
        else:
            print("✅ Adequate system memory")
    except ImportError:
        print("ℹ️  psutil not available - install with: pip install psutil")
    except Exception as e:
        print(f"⚠️  Could not check memory: {e}")
    
    # Check ports
    print("\n🌐 Network Ports:")
    import socket
    ports_to_check = [3000, 8000, 7860]
    for port in ports_to_check:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                print(f"✅ Port {port} available")
            except OSError:
                print(f"⚠️  Port {port} in use")
    
    # Check project structure
    print("\n📁 Project Structure:")
    required_paths = ['backend', 'frontend', 'backend/app.py', 'frontend/package.json']
    for path in required_paths:
        if Path(path).exists():
            print(f"✅ {path}")
        else:
            print(f"❌ {path} missing")
    
    print("\n" + "="*50)
    print("🔍 Diagnostic complete!")
    print("💡 Share this output when asking for help")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WAN22 Startup Script")
    parser.add_argument("--diagnostics", action="store_true", 
                       help="Run system diagnostics instead of starting servers")
    
    args = parser.parse_args()
    
    if args.diagnostics:
        run_diagnostics()
    else:
        main()