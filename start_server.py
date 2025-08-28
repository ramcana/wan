#!/usr/bin/env python3
"""
Startup script for WAN22 FastAPI Backend Server
Ensures proper directory and import path setup
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the directory containing this script
    script_dir = Path(__file__).parent.absolute()
    backend_dir = script_dir / "backend"
    
    # Change to backend directory
    os.chdir(backend_dir)
    
    # Add backend directory to Python path
    sys.path.insert(0, str(backend_dir))
    
    print("🚀 Starting WAN22 FastAPI Backend Server...")
    print(f"📁 Working directory: {backend_dir}")
    print(f"🌐 Server will be available at: http://localhost:8000")
    print(f"📚 API Documentation: http://localhost:8000/docs")
    print(f"🔌 WebSocket endpoint: ws://localhost:8000/ws")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Start the server using uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server failed to start: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()