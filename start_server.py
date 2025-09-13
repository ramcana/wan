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
    
    # Add project root to Python path for proper module resolution
    sys.path.insert(0, str(script_dir))
    
    # Run the backend as a module to ensure proper import resolution
    print("Starting WAN22 FastAPI Backend Server using module execution...")
    print(f"🔌 WebSocket endpoint: ws://localhost:8000/ws")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Use subprocess to run the backend as a module
        result = subprocess.run([
            sys.executable, "-m", "backend"
        ], cwd=str(script_dir))
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
