#!/usr/bin/env python3
"""
Simple script to start the FastAPI backend server
"""

import uvicorn
import os
import sys
import argparse
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def main():
    parser = argparse.ArgumentParser(description="Start WAN22 FastAPI Backend Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9000, help="Port to bind to (default: 9000)")
    parser.add_argument("--reload", action="store_true", default=True, help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level (default: info)")
    
    args = parser.parse_args()
    
    print("Starting WAN22 FastAPI Backend Server...")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    print(f"WebSocket endpoint: ws://{args.host}:{args.port}/ws")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            "app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()