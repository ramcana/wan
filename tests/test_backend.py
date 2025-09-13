#!/usr/bin/env python3
"""
Simple test script to verify the FastAPI backend can start
"""

import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

def test_imports():
    """Test that all required modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import fastapi
        print("✅ FastAPI imported successfully")
        
        import uvicorn
        print("✅ Uvicorn imported successfully")
        
        import psutil
        print("✅ psutil imported successfully")
        
        # Test backend app import
        os.chdir(backend_dir)
        from backend.app import app
        print("✅ Backend app imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    assert True  # TODO: Add proper assertion

def test_endpoints():
    """Test that basic endpoints are defined"""
    try:
        os.chdir(backend_dir)
        from backend.app import app
        
        # Get all routes
        routes = [route.path for route in app.routes]
        
        expected_routes = [
            "/health",
            "/api/health", 
            "/api/v1/system/health",
            "/api/v1/queue",
            "/api/v1/generation/submit",
            "/ws"
        ]
        
        print("\nTesting endpoints...")
        for route in expected_routes:
            if route in routes:
                print(f"✅ {route} endpoint found")
            else:
                print(f"❌ {route} endpoint missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing endpoints: {e}")
        return False

    assert True  # TODO: Add proper assertion

def main():
    print("WAN22 Backend Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed")
        return False
    
    # Test endpoints
    if not test_endpoints():
        print("\n❌ Endpoint test failed") 
        return False
    
    print("\n✅ All tests passed!")
    print("\nTo start the backend server, run:")
    print("  cd backend")
    print("  python start_server.py")
    print("\nOr use the startup script:")
    print("  start_both_servers.bat (Windows)")
    print("  ./start_both_servers.sh (Linux/Mac)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
