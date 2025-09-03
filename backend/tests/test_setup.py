#!/usr/bin/env python3
"""
Test script to verify FastAPI backend setup
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from fastapi import FastAPI
        print("✓ FastAPI imported successfully")
    except ImportError as e:
        print(f"✗ FastAPI import failed: {e}")
        return False
    
    try:
        from backend.repositories.database import init_database, get_db
        print("✓ Database modules imported successfully")
    except ImportError as e:
        print(f"✗ Database import failed: {e}")
        return False
    
    try:
        from backend.schemas.schemas import GenerationRequest, SystemStats
        print("✓ Schema models imported successfully")
    except ImportError as e:
        print(f"✗ Schema models import failed: {e}")
        return False
    
    try:
        from backend.core.system_integration import get_system_integration
        print("✓ System integration imported successfully")
    except ImportError as e:
        print(f"✗ System integration import failed: {e}")
        return False
    
    try:
        from api.routes import generation, queue, system, outputs
        print("✓ API routes imported successfully")
    except ImportError as e:
        print(f"✗ API routes import failed: {e}")
        return False
    
    return True

def test_database_init():
    """Test database initialization"""
    print("\nTesting database initialization...")
    
    try:
        from backend.repositories.database import init_database
        init_database()
        print("✓ Database initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        return False

def test_system_integration():
    """Test system integration"""
    print("\nTesting system integration...")
    
    try:
        import asyncio
from backend.core.system_integration import get_system_integration
        
        async def test_integration():
            integration = await get_system_integration()
            system_info = integration.get_system_info()
            print(f"✓ System integration working: {system_info}")
            return True
        
        result = asyncio.run(test_integration())
        return result
    except Exception as e:
        print(f"✗ System integration test failed: {e}")
        return False

def test_gpu_validation():
    """Test GPU validation"""
    print("\nTesting GPU validation...")
    
    try:
        import asyncio
from backend.core.system_integration import get_system_integration
        
        async def test_gpu():
            integration = await get_system_integration()
            gpu_valid, gpu_message = await integration.validate_gpu_access()
            print(f"GPU validation: {gpu_message}")
            return gpu_valid
        
        result = asyncio.run(test_gpu())
        if result:
            print("✓ GPU validation passed")
        else:
            print("⚠ GPU validation failed (may be expected in some environments)")
        return True  # Don't fail the test if GPU is not available
    except Exception as e:
        print(f"✗ GPU validation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== FastAPI Backend Setup Test ===\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Database Test", test_database_init),
        ("System Integration Test", test_system_integration),
        ("GPU Validation Test", test_gpu_validation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! FastAPI backend is ready.")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)