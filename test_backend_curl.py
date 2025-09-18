#!/usr/bin/env python3
"""
Comprehensive curl test script for WAN22 backend
Tests health endpoints, model recognition, and generation endpoints
"""

import requests
import json
import time
import sys
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def test_endpoint(name: str, method: str, url: str, data: Dict[Any, Any] = None, 
                 headers: Dict[str, str] = None) -> bool:
    """Test a single endpoint and return success status"""
    print(f"\n🧪 Testing {name}")
    print(f"   {method} {url}")
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=10)
        else:
            print(f"   ❌ Unsupported method: {method}")
            return False
            
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"   ✅ Success: {json.dumps(result, indent=2)[:200]}...")
                return True
            except json.JSONDecodeError:
                print(f"   ✅ Success: {response.text[:100]}...")
                return True
        else:
            print(f"   ❌ Failed: {response.text[:200]}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Connection error: {e}")
        return False

def main():
    """Run comprehensive backend tests"""
    print("🚀 WAN22 Backend Comprehensive Test Suite")
    print(f"Testing backend at: {BASE_URL}")
    print("=" * 60)
    
    tests = [
        # Health check endpoints
        ("Basic Health Check", "GET", f"{BASE_URL}/health"),
        ("API Health Check", "GET", f"{BASE_URL}/api/health"),
        ("System Health Check", "GET", f"{BASE_URL}/api/v1/system/health"),
        
        # Model-related endpoints (if available)
        ("Model Registry", "GET", f"{BASE_URL}/api/v1/models"),
        ("Model Status", "GET", f"{BASE_URL}/api/v1/models/status"),
        
        # Prompt enhancement
        ("Prompt Enhancement", "POST", f"{BASE_URL}/api/v1/prompt/enhance", {
            "prompt": "A beautiful sunset over mountains"
        }),
        
        # Generation endpoints
        ("Text-to-Video Generation", "POST", f"{BASE_URL}/api/v1/generation/submit", {
            "prompt": "A cat walking in a garden",
            "model_type": "t2v-A14B@2.2.0",
            "resolution": "1280x720",
            "steps": 20
        }),
        
        ("Image-to-Video Generation", "POST", f"{BASE_URL}/api/v1/generation/submit", {
            "prompt": "Animate this image with gentle movement",
            "model_type": "i2v-A14B@2.2.0",
            "resolution": "1280x720",
            "steps": 20
        }),
        
        # Queue management
        ("Generation Queue", "GET", f"{BASE_URL}/api/v1/queue"),
        
        # System info
        ("System Info", "GET", f"{BASE_URL}/api/v1/system/info"),
        ("Hardware Info", "GET", f"{BASE_URL}/api/v1/system/hardware"),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, method, url, *args in tests:
        data = args[0] if args else None
        success = test_endpoint(test_name, method, url, data)
        if success:
            passed += 1
        time.sleep(0.5)  # Brief pause between tests
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Backend is fully operational.")
    elif passed >= total * 0.7:
        print("⚠️  Most tests passed. Some endpoints may not be implemented yet.")
    else:
        print("❌ Many tests failed. Check backend configuration and logs.")
    
    # Additional model-specific tests
    print("\n🔍 Testing Model Recognition...")
    
    # Test model configuration loading
    model_test = test_endpoint(
        "Model Configuration Test", 
        "GET", 
        f"{BASE_URL}/api/v1/models/t2v-A14B@2.2.0"
    )
    
    if not model_test:
        print("   ℹ️  Model-specific endpoints may not be implemented yet")
    
    print("\n📋 Manual curl commands for testing:")
    print(f"curl -X GET {BASE_URL}/health")
    print(f"curl -X GET {BASE_URL}/api/v1/system/health")
    print(f'curl -X POST {BASE_URL}/api/v1/prompt/enhance \\')
    print(f'  -H "Content-Type: application/json" \\')
    print(f'  -d \'{{"prompt": "A beautiful landscape"}}\'')
    print(f'curl -X POST {BASE_URL}/api/v1/generation/submit \\')
    print(f'  -H "Content-Type: application/json" \\')
    print(f'  -d \'{{"prompt": "A cat in a garden", "model_type": "t2v-A14B@2.2.0"}}\'')

if __name__ == "__main__":
    main()