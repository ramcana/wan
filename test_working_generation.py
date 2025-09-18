#!/usr/bin/env python3
"""
Working generation test - test the endpoints that should work
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_working_endpoints():
    """Test endpoints that should work regardless of model status"""
    
    print("🚀 Testing Working Endpoints")
    print("=" * 50)
    
    tests = [
        # Health checks (should always work)
        ("Health Check", "GET", f"{BASE_URL}/health"),
        ("System Health", "GET", f"{BASE_URL}/api/v1/system/health"),
        
        # Prompt enhancement (should work)
        ("Prompt Enhancement", "POST", f"{BASE_URL}/api/v1/prompt/enhance", {
            "prompt": "A beautiful mountain landscape at sunset"
        }),
        
        # Queue status (should work)
        ("Queue Status", "GET", f"{BASE_URL}/api/v1/queue"),
        
        # Model status (should show models as detected but missing files)
        ("Model Status", "GET", f"{BASE_URL}/api/v1/models/status"),
    ]
    
    for name, method, url, *args in tests:
        data = args[0] if args else None
        print(f"\n🧪 {name}")
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=10)
            else:
                response = requests.post(url, json=data, timeout=10)
            
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if name == "Model Status":
                    # Show model status details
                    models = result.get("models", {})
                    print(f"  ✅ Found {len(models)} models:")
                    for model_id, info in models.items():
                        status = info.get("status", "unknown")
                        available = info.get("is_available", False)
                        print(f"    • {model_id}: {status} (available: {available})")
                else:
                    print(f"  ✅ Success: {json.dumps(result, indent=2)[:200]}...")
            else:
                print(f"  ❌ Failed: {response.text[:200]}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_generation_with_mock():
    """Test generation endpoint - should work in mock mode even with missing models"""
    
    print(f"\n🎬 Testing Generation (Mock Mode)")
    print("=" * 30)
    
    # Test form data (this should work based on the backend code)
    print("📝 Form Data Request:")
    try:
        data = {
            "prompt": "A cat walking through a beautiful garden",
            "model_type": "t2v-A14B@2.2.0",
            "resolution": "1280x720",
            "steps": "25"
        }
        
        response = requests.post(f"{BASE_URL}/api/v1/generation/submit", data=data, timeout=10)
        print(f"  Status: {response.status_code}")
        
        if response.headers.get('content-type', '').startswith('application/json'):
            result = response.json()
            print(f"  Response: {json.dumps(result, indent=2)}")
            
            if response.status_code == 200:
                print("  ✅ Generation request accepted!")
                task_id = result.get("task_id")
                if task_id:
                    print(f"  📋 Task ID: {task_id}")
            else:
                print("  ⚠️  Generation failed, but endpoint is responding")
        else:
            print(f"  Response: {response.text}")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

def main():
    print("🎯 WAN22 Backend Working Test Suite")
    print("Testing what actually works with current setup")
    print("=" * 60)
    
    test_working_endpoints()
    test_generation_with_mock()
    
    print("\n" + "=" * 60)
    print("📋 Manual curl commands that should work:")
    print()
    print("# Health check")
    print(f"curl {BASE_URL}/health")
    print()
    print("# Model status")
    print(f"curl {BASE_URL}/api/v1/models/status")
    print()
    print("# Prompt enhancement")
    print(f'curl -X POST {BASE_URL}/api/v1/prompt/enhance \\')
    print(f'  -H "Content-Type: application/json" \\')
    print(f'  -d \'{{"prompt": "A beautiful landscape"}}\'')
    print()
    print("# Generation (form data)")
    print(f'curl -X POST {BASE_URL}/api/v1/generation/submit \\')
    print(f'  -F "prompt=A cat in a garden" \\')
    print(f'  -F "model_type=t2v-A14B@2.2.0" \\')
    print(f'  -F "resolution=1280x720" \\')
    print(f'  -F "steps=25"')

if __name__ == "__main__":
    main()