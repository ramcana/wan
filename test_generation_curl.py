#!/usr/bin/env python3
"""
Focused curl test for WAN22 generation endpoints
Tests the actual generation submission with proper format
"""

import requests
import json
import time

BASE_URL = "http://localhost:9000"

def test_generation_json():
    """Test generation endpoint with JSON payload"""
    print("🧪 Testing Text-to-Video Generation (JSON)")
    
    payload = {
        "prompt": "A majestic eagle soaring through mountain peaks at sunset",
        "model_type": "t2v-A14B@2.2.0",
        "resolution": "1280x720",
        "steps": 25
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/generation/submit",
            json=payload,
            headers=headers,
            timeout=10
        )
        
        print(f"Status: {response.status_code}")
        if response.headers.get('content-type', '').startswith('application/json'):
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Response: {response.text}")
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_generation_form():
    """Test generation endpoint with form data"""
    print("\n🧪 Testing Text-to-Video Generation (Form Data)")
    
    data = {
        "prompt": "A peaceful lake with swans swimming gracefully",
        "model_type": "t2v-A14B@2.2.0", 
        "resolution": "1280x720",
        "steps": "30"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/generation/submit",
            data=data,
            timeout=10
        )
        
        print(f"Status: {response.status_code}")
        if response.headers.get('content-type', '').startswith('application/json'):
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Response: {response.text}")
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_status():
    """Test model status to see what models are recognized"""
    print("\n🔍 Checking Model Status")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/models/status", timeout=10)
        
        if response.status_code == 200:
            models = response.json().get("models", {})
            print(f"Found {len(models)} models:")
            
            for model_id, info in models.items():
                status = info.get("status", "unknown")
                available = info.get("is_available", False)
                print(f"  • {model_id}: {status} (available: {available})")
                
            return True
        else:
            print(f"Failed to get model status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("🚀 WAN22 Generation Endpoint Test")
    print("=" * 50)
    
    # Test model recognition first
    test_model_status()
    
    # Test generation endpoints
    json_success = test_generation_json()
    form_success = test_generation_form()
    
    print("\n" + "=" * 50)
    print("📊 Results:")
    print(f"  JSON Generation: {'✅ Pass' if json_success else '❌ Fail'}")
    print(f"  Form Generation: {'✅ Pass' if form_success else '❌ Fail'}")
    
    if json_success or form_success:
        print("\n🎉 Generation endpoints are working!")
        print("\n📋 Manual curl commands:")
        print(f"# JSON request:")
        print(f'curl -X POST {BASE_URL}/api/v1/generation/submit \\')
        print(f'  -H "Content-Type: application/json" \\')
        print(f'  -d \'{{"prompt": "A cat playing in a garden", "model_type": "t2v-A14B@2.2.0", "resolution": "1280x720", "steps": 25}}\'')
        
        print(f"\n# Form data request:")
        print(f'curl -X POST {BASE_URL}/api/v1/generation/submit \\')
        print(f'  -F "prompt=A dog running on the beach" \\')
        print(f'  -F "model_type=t2v-A14B@2.2.0" \\')
        print(f'  -F "resolution=1280x720" \\')
        print(f'  -F "steps=25"')
    else:
        print("\n❌ Generation endpoints need debugging")

if __name__ == "__main__":
    main()