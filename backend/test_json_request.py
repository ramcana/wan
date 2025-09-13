#!/usr/bin/env python3
"""
Test JSON request handling for generation endpoint
"""

import asyncio
import json
import sys
sys.path.append('.')

async def test_json_request():
    print("🧪 Testing JSON Request Handling")
    print("=" * 50)
    
    try:
        import httpx

        # Test data that matches what the frontend should send
        test_data = {
            "prompt": "A cat walking in the park, cinematic lighting, highly detailed",
            "model_type": "t2v-A14B",
            "resolution": "1280x720",
            "steps": 50,
            "lora_path": "",
            "lora_strength": 1.0
        }
        
        print(f"📤 Sending JSON request:")
        print(f"   Prompt: {test_data['prompt']}")
        print(f"   Model: {test_data['model_type']}")
        print(f"   Resolution: {test_data['resolution']}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/v1/generation/submit",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=30.0
            )
            
            print(f"\n📥 Response:")
            print(f"   Status: {response.status_code}")
            print(f"   Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success: {result}")
            else:
                print(f"   ❌ Error: {response.text}")
                
    except ImportError:
        print("❌ httpx not available, install with: pip install httpx")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_json_request())
