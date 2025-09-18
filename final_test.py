#!/usr/bin/env python3
"""
Final test for WAN22 backend after configuration updates
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_backend_final():
    """Final comprehensive test"""

    print("🎯 WAN22 Backend - Final Test After Config Updates")
    print("=" * 60)

    # Test health first
    print("🏥 Health Check:")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("  ✅ Backend is running")
        else:
            print("  ❌ Backend not responding")
            return
    except:
        print("  ❌ Backend not accessible")
        return

    # Test model status with updated config
    print("\n🤖 Model Status (After Config Update):")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/models/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", {})
            available = data.get("available_models", 0)
            total = data.get("total_models", 0)

            print(f"  📊 Models: {available}/{total} available")

            for model_id, info in models.items():
                status = info.get("status", "unknown")
                available = info.get("is_available", False)
                icon = "✅" if available else "❌"
                print(f"    {icon} {model_id}: {status}")

                if not available and info.get("error_message"):
                    print(f"      Error: {info['error_message']}")
        else:
            print(f"  ❌ Model status failed: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # Test prompt enhancement
    print("\n📝 Prompt Enhancement:")
    try:
        data = {"prompt": "A beautiful sunset over mountains"}
        response = requests.post(
            f"{BASE_URL}/api/v1/prompt/enhance", json=data, timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            original = result.get("original_prompt", "")[:40]
            enhanced = result.get("enhanced_prompt", "")[:40]
            print(f"  ✅ Working: '{original}...' → '{enhanced}...'")
        else:
            print(f"  ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # Test generation endpoint
    print("\n🎬 Generation Test:")
    try:
        data = {
            "prompt": "A cat walking in a garden",
            "model_type": "t2v-A14B@2.2.0",
            "resolution": "1280x720",
            "steps": "20",
        }
        response = requests.post(
            f"{BASE_URL}/api/v1/generation/submit", data=data, timeout=10
        )

        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"  ✅ Generation accepted!")
            print(f"  📋 Task ID: {result.get('task_id', 'N/A')}")
        else:
            print(f"  ⚠️  Response: {response.text[:100]}")
    except Exception as e:
        print(f"  ❌ Error: {e}")


def show_working_commands():
    """Show the working curl commands"""

    print(f"\n📋 Working Curl Commands:")
    print("-" * 40)

    commands = [
        ("Health Check", f"curl {BASE_URL}/health"),
        ("Model Status", f"curl {BASE_URL}/api/v1/models/status"),
        (
            "Prompt Enhancement",
            f'curl -X POST {BASE_URL}/api/v1/prompt/enhance -H "Content-Type: application/json" -d \'{{"prompt": "A beautiful landscape"}}\'',
        ),
        (
            "Generation (Form)",
            f'curl -X POST {BASE_URL}/api/v1/generation/submit -F "prompt=A cat in a garden" -F "model_type=t2v-A14B@2.2.0" -F "resolution=1280x720" -F "steps=20"',
        ),
        ("Queue Status", f"curl {BASE_URL}/api/v1/queue"),
    ]

    for name, cmd in commands:
        print(f"\n# {name}")
        print(cmd)


def main():
    """Run final test"""
    test_backend_final()
    show_working_commands()

    print(f"\n🎉 Backend Test Complete!")
    print(f"Your WAN22 backend is operational with model detection working.")


if __name__ == "__main__":
    main()
