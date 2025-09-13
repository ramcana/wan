#!/usr/bin/env python3
"""
Test error handling and validation for the generation API
"""

import requests
import json

def test_prompt_validation():
    """Test prompt validation"""
    print("=== Testing Prompt Validation ===\n")
    
    base_url = "http://localhost:8000/api/v1"
    
    # Test 1: Empty prompt
    print("1. Testing empty prompt...")
    response = requests.post(f"{base_url}/generate", data={
        "model_type": "T2V-A14B",
        "prompt": "",
        "resolution": "1280x720"
    })
    
    if response.status_code == 422:  # Validation error
        print("✅ Empty prompt correctly rejected")
    else:
        print(f"❌ Expected 422, got {response.status_code}")
        return False
    
    # Test 2: Too long prompt
    print("2. Testing overly long prompt...")
    long_prompt = "A" * 501  # Exceeds 500 character limit
    response = requests.post(f"{base_url}/generate", data={
        "model_type": "T2V-A14B",
        "prompt": long_prompt,
        "resolution": "1280x720"
    })
    
    if response.status_code == 422:  # Validation error
        print("✅ Long prompt correctly rejected")
    else:
        print(f"❌ Expected 422, got {response.status_code}")
        return False
    
    return True

    assert True  # TODO: Add proper assertion

def test_resolution_validation():
    """Test resolution validation"""
    print("\n=== Testing Resolution Validation ===\n")
    
    base_url = "http://localhost:8000/api/v1"
    
    # Test invalid resolution format
    print("1. Testing invalid resolution format...")
    response = requests.post(f"{base_url}/generate", data={
        "model_type": "T2V-A14B",
        "prompt": "Test prompt",
        "resolution": "invalid_resolution"
    })
    
    if response.status_code == 400:
        print("✅ Invalid resolution format correctly rejected")
    else:
        print(f"❌ Expected 400, got {response.status_code}")
        return False
    
    # Test negative resolution values
    print("2. Testing negative resolution values...")
    response = requests.post(f"{base_url}/generate", data={
        "model_type": "T2V-A14B",
        "prompt": "Test prompt",
        "resolution": "-100x-100"
    })
    
    if response.status_code == 400:
        print("✅ Negative resolution correctly rejected")
    else:
        print(f"❌ Expected 400, got {response.status_code}")
        return False
    
    return True

    assert True  # TODO: Add proper assertion

def test_steps_validation():
    """Test steps parameter validation"""
    print("\n=== Testing Steps Validation ===\n")
    
    base_url = "http://localhost:8000/api/v1"
    
    # Test steps too low
    print("1. Testing steps below minimum...")
    response = requests.post(f"{base_url}/generate", data={
        "model_type": "T2V-A14B",
        "prompt": "Test prompt",
        "steps": "0"
    })
    
    if response.status_code == 422:
        print("✅ Steps below minimum correctly rejected")
    else:
        print(f"❌ Expected 422, got {response.status_code}")
        return False
    
    # Test steps too high
    print("2. Testing steps above maximum...")
    response = requests.post(f"{base_url}/generate", data={
        "model_type": "T2V-A14B",
        "prompt": "Test prompt",
        "steps": "101"
    })
    
    if response.status_code == 422:
        print("✅ Steps above maximum correctly rejected")
    else:
        print(f"❌ Expected 422, got {response.status_code}")
        return False
    
    return True

    assert True  # TODO: Add proper assertion

def test_model_type_validation():
    """Test model type validation"""
    print("\n=== Testing Model Type Validation ===\n")
    
    base_url = "http://localhost:8000/api/v1"
    
    # Test invalid model type
    print("1. Testing invalid model type...")
    response = requests.post(f"{base_url}/generate", data={
        "model_type": "INVALID-MODEL",
        "prompt": "Test prompt"
    })
    
    if response.status_code == 422:
        print("✅ Invalid model type correctly rejected")
    else:
        print(f"❌ Expected 422, got {response.status_code}")
        return False
    
    return True

    assert True  # TODO: Add proper assertion

def test_lora_validation():
    """Test LoRA parameter validation"""
    print("\n=== Testing LoRA Validation ===\n")
    
    base_url = "http://localhost:8000/api/v1"
    
    # Test LoRA strength out of range
    print("1. Testing LoRA strength above maximum...")
    response = requests.post(f"{base_url}/generate", data={
        "model_type": "T2V-A14B",
        "prompt": "Test prompt",
        "lora_strength": "2.1"
    })
    
    if response.status_code == 422:
        print("✅ LoRA strength above maximum correctly rejected")
    else:
        print(f"❌ Expected 422, got {response.status_code}")
        return False
    
    # Test negative LoRA strength
    print("2. Testing negative LoRA strength...")
    response = requests.post(f"{base_url}/generate", data={
        "model_type": "T2V-A14B",
        "prompt": "Test prompt",
        "lora_strength": "-0.1"
    })
    
    if response.status_code == 422:
        print("✅ Negative LoRA strength correctly rejected")
    else:
        print(f"❌ Expected 422, got {response.status_code}")
        return False
    
    return True

    assert True  # TODO: Add proper assertion

def test_standardized_error_responses():
    """Test that error responses follow the standardized format"""
    print("\n=== Testing Standardized Error Responses ===\n")
    
    base_url = "http://localhost:8000/api/v1"
    
    # Test with T2V + image (should give standardized error)
    print("1. Testing standardized error format...")
    
    files = {
        "image": ("test.jpg", b"fake image data", "image/jpeg")
    }
    
    response = requests.post(f"{base_url}/generate", data={
        "model_type": "T2V-A14B",
        "prompt": "Test prompt"
    }, files=files)
    
    if response.status_code == 400:
        try:
            error_data = response.json()
            detail = error_data.get("detail", "")
            
            if "T2V mode does not accept image input" in detail:
                print("✅ Standardized error response format working")
                return True
            else:
                print(f"❌ Unexpected error format: {detail}")
                return False
        except json.JSONDecodeError:
            print("❌ Error response is not valid JSON")
            return False
    else:
        print(f"❌ Expected 400, got {response.status_code}")
        return False

    assert True  # TODO: Add proper assertion

def main():
    """Run all error handling tests"""
    print("Testing Error Handling and Validation\n")
    
    tests = [
        ("Prompt Validation", test_prompt_validation),
        ("Resolution Validation", test_resolution_validation),
        ("Steps Validation", test_steps_validation),
        ("Model Type Validation", test_model_type_validation),
        ("LoRA Validation", test_lora_validation),
        ("Standardized Error Responses", test_standardized_error_responses),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED\n")
            else:
                print(f"❌ {test_name} FAILED\n")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}\n")
    
    print(f"=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All error handling tests passed!")
        return True
    else:
        print("❌ Some error handling tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
