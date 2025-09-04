#!/usr/bin/env python3
"""
Test T2V generation API endpoint
"""

import requests
import time
import json

def test_t2v_generation():
    """Test T2V generation endpoint"""
    base_url = "http://localhost:8000/api/v1"
    
    print("=== Testing T2V Generation API ===\n")
    
    # Test 1: Submit T2V generation request
    print("1. Testing T2V generation request submission...")
    
    generation_data = {
        "model_type": "T2V-A14B",
        "prompt": "A beautiful sunset over mountains with flowing clouds",
        "resolution": "1280x720",
        "steps": 20  # Reduced for faster testing
    }
    
    try:
        response = requests.post(f"{base_url}/generate", data=generation_data)
        
        if response.status_code == 200:
            result = response.json()
            task_id = result["task_id"]
            print(f"✅ Generation request submitted successfully")
            print(f"   Task ID: {task_id}")
            print(f"   Status: {result['status']}")
            print(f"   Message: {result['message']}")
            print(f"   Estimated time: {result.get('estimated_time_minutes', 'N/A')} minutes")
        else:
            print(f"❌ Generation request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error submitting generation request: {e}")
        return False
    
    # Test 2: Check task status
    print(f"\n2. Checking task status...")
    
    try:
        response = requests.get(f"{base_url}/generate/{task_id}")
        
        if response.status_code == 200:
            task_info = response.json()
            print(f"✅ Task status retrieved successfully")
            print(f"   Status: {task_info['status']}")
            print(f"   Progress: {task_info['progress']}%")
            print(f"   Created: {task_info['created_at']}")
        else:
            print(f"❌ Failed to get task status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting task status: {e}")
    
    # Test 3: Monitor queue
    print(f"\n3. Monitoring queue status...")
    
    for i in range(10):  # Monitor for up to 10 iterations
        try:
            response = requests.get(f"{base_url}/queue")
            
            if response.status_code == 200:
                queue_status = response.json()
                print(f"   Queue check {i+1}: {queue_status['pending_tasks']} pending, "
                      f"{queue_status['processing_tasks']} processing, "
                      f"{queue_status['completed_tasks']} completed")
                
                # Check if our task is completed
                for task in queue_status['tasks']:
                    if task['id'] == task_id:
                        if task['status'] == 'completed':
                            print(f"✅ Task completed successfully!")
                            print(f"   Output path: {task.get('output_path', 'N/A')}")
                            return True
                        elif task['status'] == 'failed':
                            print(f"❌ Task failed: {task.get('error_message', 'Unknown error')}")
                            return False
                        else:
                            print(f"   Task status: {task['status']} ({task['progress']}%)")
                
            time.sleep(2)  # Wait 2 seconds between checks
            
        except Exception as e:
            print(f"❌ Error monitoring queue: {e}")
            break
    
    print("⚠️  Task monitoring timed out")
    return False

    assert True  # TODO: Add proper assertion

def test_t2v_validation():
    """Test T2V validation (should reject image input)"""
    print("\n=== Testing T2V Validation ===\n")
    
    base_url = "http://localhost:8000/api/v1"
    
    # Test: T2V with image should fail
    print("Testing T2V with image input (should fail)...")
    
    generation_data = {
        "model_type": "T2V-A14B",
        "prompt": "A test prompt",
        "resolution": "1280x720",
        "steps": 20
    }
    
    # Create a dummy image file
    files = {
        "image": ("test.jpg", b"fake image data", "image/jpeg")
    }
    
    try:
        response = requests.post(f"{base_url}/generate", data=generation_data, files=files)
        
        if response.status_code == 400:
            error_detail = response.json()["detail"]
            if "T2V mode does not accept image input" in error_detail:
                print("✅ T2V validation working correctly - rejected image input")
                return True
            else:
                print(f"❌ Unexpected error message: {error_detail}")
                return False
        else:
            print(f"❌ Expected 400 error, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing T2V validation: {e}")
        return False

    assert True  # TODO: Add proper assertion

def main():
    """Run all tests"""
    print("Testing T2V Generation API\n")
    
    # Test basic functionality
    generation_success = test_t2v_generation()
    
    # Test validation
    validation_success = test_t2v_validation()
    
    print(f"\n=== Test Results ===")
    print(f"T2V Generation: {'✅ PASS' if generation_success else '❌ FAIL'}")
    print(f"T2V Validation: {'✅ PASS' if validation_success else '❌ FAIL'}")
    
    if generation_success and validation_success:
        print("🎉 All T2V tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)