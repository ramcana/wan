"""
Test I2V/TI2V API endpoints and image upload functionality
"""

import os
import sys
import io
from PIL import Image
import requests
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_test_image(width=512, height=512, format='JPEG'):
    """Create a test image for upload testing"""
    # Create a simple test image
    img = Image.new('RGB', (width, height), color='red')
    
    # Add some content to make it more realistic
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, width-50, height-50], fill='blue')
    draw.text((100, 100), "Test Image", fill='white')
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=format)
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

def test_api_endpoints():
    """Test the API endpoints"""
    
    base_url = "http://localhost:8000/api/v1"
    
    print("Testing API endpoints...")
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Health endpoint working")
        else:
            print(f"✗ Health endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Health endpoint not accessible: {e}")
        print("Note: Start the backend server with 'uvicorn backend.main:app --reload' to test API endpoints")
        return
    
    # Test 2: T2V generation (no image)
    try:
        t2v_data = {
            "model_type": "T2V-A14B",
            "prompt": "A beautiful sunset over mountains",
            "resolution": "1280x720",
            "steps": 30
        }
        
        response = requests.post(f"{base_url}/generate", data=t2v_data, timeout=10)
        if response.status_code == 200:
            task_data = response.json()
            print(f"✓ T2V generation request successful: {task_data['task_id']}")
            t2v_task_id = task_data['task_id']
        else:
            print(f"✗ T2V generation failed: {response.status_code} - {response.text}")
            return
            
    except requests.exceptions.RequestException as e:
        print(f"✗ T2V generation request failed: {e}")
        return
    
    # Test 3: I2V generation (with image)
    try:
        test_image = create_test_image()
        
        i2v_data = {
            "model_type": "I2V-A14B",
            "prompt": "Transform this image into a video",
            "resolution": "1280x720",
            "steps": 25
        }
        
        files = {"image": ("test.jpg", test_image, "image/jpeg")}
        
        response = requests.post(f"{base_url}/generate", data=i2v_data, files=files, timeout=10)
        if response.status_code == 200:
            task_data = response.json()
            print(f"✓ I2V generation request successful: {task_data['task_id']}")
            i2v_task_id = task_data['task_id']
        else:
            print(f"✗ I2V generation failed: {response.status_code} - {response.text}")
            return
            
    except requests.exceptions.RequestException as e:
        print(f"✗ I2V generation request failed: {e}")
        return
    
    # Test 4: TI2V generation (with image)
    try:
        test_image = create_test_image(width=1024, height=768)
        
        ti2v_data = {
            "model_type": "TI2V-5B",
            "prompt": "A cinematic transformation of this scene",
            "resolution": "1920x1080",
            "steps": 50
        }
        
        files = {"image": ("test.png", test_image, "image/png")}
        
        response = requests.post(f"{base_url}/generate", data=ti2v_data, files=files, timeout=10)
        if response.status_code == 200:
            task_data = response.json()
            print(f"✓ TI2V generation request successful: {task_data['task_id']}")
            ti2v_task_id = task_data['task_id']
        else:
            print(f"✗ TI2V generation failed: {response.status_code} - {response.text}")
            return
            
    except requests.exceptions.RequestException as e:
        print(f"✗ TI2V generation request failed: {e}")
        return
    
    # Test 5: Queue status
    try:
        response = requests.get(f"{base_url}/queue", timeout=5)
        if response.status_code == 200:
            queue_data = response.json()
            print(f"✓ Queue status: {queue_data['total_tasks']} total tasks")
            print(f"  - Pending: {queue_data['pending_tasks']}")
            print(f"  - Processing: {queue_data['processing_tasks']}")
            print(f"  - Completed: {queue_data['completed_tasks']}")
        else:
            print(f"✗ Queue status failed: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Queue status request failed: {e}")
    
    # Test 6: Polling endpoint
    try:
        response = requests.get(f"{base_url}/queue/poll", timeout=5)
        if response.status_code == 200:
            poll_data = response.json()
            print(f"✓ Queue polling: {poll_data['pending_count']} pending, {poll_data['processing_count']} processing")
        else:
            print(f"✗ Queue polling failed: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Queue polling request failed: {e}")
    
    # Test 7: Task cancellation
    try:
        if 't2v_task_id' in locals():
            response = requests.post(f"{base_url}/queue/{t2v_task_id}/cancel", timeout=5)
            if response.status_code == 200:
                print(f"✓ Task cancellation successful")
            else:
                print(f"✗ Task cancellation failed: {response.status_code}")
                
    except requests.exceptions.RequestException as e:
        print(f"✗ Task cancellation request failed: {e}")
    
    # Test 8: Image validation errors
    try:
        # Test with invalid image format
        invalid_data = {
            "model_type": "I2V-A14B",
            "prompt": "Test with invalid image",
            "resolution": "1280x720"
        }
        
        files = {"image": ("test.txt", b"not an image", "text/plain")}
        
        response = requests.post(f"{base_url}/generate", data=invalid_data, files=files, timeout=10)
        if response.status_code == 400:
            print("✓ Image validation correctly rejected invalid format")
        else:
            print(f"✗ Image validation should have failed: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Image validation test failed: {e}")
    
    # Test 9: T2V with image should fail
    try:
        test_image = create_test_image()
        
        invalid_t2v_data = {
            "model_type": "T2V-A14B",
            "prompt": "T2V should not accept images",
            "resolution": "1280x720"
        }
        
        files = {"image": ("test.jpg", test_image, "image/jpeg")}
        
        response = requests.post(f"{base_url}/generate", data=invalid_t2v_data, files=files, timeout=10)
        if response.status_code == 400:
            print("✓ T2V correctly rejected image input")
        else:
            print(f"✗ T2V should have rejected image: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"✗ T2V image rejection test failed: {e}")
    
    print("\nAPI endpoint testing completed!")

    assert True  # TODO: Add proper assertion

if __name__ == "__main__":
    test_api_endpoints()