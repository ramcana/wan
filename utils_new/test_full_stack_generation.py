#!/usr/bin/env python3
"""
Full Stack Generation Test
Tests the complete generation workflow from React frontend to FastAPI backend
"""

import requests
import json
import time
import sys
from pathlib import Path

class GenerationTester:
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:3000"
    
    def check_services(self):
        """Check if both services are running"""
        print("🔍 Checking services...")
        
        # Check backend
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Backend is running")
            else:
                print(f"❌ Backend returned {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Backend is not accessible: {e}")
            return False
        
        # Check frontend (just check if port is open)
        try:
            response = requests.get(self.frontend_url, timeout=5)
            print("✅ Frontend is running")
        except Exception as e:
            print(f"❌ Frontend is not accessible: {e}")
            return False
        
        return True
    
    def test_system_info(self):
        """Test system information endpoint"""
        print("\n🖥️  Testing system information...")
        
        try:
            response = requests.get(f"{self.backend_url}/api/v1/system/info", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print("✅ System info retrieved successfully")
                print(f"   GPU Available: {data.get('gpu_available', 'Unknown')}")
                print(f"   GPU Name: {data.get('gpu_name', 'Unknown')}")
                print(f"   VRAM: {data.get('vram_total', 'Unknown')} MB")
                return True
            else:
                print(f"❌ System info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ System info error: {e}")
            return False
    
    def test_model_list(self):
        """Test model listing endpoint"""
        print("\n📋 Testing model listing...")
        
        try:
            response = requests.get(f"{self.backend_url}/api/v1/generation/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                print(f"✅ Found {len(models)} available models")
                for model in models[:3]:  # Show first 3 models
                    print(f"   - {model.get('name', 'Unknown')}")
                if len(models) > 3:
                    print(f"   ... and {len(models) - 3} more")
                return True
            else:
                print(f"❌ Model listing failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Model listing error: {e}")
            return False
    
    def test_queue_status(self):
        """Test queue status endpoint"""
        print("\n📊 Testing queue status...")
        
        try:
            response = requests.get(f"{self.backend_url}/api/v1/queue/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print("✅ Queue status retrieved successfully")
                print(f"   Active tasks: {data.get('active_tasks', 0)}")
                print(f"   Pending tasks: {data.get('pending_tasks', 0)}")
                print(f"   Queue size: {data.get('queue_size', 0)}")
                return True
            else:
                print(f"❌ Queue status failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Queue status error: {e}")
            return False
    
    def test_simple_generation(self):
        """Test a simple text-to-video generation"""
        print("\n🎬 Testing simple generation...")
        
        # Simple generation request
        generation_data = {
            "prompt": "A beautiful sunset over the ocean",
            "model_name": "wan22_t2v_5b",  # Default model
            "num_frames": 16,
            "width": 512,
            "height": 512,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "seed": 42
        }
        
        try:
            print("📤 Submitting generation request...")
            response = requests.post(
                f"{self.backend_url}/api/v1/generation/generate",
                json=generation_data,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                task_id = data.get('task_id')
                print(f"✅ Generation task submitted: {task_id}")
                
                # Monitor task progress
                return self.monitor_task(task_id)
            else:
                print(f"❌ Generation failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('detail', 'Unknown error')}")
                except:
                    print(f"   Raw response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return False
    
    def monitor_task(self, task_id, timeout=300):
        """Monitor a generation task"""
        print(f"⏳ Monitoring task {task_id}...")
        
        start_time = time.time()
        last_progress = -1
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{self.backend_url}/api/v1/queue/task/{task_id}",
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('status', 'unknown')
                    progress = data.get('progress', 0)
                    
                    # Show progress updates
                    if progress != last_progress:
                        print(f"   Progress: {progress}% - Status: {status}")
                        last_progress = progress
                    
                    if status == 'completed':
                        output_path = data.get('output_path')
                        print(f"✅ Generation completed!")
                        print(f"   Output: {output_path}")
                        return True
                    elif status == 'failed':
                        error = data.get('error', 'Unknown error')
                        print(f"❌ Generation failed: {error}")
                        return False
                    elif status in ['pending', 'running']:
                        time.sleep(2)  # Wait before next check
                    else:
                        print(f"⚠️  Unknown status: {status}")
                        time.sleep(2)
                else:
                    print(f"❌ Task status check failed: {response.status_code}")
                    return False
                    
            except Exception as e:
                print(f"❌ Task monitoring error: {e}")
                return False
        
        print(f"⏰ Task monitoring timed out after {timeout} seconds")
        return False
    
    def test_outputs_list(self):
        """Test outputs listing endpoint"""
        print("\n📁 Testing outputs listing...")
        
        try:
            response = requests.get(f"{self.backend_url}/api/v1/outputs/list", timeout=10)
            if response.status_code == 200:
                data = response.json()
                outputs = data.get('outputs', [])
                print(f"✅ Found {len(outputs)} output files")
                for output in outputs[:3]:  # Show first 3 outputs
                    print(f"   - {output.get('filename', 'Unknown')}")
                if len(outputs) > 3:
                    print(f"   ... and {len(outputs) - 3} more")
                return True
            else:
                print(f"❌ Outputs listing failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Outputs listing error: {e}")
            return False
    
    def run_tests(self):
        """Run all tests"""
        print("🧪 Wan2.2 Full Stack Generation Test Suite")
        print("=" * 50)
        
        tests = [
            ("Service Availability", self.check_services),
            ("System Information", self.test_system_info),
            ("Model Listing", self.test_model_list),
            ("Queue Status", self.test_queue_status),
            ("Outputs Listing", self.test_outputs_list),
            ("Simple Generation", self.test_simple_generation),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                if test_func():
                    passed += 1
                    print(f"✅ {test_name} PASSED")
                else:
                    print(f"❌ {test_name} FAILED")
            except Exception as e:
                print(f"❌ {test_name} ERROR: {e}")
        
        print(f"\n{'='*50}")
        print(f"🏁 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Your full stack is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        
        return passed == total

def main():
    """Main entry point"""
    tester = GenerationTester()
    success = tester.run_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()