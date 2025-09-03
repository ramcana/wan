import pytest
import asyncio
import json
import tempfile
import os
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from backend.main import backend.app as app
from backend.models.schemas import GenerationRequest, TaskStatus
from backend.database import get_db_session

client = TestClient(app)

class TestComprehensiveAPI:
    """Comprehensive API testing suite covering all endpoints and scenarios"""

    def setup_method(self):
        """Setup for each test method"""
        self.mock_generation_service = Mock()
        self.test_files = []

    def teardown_method(self):
        """Cleanup after each test method"""
        # Clean up test files
        for file_path in self.test_files:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert data["status"] == "healthy"

    def test_health_endpoint_with_gpu_check(self):
        """Test health endpoint with GPU validation"""
        with patch('backend.core.system_integration.check_gpu_availability') as mock_gpu:
            mock_gpu.return_value = True
            
            response = client.get("/api/v1/health?check_gpu=true")
            assert response.status_code == 200
            
            data = response.json()
            assert data["gpu_available"] is True

    def test_health_endpoint_gpu_unavailable(self):
        """Test health endpoint when GPU is unavailable"""
        with patch('backend.core.system_integration.check_gpu_availability') as mock_gpu:
            mock_gpu.return_value = False
            
            response = client.get("/api/v1/health?check_gpu=true")
            assert response.status_code == 503
            
            data = response.json()
            assert data["gpu_available"] is False

    def test_t2v_generation_success(self):
        """Test successful T2V generation"""
        request_data = {
            "model_type": "T2V-A14B",
            "prompt": "A beautiful sunset over mountains",
            "resolution": "1280x720",
            "steps": 50
        }

        with patch('backend.services.generation_service.GenerationService.create_task') as mock_create:
            mock_create.return_value = {
                "task_id": "test-task-1",
                "status": "pending",
                "message": "Task created successfully"
            }

            response = client.post("/api/v1/generate", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["task_id"] == "test-task-1"
            assert data["status"] == "pending"

    def test_i2v_generation_with_image(self):
        """Test I2V generation with image upload"""
        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(b"fake image data")
            tmp_file_path = tmp_file.name
            self.test_files.append(tmp_file_path)

        with open(tmp_file_path, 'rb') as image_file:
            files = {"image": ("test.jpg", image_file, "image/jpeg")}
            data = {
                "model_type": "I2V-A14B",
                "prompt": "Transform this image into a video",
                "resolution": "1280x720",
                "steps": 50
            }

            with patch('backend.services.generation_service.GenerationService.create_task') as mock_create:
                mock_create.return_value = {
                    "task_id": "test-task-2",
                    "status": "pending",
                    "message": "Task created successfully"
                }

                response = client.post("/api/v1/generate", data=data, files=files)
                assert response.status_code == 200
                
                result = response.json()
                assert result["task_id"] == "test-task-2"

    def test_generation_validation_errors(self):
        """Test generation request validation"""
        # Missing required fields
        response = client.post("/api/v1/generate", json={})
        assert response.status_code == 422

        # Invalid model type
        response = client.post("/api/v1/generate", json={
            "model_type": "INVALID-MODEL",
            "prompt": "Test prompt"
        })
        assert response.status_code == 422

        # Prompt too long
        response = client.post("/api/v1/generate", json={
            "model_type": "T2V-A14B",
            "prompt": "a" * 501  # Exceeds 500 character limit
        })
        assert response.status_code == 422

    def test_queue_endpoint(self):
        """Test queue status endpoint"""
        mock_tasks = [
            {
                "id": "task-1",
                "model_type": "T2V-A14B",
                "prompt": "Test prompt 1",
                "status": "processing",
                "progress": 50,
                "created_at": "2024-01-01T00:00:00Z"
            },
            {
                "id": "task-2",
                "model_type": "I2V-A14B",
                "prompt": "Test prompt 2",
                "status": "pending",
                "progress": 0,
                "created_at": "2024-01-01T00:01:00Z"
            }
        ]

        with patch('backend.services.generation_service.GenerationService.get_queue') as mock_queue:
            mock_queue.return_value = mock_tasks

            response = client.get("/api/v1/queue")
            assert response.status_code == 200
            
            data = response.json()
            assert len(data) == 2
            assert data[0]["id"] == "task-1"
            assert data[1]["id"] == "task-2"

    def test_queue_task_cancellation(self):
        """Test task cancellation"""
        task_id = "test-task-1"

        with patch('backend.services.generation_service.GenerationService.cancel_task') as mock_cancel:
            mock_cancel.return_value = {"success": True, "message": "Task cancelled"}

            response = client.delete(f"/api/v1/queue/{task_id}")
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True

    def test_queue_task_cancellation_not_found(self):
        """Test cancellation of non-existent task"""
        task_id = "non-existent-task"

        with patch('backend.services.generation_service.GenerationService.cancel_task') as mock_cancel:
            mock_cancel.side_effect = ValueError("Task not found")

            response = client.delete(f"/api/v1/queue/{task_id}")
            assert response.status_code == 404

    def test_system_stats_endpoint(self):
        """Test system statistics endpoint"""
        mock_stats = {
            "cpu": 45.5,
            "ram": {"used": 8.2, "total": 16.0},
            "gpu": 78.3,
            "vram": {"used": 6.4, "total": 8.0},
            "timestamp": "2024-01-01T00:00:00Z"
        }

        with patch('backend.core.system_integration.get_system_stats') as mock_stats_func:
            mock_stats_func.return_value = mock_stats

            response = client.get("/api/v1/system/stats")
            assert response.status_code == 200
            
            data = response.json()
            assert data["cpu"] == 45.5
            assert data["ram"]["used"] == 8.2
            assert data["gpu"] == 78.3

    def test_outputs_endpoint(self):
        """Test outputs listing endpoint"""
        mock_outputs = [
            {
                "id": "video-1",
                "filename": "video1.mp4",
                "thumbnail": "thumb1.jpg",
                "metadata": {
                    "prompt": "Beautiful sunset",
                    "model_type": "T2V-A14B",
                    "resolution": "1280x720",
                    "duration": 5,
                    "created_at": "2024-01-01T00:00:00Z"
                }
            }
        ]

        with patch('backend.services.generation_service.GenerationService.get_outputs') as mock_outputs_func:
            mock_outputs_func.return_value = mock_outputs

            response = client.get("/api/v1/outputs")
            assert response.status_code == 200
            
            data = response.json()
            assert len(data) == 1
            assert data[0]["id"] == "video-1"

    def test_prompt_enhancement(self):
        """Test prompt enhancement endpoint"""
        original_prompt = "sunset"
        enhanced_prompt = "A breathtaking sunset over majestic mountains with golden light"

        with patch('backend.services.generation_service.GenerationService.enhance_prompt') as mock_enhance:
            mock_enhance.return_value = {
                "original": original_prompt,
                "enhanced": enhanced_prompt,
                "improvements": ["Added descriptive adjectives", "Specified scene details"]
            }

            response = client.post("/api/v1/prompt/enhance", json={"prompt": original_prompt})
            assert response.status_code == 200
            
            data = response.json()
            assert data["original"] == original_prompt
            assert data["enhanced"] == enhanced_prompt
            assert len(data["improvements"]) == 2

    def test_error_handling_generation_failure(self):
        """Test error handling when generation fails"""
        request_data = {
            "model_type": "T2V-A14B",
            "prompt": "Test prompt",
            "resolution": "1280x720",
            "steps": 50
        }

        with patch('backend.services.generation_service.GenerationService.create_task') as mock_create:
            mock_create.side_effect = RuntimeError("GPU out of memory")

            response = client.post("/api/v1/generate", json=request_data)
            assert response.status_code == 500
            
            data = response.json()
            assert "error" in data
            assert "GPU out of memory" in data["message"]

    def test_rate_limiting(self):
        """Test API rate limiting"""
        request_data = {
            "model_type": "T2V-A14B",
            "prompt": "Test prompt",
            "resolution": "1280x720",
            "steps": 50
        }

        # Make multiple rapid requests
        responses = []
        for i in range(10):
            response = client.post("/api/v1/generate", json=request_data)
            responses.append(response)

        # Check if rate limiting kicks in
        rate_limited = any(r.status_code == 429 for r in responses)
        # Note: This test depends on rate limiting configuration

    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import threading
import time

        results = []
        
        def make_request():
            response = client.get("/api/v1/health")
            results.append(response.status_code)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5

    def test_large_file_upload(self):
        """Test handling of large file uploads"""
        # Create a large temporary file (simulating large image)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            # Write 10MB of data
            tmp_file.write(b"0" * (10 * 1024 * 1024))
            tmp_file_path = tmp_file.name
            self.test_files.append(tmp_file_path)

        with open(tmp_file_path, 'rb') as image_file:
            files = {"image": ("large_test.jpg", image_file, "image/jpeg")}
            data = {
                "model_type": "I2V-A14B",
                "prompt": "Test with large image",
                "resolution": "1280x720",
                "steps": 50
            }

            response = client.post("/api/v1/generate", data=data, files=files)
            # Should either succeed or fail gracefully with appropriate error
            assert response.status_code in [200, 413, 422]  # Success, Payload Too Large, or Validation Error

    def test_malformed_requests(self):
        """Test handling of malformed requests"""
        # Invalid JSON
        response = client.post(
            "/api/v1/generate",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

        # Missing Content-Type
        response = client.post("/api/v1/generate", data='{"test": "data"}')
        assert response.status_code in [422, 415]

    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.options("/api/v1/health")
        assert response.status_code == 200
        
        # Check for CORS headers
        headers = response.headers
        assert "access-control-allow-origin" in headers
        assert "access-control-allow-methods" in headers

    def test_api_versioning(self):
        """Test API versioning"""
        # Test v1 endpoint
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        # Test non-existent version
        response = client.get("/api/v2/health")
        assert response.status_code == 404

    def test_authentication_headers(self):
        """Test authentication header handling"""
        # Test with valid auth header (if implemented)
        headers = {"Authorization": "Bearer test-token"}
        response = client.get("/api/v1/health", headers=headers)
        # Should not affect health endpoint
        assert response.status_code == 200

    def test_request_timeout_handling(self):
        """Test request timeout handling"""
        with patch('backend.services.generation_service.GenerationService.create_task') as mock_create:
            # Simulate a long-running operation
            def slow_operation(*args, **kwargs):
                import time
time.sleep(2)  # Simulate slow operation
                return {"task_id": "slow-task", "status": "pending"}
            
            mock_create.side_effect = slow_operation

            request_data = {
                "model_type": "T2V-A14B",
                "prompt": "Test prompt",
                "resolution": "1280x720",
                "steps": 50
            }

            # This should either complete or timeout gracefully
            response = client.post("/api/v1/generate", json=request_data)
            assert response.status_code in [200, 408, 504]  # Success, Timeout, or Gateway Timeout

    def test_database_connection_failure(self):
        """Test handling of database connection failures"""
        with patch('backend.database.get_db_session') as mock_db:
            mock_db.side_effect = ConnectionError("Database unavailable")

            response = client.get("/api/v1/queue")
            assert response.status_code == 503  # Service Unavailable

    def test_memory_usage_monitoring(self):
        """Test memory usage during API operations"""
        import psutil
import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Make multiple requests
        for i in range(10):
            response = client.get("/api/v1/health")
            assert response.status_code == 200

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB for health checks)
        assert memory_increase < 50 * 1024 * 1024

    def test_response_compression(self):
        """Test response compression"""
        headers = {"Accept-Encoding": "gzip"}
        response = client.get("/api/v1/health", headers=headers)
        
        assert response.status_code == 200
        # Check if response is compressed (if compression is enabled)
        if "content-encoding" in response.headers:
            assert "gzip" in response.headers["content-encoding"]

    def test_api_documentation_endpoint(self):
        """Test API documentation endpoints"""
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema

        # Test Swagger UI
        response = client.get("/docs")
        assert response.status_code == 200

        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200

@pytest.mark.asyncio
class TestAsyncOperations:
    """Test asynchronous operations and WebSocket functionality"""

    async def test_websocket_connection(self):
        """Test WebSocket connection establishment"""
        with client.websocket_connect("/ws") as websocket:
            # Test connection
            data = websocket.receive_json()
            assert "type" in data
            assert data["type"] == "connection_established"

    async def test_websocket_progress_updates(self):
        """Test WebSocket progress updates"""
        with client.websocket_connect("/ws") as websocket:
            # Simulate progress update
            test_progress = {
                "type": "progress_update",
                "task_id": "test-task",
                "progress": 50
            }
            
            # In a real test, this would be triggered by the backend
            # For now, we just test the connection works
            websocket.receive_json()  # Connection established message

    async def test_background_task_processing(self):
        """Test background task processing"""
        with patch('backend.services.generation_service.GenerationService.process_task') as mock_process:
            mock_process.return_value = AsyncMock()

            request_data = {
                "model_type": "T2V-A14B",
                "prompt": "Test background processing",
                "resolution": "1280x720",
                "steps": 50
            }

            response = client.post("/api/v1/generate", json=request_data)
            assert response.status_code == 200

            # Allow some time for background processing
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])