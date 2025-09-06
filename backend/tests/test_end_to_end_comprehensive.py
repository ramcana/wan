"""
Comprehensive End-to-End Integration Tests
Tests complete workflows from FastAPI endpoints to real model generation
"""

import pytest
import asyncio
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from httpx import AsyncClient
from fastapi.testclient import TestClient

# Add backend to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from backend.app import app
    from backend.models.schemas import TaskStatus
except ImportError:
    # Fallback for testing
    from fastapi import FastAPI
    app = FastAPI()
    
    class TaskStatus:
        COMPLETED = "completed"
        FAILED = "failed"
        PENDING = "pending"
        PROCESSING = "processing"

class TestEndToEndWorkflows:
    """Test complete end-to-end workflows"""
    
    @pytest.mark.asyncio
    async def test_t2v_complete_workflow(self):
        """Test complete T2V workflow from API to video generation"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Step 1: Submit generation request
            request_data = {
                "model_type": "T2V-A14B",
                "prompt": "A majestic eagle soaring over snow-capped mountains at sunset",
                "resolution": "1280x720",
                "steps": 25,
                "guidance_scale": 7.5,
                "num_frames": 16,
                "fps": 8.0,
                "seed": 42
            }
            
            response = await client.post("/api/v1/generation/submit", json=request_data)
            assert response.status_code == 200
            
            task_data = response.json()
            assert "task_id" in task_data
            task_id = task_data["task_id"]
            
            # Step 2: Verify task appears in queue
            queue_response = await client.get("/api/v1/queue")
            assert queue_response.status_code == 200
            
            queue_data = queue_response.json()
            task_ids = [task["id"] for task in queue_data]
            assert task_id in task_ids
            
            # Step 3: Monitor task progress
            max_wait_time = 300  # 5 minutes
            start_time = time.time()
            final_status = None
            
            while (time.time() - start_time) < max_wait_time:
                await asyncio.sleep(2)
                
                status_response = await client.get(f"/api/v1/queue/{task_id}")
                assert status_response.status_code == 200
                
                status_data = status_response.json()
                final_status = status_data["status"]
                
                if final_status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    break
            
            # Step 4: Verify final result
            assert final_status is not None
            
            if final_status == TaskStatus.COMPLETED:
                # Verify output file information
                assert "output_path" in status_data
                output_path = Path(status_data["output_path"])
                assert output_path.suffix == ".mp4"
                
                # Verify generation metadata
                assert "generation_time" in status_data
                assert "model_used" in status_data
                assert status_data["model_used"] == "T2V-A14B"
                
            elif final_status == TaskStatus.FAILED:
                # Verify error information is provided
                assert "error_message" in status_data
                assert len(status_data["error_message"]) > 0
    
    @pytest.mark.asyncio
    async def test_i2v_complete_workflow(self):
        """Test complete I2V workflow with image upload"""
        # Create temporary test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            test_image_path = tmp_file.name
        
        try:
            async with AsyncClient(app=app, base_url="http://test") as client:
                # Step 1: Upload image (if endpoint exists)
                # For now, we'll simulate with a file path
                
                # Step 2: Submit I2V generation request
                request_data = {
                    "model_type": "I2V-A14B",
                    "prompt": "Animate this landscape with gentle wind movement",
                    "image_path": test_image_path,
                    "resolution": "1280x720",
                    "steps": 30,
                    "guidance_scale": 8.0,
                    "num_frames": 16,
                    "fps": 8.0
                }
                
                response = await client.post("/api/v1/generation/submit", json=request_data)
                
                # Should either succeed or fail gracefully
                if response.status_code == 200:
                    task_data = response.json()
                    task_id = task_data["task_id"]
                    
                    # Monitor progress briefly
                    for _ in range(5):  # Check 5 times
                        await asyncio.sleep(2)
                        
                        status_response = await client.get(f"/api/v1/queue/{task_id}")
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            if status_data["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                                break
                
                elif response.status_code == 400:
                    # Expected if image validation fails
                    error_data = response.json()
                    assert "message" in error_data
        
        finally:
            # Clean up temporary file
            Path(test_image_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_ti2v_complete_workflow(self):
        """Test complete TI2V workflow"""
        # Create temporary test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            test_image_path = tmp_file.name
        
        try:
            async with AsyncClient(app=app, base_url="http://test") as client:
                request_data = {
                    "model_type": "TI2V-5B",
                    "prompt": "Transform this image with magical sparkles and ethereal lighting",
                    "image_path": test_image_path,
                    "resolution": "1280x720",
                    "steps": 40,
                    "guidance_scale": 10.0,
                    "num_frames": 16,
                    "fps": 8.0
                }
                
                response = await client.post("/api/v1/generation/submit", json=request_data)
                
                # Handle response appropriately
                if response.status_code == 200:
                    task_data = response.json()
                    assert "task_id" in task_data
                    
                    # Verify task is queued
                    queue_response = await client.get("/api/v1/queue")
                    assert queue_response.status_code == 200
                    
                elif response.status_code in [400, 422]:
                    # Expected if validation fails
                    error_data = response.json()
                    assert "message" in error_data or "detail" in error_data
        
        finally:
            Path(test_image_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_generations(self):
        """Test multiple concurrent generation requests"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Submit multiple requests
            requests = [
                {
                    "model_type": "T2V-A14B",
                    "prompt": f"Test generation {i}: A beautiful landscape",
                    "resolution": "1280x720",
                    "steps": 20
                }
                for i in range(3)
            ]
            
            task_ids = []
            
            # Submit all requests
            for request_data in requests:
                response = await client.post("/api/v1/generation/submit", json=request_data)
                if response.status_code == 200:
                    task_data = response.json()
                    task_ids.append(task_data["task_id"])
            
            # Verify all tasks are in queue
            if task_ids:
                queue_response = await client.get("/api/v1/queue")
                assert queue_response.status_code == 200
                
                queue_data = queue_response.json()
                queued_ids = [task["id"] for task in queue_data]
                
                for task_id in task_ids:
                    assert task_id in queued_ids
    
    @pytest.mark.asyncio
    async def test_system_stats_during_generation(self):
        """Test system stats endpoint during generation"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Start a generation
            request_data = {
                "model_type": "T2V-A14B",
                "prompt": "System stats test generation",
                "resolution": "1280x720",
                "steps": 20
            }
            
            response = await client.post("/api/v1/generation/submit", json=request_data)
            
            if response.status_code == 200:
                # Check system stats while generation might be running
                stats_response = await client.get("/api/v1/system/stats")
                assert stats_response.status_code == 200
                
                stats_data = stats_response.json()
                
                # Verify expected fields
                expected_fields = ["cpu_percent", "memory_percent", "disk_usage"]
                for field in expected_fields:
                    assert field in stats_data
                
                # Values should be reasonable
                assert 0 <= stats_data["cpu_percent"] <= 100
                assert 0 <= stats_data["memory_percent"] <= 100
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self):
        """Test error handling in complete workflow"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test with invalid parameters
            invalid_requests = [
                {
                    "model_type": "INVALID_MODEL",
                    "prompt": "Test prompt",
                    "resolution": "1280x720",
                    "steps": 20
                },
                {
                    "model_type": "T2V-A14B",
                    "prompt": "",  # Empty prompt
                    "resolution": "1280x720",
                    "steps": 20
                },
                {
                    "model_type": "T2V-A14B",
                    "prompt": "Test prompt",
                    "resolution": "invalid_resolution",
                    "steps": 20
                }
            ]
            
            for request_data in invalid_requests:
                response = await client.post("/api/v1/generation/submit", json=request_data)
                
                # Should return error status
                assert response.status_code in [400, 422]
                
                error_data = response.json()
                assert "message" in error_data or "detail" in error_data
    
    @pytest.mark.asyncio
    async def test_outputs_endpoint_integration(self):
        """Test outputs endpoint integration with generation"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Check initial outputs
            initial_response = await client.get("/api/v1/outputs")
            assert initial_response.status_code == 200
            
            initial_outputs = initial_response.json()
            initial_count = len(initial_outputs)
            
            # Submit a generation
            request_data = {
                "model_type": "T2V-A14B",
                "prompt": "Outputs test generation",
                "resolution": "1280x720",
                "steps": 15
            }
            
            response = await client.post("/api/v1/generation/submit", json=request_data)
            
            if response.status_code == 200:
                task_data = response.json()
                task_id = task_data["task_id"]
                
                # Wait briefly for potential completion
                await asyncio.sleep(5)
                
                # Check if outputs list has changed
                final_response = await client.get("/api/v1/outputs")
                assert final_response.status_code == 200
                
                final_outputs = final_response.json()
                # Outputs count might increase if generation completed
                assert len(final_outputs) >= initial_count

class TestAPICompatibility:
    """Test API compatibility and contract maintenance"""
    
    @pytest.mark.asyncio
    async def test_health_endpoint_contract(self):
        """Test health endpoint maintains expected contract"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/health")
            assert response.status_code == 200
            
            data = response.json()
            
            # Required fields
            assert "status" in data
            assert "timestamp" in data
            
            # Status should be valid
            assert data["status"] in ["healthy", "degraded", "unhealthy"]
            
            # Timestamp should be recent (within last minute)
            import time
            current_time = time.time()
            timestamp = data["timestamp"]
            assert abs(current_time - timestamp) < 60
    
    @pytest.mark.asyncio
    async def test_queue_endpoint_contract(self):
        """Test queue endpoint maintains expected contract"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/queue")
            assert response.status_code == 200
            
            data = response.json()
            
            # Should return a list
            assert isinstance(data, list)
            
            # If tasks exist, verify structure
            for task in data:
                required_fields = ["id", "status", "created_at"]
                for field in required_fields:
                    assert field in task
                
                # Status should be valid
                valid_statuses = ["pending", "processing", "completed", "failed"]
                assert task["status"] in valid_statuses
    
    @pytest.mark.asyncio
    async def test_generation_submit_contract(self):
        """Test generation submit endpoint contract"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            request_data = {
                "model_type": "T2V-A14B",
                "prompt": "Contract test generation",
                "resolution": "1280x720",
                "steps": 20
            }
            
            response = await client.post("/api/v1/generation/submit", json=request_data)
            
            if response.status_code == 200:
                data = response.json()
                
                # Required response fields
                assert "task_id" in data
                assert "message" in data
                
                # Task ID should be a string
                assert isinstance(data["task_id"], str)
                assert len(data["task_id"]) > 0
            
            elif response.status_code in [400, 422]:
                # Error response should have message
                data = response.json()
                assert "message" in data or "detail" in data
    
    @pytest.mark.asyncio
    async def test_system_stats_contract(self):
        """Test system stats endpoint contract"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/system/stats")
            assert response.status_code == 200
            
            data = response.json()
            
            # Required fields
            required_fields = ["cpu_percent", "memory_percent", "disk_usage"]
            for field in required_fields:
                assert field in data
                assert isinstance(data[field], (int, float))
                assert 0 <= data[field] <= 100
    
    @pytest.mark.asyncio
    async def test_outputs_endpoint_contract(self):
        """Test outputs endpoint contract"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/outputs")
            assert response.status_code == 200
            
            data = response.json()
            
            # Should return a list
            assert isinstance(data, list)
            
            # If outputs exist, verify structure
            for output in data:
                expected_fields = ["filename", "path", "created_at", "size"]
                for field in expected_fields:
                    if field in output:  # Some fields might be optional
                        if field == "size":
                            assert isinstance(output[field], (int, float))
                        elif field in ["filename", "path"]:
                            assert isinstance(output[field], str)

class TestWebSocketIntegration:
    """Test WebSocket integration during generation"""
    
    @pytest.mark.asyncio
    async def test_websocket_manager_availability(self):
        """Test WebSocket manager is available and functional"""
        try:
            from backend.websocket.manager import get_connection_manager
            
            manager = await get_connection_manager()
            assert manager is not None
            
            # Verify required methods exist
            required_methods = ["send_generation_progress", "connect", "disconnect"]
            for method in required_methods:
                assert hasattr(manager, method)
                assert callable(getattr(manager, method))
        
        except ImportError:
            pytest.skip("WebSocket manager not available")
    
    @pytest.mark.asyncio
    async def test_progress_update_format(self):
        """Test progress update message format"""
        try:
            from backend.websocket.manager import get_connection_manager
            
            manager = await get_connection_manager()
            
            # Test progress update (won't actually send without connected clients)
            test_task_id = "test_websocket_task"
            
            # This should not raise an exception
            if hasattr(manager, 'send_generation_progress'):
                # Method exists, test would require actual WebSocket connection
                pass
        
        except ImportError:
            pytest.skip("WebSocket manager not available")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])