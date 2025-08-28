"""
Test WebSocket Progress Integration
Tests the enhanced WebSocket progress tracking system for real AI model generation
"""

import asyncio
import pytest
import logging
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Setup logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
async def mock_websocket_manager():
    """Create a mock WebSocket manager for testing"""
    manager = Mock()
    manager.send_generation_progress = AsyncMock()
    manager.send_detailed_generation_progress = AsyncMock()
    manager.send_model_loading_progress = AsyncMock()
    manager.send_vram_monitoring_update = AsyncMock()
    manager.send_generation_stage_notification = AsyncMock()
    return manager

@pytest.fixture
async def progress_integration(mock_websocket_manager):
    """Create a progress integration instance for testing"""
    from backend.websocket.progress_integration import ProgressIntegration
    
    integration = ProgressIntegration(mock_websocket_manager)
    await integration.initialize()
    return integration

class TestProgressIntegration:
    """Test the progress integration system"""
    
    @pytest.mark.asyncio
    async def test_start_generation_tracking(self, progress_integration, mock_websocket_manager):
        """Test starting generation tracking"""
        task_id = "test_task_123"
        model_type = "t2v-A14B"
        estimated_duration = 60.0
        
        await progress_integration.start_generation_tracking(
            task_id, model_type, estimated_duration
        )
        
        # Verify generation start notification was sent
        mock_websocket_manager.send_generation_stage_notification.assert_called_once()
        call_args = mock_websocket_manager.send_generation_stage_notification.call_args
        
        assert call_args[0][0] == task_id  # task_id
        assert call_args[0][1] == "generation_started"  # stage
        assert call_args[1]["model_type"] == model_type
        assert call_args[1]["estimated_duration"] == estimated_duration
    
    @pytest.mark.asyncio
    async def test_update_stage_progress(self, progress_integration, mock_websocket_manager):
        """Test updating stage progress"""
        from backend.websocket.progress_integration import GenerationStage
        
        # Start tracking first
        task_id = "test_task_123"
        await progress_integration.start_generation_tracking(task_id, "t2v-A14B")
        
        # Update progress
        stage = GenerationStage.LOADING_MODEL
        progress = 25
        message = "Loading T2V model"
        
        await progress_integration.update_stage_progress(stage, progress, message)
        
        # Verify detailed progress update was sent
        mock_websocket_manager.send_detailed_generation_progress.assert_called()
        call_args = mock_websocket_manager.send_detailed_generation_progress.call_args
        
        assert call_args[0][0] == task_id  # task_id
        assert call_args[0][1] == stage.value  # stage
        assert call_args[0][2] == progress  # progress
        assert call_args[0][3] == message  # message
    
    @pytest.mark.asyncio
    async def test_update_model_loading_progress(self, progress_integration, mock_websocket_manager):
        """Test model loading progress updates"""
        # Start tracking first
        task_id = "test_task_123"
        await progress_integration.start_generation_tracking(task_id, "t2v-A14B")
        
        # Update model loading progress
        model_type = "t2v-A14B"
        progress = 50
        status = "Downloading model files"
        
        await progress_integration.update_model_loading_progress(model_type, progress, status)
        
        # Verify model loading progress was sent
        mock_websocket_manager.send_model_loading_progress.assert_called_once_with(
            task_id, model_type, progress, status
        )
    
    @pytest.mark.asyncio
    async def test_update_generation_step_progress(self, progress_integration, mock_websocket_manager):
        """Test generation step progress updates"""
        # Start tracking first
        task_id = "test_task_123"
        await progress_integration.start_generation_tracking(task_id, "t2v-A14B")
        
        # Update generation step progress
        current_step = 5
        total_steps = 20
        
        await progress_integration.update_generation_step_progress(current_step, total_steps)
        
        # Verify detailed progress update was sent
        mock_websocket_manager.send_detailed_generation_progress.assert_called()
        call_args = mock_websocket_manager.send_detailed_generation_progress.call_args
        
        assert call_args[1]["current_step"] == current_step
        assert call_args[1]["total_steps"] == total_steps
    
    @pytest.mark.asyncio
    async def test_complete_generation_tracking_success(self, progress_integration, mock_websocket_manager):
        """Test completing generation tracking successfully"""
        from backend.websocket.progress_integration import GenerationStage
        
        # Start tracking first
        task_id = "test_task_123"
        await progress_integration.start_generation_tracking(task_id, "t2v-A14B")
        
        # Complete tracking
        output_path = "/path/to/output.mp4"
        await progress_integration.complete_generation_tracking(
            success=True, output_path=output_path
        )
        
        # Verify completion progress update was sent
        mock_websocket_manager.send_detailed_generation_progress.assert_called()
        call_args = mock_websocket_manager.send_detailed_generation_progress.call_args
        
        assert call_args[0][1] == GenerationStage.COMPLETED.value  # stage
        assert call_args[0][2] == 100  # progress
        assert call_args[1]["output_path"] == output_path
    
    @pytest.mark.asyncio
    async def test_complete_generation_tracking_failure(self, progress_integration, mock_websocket_manager):
        """Test completing generation tracking with failure"""
        from backend.websocket.progress_integration import GenerationStage
        
        # Start tracking first
        task_id = "test_task_123"
        await progress_integration.start_generation_tracking(task_id, "t2v-A14B")
        
        # Complete tracking with error
        error_message = "CUDA out of memory"
        await progress_integration.complete_generation_tracking(
            success=False, error_message=error_message
        )
        
        # Verify failure progress update was sent
        mock_websocket_manager.send_detailed_generation_progress.assert_called()
        call_args = mock_websocket_manager.send_detailed_generation_progress.call_args
        
        assert call_args[0][1] == GenerationStage.FAILED.value  # stage
        assert call_args[0][2] == 0  # progress
        assert call_args[1]["error_message"] == error_message

class TestWebSocketManagerEnhancements:
    """Test the enhanced WebSocket manager functionality"""
    
    @pytest.mark.asyncio
    async def test_send_detailed_generation_progress(self):
        """Test sending detailed generation progress"""
        from backend.websocket.manager import ConnectionManager
        
        manager = ConnectionManager()
        
        # Mock the broadcast_to_topic method
        manager.broadcast_to_topic = AsyncMock()
        
        task_id = "test_task_123"
        stage = "loading_model"
        progress = 50
        message = "Loading T2V model"
        
        await manager.send_detailed_generation_progress(
            task_id, stage, progress, message, 
            estimated_time_remaining=30.0
        )
        
        # Verify broadcast was called
        manager.broadcast_to_topic.assert_called_once()
        call_args = manager.broadcast_to_topic.call_args
        
        message_data = call_args[0][0]
        assert message_data["type"] == "detailed_generation_progress"
        assert message_data["data"]["task_id"] == task_id
        assert message_data["data"]["stage"] == stage
        assert message_data["data"]["progress"] == progress
        assert message_data["data"]["message"] == message
        assert message_data["data"]["estimated_time_remaining"] == 30.0
    
    @pytest.mark.asyncio
    async def test_send_model_loading_progress(self):
        """Test sending model loading progress"""
        from backend.websocket.manager import ConnectionManager
        
        manager = ConnectionManager()
        manager.broadcast_to_topic = AsyncMock()
        
        task_id = "test_task_123"
        model_type = "t2v-A14B"
        progress = 75
        status = "Model loaded successfully"
        
        await manager.send_model_loading_progress(
            task_id, model_type, progress, status
        )
        
        # Verify broadcast was called
        manager.broadcast_to_topic.assert_called_once()
        call_args = manager.broadcast_to_topic.call_args
        
        message_data = call_args[0][0]
        assert message_data["type"] == "model_loading_progress"
        assert message_data["data"]["task_id"] == task_id
        assert message_data["data"]["model_type"] == model_type
        assert message_data["data"]["progress"] == progress
        assert message_data["data"]["status"] == status
    
    @pytest.mark.asyncio
    async def test_send_vram_monitoring_update(self):
        """Test sending VRAM monitoring updates"""
        from backend.websocket.manager import ConnectionManager
        
        manager = ConnectionManager()
        manager.broadcast_to_topic = AsyncMock()
        
        vram_data = {
            "allocated_mb": 8192.0,
            "total_mb": 16384.0,
            "allocated_percent": 50.0,
            "warning_level": "normal"
        }
        
        await manager.send_vram_monitoring_update(vram_data, task_id="test_task_123")
        
        # Verify broadcast was called
        manager.broadcast_to_topic.assert_called_once()
        call_args = manager.broadcast_to_topic.call_args
        
        message_data = call_args[0][0]
        assert message_data["type"] == "vram_monitoring"
        assert message_data["data"]["allocated_mb"] == 8192.0
        assert message_data["data"]["task_id"] == "test_task_123"
    
    @pytest.mark.asyncio
    async def test_send_generation_stage_notification(self):
        """Test sending generation stage notifications"""
        from backend.websocket.manager import ConnectionManager
        
        manager = ConnectionManager()
        manager.broadcast_to_topic = AsyncMock()
        
        task_id = "test_task_123"
        stage = "generating"
        stage_progress = 45
        
        await manager.send_generation_stage_notification(
            task_id, stage, stage_progress, 
            stage_message="Generating frames"
        )
        
        # Verify broadcast was called
        manager.broadcast_to_topic.assert_called_once()
        call_args = manager.broadcast_to_topic.call_args
        
        message_data = call_args[0][0]
        assert message_data["type"] == "generation_stage"
        assert message_data["data"]["task_id"] == task_id
        assert message_data["data"]["stage"] == stage
        assert message_data["data"]["stage_progress"] == stage_progress
        assert message_data["data"]["stage_message"] == "Generating frames"

class TestVRAMMonitoring:
    """Test VRAM monitoring functionality"""
    
    @pytest.mark.asyncio
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.current_device', return_value=0)
    @patch('torch.cuda.memory_allocated', return_value=8589934592)  # 8GB in bytes
    @patch('torch.cuda.memory_reserved', return_value=10737418240)  # 10GB in bytes
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.get_device_name', return_value="NVIDIA RTX 4080")
    async def test_vram_monitoring_loop(self, mock_device_name, mock_device_props, 
                                       mock_reserved, mock_allocated, mock_device, mock_available):
        """Test VRAM monitoring loop functionality"""
        from backend.websocket.progress_integration import ProgressIntegration
        
        # Mock device properties
        mock_props = Mock()
        mock_props.total_memory = 17179869184  # 16GB in bytes
        mock_device_props.return_value = mock_props
        
        # Create progress integration with mock WebSocket manager
        mock_websocket_manager = Mock()
        mock_websocket_manager.send_vram_monitoring_update = AsyncMock()
        
        integration = ProgressIntegration(mock_websocket_manager)
        await integration.initialize()
        
        # Start generation tracking to activate VRAM monitoring
        await integration.start_generation_tracking("test_task", "t2v-A14B")
        
        # Wait a short time for monitoring to run
        await asyncio.sleep(1.5)
        
        # Stop monitoring
        await integration._stop_vram_monitoring()
        
        # Verify VRAM monitoring updates were sent
        assert mock_websocket_manager.send_vram_monitoring_update.called
        
        # Check the VRAM data structure
        call_args = mock_websocket_manager.send_vram_monitoring_update.call_args_list[0]
        vram_data = call_args[0][0]
        
        assert "allocated_mb" in vram_data
        assert "total_mb" in vram_data
        assert "allocated_percent" in vram_data
        assert "warning_level" in vram_data
        assert vram_data["device_name"] == "NVIDIA RTX 4080"

if __name__ == "__main__":
    # Run basic functionality test
    async def test_basic_functionality():
        """Basic functionality test"""
        logger.info("Testing WebSocket Progress Integration...")
        
        try:
            from backend.websocket.progress_integration import get_progress_integration
            
            # Get progress integration instance
            progress_integration = await get_progress_integration()
            logger.info("✓ Progress integration initialized successfully")
            
            # Test starting generation tracking
            task_id = "test_task_123"
            await progress_integration.start_generation_tracking(task_id, "t2v-A14B", 60.0)
            logger.info("✓ Generation tracking started successfully")
            
            # Test stage progress updates
            from backend.websocket.progress_integration import GenerationStage
            await progress_integration.update_stage_progress(
                GenerationStage.LOADING_MODEL, 25, "Loading T2V model"
            )
            logger.info("✓ Stage progress update sent successfully")
            
            # Test model loading progress
            await progress_integration.update_model_loading_progress(
                "t2v-A14B", 50, "Model loading in progress"
            )
            logger.info("✓ Model loading progress sent successfully")
            
            # Test generation step progress
            await progress_integration.update_generation_step_progress(5, 20)
            logger.info("✓ Generation step progress sent successfully")
            
            # Test completion
            await progress_integration.complete_generation_tracking(
                success=True, output_path="/test/output.mp4"
            )
            logger.info("✓ Generation tracking completed successfully")
            
            logger.info("All tests passed! WebSocket Progress Integration is working correctly.")
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise
    
    # Run the test
    asyncio.run(test_basic_functionality())