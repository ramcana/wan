"""
Simple Tests for Model Availability Manager
Tests core functionality without complex dependencies
"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import the components to test
from backend.core.model_availability_manager import (
    ModelAvailabilityManager,
    ModelAvailabilityStatus,
    ModelPriority,
    DetailedModelStatus,
    ModelRequestResult,
    CleanupResult,
    RetentionPolicy
)


class TestModelAvailabilityManagerSimple:
    """Simple test suite for ModelAvailabilityManager core functionality"""
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary models directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_model_manager(self):
        """Create simple mock ModelManager"""
        mock_manager = Mock()
        mock_manager.get_model_id = Mock(return_value="test-model-id")
        mock_manager.get_model_status = Mock(return_value={
            "model_id": "test-model-id",
            "is_cached": True,
            "is_loaded": False,
            "is_valid": True,
            "size_mb": 8500.0
        })
        return mock_manager
    
    @pytest.fixture
    async def simple_availability_manager(self, temp_models_dir, mock_model_manager):
        """Create simple ModelAvailabilityManager instance for testing"""
        manager = ModelAvailabilityManager(
            model_manager=mock_model_manager,
            models_dir=temp_models_dir
        )
        
        # Mock the downloader and health monitor to avoid complex dependencies
        manager.downloader = AsyncMock()
        manager.downloader.__aenter__ = AsyncMock(return_value=manager.downloader)
        manager.downloader.__aexit__ = AsyncMock(return_value=None)
        manager.downloader.add_progress_callback = Mock()
        manager.downloader.get_download_progress = AsyncMock(return_value=None)
        
        manager.health_monitor = Mock()
        manager.health_monitor.health_data_dir = Path(temp_models_dir) / ".health"
        manager.health_monitor.health_data_dir.mkdir(exist_ok=True)
        manager.health_monitor.add_health_callback = Mock()
        
        # Initialize the manager
        await manager.initialize()
        
        yield manager
        
        # Cleanup
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_basic_initialization(self, temp_models_dir):
        """Test basic manager initialization"""
        mock_manager = Mock()
        mock_manager.get_model_id = Mock(return_value="test-model")
        
        manager = ModelAvailabilityManager(
            model_manager=mock_manager,
            models_dir=temp_models_dir
        )
        
        # Test initialization
        success = await manager.initialize()
        assert success is True
        
        # Check that directories are created
        assert Path(temp_models_dir).exists()
        assert (Path(temp_models_dir) / ".analytics").exists()
        
        # Cleanup
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_detailed_model_status_creation(self, simple_availability_manager):
        """Test creation of DetailedModelStatus objects"""
        # Test status creation
        status = await simple_availability_manager._check_single_model_availability("t2v-A14B")
        
        # Verify status structure
        assert isinstance(status, DetailedModelStatus)
        assert status.model_type == "t2v-A14B"
        assert hasattr(status, 'availability_status')
        assert hasattr(status, 'is_available')
        assert hasattr(status, 'size_mb')
        assert hasattr(status, 'priority')
    
    @pytest.mark.asyncio
    async def test_usage_tracking(self, simple_availability_manager):
        """Test model usage tracking functionality"""
        model_type = "t2v-A14B"
        
        # Track usage multiple times
        await simple_availability_manager._track_model_usage(model_type)
        await simple_availability_manager._track_model_usage(model_type)
        await simple_availability_manager._track_model_usage(model_type)
        
        # Verify usage data
        assert model_type in simple_availability_manager._usage_data
        usage_data = simple_availability_manager._usage_data[model_type]
        
        assert usage_data["total_uses"] == 3
        assert usage_data["last_used"] is not None
        assert len(usage_data["usage_history"]) == 3
        assert usage_data["usage_frequency"] > 0
    
    @pytest.mark.asyncio
    async def test_download_queue_management(self, simple_availability_manager):
        """Test download queue management"""
        # Queue models with different priorities
        await simple_availability_manager._queue_model_download("t2v-A14B", ModelPriority.LOW)
        await simple_availability_manager._queue_model_download("i2v-A14B", ModelPriority.HIGH)
        await simple_availability_manager._queue_model_download("ti2v-5B", ModelPriority.CRITICAL)
        
        # Verify queue ordering (highest priority first)
        queue = simple_availability_manager._download_queue
        assert len(queue) == 3
        
        # Check that critical priority comes first
        priorities = [item[1] for item in queue]
        assert ModelPriority.CRITICAL in priorities
        
        # Test updating priority
        await simple_availability_manager._queue_model_download("t2v-A14B", ModelPriority.CRITICAL)
        
        # Verify no duplicates and updated priority
        model_types = [item[0] for item in simple_availability_manager._download_queue]
        assert model_types.count("t2v-A14B") == 1
    
    @pytest.mark.asyncio
    async def test_analytics_persistence(self, simple_availability_manager):
        """Test that analytics data persists"""
        model_type = "t2v-A14B"
        
        # Track some usage
        await simple_availability_manager._track_model_usage(model_type)
        await simple_availability_manager._track_model_usage(model_type)
        
        # Save analytics
        await simple_availability_manager._save_usage_analytics()
        
        # Verify analytics file exists
        analytics_file = simple_availability_manager.analytics_dir / "usage_analytics.json"
        assert analytics_file.exists()
        
        # Verify data was saved
        with open(analytics_file, 'r') as f:
            saved_data = json.load(f)
        
        assert model_type in saved_data
        assert saved_data[model_type]["total_uses"] == 2
    
    @pytest.mark.asyncio
    async def test_model_request_handling_available(self, simple_availability_manager):
        """Test handling request for available model"""
        # Mock model as available
        simple_availability_manager.model_manager.get_model_status.return_value = {
            "model_id": "test-model-id",
            "is_cached": True,
            "is_loaded": False,
            "is_valid": True,
            "size_mb": 8500.0
        }
        
        # Test request handling
        result = await simple_availability_manager.handle_model_request("t2v-A14B")
        
        # Verify results
        assert isinstance(result, ModelRequestResult)
        assert result.success is True
        assert result.availability_status == ModelAvailabilityStatus.AVAILABLE
    
    @pytest.mark.asyncio
    async def test_model_request_handling_missing(self, simple_availability_manager):
        """Test handling request for missing model"""
        # Mock model as missing
        simple_availability_manager.model_manager.get_model_status.return_value = {
            "model_id": "test-model-id",
            "is_cached": False,
            "is_loaded": False,
            "is_valid": False,
            "size_mb": 0.0
        }
        
        # Test request handling
        result = await simple_availability_manager.handle_model_request("i2v-A14B")
        
        # Verify results
        assert isinstance(result, ModelRequestResult)
        assert result.success is False
        assert result.availability_status == ModelAvailabilityStatus.MISSING
        assert result.action_required == "download_required"
    
    @pytest.mark.asyncio
    async def test_cleanup_recommendations(self, simple_availability_manager):
        """Test cleanup recommendations generation"""
        # Setup usage data with old usage
        old_date = datetime.now() - timedelta(days=45)
        simple_availability_manager._usage_data = {
            "t2v-A14B": {
                "total_uses": 5,
                "last_used": old_date.isoformat(),
                "usage_frequency": 0.05,  # Very low usage
                "usage_history": []
            }
        }
        
        # Mock model as available
        simple_availability_manager.model_manager.get_model_status.return_value = {
            "model_id": "test-model-id",
            "is_cached": True,
            "is_loaded": False,
            "is_valid": True,
            "size_mb": 8500.0
        }
        
        # Test cleanup
        policy = RetentionPolicy(max_unused_days=30, min_usage_frequency=0.1)
        result = await simple_availability_manager.cleanup_unused_models(policy)
        
        # Verify results
        assert isinstance(result, CleanupResult)
        assert result.success is True
        assert len(result.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_priority_assignment(self, simple_availability_manager):
        """Test model priority assignment based on usage"""
        # Test high usage model
        simple_availability_manager._usage_data = {
            "t2v-A14B": {
                "usage_frequency": 2.0,  # High usage
                "last_used": datetime.now().isoformat()
            }
        }
        
        status = await simple_availability_manager._check_single_model_availability("t2v-A14B")
        assert status.priority == ModelPriority.HIGH
        
        # Test low usage model
        simple_availability_manager._usage_data = {
            "i2v-A14B": {
                "usage_frequency": 0.05,  # Low usage
                "last_used": (datetime.now() - timedelta(days=20)).isoformat()
            }
        }
        
        status = await simple_availability_manager._check_single_model_availability("i2v-A14B")
        assert status.priority == ModelPriority.LOW
    
    @pytest.mark.asyncio
    async def test_callback_registration(self, simple_availability_manager):
        """Test callback registration"""
        # Create mock callbacks
        status_callback = Mock()
        download_callback = Mock()
        
        # Register callbacks
        simple_availability_manager.add_status_callback(status_callback)
        simple_availability_manager.add_download_callback(download_callback)
        
        # Verify callbacks are registered
        assert status_callback in simple_availability_manager._status_callbacks
        assert download_callback in simple_availability_manager._download_callbacks
    
    @pytest.mark.asyncio
    async def test_error_handling(self, simple_availability_manager):
        """Test error handling in various scenarios"""
        # Test with model manager that raises exceptions
        simple_availability_manager.model_manager.get_model_status.side_effect = Exception("Test error")
        
        # Test that errors are handled gracefully
        result = await simple_availability_manager.handle_model_request("t2v-A14B")
        
        assert isinstance(result, ModelRequestResult)
        assert result.success is False
        assert result.availability_status == ModelAvailabilityStatus.UNKNOWN
        assert result.error_details is not None
    
    @pytest.mark.asyncio
    async def test_comprehensive_status_structure(self, simple_availability_manager):
        """Test comprehensive model status structure"""
        # Get comprehensive status
        status_dict = await simple_availability_manager.get_comprehensive_model_status()
        
        # Verify structure
        assert isinstance(status_dict, dict)
        
        # Check that we get status for supported models
        supported_models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
        for model_type in supported_models:
            if model_type in status_dict:
                status = status_dict[model_type]
                assert isinstance(status, DetailedModelStatus)
                assert hasattr(status, 'model_id')
                assert hasattr(status, 'availability_status')
                assert hasattr(status, 'priority')
                assert hasattr(status, 'usage_frequency')
    
    def test_model_availability_status_enum(self):
        """Test ModelAvailabilityStatus enum values"""
        # Test all enum values exist
        assert ModelAvailabilityStatus.AVAILABLE
        assert ModelAvailabilityStatus.DOWNLOADING
        assert ModelAvailabilityStatus.MISSING
        assert ModelAvailabilityStatus.CORRUPTED
        assert ModelAvailabilityStatus.UPDATING
        assert ModelAvailabilityStatus.QUEUED
        assert ModelAvailabilityStatus.PAUSED
        assert ModelAvailabilityStatus.FAILED
        assert ModelAvailabilityStatus.UNKNOWN
    
    def test_model_priority_enum(self):
        """Test ModelPriority enum values"""
        # Test all enum values exist
        assert ModelPriority.CRITICAL
        assert ModelPriority.HIGH
        assert ModelPriority.MEDIUM
        assert ModelPriority.LOW
    
    def test_detailed_model_status_dataclass(self):
        """Test DetailedModelStatus dataclass structure"""
        # Create a status object
        status = DetailedModelStatus(
            model_id="test-model",
            model_type="t2v-A14B",
            is_available=True,
            is_loaded=False,
            size_mb=8500.0,
            availability_status=ModelAvailabilityStatus.AVAILABLE
        )
        
        # Test required fields
        assert status.model_id == "test-model"
        assert status.model_type == "t2v-A14B"
        assert status.is_available is True
        assert status.availability_status == ModelAvailabilityStatus.AVAILABLE
        
        # Test default values
        assert status.integrity_score == 1.0
        assert status.performance_score == 1.0
        assert status.usage_frequency == 0.0
        assert status.priority == ModelPriority.MEDIUM
    
    def test_retention_policy_dataclass(self):
        """Test RetentionPolicy dataclass structure"""
        # Test default policy
        policy = RetentionPolicy()
        assert policy.max_unused_days == 30
        assert policy.max_storage_usage_percent == 80.0
        assert policy.min_usage_frequency == 0.1
        assert policy.preserve_recently_downloaded is True
        assert policy.preserve_high_priority is True
        
        # Test custom policy
        custom_policy = RetentionPolicy(
            max_unused_days=60,
            max_storage_usage_percent=90.0,
            min_usage_frequency=0.05
        )
        assert custom_policy.max_unused_days == 60
        assert custom_policy.max_storage_usage_percent == 90.0
        assert custom_policy.min_usage_frequency == 0.05


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])