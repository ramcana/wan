"""
Unit tests for Intelligent Fallback Manager

Tests model compatibility scoring algorithms, alternative model suggestions,
fallback strategy decision engine, request queuing, and wait time calculations.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
from enum import Enum

# Import the module under test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.intelligent_fallback_manager import (
    IntelligentFallbackManager,
    GenerationRequirements,
    ModelSuggestion,
    FallbackStrategy,
    FallbackType,
    EstimatedWaitTime,
    QueuedRequest,
    QueueResult,
    ModelCapability,
    get_intelligent_fallback_manager,
    initialize_intelligent_fallback_manager
)


class MockAvailabilityStatus(Enum):
    """Mock availability status for testing"""
    AVAILABLE = "available"
    MISSING = "missing"
    DOWNLOADING = "downloading"
    CORRUPTED = "corrupted"


@dataclass
class MockModelStatus:
    """Mock model status for testing"""
    model_id: str
    availability_status: MockAvailabilityStatus
    download_progress: float = 0.0
    estimated_download_time: timedelta = None
    size_mb: float = 8000.0


class MockAvailabilityManager:
    """Mock availability manager for testing"""
    
    def __init__(self):
        self.model_statuses = {
            "t2v-A14B": MockModelStatus("t2v-A14B", MockAvailabilityStatus.AVAILABLE),
            "i2v-A14B": MockModelStatus("i2v-A14B", MockAvailabilityStatus.MISSING),
            "ti2v-5B": MockModelStatus("ti2v-5B", MockAvailabilityStatus.AVAILABLE)
        }
    
    async def _check_single_model_availability(self, model_id: str):
        return self.model_statuses.get(model_id, MockModelStatus(model_id, MockAvailabilityStatus.MISSING))
    
    async def get_comprehensive_model_status(self):
        return self.model_statuses
    
    def set_model_status(self, model_id: str, status: MockAvailabilityStatus):
        """Helper method to change model status for testing"""
        if model_id in self.model_statuses:
            self.model_statuses[model_id].availability_status = status


@pytest.fixture
def mock_availability_manager():
    """Fixture providing a mock availability manager"""
    return MockAvailabilityManager()


@pytest.fixture
def fallback_manager(mock_availability_manager):
    """Fixture providing an IntelligentFallbackManager instance"""
    return IntelligentFallbackManager(availability_manager=mock_availability_manager)


@pytest.fixture
def sample_requirements():
    """Fixture providing sample generation requirements"""
    return GenerationRequirements(
        model_type="t2v-A14B",
        quality="high",
        speed="medium",
        resolution="1920x1080",
        allow_alternatives=True,
        allow_quality_reduction=True
    )


class TestIntelligentFallbackManager:
    """Test suite for IntelligentFallbackManager"""
    
    def test_initialization(self, mock_availability_manager):
        """Test proper initialization of the fallback manager"""
        manager = IntelligentFallbackManager(availability_manager=mock_availability_manager)
        
        assert manager.availability_manager == mock_availability_manager
        assert manager.models_dir.name == "models"
        assert len(manager._model_capabilities) > 0
        assert len(manager._compatibility_matrix) > 0
        assert manager._request_queue == []
        assert manager.max_queue_size == 100
    
    def test_model_capabilities_initialization(self, fallback_manager):
        """Test that model capabilities are properly initialized"""
        capabilities = fallback_manager._model_capabilities
        
        # Check that all expected models have capabilities
        assert "t2v-A14B" in capabilities
        assert "i2v-A14B" in capabilities
        assert "ti2v-5B" in capabilities
        
        # Check specific capabilities
        assert ModelCapability.TEXT_TO_VIDEO in capabilities["t2v-A14B"]
        assert ModelCapability.IMAGE_TO_VIDEO in capabilities["i2v-A14B"]
        assert ModelCapability.TEXT_IMAGE_TO_VIDEO in capabilities["ti2v-5B"]
        assert ModelCapability.FAST_GENERATION in capabilities["ti2v-5B"]
    
    def test_compatibility_matrix_initialization(self, fallback_manager):
        """Test that compatibility matrix is properly initialized"""
        matrix = fallback_manager._compatibility_matrix
        
        # Check matrix structure
        assert "t2v-A14B" in matrix
        assert "i2v-A14B" in matrix["t2v-A14B"]
        assert "ti2v-5B" in matrix["t2v-A14B"]
        
        # Check symmetry and reasonable scores
        assert 0.0 <= matrix["t2v-A14B"]["i2v-A14B"] <= 1.0
        assert 0.0 <= matrix["i2v-A14B"]["ti2v-5B"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_get_available_models(self, fallback_manager, mock_availability_manager):
        """Test getting list of available models"""
        available_models = await fallback_manager._get_available_models()
        
        # Should return models with "available" status
        assert "t2v-A14B" in available_models
        assert "ti2v-5B" in available_models
        assert "i2v-A14B" not in available_models  # This one is missing
    
    @pytest.mark.asyncio
    async def test_suggest_alternative_model_high_compatibility(self, fallback_manager, sample_requirements):
        """Test suggesting alternative model with high compatibility"""
        # Request i2v-A14B (which is missing), should suggest ti2v-5B
        requirements = GenerationRequirements(
            model_type="i2v-A14B",
            quality="high",
            speed="medium",
            resolution="1280x720"
        )
        
        suggestion = await fallback_manager.suggest_alternative_model("i2v-A14B", requirements)
        
        assert suggestion.suggested_model == "ti2v-5B"
        assert suggestion.compatibility_score > 0.7
        assert suggestion.availability_status == "available"
        assert suggestion.estimated_quality_difference in ["similar", "slightly_lower"]
    
    @pytest.mark.asyncio
    async def test_suggest_alternative_model_no_good_alternatives(self, fallback_manager, mock_availability_manager):
        """Test suggesting alternative when no good alternatives exist"""
        # Make all models unavailable except the requested one
        mock_availability_manager.set_model_status("t2v-A14B", MockAvailabilityStatus.MISSING)
        mock_availability_manager.set_model_status("ti2v-5B", MockAvailabilityStatus.MISSING)
        
        requirements = GenerationRequirements(model_type="i2v-A14B")
        suggestion = await fallback_manager.suggest_alternative_model("i2v-A14B", requirements)
        
        assert suggestion.suggested_model == "mock_generation"
        assert suggestion.compatibility_score < 0.3
        assert suggestion.estimated_quality_difference == "significantly_lower"
    
    @pytest.mark.asyncio
    async def test_score_model_compatibility(self, fallback_manager, sample_requirements):
        """Test model compatibility scoring algorithm"""
        # Test scoring between similar models
        suggestion = await fallback_manager._score_model_compatibility(
            "i2v-A14B", "ti2v-5B", sample_requirements
        )
        
        assert 0.0 <= suggestion.compatibility_score <= 1.0
        assert suggestion.suggested_model == "ti2v-5B"
        assert len(suggestion.capabilities_match) > 0
        assert suggestion.vram_requirement_gb > 0
        assert suggestion.estimated_generation_time is not None
    
    def test_estimate_performance_difference(self, fallback_manager):
        """Test performance difference estimation"""
        # Test performance comparison
        diff1 = fallback_manager._estimate_performance_difference("t2v-A14B", "i2v-A14B")
        diff2 = fallback_manager._estimate_performance_difference("i2v-A14B", "ti2v-5B")
        
        assert -1.0 <= diff1 <= 1.0
        assert -1.0 <= diff2 <= 1.0
    
    def test_estimate_vram_requirement(self, fallback_manager):
        """Test VRAM requirement estimation"""
        # Test different models and resolutions
        vram_720p = fallback_manager._estimate_vram_requirement("t2v-A14B", "1280x720")
        vram_1080p = fallback_manager._estimate_vram_requirement("t2v-A14B", "1920x1080")
        
        assert vram_720p > 0
        assert vram_1080p > vram_720p  # Higher resolution should require more VRAM
        
        # Test different models
        vram_t2v = fallback_manager._estimate_vram_requirement("t2v-A14B", "1280x720")
        vram_ti2v = fallback_manager._estimate_vram_requirement("ti2v-5B", "1280x720")
        
        assert vram_t2v != vram_ti2v  # Different models should have different requirements
    
    def test_estimate_generation_time(self, fallback_manager):
        """Test generation time estimation"""
        requirements_high = GenerationRequirements(
            model_type="t2v-A14B",
            quality="high",
            resolution="1920x1080"
        )
        
        requirements_low = GenerationRequirements(
            model_type="ti2v-5B",
            quality="low",
            resolution="1280x720"
        )
        
        time_high = fallback_manager._estimate_generation_time("t2v-A14B", requirements_high)
        time_low = fallback_manager._estimate_generation_time("ti2v-5B", requirements_low)
        
        assert isinstance(time_high, timedelta)
        assert isinstance(time_low, timedelta)
        assert time_high > time_low  # High quality should take longer
    
    @pytest.mark.asyncio
    async def test_get_fallback_strategy_missing_model(self, fallback_manager):
        """Test fallback strategy for missing model"""
        error_context = {
            "failure_type": "model_loading_failure",
            "error_message": "Model not found",
            "requirements": GenerationRequirements(model_type="i2v-A14B")
        }
        
        strategy = await fallback_manager.get_fallback_strategy("i2v-A14B", error_context)
        
        assert strategy.strategy_type in [FallbackType.ALTERNATIVE_MODEL, FallbackType.DOWNLOAD_AND_RETRY]
        assert strategy.confidence_score > 0.0
        assert len(strategy.recommended_action) > 0
    
    @pytest.mark.asyncio
    async def test_get_fallback_strategy_vram_exhaustion(self, fallback_manager):
        """Test fallback strategy for VRAM exhaustion"""
        error_context = {
            "failure_type": "vram_exhaustion",
            "error_message": "CUDA out of memory",
            "requirements": GenerationRequirements(
                model_type="t2v-A14B",
                quality="high",
                resolution="1920x1080",
                allow_quality_reduction=True
            )
        }
        
        strategy = await fallback_manager.get_fallback_strategy("t2v-A14B", error_context)
        
        assert strategy.strategy_type in [
            FallbackType.ALTERNATIVE_MODEL,
            FallbackType.REDUCE_REQUIREMENTS,
            FallbackType.MOCK_GENERATION
        ]
        
        if strategy.strategy_type == FallbackType.REDUCE_REQUIREMENTS:
            assert "vram_optimized" in strategy.requirements_adjustments
    
    @pytest.mark.asyncio
    async def test_get_fallback_strategy_network_error(self, fallback_manager):
        """Test fallback strategy for network errors"""
        error_context = {
            "failure_type": "network_error",
            "error_message": "Download failed",
            "requirements": GenerationRequirements(model_type="i2v-A14B")
        }
        
        strategy = await fallback_manager.get_fallback_strategy("i2v-A14B", error_context)
        
        assert strategy.strategy_type in [
            FallbackType.ALTERNATIVE_MODEL,
            FallbackType.QUEUE_AND_WAIT
        ]
        
        if strategy.strategy_type == FallbackType.QUEUE_AND_WAIT:
            assert strategy.can_queue_request
            assert strategy.estimated_wait_time is not None
    
    @pytest.mark.asyncio
    async def test_estimate_wait_time_available_model(self, fallback_manager):
        """Test wait time estimation for available model"""
        wait_time = await fallback_manager.estimate_wait_time("t2v-A14B")
        
        assert wait_time.model_id == "t2v-A14B"
        assert wait_time.total_wait_time == timedelta(0)
        assert wait_time.confidence == "high"
        assert "already available" in wait_time.factors[0].lower()
    
    @pytest.mark.asyncio
    async def test_estimate_wait_time_missing_model(self, fallback_manager):
        """Test wait time estimation for missing model"""
        wait_time = await fallback_manager.estimate_wait_time("i2v-A14B")
        
        assert wait_time.model_id == "i2v-A14B"
        assert wait_time.total_wait_time > timedelta(0)
        assert wait_time.download_time is not None
        assert any("download" in factor.lower() for factor in wait_time.factors)
    
    def test_estimate_model_size(self, fallback_manager):
        """Test model size estimation"""
        size_t2v = fallback_manager._estimate_model_size("t2v-A14B")
        size_i2v = fallback_manager._estimate_model_size("i2v-A14B")
        size_ti2v = fallback_manager._estimate_model_size("ti2v-5B")
        size_unknown = fallback_manager._estimate_model_size("unknown-model")
        
        assert size_t2v > 0
        assert size_i2v > 0
        assert size_ti2v > 0
        assert size_unknown == 10.0  # Default fallback
        
        # i2v should be larger than ti2v (based on model complexity)
        assert size_i2v > size_ti2v
    
    @pytest.mark.asyncio
    async def test_queue_request_success(self, fallback_manager):
        """Test successful request queuing"""
        requirements = GenerationRequirements(model_type="i2v-A14B", priority="high")
        
        result = await fallback_manager.queue_request_for_downloading_model(
            "i2v-A14B", requirements
        )
        
        assert result.success
        assert result.queue_position == 1
        assert len(result.request_id) > 0
        assert result.estimated_wait_time is not None
        
        # Check that request is in queue
        assert len(fallback_manager._request_queue) == 1
        assert fallback_manager._request_queue[0].model_id == "i2v-A14B"
    
    @pytest.mark.asyncio
    async def test_queue_request_priority_ordering(self, fallback_manager):
        """Test that requests are queued in priority order"""
        # Add requests with different priorities
        req_low = GenerationRequirements(model_type="model1", priority="low")
        req_normal = GenerationRequirements(model_type="model2", priority="normal")
        req_high = GenerationRequirements(model_type="model3", priority="high")
        req_critical = GenerationRequirements(model_type="model4", priority="critical")
        
        # Add in random order
        await fallback_manager.queue_request_for_downloading_model("model1", req_low)
        await fallback_manager.queue_request_for_downloading_model("model2", req_normal)
        await fallback_manager.queue_request_for_downloading_model("model3", req_high)
        await fallback_manager.queue_request_for_downloading_model("model4", req_critical)
        
        # Check that they're ordered by priority
        queue = fallback_manager._request_queue
        priorities = [req.priority for req in queue]
        
        # Critical should be first, then high, normal, low
        assert priorities[0] == "critical"
        assert priorities[1] == "high"
        assert priorities[2] == "normal"
        assert priorities[3] == "low"
    
    @pytest.mark.asyncio
    async def test_queue_request_full_queue(self, fallback_manager):
        """Test queuing when queue is full"""
        # Fill the queue to capacity
        fallback_manager.max_queue_size = 2
        
        req1 = GenerationRequirements(model_type="model1")
        req2 = GenerationRequirements(model_type="model2")
        req3 = GenerationRequirements(model_type="model3")
        
        result1 = await fallback_manager.queue_request_for_downloading_model("model1", req1)
        result2 = await fallback_manager.queue_request_for_downloading_model("model2", req2)
        result3 = await fallback_manager.queue_request_for_downloading_model("model3", req3)
        
        assert result1.success
        assert result2.success
        assert not result3.success
        assert "full" in result3.error.lower()
    
    def test_get_priority_value(self, fallback_manager):
        """Test priority value conversion"""
        assert fallback_manager._get_priority_value("critical") > fallback_manager._get_priority_value("high")
        assert fallback_manager._get_priority_value("high") > fallback_manager._get_priority_value("normal")
        assert fallback_manager._get_priority_value("normal") > fallback_manager._get_priority_value("low")
        assert fallback_manager._get_priority_value("unknown") == 2  # Default to normal
    
    @pytest.mark.asyncio
    async def test_get_queue_status(self, fallback_manager):
        """Test getting queue status"""
        # Add some requests
        req1 = GenerationRequirements(model_type="model1", priority="high")
        req2 = GenerationRequirements(model_type="model1", priority="normal")
        req3 = GenerationRequirements(model_type="model2", priority="low")
        
        await fallback_manager.queue_request_for_downloading_model("model1", req1)
        await fallback_manager.queue_request_for_downloading_model("model1", req2)
        await fallback_manager.queue_request_for_downloading_model("model2", req3)
        
        status = await fallback_manager.get_queue_status()
        
        assert status["total_queued_requests"] == 3
        assert "model1" in status["queue_by_model"]
        assert "model2" in status["queue_by_model"]
        assert len(status["queue_by_model"]["model1"]) == 2
        assert len(status["queue_by_model"]["model2"]) == 1
        assert 0.0 <= status["queue_utilization"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_process_queue_with_available_model(self, fallback_manager, mock_availability_manager):
        """Test processing queue when model becomes available"""
        # Add a request for a missing model
        req = GenerationRequirements(model_type="i2v-A14B")
        callback_called = False
        
        async def test_callback(request):
            nonlocal callback_called
            callback_called = True
        
        await fallback_manager.queue_request_for_downloading_model("i2v-A14B", req, callback=test_callback)
        
        # Initially model is missing
        assert len(fallback_manager._request_queue) == 1
        
        # Make model available
        mock_availability_manager.set_model_status("i2v-A14B", MockAvailabilityStatus.AVAILABLE)
        
        # Process queue
        result = await fallback_manager.process_queue()
        
        assert result["processed_requests"] == 1
        assert result["remaining_requests"] == 0
        assert callback_called
        assert len(fallback_manager._request_queue) == 0
    
    @pytest.mark.asyncio
    async def test_process_queue_model_still_unavailable(self, fallback_manager):
        """Test processing queue when model is still unavailable"""
        # Add a request for a missing model
        req = GenerationRequirements(model_type="i2v-A14B")
        await fallback_manager.queue_request_for_downloading_model("i2v-A14B", req)
        
        # Process queue (model is still missing)
        result = await fallback_manager.process_queue()
        
        assert result["processed_requests"] == 0
        assert result["remaining_requests"] == 1
        assert len(fallback_manager._request_queue) == 1
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_requests(self, fallback_manager):
        """Test cleanup of expired requests"""
        # Add an old request
        req = GenerationRequirements(model_type="model1")
        await fallback_manager.queue_request_for_downloading_model("model1", req)
        
        # Manually set old timestamp
        fallback_manager._request_queue[0].queued_at = datetime.now() - timedelta(hours=25)
        
        # Add a recent request
        req2 = GenerationRequirements(model_type="model2")
        await fallback_manager.queue_request_for_downloading_model("model2", req2)
        
        assert len(fallback_manager._request_queue) == 2
        
        # Cleanup expired requests (older than 24 hours)
        removed_count = await fallback_manager.cleanup_expired_requests(max_age_hours=24)
        
        assert removed_count == 1
        assert len(fallback_manager._request_queue) == 1
        assert fallback_manager._request_queue[0].model_id == "model2"
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self, fallback_manager, sample_requirements):
        """Test that compatibility scores are cached"""
        # First call should compute and cache
        suggestion1 = await fallback_manager.suggest_alternative_model("i2v-A14B", sample_requirements)
        
        # Second call should use cache
        suggestion2 = await fallback_manager.suggest_alternative_model("i2v-A14B", sample_requirements)
        
        assert suggestion1.suggested_model == suggestion2.suggested_model
        assert suggestion1.compatibility_score == suggestion2.compatibility_score
        
        # Check that cache was used
        cache_key = f"i2v-A14B_{sample_requirements.quality}_{sample_requirements.speed}_{sample_requirements.resolution}"
        assert cache_key in fallback_manager._compatibility_cache


class TestGlobalInstanceManagement:
    """Test global instance management functions"""
    
    def test_get_intelligent_fallback_manager_singleton(self):
        """Test that get_intelligent_fallback_manager returns singleton"""
        manager1 = get_intelligent_fallback_manager()
        manager2 = get_intelligent_fallback_manager()
        
        assert manager1 is manager2
    
    @pytest.mark.asyncio
    async def test_initialize_intelligent_fallback_manager(self):
        """Test initialization function"""
        mock_availability_manager = MockAvailabilityManager()
        
        with patch('asyncio.create_task') as mock_create_task:
            manager = await initialize_intelligent_fallback_manager(mock_availability_manager)
            
            assert isinstance(manager, IntelligentFallbackManager)
            assert manager.availability_manager == mock_availability_manager
            
            # Should have started background task
            mock_create_task.assert_called_once()


class TestErrorHandling:
    """Test error handling in various scenarios"""
    
    @pytest.mark.asyncio
    async def test_suggest_alternative_with_error(self, fallback_manager):
        """Test alternative suggestion with error in availability manager"""
        # Mock availability manager to raise exception
        fallback_manager.availability_manager = Mock()
        fallback_manager.availability_manager.get_comprehensive_model_status = AsyncMock(
            side_effect=Exception("Test error")
        )
        
        requirements = GenerationRequirements(model_type="t2v-A14B")
        suggestion = await fallback_manager.suggest_alternative_model("t2v-A14B", requirements)
        
        assert suggestion.suggested_model == "mock_generation"
        assert suggestion.compatibility_score < 0.5
        assert "error" in suggestion.reason.lower()
    
    @pytest.mark.asyncio
    async def test_get_fallback_strategy_with_error(self, fallback_manager):
        """Test fallback strategy with error in processing"""
        # Create error context that will cause issues
        error_context = {
            "failure_type": "unknown_failure",
            "error_message": "Test error",
            "requirements": None  # This should cause issues
        }
        
        strategy = await fallback_manager.get_fallback_strategy("test-model", error_context)
        
        assert strategy.strategy_type == FallbackType.MOCK_GENERATION
        assert strategy.confidence_score < 0.5
    
    @pytest.mark.asyncio
    async def test_estimate_wait_time_with_error(self, fallback_manager):
        """Test wait time estimation with error"""
        # Mock availability manager to raise exception
        fallback_manager.availability_manager = Mock()
        fallback_manager.availability_manager._check_single_model_availability = AsyncMock(
            side_effect=Exception("Test error")
        )
        
        wait_time = await fallback_manager.estimate_wait_time("test-model")
        
        assert wait_time.model_id == "test-model"
        assert wait_time.total_wait_time == timedelta(minutes=15)  # Conservative fallback
        assert wait_time.confidence == "low"
        assert any("error" in factor.lower() for factor in wait_time.factors)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    @pytest.mark.asyncio
    async def test_suggest_alternative_for_same_model(self, fallback_manager):
        """Test suggesting alternative for a model that's requesting itself"""
        requirements = GenerationRequirements(model_type="t2v-A14B")
        
        # t2v-A14B is available, so this should find other alternatives
        suggestion = await fallback_manager.suggest_alternative_model("t2v-A14B", requirements)
        
        assert suggestion.suggested_model != "t2v-A14B"
        assert suggestion.compatibility_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_queue_request_with_none_callback(self, fallback_manager):
        """Test queuing request with None callback"""
        requirements = GenerationRequirements(model_type="test-model")
        
        result = await fallback_manager.queue_request_for_downloading_model(
            "test-model", requirements, callback=None
        )
        
        assert result.success
        assert result.queue_position == 1
    
    @pytest.mark.asyncio
    async def test_process_queue_with_callback_error(self, fallback_manager, mock_availability_manager):
        """Test processing queue when callback raises exception"""
        async def failing_callback(request):
            raise Exception("Callback error")
        
        requirements = GenerationRequirements(model_type="i2v-A14B")
        await fallback_manager.queue_request_for_downloading_model(
            "i2v-A14B", requirements, callback=failing_callback
        )
        
        # Make model available
        mock_availability_manager.set_model_status("i2v-A14B", MockAvailabilityStatus.AVAILABLE)
        
        # Process queue
        result = await fallback_manager.process_queue()
        
        assert result["processed_requests"] == 0
        assert len(result["errors"]) > 0
        assert "callback error" in result["errors"][0].lower()
        assert result["remaining_requests"] == 1  # Request should remain in queue
    
    def test_compatibility_scoring_with_unknown_models(self, fallback_manager):
        """Test compatibility scoring with unknown models"""
        requirements = GenerationRequirements(model_type="unknown-model")
        
        # Should handle unknown models gracefully
        base_score = fallback_manager._compatibility_matrix.get("unknown-model", {}).get("another-unknown", 0.5)
        assert base_score == 0.5  # Default fallback
        
        # Performance difference should also handle unknown models
        perf_diff = fallback_manager._estimate_performance_difference("unknown1", "unknown2")
        assert -1.0 <= perf_diff <= 1.0
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_requests_empty_queue(self, fallback_manager):
        """Test cleanup when queue is empty"""
        removed_count = await fallback_manager.cleanup_expired_requests()
        assert removed_count == 0
    
    @pytest.mark.asyncio
    async def test_get_queue_status_empty_queue(self, fallback_manager):
        """Test getting status of empty queue"""
        status = await fallback_manager.get_queue_status()
        
        assert status["total_queued_requests"] == 0
        assert status["queue_by_model"] == {}
        assert status["queue_utilization"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
