"""
Tests for the Fallback and Recovery System
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from backend.core.fallback_recovery_system import (
    FallbackRecoverySystem, FailureType, RecoveryAction, 
    RecoveryAttempt, SystemHealthStatus
)

class TestFallbackRecoverySystem:
    """Test cases for the FallbackRecoverySystem"""
    
    @pytest.fixture
    def mock_generation_service(self):
        """Create a mock generation service"""
        service = Mock()
        service.use_real_generation = True
        service.fallback_to_simulation = True
        service.model_integration_bridge = Mock()
        service.real_generation_pipeline = Mock()
        service.wan22_system_optimizer = Mock()
        service.websocket_manager = Mock()
        return service
    
    @pytest.fixture
    def mock_websocket_manager(self):
        """Create a mock WebSocket manager"""
        manager = AsyncMock()
        manager.broadcast = AsyncMock()
        manager.send_alert = AsyncMock()
        return manager
    
    @pytest.fixture
    def recovery_system(self, mock_generation_service, mock_websocket_manager):
        """Create a FallbackRecoverySystem instance for testing"""
        return FallbackRecoverySystem(
            generation_service=mock_generation_service,
            websocket_manager=mock_websocket_manager
        )
    
    def test_initialization(self, recovery_system):
        """Test that the recovery system initializes correctly"""
        assert recovery_system is not None
        assert recovery_system.recovery_attempts == []
        assert recovery_system.max_recovery_attempts == 3
        assert recovery_system.recovery_cooldown_seconds == 30
        assert not recovery_system.mock_generation_enabled
        assert not recovery_system.degraded_mode_active
        assert not recovery_system.health_monitoring_active
    
    def test_recovery_strategies_initialization(self, recovery_system):
        """Test that recovery strategies are properly initialized"""
        strategies = recovery_system.recovery_strategies
        
        # Check that all failure types have strategies
        for failure_type in FailureType:
            assert failure_type in strategies
            assert len(strategies[failure_type]) > 0
        
        # Check specific strategies
        assert RecoveryAction.FALLBACK_TO_MOCK in strategies[FailureType.MODEL_LOADING_FAILURE]
        assert RecoveryAction.APPLY_VRAM_OPTIMIZATION in strategies[FailureType.VRAM_EXHAUSTION]
        assert RecoveryAction.RESTART_PIPELINE in strategies[FailureType.GENERATION_PIPELINE_ERROR]
    
    @pytest.mark.asyncio
    async def test_fallback_to_mock_generation(self, recovery_system):
        """Test fallback to mock generation"""
        # Test successful fallback
        success = await recovery_system._fallback_to_mock_generation()
        
        assert success is True
        assert recovery_system.mock_generation_enabled is True
        assert recovery_system.generation_service.use_real_generation is False
        assert recovery_system.generation_service.fallback_to_simulation is True
        
        # Verify WebSocket notification was sent
        recovery_system.websocket_manager.broadcast.assert_called_once()
        call_args = recovery_system.websocket_manager.broadcast.call_args[0][0]
        assert call_args["type"] == "system_status"
        assert call_args["data"]["mock_mode_enabled"] is True
    
    @pytest.mark.asyncio
    async def test_clear_gpu_cache(self, recovery_system):
        """Test GPU cache clearing"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache') as mock_empty_cache, \
             patch('gc.collect') as mock_gc_collect:
            
            success = await recovery_system._clear_gpu_cache()
            
            assert success is True
            mock_empty_cache.assert_called_once()
            mock_gc_collect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_clear_gpu_cache_no_cuda(self, recovery_system):
        """Test GPU cache clearing when CUDA is not available"""
        with patch('torch.cuda.is_available', return_value=False):
            success = await recovery_system._clear_gpu_cache()
            assert success is False
    
    @pytest.mark.asyncio
    async def test_retry_model_download(self, recovery_system):
        """Test model download retry logic"""
        # Mock successful download
        recovery_system.generation_service.model_integration_bridge.ensure_model_available = AsyncMock(return_value=True)
        
        context = {"model_type": "t2v-A14B"}
        success = await recovery_system._retry_model_download(context)
        
        assert success is True
        recovery_system.generation_service.model_integration_bridge.ensure_model_available.assert_called_with("t2v-A14B")
    
    @pytest.mark.asyncio
    async def test_retry_model_download_failure(self, recovery_system):
        """Test model download retry logic with failures"""
        # Mock failed downloads
        recovery_system.generation_service.model_integration_bridge.ensure_model_available = AsyncMock(return_value=False)
        
        context = {"model_type": "t2v-A14B"}
        success = await recovery_system._retry_model_download(context)
        
        assert success is False
        # Should have tried multiple times
        assert recovery_system.generation_service.model_integration_bridge.ensure_model_available.call_count == 3
    
    @pytest.mark.asyncio
    async def test_apply_vram_optimization(self, recovery_system):
        """Test VRAM optimization application"""
        # Mock successful optimization
        opt_result = Mock()
        opt_result.success = True
        opt_result.optimizations_applied = ["quantization", "offloading"]
        
        recovery_system.generation_service.wan22_system_optimizer.apply_hardware_optimizations.return_value = opt_result
        recovery_system.generation_service._apply_vram_optimizations = AsyncMock()
        
        success = await recovery_system._apply_vram_optimization({})
        
        assert success is True
        recovery_system.generation_service.wan22_system_optimizer.apply_hardware_optimizations.assert_called_once()
        recovery_system.generation_service._apply_vram_optimizations.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_restart_generation_pipeline(self, recovery_system):
        """Test generation pipeline restart"""
        # Mock successful pipeline restart
        recovery_system.generation_service.real_generation_pipeline.initialize = AsyncMock(return_value=True)
        recovery_system.generation_service.real_generation_pipeline._pipeline_cache = {}
        
        success = await recovery_system._restart_generation_pipeline()
        
        assert success is True
        recovery_system.generation_service.real_generation_pipeline.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_failure_success(self, recovery_system):
        """Test successful failure handling"""
        # Mock successful recovery action
        recovery_system._execute_recovery_action = AsyncMock(return_value=True)
        
        error = Exception("Test model loading error")
        success, message = await recovery_system.handle_failure(
            FailureType.MODEL_LOADING_FAILURE, error, {"test": "context"}
        )
        
        assert success is True
        assert "Recovery successful" in message
        assert len(recovery_system.recovery_attempts) == 1
        
        attempt = recovery_system.recovery_attempts[0]
        assert attempt.failure_type == FailureType.MODEL_LOADING_FAILURE
        assert attempt.success is True
    
    @pytest.mark.asyncio
    async def test_handle_failure_all_strategies_fail(self, recovery_system):
        """Test failure handling when all strategies fail"""
        # Mock all recovery actions to fail
        recovery_system._execute_recovery_action = AsyncMock(return_value=False)
        
        error = Exception("Test error")
        success, message = await recovery_system.handle_failure(
            FailureType.MODEL_LOADING_FAILURE, error, {}
        )
        
        assert success is False
        assert "All recovery strategies failed" in message
        
        # Should have attempted all strategies for model loading failure
        expected_strategies = recovery_system.recovery_strategies[FailureType.MODEL_LOADING_FAILURE]
        assert len(recovery_system.recovery_attempts) == len(expected_strategies)
    
    @pytest.mark.asyncio
    async def test_cooldown_mechanism(self, recovery_system):
        """Test recovery cooldown mechanism"""
        # Set a short cooldown for testing
        recovery_system.recovery_cooldown_seconds = 1
        
        # First recovery attempt
        recovery_system._execute_recovery_action = AsyncMock(return_value=True)
        error = Exception("Test error")
        
        success1, _ = await recovery_system.handle_failure(FailureType.MODEL_LOADING_FAILURE, error, {})
        assert success1 is True
        
        # Immediate second attempt should be blocked by cooldown
        success2, message2 = await recovery_system.handle_failure(FailureType.MODEL_LOADING_FAILURE, error, {})
        assert success2 is False
        assert "cooldown" in message2.lower()
        
        # Wait for cooldown to expire
        await asyncio.sleep(1.1)
        
        # Third attempt should work
        success3, _ = await recovery_system.handle_failure(FailureType.MODEL_LOADING_FAILURE, error, {})
        assert success3 is True
    
    @pytest.mark.asyncio
    async def test_get_system_health_status(self, recovery_system):
        """Test system health status retrieval"""
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.current_device', return_value=0), \
             patch('torch.cuda.get_device_properties') as mock_props, \
             patch('torch.cuda.memory_allocated', return_value=4 * 1024**3):  # 4GB
            
            # Mock memory info
            mock_memory.return_value.percent = 60.0
            
            # Mock GPU properties
            mock_props.return_value.total_memory = 16 * 1024**3  # 16GB
            
            health_status = await recovery_system.get_system_health_status()
            
            assert health_status.overall_status in ["healthy", "degraded", "critical"]
            assert health_status.cpu_usage_percent == 50.0
            assert health_status.memory_usage_percent == 60.0
            assert health_status.gpu_available is True
            assert 0 <= health_status.vram_usage_percent <= 100
    
    @pytest.mark.asyncio
    async def test_get_system_health_status_critical(self, recovery_system):
        """Test system health status with critical conditions"""
        with patch('psutil.cpu_percent', return_value=98.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.current_device', return_value=0), \
             patch('torch.cuda.get_device_properties') as mock_props, \
             patch('torch.cuda.memory_allocated', return_value=15 * 1024**3):  # 15GB
            
            # Mock high memory usage
            mock_memory.return_value.percent = 97.0
            
            # Mock GPU properties (16GB total)
            mock_props.return_value.total_memory = 16 * 1024**3
            
            health_status = await recovery_system.get_system_health_status()
            
            assert health_status.overall_status == "critical"
            assert len(health_status.issues) > 0
            assert len(health_status.recommendations) > 0
            assert "Critical CPU usage" in health_status.issues
            assert "Critical memory usage" in health_status.issues
    
    def test_get_recovery_statistics(self, recovery_system):
        """Test recovery statistics generation"""
        # Add some mock recovery attempts
        recovery_system.recovery_attempts = [
            RecoveryAttempt(
                failure_type=FailureType.MODEL_LOADING_FAILURE,
                action=RecoveryAction.FALLBACK_TO_MOCK,
                timestamp=datetime.now(),
                success=True,
                recovery_time_seconds=2.5
            ),
            RecoveryAttempt(
                failure_type=FailureType.VRAM_EXHAUSTION,
                action=RecoveryAction.APPLY_VRAM_OPTIMIZATION,
                timestamp=datetime.now(),
                success=False,
                recovery_time_seconds=1.0
            )
        ]
        
        stats = recovery_system.get_recovery_statistics()
        
        assert stats["total_attempts"] == 2
        assert stats["successful_attempts"] == 1
        assert stats["success_rate"] == 50.0
        assert stats["average_recovery_time"] == 2.5  # Only successful attempts counted
        assert "model_loading_failure" in stats["failure_types"]
        assert "vram_exhaustion" in stats["failure_types"]
    
    def test_reset_recovery_state(self, recovery_system):
        """Test recovery state reset"""
        # Set some state
        recovery_system.mock_generation_enabled = True
        recovery_system.degraded_mode_active = True
        recovery_system.critical_failures["test"] = datetime.now()
        recovery_system.last_recovery_attempt[FailureType.MODEL_LOADING_FAILURE] = datetime.now()
        
        # Reset state
        recovery_system.reset_recovery_state()
        
        assert recovery_system.mock_generation_enabled is False
        assert recovery_system.degraded_mode_active is False
        assert len(recovery_system.critical_failures) == 0
        assert len(recovery_system.last_recovery_attempt) == 0
        assert recovery_system.generation_service.use_real_generation is True
        assert recovery_system.generation_service.fallback_to_simulation is False
    
    def test_health_monitoring_start_stop(self, recovery_system):
        """Test health monitoring start and stop"""
        # Start monitoring
        assert not recovery_system.health_monitoring_active
        recovery_system.start_health_monitoring()
        assert recovery_system.health_monitoring_active
        assert recovery_system.health_monitor_thread is not None
        
        # Stop monitoring
        recovery_system.stop_health_monitoring()
        assert not recovery_system.health_monitoring_active
    
    @pytest.mark.asyncio
    async def test_reduce_generation_parameters(self, recovery_system):
        """Test automatic parameter reduction"""
        context = {
            "generation_params": {
                "resolution": "1920x1080",
                "steps": 50,
                "num_frames": 32
            }
        }
        
        success = await recovery_system._reduce_generation_parameters(context)
        
        assert success is True
        reduced_params = context["generation_params"]
        assert reduced_params["resolution"] == "1280x720"
        assert reduced_params["steps"] == 20
        assert reduced_params["num_frames"] == 8
        assert context["parameters_reduced"] is True
    
    @pytest.mark.asyncio
    async def test_enable_cpu_offload(self, recovery_system):
        """Test CPU offloading enablement"""
        success = await recovery_system._enable_cpu_offload()
        
        assert success is True
        assert recovery_system.generation_service.enable_model_offloading is True
    
    @pytest.mark.asyncio
    async def test_perform_system_health_check(self, recovery_system):
        """Test system health check performance"""
        # Mock healthy system
        recovery_system.get_system_health_status = AsyncMock()
        recovery_system.get_system_health_status.return_value = SystemHealthStatus(
            overall_status="healthy",
            cpu_usage_percent=50.0,
            memory_usage_percent=60.0,
            vram_usage_percent=40.0,
            gpu_available=True,
            model_loading_functional=True,
            generation_pipeline_functional=True,
            last_check_timestamp=datetime.now()
        )
        
        success = await recovery_system._perform_system_health_check()
        assert success is True
        
        # Mock critical system
        recovery_system.get_system_health_status.return_value.overall_status = "critical"
        success = await recovery_system._perform_system_health_check()
        assert success is False
        
        # Mock degraded system (should still return True)
        recovery_system.get_system_health_status.return_value.overall_status = "degraded"
        success = await recovery_system._perform_system_health_check()
        assert success is True


if __name__ == "__main__":
    pytest.main([__file__])