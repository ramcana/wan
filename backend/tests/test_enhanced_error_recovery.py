"""
Tests for Enhanced Error Recovery System

This module tests the sophisticated error categorization, multi-strategy recovery,
intelligent fallback integration, automatic repair triggers, and user-friendly
error messages with actionable recovery steps.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

# Import the enhanced error recovery system
from backend.core.enhanced_error_recovery import (
    EnhancedErrorRecovery,
    EnhancedFailureType,
    RecoveryStrategy,
    ErrorSeverity,
    ErrorContext,
    RecoveryResult,
    RecoveryMetrics,
    create_enhanced_error_recovery
)

# Import related components for mocking
from backend.core.fallback_recovery_system import (
    FallbackRecoverySystem, FailureType, RecoveryAction
)


class TestEnhancedErrorRecovery:
    """Test suite for EnhancedErrorRecovery class"""
    
    @pytest.fixture
    def mock_base_recovery(self):
        """Create mock base recovery system"""
        mock = AsyncMock(spec=FallbackRecoverySystem)
        mock._clear_gpu_cache = AsyncMock(return_value=True)
        mock._apply_vram_optimization = AsyncMock(return_value=True)
        mock._enable_cpu_offload = AsyncMock(return_value=True)
        mock._restart_generation_pipeline = AsyncMock(return_value=True)
        mock._fallback_to_mock_generation = AsyncMock(return_value=True)
        mock.handle_failure = AsyncMock(return_value=(True, "Recovery successful"))
        return mock
    
    @pytest.fixture
    def mock_availability_manager(self):
        """Create mock model availability manager"""
        mock = AsyncMock()
        mock.get_model_status = AsyncMock()
        mock.get_cleanup_suggestions = AsyncMock(return_value=["cleanup_suggestion"])
        return mock
    
    @pytest.fixture
    def mock_fallback_manager(self):
        """Create mock intelligent fallback manager"""
        mock = AsyncMock()
        mock.get_fallback_strategy = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_health_monitor(self):
        """Create mock model health monitor"""
        mock = AsyncMock()
        mock.check_model_integrity = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_enhanced_downloader(self):
        """Create mock enhanced model downloader"""
        mock = AsyncMock()
        mock.download_with_retry = AsyncMock()
        mock.verify_and_repair_model = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_websocket_manager(self):
        """Create mock WebSocket manager"""
        mock = AsyncMock()
        mock.broadcast = AsyncMock()
        return mock
    
    @pytest.fixture
    def enhanced_recovery(
        self,
        mock_base_recovery,
        mock_availability_manager,
        mock_fallback_manager,
        mock_health_monitor,
        mock_enhanced_downloader,
        mock_websocket_manager
    ):
        """Create enhanced error recovery system with mocked dependencies"""
        return EnhancedErrorRecovery(
            base_recovery_system=mock_base_recovery,
            model_availability_manager=mock_availability_manager,
            intelligent_fallback_manager=mock_fallback_manager,
            model_health_monitor=mock_health_monitor,
            enhanced_downloader=mock_enhanced_downloader,
            websocket_manager=mock_websocket_manager
        ) 
   
    def test_initialization(self, enhanced_recovery):
        """Test enhanced error recovery system initialization"""
        assert enhanced_recovery is not None
        assert enhanced_recovery.max_recovery_attempts == 5
        assert enhanced_recovery.recovery_timeout_seconds == 300
        assert enhanced_recovery.user_intervention_threshold == 3
        assert len(enhanced_recovery.strategy_mapping) > 0
        assert len(enhanced_recovery.error_messages) > 0
    
    def test_strategy_mapping_completeness(self, enhanced_recovery):
        """Test that all enhanced failure types have recovery strategies"""
        for failure_type in EnhancedFailureType:
            assert failure_type in enhanced_recovery.strategy_mapping
            strategies = enhanced_recovery.strategy_mapping[failure_type]
            assert len(strategies) > 0
            assert all(isinstance(s, RecoveryStrategy) for s in strategies)
    
    def test_error_messages_completeness(self, enhanced_recovery):
        """Test that error messages are properly configured"""
        # Check that key failure types have error messages
        key_failure_types = [
            EnhancedFailureType.MODEL_DOWNLOAD_FAILURE,
            EnhancedFailureType.MODEL_CORRUPTION_DETECTED,
            EnhancedFailureType.VRAM_EXHAUSTION,
            EnhancedFailureType.NETWORK_CONNECTIVITY_LOSS
        ]
        
        for failure_type in key_failure_types:
            assert failure_type in enhanced_recovery.error_messages
            error_info = enhanced_recovery.error_messages[failure_type]
            assert "title" in error_info
            assert "message" in error_info
            assert "user_message" in error_info
            assert "steps" in error_info
    
    @pytest.mark.asyncio
    async def test_categorize_error_model_download_failure(self, enhanced_recovery):
        """Test error categorization for model download failures"""
        error = Exception("Failed to download model from server")
        context = {
            "model_id": "test-model",
            "operation": "download",
            "user_parameters": {"quality": "high"}
        }
        
        error_context = await enhanced_recovery.categorize_error(error, context)
        
        assert error_context.failure_type == EnhancedFailureType.MODEL_DOWNLOAD_FAILURE
        assert error_context.model_id == "test-model"
        assert error_context.operation == "download"
        assert error_context.user_parameters["quality"] == "high"
        assert error_context.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM, ErrorSeverity.HIGH]
    
    @pytest.mark.asyncio
    async def test_categorize_error_vram_exhaustion(self, enhanced_recovery):
        """Test error categorization for VRAM exhaustion"""
        error = Exception("CUDA out of memory")
        context = {
            "model_id": "large-model",
            "operation": "generation"
        }
        
        error_context = await enhanced_recovery.categorize_error(error, context)
        
        assert error_context.failure_type == EnhancedFailureType.VRAM_EXHAUSTION
        assert error_context.severity == ErrorSeverity.HIGH
    
    @pytest.mark.asyncio
    async def test_categorize_error_corruption_detected(self, enhanced_recovery):
        """Test error categorization for model corruption"""
        error = Exception("Model file checksum mismatch - corruption detected")
        context = {"model_id": "corrupted-model"}
        
        error_context = await enhanced_recovery.categorize_error(error, context)
        
        assert error_context.failure_type == EnhancedFailureType.MODEL_CORRUPTION_DETECTED
        assert error_context.severity == ErrorSeverity.HIGH
    
    @pytest.mark.asyncio
    async def test_categorize_error_network_connectivity(self, enhanced_recovery):
        """Test error categorization for network issues"""
        error = Exception("Connection timeout - server unreachable")
        context = {"operation": "download"}
        
        error_context = await enhanced_recovery.categorize_error(error, context)
        
        assert error_context.failure_type == EnhancedFailureType.NETWORK_CONNECTIVITY_LOSS
        assert error_context.severity == ErrorSeverity.MEDIUM
    
    @pytest.mark.asyncio
    async def test_categorize_error_permission_denied(self, enhanced_recovery):
        """Test error categorization for permission issues"""
        error = Exception("Permission denied - access to file forbidden")
        context = {"operation": "file_access"}
        
        error_context = await enhanced_recovery.categorize_error(error, context)
        
        assert error_context.failure_type == EnhancedFailureType.PERMISSION_DENIED
        assert error_context.severity == ErrorSeverity.CRITICAL
    
    @pytest.mark.asyncio
    async def test_immediate_retry_strategy_success(self, enhanced_recovery, mock_enhanced_downloader):
        """Test successful immediate retry strategy"""
        # Mock successful download retry
        mock_result = Mock()
        mock_result.success = True
        mock_enhanced_downloader.download_with_retry.return_value = mock_result
        
        error_context = ErrorContext(
            failure_type=EnhancedFailureType.MODEL_DOWNLOAD_FAILURE,
            original_error=Exception("Download failed"),
            severity=ErrorSeverity.MEDIUM,
            model_id="test-model"
        )
        
        result = await enhanced_recovery._immediate_retry_strategy(error_context)
        
        assert result.success is True
        assert result.strategy_used == RecoveryStrategy.IMMEDIATE_RETRY
        assert "successful" in result.message.lower()
        assert "retry" in result.user_message.lower()
        mock_enhanced_downloader.download_with_retry.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_immediate_retry_strategy_failure(self, enhanced_recovery, mock_enhanced_downloader):
        """Test failed immediate retry strategy"""
        # Mock failed download retry
        mock_result = Mock()
        mock_result.success = False
        mock_enhanced_downloader.download_with_retry.return_value = mock_result
        
        error_context = ErrorContext(
            failure_type=EnhancedFailureType.MODEL_DOWNLOAD_FAILURE,
            original_error=Exception("Download failed"),
            severity=ErrorSeverity.MEDIUM,
            model_id="test-model"
        )
        
        result = await enhanced_recovery._immediate_retry_strategy(error_context)
        
        assert result.success is False
        assert result.strategy_used == RecoveryStrategy.IMMEDIATE_RETRY
        assert "failed" in result.message.lower()
        assert result.system_changes["all_retries_failed"] is True
    
    @pytest.mark.asyncio
    async def test_intelligent_fallback_strategy_alternative_model(
        self, enhanced_recovery, mock_fallback_manager, mock_availability_manager
    ):
        """Test intelligent fallback with alternative model"""
        # Mock fallback strategy with alternative model
        from backend.core.intelligent_fallback_manager import FallbackStrategy, FallbackType as EnhancedFallbackType
        
        mock_strategy = Mock()
        mock_strategy.strategy_type = EnhancedFallbackType.ALTERNATIVE_MODEL
        mock_strategy.alternative_model = "alternative-model"
        mock_strategy.recommended_action = "Use compatible alternative"
        mock_fallback_manager.get_fallback_strategy.return_value = mock_strategy
        
        # Mock model availability check
        mock_status = Mock()
        mock_status.is_available = True
        mock_availability_manager.get_model_status.return_value = mock_status
        
        error_context = ErrorContext(
            failure_type=EnhancedFailureType.MODEL_DOWNLOAD_FAILURE,
            original_error=Exception("Model unavailable"),
            severity=ErrorSeverity.MEDIUM,
            model_id="unavailable-model"
        )
        
        result = await enhanced_recovery._intelligent_fallback_strategy(error_context)
        
        assert result.success is True
        assert result.strategy_used == RecoveryStrategy.INTELLIGENT_FALLBACK
        assert "alternative" in result.message.lower()
        assert len(result.alternative_options) > 0
        assert result.system_changes["alternative_model_used"] == "alternative-model"
    
    @pytest.mark.asyncio
    async def test_automatic_repair_strategy_corruption(
        self, enhanced_recovery, mock_health_monitor, mock_enhanced_downloader
    ):
        """Test automatic repair strategy for corruption"""
        # Mock integrity check failure
        mock_integrity = Mock()
        mock_integrity.is_healthy = False
        mock_health_monitor.check_model_integrity.return_value = mock_integrity
        
        # Mock successful repair
        mock_repair = Mock()
        mock_repair.success = True
        mock_enhanced_downloader.verify_and_repair_model.return_value = mock_repair
        
        error_context = ErrorContext(
            failure_type=EnhancedFailureType.MODEL_CORRUPTION_DETECTED,
            original_error=Exception("Corruption detected"),
            severity=ErrorSeverity.HIGH,
            model_id="corrupted-model"
        )
        
        result = await enhanced_recovery._automatic_repair_strategy(error_context)
        
        assert result.success is True
        assert result.strategy_used == RecoveryStrategy.AUTOMATIC_REPAIR
        assert "repaired" in result.message.lower()
        assert result.system_changes["model_repaired"] == "corrupted-model"
        mock_health_monitor.check_model_integrity.assert_called_once_with("corrupted-model")
        mock_enhanced_downloader.verify_and_repair_model.assert_called_once_with("corrupted-model")
    
    @pytest.mark.asyncio
    async def test_parameter_adjustment_strategy_vram_exhaustion(self, enhanced_recovery):
        """Test parameter adjustment for VRAM exhaustion"""
        error_context = ErrorContext(
            failure_type=EnhancedFailureType.VRAM_EXHAUSTION,
            original_error=Exception("CUDA out of memory"),
            severity=ErrorSeverity.HIGH,
            user_parameters={
                "resolution": "1920x1080",
                "steps": 30,
                "num_frames": 24
            }
        )
        
        result = await enhanced_recovery._parameter_adjustment_strategy(error_context)
        
        assert result.success is True
        assert result.strategy_used == RecoveryStrategy.PARAMETER_ADJUSTMENT
        assert error_context.user_parameters["resolution"] == "1280x720"
        assert error_context.user_parameters["steps"] == 20
        assert error_context.user_parameters["num_frames"] == 16
        assert len(result.system_changes["adjustments_made"]) > 0
    
    @pytest.mark.asyncio
    async def test_parameter_adjustment_strategy_invalid_parameters(self, enhanced_recovery):
        """Test parameter adjustment for invalid parameters"""
        error_context = ErrorContext(
            failure_type=EnhancedFailureType.INVALID_PARAMETERS,
            original_error=Exception("Invalid resolution"),
            severity=ErrorSeverity.MEDIUM,
            user_parameters={
                "resolution": "invalid_resolution",
                "steps": -5,
                "num_frames": 100
            }
        )
        
        result = await enhanced_recovery._parameter_adjustment_strategy(error_context)
        
        assert result.success is True
        assert result.strategy_used == RecoveryStrategy.PARAMETER_ADJUSTMENT
        assert error_context.user_parameters["resolution"] == "1280x720"
        assert error_context.user_parameters["steps"] == 20
        assert error_context.user_parameters["num_frames"] == 16
        assert "fixed_invalid_resolution" in result.system_changes["adjustments_made"]
    
    @pytest.mark.asyncio
    async def test_resource_optimization_strategy(self, enhanced_recovery, mock_base_recovery):
        """Test resource optimization strategy"""
        error_context = ErrorContext(
            failure_type=EnhancedFailureType.VRAM_EXHAUSTION,
            original_error=Exception("Out of memory"),
            severity=ErrorSeverity.HIGH
        )
        
        result = await enhanced_recovery._resource_optimization_strategy(error_context)
        
        assert result.success is True
        assert result.strategy_used == RecoveryStrategy.RESOURCE_OPTIMIZATION
        assert len(result.system_changes["optimizations_applied"]) > 0
        mock_base_recovery._clear_gpu_cache.assert_called_once()
        mock_base_recovery._apply_vram_optimization.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_strategy(self, enhanced_recovery, mock_base_recovery):
        """Test graceful degradation strategy"""
        error_context = ErrorContext(
            failure_type=EnhancedFailureType.NETWORK_CONNECTIVITY_LOSS,
            original_error=Exception("Network unreachable"),
            severity=ErrorSeverity.MEDIUM
        )
        
        result = await enhanced_recovery._graceful_degradation_strategy(error_context)
        
        assert result.success is True
        assert result.strategy_used == RecoveryStrategy.GRACEFUL_DEGRADATION
        assert result.system_changes["degradation_mode"] is True
        mock_base_recovery._fallback_to_mock_generation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_user_intervention_strategy(self, enhanced_recovery):
        """Test user intervention strategy"""
        error_context = ErrorContext(
            failure_type=EnhancedFailureType.PERMISSION_DENIED,
            original_error=Exception("Access denied"),
            severity=ErrorSeverity.CRITICAL
        )
        
        result = await enhanced_recovery._user_intervention_strategy(error_context)
        
        assert result.success is False  # User intervention means automatic recovery failed
        assert result.strategy_used == RecoveryStrategy.USER_INTERVENTION
        assert result.requires_user_action is True
        assert len(result.actionable_steps) > 0
        assert result.system_changes["user_intervention_required"] is True    

    @pytest.mark.asyncio
    async def test_handle_enhanced_failure_success(self, enhanced_recovery, mock_enhanced_downloader):
        """Test successful enhanced failure handling"""
        # Mock successful download retry
        mock_result = Mock()
        mock_result.success = True
        mock_enhanced_downloader.download_with_retry.return_value = mock_result
        
        error_context = ErrorContext(
            failure_type=EnhancedFailureType.MODEL_DOWNLOAD_FAILURE,
            original_error=Exception("Download failed"),
            severity=ErrorSeverity.MEDIUM,
            model_id="test-model",
            correlation_id="test-123"
        )
        
        result = await enhanced_recovery.handle_enhanced_failure(error_context)
        
        assert result.success is True
        assert result.recovery_time_seconds > 0
        assert enhanced_recovery.recovery_metrics.successful_recoveries == 1
        assert enhanced_recovery.recovery_metrics.total_attempts == 1
    
    @pytest.mark.asyncio
    async def test_handle_enhanced_failure_all_strategies_fail(self, enhanced_recovery, mock_enhanced_downloader):
        """Test enhanced failure handling when all strategies fail"""
        # Mock all strategies failing
        mock_result = Mock()
        mock_result.success = False
        mock_enhanced_downloader.download_with_retry.return_value = mock_result
        
        error_context = ErrorContext(
            failure_type=EnhancedFailureType.MODEL_DOWNLOAD_FAILURE,
            original_error=Exception("Download failed"),
            severity=ErrorSeverity.MEDIUM,
            model_id="test-model"
        )
        
        result = await enhanced_recovery.handle_enhanced_failure(error_context)
        
        assert result.success is False
        assert result.requires_user_action is True
        assert enhanced_recovery.recovery_metrics.failed_recoveries == 1
        assert enhanced_recovery.recovery_metrics.total_attempts == 1
    
    @pytest.mark.asyncio
    async def test_handle_enhanced_failure_user_intervention_threshold(self, enhanced_recovery):
        """Test user intervention threshold"""
        error_context = ErrorContext(
            failure_type=EnhancedFailureType.MODEL_DOWNLOAD_FAILURE,
            original_error=Exception("Download failed"),
            severity=ErrorSeverity.MEDIUM,
            model_id="test-model",
            previous_attempts=["retry1", "retry2", "retry3", "retry4"]  # Exceeds threshold
        )
        
        result = await enhanced_recovery.handle_enhanced_failure(error_context)
        
        assert result.success is False
        assert result.requires_user_action is True
        assert result.strategy_used == RecoveryStrategy.USER_INTERVENTION
        assert "maximum recovery attempts exceeded" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_recovery_metrics_tracking(self, enhanced_recovery, mock_enhanced_downloader):
        """Test recovery metrics tracking"""
        # Mock successful recovery
        mock_result = Mock()
        mock_result.success = True
        mock_enhanced_downloader.download_with_retry.return_value = mock_result
        
        error_context = ErrorContext(
            failure_type=EnhancedFailureType.MODEL_DOWNLOAD_FAILURE,
            original_error=Exception("Download failed"),
            severity=ErrorSeverity.MEDIUM,
            model_id="test-model"
        )
        
        # Handle multiple failures to test metrics
        await enhanced_recovery.handle_enhanced_failure(error_context)
        await enhanced_recovery.handle_enhanced_failure(error_context)
        
        metrics = await enhanced_recovery.get_recovery_metrics()
        
        assert metrics.total_attempts == 2
        assert metrics.successful_recoveries == 2
        assert metrics.failed_recoveries == 0
        assert EnhancedFailureType.MODEL_DOWNLOAD_FAILURE in metrics.failure_type_frequencies
        assert metrics.failure_type_frequencies[EnhancedFailureType.MODEL_DOWNLOAD_FAILURE] == 2
        assert RecoveryStrategy.IMMEDIATE_RETRY in metrics.strategy_success_rates
    
    @pytest.mark.asyncio
    async def test_recovery_history_tracking(self, enhanced_recovery, mock_enhanced_downloader):
        """Test recovery history tracking"""
        # Mock successful recovery
        mock_result = Mock()
        mock_result.success = True
        mock_enhanced_downloader.download_with_retry.return_value = mock_result
        
        error_context = ErrorContext(
            failure_type=EnhancedFailureType.MODEL_DOWNLOAD_FAILURE,
            original_error=Exception("Download failed"),
            severity=ErrorSeverity.MEDIUM,
            model_id="test-model"
        )
        
        await enhanced_recovery.handle_enhanced_failure(error_context)
        
        history = await enhanced_recovery.get_recovery_history(limit=10)
        
        assert len(history) > 0
        # Note: The history comes from the base recovery system's recovery_attempts
        # In a real scenario, this would be populated by the recovery process
    
    @pytest.mark.asyncio
    async def test_websocket_notifications(self, enhanced_recovery, mock_websocket_manager, mock_enhanced_downloader):
        """Test WebSocket notifications during recovery"""
        # Mock successful recovery
        mock_result = Mock()
        mock_result.success = True
        mock_enhanced_downloader.download_with_retry.return_value = mock_result
        
        error_context = ErrorContext(
            failure_type=EnhancedFailureType.MODEL_DOWNLOAD_FAILURE,
            original_error=Exception("Download failed"),
            severity=ErrorSeverity.MEDIUM,
            model_id="test-model",
            correlation_id="test-123"
        )
        
        await enhanced_recovery.handle_enhanced_failure(error_context)
        
        # Verify WebSocket notifications were sent
        assert mock_websocket_manager.broadcast.call_count >= 1
        
        # Check the notification content
        calls = mock_websocket_manager.broadcast.call_args_list
        success_notification = None
        
        for call in calls:
            notification = call[0][0]
            if notification.get("type") == "recovery_success":
                success_notification = notification
                break
        
        assert success_notification is not None
        assert success_notification["correlation_id"] == "test-123"
        assert success_notification["data"]["success"] is True
    
    def test_convert_to_base_failure_type(self, enhanced_recovery):
        """Test conversion from enhanced to base failure types"""
        test_cases = [
            (EnhancedFailureType.MODEL_DOWNLOAD_FAILURE, FailureType.MODEL_LOADING_FAILURE),
            (EnhancedFailureType.VRAM_EXHAUSTION, FailureType.VRAM_EXHAUSTION),
            (EnhancedFailureType.NETWORK_CONNECTIVITY_LOSS, FailureType.NETWORK_ERROR),
            (EnhancedFailureType.PERMISSION_DENIED, FailureType.SYSTEM_RESOURCE_ERROR)
        ]
        
        for enhanced_type, expected_base_type in test_cases:
            result = enhanced_recovery._convert_to_base_failure_type(enhanced_type)
            assert result == expected_base_type
    
    def test_update_strategy_success_rate(self, enhanced_recovery):
        """Test strategy success rate tracking"""
        strategy = RecoveryStrategy.IMMEDIATE_RETRY
        
        # Test initial success
        enhanced_recovery._update_strategy_success_rate(strategy, True)
        assert enhanced_recovery.recovery_metrics.strategy_success_rates[strategy] == 1.0
        
        # Test failure
        enhanced_recovery._update_strategy_success_rate(strategy, False)
        success_rate = enhanced_recovery.recovery_metrics.strategy_success_rates[strategy]
        assert 0.0 < success_rate < 1.0  # Should be between 0 and 1
    
    def test_reset_metrics(self, enhanced_recovery):
        """Test metrics reset functionality"""
        # Add some data
        enhanced_recovery.recovery_metrics.total_attempts = 5
        enhanced_recovery.recovery_metrics.successful_recoveries = 3
        enhanced_recovery.recovery_attempts.append(Mock())
        
        # Reset
        enhanced_recovery.reset_metrics()
        
        # Verify reset
        assert enhanced_recovery.recovery_metrics.total_attempts == 0
        assert enhanced_recovery.recovery_metrics.successful_recoveries == 0
        assert len(enhanced_recovery.recovery_attempts) == 0


class TestErrorContextCreation:
    """Test error context creation and categorization"""
    
    @pytest.fixture
    def enhanced_recovery(self):
        """Create minimal enhanced recovery for testing"""
        return EnhancedErrorRecovery()
    
    @pytest.mark.asyncio
    async def test_error_categorization_comprehensive(self, enhanced_recovery):
        """Test comprehensive error categorization"""
        test_cases = [
            # Model-related errors
            ("Failed to download model", {}, EnhancedFailureType.MODEL_DOWNLOAD_FAILURE),
            ("Model file corruption detected", {}, EnhancedFailureType.MODEL_CORRUPTION_DETECTED),
            ("Version mismatch in model", {}, EnhancedFailureType.MODEL_VERSION_MISMATCH),
            ("Model loading timeout", {}, EnhancedFailureType.MODEL_LOADING_TIMEOUT),
            ("Model compatibility error", {}, EnhancedFailureType.MODEL_COMPATIBILITY_ERROR),
            
            # Resource-related errors
            ("CUDA out of memory", {}, EnhancedFailureType.VRAM_EXHAUSTION),
            ("No space left on device", {}, EnhancedFailureType.STORAGE_SPACE_INSUFFICIENT),
            ("Network connection lost", {}, EnhancedFailureType.NETWORK_CONNECTIVITY_LOSS),
            ("Bandwidth limit exceeded", {}, EnhancedFailureType.BANDWIDTH_LIMITATION),
            
            # System-related errors
            ("Permission denied", {}, EnhancedFailureType.PERMISSION_DENIED),
            ("Invalid parameter value", {}, EnhancedFailureType.INVALID_PARAMETERS),
            ("Operation not supported", {}, EnhancedFailureType.UNSUPPORTED_OPERATION),
            ("Missing dependency module", {}, EnhancedFailureType.DEPENDENCY_MISSING)
        ]
        
        for error_message, context, expected_type in test_cases:
            error = Exception(error_message)
            error_context = await enhanced_recovery.categorize_error(error, context)
            assert error_context.failure_type == expected_type, f"Failed for: {error_message}"
    
    @pytest.mark.asyncio
    async def test_severity_determination(self, enhanced_recovery):
        """Test error severity determination"""
        test_cases = [
            (EnhancedFailureType.DEPENDENCY_MISSING, ErrorSeverity.CRITICAL),
            (EnhancedFailureType.PERMISSION_DENIED, ErrorSeverity.CRITICAL),
            (EnhancedFailureType.MODEL_CORRUPTION_DETECTED, ErrorSeverity.HIGH),
            (EnhancedFailureType.VRAM_EXHAUSTION, ErrorSeverity.HIGH),
            (EnhancedFailureType.MODEL_DOWNLOAD_FAILURE, ErrorSeverity.MEDIUM),
            (EnhancedFailureType.NETWORK_CONNECTIVITY_LOSS, ErrorSeverity.MEDIUM)
        ]
        
        for failure_type, expected_severity in test_cases:
            severity = enhanced_recovery._determine_error_severity(
                failure_type, Exception("test"), {}
            )
            assert severity == expected_severity


class TestConvenienceFunction:
    """Test the convenience function for creating enhanced error recovery"""
    
    def test_create_enhanced_error_recovery_minimal(self):
        """Test creating enhanced error recovery with minimal parameters"""
        recovery = create_enhanced_error_recovery()
        
        assert recovery is not None
        assert isinstance(recovery, EnhancedErrorRecovery)
    
    def test_create_enhanced_error_recovery_with_generation_service(self):
        """Test creating enhanced error recovery with generation service"""
        mock_generation_service = Mock()
        mock_websocket_manager = Mock()
        
        recovery = create_enhanced_error_recovery(
            generation_service=mock_generation_service,
            websocket_manager=mock_websocket_manager
        )
        
        assert recovery is not None
        assert recovery.websocket_manager == mock_websocket_manager
    
    def test_create_enhanced_error_recovery_with_all_components(self):
        """Test creating enhanced error recovery with all components"""
        mock_generation_service = Mock()
        mock_websocket_manager = Mock()
        mock_availability_manager = Mock()
        mock_fallback_manager = Mock()
        mock_health_monitor = Mock()
        mock_downloader = Mock()
        
        recovery = create_enhanced_error_recovery(
            generation_service=mock_generation_service,
            websocket_manager=mock_websocket_manager,
            model_availability_manager=mock_availability_manager,
            intelligent_fallback_manager=mock_fallback_manager,
            model_health_monitor=mock_health_monitor,
            enhanced_downloader=mock_downloader
        )
        
        assert recovery is not None
        assert recovery.availability_manager == mock_availability_manager
        assert recovery.fallback_manager == mock_fallback_manager
        assert recovery.health_monitor == mock_health_monitor
        assert recovery.enhanced_downloader == mock_downloader


class TestIntegrationScenarios:
    """Test integration scenarios with various failure types"""
    
    @pytest.fixture
    def full_recovery_system(self):
        """Create a full recovery system with all mocked components"""
        mock_base_recovery = AsyncMock(spec=FallbackRecoverySystem)
        mock_base_recovery._clear_gpu_cache = AsyncMock(return_value=True)
        mock_base_recovery._apply_vram_optimization = AsyncMock(return_value=True)
        mock_base_recovery._enable_cpu_offload = AsyncMock(return_value=True)
        mock_base_recovery._restart_generation_pipeline = AsyncMock(return_value=True)
        mock_base_recovery._fallback_to_mock_generation = AsyncMock(return_value=True)
        
        return EnhancedErrorRecovery(base_recovery_system=mock_base_recovery)
    
    @pytest.mark.asyncio
    async def test_model_download_failure_scenario(self, full_recovery_system):
        """Test complete model download failure scenario"""
        error_context = ErrorContext(
            failure_type=EnhancedFailureType.MODEL_DOWNLOAD_FAILURE,
            original_error=Exception("Download failed due to network timeout"),
            severity=ErrorSeverity.MEDIUM,
            model_id="test-model-v1",
            operation="download",
            user_parameters={"quality": "high", "resolution": "1920x1080"}
        )
        
        result = await full_recovery_system.handle_enhanced_failure(error_context)
        
        # Should attempt recovery strategies
        assert result is not None
        assert isinstance(result, RecoveryResult)
    
    @pytest.mark.asyncio
    async def test_vram_exhaustion_scenario(self, full_recovery_system):
        """Test complete VRAM exhaustion scenario"""
        error_context = ErrorContext(
            failure_type=EnhancedFailureType.VRAM_EXHAUSTION,
            original_error=Exception("CUDA out of memory: tried to allocate 2.5GB"),
            severity=ErrorSeverity.HIGH,
            model_id="large-model",
            operation="generation",
            user_parameters={
                "resolution": "1920x1080",
                "steps": 50,
                "num_frames": 32
            }
        )
        
        result = await full_recovery_system.handle_enhanced_failure(error_context)
        
        # Should attempt resource optimization and parameter adjustment
        assert result is not None
        assert isinstance(result, RecoveryResult)
    
    @pytest.mark.asyncio
    async def test_model_corruption_scenario(self, full_recovery_system):
        """Test complete model corruption scenario"""
        error_context = ErrorContext(
            failure_type=EnhancedFailureType.MODEL_CORRUPTION_DETECTED,
            original_error=Exception("Checksum verification failed - file corrupted"),
            severity=ErrorSeverity.HIGH,
            model_id="corrupted-model",
            operation="load"
        )
        
        result = await full_recovery_system.handle_enhanced_failure(error_context)
        
        # Should attempt automatic repair
        assert result is not None
        assert isinstance(result, RecoveryResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])