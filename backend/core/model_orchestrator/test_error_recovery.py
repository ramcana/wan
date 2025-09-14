"""
Tests for the error recovery and retry logic system.
"""

import time
import pytest
from unittest.mock import Mock, patch
from typing import List

from .error_recovery import (
    ErrorRecoveryManager,
    RetryConfig,
    RetryStrategy,
    FailureCategory,
    RecoveryContext,
    retry_operation,
    with_retry
)
from .exceptions import (
    ModelOrchestratorError,
    ErrorCode,
    NoSpaceError,
    ChecksumError,
    LockTimeoutError
)
from .logging_config import configure_logging


@pytest.fixture
def recovery_manager():
    """Create an error recovery manager for testing."""
    configure_logging(level="DEBUG", structured=True)
    return ErrorRecoveryManager()


@pytest.fixture
def custom_config():
    """Create a custom retry configuration for testing."""
    return RetryConfig(
        max_attempts=3,
        base_delay=0.1,
        max_delay=1.0,
        backoff_factor=2.0,
        jitter=False,  # Disable jitter for predictable testing
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF
    )


class TestRetryConfiguration:
    """Test retry configuration and strategy selection."""
    
    def test_default_configuration(self, recovery_manager):
        """Test default retry configuration."""
        config = recovery_manager.config
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.backoff_factor == 2.0
        assert config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
    
    def test_custom_configuration(self):
        """Test custom retry configuration."""
        custom_config = RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=10.0,
            backoff_factor=1.5,
            strategy=RetryStrategy.LINEAR_BACKOFF
        )
        
        manager = ErrorRecoveryManager(custom_config)
        assert manager.config.max_attempts == 5
        assert manager.config.base_delay == 0.5
        assert manager.config.strategy == RetryStrategy.LINEAR_BACKOFF
    
    def test_error_specific_overrides(self):
        """Test error-specific retry configuration overrides."""
        base_config = RetryConfig(max_attempts=3)
        base_config.error_overrides[ErrorCode.RATE_LIMIT] = RetryConfig(
            max_attempts=10,
            base_delay=5.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        
        manager = ErrorRecoveryManager(base_config)
        
        # Test regular error uses category-based config (transient errors have max_attempts=5)
        regular_error = ModelOrchestratorError("test", ErrorCode.NETWORK_TIMEOUT)
        config = manager.get_retry_config(regular_error)
        assert config.max_attempts == 5  # From transient category strategy
        
        # Test rate limit error uses override
        rate_limit_error = ModelOrchestratorError("test", ErrorCode.RATE_LIMIT)
        config = manager.get_retry_config(rate_limit_error)
        assert config.max_attempts == 10
        assert config.base_delay == 5.0


class TestFailureCategories:
    """Test failure category classification."""
    
    def test_transient_failures(self, recovery_manager):
        """Test classification of transient failures."""
        errors = [
            ModelOrchestratorError("timeout", ErrorCode.NETWORK_TIMEOUT),
            ModelOrchestratorError("unavailable", ErrorCode.SOURCE_UNAVAILABLE),
            ConnectionError("connection failed"),
            TimeoutError("operation timed out")
        ]
        
        for error in errors:
            category = recovery_manager.get_failure_category(error)
            assert category == FailureCategory.TRANSIENT
    
    def test_permanent_failures(self, recovery_manager):
        """Test classification of permanent failures."""
        errors = [
            ModelOrchestratorError("not found", ErrorCode.MODEL_NOT_FOUND),
            ModelOrchestratorError("invalid", ErrorCode.INVALID_MODEL_ID),
            ModelOrchestratorError("config", ErrorCode.INVALID_CONFIG)
        ]
        
        for error in errors:
            category = recovery_manager.get_failure_category(error)
            assert category == FailureCategory.PERMANENT
    
    def test_resource_failures(self, recovery_manager):
        """Test classification of resource failures."""
        errors = [
            NoSpaceError(1000, 500, "/tmp"),
            ModelOrchestratorError("permission", ErrorCode.PERMISSION_DENIED),
            PermissionError("access denied")
        ]
        
        for error in errors:
            category = recovery_manager.get_failure_category(error)
            assert category == FailureCategory.RESOURCE
    
    def test_integrity_failures(self, recovery_manager):
        """Test classification of integrity failures."""
        errors = [
            ChecksumError("/path/file", "expected", "actual"),
            ModelOrchestratorError("size", ErrorCode.SIZE_MISMATCH),
            ModelOrchestratorError("incomplete", ErrorCode.INCOMPLETE_DOWNLOAD)
        ]
        
        for error in errors:
            category = recovery_manager.get_failure_category(error)
            assert category == FailureCategory.INTEGRITY


class TestDelayCalculation:
    """Test retry delay calculation strategies."""
    
    def test_exponential_backoff(self, recovery_manager):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            base_delay=1.0,
            backoff_factor=2.0,
            max_delay=10.0,
            jitter=False,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        
        # Test delay progression
        assert recovery_manager.calculate_delay(1, config) == 1.0
        assert recovery_manager.calculate_delay(2, config) == 2.0
        assert recovery_manager.calculate_delay(3, config) == 4.0
        assert recovery_manager.calculate_delay(4, config) == 8.0
        
        # Test max delay cap
        assert recovery_manager.calculate_delay(5, config) == 10.0
    
    def test_linear_backoff(self, recovery_manager):
        """Test linear backoff delay calculation."""
        config = RetryConfig(
            base_delay=2.0,
            max_delay=10.0,
            jitter=False,
            strategy=RetryStrategy.LINEAR_BACKOFF
        )
        
        assert recovery_manager.calculate_delay(1, config) == 2.0
        assert recovery_manager.calculate_delay(2, config) == 4.0
        assert recovery_manager.calculate_delay(3, config) == 6.0
        assert recovery_manager.calculate_delay(4, config) == 8.0
        assert recovery_manager.calculate_delay(5, config) == 10.0  # Capped
    
    def test_fixed_delay(self, recovery_manager):
        """Test fixed delay strategy."""
        config = RetryConfig(
            base_delay=3.0,
            jitter=False,
            strategy=RetryStrategy.FIXED_DELAY
        )
        
        for attempt in range(1, 6):
            assert recovery_manager.calculate_delay(attempt, config) == 3.0
    
    def test_no_retry(self, recovery_manager):
        """Test no retry strategy."""
        config = RetryConfig(strategy=RetryStrategy.NO_RETRY)
        
        for attempt in range(1, 6):
            assert recovery_manager.calculate_delay(attempt, config) == 0.0
    
    def test_jitter_application(self, recovery_manager):
        """Test that jitter is applied when enabled."""
        config = RetryConfig(
            base_delay=1.0,
            jitter=True,
            strategy=RetryStrategy.FIXED_DELAY
        )
        
        # With jitter, delays should vary slightly
        delays = [recovery_manager.calculate_delay(1, config) for _ in range(10)]
        
        # All delays should be close to base_delay but not identical
        assert all(0.9 <= delay <= 1.1 for delay in delays)
        assert len(set(delays)) > 1  # Should have some variation


class TestRetryLogic:
    """Test the core retry logic."""
    
    def test_successful_operation_no_retry(self, recovery_manager):
        """Test that successful operations don't retry."""
        call_count = 0
        
        def successful_operation():
            nonlocal call_count
            call_count += 1
            return "success"
        
        with recovery_manager.recovery_context("test_operation") as context:
            result = recovery_manager.retry_with_recovery(
                successful_operation, context
            )
        
        assert result == "success"
        assert call_count == 1
        assert context.attempt == 1
    
    def test_transient_failure_retry(self, recovery_manager, custom_config):
        """Test retry behavior for transient failures."""
        recovery_manager.config = custom_config
        call_count = 0
        
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            
            if call_count < 3:
                raise ModelOrchestratorError("transient", ErrorCode.NETWORK_TIMEOUT)
            
            return "success_after_retry"
        
        with recovery_manager.recovery_context("test_operation") as context:
            result = recovery_manager.retry_with_recovery(
                flaky_operation, context
            )
        
        assert result == "success_after_retry"
        assert call_count == 3
        assert context.attempt == 3
    
    def test_permanent_failure_no_retry(self, recovery_manager):
        """Test that permanent failures don't retry."""
        call_count = 0
        
        def permanent_failure():
            nonlocal call_count
            call_count += 1
            raise ModelOrchestratorError("permanent", ErrorCode.MODEL_NOT_FOUND)
        
        with recovery_manager.recovery_context("test_operation") as context:
            with pytest.raises(ModelOrchestratorError) as exc_info:
                recovery_manager.retry_with_recovery(
                    permanent_failure, context
                )
        
        assert exc_info.value.error_code == ErrorCode.MODEL_NOT_FOUND
        assert call_count == 1
        assert context.attempt == 1
    
    def test_max_attempts_exceeded(self, custom_config):
        """Test behavior when max attempts are exceeded."""
        # Create manager with custom config to ensure it's used
        recovery_manager = ErrorRecoveryManager(custom_config)
        call_count = 0
        
        def always_failing():
            nonlocal call_count
            call_count += 1
            raise ModelOrchestratorError("always fails", ErrorCode.NETWORK_TIMEOUT)
        
        with recovery_manager.recovery_context("test_operation") as context:
            with pytest.raises(ModelOrchestratorError):
                recovery_manager.retry_with_recovery(
                    always_failing, context
                )
        
        # Network timeout is a transient error, so it uses category strategy (max_attempts=5)
        assert call_count == 5
        assert context.attempt == 5
    
    def test_delay_between_retries(self):
        """Test that delays are applied between retries."""
        # Use a permanent error to avoid category-based overrides
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            jitter=False,
            strategy=RetryStrategy.FIXED_DELAY
        )
        # Override permanent error to allow retries for testing
        config.error_overrides[ErrorCode.INVALID_MODEL_ID] = config
        
        recovery_manager = ErrorRecoveryManager(config)
        
        call_times = []
        
        def timed_failure():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ModelOrchestratorError("retry", ErrorCode.INVALID_MODEL_ID)
            return "success"
        
        with recovery_manager.recovery_context("test_operation") as context:
            result = recovery_manager.retry_with_recovery(
                timed_failure, context
            )
        
        assert result == "success"
        assert len(call_times) == 3
        
        # Verify delays (with some tolerance for execution time)
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        
        assert 0.08 <= delay1 <= 0.15  # ~0.1s with tolerance
        assert 0.08 <= delay2 <= 0.15


class TestRecoveryStrategies:
    """Test specific recovery strategies for different error types."""
    
    def test_auth_failure_recovery(self, recovery_manager):
        """Test recovery strategy for authentication failures."""
        with patch.object(recovery_manager, '_handle_auth_failure') as mock_handler:
            error = ModelOrchestratorError("auth failed", ErrorCode.AUTH_FAIL)
            context = RecoveryContext("test", "auth_test")
            
            recovery_manager._apply_recovery_strategy(error, context)
            
            mock_handler.assert_called_once_with(error, context)
    
    def test_rate_limit_recovery(self, recovery_manager):
        """Test recovery strategy for rate limiting."""
        with patch.object(recovery_manager, '_handle_rate_limit') as mock_handler:
            error = ModelOrchestratorError("rate limited", ErrorCode.RATE_LIMIT)
            context = RecoveryContext("test", "rate_limit_test")
            
            recovery_manager._apply_recovery_strategy(error, context)
            
            mock_handler.assert_called_once_with(error, context)
    
    def test_resource_failure_recovery(self, recovery_manager):
        """Test recovery strategy for resource failures."""
        with patch.object(recovery_manager, '_handle_resource_failure') as mock_handler:
            error = NoSpaceError(1000, 500, "/tmp")
            context = RecoveryContext("test", "resource_test")
            
            recovery_manager._apply_recovery_strategy(error, context)
            
            mock_handler.assert_called_once_with(error, context)
    
    def test_integrity_failure_recovery(self, recovery_manager):
        """Test recovery strategy for integrity failures."""
        with patch.object(recovery_manager, '_handle_integrity_failure') as mock_handler:
            error = ChecksumError("/path", "expected", "actual")
            context = RecoveryContext("test", "integrity_test")
            
            recovery_manager._apply_recovery_strategy(error, context)
            
            mock_handler.assert_called_once_with(error, context)
    
    def test_concurrency_failure_recovery(self, recovery_manager):
        """Test recovery strategy for concurrency failures."""
        with patch.object(recovery_manager, '_handle_concurrency_failure') as mock_handler:
            error = LockTimeoutError("lock timeout", "test-model", 5.0)
            context = RecoveryContext("test", "concurrency_test")
            
            recovery_manager._apply_recovery_strategy(error, context)
            
            mock_handler.assert_called_once_with(error, context)


class TestConvenienceFunctions:
    """Test convenience functions and decorators."""
    
    def test_retry_operation_function(self, custom_config):
        """Test the retry_operation convenience function."""
        call_count = 0
        
        def test_operation(value: str) -> str:
            nonlocal call_count
            call_count += 1
            
            if call_count < 2:
                raise ModelOrchestratorError("retry", ErrorCode.NETWORK_TIMEOUT)
            
            return f"processed_{value}"
        
        result = retry_operation(
            test_operation,
            operation="test_retry_operation",
            config=custom_config,
            value="test_input"
        )
        
        assert result == "processed_test_input"
        assert call_count == 2
    
    def test_with_retry_decorator(self, custom_config):
        """Test the with_retry decorator."""
        call_count = 0
        
        @with_retry("decorated_operation", config=custom_config)
        def decorated_function(multiplier: int) -> int:
            nonlocal call_count
            call_count += 1
            
            if call_count < 3:
                raise ModelOrchestratorError("retry", ErrorCode.NETWORK_TIMEOUT)
            
            return call_count * multiplier
        
        result = decorated_function(5)
        
        assert result == 15  # 3 * 5
        assert call_count == 3
    
    def test_decorator_with_context_kwargs(self):
        """Test decorator with additional context kwargs."""
        @with_retry("test_operation", model_id="test-model", variant="fp16")
        def context_function() -> str:
            return "success"
        
        # This should not raise an exception
        result = context_function()
        assert result == "success"


class TestRecoveryContext:
    """Test recovery context management."""
    
    def test_context_creation(self, recovery_manager):
        """Test recovery context creation and management."""
        with recovery_manager.recovery_context(
            "test_operation",
            model_id="test-model",
            variant="fp16",
            source_url="https://example.com"
        ) as context:
            assert context.operation == "test_operation"
            assert context.model_id == "test-model"
            assert context.variant == "fp16"
            assert context.source_url == "https://example.com"
            assert context.attempt == 1
            assert context.correlation_id is not None
            assert isinstance(context.start_time, float)
    
    def test_context_attempt_tracking(self):
        """Test that context tracks attempt numbers correctly."""
        def failing_operation():
            raise ModelOrchestratorError("test", ErrorCode.NETWORK_TIMEOUT)
        
        config = RetryConfig(max_attempts=3, base_delay=0.01, jitter=False)
        recovery_manager = ErrorRecoveryManager(config)
        
        with recovery_manager.recovery_context("test") as context:
            with pytest.raises(ModelOrchestratorError):
                recovery_manager.retry_with_recovery(failing_operation, context)
            
            # Network timeout is transient, so it uses category strategy (max_attempts=5)
            assert context.attempt == 5


class TestErrorRecoveryIntegration:
    """Test integration of error recovery with other components."""
    
    def test_recovery_with_logging(self, recovery_manager, caplog):
        """Test that recovery operations are properly logged."""
        call_count = 0
        
        def logged_operation():
            nonlocal call_count
            call_count += 1
            
            if call_count < 2:
                raise ModelOrchestratorError("logged error", ErrorCode.NETWORK_TIMEOUT)
            
            return "logged_success"
        
        with recovery_manager.recovery_context("logged_test") as context:
            result = recovery_manager.retry_with_recovery(logged_operation, context)
        
        assert result == "logged_success"
        
        # Check that appropriate log messages were generated
        log_messages = [record.message for record in caplog.records]
        assert any("Starting operation" in msg for msg in log_messages)
        assert any("Operation failed" in msg for msg in log_messages)
        assert any("Operation completed" in msg for msg in log_messages)
    
    def test_recovery_preserves_exception_details(self, recovery_manager):
        """Test that recovery preserves original exception details."""
        original_error = ChecksumError("/test/path", "expected_hash", "actual_hash")
        
        def failing_operation():
            raise original_error
        
        with recovery_manager.recovery_context("test") as context:
            with pytest.raises(ChecksumError) as exc_info:
                recovery_manager.retry_with_recovery(failing_operation, context)
        
        # Verify the original exception details are preserved
        caught_error = exc_info.value
        assert caught_error.error_code == ErrorCode.CHECKSUM_FAIL
        assert caught_error.details["file_path"] == "/test/path"
        assert caught_error.details["expected"] == "expected_hash"
        assert caught_error.details["actual"] == "actual_hash"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])