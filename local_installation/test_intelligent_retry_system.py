"""
Comprehensive test suite for the Intelligent Retry System.

This test suite covers:
- Configurable retry counts and user control
- Exponential backoff with jitter for network operations
- Retry strategy selection based on error type and context
- User prompt functionality for retry configuration during failures
- Integration with the existing error handler
"""

import pytest
import time
import logging
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Import the modules to test
from scripts.intelligent_retry_system import (
    IntelligentRetrySystem, RetryConfiguration, RetryStrategy, RetryDecision,
    UserInteractionHandler, RetryStrategySelector, RetryAttempt, RetrySession
)
from scripts.error_handler import ComprehensiveErrorHandler
from interfaces import ErrorCategory, InstallationError


class TestRetryConfiguration:
    """Test retry configuration functionality."""
    
    def test_default_configuration(self):
        """Test default retry configuration values."""
        config = RetryConfiguration()
        
        assert config.max_attempts == 3
        assert config.base_delay == 2.0
        assert config.max_delay == 60.0
        assert config.backoff_multiplier == 2.0
        assert config.jitter is True
        assert config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert config.user_prompt is True
    
    def test_custom_configuration(self):
        """Test custom retry configuration."""
        config = RetryConfiguration(
            max_attempts=5,
            base_delay=1.0,
            max_delay=30.0,
            backoff_multiplier=1.5,
            jitter=False,
            strategy=RetryStrategy.LINEAR_BACKOFF,
            user_prompt=False
        )
        
        assert config.max_attempts == 5
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
        assert config.backoff_multiplier == 1.5
        assert config.jitter is False
        assert config.strategy == RetryStrategy.LINEAR_BACKOFF
        assert config.user_prompt is False
    
    def test_error_specific_configurations(self):
        """Test error-specific configuration values."""
        config = RetryConfiguration()
        
        assert config.network_max_attempts == 5
        assert config.system_max_attempts == 3
        assert config.configuration_max_attempts == 2
        assert config.permission_max_attempts == 1


class TestUserInteractionHandler:
    """Test user interaction handling for retry decisions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = UserInteractionHandler()
        self.config = RetryConfiguration()
    
    def test_no_prompt_when_disabled(self):
        """Test that no prompt is shown when user_prompt is disabled."""
        config = RetryConfiguration(user_prompt=False)
        error = ValueError("Test error")
        
        decision = self.handler.prompt_user_for_retry_decision(
            error, 1, "test_operation", config
        )
        
        assert decision == RetryDecision.RETRY
    
    def test_saved_preference_used(self):
        """Test that saved user preferences are used."""
        self.handler.user_preferences["ValueError"] = "skip"
        error = ValueError("Test error")
        
        decision = self.handler.prompt_user_for_retry_decision(
            error, 1, "test_operation", self.config
        )
        
        assert decision == RetryDecision.SKIP
    
    @patch('builtins.input', side_effect=['1'])
    def test_user_chooses_retry(self, mock_input):
        """Test user choosing to retry."""
        error = ValueError("Test error")
        
        decision = self.handler.prompt_user_for_retry_decision(
            error, 1, "test_operation", self.config
        )
        
        assert decision == RetryDecision.RETRY
    
    @patch('builtins.input', side_effect=['3'])
    def test_user_chooses_skip(self, mock_input):
        """Test user choosing to skip."""
        error = ValueError("Test error")
        
        decision = self.handler.prompt_user_for_retry_decision(
            error, 1, "test_operation", self.config
        )
        
        assert decision == RetryDecision.SKIP
    
    @patch('builtins.input', side_effect=['4'])
    def test_user_chooses_abort(self, mock_input):
        """Test user choosing to abort."""
        error = ValueError("Test error")
        
        decision = self.handler.prompt_user_for_retry_decision(
            error, 1, "test_operation", self.config
        )
        
        assert decision == RetryDecision.ABORT
    
    @patch('builtins.input', side_effect=['2', '5', '1.5', '30', '2', 'y'])
    def test_user_configures_settings(self, mock_input):
        """Test user configuring retry settings."""
        error = ValueError("Test error")
        
        decision = self.handler.prompt_user_for_retry_decision(
            error, 1, "test_operation", self.config
        )
        
        assert decision == RetryDecision.CONFIGURE
        assert self.config.max_attempts == 5
        assert self.config.base_delay == 1.5
        assert self.config.max_delay == 30.0
        assert self.config.strategy == RetryStrategy.LINEAR_BACKOFF
        assert self.config.jitter is True
    
    @patch('builtins.input', side_effect=['5', '1'])
    def test_user_remembers_choice(self, mock_input):
        """Test user choosing to remember decision."""
        error = ValueError("Test error")
        
        decision = self.handler.prompt_user_for_retry_decision(
            error, 1, "test_operation", self.config
        )
        
        assert decision == RetryDecision.RETRY
        assert self.handler.user_preferences["ValueError"] == "retry"
    
    def test_delay_calculation_exponential(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfiguration(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=2.0,
            backoff_multiplier=2.0,
            jitter=False
        )
        
        delay = self.handler._calculate_next_delay(0, config)
        assert delay == 2.0
        
        delay = self.handler._calculate_next_delay(1, config)
        assert delay == 4.0
        
        delay = self.handler._calculate_next_delay(2, config)
        assert delay == 8.0
    
    def test_delay_calculation_linear(self):
        """Test linear backoff delay calculation."""
        config = RetryConfiguration(
            strategy=RetryStrategy.LINEAR_BACKOFF,
            base_delay=2.0,
            jitter=False
        )
        
        delay = self.handler._calculate_next_delay(0, config)
        assert delay == 2.0
        
        delay = self.handler._calculate_next_delay(1, config)
        assert delay == 4.0
        
        delay = self.handler._calculate_next_delay(2, config)
        assert delay == 6.0
    
    def test_delay_calculation_fixed(self):
        """Test fixed delay calculation."""
        config = RetryConfiguration(
            strategy=RetryStrategy.FIXED_DELAY,
            base_delay=2.0,
            jitter=False
        )
        
        delay = self.handler._calculate_next_delay(0, config)
        assert delay == 2.0
        
        delay = self.handler._calculate_next_delay(1, config)
        assert delay == 2.0
        
        delay = self.handler._calculate_next_delay(2, config)
        assert delay == 2.0
    
    def test_delay_calculation_fibonacci(self):
        """Test Fibonacci delay calculation."""
        config = RetryConfiguration(
            strategy=RetryStrategy.FIBONACCI,
            base_delay=1.0,
            jitter=False
        )
        
        delay = self.handler._calculate_next_delay(0, config)
        assert delay == 1.0  # 1 * fib(1) = 1 * 1
        
        delay = self.handler._calculate_next_delay(1, config)
        assert delay == 2.0  # 1 * fib(2) = 1 * 2
        
        delay = self.handler._calculate_next_delay(2, config)
        assert delay == 3.0  # 1 * fib(3) = 1 * 3
    
    def test_max_delay_limit(self):
        """Test that delays are limited by max_delay."""
        config = RetryConfiguration(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=10.0,
            backoff_multiplier=2.0,
            max_delay=15.0,
            jitter=False
        )
        
        delay = self.handler._calculate_next_delay(2, config)  # Would be 40.0 without limit
        assert delay == 15.0
    
    def test_jitter_application(self):
        """Test that jitter is applied when enabled."""
        config = RetryConfiguration(
            strategy=RetryStrategy.FIXED_DELAY,
            base_delay=10.0,
            jitter=True
        )
        
        # Run multiple times to check jitter variation
        delays = [self.handler._calculate_next_delay(0, config) for _ in range(10)]
        
        # All delays should be between 5.0 and 10.0 (50% to 100% of base)
        assert all(5.0 <= delay <= 10.0 for delay in delays)
        
        # Should have some variation (not all the same)
        assert len(set(delays)) > 1


class TestRetryStrategySelector:
    """Test retry strategy selection based on error type and context."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.selector = RetryStrategySelector()
    
    def test_network_error_strategy(self):
        """Test strategy selection for network errors."""
        error = ConnectionError("Network connection failed")
        config = self.selector.select_strategy(error, ErrorCategory.NETWORK)
        
        assert config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.jitter is True
    
    def test_system_error_strategy(self):
        """Test strategy selection for system errors."""
        error = OSError("System error occurred")
        config = self.selector.select_strategy(error, ErrorCategory.SYSTEM)
        
        assert config.strategy == RetryStrategy.LINEAR_BACKOFF
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.jitter is True
    
    def test_configuration_error_strategy(self):
        """Test strategy selection for configuration errors."""
        error = ValueError("Invalid configuration value")
        config = self.selector.select_strategy(error, ErrorCategory.CONFIGURATION)
        
        assert config.strategy == RetryStrategy.FIXED_DELAY
        assert config.max_attempts == 2
        assert config.base_delay == 0.5
        assert config.jitter is False
    
    def test_permission_error_strategy(self):
        """Test strategy selection for permission errors."""
        error = PermissionError("Access denied")
        config = self.selector.select_strategy(error, ErrorCategory.PERMISSION)
        
        assert config.strategy == RetryStrategy.IMMEDIATE
        assert config.max_attempts == 1
        assert config.base_delay == 0.0
        assert config.jitter is False
    
    def test_error_categorization_network(self):
        """Test automatic error categorization for network errors."""
        errors = [
            ConnectionError("Connection failed"),
            TimeoutError("Request timed out"),
            Exception("HTTP 500 error"),
            Exception("DNS resolution failed"),
            Exception("SSL certificate error")
        ]
        
        for error in errors:
            category = self.selector._categorize_error(error)
            assert category == ErrorCategory.NETWORK
    
    def test_error_categorization_permission(self):
        """Test automatic error categorization for permission errors."""
        errors = [
            PermissionError("Permission denied"),
            Exception("Access denied"),
            Exception("Unauthorized access"),
            Exception("Administrator privileges required")
        ]
        
        for error in errors:
            category = self.selector._categorize_error(error)
            assert category == ErrorCategory.PERMISSION
    
    def test_error_categorization_configuration(self):
        """Test automatic error categorization for configuration errors."""
        errors = [
            ValueError("Invalid configuration"),
            SyntaxError("JSON parse error"),
            KeyError("Missing configuration key"),
            Exception("Invalid parameter value")
        ]
        
        for error in errors:
            category = self.selector._categorize_error(error)
            assert category == ErrorCategory.CONFIGURATION
    
    def test_context_adjustments_file_size(self):
        """Test context-based adjustments for large files."""
        error = ConnectionError("Download failed")
        context = {'file_size_mb': 1500}  # Large file
        
        config = self.selector.select_strategy(error, ErrorCategory.NETWORK, context)
        
        # Should have increased delays for large files
        base_config = self.selector.strategy_mappings[ErrorCategory.NETWORK]
        assert config.max_delay > base_config['max_delay']
        assert config.base_delay > base_config['base_delay']
    
    def test_context_adjustments_network_speed(self):
        """Test context-based adjustments for slow networks."""
        error = ConnectionError("Download failed")
        context = {'network_speed': 'slow'}
        
        config = self.selector.select_strategy(error, ErrorCategory.NETWORK, context)
        
        # Should have more attempts and longer delays for slow networks
        base_config = self.selector.strategy_mappings[ErrorCategory.NETWORK]
        assert config.max_attempts > base_config['max_attempts']
        assert config.max_delay > base_config['max_delay']
    
    def test_context_adjustments_system_load(self):
        """Test context-based adjustments for high system load."""
        error = OSError("System busy")
        context = {'system_load': 'high'}
        
        config = self.selector.select_strategy(error, ErrorCategory.SYSTEM, context)
        
        # Should have longer delays for high system load
        base_config = self.selector.strategy_mappings[ErrorCategory.SYSTEM]
        assert config.base_delay > base_config['base_delay']
        assert config.max_delay > base_config['max_delay']
    
    def test_context_adjustments_critical_operation(self):
        """Test context-based adjustments for critical operations."""
        error = ConnectionError("Critical operation failed")
        context = {'critical': True}
        
        config = self.selector.select_strategy(error, ErrorCategory.NETWORK, context)
        
        # Should have more attempts for critical operations
        base_config = self.selector.strategy_mappings[ErrorCategory.NETWORK]
        assert config.max_attempts > base_config['max_attempts']


class TestIntelligentRetrySystem:
    """Test the main intelligent retry system functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.retry_system = IntelligentRetrySystem("/test/path")
        self.retry_system.logger = Mock()
    
    def test_successful_operation_no_retry(self):
        """Test successful operation that doesn't need retry."""
        def successful_operation():
            return "success"
        
        result = self.retry_system.execute_with_retry(
            successful_operation, "test_operation"
        )
        
        assert result == "success"
        assert len(self.retry_system.session_history) == 1
        
        session = self.retry_system.session_history[0]
        assert session.successful is True
        assert session.total_attempts == 1
        assert len(session.attempts) == 1
        assert session.attempts[0].success is True
    
    def test_operation_fails_then_succeeds(self):
        """Test operation that fails once then succeeds."""
        call_count = 0
        
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Network error")
            return "success"
        
        with patch.object(self.retry_system.user_interaction_handler, 
                         'prompt_user_for_retry_decision', 
                         return_value=RetryDecision.RETRY):
            result = self.retry_system.execute_with_retry(
                flaky_operation, "test_operation", ErrorCategory.NETWORK
            )
        
        assert result == "success"
        assert call_count == 2
        
        session = self.retry_system.session_history[0]
        assert session.successful is True
        assert session.total_attempts == 2
        assert len(session.attempts) == 2
        assert session.attempts[0].success is False
        assert session.attempts[1].success is True
    
    def test_operation_fails_all_attempts(self):
        """Test operation that fails all retry attempts."""
        def failing_operation():
            raise ConnectionError("Persistent network error")
        
        config = RetryConfiguration(max_attempts=3, user_prompt=False)
        
        with pytest.raises(ConnectionError):
            self.retry_system.execute_with_retry(
                failing_operation, "test_operation", 
                ErrorCategory.NETWORK, custom_config=config
            )
        
        session = self.retry_system.session_history[0]
        assert session.successful is False
        assert session.total_attempts == 3
        assert len(session.attempts) == 3
        assert all(not attempt.success for attempt in session.attempts)
        assert isinstance(session.final_error, ConnectionError)
    
    def test_user_aborts_retry(self):
        """Test user choosing to abort retry."""
        def failing_operation():
            raise ConnectionError("Network error")
        
        with patch.object(self.retry_system.user_interaction_handler, 
                         'prompt_user_for_retry_decision', 
                         return_value=RetryDecision.ABORT):
            with pytest.raises(ConnectionError):
                self.retry_system.execute_with_retry(
                    failing_operation, "test_operation", ErrorCategory.NETWORK
                )
        
        session = self.retry_system.session_history[0]
        assert session.successful is False
        assert session.total_attempts == 1
        assert "abort" in session.user_decisions
    
    def test_user_skips_operation(self):
        """Test user choosing to skip operation."""
        def failing_operation():
            raise ConnectionError("Network error")
        
        with patch.object(self.retry_system.user_interaction_handler, 
                         'prompt_user_for_retry_decision', 
                         return_value=RetryDecision.SKIP):
            result = self.retry_system.execute_with_retry(
                failing_operation, "test_operation", ErrorCategory.NETWORK
            )
        
        assert result is None
        
        session = self.retry_system.session_history[0]
        assert session.successful is False
        assert session.total_attempts == 1
        assert "skip" in session.user_decisions
    
    def test_non_retryable_error(self):
        """Test that non-retryable errors are not retried."""
        def failing_operation():
            raise KeyboardInterrupt("User interrupted")
        
        with pytest.raises(KeyboardInterrupt):
            self.retry_system.execute_with_retry(
                failing_operation, "test_operation"
            )
        
        session = self.retry_system.session_history[0]
        assert session.successful is False
        assert session.total_attempts == 1
    
    def test_delay_calculation_and_timing(self):
        """Test that delays are calculated and applied correctly."""
        call_times = []
        
        def failing_operation():
            call_times.append(time.time())
            raise ConnectionError("Network error")
        
        config = RetryConfiguration(
            max_attempts=3, 
            base_delay=0.1,  # Short delay for testing
            strategy=RetryStrategy.FIXED_DELAY,
            jitter=False,
            user_prompt=False
        )
        
        with pytest.raises(ConnectionError):
            self.retry_system.execute_with_retry(
                failing_operation, "test_operation", 
                ErrorCategory.NETWORK, custom_config=config
            )
        
        # Check that delays were applied between attempts
        assert len(call_times) == 3
        
        # Allow some tolerance for timing
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        
        assert 0.08 <= delay1 <= 0.15  # ~0.1s with tolerance
        assert 0.08 <= delay2 <= 0.15  # ~0.1s with tolerance
    
    def test_session_statistics(self):
        """Test session statistics calculation."""
        # Run some successful and failed operations
        def success():
            return "ok"
        
        def failure():
            raise ValueError("Error")
        
        config = RetryConfiguration(max_attempts=1, user_prompt=False)
        
        # Successful operation
        self.retry_system.execute_with_retry(success, "success_op")
        
        # Failed operation
        with pytest.raises(ValueError):
            self.retry_system.execute_with_retry(
                failure, "failure_op", custom_config=config
            )
        
        stats = self.retry_system.get_session_statistics()
        
        assert stats['total_sessions'] == 2
        assert stats['successful_sessions'] == 1
        assert stats['failed_sessions'] == 1
        assert stats['success_rate'] == 0.5
        assert stats['average_attempts'] == 1.0
        assert len(stats['most_common_errors']) == 1
        assert stats['most_common_errors'][0][0] == 'ValueError'
    
    def test_active_session_management(self):
        """Test active session tracking and cancellation."""
        def long_running_operation():
            time.sleep(0.1)
            return "done"
        
        # Start operation in background (simulate)
        import threading
        
        def run_operation():
            self.retry_system.execute_with_retry(
                long_running_operation, "long_op"
            )
        
        thread = threading.Thread(target=run_operation)
        thread.start()
        
        # Check active sessions
        time.sleep(0.05)  # Let operation start
        active = self.retry_system.get_active_sessions()
        
        # Wait for completion
        thread.join()
        
        # Should be no active sessions after completion
        active_after = self.retry_system.get_active_sessions()
        assert len(active_after) == 0
    
    def test_global_configuration(self):
        """Test setting global retry configuration."""
        new_config = RetryConfiguration(
            max_attempts=10,
            base_delay=5.0,
            strategy=RetryStrategy.LINEAR_BACKOFF
        )
        
        self.retry_system.set_global_configuration(new_config)
        
        assert self.retry_system.global_config.max_attempts == 10
        assert self.retry_system.global_config.base_delay == 5.0
        assert self.retry_system.global_config.strategy == RetryStrategy.LINEAR_BACKOFF


class TestErrorHandlerIntegration:
    """Test integration with the existing error handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ComprehensiveErrorHandler("/test/path")
        self.error_handler.logger = Mock()
    
    def test_intelligent_retry_integration(self):
        """Test that error handler integrates with intelligent retry system."""
        def successful_operation():
            return "success"
        
        result = self.error_handler.execute_with_intelligent_retry(
            successful_operation, "test_operation"
        )
        
        assert result == "success"
        
        # Check that retry system was used
        assert len(self.error_handler.intelligent_retry_system.session_history) == 1
    
    def test_retry_configuration_update(self):
        """Test updating retry configuration through error handler."""
        new_config = RetryConfiguration(
            max_attempts=7,
            base_delay=3.0,
            strategy=RetryStrategy.FIBONACCI
        )
        
        self.error_handler.configure_retry_behavior(ErrorCategory.NETWORK, new_config)
        
        # Check legacy config was updated
        legacy_config = self.error_handler.retry_configs[ErrorCategory.NETWORK]
        assert legacy_config.max_attempts == 7
        assert legacy_config.base_delay == 3.0
        
        # Check intelligent retry system was updated
        strategy_mapping = self.error_handler.intelligent_retry_system.strategy_selector.strategy_mappings[ErrorCategory.NETWORK]
        assert strategy_mapping['max_attempts'] == 7
        assert strategy_mapping['base_delay'] == 3.0
        assert strategy_mapping['strategy'] == RetryStrategy.FIBONACCI
    
    def test_retry_statistics_combination(self):
        """Test combined retry statistics from both systems."""
        # Run operation through intelligent retry system
        def success():
            return "ok"
        
        self.error_handler.execute_with_intelligent_retry(success, "test_op")
        
        stats = self.error_handler.get_retry_statistics()
        
        assert 'intelligent_retry_system' in stats
        assert 'error_handler_stats' in stats
        assert 'active_retry_sessions' in stats
        assert 'retry_configurations' in stats
        
        # Check intelligent retry stats
        intelligent_stats = stats['intelligent_retry_system']
        assert intelligent_stats['total_sessions'] == 1
        assert intelligent_stats['successful_sessions'] == 1
    
    def test_user_prompt_control(self):
        """Test controlling user prompts through error handler."""
        self.error_handler.set_user_prompt_enabled(False)
        
        assert self.error_handler.intelligent_retry_system.global_config.user_prompt is False
        
        self.error_handler.set_user_prompt_enabled(True)
        
        assert self.error_handler.intelligent_retry_system.global_config.user_prompt is True
    
    def test_active_session_management_through_handler(self):
        """Test managing active sessions through error handler."""
        # Mock an active session
        from scripts.intelligent_retry_system import RetrySession
        
        session = RetrySession("test_op", datetime.now())
        self.error_handler.intelligent_retry_system.active_sessions["test_op"] = session
        
        # Get active sessions
        active = self.error_handler.get_active_retry_sessions()
        assert "test_op" in active
        assert active["test_op"]["operation_name"] == "test_op"
        
        # Cancel session
        cancelled = self.error_handler.cancel_active_retry_session("test_op")
        assert cancelled is True
        
        # Should be no active sessions now
        active_after = self.error_handler.get_active_retry_sessions()
        assert len(active_after) == 0


class TestNetworkOperationsWithExponentialBackoff:
    """Test network operations with exponential backoff specifically."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.retry_system = IntelligentRetrySystem("/test/path")
        self.retry_system.logger = Mock()
    
    def test_network_download_with_exponential_backoff(self):
        """Test network download operation with exponential backoff."""
        attempt_count = 0
        attempt_times = []
        
        def download_operation():
            nonlocal attempt_count
            attempt_count += 1
            attempt_times.append(time.time())
            
            if attempt_count < 3:
                raise ConnectionError(f"Download failed (attempt {attempt_count})")
            return f"Downloaded {attempt_count} MB"
        
        config = RetryConfiguration(
            max_attempts=5,
            base_delay=0.1,  # Short for testing
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            backoff_multiplier=2.0,
            jitter=False,
            user_prompt=False
        )
        
        result = self.retry_system.execute_with_retry(
            download_operation, "network_download", 
            ErrorCategory.NETWORK, custom_config=config
        )
        
        assert result == "Downloaded 3 MB"
        assert attempt_count == 3
        
        # Check exponential backoff timing
        assert len(attempt_times) == 3
        
        # First retry delay should be ~0.1s
        delay1 = attempt_times[1] - attempt_times[0]
        assert 0.08 <= delay1 <= 0.15
        
        # Second retry delay should be ~0.2s (2 * 0.1)
        delay2 = attempt_times[2] - attempt_times[1]
        assert 0.18 <= delay2 <= 0.25
    
    def test_network_timeout_with_jitter(self):
        """Test network timeout handling with jitter."""
        def timeout_operation():
            raise TimeoutError("Request timed out")
        
        config = RetryConfiguration(
            max_attempts=3,
            base_delay=0.1,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=True,
            user_prompt=False
        )
        
        start_time = time.time()
        
        with pytest.raises(TimeoutError):
            self.retry_system.execute_with_retry(
                timeout_operation, "network_timeout",
                ErrorCategory.NETWORK, custom_config=config
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should have taken some time for delays (with jitter variation)
        # Expected: ~0.1 + ~0.2 = ~0.3s, but with jitter could be 0.15-0.45s
        assert 0.1 <= total_time <= 0.6
        
        session = self.retry_system.session_history[0]
        assert session.total_attempts == 3
        assert all(isinstance(attempt.error, TimeoutError) for attempt in session.attempts)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])