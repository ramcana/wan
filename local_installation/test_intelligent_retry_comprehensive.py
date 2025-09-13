"""
Comprehensive test scenarios for the Intelligent Retry System.

This test demonstrates:
1. Configurable retry counts and user control
2. Exponential backoff with jitter for network operations
3. Retry strategy selection based on error type and context
4. Integration with existing error handler
"""

import sys
import time
import logging
from scripts.retry_system import IntelligentRetrySystem, RetryConfiguration, RetryStrategy
from scripts.error_handler import ComprehensiveErrorHandler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_configurable_retry_counts():
    """Test configurable retry counts with different configurations."""
    print("Testing configurable retry counts...")
    
    retry_system = IntelligentRetrySystem("/test/path")
    
    # Test with different max attempts
    configs = [
        RetryConfiguration(max_attempts=1, user_prompt=False),
        RetryConfiguration(max_attempts=3, user_prompt=False),
        RetryConfiguration(max_attempts=5, user_prompt=False)
    ]
    
    for i, config in enumerate(configs):
        attempt_count = 0
        
        def counting_failure():
            nonlocal attempt_count
            attempt_count += 1
            raise ConnectionError(f"Failure {attempt_count}")
        
        try:
            retry_system.execute_with_retry(
                counting_failure, f"test_config_{i}", custom_config=config
            )
        except ConnectionError:
            pass
        
        # Verify the correct number of attempts were made
        session = retry_system.session_history[-1]
        assert session.total_attempts == config.max_attempts, \
            f"Expected {config.max_attempts} attempts, got {session.total_attempts}"
        assert attempt_count == config.max_attempts
    
    print("✓ Configurable retry counts test passed")


def test_exponential_backoff_with_jitter():
    """Test exponential backoff with and without jitter."""
    print("Testing exponential backoff with jitter...")
    
    retry_system = IntelligentRetrySystem("/test/path")
    
    # Test without jitter (predictable delays)
    attempt_times_no_jitter = []
    attempt_count = 0
    
    def network_failure_no_jitter():
        nonlocal attempt_count
        attempt_count += 1
        attempt_times_no_jitter.append(time.time())
        if attempt_count < 4:
            raise ConnectionError("Network timeout")
        return "success"
    
    config_no_jitter = RetryConfiguration(
        max_attempts=5,
        base_delay=0.1,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        backoff_multiplier=2.0,
        jitter=False,
        user_prompt=False
    )
    
    result = retry_system.execute_with_retry(
        network_failure_no_jitter, "network_no_jitter", custom_config=config_no_jitter
    )
    
    assert result == "success"
    assert len(attempt_times_no_jitter) == 4
    
    # Verify exponential backoff delays (0.1, 0.2, 0.4)
    delays_no_jitter = [
        attempt_times_no_jitter[i+1] - attempt_times_no_jitter[i] 
        for i in range(len(attempt_times_no_jitter)-1)
    ]
    
    expected_delays = [0.1, 0.2, 0.4]
    for i, (actual, expected) in enumerate(zip(delays_no_jitter, expected_delays)):
        assert abs(actual - expected) < 0.05, \
            f"Delay {i}: expected ~{expected}s, got {actual:.3f}s"
    
    # Test with jitter (variable delays)
    attempt_times_jitter = []
    attempt_count = 0
    
    def network_failure_jitter():
        nonlocal attempt_count
        attempt_count += 1
        attempt_times_jitter.append(time.time())
        if attempt_count < 4:
            raise ConnectionError("Network timeout with jitter")
        return "success with jitter"
    
    config_jitter = RetryConfiguration(
        max_attempts=5,
        base_delay=0.1,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        backoff_multiplier=2.0,
        jitter=True,
        user_prompt=False
    )
    
    result = retry_system.execute_with_retry(
        network_failure_jitter, "network_jitter", custom_config=config_jitter
    )
    
    assert result == "success with jitter"
    assert len(attempt_times_jitter) == 4
    
    # Verify jitter is applied (delays should be 50-100% of expected)
    delays_jitter = [
        attempt_times_jitter[i+1] - attempt_times_jitter[i] 
        for i in range(len(attempt_times_jitter)-1)
    ]
    
    for i, (jitter_delay, expected) in enumerate(zip(delays_jitter, expected_delays)):
        min_expected = expected * 0.5
        max_expected = expected * 1.0
        assert min_expected <= jitter_delay <= max_expected, \
            f"Jitter delay {i}: expected {min_expected:.3f}-{max_expected:.3f}s, got {jitter_delay:.3f}s"
    
    print("✓ Exponential backoff with jitter test passed")


def test_retry_strategy_selection():
    """Test different retry strategies."""
    print("Testing retry strategy selection...")
    
    retry_system = IntelligentRetrySystem("/test/path")
    
    # Test all retry strategies
    strategies_to_test = [
        (RetryStrategy.EXPONENTIAL_BACKOFF, "exponential"),
        (RetryStrategy.LINEAR_BACKOFF, "linear"),
        (RetryStrategy.FIXED_DELAY, "fixed"),
        (RetryStrategy.IMMEDIATE, "immediate")
    ]
    
    for strategy, name in strategies_to_test:
        attempt_times = []
        attempt_count = 0
        
        def strategy_test():
            nonlocal attempt_count
            attempt_count += 1
            attempt_times.append(time.time())
            if attempt_count < 3:
                raise ValueError(f"{name} strategy test error")
            return f"{name} success"
        
        config = RetryConfiguration(
            max_attempts=4,
            base_delay=0.05,
            strategy=strategy,
            backoff_multiplier=2.0,
            jitter=False,
            user_prompt=False
        )
        
        result = retry_system.execute_with_retry(
            strategy_test, f"strategy_{name}", custom_config=config
        )
        
        assert result == f"{name} success"
        assert len(attempt_times) == 3
        
        # Verify strategy-specific delay patterns
        if len(attempt_times) >= 3:
            delay1 = attempt_times[1] - attempt_times[0]
            delay2 = attempt_times[2] - attempt_times[1]
            
            if strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
                # Delays: 0.05, 0.10
                assert 0.04 <= delay1 <= 0.06, f"Exponential delay1: {delay1}"
                assert 0.09 <= delay2 <= 0.11, f"Exponential delay2: {delay2}"
            elif strategy == RetryStrategy.LINEAR_BACKOFF:
                # Delays: 0.05 * (0+1) = 0.05, 0.05 * (1+1) = 0.10
                assert 0.04 <= delay1 <= 0.06, f"Linear delay1: {delay1}"
                assert 0.09 <= delay2 <= 0.11, f"Linear delay2: {delay2}"
            elif strategy == RetryStrategy.FIXED_DELAY:
                # Delays: 0.05, 0.05
                assert 0.04 <= delay1 <= 0.06, f"Fixed delay1: {delay1}"
                assert 0.04 <= delay2 <= 0.06, f"Fixed delay2: {delay2}"
            elif strategy == RetryStrategy.IMMEDIATE:
                # Delays: 0, 0
                assert delay1 < 0.01, f"Immediate delay1: {delay1}"
                assert delay2 < 0.01, f"Immediate delay2: {delay2}"
    
    print("✓ Retry strategy selection test passed")


def test_error_handler_integration():
    """Test integration with the existing error handler."""
    print("Testing error handler integration...")
    
    error_handler = ComprehensiveErrorHandler("/test/path")
    
    # Test successful operation through error handler
    def successful_operation():
        return "integrated success"
    
    result = error_handler.execute_with_intelligent_retry(
        successful_operation, "integration_test"
    )
    
    assert result == "integrated success"
    
    # Verify retry system was used
    retry_stats = error_handler.get_retry_statistics()
    assert retry_stats['intelligent_retry_system']['total_sessions'] >= 1
    assert retry_stats['intelligent_retry_system']['successful_sessions'] >= 1
    
    # Test retry configuration through error handler
    from scripts.interfaces import ErrorCategory
    
    new_config = RetryConfiguration(
        max_attempts=7,
        base_delay=1.5,
        strategy=RetryStrategy.LINEAR_BACKOFF
    )
    
    error_handler.configure_retry_behavior(ErrorCategory.NETWORK, new_config)
    
    # Verify configuration was applied
    retry_configs = retry_stats['retry_configurations']
    assert 'network' in retry_configs
    
    print("✓ Error handler integration test passed")


def test_max_delay_limits():
    """Test that max delay limits are respected."""
    print("Testing max delay limits...")
    
    retry_system = IntelligentRetrySystem("/test/path")
    
    attempt_times = []
    attempt_count = 0
    
    def delay_limit_test():
        nonlocal attempt_count
        attempt_count += 1
        attempt_times.append(time.time())
        if attempt_count < 5:
            raise ConnectionError("Testing delay limits")
        return "delay limit success"
    
    config = RetryConfiguration(
        max_attempts=6,
        base_delay=1.0,
        max_delay=2.0,  # Limit delays to 2 seconds
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        backoff_multiplier=3.0,  # Would normally create very long delays
        jitter=False,
        user_prompt=False
    )
    
    result = retry_system.execute_with_retry(
        delay_limit_test, "delay_limit_test", custom_config=config
    )
    
    assert result == "delay limit success"
    assert len(attempt_times) == 5
    
    # Verify that no delay exceeds max_delay
    delays = [
        attempt_times[i+1] - attempt_times[i] 
        for i in range(len(attempt_times)-1)
    ]
    
    for i, delay in enumerate(delays):
        assert delay <= config.max_delay + 0.1, \
            f"Delay {i} ({delay:.3f}s) exceeded max_delay ({config.max_delay}s)"
    
    print("✓ Max delay limits test passed")


def test_session_management():
    """Test session tracking and management."""
    print("Testing session management...")
    
    retry_system = IntelligentRetrySystem("/test/path")
    
    # Test multiple concurrent-like sessions
    def operation1():
        time.sleep(0.01)
        return "op1 result"
    
    def operation2():
        raise ValueError("op2 error")
    
    def operation3():
        return "op3 result"
    
    # Execute operations
    result1 = retry_system.execute_with_retry(operation1, "operation1")
    assert result1 == "op1 result"
    
    try:
        retry_system.execute_with_retry(
            operation2, "operation2", 
            custom_config=RetryConfiguration(max_attempts=1, user_prompt=False)
        )
    except ValueError:
        pass
    
    result3 = retry_system.execute_with_retry(operation3, "operation3")
    assert result3 == "op3 result"
    
    # Verify session history
    assert len(retry_system.session_history) == 3
    
    sessions = retry_system.session_history
    assert sessions[0].operation_name == "operation1"
    assert sessions[0].successful is True
    assert sessions[1].operation_name == "operation2"
    assert sessions[1].successful is False
    assert sessions[2].operation_name == "operation3"
    assert sessions[2].successful is True
    
    # Test session statistics
    stats = retry_system.get_session_statistics()
    assert stats['total_sessions'] == 3
    assert stats['successful_sessions'] == 2
    assert stats['failed_sessions'] == 1
    assert abs(stats['success_rate'] - (2/3)) < 0.01
    
    print("✓ Session management test passed")


def main():
    """Run all comprehensive tests."""
    print("Running comprehensive intelligent retry system tests...")
    print("=" * 60)
    
    try:
        test_configurable_retry_counts()
        test_exponential_backoff_with_jitter()
        test_retry_strategy_selection()
        test_error_handler_integration()
        test_max_delay_limits()
        test_session_management()
        
        print("=" * 60)
        print("✓ All comprehensive tests passed!")
        print("\nThe intelligent retry system successfully implements:")
        print("  • Configurable retry counts and user control")
        print("  • Exponential backoff with jitter for network operations")
        print("  • Retry strategy selection based on error type and context")
        print("  • Integration with existing error handler")
        print("  • Comprehensive session tracking and analytics")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
