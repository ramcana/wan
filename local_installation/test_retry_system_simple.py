"""
Simple test for the intelligent retry system.
"""

import sys
import time
import logging
from scripts.retry_system import IntelligentRetrySystem, RetryConfiguration, RetryStrategy

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_successful_operation():
    """Test a successful operation that doesn't need retry."""
    print("Testing successful operation...")
    
    retry_system = IntelligentRetrySystem("/test/path")
    
    def success_operation():
        return "success"
    
    result = retry_system.execute_with_retry(success_operation, "test_success")
    
    assert result == "success"
    assert len(retry_system.session_history) == 1
    assert retry_system.session_history[0].successful is True
    assert retry_system.session_history[0].total_attempts == 1
    
    print("✓ Successful operation test passed")


def test_retry_with_exponential_backoff():
    """Test retry with exponential backoff."""
    print("Testing retry with exponential backoff...")
    
    retry_system = IntelligentRetrySystem("/test/path")
    
    attempt_count = 0
    attempt_times = []
    
    def flaky_operation():
        nonlocal attempt_count
        attempt_count += 1
        attempt_times.append(time.time())
        
        if attempt_count < 3:
            raise ConnectionError(f"Network error on attempt {attempt_count}")
        return f"success after {attempt_count} attempts"
    
    config = RetryConfiguration(
        max_attempts=5,
        base_delay=0.1,  # Short delay for testing
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        backoff_multiplier=2.0,
        jitter=False,
        user_prompt=False
    )
    
    result = retry_system.execute_with_retry(
        flaky_operation, "test_retry", custom_config=config
    )
    
    assert result == "success after 3 attempts"
    assert attempt_count == 3
    assert len(attempt_times) == 3
    
    # Check exponential backoff timing
    if len(attempt_times) >= 3:
        delay1 = attempt_times[1] - attempt_times[0]
        delay2 = attempt_times[2] - attempt_times[1]
        
        # First retry delay should be ~0.1s, second should be ~0.2s
        assert 0.08 <= delay1 <= 0.15, f"First delay was {delay1}, expected ~0.1"
        assert 0.18 <= delay2 <= 0.25, f"Second delay was {delay2}, expected ~0.2"
    
    print("✓ Exponential backoff test passed")


def test_max_attempts_exceeded():
    """Test behavior when max attempts are exceeded."""
    print("Testing max attempts exceeded...")
    
    retry_system = IntelligentRetrySystem("/test/path")
    
    def always_fail():
        raise ValueError("Always fails")
    
    config = RetryConfiguration(
        max_attempts=3,
        base_delay=0.01,  # Very short delay for testing
        user_prompt=False
    )
    
    try:
        retry_system.execute_with_retry(
            always_fail, "test_fail", custom_config=config
        )
        assert False, "Should have raised an exception"
    except ValueError as e:
        assert str(e) == "Always fails"
    
    session = retry_system.session_history[0]
    assert session.successful is False
    assert session.total_attempts == 3
    assert len(session.attempts) == 3
    assert all(not attempt.success for attempt in session.attempts)
    
    print("✓ Max attempts exceeded test passed")


def test_session_statistics():
    """Test session statistics calculation."""
    print("Testing session statistics...")
    
    retry_system = IntelligentRetrySystem("/test/path")
    
    # Run some operations
    def success():
        return "ok"
    
    def failure():
        raise ValueError("Error")
    
    config = RetryConfiguration(max_attempts=1, user_prompt=False)
    
    # Successful operation
    retry_system.execute_with_retry(success, "success_op")
    
    # Failed operation
    try:
        retry_system.execute_with_retry(failure, "failure_op", custom_config=config)
    except ValueError:
        pass
    
    stats = retry_system.get_session_statistics()
    
    assert stats['total_sessions'] == 2
    assert stats['successful_sessions'] == 1
    assert stats['failed_sessions'] == 1
    assert stats['success_rate'] == 0.5
    assert stats['average_attempts'] == 1.0
    
    print("✓ Session statistics test passed")


def test_different_retry_strategies():
    """Test different retry strategies."""
    print("Testing different retry strategies...")
    
    retry_system = IntelligentRetrySystem("/test/path")
    
    # Test linear backoff
    attempt_times = []
    attempt_count = 0
    
    def linear_fail():
        nonlocal attempt_count
        attempt_count += 1
        attempt_times.append(time.time())
        if attempt_count < 3:
            raise ConnectionError("Linear test error")
        return "linear success"
    
    config = RetryConfiguration(
        max_attempts=4,
        base_delay=0.05,
        strategy=RetryStrategy.LINEAR_BACKOFF,
        jitter=False,
        user_prompt=False
    )
    
    result = retry_system.execute_with_retry(
        linear_fail, "linear_test", custom_config=config
    )
    
    assert result == "linear success"
    assert attempt_count == 3
    
    # Check linear backoff timing (delays should be 0.05, 0.10, ...)
    if len(attempt_times) >= 3:
        delay1 = attempt_times[1] - attempt_times[0]
        delay2 = attempt_times[2] - attempt_times[1]
        
        assert 0.04 <= delay1 <= 0.08, f"First delay was {delay1}, expected ~0.05"
        assert 0.08 <= delay2 <= 0.12, f"Second delay was {delay2}, expected ~0.10"
    
    print("✓ Different retry strategies test passed")


def main():
    """Run all tests."""
    print("Running intelligent retry system tests...")
    print("=" * 50)
    
    try:
        test_successful_operation()
        test_retry_with_exponential_backoff()
        test_max_attempts_exceeded()
        test_session_statistics()
        test_different_retry_strategies()
        
        print("=" * 50)
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()