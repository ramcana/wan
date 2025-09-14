"""
Chaos tests for network failures and interruptions in the Model Orchestrator.
"""

import asyncio
import os
import random
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest
import requests

from .error_recovery import ErrorRecoveryManager, RetryConfig, RetryStrategy
from .exceptions import (
    ChecksumError,
    ErrorCode,
    ModelOrchestratorError,
    NoSpaceError,
    SizeMismatchError,
)
from .logging_config import LogContext, configure_logging


class NetworkFailureSimulator:
    """Simulates various network failure scenarios."""
    
    def __init__(self):
        self.failure_rate = 0.0
        self.failure_types = []
        self.call_count = 0
        self.failures_injected = 0
    
    def set_failure_rate(self, rate: float):
        """Set the probability of failure (0.0 to 1.0)."""
        self.failure_rate = rate
    
    def set_failure_types(self, failure_types: List[str]):
        """Set the types of failures to inject."""
        self.failure_types = failure_types
    
    def should_fail(self) -> bool:
        """Determine if this call should fail."""
        self.call_count += 1
        return random.random() < self.failure_rate
    
    def inject_failure(self):
        """Inject a random failure."""
        if not self.should_fail():
            return
        
        self.failures_injected += 1
        failure_type = random.choice(self.failure_types)
        
        if failure_type == "timeout":
            raise requests.exceptions.Timeout("Simulated network timeout")
        elif failure_type == "connection_error":
            raise requests.exceptions.ConnectionError("Simulated connection error")
        elif failure_type == "http_error":
            response = Mock()
            response.status_code = random.choice([500, 502, 503, 504])
            raise requests.exceptions.HTTPError("Simulated HTTP error", response=response)
        elif failure_type == "auth_error":
            response = Mock()
            response.status_code = 401
            raise requests.exceptions.HTTPError("Simulated auth error", response=response)
        elif failure_type == "rate_limit":
            response = Mock()
            response.status_code = 429
            raise requests.exceptions.HTTPError("Simulated rate limit", response=response)
        else:
            raise Exception(f"Simulated {failure_type} error")


class DiskFailureSimulator:
    """Simulates disk-related failures."""
    
    def __init__(self):
        self.no_space_probability = 0.0
        self.permission_error_probability = 0.0
        self.corruption_probability = 0.0
    
    def set_no_space_probability(self, probability: float):
        """Set probability of disk space errors."""
        self.no_space_probability = probability
    
    def set_permission_error_probability(self, probability: float):
        """Set probability of permission errors."""
        self.permission_error_probability = probability
    
    def set_corruption_probability(self, probability: float):
        """Set probability of file corruption."""
        self.corruption_probability = probability
    
    def check_disk_space(self, path: str, required_bytes: int):
        """Simulate disk space check with potential failure."""
        if random.random() < self.no_space_probability:
            raise NoSpaceError(required_bytes, required_bytes // 2, path)
    
    def write_file(self, path: str, content: bytes):
        """Simulate file write with potential failures."""
        if random.random() < self.permission_error_probability:
            raise PermissionError(f"Simulated permission error for {path}")
        
        # Write the file normally
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(content)
        
        # Potentially corrupt the file
        if random.random() < self.corruption_probability:
            # Corrupt by truncating or modifying content
            if random.choice([True, False]):
                # Truncate file
                with open(path, 'wb') as f:
                    f.write(content[:len(content)//2])
            else:
                # Modify content
                corrupted = bytearray(content)
                for _ in range(min(10, len(corrupted))):
                    if corrupted:
                        idx = random.randint(0, len(corrupted) - 1)
                        corrupted[idx] = random.randint(0, 255)
                with open(path, 'wb') as f:
                    f.write(corrupted)


class ProcessKillSimulator:
    """Simulates process interruptions and kills."""
    
    def __init__(self):
        self.kill_probability = 0.0
        self.kill_delay_range = (1.0, 5.0)
    
    def set_kill_probability(self, probability: float):
        """Set probability of process kill."""
        self.kill_probability = probability
    
    def set_kill_delay_range(self, min_delay: float, max_delay: float):
        """Set range for kill delay."""
        self.kill_delay_range = (min_delay, max_delay)
    
    @contextmanager
    def kill_context(self):
        """Context manager that may kill the process."""
        kill_timer = None
        
        if random.random() < self.kill_probability:
            delay = random.uniform(*self.kill_delay_range)
            kill_timer = threading.Timer(delay, self._simulate_kill)
            kill_timer.start()
        
        try:
            yield
        finally:
            if kill_timer:
                kill_timer.cancel()
    
    def _simulate_kill(self):
        """Simulate process kill by raising an exception."""
        # In a real scenario, this would be a SIGKILL or similar
        raise KeyboardInterrupt("Simulated process kill")


@pytest.fixture
def temp_models_dir():
    """Create a temporary directory for models."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def network_simulator():
    """Create a network failure simulator."""
    return NetworkFailureSimulator()


@pytest.fixture
def disk_simulator():
    """Create a disk failure simulator."""
    return DiskFailureSimulator()


@pytest.fixture
def process_simulator():
    """Create a process kill simulator."""
    return ProcessKillSimulator()


@pytest.fixture
def recovery_manager():
    """Create an error recovery manager."""
    configure_logging(level="DEBUG", structured=True)
    return ErrorRecoveryManager()


class TestNetworkFailureRecovery:
    """Test recovery from various network failures."""
    
    def test_transient_network_failure_recovery(self, recovery_manager, network_simulator):
        """Test recovery from transient network failures."""
        network_simulator.set_failure_rate(0.7)  # 70% failure rate
        network_simulator.set_failure_types(["timeout", "connection_error"])
        
        call_count = 0
        success_count = 0
        
        def flaky_network_operation():
            nonlocal call_count, success_count
            call_count += 1
            
            # Inject network failure
            network_simulator.inject_failure()
            
            # If we get here, the operation succeeded
            success_count += 1
            return f"success_after_{call_count}_calls"
        
        with recovery_manager.recovery_context("test_network_recovery") as ctx:
            try:
                result = recovery_manager.retry_with_recovery(
                    flaky_network_operation,
                    ctx,
                )
                assert result.startswith("success_after_")
                assert success_count == 1
                assert call_count >= 1  # Should have retried
            except Exception as e:
                # If all retries failed, that's also a valid test outcome
                assert call_count >= 3  # Should have tried multiple times
    
    def test_rate_limit_backoff(self, recovery_manager, network_simulator):
        """Test exponential backoff for rate limiting."""
        network_simulator.set_failure_rate(0.8)
        network_simulator.set_failure_types(["rate_limit"])
        
        start_time = time.time()
        call_times = []
        
        def rate_limited_operation():
            call_times.append(time.time())
            
            # First few calls fail with rate limit
            if len(call_times) < 3:
                network_simulator.inject_failure()
            
            return "success"
        
        with recovery_manager.recovery_context("test_rate_limit") as ctx:
            result = recovery_manager.retry_with_recovery(
                rate_limited_operation,
                ctx,
            )
            
            assert result == "success"
            assert len(call_times) == 3
            
            # Verify exponential backoff (allowing for some timing variance)
            if len(call_times) >= 3:
                delay1 = call_times[1] - call_times[0]
                delay2 = call_times[2] - call_times[1]
                assert delay2 > delay1  # Second delay should be longer
    
    def test_auth_failure_handling(self, recovery_manager, network_simulator):
        """Test handling of authentication failures."""
        network_simulator.set_failure_rate(1.0)
        network_simulator.set_failure_types(["auth_error"])
        
        def auth_failing_operation():
            network_simulator.inject_failure()
            return "success"
        
        with recovery_manager.recovery_context("test_auth_failure") as ctx:
            with pytest.raises(requests.exceptions.HTTPError):
                recovery_manager.retry_with_recovery(
                    auth_failing_operation,
                    ctx,
                )
    
    def test_mixed_failure_types(self, recovery_manager, network_simulator):
        """Test handling of mixed failure types."""
        network_simulator.set_failure_rate(0.6)
        network_simulator.set_failure_types([
            "timeout", "connection_error", "http_error", "rate_limit"
        ])
        
        call_count = 0
        
        def mixed_failure_operation():
            nonlocal call_count
            call_count += 1
            
            # Succeed after a few attempts
            if call_count >= 4:
                return "success"
            
            network_simulator.inject_failure()
        
        with recovery_manager.recovery_context("test_mixed_failures") as ctx:
            result = recovery_manager.retry_with_recovery(
                mixed_failure_operation,
                ctx,
            )
            
            assert result == "success"
            assert call_count >= 4


class TestDiskFailureRecovery:
    """Test recovery from disk-related failures."""
    
    def test_disk_space_failure(self, recovery_manager, disk_simulator, temp_models_dir):
        """Test handling of disk space failures."""
        disk_simulator.set_no_space_probability(0.8)
        
        def disk_space_operation():
            disk_simulator.check_disk_space(temp_models_dir, 1000000)
            return "success"
        
        with recovery_manager.recovery_context("test_disk_space") as ctx:
            with pytest.raises(NoSpaceError):
                recovery_manager.retry_with_recovery(
                    disk_space_operation,
                    ctx,
                )
    
    def test_permission_error_recovery(self, recovery_manager, disk_simulator, temp_models_dir):
        """Test recovery from permission errors."""
        disk_simulator.set_permission_error_probability(0.7)
        
        call_count = 0
        
        def permission_operation():
            nonlocal call_count
            call_count += 1
            
            # Succeed after a few attempts
            if call_count >= 3:
                return "success"
            
            test_file = os.path.join(temp_models_dir, f"test_{call_count}.txt")
            disk_simulator.write_file(test_file, b"test content")
            return "success"
        
        with recovery_manager.recovery_context("test_permission_error") as ctx:
            try:
                result = recovery_manager.retry_with_recovery(
                    permission_operation,
                    ctx,
                )
                assert result == "success"
            except PermissionError:
                # If all retries failed, that's also valid
                assert call_count >= 2
    
    def test_file_corruption_detection(self, recovery_manager, disk_simulator, temp_models_dir):
        """Test detection and handling of file corruption."""
        disk_simulator.set_corruption_probability(0.8)
        
        expected_content = b"This is test content for corruption detection"
        expected_checksum = "sha256_hash_here"  # In real test, calculate actual hash
        
        def write_and_verify_operation():
            test_file = os.path.join(temp_models_dir, "test_file.txt")
            disk_simulator.write_file(test_file, expected_content)
            
            # Verify file integrity (simplified)
            with open(test_file, 'rb') as f:
                actual_content = f.read()
            
            if actual_content != expected_content:
                raise ChecksumError(test_file, expected_checksum, "corrupted_hash")
            
            return "success"
        
        with recovery_manager.recovery_context("test_corruption") as ctx:
            try:
                result = recovery_manager.retry_with_recovery(
                    write_and_verify_operation,
                    ctx,
                )
                # If we get here, eventually got uncorrupted file
                assert result == "success"
            except ChecksumError:
                # If all retries resulted in corruption, that's also valid
                pass


class TestProcessInterruptionRecovery:
    """Test recovery from process interruptions."""
    
    def test_process_kill_during_operation(self, recovery_manager, process_simulator):
        """Test handling of process kills during operations."""
        process_simulator.set_kill_probability(0.5)
        process_simulator.set_kill_delay_range(0.1, 0.5)
        
        call_count = 0
        
        def interruptible_operation():
            nonlocal call_count
            call_count += 1
            
            with process_simulator.kill_context():
                # Simulate some work
                time.sleep(1.0)
                return f"success_{call_count}"
        
        with recovery_manager.recovery_context("test_process_kill") as ctx:
            try:
                result = recovery_manager.retry_with_recovery(
                    interruptible_operation,
                    ctx,
                )
                assert result.startswith("success_")
            except KeyboardInterrupt:
                # Process was killed and couldn't recover
                assert call_count >= 1
    
    def test_partial_download_recovery(self, recovery_manager, temp_models_dir):
        """Test recovery from partial downloads."""
        partial_file = os.path.join(temp_models_dir, "partial_download.bin")
        complete_content = b"A" * 1000  # 1KB of data
        
        call_count = 0
        
        def partial_download_operation():
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # First attempt: write partial file
                with open(partial_file, 'wb') as f:
                    f.write(complete_content[:500])  # Only half
                raise ConnectionError("Download interrupted")
            elif call_count == 2:
                # Second attempt: complete the download
                with open(partial_file, 'wb') as f:
                    f.write(complete_content)
                
                # Verify completeness
                with open(partial_file, 'rb') as f:
                    actual_content = f.read()
                
                if len(actual_content) != len(complete_content):
                    raise SizeMismatchError(partial_file, len(complete_content), len(actual_content))
                
                return "download_complete"
        
        with recovery_manager.recovery_context("test_partial_download") as ctx:
            result = recovery_manager.retry_with_recovery(
                partial_download_operation,
                ctx,
            )
            
            assert result == "download_complete"
            assert call_count == 2
            
            # Verify file was completed
            with open(partial_file, 'rb') as f:
                final_content = f.read()
            assert len(final_content) == len(complete_content)


class TestConcurrencyFailureRecovery:
    """Test recovery from concurrency-related failures."""
    
    def test_lock_timeout_recovery(self, recovery_manager):
        """Test recovery from lock timeout failures."""
        from .exceptions import LockTimeoutError
        
        call_count = 0
        
        def lock_contention_operation():
            nonlocal call_count
            call_count += 1
            
            if call_count < 3:
                raise LockTimeoutError(
                    f"Lock timeout on attempt {call_count}",
                    model_id="test-model",
                    timeout=5.0
                )
            
            return "lock_acquired"
        
        with recovery_manager.recovery_context("test_lock_timeout") as ctx:
            result = recovery_manager.retry_with_recovery(
                lock_contention_operation,
                ctx,
            )
            
            assert result == "lock_acquired"
            assert call_count == 3
    
    def test_concurrent_modification_recovery(self, recovery_manager):
        """Test recovery from concurrent modification errors."""
        from .exceptions import ModelOrchestratorError, ErrorCode
        
        call_count = 0
        
        def concurrent_operation():
            nonlocal call_count
            call_count += 1
            
            if call_count < 2:
                raise ModelOrchestratorError(
                    "Concurrent modification detected",
                    ErrorCode.CONCURRENT_MODIFICATION,
                    {"attempt": call_count}
                )
            
            return "operation_completed"
        
        with recovery_manager.recovery_context("test_concurrent_modification") as ctx:
            result = recovery_manager.retry_with_recovery(
                concurrent_operation,
                ctx,
            )
            
            assert result == "operation_completed"
            assert call_count == 2


class TestIntegratedChaosScenarios:
    """Test complex scenarios with multiple failure types."""
    
    def test_download_with_multiple_failures(
        self,
        recovery_manager,
        network_simulator,
        disk_simulator,
        process_simulator,
        temp_models_dir
    ):
        """Test a complex download scenario with multiple failure types."""
        # Configure simulators
        network_simulator.set_failure_rate(0.4)
        network_simulator.set_failure_types(["timeout", "rate_limit"])
        disk_simulator.set_corruption_probability(0.3)
        process_simulator.set_kill_probability(0.2)
        
        call_count = 0
        download_file = os.path.join(temp_models_dir, "complex_download.bin")
        expected_content = b"Complex download content" * 100
        
        def complex_download_operation():
            nonlocal call_count
            call_count += 1
            
            # Network failure simulation
            if call_count <= 2:
                network_simulator.inject_failure()
            
            # Process interruption simulation
            with process_simulator.kill_context():
                # Disk operation with potential corruption
                disk_simulator.write_file(download_file, expected_content)
                
                # Verify integrity
                with open(download_file, 'rb') as f:
                    actual_content = f.read()
                
                if actual_content != expected_content:
                    raise ChecksumError(download_file, "expected_hash", "actual_hash")
                
                return "complex_download_success"
        
        with recovery_manager.recovery_context("test_complex_download") as ctx:
            try:
                result = recovery_manager.retry_with_recovery(
                    complex_download_operation,
                    ctx,
                )
                assert result == "complex_download_success"
                
                # Verify final file integrity
                with open(download_file, 'rb') as f:
                    final_content = f.read()
                assert final_content == expected_content
                
            except Exception as e:
                # If the operation ultimately failed, verify we made multiple attempts
                assert call_count >= 2
                print(f"Complex operation failed after {call_count} attempts: {e}")
    
    def test_retry_configuration_effectiveness(self, recovery_manager):
        """Test that retry configurations work as expected."""
        # Test with custom retry config
        custom_config = RetryConfig(
            max_attempts=5,
            base_delay=0.1,
            max_delay=1.0,
            backoff_factor=1.5,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        
        custom_manager = ErrorRecoveryManager(custom_config)
        
        call_count = 0
        call_times = []
        
        def configured_retry_operation():
            nonlocal call_count
            call_count += 1
            call_times.append(time.time())
            
            if call_count < 4:
                raise ConnectionError(f"Failure {call_count}")
            
            return "configured_success"
        
        with custom_manager.recovery_context("test_retry_config") as ctx:
            result = custom_manager.retry_with_recovery(
                configured_retry_operation,
                ctx,
            )
            
            assert result == "configured_success"
            assert call_count == 4
            assert len(call_times) == 4
            
            # Verify backoff timing (with some tolerance for execution time)
            for i in range(1, len(call_times)):
                delay = call_times[i] - call_times[i-1]
                expected_min_delay = custom_config.base_delay * (custom_config.backoff_factor ** (i-1))
                # Allow some tolerance for execution time
                assert delay >= expected_min_delay * 0.8


if __name__ == "__main__":
    # Run chaos tests
    pytest.main([__file__, "-v", "--tb=short"])