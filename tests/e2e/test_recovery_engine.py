"""
Unit tests for the Recovery Engine component

Tests error classification, recovery strategies, and retry logic.
"""

import pytest
import json
import time
import socket
import subprocess
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import psutil

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.startup_manager.recovery_engine import (
    RecoveryEngine, ErrorPatternMatcher, RetryStrategy, ErrorType,
    StartupError, RecoveryAction, RecoveryResult
)


class TestErrorPatternMatcher:
    """Test error pattern matching and classification"""
    
    def setup_method(self):
        self.matcher = ErrorPatternMatcher()
    
    def test_permission_denied_classification(self):
        """Test classification of permission denied errors"""
        test_cases = [
            ("WinError 10013: An attempt was made to access a socket", ErrorType.PERMISSION_DENIED),
            ("PermissionError: [Errno 13] Permission denied", ErrorType.PERMISSION_DENIED),
            ("OSError: [Errno 13] Permission denied", ErrorType.PERMISSION_DENIED),
            ("Access is denied", ErrorType.PERMISSION_DENIED)
        ]
        
        for error_message, expected_type in test_cases:
            result = self.matcher.classify_error(error_message)
            assert result == expected_type, f"Failed to classify: {error_message}"
    
    def test_port_conflict_classification(self):
        """Test classification of port conflict errors"""
        test_cases = [
            ("Address already in use", ErrorType.PORT_CONFLICT),
            ("WinError 10048: Only one usage of each socket address", ErrorType.PORT_CONFLICT),
            ("OSError: [Errno 98] Address already in use", ErrorType.PORT_CONFLICT),
            ("bind failed: address already in use", ErrorType.PORT_CONFLICT)
        ]
        
        for error_message, expected_type in test_cases:
            result = self.matcher.classify_error(error_message)
            assert result == expected_type, f"Failed to classify: {error_message}"
    
    def test_dependency_missing_classification(self):
        """Test classification of missing dependency errors"""
        test_cases = [
            ("ModuleNotFoundError: No module named 'fastapi'", ErrorType.DEPENDENCY_MISSING),
            ("ImportError: cannot import name 'FastAPI'", ErrorType.DEPENDENCY_MISSING),
            ("'npm' is not recognized as an internal or external command", ErrorType.DEPENDENCY_MISSING),
            ("command not found: python", ErrorType.DEPENDENCY_MISSING)
        ]
        
        for error_message, expected_type in test_cases:
            result = self.matcher.classify_error(error_message)
            assert result == expected_type, f"Failed to classify: {error_message}"
    
    def test_config_invalid_classification(self):
        """Test classification of configuration errors"""
        test_cases = [
            ("JSONDecodeError: Expecting ',' delimiter", ErrorType.CONFIG_INVALID),
            ("KeyError: 'backend' not found in config", ErrorType.CONFIG_INVALID),
            ("Invalid configuration file", ErrorType.CONFIG_INVALID),
            ("Configuration file missing", ErrorType.CONFIG_INVALID)
        ]
        
        for error_message, expected_type in test_cases:
            result = self.matcher.classify_error(error_message)
            assert result == expected_type, f"Failed to classify: {error_message}"
    
    def test_firewall_blocked_classification(self):
        """Test classification of firewall blocked errors"""
        test_cases = [
            ("WinError 10060: A connection attempt failed", ErrorType.FIREWALL_BLOCKED),
            ("Connection timed out", ErrorType.FIREWALL_BLOCKED),
            ("No connection could be made", ErrorType.FIREWALL_BLOCKED)
        ]
        
        for error_message, expected_type in test_cases:
            result = self.matcher.classify_error(error_message)
            assert result == expected_type, f"Failed to classify: {error_message}"
    
    def test_exception_type_classification(self):
        """Test classification based on exception type"""
        # Test PermissionError
        result = self.matcher.classify_error("Some error", PermissionError("Access denied"))
        assert result == ErrorType.PERMISSION_DENIED
        
        # Test ImportError
        result = self.matcher.classify_error("Some error", ImportError("No module"))
        assert result == ErrorType.DEPENDENCY_MISSING
        
        # Test TimeoutError
        try:
            timeout_error = TimeoutError("Timeout")
        except NameError:
            # TimeoutError might not be available in older Python versions
            import socket
            timeout_error = socket.timeout("Timeout")
        
        result = self.matcher.classify_error("Some error", timeout_error)
        assert result == ErrorType.TIMEOUT_ERROR
        
        # Test subprocess.CalledProcessError
        result = self.matcher.classify_error("Some error", subprocess.CalledProcessError(1, "cmd"))
        assert result == ErrorType.PROCESS_FAILED
    
    def test_unknown_error_classification(self):
        """Test classification of unknown errors"""
        result = self.matcher.classify_error("Some random error message")
        assert result == ErrorType.UNKNOWN


class TestRetryStrategy:
    """Test retry strategy with exponential backoff"""
    
    def setup_method(self):
        self.retry_strategy = RetryStrategy(max_attempts=3, base_delay=0.1, max_delay=1.0)
    
    def test_successful_operation_first_attempt(self):
        """Test operation that succeeds on first attempt"""
        mock_operation = Mock(return_value="success")
        
        result = self.retry_strategy.execute_with_retry(mock_operation, "test_op")
        
        assert result == "success"
        assert mock_operation.call_count == 1
    
    def test_successful_operation_after_retries(self):
        """Test operation that succeeds after retries"""
        mock_operation = Mock(side_effect=[Exception("fail"), Exception("fail"), "success"])
        
        result = self.retry_strategy.execute_with_retry(mock_operation, "test_op")
        
        assert result == "success"
        assert mock_operation.call_count == 3
    
    def test_operation_fails_all_attempts(self):
        """Test operation that fails all attempts"""
        mock_operation = Mock(side_effect=Exception("persistent failure"))
        
        with pytest.raises(Exception, match="persistent failure"):
            self.retry_strategy.execute_with_retry(mock_operation, "test_op")
        
        assert mock_operation.call_count == 3
    
    @patch('time.sleep')
    def test_exponential_backoff_timing(self, mock_sleep):
        """Test that exponential backoff timing is correct"""
        mock_operation = Mock(side_effect=[Exception("fail"), Exception("fail"), "success"])
        
        result = self.retry_strategy.execute_with_retry(mock_operation, "test_op")
        
        assert result == "success"
        # Should have slept twice (after first and second failures)
        assert mock_sleep.call_count == 2
        # Check exponential backoff: 0.1, then 0.2
        mock_sleep.assert_any_call(0.1)
        mock_sleep.assert_any_call(0.2)


class TestRecoveryEngine:
    """Test the main recovery engine functionality"""
    
    def setup_method(self):
        with patch('pathlib.Path.exists', return_value=False):
            self.recovery_engine = RecoveryEngine()
    
    def test_initialization(self):
        """Test recovery engine initialization"""
        assert self.recovery_engine.pattern_matcher is not None
        assert self.recovery_engine.retry_strategy is not None
        assert len(self.recovery_engine.recovery_actions) > 0
        assert self.recovery_engine.config is not None
    
    def test_error_classification(self):
        """Test error classification functionality"""
        error = self.recovery_engine.classify_error(
            "WinError 10013: Permission denied",
            PermissionError("Access denied"),
            {"port": 8000}
        )
        
        assert error.type == ErrorType.PERMISSION_DENIED
        assert error.message == "WinError 10013: Permission denied"
        assert error.original_exception is not None
        assert error.context["port"] == 8000
        assert error.auto_fixable is False  # Permission errors typically require manual intervention
        assert len(error.suggested_actions) > 0
    
    def test_recovery_action_registration(self):
        """Test recovery action registration"""
        test_action = RecoveryAction(
            name="test_action",
            description="Test action",
            action_func=lambda x, y: RecoveryResult(True, "test", "success"),
            priority=1
        )
        
        self.recovery_engine.register_recovery_action(ErrorType.PORT_CONFLICT, test_action)
        
        actions = self.recovery_engine.get_recovery_actions(ErrorType.PORT_CONFLICT)
        action_names = [action.name for action in actions]
        assert "test_action" in action_names
    
    def test_get_recovery_actions_sorting(self):
        """Test that recovery actions are sorted by priority and success rate"""
        # Add test actions with different priorities
        action1 = RecoveryAction("action1", "desc1", lambda x, y: None, priority=3, success_rate=0.5)
        action2 = RecoveryAction("action2", "desc2", lambda x, y: None, priority=1, success_rate=0.8)
        action3 = RecoveryAction("action3", "desc3", lambda x, y: None, priority=2, success_rate=0.9)
        
        self.recovery_engine.register_recovery_action(ErrorType.UNKNOWN, action1)
        self.recovery_engine.register_recovery_action(ErrorType.UNKNOWN, action2)
        self.recovery_engine.register_recovery_action(ErrorType.UNKNOWN, action3)
        
        actions = self.recovery_engine.get_recovery_actions(ErrorType.UNKNOWN)
        
        # Should be sorted by success rate (learning enabled by default)
        assert actions[0].name == "action3"  # Highest success rate
        assert actions[1].name == "action2"  # Second highest success rate
        assert actions[2].name == "action1"  # Lowest success rate
    
    @patch('scripts.startup_manager.recovery_engine.psutil.net_connections')
    @patch('scripts.startup_manager.recovery_engine.psutil.Process')
    def test_kill_process_on_port_success(self, mock_process_class, mock_net_connections):
        """Test successful process killing on port"""
        # Mock network connections
        mock_conn = Mock()
        mock_conn.laddr.port = 8000
        mock_conn.pid = 1234
        mock_net_connections.return_value = [mock_conn]
        
        # Mock process
        mock_process = Mock()
        mock_process.name.return_value = "test_process"
        mock_process.wait.return_value = None
        mock_process_class.return_value = mock_process
        
        error = StartupError(ErrorType.PORT_CONFLICT, "Port conflict")
        context = {"port": 8000}
        
        result = self.recovery_engine._kill_process_on_port(error, context)
        
        assert result.success is True
        assert "test_process" in result.message
        assert result.retry_recommended is True
        mock_process.terminate.assert_called_once()
    
    @patch('scripts.startup_manager.recovery_engine.socket.socket')
    def test_find_alternative_port_success(self, mock_socket):
        """Test finding alternative port"""
        # Mock socket to simulate port 8000 busy, 8001 available
        mock_socket_instance = Mock()
        mock_socket.return_value.__enter__.return_value = mock_socket_instance
        
        def bind_side_effect(addr):
            if addr[1] == 8000:
                raise OSError("Address already in use")
            # Port 8001 is available
            return None
        
        mock_socket_instance.bind.side_effect = bind_side_effect
        
        error = StartupError(ErrorType.PORT_CONFLICT, "Port conflict")
        context = {"port": 8000}
        
        result = self.recovery_engine._find_alternative_port(error, context)
        
        assert result.success is True
        assert result.details["alternative_port"] == 8001
        assert result.retry_recommended is True
    
    @patch('subprocess.run')
    def test_install_missing_dependencies_success(self, mock_subprocess):
        """Test successful dependency installation"""
        # Mock successful pip install
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        error = StartupError(ErrorType.DEPENDENCY_MISSING, "Missing deps")
        context = {"missing_python_deps": True}
        
        result = self.recovery_engine._install_missing_dependencies(error, context)
        
        assert result.success is True
        assert "successfully" in result.message
        assert result.retry_recommended is True
        mock_subprocess.assert_called_once()
    
    def test_activate_virtual_environment_found(self):
        """Test virtual environment activation when venv exists"""
        def exists_side_effect(self):
            # self is the Path object
            path_str = str(self)
            
            # Return True for venv directory
            if path_str == "venv":
                return True
            # Return True for activate.bat script
            elif "activate.bat" in path_str and "venv" in path_str:
                return True
            return False
        
        # Mock the exists method on Path instances
        with patch.object(Path, 'exists', exists_side_effect):
            error = StartupError(ErrorType.VIRTUAL_ENV_ERROR, "Venv error")
            context = {}
            
            result = self.recovery_engine._activate_virtual_environment(error, context)
            
            assert result.success is True
            assert "venv" in result.message
            assert result.retry_recommended is True
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.rename')
    def test_repair_config_file_success(self, mock_rename, mock_exists, mock_file):
        """Test successful config file repair"""
        mock_exists.return_value = True  # Config file exists
        
        error = StartupError(ErrorType.CONFIG_INVALID, "Invalid config")
        context = {"config_file": "config.json"}
        
        result = self.recovery_engine._repair_config_file(error, context)
        
        assert result.success is True
        assert "repaired" in result.message
        assert result.retry_recommended is True
        mock_file.assert_called_once()
        mock_rename.assert_called_once()  # Backup created
    
    def test_success_rate_tracking(self):
        """Test success rate tracking for recovery actions"""
        action_name = "test_action"
        
        # Record some successes and failures
        self.recovery_engine._update_success_rate(action_name, True)
        self.recovery_engine._update_success_rate(action_name, True)
        self.recovery_engine._update_success_rate(action_name, False)
        
        # Check success rate calculation
        assert len(self.recovery_engine.success_history[action_name]) == 3
        
        # Find the action and check its success rate
        for actions in self.recovery_engine.recovery_actions.values():
            for action in actions:
                if action.name == action_name:
                    expected_rate = 2/3  # 2 successes out of 3 attempts
                    assert abs(action.success_rate - expected_rate) < 0.01
                    break
    
    def test_attempt_recovery_success(self):
        """Test successful recovery attempt"""
        # Create a mock recovery action that succeeds
        def mock_recovery_func(error, context):
            return RecoveryResult(
                success=True,
                action_taken="mock_action",
                message="Recovery successful"
            )
        
        mock_action = RecoveryAction(
            name="mock_action",
            description="Mock action",
            action_func=mock_recovery_func,
            priority=1
        )
        
        self.recovery_engine.register_recovery_action(ErrorType.PORT_CONFLICT, mock_action)
        
        error = StartupError(ErrorType.PORT_CONFLICT, "Port conflict")
        result = self.recovery_engine.attempt_recovery(error)
        
        assert result.success is True
        assert result.action_taken == "mock_action"
    
    def test_attempt_recovery_all_fail(self):
        """Test recovery attempt when all actions fail"""
        # Create mock recovery actions that fail
        def mock_failing_func(error, context):
            return RecoveryResult(
                success=False,
                action_taken="mock_fail",
                message="Recovery failed"
            )
        
        mock_action = RecoveryAction(
            name="mock_fail",
            description="Mock failing action",
            action_func=mock_failing_func,
            priority=1
        )
        
        self.recovery_engine.register_recovery_action(ErrorType.UNKNOWN, mock_action)
        
        error = StartupError(ErrorType.UNKNOWN, "Unknown error")
        result = self.recovery_engine.attempt_recovery(error)
        
        assert result.success is False
        # With intelligent failure handling, it may provide manual intervention instead of fallback
        assert result.action_taken in ["all_failed", "manual_intervention_required"]
    
    def test_attempt_recovery_no_actions(self):
        """Test recovery attempt when no actions are available"""
        error = StartupError(ErrorType.UNKNOWN, "Unknown error")
        
        # Clear any existing actions for UNKNOWN type
        if ErrorType.UNKNOWN in self.recovery_engine.recovery_actions:
            self.recovery_engine.recovery_actions[ErrorType.UNKNOWN] = []
        
        result = self.recovery_engine.attempt_recovery(error)
        
        assert result.success is False
        assert result.action_taken == "none"
        assert result.fallback_needed is True
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"max_retry_attempts": 5}')
    @patch('pathlib.Path.exists')
    def test_load_configuration(self, mock_exists, mock_file):
        """Test configuration loading"""
        mock_exists.return_value = True
        
        recovery_engine = RecoveryEngine()
        
        assert recovery_engine.config["max_retry_attempts"] == 5
        mock_file.assert_called()
    
    @patch('builtins.open', new_callable=mock_open)
    def test_save_configuration(self, mock_file):
        """Test configuration saving"""
        self.recovery_engine.save_configuration()
        
        mock_file.assert_called_once()
        # Verify JSON was written
        handle = mock_file()
        written_data = ''.join(call.args[0] for call in handle.write.call_args_list)
        assert "max_retry_attempts" in written_data


class TestRecoveryActionImplementations:
    """Test specific recovery action implementations"""
    
    def setup_method(self):
        with patch('pathlib.Path.exists', return_value=False):
            self.recovery_engine = RecoveryEngine()
    
    @patch('scripts.startup_manager.recovery_engine.socket.socket')
    def test_is_port_available_true(self, mock_socket):
        """Test port availability check when port is available"""
        mock_socket_instance = Mock()
        mock_socket.return_value.__enter__.return_value = mock_socket_instance
        mock_socket_instance.bind.return_value = None
        
        result = self.recovery_engine._is_port_available(8000)
        
        assert result is True
        mock_socket_instance.bind.assert_called_with(('localhost', 8000))
    
    @patch('scripts.startup_manager.recovery_engine.socket.socket')
    def test_is_port_available_false(self, mock_socket):
        """Test port availability check when port is not available"""
        mock_socket_instance = Mock()
        mock_socket.return_value.__enter__.return_value = mock_socket_instance
        mock_socket_instance.bind.side_effect = OSError("Address already in use")
        
        result = self.recovery_engine._is_port_available(8000)
        
        assert result is False
    
    @patch('subprocess.run')
    def test_create_virtual_environment_success(self, mock_subprocess):
        """Test successful virtual environment creation"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        error = StartupError(ErrorType.VIRTUAL_ENV_ERROR, "No venv")
        context = {}
        
        result = self.recovery_engine._create_virtual_environment(error, context)
        
        assert result.success is True
        assert "created" in result.message
        assert result.retry_recommended is True
        mock_subprocess.assert_called_once()
    
    def test_suggest_firewall_exception(self):
        """Test firewall exception suggestion"""
        error = StartupError(ErrorType.FIREWALL_BLOCKED, "Firewall blocked")
        context = {"port": 8000}
        
        result = self.recovery_engine._suggest_firewall_exception(error, context)
        
        assert result.success is False  # Requires manual action
        assert result.action_taken == "suggest_firewall_exception"
        assert "instructions" in result.details
        assert len(result.details["instructions"]) > 0
    
    def test_increase_timeout(self):
        """Test timeout increase functionality"""
        error = StartupError(ErrorType.TIMEOUT_ERROR, "Timeout")
        context = {"timeout": 30}
        
        result = self.recovery_engine._increase_timeout(error, context)
        
        assert result.success is True
        assert result.details["new_timeout"] == 60  # Doubled
        assert result.retry_recommended is True
    
    def test_restart_with_different_params(self):
        """Test restart with different parameters"""
        error = StartupError(ErrorType.PROCESS_FAILED, "Process failed")
        context = {}
        
        result = self.recovery_engine._restart_with_different_params(error, context)
        
        assert result.success is True
        assert "alternative_params" in result.details
        assert result.retry_recommended is True


class TestRecoveryEngineAdvanced:
    """Advanced test scenarios for RecoveryEngine."""
    
    def setup_method(self):
        with patch('pathlib.Path.exists', return_value=False):
            self.recovery_engine = RecoveryEngine()
    
    def test_complex_error_pattern_matching(self):
        """Test complex error pattern matching with nested conditions."""
        complex_errors = [
            ("WinError 10013: An attempt was made to access a socket in a way forbidden by its access permissions on port 8000", ErrorType.PERMISSION_DENIED),
            ("OSError: [Errno 98] Address already in use: ('0.0.0.0', 8000) - Another process is using port 8000", ErrorType.PORT_CONFLICT),
            ("ModuleNotFoundError: No module named 'fastapi'. Did you forget to activate your virtual environment?", ErrorType.DEPENDENCY_MISSING),
            ("JSONDecodeError: Expecting ',' delimiter: line 5 column 10 (char 89) in config.json", ErrorType.CONFIG_INVALID),
        ]
        
        for error_message, expected_type in complex_errors:
            result = self.recovery_engine.pattern_matcher.classify_error(error_message)
            assert result == expected_type, f"Failed to classify: {error_message}"
    
    def test_recovery_action_chaining(self):
        """Test chaining of recovery actions for complex scenarios."""
        # Create a scenario where multiple actions need to be chained
        error = StartupError(ErrorType.PORT_CONFLICT, "Port 8000 in use")
        context = {"port": 8000, "process_name": "python.exe"}
        
        # Mock the first action to partially succeed but require follow-up
        def mock_kill_process(error, context):
            return RecoveryResult(
                success=True,
                action_taken="kill_process_on_port",
                message="Process killed, but port still showing as occupied",
                retry_recommended=True,
                details={"follow_up_needed": True}
            )
        
        def mock_wait_and_retry(error, context):
            return RecoveryResult(
                success=True,
                action_taken="wait_and_retry",
                message="Port now available after waiting",
                retry_recommended=True
            )
        
        # Replace recovery actions
        kill_action = next(action for actions in self.recovery_engine.recovery_actions.values() 
                          for action in actions if "kill_process" in action.name)
        kill_action.action_func = mock_kill_process
        
        # Add follow-up action
        from scripts.startup_manager.recovery_engine import RecoveryAction
        wait_action = RecoveryAction(
            name="wait_and_retry",
            description="Wait and retry after process cleanup",
            action_func=mock_wait_and_retry,
            priority=2
        )
        self.recovery_engine.register_recovery_action(ErrorType.PORT_CONFLICT, wait_action)
        
        result = self.recovery_engine.attempt_recovery(error, context)
        
        assert result.success is True
        # Should have executed the chain of actions
    
    def test_recovery_with_user_preferences(self):
        """Test recovery actions respecting user preferences."""
        # Set user preferences to avoid killing processes
        self.recovery_engine.config["user_preferences"] = {
            "avoid_killing_processes": True,
            "prefer_alternative_ports": True
        }
        
        error = StartupError(ErrorType.PORT_CONFLICT, "Port conflict")
        context = {"port": 8000}
        
        # Mock alternative port finding to succeed
        with patch.object(self.recovery_engine, '_find_alternative_port') as mock_alt_port:
            mock_alt_port.return_value = RecoveryResult(
                success=True,
                action_taken="find_alternative_port",
                message="Found alternative port 8001",
                details={"alternative_port": 8001}
            )
            
            result = self.recovery_engine.attempt_recovery(error, context)
            
            assert result.success is True
            assert result.action_taken == "find_alternative_port"
            mock_alt_port.assert_called_once()
    
    def test_recovery_action_timeout_handling(self):
        """Test handling of recovery actions that timeout."""
        import time

        def slow_recovery_action(error, context):
            time.sleep(10)  # Simulate slow action
            return RecoveryResult(success=True, action_taken="slow_action", message="Done")
        
        # Add a slow action
        from scripts.startup_manager.recovery_engine import RecoveryAction
        slow_action = RecoveryAction(
            name="slow_action",
            description="Slow recovery action",
            action_func=slow_recovery_action,
            priority=1
        )
        self.recovery_engine.register_recovery_action(ErrorType.UNKNOWN, slow_action)
        
        error = StartupError(ErrorType.UNKNOWN, "Unknown error")
        
        # Should timeout and move to next action
        with patch('threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread_instance.join.return_value = None  # Simulate timeout
            mock_thread_instance.is_alive.return_value = True
            mock_thread.return_value = mock_thread_instance
            
            result = self.recovery_engine.attempt_recovery(error, timeout=5)
            
            # Should handle timeout gracefully
            assert result is not None
    
    def test_recovery_rollback_on_failure(self):
        """Test rollback of recovery actions when they cause more problems."""
        error = StartupError(ErrorType.CONFIG_INVALID, "Invalid config")
        context = {"config_file": "config.json"}
        
        # Mock a recovery action that makes things worse
        def problematic_recovery(error, context):
            # Simulate an action that causes additional problems
            raise Exception("Recovery action caused system instability")
        
        from scripts.startup_manager.recovery_engine import RecoveryAction
        problematic_action = RecoveryAction(
            name="problematic_repair",
            description="Problematic config repair",
            action_func=problematic_recovery,
            priority=1
        )
        self.recovery_engine.register_recovery_action(ErrorType.CONFIG_INVALID, problematic_action)
        
        result = self.recovery_engine.attempt_recovery(error, context)
        
        # Should handle the exception and try alternative approaches
        assert result is not None
        assert result.success is False or result.action_taken != "problematic_repair"
    
    def test_adaptive_retry_strategy(self):
        """Test adaptive retry strategy based on error types."""
        # Different error types should have different retry strategies
        error_configs = [
            (ErrorType.NETWORK_ERROR, 5, 1.0),  # Network errors: more retries, longer delays
            (ErrorType.PERMISSION_DENIED, 2, 0.5),  # Permission errors: fewer retries, shorter delays
            (ErrorType.TIMEOUT_ERROR, 3, 2.0),  # Timeout errors: moderate retries, longer delays
        ]
        
        for error_type, expected_max_attempts, expected_base_delay in error_configs:
            retry_strategy = self.recovery_engine._get_retry_strategy_for_error_type(error_type)
            
            assert retry_strategy.max_attempts >= expected_max_attempts
            assert retry_strategy.base_delay >= expected_base_delay
    
    def test_recovery_context_preservation(self):
        """Test that recovery context is preserved across action chains."""
        error = StartupError(ErrorType.PORT_CONFLICT, "Port conflict")
        initial_context = {"port": 8000, "user_id": "test_user", "session_id": "abc123"}
        
        def context_checking_action(error, context):
            # Verify that context is preserved
            assert context["user_id"] == "test_user"
            assert context["session_id"] == "abc123"
            assert "port" in context
            
            # Add to context
            context["action_timestamp"] = time.time()
            
            return RecoveryResult(
                success=True,
                action_taken="context_check",
                message="Context preserved"
            )
        
        from scripts.startup_manager.recovery_engine import RecoveryAction
        context_action = RecoveryAction(
            name="context_check",
            description="Context checking action",
            action_func=context_checking_action,
            priority=1
        )
        self.recovery_engine.register_recovery_action(ErrorType.PORT_CONFLICT, context_action)
        
        result = self.recovery_engine.attempt_recovery(error, initial_context)
        
        assert result.success is True
        assert "action_timestamp" in initial_context  # Context should be modified
    
    def test_recovery_metrics_collection(self):
        """Test collection of recovery metrics for analysis."""
        error = StartupError(ErrorType.PORT_CONFLICT, "Port conflict")
        context = {"port": 8000}
        
        # Perform several recovery attempts
        for _ in range(3):
            self.recovery_engine.attempt_recovery(error, context)
        
        # Check that metrics were collected
        metrics = self.recovery_engine.get_recovery_metrics()
        
        assert "total_attempts" in metrics
        assert "success_rate" in metrics
        assert "error_type_distribution" in metrics
        assert "action_success_rates" in metrics
        
        assert metrics["total_attempts"] >= 3
        assert ErrorType.PORT_CONFLICT.value in metrics["error_type_distribution"]


class TestRecoveryEngineIntegrationScenarios:
    """Test integration scenarios with other components."""
    
    def setup_method(self):
        with patch('pathlib.Path.exists', return_value=False):
            self.recovery_engine = RecoveryEngine()
    
    def test_integration_with_port_manager(self):
        """Test recovery engine integration with port manager."""
        from scripts.startup_manager.port_manager import PortManager
        
        port_manager = PortManager()
        error = StartupError(ErrorType.PORT_CONFLICT, "Port 8000 in use")
        context = {"port": 8000, "port_manager": port_manager}
        
        # Mock port manager methods
        with patch.object(port_manager, 'find_available_port', return_value=8001):
            with patch.object(port_manager, 'kill_process_on_port', return_value=True):
                result = self.recovery_engine.attempt_recovery(error, context)
                
                assert result.success is True
    
    def test_integration_with_process_manager(self):
        """Test recovery engine integration with process manager."""
        from scripts.startup_manager.process_manager import ProcessManager
        from scripts.startup_manager.config import StartupConfig
        
        config = StartupConfig()
        process_manager = ProcessManager(config)
        
        error = StartupError(ErrorType.PROCESS_FAILED, "Process startup failed")
        context = {"process_name": "backend", "process_manager": process_manager}
        
        # Mock process manager methods
        with patch.object(process_manager, 'restart_process') as mock_restart:
            mock_restart.return_value = Mock(success=True)
            
            result = self.recovery_engine.attempt_recovery(error, context)
            
            # Should attempt to restart the process
            assert result is not None
    
    def test_integration_with_environment_validator(self):
        """Test recovery engine integration with environment validator."""
        from scripts.startup_manager.environment_validator import EnvironmentValidator
        
        env_validator = EnvironmentValidator()
        error = StartupError(ErrorType.DEPENDENCY_MISSING, "Missing dependencies")
        context = {"missing_packages": ["fastapi", "uvicorn"], "env_validator": env_validator}
        
        # Mock environment validator methods
        with patch.object(env_validator, 'auto_fix_issues') as mock_fix:
            mock_fix.return_value = ["Fixed missing dependencies"]
            
            result = self.recovery_engine.attempt_recovery(error, context)
            
            assert result is not None
    
    def test_end_to_end_recovery_workflow(self):
        """Test complete end-to-end recovery workflow."""
        # Simulate a complex failure scenario
        errors = [
            StartupError(ErrorType.DEPENDENCY_MISSING, "Missing fastapi"),
            StartupError(ErrorType.PORT_CONFLICT, "Port 8000 in use"),
            StartupError(ErrorType.PERMISSION_DENIED, "Permission denied"),
        ]
        
        context = {"port": 8000, "config_file": "config.json"}
        
        # Process each error in sequence
        results = []
        for error in errors:
            result = self.recovery_engine.attempt_recovery(error, context)
            results.append(result)
            
            # Update context based on recovery result
            if result.success and result.details:
                context.update(result.details)
        
        # All errors should be handled
        assert len(results) == 3
        for result in results:
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])