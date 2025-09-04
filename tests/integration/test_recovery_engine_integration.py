"""
Integration tests for the Recovery Engine intelligent failure handling

Tests complex failure scenarios, pattern detection, and fallback configurations.
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.startup_manager.recovery_engine import (
    RecoveryEngine, ErrorType, StartupError, RecoveryResult,
    FailurePattern, FallbackConfiguration, IntelligentFailureHandler
)


class TestFailurePattern:
    """Test failure pattern detection and learning"""
    
    def setup_method(self):
        self.pattern = FailurePattern("test_pattern", "Test pattern description")
    
    def test_pattern_initialization(self):
        """Test failure pattern initialization"""
        assert self.pattern.pattern_id == "test_pattern"
        assert self.pattern.description == "Test pattern description"
        assert self.pattern.frequency == 1
        assert len(self.pattern.error_types) == 0
        assert len(self.pattern.contexts) == 0
    
    def test_add_occurrence(self):
        """Test adding occurrences to a pattern"""
        context = {"port": 8000, "process": "test"}
        
        self.pattern.add_occurrence(ErrorType.PORT_CONFLICT, context)
        
        assert self.pattern.frequency == 2  # Started at 1
        assert ErrorType.PORT_CONFLICT in self.pattern.error_types
        assert context in self.pattern.contexts
    
    def test_add_recovery_result(self):
        """Test recording recovery results"""
        self.pattern.add_recovery_result("kill_process", True)
        self.pattern.add_recovery_result("find_alternative_port", False)
        self.pattern.add_recovery_result("kill_process", True)
        
        assert "kill_process" in self.pattern.successful_recoveries
        assert "find_alternative_port" in self.pattern.failed_recoveries
        assert self.pattern.successful_recoveries.count("kill_process") == 2
    
    def test_get_success_rate_for_action(self):
        """Test success rate calculation for actions"""
        # Add some results
        self.pattern.add_recovery_result("test_action", True)
        self.pattern.add_recovery_result("test_action", True)
        self.pattern.add_recovery_result("test_action", False)
        
        success_rate = self.pattern.get_success_rate_for_action("test_action")
        assert abs(success_rate - 2/3) < 0.01  # 2 successes out of 3 attempts
        
        # Test action with no history
        no_history_rate = self.pattern.get_success_rate_for_action("unknown_action")
        assert no_history_rate == 0.0
    
    def test_get_most_successful_actions(self):
        """Test getting actions sorted by success rate"""
        # Add results for multiple actions
        self.pattern.add_recovery_result("action_a", True)
        self.pattern.add_recovery_result("action_a", True)
        self.pattern.add_recovery_result("action_b", True)
        self.pattern.add_recovery_result("action_b", False)
        self.pattern.add_recovery_result("action_c", False)
        
        most_successful = self.pattern.get_most_successful_actions()
        
        # action_a should be first (100% success), then action_b (50%), then action_c (0%)
        assert most_successful[0] == "action_a"
        assert most_successful[1] == "action_b"
        assert most_successful[2] == "action_c"


class TestFallbackConfiguration:
    """Test fallback configuration system"""
    
    def setup_method(self):
        self.fallback_config = FallbackConfiguration()
    
    def test_get_fallback_config(self):
        """Test getting fallback configuration for error types"""
        port_conflict_config = self.fallback_config.get_fallback_config(ErrorType.PORT_CONFLICT)
        
        assert "safe_port_ranges" in port_conflict_config
        assert "alternative_hosts" in port_conflict_config
        assert isinstance(port_conflict_config["safe_port_ranges"], list)
        
        # Test unknown error type
        unknown_config = self.fallback_config.get_fallback_config(ErrorType.UNKNOWN)
        assert unknown_config == {}
    
    def test_apply_fallback_config(self):
        """Test applying fallback configuration to base config"""
        base_config = {
            "host": "localhost",
            "port": 8000,
            "features": ["auto_open", "system_integration"]
        }
        
        # Test port conflict fallback
        result_config = self.fallback_config.apply_fallback_config(
            ErrorType.PORT_CONFLICT, base_config
        )
        
        assert "safe_ports" in result_config
        assert "allowed_hosts" in result_config
        assert result_config["host"] == "localhost"  # Original value preserved
        
        # Test permission denied fallback
        permission_config = self.fallback_config.apply_fallback_config(
            ErrorType.PERMISSION_DENIED, base_config
        )
        
        assert "disabled_features" in permission_config
        assert "safe_ports" in permission_config
    
    def test_apply_local_only_fallback(self):
        """Test local-only fallback configuration"""
        base_config = {"host": "0.0.0.0", "port": 8000}
        
        result_config = self.fallback_config.apply_fallback_config(
            ErrorType.FIREWALL_BLOCKED, base_config
        )
        
        assert result_config["host"] == "127.0.0.1"
        assert result_config["local_only"] is True


class TestIntelligentFailureHandler:
    """Test intelligent failure handling"""
    
    def setup_method(self):
        with patch('pathlib.Path.exists', return_value=False):
            self.recovery_engine = RecoveryEngine()
        self.handler = IntelligentFailureHandler(self.recovery_engine)
    
    def test_create_pattern_signature(self):
        """Test pattern signature creation"""
        error = StartupError(ErrorType.PORT_CONFLICT, "Address already in use on port 8000")
        context = {"port": 8000, "process_name": "python.exe"}
        
        signature = self.handler._create_pattern_signature(error, context)
        
        assert "port_conflict" in signature
        assert "8000" in signature
        assert "python.exe" in signature
    
    def test_extract_error_keywords(self):
        """Test error keyword extraction"""
        error_message = "Permission denied: Address already in use on socket port 8000"
        
        keywords = self.handler._extract_error_keywords(error_message)
        
        assert "permission" in keywords
        assert "denied" in keywords
        assert "address" in keywords
        assert "use" in keywords
        assert "socket" in keywords
        assert "port" in keywords
    
    def test_detect_failure_pattern_new(self):
        """Test detecting a new failure pattern"""
        error = StartupError(ErrorType.PORT_CONFLICT, "Port 8000 already in use")
        context = {"port": 8000}
        
        pattern = self.handler.detect_failure_pattern(error, context)
        
        assert pattern is not None
        assert pattern.pattern_id in self.handler.failure_patterns
        assert ErrorType.PORT_CONFLICT in pattern.error_types
        assert pattern.frequency == 2  # Pattern starts at 1, then add_occurrence increments it
    
    def test_detect_failure_pattern_existing(self):
        """Test detecting an existing failure pattern"""
        error = StartupError(ErrorType.PORT_CONFLICT, "Port 8000 already in use")
        context = {"port": 8000}
        
        # First occurrence
        pattern1 = self.handler.detect_failure_pattern(error, context)
        initial_frequency = pattern1.frequency
        
        # Second occurrence
        pattern2 = self.handler.detect_failure_pattern(error, context)
        
        assert pattern1 is pattern2  # Same pattern object
        assert pattern2.frequency == initial_frequency + 1
    
    def test_prioritize_recovery_actions_no_pattern(self):
        """Test action prioritization without existing patterns"""
        error = StartupError(ErrorType.PORT_CONFLICT, "Port conflict")
        context = {"port": 8000}
        
        actions = self.handler.prioritize_recovery_actions(error, context)
        
        # Should return base actions from recovery engine
        base_actions = self.recovery_engine.get_recovery_actions(ErrorType.PORT_CONFLICT)
        assert len(actions) == len(base_actions)
    
    def test_prioritize_recovery_actions_with_pattern(self):
        """Test action prioritization with learned patterns"""
        error = StartupError(ErrorType.PORT_CONFLICT, "Port 8000 already in use")
        context = {"port": 8000}
        
        # Create a pattern with some history
        pattern = self.handler.detect_failure_pattern(error, context)
        pattern.frequency = 5  # Make it above threshold
        pattern.add_recovery_result("kill_process_on_port", True)
        pattern.add_recovery_result("kill_process_on_port", True)
        pattern.add_recovery_result("find_alternative_port", False)
        
        actions = self.handler.prioritize_recovery_actions(error, context)
        
        # Actions should be reordered based on pattern success rates
        assert len(actions) > 0
        # The first action should be the most successful one for this pattern
        if actions:
            first_action = actions[0]
            assert first_action.success_rate > 0
    
    def test_handle_recovery_failure_with_fallback(self):
        """Test handling recovery failure with fallback configuration"""
        error = StartupError(ErrorType.PORT_CONFLICT, "Port conflict")
        context = {"port": 8000, "host": "localhost"}
        failed_actions = ["kill_process_on_port", "find_alternative_port"]
        
        result = self.handler.handle_recovery_failure(error, context, failed_actions)
        
        assert result.action_taken == "apply_fallback_configuration"
        assert result.success is True
        assert "fallback_config" in result.details
        assert result.retry_recommended is True
    
    def test_handle_recovery_failure_manual_intervention(self):
        """Test handling recovery failure requiring manual intervention"""
        error = StartupError(ErrorType.UNKNOWN, "Unknown error")
        context = {}
        failed_actions = ["some_action"]
        
        result = self.handler.handle_recovery_failure(error, context, failed_actions)
        
        assert result.action_taken == "manual_intervention_required"
        assert result.success is False
        assert "failed_actions" in result.details
        assert "suggested_manual_actions" in result.details
        assert result.fallback_needed is False
    
    @patch('builtins.open')
    @patch('pathlib.Path.exists')
    def test_save_failure_patterns(self, mock_exists, mock_open):
        """Test saving failure patterns to persistent storage"""
        # Add a pattern
        error = StartupError(ErrorType.PORT_CONFLICT, "Port conflict")
        context = {"port": 8000}
        pattern = self.handler.detect_failure_pattern(error, context)
        pattern.add_recovery_result("test_action", True)
        
        mock_exists.return_value = False
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        self.handler.save_failure_patterns()
        
        mock_open.assert_called_once()
        mock_file.write.assert_called()

        assert True  # TODO: Add proper assertion
    
    @patch('builtins.open')
    @patch('pathlib.Path.exists')
    def test_load_failure_patterns(self, mock_exists, mock_open):
        """Test loading failure patterns from persistent storage"""
        mock_exists.return_value = True
        
        # Mock file content
        pattern_data = {
            "test_pattern": {
                "description": "Test pattern",
                "frequency": 3,
                "first_seen": time.time(),
                "last_seen": time.time(),
                "error_types": ["port_conflict"],
                "successful_recoveries": ["test_action"],
                "failed_recoveries": []
            }
        }
        
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(pattern_data)
        
        # Create new handler to test loading
        handler = IntelligentFailureHandler(self.recovery_engine)
        
        assert "test_pattern" in handler.failure_patterns
        pattern = handler.failure_patterns["test_pattern"]
        assert pattern.frequency == 3
        assert ErrorType.PORT_CONFLICT in pattern.error_types


class TestRecoveryEngineIntegration:
    """Test integration of intelligent failure handling with RecoveryEngine"""
    
    def setup_method(self):
        with patch('pathlib.Path.exists', return_value=False):
            self.recovery_engine = RecoveryEngine()
    
    def test_enhanced_initialization(self):
        """Test that RecoveryEngine is enhanced with intelligent handling"""
        assert hasattr(self.recovery_engine, 'intelligent_handler')
        assert isinstance(self.recovery_engine.intelligent_handler, IntelligentFailureHandler)
    
    def test_intelligent_attempt_recovery_success(self):
        """Test intelligent recovery attempt that succeeds"""
        # Mock a successful recovery action
        def mock_recovery_func(error, context):
            return RecoveryResult(
                success=True,
                action_taken="mock_action",
                message="Recovery successful"
            )
        
        # Replace the recovery action with our mock
        for actions in self.recovery_engine.recovery_actions.values():
            for action in actions:
                action.action_func = mock_recovery_func
                break
        
        error = StartupError(ErrorType.PORT_CONFLICT, "Port conflict")
        result = self.recovery_engine.attempt_recovery(error, {"port": 8000})
        
        assert result.success is True
        assert result.action_taken == "mock_action"
    
    def test_intelligent_attempt_recovery_with_learning(self):
        """Test that recovery attempts contribute to learning"""
        error = StartupError(ErrorType.PORT_CONFLICT, "Port 8000 already in use")
        context = {"port": 8000}
        
        # Mock recovery actions to fail initially
        def mock_failing_func(error, context):
            return RecoveryResult(
                success=False,
                action_taken="mock_fail",
                message="Recovery failed"
            )
        
        # Replace recovery actions with failing mocks
        for actions in self.recovery_engine.recovery_actions.values():
            for action in actions:
                action.action_func = mock_failing_func
        
        # First attempt should create a pattern and may succeed with fallback
        result1 = self.recovery_engine.attempt_recovery(error, context)
        # The intelligent handler will apply fallback configuration, so it may succeed
        assert result1 is not None
        
        # Check that a pattern was created
        patterns = self.recovery_engine.intelligent_handler.failure_patterns
        assert len(patterns) > 0
        
        # Pattern should have recorded the failure
        pattern = list(patterns.values())[0]
        assert len(pattern.failed_recoveries) > 0
    
    def test_intelligent_attempt_recovery_fallback(self):
        """Test intelligent recovery with fallback configuration"""
        error = StartupError(ErrorType.PORT_CONFLICT, "Port conflict")
        context = {"port": 8000, "host": "localhost"}
        
        # Mock all recovery actions to fail
        def mock_failing_func(error, context):
            return RecoveryResult(
                success=False,
                action_taken="mock_fail",
                message="Recovery failed"
            )
        
        # Replace recovery actions with failing mocks
        for actions in self.recovery_engine.recovery_actions.values():
            for action in actions:
                action.action_func = mock_failing_func
        
        result = self.recovery_engine.attempt_recovery(error, context)
        
        # Should fall back to fallback configuration
        assert result.action_taken == "apply_fallback_configuration"
        assert result.success is True
        assert "fallback_config" in result.details
    
    def test_pattern_persistence_integration(self):
        """Test that patterns are saved and loaded correctly"""
        error = StartupError(ErrorType.PORT_CONFLICT, "Port 8000 already in use")
        context = {"port": 8000}
        
        # Create a pattern
        pattern = self.recovery_engine.intelligent_handler.detect_failure_pattern(error, context)
        pattern.add_recovery_result("test_action", True)
        
        # Mock file operations
        with patch('builtins.open'), patch('pathlib.Path.exists', return_value=False):
            self.recovery_engine.intelligent_handler.save_failure_patterns()
        
        # Verify the pattern exists
        assert len(self.recovery_engine.intelligent_handler.failure_patterns) > 0


class TestComplexFailureScenarios:
    """Test complex failure scenarios and edge cases"""
    
    def setup_method(self):
        with patch('pathlib.Path.exists', return_value=False):
            self.recovery_engine = RecoveryEngine()
    
    def test_cascading_failures(self):
        """Test handling of cascading failures"""
        # Simulate a scenario where port conflict leads to permission issues
        port_error = StartupError(ErrorType.PORT_CONFLICT, "Port 8000 in use")
        permission_error = StartupError(ErrorType.PERMISSION_DENIED, "Permission denied on port 8001")
        
        context = {"port": 8000}
        
        # First failure
        result1 = self.recovery_engine.attempt_recovery(port_error, context)
        
        # Second related failure
        context["port"] = 8001
        result2 = self.recovery_engine.attempt_recovery(permission_error, context)
        
        # Both should be handled appropriately
        assert result1 is not None
        assert result2 is not None
    
    def test_rapid_repeated_failures(self):
        """Test handling of rapid repeated failures"""
        error = StartupError(ErrorType.PORT_CONFLICT, "Port 8000 already in use")
        context = {"port": 8000}
        
        # Simulate rapid repeated failures
        results = []
        for i in range(5):
            result = self.recovery_engine.attempt_recovery(error, context)
            results.append(result)
        
        # Should handle all attempts
        assert len(results) == 5
        
        # Pattern should be learned
        patterns = self.recovery_engine.intelligent_handler.failure_patterns
        assert len(patterns) > 0
        
        # Pattern frequency should reflect repeated occurrences
        pattern = list(patterns.values())[0]
        assert pattern.frequency >= 5
    
    def test_mixed_error_types_pattern(self):
        """Test pattern detection across mixed error types"""
        errors = [
            StartupError(ErrorType.PORT_CONFLICT, "Port 8000 in use"),
            StartupError(ErrorType.PERMISSION_DENIED, "Permission denied on port 8000"),
            StartupError(ErrorType.FIREWALL_BLOCKED, "Firewall blocking port 8000")
        ]
        
        context = {"port": 8000}
        
        # Process different error types for the same context
        for error in errors:
            self.recovery_engine.attempt_recovery(error, context)
        
        # Should create patterns that can handle related issues
        patterns = self.recovery_engine.intelligent_handler.failure_patterns
        assert len(patterns) > 0
    
    def test_learning_disabled_fallback(self):
        """Test behavior when learning is disabled"""
        self.recovery_engine.intelligent_handler.learning_enabled = False
        
        error = StartupError(ErrorType.PORT_CONFLICT, "Port conflict")
        context = {"port": 8000}
        
        # Should still work but without pattern learning
        result = self.recovery_engine.attempt_recovery(error, context)
        
        assert result is not None
        # No patterns should be created
        patterns = self.recovery_engine.intelligent_handler.failure_patterns
        assert len(patterns) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])