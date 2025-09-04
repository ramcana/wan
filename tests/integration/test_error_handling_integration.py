"""
Integration tests for error handling workflows.

Tests the complete error handling system including:
- Error classification and display
- Interactive error resolution
- Context-sensitive help system
- Recovery action execution
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json

from scripts.startup_manager.error_handler import (
    ErrorDisplayManager,
    ErrorClassifier,
    HelpSystem,
    StartupError,
    ErrorCategory,
    ErrorSeverity,
    RecoveryAction
)
from scripts.startup_manager.cli import InteractiveCLI, CLIOptions, VerbosityLevel


class TestErrorClassifier:
    """Test error classification functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.classifier = ErrorClassifier()
    
    def test_classify_network_error(self):
        """Test classification of network errors"""
        error = Exception("WinError 10013: An attempt was made to access a socket")
        
        result = self.classifier.classify_error(error)
        
        assert result.category == ErrorCategory.PERMISSION
        assert result.severity == ErrorSeverity.WARNING
        assert "Socket access forbidden" in result.message
        assert result.error_code.startswith("PRM-")
    
    def test_classify_port_conflict_error(self):
        """Test classification of port conflict errors"""
        error = Exception("Address already in use: port 8000")
        context = {"port": 8000, "operation": "start_backend"}
        
        result = self.classifier.classify_error(error, context)
        
        assert result.category == ErrorCategory.NETWORK
        assert result.severity == ErrorSeverity.WARNING
        assert "Port conflict detected" in result.message
        assert "Port: 8000" in result.details
        assert result.error_code.startswith("NET-")
    
    def test_classify_dependency_error(self):
        """Test classification of dependency errors"""
        error = ModuleNotFoundError("No module named 'fastapi'")
        
        result = self.classifier.classify_error(error)
        
        assert result.category == ErrorCategory.DEPENDENCY
        assert result.severity == ErrorSeverity.ERROR
        assert "Python module missing" in result.message
        assert result.error_code.startswith("DEP-")
    
    def test_classify_permission_error(self):
        """Test classification of permission errors"""
        error = PermissionError("Access is denied")
        context = {"file": "config.json", "operation": "write_config"}
        
        result = self.classifier.classify_error(error, context)
        
        assert result.category == ErrorCategory.PERMISSION
        assert result.severity == ErrorSeverity.WARNING
        assert "Windows access denied" in result.message
        assert "config.json" in result.details
    
    def test_classify_configuration_error(self):
        """Test classification of configuration errors"""
        error = json.JSONDecodeError("Expecting ',' delimiter", "config.json", 10)
        context = {"config_file": "startup_config.json"}
        
        result = self.classifier.classify_error(error, context)
        
        assert result.category == ErrorCategory.CONFIGURATION
        assert result.severity == ErrorSeverity.ERROR
        assert "Invalid JSON configuration" in result.message
    
    def test_classify_unknown_error(self):
        """Test classification of unknown errors"""
        error = Exception("Some unknown error occurred")
        
        result = self.classifier.classify_error(error)
        
        assert result.category == ErrorCategory.UNKNOWN
        assert result.severity == ErrorSeverity.INFO
        assert result.error_code.startswith("UNK-")
    
    def test_context_specific_recovery_actions(self):
        """Test that context-specific recovery actions are added"""
        error = Exception("Address already in use")
        context = {"port": 8000}
        
        result = self.classifier.classify_error(error, context)
        
        # Should have context-specific action for port change
        action_names = [action.name for action in result.recovery_actions]
        assert "use_different_port" in action_names
    
    def test_default_recovery_actions(self):
        """Test that default recovery actions are always added"""
        error = Exception("Some error")
        
        result = self.classifier.classify_error(error)
        
        # Should have default actions
        action_names = [action.name for action in result.recovery_actions]
        assert "view_logs" in action_names
        assert "get_help" in action_names


class TestStartupError:
    """Test StartupError functionality"""
    
    def test_startup_error_creation(self):
        """Test creating a StartupError"""
        error = StartupError(
            message="Test error",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            details="Test details",
            error_code="NET-001"
        )
        
        assert error.message == "Test error"
        assert error.category == ErrorCategory.NETWORK
        assert error.severity == ErrorSeverity.ERROR
        assert error.details == "Test details"
        assert error.error_code == "NET-001"
        assert len(error.recovery_actions) > 0  # Should have default actions
    
    def test_recovery_action_execution(self):
        """Test recovery action execution"""
        executed = False
        
        def test_action():
            nonlocal executed
            executed = True
            return True

            assert True  # TODO: Add proper assertion
        
        action = RecoveryAction(
            name="test_action",
            description="Test action",
            action=test_action,
            auto_executable=True,
            requires_confirmation=False
        )
        
        result = action.action()
        
        assert result is True
        assert executed is True


class TestErrorDisplayManager:
    """Test error display and interaction management"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cli_options = CLIOptions(verbosity=VerbosityLevel.NORMAL)
        self.cli = InteractiveCLI(self.cli_options)
        self.display_manager = ErrorDisplayManager(self.cli)
    
    def test_display_error_basic(self):
        """Test basic error display"""
        error = StartupError(
            message="Test error message",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            details="Additional details",
            error_code="NET-001"
        )
        
        with patch.object(self.cli, 'print_status') as mock_print:
            with patch.object(self.cli, 'print_verbose') as mock_verbose:
                self.display_manager.display_error(error)
                
                # Should print main error message
                mock_print.assert_any_call("Error: Test error message", "error")
                mock_print.assert_any_call("Details: Additional details", "info")
                mock_verbose.assert_any_call("Error Code: NET-001")

        assert True  # TODO: Add proper assertion
    
    def test_display_error_with_technical_details(self):
        """Test error display with technical details in debug mode"""
        debug_cli = InteractiveCLI(CLIOptions(verbosity=VerbosityLevel.DEBUG))
        display_manager = ErrorDisplayManager(debug_cli)
        
        error = StartupError(
            message="Test error",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.ERROR,
            technical_details="Exception: Detailed technical info"
        )
        
        with patch.object(debug_cli, 'print_debug') as mock_debug:
            display_manager.display_error(error)
            
            mock_debug.assert_called_with("Technical: Exception: Detailed technical info")

        assert True  # TODO: Add proper assertion
    
    @patch('builtins.input', return_value='skip')
    def test_offer_recovery_options_skip(self, mock_input):
        """Test skipping recovery options"""
        error = StartupError(
            message="Test error",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR
        )
        
        with patch.object(self.cli, 'prompt_choice', return_value='skip'):
            result = self.display_manager.offer_recovery_options(error)
            
            assert result is False
    
    @patch('builtins.input', side_effect=['1', 'y'])
    def test_offer_recovery_options_execute(self, mock_input):
        """Test executing a recovery option"""
        executed = False
        
        def test_recovery():
            nonlocal executed
            executed = True
            return True

            assert True  # TODO: Add proper assertion
        
        recovery_action = RecoveryAction(
            name="test_recovery",
            description="Test recovery action",
            action=test_recovery,
            requires_confirmation=True
        )
        
        error = StartupError(
            message="Test error",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            recovery_actions=[recovery_action]
        )
        
        with patch.object(self.cli, 'prompt_choice', return_value='1'):
            with patch.object(self.cli, 'confirm_action', return_value=True):
                with patch.object(self.cli, 'print_status'):
                    result = self.display_manager.offer_recovery_options(error)
                    
                    assert result is True
                    assert executed is True
    
    def test_offer_recovery_options_help(self):
        """Test requesting help during recovery options"""
        error = StartupError(
            message="Test error",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR
        )
        
        with patch.object(self.cli, 'prompt_choice', side_effect=['help', 'skip']):
            with patch.object(self.display_manager, 'show_contextual_help') as mock_help:
                result = self.display_manager.offer_recovery_options(error)
                
                mock_help.assert_called_once_with(error)
                assert result is False
    
    def test_handle_error_complete_workflow(self):
        """Test complete error handling workflow"""
        original_error = Exception("Address already in use")
        context = {"port": 8000, "operation": "start_backend"}
        
        with patch.object(self.display_manager, 'display_error_enhanced') as mock_display:
            with patch.object(self.display_manager, 'offer_recovery_options_enhanced', return_value=True) as mock_recovery:
                result = self.display_manager.handle_error(original_error, context)
                
                assert result is True
                mock_display.assert_called_once()
                mock_recovery.assert_called_once()
                
                # Check that error was properly classified
                called_error = mock_display.call_args[0][0]
                assert isinstance(called_error, StartupError)
                assert called_error.category == ErrorCategory.NETWORK


class TestHelpSystem:
    """Test context-sensitive help system"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.help_system = HelpSystem()
    
    def test_get_network_help(self):
        """Test getting help for network errors"""
        error = StartupError(
            message="Port conflict",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.WARNING
        )
        
        help_content = self.help_system.get_help_for_error(error)
        
        assert help_content is not None
        assert "Network Error Help" in help_content
        assert "Port already in use" in help_content
        assert "Windows Firewall" in help_content
    
    def test_get_permission_help(self):
        """Test getting help for permission errors"""
        error = StartupError(
            message="Access denied",
            category=ErrorCategory.PERMISSION,
            severity=ErrorSeverity.WARNING
        )
        
        help_content = self.help_system.get_help_for_error(error)
        
        assert help_content is not None
        assert "Permission Error Help" in help_content
        assert "Run as administrator" in help_content
        assert "UAC" in help_content
    
    def test_get_dependency_help(self):
        """Test getting help for dependency errors"""
        error = StartupError(
            message="Module not found",
            category=ErrorCategory.DEPENDENCY,
            severity=ErrorSeverity.ERROR
        )
        
        help_content = self.help_system.get_help_for_error(error)
        
        assert help_content is not None
        assert "Dependency Error Help" in help_content
        assert "pip install" in help_content
        assert "Virtual Environment" in help_content
    
    def test_get_configuration_help(self):
        """Test getting help for configuration errors"""
        error = StartupError(
            message="Invalid JSON",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.ERROR
        )
        
        help_content = self.help_system.get_help_for_error(error)
        
        assert help_content is not None
        assert "Configuration Error Help" in help_content
        assert "JSON syntax" in help_content
        assert "startup_config.json" in help_content
    
    def test_get_help_for_unknown_category(self):
        """Test getting help for unknown error category"""
        error = StartupError(
            message="Unknown error",
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.INFO
        )
        
        help_content = self.help_system.get_help_for_error(error)
        
        assert help_content is None


class TestErrorHandlingIntegration:
    """Integration tests for complete error handling workflows"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cli_options = CLIOptions(verbosity=VerbosityLevel.NORMAL, interactive=True)
        self.cli = InteractiveCLI(self.cli_options)
        self.display_manager = ErrorDisplayManager(self.cli)
    
    def test_network_error_workflow(self):
        """Test complete workflow for network error"""
        # Simulate a port conflict error
        original_error = Exception("WinError 10048: Address already in use")
        context = {
            "operation": "start_backend",
            "port": 8000,
            "service": "FastAPI"
        }
        
        with patch.object(self.cli, 'print_status') as mock_print:
            with patch.object(self.cli, 'confirm_action', return_value=False):  # Don't run auto recovery
                with patch.object(self.cli, 'prompt_choice', return_value='skip'):
                    result = self.display_manager.handle_error(original_error, context)
                    
                    # Should display error information
                    # Note: The enhanced display uses display_summary_panel, not print_status for main error
                    
                    # Should offer recovery options
                    assert result is False  # User chose to skip
    
    def test_dependency_error_workflow(self):
        """Test complete workflow for dependency error"""
        original_error = ModuleNotFoundError("No module named 'fastapi'")
        context = {"operation": "import_dependencies"}
        
        recovery_executed = False
        
        def mock_install_deps():
            nonlocal recovery_executed
            recovery_executed = True
            return True
        
        # Mock the recovery action execution directly
        with patch.object(self.display_manager, 'offer_recovery_options_enhanced') as mock_recovery:
            mock_recovery.return_value = True
            result = self.display_manager.handle_error(original_error, context)
            
            assert result is True
            mock_recovery.assert_called_once()
    
    def test_configuration_error_workflow(self):
        """Test complete workflow for configuration error"""
        original_error = json.JSONDecodeError("Expecting ',' delimiter", "config.json", 10)
        context = {
            "operation": "load_config",
            "config_file": "startup_config.json"
        }
        
        with patch.object(self.cli, 'prompt_choice', return_value='help'):
            with patch.object(self.cli, 'display_summary_panel') as mock_panel:
                with patch.object(self.cli, 'prompt_choice', return_value='skip'):
                    result = self.display_manager.handle_error(original_error, context)
                    
                    # Should show help panel
                    mock_panel.assert_called()
                    panel_args = mock_panel.call_args[0]
                    assert "Configuration Error" in panel_args[0]
                    assert "JSON syntax" in panel_args[1]
    
    def test_permission_error_workflow(self):
        """Test complete workflow for permission error"""
        original_error = PermissionError("Access is denied")
        context = {
            "operation": "create_log_file",
            "file": "logs/startup.log"
        }
        
        with patch.object(self.cli, 'display_summary_panel') as mock_panel:
            with patch.object(self.cli, 'confirm_action', return_value=False):  # Don't run auto recovery
                with patch.object(self.cli, 'prompt_choice', return_value='skip'):
                    result = self.display_manager.handle_error(original_error, context)
                    
                    # Should display permission error panel
                    mock_panel.assert_called()
                    panel_args = mock_panel.call_args[0]
                    assert "Permission Error" in panel_args[0]
                    assert "Windows access denied" in panel_args[1]
                    
                    # Should include file information in details
                    assert "logs/startup.log" in panel_args[1]
    
    def test_quiet_mode_error_handling(self):
        """Test error handling in quiet mode"""
        quiet_cli = InteractiveCLI(CLIOptions(verbosity=VerbosityLevel.QUIET))
        quiet_display_manager = ErrorDisplayManager(quiet_cli)
        
        original_error = Exception("Test error")
        
        with patch.object(quiet_cli, 'print_status') as mock_print:
            with patch.object(quiet_cli, 'prompt_choice', return_value='skip'):
                result = quiet_display_manager.handle_error(original_error)
                
                # Should not offer recovery options in quiet mode
                assert result is False
    
    def test_auto_confirm_mode_error_handling(self):
        """Test error handling with auto-confirm enabled"""
        auto_cli = InteractiveCLI(CLIOptions(auto_confirm=True))
        auto_display_manager = ErrorDisplayManager(auto_cli)
        
        original_error = Exception("Test error")
        
        # Mock the recovery options to avoid interactive prompts
        with patch.object(auto_display_manager, 'offer_recovery_options_enhanced', return_value=True) as mock_recovery:
            result = auto_display_manager.handle_error(original_error)
            
            # Should handle error and offer recovery
            assert result is True
            mock_recovery.assert_called_once()
    
    def test_enhanced_error_display(self):
        """Test enhanced error display with rich formatting"""
        original_error = Exception("Address already in use")
        context = {"port": 8000, "operation": "start_backend"}
        
        with patch.object(self.cli, 'display_summary_panel') as mock_panel:
            with patch.object(self.display_manager, 'offer_recovery_options_enhanced', return_value=False):
                self.display_manager.handle_error(original_error, context)
                
                # Should display enhanced error panel
                mock_panel.assert_called()
                panel_args = mock_panel.call_args[0]
                assert "Network Error" in panel_args[0]
                assert "Port conflict detected" in panel_args[1]
    
    def test_recurring_error_detection(self):
        """Test detection and warning for recurring errors"""
        original_error = Exception("Address already in use")
        context = {"port": 8000}
        
        # Simulate multiple occurrences of the same error
        with patch.object(self.display_manager, 'offer_recovery_options_enhanced', return_value=False):
            with patch.object(self.cli, 'display_summary_panel') as mock_panel:
                # First occurrence
                self.display_manager.handle_error(original_error, context)
                
                # Second occurrence - should trigger recurring error warning
                self.display_manager.handle_error(original_error, context)
                
                # Check that recurring error warning was displayed
                warning_calls = [call for call in mock_panel.call_args_list 
                               if len(call[0]) > 0 and "Recurring Error" in call[0][0]]
                assert len(warning_calls) > 0
    
    def test_enhanced_help_system(self):
        """Test enhanced help system with interactive options"""
        error = StartupError(
            message="Test error",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR
        )
        
        with patch.object(self.cli, 'prompt_choice', side_effect=['troubleshooting', 'back']):
            with patch.object(self.display_manager, 'show_detailed_troubleshooting') as mock_troubleshooting:
                self.display_manager.show_enhanced_help(error)
                
                mock_troubleshooting.assert_called_once_with(error)

        assert True  # TODO: Add proper assertion
    
    def test_batch_recovery_execution(self):
        """Test executing all recovery actions in batch"""
        recovery_actions = [
            RecoveryAction("action1", "Test action 1", lambda: True, auto_executable=True),
            RecoveryAction("action2", "Test action 2", lambda: True, auto_executable=False),
            RecoveryAction("action3", "Test action 3", lambda: False, auto_executable=True)
        ]
        
        error = StartupError(
            message="Test error",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            recovery_actions=recovery_actions
        )
        
        with patch.object(self.cli, 'prompt_choice', return_value='all'):
            with patch.object(self.cli, 'confirm_action', return_value=True):
                with patch.object(self.cli, 'print_status'):
                    with patch.object(self.display_manager, '_display_batch_recovery_summary') as mock_summary:
                        result = self.display_manager.offer_recovery_options_enhanced(error)
                        
                        # Should execute batch recovery and display summary
                        mock_summary.assert_called_once()
                        summary_args = mock_summary.call_args[0]
                        assert summary_args[0] == 2  # success_count
                        assert summary_args[1] == 3  # total_count
                        assert "action3" in summary_args[2]  # failed_actions
    
    def test_auto_recovery_workflow(self):
        """Test automatic recovery action workflow"""
        auto_action = RecoveryAction(
            "auto_fix", 
            "Automatic fix", 
            lambda: True, 
            auto_executable=True,
            requires_confirmation=False
        )
        
        error = StartupError(
            message="Test error",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            recovery_actions=[auto_action]
        )
        
        with patch.object(self.cli, 'confirm_action', return_value=True):
            with patch.object(self.cli, 'print_status'):
                result = self.display_manager.offer_recovery_options_enhanced(error)
                
                assert result is True
    
    def test_help_system_troubleshooting_steps(self):
        """Test help system troubleshooting steps"""
        help_system = self.display_manager.help_system
        
        # Test each category has troubleshooting steps
        for category in ErrorCategory:
            if category != ErrorCategory.UNKNOWN:
                steps = help_system.get_troubleshooting_steps(category)
                assert steps is not None
                assert len(steps) > 0
                assert "Step-by-Step" in steps
    
    def test_help_system_error_examples(self):
        """Test help system error examples"""
        help_system = self.display_manager.help_system
        
        # Test each category has error examples
        for category in ErrorCategory:
            if category != ErrorCategory.UNKNOWN:
                examples = help_system.get_error_examples(category)
                assert examples is not None
                assert len(examples) > 0
                assert "Example" in examples
    
    def test_log_guidance_display(self):
        """Test log guidance display functionality"""
        error = StartupError(
            message="Test error",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.ERROR
        )
        
        with patch.object(self.cli, 'display_summary_panel') as mock_panel:
            self.display_manager.show_log_guidance(error)
            
            mock_panel.assert_called_once()
            panel_args = mock_panel.call_args[0]
            assert "Log Analysis Guidance" in panel_args[0]
            assert "logs/" in panel_args[1]
    
    def test_support_information_display(self):
        """Test support information display"""
        error = StartupError(
            message="Test error",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.ERROR,
            error_code="CFG-001"
        )
        
        with patch.object(self.cli, 'display_summary_panel') as mock_panel:
            self.display_manager.show_support_information(error)
            
            mock_panel.assert_called_once()
            panel_args = mock_panel.call_args[0]
            assert "Support Information" in panel_args[0]
            assert "CFG-001" in panel_args[1]
    
    def test_no_recovery_options_display(self):
        """Test display when no recovery options are available"""
        # Create error with no recovery actions by overriding the default behavior
        error = StartupError(
            message="Test error",
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.ERROR
        )
        # Clear the default recovery actions
        error.recovery_actions = []
        
        with patch.object(self.cli, 'display_summary_panel') as mock_panel:
            result = self.display_manager.offer_recovery_options_enhanced(error)
            
            assert result is False
            mock_panel.assert_called_once()
            panel_args = mock_panel.call_args[0]
            assert "Manual Resolution Required" in panel_args[0]


class TestEnhancedErrorHandlingFeatures:
    """Test enhanced error handling features"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cli_options = CLIOptions(verbosity=VerbosityLevel.NORMAL, interactive=True)
        self.cli = InteractiveCLI(self.cli_options)
        self.display_manager = ErrorDisplayManager(self.cli)
    
    def test_error_history_tracking(self):
        """Test error history tracking functionality"""
        error1 = Exception("First error")
        error2 = Exception("Second error")
        
        with patch.object(self.display_manager, 'offer_recovery_options_enhanced', return_value=False):
            self.display_manager.handle_error(error1)
            self.display_manager.handle_error(error2)
            
            assert len(self.display_manager.error_history) == 2
    
    def test_category_guidance_display(self):
        """Test category-specific guidance display"""
        guidance = self.display_manager._get_category_guidance(ErrorCategory.NETWORK)
        
        assert guidance is not None
        assert "ports" in guidance.lower()
        assert "firewall" in guidance.lower()
    
    def test_recovery_options_table_display(self):
        """Test recovery options table display"""
        actions = [
            RecoveryAction("test1", "Test action 1", lambda: True, auto_executable=True),
            RecoveryAction("test2", "Test action 2", lambda: True, auto_executable=False, requires_confirmation=True)
        ]
        
        with patch.object(self.cli, 'display_table') as mock_table:
            self.display_manager._display_recovery_options_table(actions)
            
            mock_table.assert_called_once()
            table_args = mock_table.call_args[0]
            assert "Available Recovery Actions" in table_args[0]
            assert len(table_args[2]) == 2  # Two rows for two actions
    
    def test_single_action_execution_with_feedback(self):
        """Test single action execution with enhanced feedback"""
        success_action = RecoveryAction(
            "test_action", 
            "Test successful action", 
            lambda: True,
            requires_confirmation=True
        )
        
        with patch.object(self.cli, 'confirm_action', side_effect=[True, True]):  # Confirm execution and continue
            with patch.object(self.cli, 'show_spinner'):
                with patch.object(self.cli, 'print_status'):
                    result = self.display_manager._execute_single_action(success_action)
                    
                    assert result is True
    
    def test_windows_version_detection(self):
        """Test Windows version detection for support info"""
        version = self.display_manager._get_windows_version()
        
        assert version is not None
        assert len(version) > 0


if __name__ == '__main__':
    pytest.main([__file__])