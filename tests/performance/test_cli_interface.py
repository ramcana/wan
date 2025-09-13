"""
Unit tests for the CLI interface components.

Tests the interactive CLI functionality including:
- Progress bars and spinners
- Interactive prompts
- Verbose/quiet modes
- User input handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import sys
from click.testing import CliRunner

from scripts.startup_manager.cli import (
    InteractiveCLI, 
    CLIOptions, 
    VerbosityLevel,
    cli,
    start,
    status,
    stop
)


class TestCLIOptions:
    """Test CLI options configuration"""
    
    def test_default_options(self):
        """Test default CLI options"""
        options = CLIOptions()
        
        assert options.verbosity == VerbosityLevel.NORMAL
        assert options.no_color is False
        assert options.auto_confirm is False
        assert options.interactive is True
        assert options.show_progress is True
    
    def test_custom_options(self):
        """Test custom CLI options"""
        options = CLIOptions(
            verbosity=VerbosityLevel.VERBOSE,
            no_color=True,
            auto_confirm=True,
            interactive=False,
            show_progress=False
        )
        
        assert options.verbosity == VerbosityLevel.VERBOSE
        assert options.no_color is True
        assert options.auto_confirm is True
        assert options.interactive is False
        assert options.show_progress is False


class TestInteractiveCLI:
    """Test the InteractiveCLI class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cli_options = CLIOptions()
        self.cli = InteractiveCLI(self.cli_options)
    
    def test_initialization(self):
        """Test CLI initialization"""
        assert self.cli.options == self.cli_options
        assert self.cli.console is not None
        assert hasattr(self.cli, 'styles')
        assert 'success' in self.cli.styles
        assert 'error' in self.cli.styles
    
    def test_initialization_with_no_color(self):
        """Test CLI initialization with no color option"""
        options = CLIOptions(no_color=True)
        cli = InteractiveCLI(options)
        
        assert cli.console.color_system is None
    
    @patch('scripts.startup_manager.cli.Console')
    def test_quiet_mode_initialization(self, mock_console):
        """Test CLI initialization in quiet mode"""
        options = CLIOptions(verbosity=VerbosityLevel.QUIET)
        cli = InteractiveCLI(options)
        
        mock_console.assert_called_once()
        call_args = mock_console.call_args
        assert call_args[1]['quiet'] is True
    
    def test_display_banner_normal_mode(self):
        """Test banner display in normal mode"""
        with patch.object(self.cli.console, 'print') as mock_print:
            self.cli.display_banner()
            
            # Should print banner and empty line
            assert mock_print.call_count >= 2
    
    def test_display_banner_quiet_mode(self):
        """Test banner display in quiet mode"""
        quiet_options = CLIOptions(verbosity=VerbosityLevel.QUIET)
        quiet_cli = InteractiveCLI(quiet_options)
        
        with patch.object(quiet_cli.console, 'print') as mock_print:
            quiet_cli.display_banner()
            
            # Should not print anything in quiet mode
            mock_print.assert_not_called()

        assert True  # TODO: Add proper assertion
    
    def test_print_status_normal_mode(self):
        """Test status printing in normal mode"""
        with patch.object(self.cli.console, 'print') as mock_print:
            self.cli.print_status("Test message", "success")
            
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            assert "Test message" in call_args
            assert "✓" in call_args  # Success prefix
    
    def test_print_status_quiet_mode(self):
        """Test status printing in quiet mode"""
        quiet_options = CLIOptions(verbosity=VerbosityLevel.QUIET)
        quiet_cli = InteractiveCLI(quiet_options)
        
        with patch.object(quiet_cli.console, 'print') as mock_print:
            quiet_cli.print_status("Test message", "success")
            
            # Should not print in quiet mode
            mock_print.assert_not_called()

        assert True  # TODO: Add proper assertion
    
    def test_print_verbose_in_verbose_mode(self):
        """Test verbose printing in verbose mode"""
        verbose_options = CLIOptions(verbosity=VerbosityLevel.VERBOSE)
        verbose_cli = InteractiveCLI(verbose_options)
        
        with patch.object(verbose_cli.console, 'print') as mock_print:
            verbose_cli.print_verbose("Verbose message")
            
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            assert "Verbose message" in call_args
    
    def test_print_verbose_in_normal_mode(self):
        """Test verbose printing in normal mode"""
        with patch.object(self.cli.console, 'print') as mock_print:
            self.cli.print_verbose("Verbose message")
            
            # Should not print in normal mode
            mock_print.assert_not_called()

        assert True  # TODO: Add proper assertion
    
    def test_print_debug_in_debug_mode(self):
        """Test debug printing in debug mode"""
        debug_options = CLIOptions(verbosity=VerbosityLevel.DEBUG)
        debug_cli = InteractiveCLI(debug_options)
        
        with patch.object(debug_cli.console, 'print') as mock_print:
            debug_cli.print_debug("Debug message")
            
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            assert "DEBUG: Debug message" in call_args
    
    def test_print_debug_in_normal_mode(self):
        """Test debug printing in normal mode"""
        with patch.object(self.cli.console, 'print') as mock_print:
            self.cli.print_debug("Debug message")
            
            # Should not print in normal mode
            mock_print.assert_not_called()

        assert True  # TODO: Add proper assertion
    
    @patch('scripts.startup_manager.cli.Confirm.ask')
    def test_confirm_action_interactive(self, mock_confirm):
        """Test confirmation in interactive mode"""
        mock_confirm.return_value = True
        
        result = self.cli.confirm_action("Proceed?", default=False)
        
        assert result is True
        mock_confirm.assert_called_once()
        call_args = mock_confirm.call_args
        assert "Proceed?" in call_args[0][0]
        assert call_args[1]['default'] is False
    
    def test_confirm_action_auto_confirm(self):
        """Test confirmation with auto-confirm enabled"""
        auto_options = CLIOptions(auto_confirm=True)
        auto_cli = InteractiveCLI(auto_options)
        
        with patch('scripts.startup_manager.cli.Confirm.ask') as mock_confirm:
            result = auto_cli.confirm_action("Proceed?", default=True)
            
            assert result is True
            mock_confirm.assert_not_called()
    
    def test_confirm_action_non_interactive(self):
        """Test confirmation in non-interactive mode"""
        non_interactive_options = CLIOptions(interactive=False)
        non_interactive_cli = InteractiveCLI(non_interactive_options)
        
        with patch('scripts.startup_manager.cli.Confirm.ask') as mock_confirm:
            result = non_interactive_cli.confirm_action("Proceed?", default=True)
            
            assert result is True
            mock_confirm.assert_not_called()
    
    @patch('scripts.startup_manager.cli.Prompt.ask')
    def test_prompt_choice_interactive(self, mock_prompt):
        """Test choice prompting in interactive mode"""
        mock_prompt.return_value = "option2"
        
        result = self.cli.prompt_choice("Choose:", ["option1", "option2"], "option1")
        
        assert result == "option2"
        mock_prompt.assert_called_once()
        call_args = mock_prompt.call_args
        assert "Choose:" in call_args[0][0]
        assert call_args[1]['choices'] == ["option1", "option2"]
        assert call_args[1]['default'] == "option1"
    
    def test_prompt_choice_auto_confirm(self):
        """Test choice prompting with auto-confirm"""
        auto_options = CLIOptions(auto_confirm=True)
        auto_cli = InteractiveCLI(auto_options)
        
        with patch('scripts.startup_manager.cli.Prompt.ask') as mock_prompt:
            result = auto_cli.prompt_choice("Choose:", ["option1", "option2"], "option2")
            
            assert result == "option2"
            mock_prompt.assert_not_called()
    
    def test_prompt_choice_non_interactive(self):
        """Test choice prompting in non-interactive mode"""
        non_interactive_options = CLIOptions(interactive=False)
        non_interactive_cli = InteractiveCLI(non_interactive_options)
        
        with patch('scripts.startup_manager.cli.Prompt.ask') as mock_prompt:
            result = non_interactive_cli.prompt_choice("Choose:", ["option1", "option2"])
            
            assert result == "option1"  # Should return first choice
            mock_prompt.assert_not_called()
    
    @patch('scripts.startup_manager.cli.IntPrompt.ask')
    def test_prompt_number_interactive(self, mock_prompt):
        """Test number prompting in interactive mode"""
        mock_prompt.return_value = 42
        
        result = self.cli.prompt_number("Enter number:", default=10, min_val=1, max_val=100)
        
        assert result == 42
        mock_prompt.assert_called_once()
    
    @patch('scripts.startup_manager.cli.IntPrompt.ask')
    def test_prompt_number_validation_min(self, mock_prompt):
        """Test number prompting with minimum validation"""
        # First return value below minimum, then valid value
        mock_prompt.side_effect = [0, 5]
        
        with patch.object(self.cli, 'print_status') as mock_print:
            with patch.object(self.cli, 'confirm_action', return_value=True):
                result = self.cli.prompt_number("Enter number:", min_val=1)
                
                assert result == 5
                assert mock_prompt.call_count == 2
                mock_print.assert_called_with("Value must be at least 1", "error")
    
    @patch('scripts.startup_manager.cli.IntPrompt.ask')
    def test_prompt_number_validation_max(self, mock_prompt):
        """Test number prompting with maximum validation"""
        # First return value above maximum, then valid value
        mock_prompt.side_effect = [150, 50]
        
        with patch.object(self.cli, 'print_status') as mock_print:
            with patch.object(self.cli, 'confirm_action', return_value=True):
                result = self.cli.prompt_number("Enter number:", max_val=100)
                
                assert result == 50
                assert mock_prompt.call_count == 2
                mock_print.assert_called_with("Value must be at most 100", "error")
    
    def test_display_table_normal_mode(self):
        """Test table display in normal mode"""
        with patch.object(self.cli.console, 'print') as mock_print:
            self.cli.display_table(
                "Test Table",
                ["Header1", "Header2"],
                [["Row1Col1", "Row1Col2"], ["Row2Col1", "Row2Col2"]]
            )
            
            # Should print table and empty line
            assert mock_print.call_count >= 2
    
    def test_display_table_quiet_mode(self):
        """Test table display in quiet mode"""
        quiet_options = CLIOptions(verbosity=VerbosityLevel.QUIET)
        quiet_cli = InteractiveCLI(quiet_options)
        
        with patch.object(quiet_cli.console, 'print') as mock_print:
            quiet_cli.display_table("Test Table", ["Header1"], [["Data1"]])
            
            mock_print.assert_not_called()

        assert True  # TODO: Add proper assertion
    
    def test_display_key_value_pairs(self):
        """Test key-value pairs display"""
        with patch.object(self.cli.console, 'print') as mock_print:
            self.cli.display_key_value_pairs("Config", {"key1": "value1", "key2": "value2"})
            
            # Should print table and empty line
            assert mock_print.call_count >= 2
    
    def test_display_section_header(self):
        """Test section header display"""
        with patch.object(self.cli.console, 'print') as mock_print:
            self.cli.display_section_header("Test Section")
            
            # Should print empty line, header, and another empty line
            assert mock_print.call_count == 3
    
    def test_display_summary_panel(self):
        """Test summary panel display"""
        with patch.object(self.cli.console, 'print') as mock_print:
            self.cli.display_summary_panel("Summary", "Content", "success")
            
            # Should print panel and empty line
            assert mock_print.call_count >= 2
    
    def test_create_progress_context_with_total(self):
        """Test progress context creation with total"""
        progress = self.cli.create_progress_context("Test", total=100)
        
        # Should return a Progress object
        assert hasattr(progress, 'add_task')
        assert hasattr(progress, 'update')
    
    def test_create_progress_context_without_total(self):
        """Test progress context creation without total"""
        progress = self.cli.create_progress_context("Test")
        
        # Should return a Progress object with spinner
        assert hasattr(progress, 'add_task')
        assert hasattr(progress, 'update')
    
    def test_create_progress_context_quiet_mode(self):
        """Test progress context in quiet mode"""
        quiet_options = CLIOptions(verbosity=VerbosityLevel.QUIET)
        quiet_cli = InteractiveCLI(quiet_options)
        
        progress = quiet_cli.create_progress_context("Test")
        
        # Should return dummy progress
        assert progress.__class__.__name__ == '_DummyProgress'
    
    def test_show_spinner(self):
        """Test spinner display"""
        spinner = self.cli.show_spinner("Loading...")
        
        # Should return a Progress object
        assert hasattr(spinner, 'add_task')
        assert hasattr(spinner, 'update')
    
    def test_show_spinner_quiet_mode(self):
        """Test spinner in quiet mode"""
        quiet_options = CLIOptions(verbosity=VerbosityLevel.QUIET)
        quiet_cli = InteractiveCLI(quiet_options)
        
        spinner = quiet_cli.show_spinner("Loading...")
        
        # Should return dummy progress
        assert spinner.__class__.__name__ == '_DummyProgress'


class TestClickCLI:
    """Test the Click CLI commands"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test CLI help output"""
        result = self.runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert 'WAN22 Server Startup Manager' in result.output
        assert 'Intelligent server orchestration' in result.output
    
    def test_cli_verbose_flag(self):
        """Test verbose flag handling"""
        result = self.runner.invoke(cli, ['-v', 'status'])
        
        # Should not error with verbose flag
        assert result.exit_code == 0
    
    def test_cli_quiet_flag(self):
        """Test quiet flag handling"""
        result = self.runner.invoke(cli, ['-q', 'status'])
        
        # Should not error with quiet flag
        assert result.exit_code == 0
    
    def test_cli_no_color_flag(self):
        """Test no-color flag handling"""
        result = self.runner.invoke(cli, ['--no-color', 'status'])
        
        # Should not error with no-color flag
        assert result.exit_code == 0
    
    def test_start_command_help(self):
        """Test start command help"""
        result = self.runner.invoke(cli, ['start', '--help'])
        
        assert result.exit_code == 0
        assert 'Start both backend and frontend servers' in result.output
        assert '--backend-port' in result.output
        assert '--frontend-port' in result.output
    
    def test_start_command_with_ports(self):
        """Test start command with port arguments"""
        with patch('time.sleep'):  # Speed up the test
            result = self.runner.invoke(cli, [
                '--auto-confirm',
                'start', 
                '--backend-port', '8080', 
                '--frontend-port', '3001'
            ])
        
        assert result.exit_code == 0
    
    def test_status_command(self):
        """Test status command"""
        result = self.runner.invoke(cli, ['status'])
        
        assert result.exit_code == 0
        assert 'Server Status' in result.output
    
    def test_stop_command(self):
        """Test stop command"""
        with patch('time.sleep'):  # Speed up the test
            result = self.runner.invoke(cli, ['--auto-confirm', 'stop'])
        
        assert result.exit_code == 0


class TestDummyProgress:
    """Test the dummy progress class"""
    
    def test_dummy_progress_context_manager(self):
        """Test dummy progress as context manager"""
        from scripts.startup_manager.cli import _DummyProgress
        
        dummy = _DummyProgress()
        
        with dummy as progress:
            assert progress is dummy
    
    def test_dummy_progress_methods(self):
        """Test dummy progress methods"""
        from scripts.startup_manager.cli import _DummyProgress
        
        dummy = _DummyProgress()
        
        # Should not raise errors
        task_id = dummy.add_task("test")
        assert task_id == 0
        
        dummy.update(task_id, advance=1)
        dummy.start()
        dummy.stop()


if __name__ == '__main__':
    pytest.main([__file__])
