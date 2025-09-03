"""Tests for the unified CLI system"""

import pytest
import subprocess
import sys
from pathlib import Path
from typer.testing import CliRunner

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cli.main import app

class TestUnifiedCLI:
    """Test the unified CLI functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test that CLI help works"""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "WAN Project Quality & Maintenance Toolkit" in result.output
    
    def test_status_command(self):
        """Test the status command"""
        result = self.runner.invoke(app, ["status"])
        # Should not crash, even if some checks fail
        assert result.exit_code in [0, 1]
        assert "project status" in result.output.lower() or "health" in result.output.lower()
    
    def test_quick_command(self):
        """Test the quick validation command"""
        result = self.runner.invoke(app, ["quick"])
        # Should not crash
        assert result.exit_code in [0, 1]
        assert "validation" in result.output.lower()
    
    def test_test_commands(self):
        """Test test-related commands"""
        # Test help
        result = self.runner.invoke(app, ["test", "--help"])
        assert result.exit_code == 0
        assert "testing" in result.output.lower()
        
        # Test run command help
        result = self.runner.invoke(app, ["test", "run", "--help"])
        assert result.exit_code == 0
    
    def test_clean_commands(self):
        """Test cleanup-related commands"""
        # Test help
        result = self.runner.invoke(app, ["clean", "--help"])
        assert result.exit_code == 0
        assert "cleanup" in result.output.lower()
        
        # Test duplicates dry run
        result = self.runner.invoke(app, ["clean", "duplicates"])
        assert result.exit_code in [0, 1]  # May find duplicates or not
    
    def test_config_commands(self):
        """Test configuration-related commands"""
        # Test help
        result = self.runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "configuration" in result.output.lower()
        
        # Test show command
        result = self.runner.invoke(app, ["config", "show"])
        assert result.exit_code in [0, 1]  # May have config issues
    
    def test_docs_commands(self):
        """Test documentation-related commands"""
        # Test help
        result = self.runner.invoke(app, ["docs", "--help"])
        assert result.exit_code == 0
        assert "documentation" in result.output.lower()
        
        # Test structure command
        result = self.runner.invoke(app, ["docs", "structure"])
        assert result.exit_code in [0, 1]
    
    def test_quality_commands(self):
        """Test quality-related commands"""
        # Test help
        result = self.runner.invoke(app, ["quality", "--help"])
        assert result.exit_code == 0
        assert "quality" in result.output.lower()
        
        # Test check command (dry run)
        result = self.runner.invoke(app, ["quality", "check"])
        assert result.exit_code in [0, 1]  # May find quality issues
    
    def test_health_commands(self):
        """Test health-related commands"""
        # Test help
        result = self.runner.invoke(app, ["health", "--help"])
        assert result.exit_code == 0
        assert "health" in result.output.lower()
        
        # Test quick check
        result = self.runner.invoke(app, ["health", "check", "--quick"])
        assert result.exit_code in [0, 1]
    
    def test_deploy_commands(self):
        """Test deployment-related commands"""
        # Test help
        result = self.runner.invoke(app, ["deploy", "--help"])
        assert result.exit_code == 0
        assert "deployment" in result.output.lower()
    
    def test_verbose_mode(self):
        """Test verbose mode works"""
        result = self.runner.invoke(app, ["--verbose", "status"])
        assert result.exit_code in [0, 1]
        # In verbose mode, should see more output
        assert len(result.output) > 0


class TestCLIIntegration:
    """Test CLI integration with actual subprocess calls"""
    
    def test_cli_as_module(self):
        """Test running CLI as Python module"""
        result = subprocess.run(
            [sys.executable, "-m", "cli.main", "--help"],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        assert result.returncode == 0
        assert "WAN Project" in result.stdout
    
    def test_cli_script_exists(self):
        """Test that CLI script exists and is executable"""
        cli_script = project_root / "wan-cli"
        assert cli_script.exists()
        
        # Test it can be executed
        result = subprocess.run(
            [sys.executable, str(cli_script), "--help"],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        assert result.returncode == 0


class TestCLIWorkflows:
    """Test common CLI workflows"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
    
    def test_development_workflow(self):
        """Test a typical development workflow"""
        # Morning routine
        result = self.runner.invoke(app, ["status"])
        assert result.exit_code in [0, 1]
        
        result = self.runner.invoke(app, ["quick"])
        assert result.exit_code in [0, 1]
        
        # Before commit workflow
        result = self.runner.invoke(app, ["clean", "imports"])  # Dry run
        assert result.exit_code in [0, 1]
        
        result = self.runner.invoke(app, ["quality", "check"])  # Dry run
        assert result.exit_code in [0, 1]
    
    def test_maintenance_workflow(self):
        """Test a maintenance workflow"""
        # Health check
        result = self.runner.invoke(app, ["health", "check", "--quick"])
        assert result.exit_code in [0, 1]
        
        # Cleanup check (dry run)
        result = self.runner.invoke(app, ["clean", "all"])
        assert result.exit_code in [0, 1]
    
    def test_error_handling(self):
        """Test that CLI handles errors gracefully"""
        # Invalid command
        result = self.runner.invoke(app, ["nonexistent-command"])
        assert result.exit_code != 0
        assert "No such command" in result.output or "Usage:" in result.output
        
        # Invalid subcommand
        result = self.runner.invoke(app, ["test", "nonexistent-subcommand"])
        assert result.exit_code != 0


class TestCLIPerformance:
    """Test CLI performance characteristics"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
    
    def test_help_is_fast(self):
        """Test that help commands are fast"""
        import time
        
        start_time = time.time()
        result = self.runner.invoke(app, ["--help"])
        end_time = time.time()
        
        assert result.exit_code == 0
        assert (end_time - start_time) < 2.0  # Should be under 2 seconds
    
    def test_quick_command_is_fast(self):
        """Test that quick command is reasonably fast"""
        import time
        
        start_time = time.time()
        result = self.runner.invoke(app, ["quick"])
        end_time = time.time()
        
        assert result.exit_code in [0, 1]
        assert (end_time - start_time) < 30.0  # Should be under 30 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])