"""
Comprehensive tests for the Dependency Recovery System.

Tests all aspects of dependency failure recovery including:
- Virtual environment recreation
- Alternative package source selection
- Version fallback strategies
- Offline package installation
- Recovery strategy selection and execution
"""

import pytest
import tempfile
import shutil
import subprocess
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any, List

# Add the scripts directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from dependency_recovery import (
    DependencyRecovery, RecoveryStrategy, PackageSource, VersionFallback
)
from interfaces import InstallationError, ErrorCategory, HardwareProfile
from setup_dependencies import DependencyManager


class TestDependencyRecovery:
    """Test suite for DependencyRecovery class."""
    
    @pytest.fixture
    def temp_installation_path(self):
        """Create temporary installation directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_dependency_manager(self):
        """Create mock dependency manager."""
        mock_manager = Mock(spec=DependencyManager)
        mock_python_handler = Mock()
        mock_python_handler.get_venv_python_executable.return_value = "/path/to/python"
        mock_python_handler.get_python_executable.return_value = "/path/to/python"
        mock_manager.python_handler = mock_python_handler
        return mock_manager
    
    @pytest.fixture
    def dependency_recovery(self, temp_installation_path, mock_dependency_manager):
        """Create DependencyRecovery instance for testing."""
        return DependencyRecovery(
            str(temp_installation_path), 
            dependency_manager=mock_dependency_manager
        )
    
    def test_initialization(self, dependency_recovery, temp_installation_path):
        """Test proper initialization of DependencyRecovery."""
        assert dependency_recovery.installation_path == temp_installation_path
        assert dependency_recovery.package_cache_dir.exists()
        assert dependency_recovery.offline_packages_dir.exists()
        assert len(dependency_recovery.ALTERNATIVE_SOURCES) > 0
        assert len(dependency_recovery.VERSION_FALLBACKS) > 0
        assert len(dependency_recovery.RECOVERY_STRATEGIES) > 0
    
    def test_recovery_strategy_configuration(self, dependency_recovery):
        """Test that recovery strategies are properly configured."""
        strategies = dependency_recovery.RECOVERY_STRATEGIES
        
        # Check that strategies are ordered by priority
        priorities = [s.priority for s in strategies]
        assert priorities == sorted(priorities)
        
        # Check that all strategies have required fields
        for strategy in strategies:
            assert strategy.name
            assert strategy.description
            assert isinstance(strategy.priority, int)
            assert isinstance(strategy.applicable_errors, list)
            assert 0.0 <= strategy.success_rate <= 1.0
    
    def test_alternative_sources_configuration(self, dependency_recovery):
        """Test that alternative sources are properly configured."""
        sources = dependency_recovery.ALTERNATIVE_SOURCES
        
        for source in sources:
            assert source.name
            assert source.index_url.startswith("https://")
            assert 0.0 <= source.reliability_score <= 1.0
    
    def test_version_fallbacks_configuration(self, dependency_recovery):
        """Test that version fallbacks are properly configured."""
        fallbacks = dependency_recovery.VERSION_FALLBACKS
        
        # Check critical packages have fallbacks
        critical_packages = ["torch", "transformers", "diffusers", "numpy"]
        for package in critical_packages:
            assert package in fallbacks
            fallback = fallbacks[package]
            assert fallback.package_name == package
            assert fallback.preferred_version
            assert len(fallback.fallback_versions) > 0
    
    @patch('subprocess.run')
    def test_recreate_virtual_environment_success(self, mock_run, dependency_recovery):
        """Test successful virtual environment recreation."""
        mock_run.return_value = Mock(returncode=0)
        
        # Mock the python handler method to fail initially
        dependency_recovery.python_handler.create_virtual_environment = Mock(return_value=False)
        
        result = dependency_recovery.recreate_virtual_environment()
        
        assert result is True
        mock_run.assert_called()
    
    @patch('subprocess.run')
    def test_recreate_virtual_environment_with_hardware_profile(self, mock_run, dependency_recovery):
        """Test virtual environment recreation with hardware profile."""
        mock_run.return_value = Mock(returncode=0)
        
        # Create mock hardware profile
        hardware_profile = Mock(spec=HardwareProfile)
        
        # Mock the python handler method to succeed
        dependency_recovery.python_handler.create_virtual_environment = Mock(return_value=True)
        
        result = dependency_recovery.recreate_virtual_environment(
            hardware_profile=hardware_profile
        )
        
        assert result is True
        dependency_recovery.python_handler.create_virtual_environment.assert_called_once_with(
            str(dependency_recovery.installation_path / "venv"), 
            hardware_profile
        )
    
    @patch('shutil.rmtree')
    @patch('subprocess.run')
    def test_recreate_virtual_environment_removes_existing(self, mock_run, mock_rmtree, 
                                                         dependency_recovery, temp_installation_path):
        """Test that existing virtual environment is removed before recreation."""
        mock_run.return_value = Mock(returncode=0)
        
        # Create existing venv directory
        venv_dir = temp_installation_path / "venv"
        venv_dir.mkdir()
        
        dependency_recovery.python_handler.create_virtual_environment = Mock(return_value=False)
        
        result = dependency_recovery.recreate_virtual_environment()
        
        # Should attempt to remove existing directory
        mock_rmtree.assert_called_with(venv_dir, ignore_errors=True)
    
    @patch('subprocess.run')
    def test_install_with_alternative_sources_success(self, mock_run, dependency_recovery):
        """Test successful installation with alternative sources."""
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")
        
        requirements = ["torch==2.1.0", "transformers>=4.30.0"]
        result = dependency_recovery.install_with_alternative_sources(requirements)
        
        assert result is True
        mock_run.assert_called()
        
        # Check that the command includes alternative source
        call_args = mock_run.call_args[0][0]
        assert "--index-url" in call_args
    
    @patch('subprocess.run')
    def test_install_with_alternative_sources_fallback(self, mock_run, dependency_recovery):
        """Test fallback to different sources when first fails."""
        # First call fails, second succeeds
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, "pip", stderr="Network error"),
            Mock(returncode=0, stderr="", stdout="")
        ]
        
        requirements = ["torch==2.1.0"]
        result = dependency_recovery.install_with_alternative_sources(requirements, max_attempts=2)
        
        assert result is True
        assert mock_run.call_count == 2
    
    @patch('subprocess.run')
    def test_install_with_alternative_sources_all_fail(self, mock_run, dependency_recovery):
        """Test when all alternative sources fail."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "pip", stderr="Network error")
        
        requirements = ["torch==2.1.0"]
        result = dependency_recovery.install_with_alternative_sources(requirements, max_attempts=2)
        
        assert result is False
        assert mock_run.call_count == 2
    
    @patch('subprocess.run')
    def test_apply_version_fallbacks_success(self, mock_run, dependency_recovery):
        """Test successful version fallback application."""
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")
        
        failed_packages = ["torch", "transformers"]
        result = dependency_recovery.apply_version_fallbacks(failed_packages)
        
        assert result is True
        mock_run.assert_called()
        
        # Check that fallback versions are used
        call_args = mock_run.call_args[0][0]
        assert any("torch==" in arg for arg in call_args)
        assert any("transformers==" in arg for arg in call_args)
    
    @patch('subprocess.run')
    def test_apply_version_fallbacks_unknown_package(self, mock_run, dependency_recovery):
        """Test version fallback with unknown package."""
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")
        
        failed_packages = ["unknown_package"]
        result = dependency_recovery.apply_version_fallbacks(failed_packages)
        
        assert result is True
        mock_run.assert_called()
        
        # Should install without version constraint
        call_args = mock_run.call_args[0][0]
        assert "unknown_package" in call_args
    
    @patch('subprocess.run')
    def test_setup_offline_installation_success(self, mock_run, dependency_recovery):
        """Test successful offline installation setup."""
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")
        
        requirements = ["torch==2.1.0", "transformers>=4.30.0"]
        result = dependency_recovery.setup_offline_installation(requirements)
        
        assert result is True
        mock_run.assert_called()
        
        # Check that download command is used
        call_args = mock_run.call_args[0][0]
        assert "download" in call_args
        assert "--dest" in call_args
        
        # Check that requirements file is created
        requirements_file = dependency_recovery.offline_packages_dir / "requirements.txt"
        assert requirements_file.exists()
    
    @patch('subprocess.run')
    def test_install_from_offline_cache_success(self, mock_run, dependency_recovery):
        """Test successful installation from offline cache."""
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")
        
        # Create mock requirements file
        requirements_file = dependency_recovery.offline_packages_dir / "requirements.txt"
        requirements_file.write_text("torch==2.1.0\ntransformers>=4.30.0\n")
        
        result = dependency_recovery.install_from_offline_cache()
        
        assert result is True
        mock_run.assert_called()
        
        # Check that offline installation flags are used
        call_args = mock_run.call_args[0][0]
        assert "--find-links" in call_args
        assert "--no-index" in call_args
        assert "-r" in call_args
    
    def test_install_from_offline_cache_no_directory(self, dependency_recovery):
        """Test offline installation when cache directory doesn't exist."""
        # Remove the offline directory
        shutil.rmtree(dependency_recovery.offline_packages_dir)
        
        result = dependency_recovery.install_from_offline_cache()
        
        assert result is False
    
    @patch('subprocess.run')
    def test_install_from_offline_cache_with_wheels(self, mock_run, dependency_recovery):
        """Test offline installation with wheel files."""
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")
        
        # Create mock wheel files
        wheel1 = dependency_recovery.offline_packages_dir / "torch-2.1.0-py3-none-any.whl"
        wheel2 = dependency_recovery.offline_packages_dir / "transformers-4.30.0-py3-none-any.whl"
        wheel1.touch()
        wheel2.touch()
        
        result = dependency_recovery.install_from_offline_cache()
        
        assert result is True
        mock_run.assert_called()
        
        # Check that wheel files are included
        call_args = mock_run.call_args[0][0]
        assert str(wheel1) in call_args
        assert str(wheel2) in call_args
    
    def test_analyze_error_for_strategies(self, dependency_recovery):
        """Test error analysis for strategy selection."""
        # Test network error
        network_error = Exception("Connection timeout occurred")
        strategies = dependency_recovery._analyze_error_for_strategies(network_error, {})
        
        strategy_names = [s.name for s in strategies]
        assert "alternative_source" in strategy_names
        
        # Test cache error
        cache_error = Exception("Cache corruption detected")
        strategies = dependency_recovery._analyze_error_for_strategies(cache_error, {})
        
        strategy_names = [s.name for s in strategies]
        assert "retry_with_cache_clear" in strategy_names
        
        # Test version error
        version_error = Exception("No matching version found")
        strategies = dependency_recovery._analyze_error_for_strategies(version_error, {})
        
        strategy_names = [s.name for s in strategies]
        assert "version_fallback" in strategy_names
    
    def test_analyze_error_fallback_strategies(self, dependency_recovery):
        """Test that fallback strategies are provided for unknown errors."""
        unknown_error = Exception("Some unknown error")
        strategies = dependency_recovery._analyze_error_for_strategies(unknown_error, {})
        
        # Should include general fallback strategies
        strategy_names = [s.name for s in strategies]
        assert "retry_with_cache_clear" in strategy_names
        assert "alternative_source" in strategy_names
    
    @patch('subprocess.run')
    def test_retry_with_cache_clear_success(self, mock_run, dependency_recovery):
        """Test successful retry with cache clear."""
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")
        
        context = {"requirements": ["torch==2.1.0"]}
        result = dependency_recovery._retry_with_cache_clear(context)
        
        assert result is True
        assert mock_run.call_count == 2  # Cache purge + install
        
        # Check cache purge command
        first_call = mock_run.call_args_list[0][0][0]
        assert "cache" in first_call
        assert "purge" in first_call
        
        # Check install command
        second_call = mock_run.call_args_list[1][0][0]
        assert "install" in second_call
        assert "--no-cache-dir" in second_call
        assert "--force-reinstall" in second_call
    
    @patch.object(DependencyRecovery, '_retry_with_cache_clear')
    @patch.object(DependencyRecovery, '_try_alternative_sources')
    def test_execute_recovery_strategy_routing(self, mock_alt_sources, mock_cache_clear, 
                                             dependency_recovery):
        """Test that recovery strategies are routed correctly."""
        mock_cache_clear.return_value = True
        mock_alt_sources.return_value = True
        
        # Test cache clear strategy
        cache_strategy = RecoveryStrategy(
            name="retry_with_cache_clear",
            description="Test",
            priority=1,
            applicable_errors=[],
            success_rate=0.5
        )
        
        result = dependency_recovery._execute_recovery_strategy(cache_strategy, Exception(), {})
        assert result is True
        mock_cache_clear.assert_called_once()
        
        # Test alternative source strategy
        alt_strategy = RecoveryStrategy(
            name="alternative_source",
            description="Test",
            priority=1,
            applicable_errors=[],
            success_rate=0.5
        )
        
        result = dependency_recovery._execute_recovery_strategy(alt_strategy, Exception(), {})
        assert result is True
        mock_alt_sources.assert_called_once()
    
    def test_execute_recovery_strategy_unknown(self, dependency_recovery):
        """Test handling of unknown recovery strategy."""
        unknown_strategy = RecoveryStrategy(
            name="unknown_strategy",
            description="Test",
            priority=1,
            applicable_errors=[],
            success_rate=0.5
        )
        
        result = dependency_recovery._execute_recovery_strategy(unknown_strategy, Exception(), {})
        assert result is False
    
    @patch.object(DependencyRecovery, '_execute_recovery_strategy')
    def test_recover_dependency_failure_success(self, mock_execute, dependency_recovery):
        """Test successful dependency failure recovery."""
        mock_execute.return_value = True
        
        error = Exception("Network timeout")
        context = {"requirements": ["torch==2.1.0"]}
        
        result = dependency_recovery.recover_dependency_failure(error, context)
        
        assert result is True
        mock_execute.assert_called_once()
    
    @patch.object(DependencyRecovery, '_execute_recovery_strategy')
    def test_recover_dependency_failure_all_strategies_fail(self, mock_execute, dependency_recovery):
        """Test when all recovery strategies fail."""
        mock_execute.return_value = False
        
        error = Exception("Network timeout")
        context = {"requirements": ["torch==2.1.0"]}
        
        result = dependency_recovery.recover_dependency_failure(error, context)
        
        assert result is False
        # Should try multiple strategies
        assert mock_execute.call_count > 1
    
    @patch.object(DependencyRecovery, '_execute_recovery_strategy')
    def test_recover_dependency_failure_strategy_exception(self, mock_execute, dependency_recovery):
        """Test handling of exceptions during strategy execution."""
        # First strategy raises exception, second succeeds
        mock_execute.side_effect = [Exception("Strategy failed"), True]
        
        error = Exception("Network timeout")
        context = {"requirements": ["torch==2.1.0"]}
        
        result = dependency_recovery.recover_dependency_failure(error, context)
        
        assert result is True
        assert mock_execute.call_count == 2
    
    def test_log_recovery_success(self, dependency_recovery):
        """Test logging of successful recovery."""
        strategy = RecoveryStrategy(
            name="test_strategy",
            description="Test",
            priority=1,
            applicable_errors=[],
            success_rate=0.5
        )
        
        error = Exception("Test error")
        context = {"test": "context"}
        
        dependency_recovery._log_recovery_success(strategy, error, context)
        
        assert len(dependency_recovery.recovery_log) == 1
        
        log_entry = dependency_recovery.recovery_log[0]
        assert log_entry["strategy"] == "test_strategy"
        assert log_entry["error_type"] == "Exception"
        assert log_entry["error_message"] == "Test error"
        assert log_entry["context"] == context
        assert "timestamp" in log_entry
    
    def test_get_recovery_statistics_empty(self, dependency_recovery):
        """Test recovery statistics with no recovery attempts."""
        stats = dependency_recovery.get_recovery_statistics()
        
        assert stats["total_recoveries"] == 0
        assert stats["strategies"] == {}
    
    def test_get_recovery_statistics_with_data(self, dependency_recovery):
        """Test recovery statistics with recovery data."""
        # Add some mock recovery log entries
        dependency_recovery.recovery_log = [
            {"strategy": "retry_with_cache_clear", "timestamp": "2023-01-01"},
            {"strategy": "alternative_source", "timestamp": "2023-01-02"},
            {"strategy": "retry_with_cache_clear", "timestamp": "2023-01-03"}
        ]
        
        stats = dependency_recovery.get_recovery_statistics()
        
        assert stats["total_recoveries"] == 3
        assert stats["strategies"]["retry_with_cache_clear"]["attempts"] == 2
        assert stats["strategies"]["alternative_source"]["attempts"] == 1


class TestIntegrationScenarios:
    """Integration tests for common dependency failure scenarios."""
    
    @pytest.fixture
    def temp_installation_path(self):
        """Create temporary installation directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def dependency_recovery(self, temp_installation_path):
        """Create DependencyRecovery instance for integration testing."""
        mock_manager = Mock(spec=DependencyManager)
        mock_python_handler = Mock()
        mock_python_handler.get_venv_python_executable.return_value = "/path/to/python"
        mock_python_handler.get_python_executable.return_value = "/path/to/python"
        mock_manager.python_handler = mock_python_handler
        
        return DependencyRecovery(str(temp_installation_path), dependency_manager=mock_manager)
    
    @patch('subprocess.run')
    def test_network_failure_recovery_scenario(self, mock_run, dependency_recovery):
        """Test complete network failure recovery scenario."""
        # Simulate network failure followed by success with alternative source
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, "pip", stderr="Network timeout"),  # Cache clear fails
            subprocess.CalledProcessError(1, "pip", stderr="Network timeout"),  # Retry fails
            Mock(returncode=0, stderr="", stdout="")  # Alternative source succeeds
        ]
        
        error = Exception("Network timeout during package installation")
        context = {"requirements": ["torch==2.1.0", "transformers>=4.30.0"]}
        
        result = dependency_recovery.recover_dependency_failure(error, context)
        
        assert result is True
        assert mock_run.call_count == 3
    
    @patch('subprocess.run')
    def test_version_conflict_recovery_scenario(self, mock_run, dependency_recovery):
        """Test version conflict recovery scenario."""
        # Simulate version conflict resolved by fallback
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, "pip", stderr="Cache purge failed"),  # Cache clear fails
            subprocess.CalledProcessError(1, "pip", stderr="Network error"),  # Alternative source fails
            Mock(returncode=0, stderr="", stdout="")  # Version fallback succeeds
        ]
        
        error = Exception("No matching version found for torch>=2.2.0")
        context = {"requirements": ["torch>=2.2.0"], "failed_packages": ["torch"]}
        
        result = dependency_recovery.recover_dependency_failure(error, context)
        
        assert result is True
        assert mock_run.call_count == 3
    
    @patch('subprocess.run')
    @patch('shutil.rmtree')
    def test_environment_corruption_recovery_scenario(self, mock_rmtree, mock_run, 
                                                    dependency_recovery):
        """Test virtual environment corruption recovery scenario."""
        # Simulate environment corruption resolved by recreation
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, "pip", stderr="Cache error"),  # Cache clear fails
            subprocess.CalledProcessError(1, "pip", stderr="Network error"),  # Alternative source fails
            subprocess.CalledProcessError(1, "pip", stderr="Version error"),  # Version fallback fails
            Mock(returncode=0, stderr="", stdout=""),  # Venv recreation succeeds
            Mock(returncode=0, stderr="", stdout="")   # Package installation succeeds
        ]
        
        # Mock python handler to fail initially then succeed
        dependency_recovery.python_handler.create_virtual_environment = Mock(return_value=False)
        
        error = Exception("Virtual environment is corrupted")
        context = {"requirements": ["torch==2.1.0"]}
        
        result = dependency_recovery.recover_dependency_failure(error, context)
        
        assert result is True
        # Should attempt venv recreation
        mock_rmtree.assert_called()
        mock_run.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])