"""
Unit tests for startup manager utility functions.
"""

import os
import sys
import pytest
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.startup_manager.utils import SystemDetector, PathManager


class TestSystemDetector:
    """Test cases for SystemDetector utility class."""
    
    def test_get_system_info(self):
        """Test getting comprehensive system information."""
        info = SystemDetector.get_system_info()
        
        # Check that all expected keys are present
        expected_keys = [
            "platform", "platform_version", "architecture", "processor",
            "python_version", "python_executable", "working_directory",
            "user", "is_admin", "virtual_env"
        ]
        
        for key in expected_keys:
            assert key in info
        
        # Check that values are reasonable
        assert isinstance(info["platform"], str)
        assert isinstance(info["is_admin"], bool)
        assert isinstance(info["virtual_env"], dict)
    
    def test_is_windows(self):
        """Test Windows detection."""
        with patch('platform.system', return_value='Windows'):
            assert SystemDetector.is_windows() is True
        
        with patch('platform.system', return_value='Linux'):
            assert SystemDetector.is_windows() is False
        
        with patch('platform.system', return_value='Darwin'):
            assert SystemDetector.is_windows() is False
    
    def test_is_admin_windows(self):
        """Test admin detection on Windows."""
        with patch('platform.system', return_value='Windows'):
            # Mock Windows admin check
            with patch('ctypes.windll.shell32.IsUserAnAdmin', return_value=1):
                assert SystemDetector.is_admin() is True
            
            with patch('ctypes.windll.shell32.IsUserAnAdmin', return_value=0):
                assert SystemDetector.is_admin() is False
            
            # Test exception handling
            with patch('ctypes.windll.shell32.IsUserAnAdmin', side_effect=Exception()):
                assert SystemDetector.is_admin() is False
    
    def test_is_admin_unix(self):
        """Test admin detection on Unix systems."""
        with patch('platform.system', return_value='Linux'):
            with patch('os.geteuid', return_value=0, create=True):
                assert SystemDetector.is_admin() is True
            
            with patch('os.geteuid', return_value=1000, create=True):
                assert SystemDetector.is_admin() is False
    
    def test_get_virtual_env_info_no_venv(self):
        """Test virtual environment detection when no venv is active."""
        # Mock no virtual environment
        with patch.object(sys, 'prefix', '/usr/local'), \
             patch.object(sys, 'base_prefix', '/usr/local'):
            
            venv_info = SystemDetector.get_virtual_env_info()
            assert venv_info["active"] is False
            assert venv_info["path"] is None
            assert venv_info["type"] is None
    
    def test_get_virtual_env_info_with_venv(self):
        """Test virtual environment detection when venv is active."""
        # Mock virtual environment active
        with patch.object(sys, 'prefix', '/path/to/venv'), \
             patch.object(sys, 'base_prefix', '/usr/local'), \
             patch('os.path.exists', return_value=True):
            
            venv_info = SystemDetector.get_virtual_env_info()
            assert venv_info["active"] is True
            assert venv_info["path"] == '/path/to/venv'
            assert venv_info["type"] == 'venv'
    
    def test_get_virtual_env_info_conda(self):
        """Test conda environment detection."""
        with patch.object(sys, 'prefix', '/path/to/conda/envs/myenv'), \
             patch.object(sys, 'base_prefix', '/usr/local'):
            
            venv_info = SystemDetector.get_virtual_env_info()
            assert venv_info["active"] is True
            assert venv_info["type"] == 'conda'
    
    def test_check_python_version(self):
        """Test Python version checking."""
        # Test current version meets requirement
        with patch.object(sys, 'version_info', (3, 9, 0)):
            result = SystemDetector.check_python_version((3, 8))
            assert result["meets_requirement"] is True
            assert result["current_version"] == "3.9"
            assert result["required_version"] == "3.8"
        
        # Test current version doesn't meet requirement
        with patch.object(sys, 'version_info', (3, 7, 0)):
            result = SystemDetector.check_python_version((3, 8))
            assert result["meets_requirement"] is False
            assert result["current_version"] == "3.7"
    
    @patch('shutil.which')
    def test_check_command_available_found(self, mock_which):
        """Test command availability check when command is found."""
        mock_which.return_value = '/usr/bin/node'
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='v16.14.0')
            
            result = SystemDetector.check_command_available('node')
            
            assert result["available"] is True
            assert result["path"] == '/usr/bin/node'
            assert result["version"] == 'v16.14.0'
    
    @patch('shutil.which')
    def test_check_command_available_not_found(self, mock_which):
        """Test command availability check when command is not found."""
        mock_which.return_value = None
        
        result = SystemDetector.check_command_available('nonexistent')
        
        assert result["available"] is False
        assert result["path"] is None
        assert result["version"] is None
    
    @patch('shutil.which')
    def test_check_command_available_version_error(self, mock_which):
        """Test command availability when version check fails."""
        mock_which.return_value = '/usr/bin/node'
        
        with patch('subprocess.run', side_effect=subprocess.SubprocessError()):
            result = SystemDetector.check_command_available('node')
            
            assert result["available"] is True
            assert result["path"] == '/usr/bin/node'
            assert result["version"] is None
    
    def test_check_node_environment(self):
        """Test Node.js environment checking."""
        with patch.object(SystemDetector, 'check_command_available') as mock_check:
            # Mock both node and npm available
            mock_check.side_effect = [
                {"available": True, "path": "/usr/bin/node", "version": "v16.14.0"},
                {"available": True, "path": "/usr/bin/npm", "version": "8.3.1"}
            ]
            
            result = SystemDetector.check_node_environment()
            
            assert result["environment_ready"] is True
            assert result["node"]["available"] is True
            assert result["npm"]["available"] is True
            
            # Mock node missing
            mock_check.side_effect = [
                {"available": False, "path": None, "version": None},
                {"available": True, "path": "/usr/bin/npm", "version": "8.3.1"}
            ]
            
            result = SystemDetector.check_node_environment()
            assert result["environment_ready"] is False


class TestPathManager:
    """Test cases for PathManager utility class."""
    
    def test_init_with_explicit_root(self):
        """Test PathManager initialization with explicit project root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            path_manager = PathManager(root_path)
            
            assert path_manager.project_root == root_path
    
    def test_init_finds_project_root(self):
        """Test PathManager finding project root automatically."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            
            # Create project structure
            (root_path / "backend").mkdir()
            (root_path / "frontend").mkdir()
            (root_path / "README.md").touch()
            
            # Create subdirectory and initialize PathManager from there
            sub_dir = root_path / "subdir"
            sub_dir.mkdir()
            
            with patch('pathlib.Path.cwd', return_value=sub_dir):
                path_manager = PathManager()
                assert path_manager.project_root == root_path
    
    def test_get_path_methods(self):
        """Test various path getter methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            path_manager = PathManager(root_path)
            
            assert path_manager.get_backend_path() == root_path / "backend"
            assert path_manager.get_frontend_path() == root_path / "frontend"
            assert path_manager.get_scripts_path() == root_path / "scripts"
            assert path_manager.get_config_path() == root_path / "startup_config.json"
            assert path_manager.get_config_path("custom.json") == root_path / "custom.json"
    
    def test_get_logs_path_creates_directory(self):
        """Test that get_logs_path creates the directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            path_manager = PathManager(root_path)
            
            logs_path = path_manager.get_logs_path()
            
            assert logs_path == root_path / "logs"
            assert logs_path.exists()
            assert logs_path.is_dir()
    
    def test_validate_project_structure_valid(self):
        """Test project structure validation with valid structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            
            # Create valid project structure
            backend_dir = root_path / "backend"
            frontend_dir = root_path / "frontend"
            backend_dir.mkdir()
            frontend_dir.mkdir()
            
            (backend_dir / "main.py").touch()
            (backend_dir / "requirements.txt").touch()
            (frontend_dir / "package.json").touch()
            (root_path / "README.md").touch()
            
            path_manager = PathManager(root_path)
            result = path_manager.validate_project_structure()
            
            assert result["valid"] is True
            assert len(result["missing_directories"]) == 0
            assert len(result["missing_files"]) == 0
    
    def test_validate_project_structure_missing_files(self):
        """Test project structure validation with missing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            
            # Create partial project structure
            backend_dir = root_path / "backend"
            frontend_dir = root_path / "frontend"
            backend_dir.mkdir()
            frontend_dir.mkdir()
            
            # Missing required files
            
            path_manager = PathManager(root_path)
            result = path_manager.validate_project_structure()
            
            assert result["valid"] is False
            assert len(result["missing_files"]) > 0
            assert any("main.py" in path for path in result["missing_files"])
    
    def test_create_directory_structure(self):
        """Test creating missing directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            path_manager = PathManager(root_path)
            
            created_dirs = path_manager.create_directory_structure()
            
            assert len(created_dirs) > 0
            assert path_manager.get_logs_path().exists()
            assert path_manager.get_scripts_path().exists()
            assert (path_manager.get_scripts_path() / "startup_manager").exists()
    
    def test_get_relative_path(self):
        """Test getting relative paths from project root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            path_manager = PathManager(root_path)
            
            backend_path = root_path / "backend" / "main.py"
            relative_path = path_manager.get_relative_path(backend_path)
            
            assert relative_path == "backend/main.py" or relative_path == "backend\\main.py"
    
    def test_resolve_path(self):
        """Test resolving path strings relative to project root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            path_manager = PathManager(root_path)
            
            # Test relative path
            resolved = path_manager.resolve_path("backend/main.py")
            assert resolved == root_path / "backend" / "main.py"
            
            # Test absolute path
            abs_path = "/absolute/path"
            resolved = path_manager.resolve_path(abs_path)
            assert str(resolved) == abs_path


if __name__ == "__main__":
    pytest.main([__file__])
