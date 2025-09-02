"""
Integration tests for Configuration Validator components.
Tests configuration validation, repair, and file creation functionality.
"""

import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock
import pytest

# Add the scripts directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from scripts.startup_manager.environment_validator import ConfigurationValidator, ValidationIssue, ValidationStatus
    ConfigurationValidator,
    ValidationIssue,
    ValidationStatus
)


class TestConfigurationValidator:
    """Test cases for ConfigurationValidator class."""
    
    def setup_method(self):
        """Set up test fixtures with temporary directory."""
        self.validator = ConfigurationValidator()
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = Path.cwd()
        
        # Create temporary backend and frontend directories
        self.backend_dir = Path(self.temp_dir) / "backend"
        self.frontend_dir = Path(self.temp_dir) / "frontend"
        self.backend_dir.mkdir(parents=True, exist_ok=True)
        self.frontend_dir.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_backend_config_missing_file(self):
        """Test backend config validation with missing config.json."""
        with patch('pathlib.Path') as mock_path:
            mock_config_path = Mock()
            mock_config_path.exists.return_value = False
            mock_path.return_value = mock_config_path
            
            issues = self.validator.validate_backend_config()
            
            assert len(issues) == 1
            assert issues[0].component == "backend_config"
            assert issues[0].issue_type == "missing_config_file"
            assert issues[0].status == ValidationStatus.FAILED
            assert issues[0].auto_fixable
    
    def test_validate_backend_config_invalid_json(self):
        """Test backend config validation with invalid JSON."""
        config_path = self.backend_dir / "config.json"
        config_path.write_text("{ invalid json }", encoding='utf-8')
        
        with patch('pathlib.Path', return_value=config_path):
            issues = self.validator.validate_backend_config()
            
            assert len(issues) == 1
            assert issues[0].component == "backend_config"
            assert issues[0].issue_type == "invalid_json"
            assert issues[0].status == ValidationStatus.FAILED
            assert issues[0].auto_fixable
    
    def test_validate_backend_config_missing_fields(self):
        """Test backend config validation with missing required fields."""
        config_data = {
            "model_path": "models/",
            "device": "auto"
            # Missing other required fields
        }
        
        config_path = self.backend_dir / "config.json"
        config_path.write_text(json.dumps(config_data), encoding='utf-8')
        
        with patch('pathlib.Path', return_value=config_path):
            issues = self.validator.validate_backend_config()
            
            missing_field_issue = next((issue for issue in issues if issue.issue_type == "missing_required_fields"), None)
            assert missing_field_issue is not None
            assert missing_field_issue.status == ValidationStatus.FAILED
            assert missing_field_issue.auto_fixable
            assert "max_memory" in missing_field_issue.details["missing_fields"]
    
    def test_validate_backend_config_invalid_types(self):
        """Test backend config validation with invalid field types."""
        config_data = {
            "model_path": "models/",
            "device": "auto",
            "max_memory": "invalid",  # Should be number
            "batch_size": "1",        # Should be int
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "height": 512,
            "width": 512,
            "num_frames": 16,
            "fps": 8
        }
        
        config_path = self.backend_dir / "config.json"
        config_path.write_text(json.dumps(config_data), encoding='utf-8')
        
        with patch('pathlib.Path', return_value=config_path):
            issues = self.validator.validate_backend_config()
            
            type_issue = next((issue for issue in issues if issue.issue_type == "invalid_field_types"), None)
            assert type_issue is not None
            assert type_issue.status == ValidationStatus.WARNING
            assert type_issue.auto_fixable
    
    def test_validate_backend_config_invalid_values(self):
        """Test backend config validation with invalid field values."""
        config_data = {
            "model_path": "models/",
            "device": "invalid_device",  # Invalid device
            "max_memory": -1,            # Invalid memory
            "batch_size": 1,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "height": -512,              # Invalid resolution
            "width": 512,
            "num_frames": 16,
            "fps": 8
        }
        
        config_path = self.backend_dir / "config.json"
        config_path.write_text(json.dumps(config_data), encoding='utf-8')
        
        with patch('pathlib.Path', return_value=config_path):
            issues = self.validator.validate_backend_config()
            
            # Should find device, memory, and resolution issues
            device_issue = next((issue for issue in issues if issue.issue_type == "invalid_device_setting"), None)
            memory_issue = next((issue for issue in issues if issue.issue_type == "invalid_memory_setting"), None)
            resolution_issue = next((issue for issue in issues if issue.issue_type == "invalid_resolution"), None)
            
            assert device_issue is not None
            assert memory_issue is not None
            assert resolution_issue is not None
    
    def test_validate_backend_config_valid(self):
        """Test backend config validation with valid configuration."""
        config_data = {
            "model_path": "models/",
            "device": "auto",
            "max_memory": 8.0,
            "batch_size": 1,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "height": 512,
            "width": 512,
            "num_frames": 16,
            "fps": 8
        }
        
        config_path = self.backend_dir / "config.json"
        config_path.write_text(json.dumps(config_data), encoding='utf-8')
        
        with patch('pathlib.Path', return_value=config_path):
            issues = self.validator.validate_backend_config()
            
            # Should have no critical issues
            critical_issues = [issue for issue in issues if issue.status == ValidationStatus.FAILED]
            assert len(critical_issues) == 0
    
    def test_validate_frontend_config_missing_package_json(self):
        """Test frontend config validation with missing package.json."""
        with patch('pathlib.Path') as mock_path:
            mock_package_path = Mock()
            mock_package_path.exists.return_value = False
            
            def path_constructor(path_str):
                if "package.json" in str(path_str):
                    return mock_package_path
                else:
                    mock_obj = Mock()
                    mock_obj.exists.return_value = True
                    return mock_obj
            
            mock_path.side_effect = path_constructor
            
            issues = self.validator.validate_frontend_config()
            
            package_issue = next((issue for issue in issues if issue.issue_type == "missing_package_json"), None)
            assert package_issue is not None
            assert package_issue.status == ValidationStatus.FAILED
            assert package_issue.auto_fixable
    
    def test_validate_frontend_config_invalid_package_json(self):
        """Test frontend config validation with invalid package.json."""
        package_path = self.frontend_dir / "package.json"
        package_path.write_text("{ invalid json }", encoding='utf-8')
        
        with patch('pathlib.Path') as mock_path:
            def path_constructor(path_str):
                if "package.json" in str(path_str):
                    return package_path
                else:
                    mock_obj = Mock()
                    mock_obj.exists.return_value = True
                    return mock_obj
            
            mock_path.side_effect = path_constructor
            
            issues = self.validator.validate_frontend_config()
            
            json_issue = next((issue for issue in issues if issue.issue_type == "invalid_package_json"), None)
            assert json_issue is not None
            assert json_issue.status == ValidationStatus.FAILED
            assert json_issue.auto_fixable
    
    def test_validate_frontend_config_missing_vite_config(self):
        """Test frontend config validation with missing vite.config.ts."""
        # Create valid package.json
        package_data = {
            "name": "test-app",
            "version": "1.0.0",
            "scripts": {"dev": "vite", "build": "vite build", "preview": "vite preview"},
            "dependencies": {},
            "devDependencies": {}
        }
        package_path = self.frontend_dir / "package.json"
        package_path.write_text(json.dumps(package_data), encoding='utf-8')
        
        with patch('pathlib.Path') as mock_path:
            def path_constructor(path_str):
                if "package.json" in str(path_str):
                    return package_path
                elif "vite.config.ts" in str(path_str):
                    mock_vite = Mock()
                    mock_vite.exists.return_value = False
                    return mock_vite
                else:
                    mock_obj = Mock()
                    mock_obj.exists.return_value = True
                    return mock_obj
            
            mock_path.side_effect = path_constructor
            
            issues = self.validator.validate_frontend_config()
            
            vite_issue = next((issue for issue in issues if issue.issue_type == "missing_vite_config"), None)
            assert vite_issue is not None
            assert vite_issue.status == ValidationStatus.WARNING
            assert vite_issue.auto_fixable
    
    def test_auto_repair_config_create_defaults(self):
        """Test automatic repair by creating default configuration files."""
        # Create issues for missing files
        backend_issue = ValidationIssue(
            component="backend_config",
            issue_type="missing_config_file",
            message="Backend config missing",
            status=ValidationStatus.FAILED,
            auto_fixable=True,
            fix_command="create_default_backend_config"
        )
        
        frontend_issue = ValidationIssue(
            component="frontend_config",
            issue_type="missing_package_json",
            message="Frontend package.json missing",
            status=ValidationStatus.FAILED,
            auto_fixable=True,
            fix_command="create_default_package_json"
        )
        
        vite_issue = ValidationIssue(
            component="frontend_config",
            issue_type="missing_vite_config",
            message="Vite config missing",
            status=ValidationStatus.WARNING,
            auto_fixable=True,
            fix_command="create_default_vite_config"
        )
        
        issues = [backend_issue, frontend_issue, vite_issue]
        
        # Mock the file creation methods to avoid actual file operations
        with patch.object(self.validator, '_create_default_backend_config') as mock_backend, \
             patch.object(self.validator, '_create_default_package_json') as mock_package, \
             patch.object(self.validator, '_create_default_vite_config') as mock_vite:
            
            repairs = self.validator.auto_repair_config(issues)
            
            assert len(repairs) == 3
            assert "Created default backend config.json" in repairs
            assert "Created default frontend package.json" in repairs
            assert "Created default vite.config.ts" in repairs
            
            # Verify methods were called
            mock_backend.assert_called_once()
            mock_package.assert_called_once()
            mock_vite.assert_called_once()
            
            # Verify issue statuses were updated
            assert backend_issue.status == ValidationStatus.FIXED
            assert frontend_issue.status == ValidationStatus.FIXED
            assert vite_issue.status == ValidationStatus.FIXED
    
    def test_auto_repair_config_add_missing_fields(self):
        """Test automatic repair by adding missing configuration fields."""
        issue = ValidationIssue(
            component="backend_config",
            issue_type="missing_required_fields",
            message="Missing fields",
            status=ValidationStatus.FAILED,
            auto_fixable=True,
            fix_command="add_missing_backend_config_fields",
            details={"missing_fields": ["max_memory", "batch_size"]}
        )
        
        with patch.object(self.validator, '_add_missing_backend_config_fields') as mock_add_fields:
            repairs = self.validator.auto_repair_config([issue])
            
            assert len(repairs) == 1
            assert "Added missing backend config fields: max_memory, batch_size" in repairs[0]
            mock_add_fields.assert_called_once_with(["max_memory", "batch_size"])
            assert issue.status == ValidationStatus.FIXED
    
    def test_create_default_backend_config(self):
        """Test creation of default backend configuration."""
        config_path = self.backend_dir / "config.json"
        
        with patch('pathlib.Path', return_value=config_path):
            self.validator._create_default_backend_config()
            
            assert config_path.exists()
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Verify all required fields are present
            for field in self.validator.required_backend_config_fields:
                assert field in config_data
            
            # Verify some specific values
            assert config_data["device"] == "auto"
            assert config_data["height"] == 512
            assert config_data["width"] == 512
    
    def test_create_default_package_json(self):
        """Test creation of default package.json."""
        package_path = self.frontend_dir / "package.json"
        
        with patch('pathlib.Path', return_value=package_path):
            self.validator._create_default_package_json()
            
            assert package_path.exists()
            
            with open(package_path, 'r', encoding='utf-8') as f:
                package_data = json.load(f)
            
            # Verify required fields are present
            assert "name" in package_data
            assert "version" in package_data
            assert "scripts" in package_data
            assert "dependencies" in package_data
            assert "devDependencies" in package_data
            
            # Verify required scripts
            assert "dev" in package_data["scripts"]
            assert "build" in package_data["scripts"]
            assert "preview" in package_data["scripts"]
    
    def test_create_default_vite_config(self):
        """Test creation of default vite.config.ts."""
        vite_path = self.frontend_dir / "vite.config.ts"
        
        with patch('pathlib.Path', return_value=vite_path):
            self.validator._create_default_vite_config()
            
            assert vite_path.exists()
            
            content = vite_path.read_text(encoding='utf-8')
            
            # Verify required patterns are present
            assert "import" in content
            assert "defineConfig" in content
            assert "export default" in content
            assert "plugins: [react()]" in content


if __name__ == "__main__":
    pytest.main([__file__])