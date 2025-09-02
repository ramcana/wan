"""
Simple integration tests for Configuration Validator.
Tests actual functionality without complex mocking.
"""

import json
import tempfile
import shutil
from pathlib import Path
import pytest

# Add the scripts directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from scripts.startup_manager.environment_validator import ConfigurationValidator, ValidationStatus
    ConfigurationValidator,
    ValidationStatus
)


class TestConfigurationIntegration:
    """Integration tests for ConfigurationValidator using real files."""
    
    def setup_method(self):
        """Set up test fixtures with temporary directory."""
        self.validator = ConfigurationValidator()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.backend_dir = self.temp_dir / "backend"
        self.frontend_dir = self.temp_dir / "frontend"
        self.backend_dir.mkdir(parents=True, exist_ok=True)
        self.frontend_dir.mkdir(parents=True, exist_ok=True)
        
        # Change to temp directory for tests
        self.original_cwd = Path.cwd()
        import os
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import os
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_default_backend_config_integration(self):
        """Test creating default backend config in real filesystem."""
        # Ensure config doesn't exist
        config_path = Path("backend/config.json")
        assert not config_path.exists()
        
        # Create default config
        self.validator._create_default_backend_config()
        
        # Verify file was created
        assert config_path.exists()
        
        # Verify content
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Check all required fields are present
        for field in self.validator.required_backend_config_fields:
            assert field in config_data
        
        # Check specific values
        assert config_data["device"] == "auto"
        assert config_data["height"] == 512
        assert config_data["width"] == 512
        assert isinstance(config_data["max_memory"], (int, float))
    
    def test_create_default_package_json_integration(self):
        """Test creating default package.json in real filesystem."""
        # Ensure package.json doesn't exist
        package_path = Path("frontend/package.json")
        assert not package_path.exists()
        
        # Create default package.json
        self.validator._create_default_package_json()
        
        # Verify file was created
        assert package_path.exists()
        
        # Verify content
        with open(package_path, 'r', encoding='utf-8') as f:
            package_data = json.load(f)
        
        # Check required fields
        assert "name" in package_data
        assert "version" in package_data
        assert "scripts" in package_data
        assert "dependencies" in package_data
        assert "devDependencies" in package_data
        
        # Check required scripts
        assert "dev" in package_data["scripts"]
        assert "build" in package_data["scripts"]
        assert "preview" in package_data["scripts"]
    
    def test_create_default_vite_config_integration(self):
        """Test creating default vite.config.ts in real filesystem."""
        # Ensure vite.config.ts doesn't exist
        vite_path = Path("frontend/vite.config.ts")
        assert not vite_path.exists()
        
        # Create default vite config
        self.validator._create_default_vite_config()
        
        # Verify file was created
        assert vite_path.exists()
        
        # Verify content
        content = vite_path.read_text(encoding='utf-8')
        
        # Check required patterns
        assert "import" in content
        assert "defineConfig" in content
        assert "export default" in content
        assert "plugins: [react()]" in content
    
    def test_validate_backend_config_with_real_files(self):
        """Test backend config validation with real files."""
        # Create a valid config file
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
        
        config_path = Path("backend/config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
        # Validate the config
        issues = self.validator.validate_backend_config()
        
        # Should have no critical issues
        critical_issues = [issue for issue in issues if issue.status == ValidationStatus.FAILED]
        assert len(critical_issues) == 0
    
    def test_validate_backend_config_missing_file(self):
        """Test backend config validation when file is missing."""
        # Ensure no config file exists
        config_path = Path("backend/config.json")
        if config_path.exists():
            config_path.unlink()
        
        # Validate (should find missing file)
        issues = self.validator.validate_backend_config()
        
        # Should find missing file issue
        missing_file_issue = next((issue for issue in issues if issue.issue_type == "missing_config_file"), None)
        assert missing_file_issue is not None
        assert missing_file_issue.status == ValidationStatus.FAILED
        assert missing_file_issue.auto_fixable
    
    def test_validate_backend_config_invalid_json(self):
        """Test backend config validation with invalid JSON."""
        # Create invalid JSON file
        config_path = Path("backend/config.json")
        config_path.write_text("{ invalid json }", encoding='utf-8')
        
        # Validate (should find JSON error)
        issues = self.validator.validate_backend_config()
        
        # Should find JSON issue
        json_issue = next((issue for issue in issues if issue.issue_type == "invalid_json"), None)
        assert json_issue is not None
        assert json_issue.status == ValidationStatus.FAILED
        assert json_issue.auto_fixable
    
    def test_validate_backend_config_missing_fields(self):
        """Test backend config validation with missing required fields."""
        # Create config with missing fields
        config_data = {
            "model_path": "models/",
            "device": "auto"
            # Missing other required fields
        }
        
        config_path = Path("backend/config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
        # Validate (should find missing fields)
        issues = self.validator.validate_backend_config()
        
        # Should find missing fields issue
        missing_fields_issue = next((issue for issue in issues if issue.issue_type == "missing_required_fields"), None)
        assert missing_fields_issue is not None
        assert missing_fields_issue.status == ValidationStatus.FAILED
        assert missing_fields_issue.auto_fixable
        assert "max_memory" in missing_fields_issue.details["missing_fields"]
    
    def test_auto_repair_missing_backend_config(self):
        """Test auto-repair of missing backend config file."""
        # Ensure no config exists
        config_path = Path("backend/config.json")
        if config_path.exists():
            config_path.unlink()
        
        # Get validation issues
        issues = self.validator.validate_backend_config()
        
        # Apply auto-repair
        repairs = self.validator.auto_repair_config(issues)
        
        # Should have applied repair
        assert len(repairs) > 0
        assert any("Created default backend config.json" in repair for repair in repairs)
        
        # Config file should now exist
        assert config_path.exists()
        
        # Validate the created config
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Should have all required fields
        for field in self.validator.required_backend_config_fields:
            assert field in config_data
    
    def test_validate_frontend_config_with_real_files(self):
        """Test frontend config validation with real files."""
        # Create valid package.json
        package_data = {
            "name": "test-app",
            "version": "1.0.0",
            "scripts": {
                "dev": "vite",
                "build": "vite build",
                "preview": "vite preview"
            },
            "dependencies": {"react": "^18.0.0"},
            "devDependencies": {"vite": "^4.0.0"}
        }
        
        package_path = Path("frontend/package.json")
        with open(package_path, 'w', encoding='utf-8') as f:
            json.dump(package_data, f, indent=2)
        
        # Create valid vite.config.ts
        vite_content = '''import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()]
})
'''
        vite_path = Path("frontend/vite.config.ts")
        vite_path.write_text(vite_content, encoding='utf-8')
        
        # Validate frontend config
        issues = self.validator.validate_frontend_config()
        
        # Should have no critical issues
        critical_issues = [issue for issue in issues if issue.status == ValidationStatus.FAILED]
        assert len(critical_issues) == 0


if __name__ == "__main__":
    pytest.main([__file__])