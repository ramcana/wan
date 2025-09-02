"""
Test suite for the automated quality enforcement system.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import yaml
import json

from tools.code_quality.enforcement.pre_commit_hooks import PreCommitHookManager
from tools.code_quality.enforcement.ci_integration import CIIntegration
from tools.code_quality.enforcement.enforcement_cli import EnforcementCLI


class TestPreCommitHookManager:
    """Test pre-commit hook management."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.hook_manager = PreCommitHookManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_install_hooks_with_pre_commit(self):
        """Test installing hooks with pre-commit available."""
        with patch.object(self.hook_manager, '_is_pre_commit_available', return_value=True):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 0
                
                result = self.hook_manager.install_hooks()
                
                assert result is True
                assert self.hook_manager.config_file.exists()
                mock_run.assert_called()
    
    def test_install_manual_hooks(self):
        """Test installing manual hooks when pre-commit not available."""
        with patch.object(self.hook_manager, '_is_pre_commit_available', return_value=False):
            result = self.hook_manager.install_hooks()
            
            assert result is True
            hook_script = self.temp_dir / ".git" / "hooks" / "pre-commit"
            assert hook_script.exists()
            assert hook_script.is_file()
    
    def test_validate_config_valid(self):
        """Test validating valid configuration."""
        config = {
            'repos': [
                {
                    'repo': 'https://github.com/pre-commit/pre-commit-hooks',
                    'rev': 'v4.4.0',
                    'hooks': [{'id': 'trailing-whitespace'}]
                }
            ]
        }
        
        with open(self.hook_manager.config_file, 'w') as f:
            yaml.dump(config, f)
        
        result = self.hook_manager.validate_config()
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
    
    def test_validate_config_invalid(self):
        """Test validating invalid configuration."""
        config = {'invalid': 'config'}
        
        with open(self.hook_manager.config_file, 'w') as f:
            yaml.dump(config, f)
        
        result = self.hook_manager.validate_config()
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
    
    def test_run_hooks_success(self):
        """Test running hooks successfully."""
        with patch.object(self.hook_manager, '_is_pre_commit_available', return_value=True):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "All hooks passed"
                mock_run.return_value.stderr = ""
                
                result = self.hook_manager.run_hooks()
                
                assert result['success'] is True
                assert 'output' in result
    
    def test_get_hook_status(self):
        """Test getting hook status."""
        status = self.hook_manager.get_hook_status()
        
        assert 'installed' in status
        assert 'pre_commit_available' in status
        assert 'config_exists' in status
        assert 'config_valid' in status


class TestCIIntegration:
    """Test CI/CD integration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.ci_integration = CIIntegration(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_setup_github_actions(self):
        """Test setting up GitHub Actions workflow."""
        result = self.ci_integration.setup_github_actions()
        
        assert result is True
        workflow_file = self.temp_dir / ".github" / "workflows" / "code-quality.yml"
        assert workflow_file.exists()
        
        # Validate workflow content
        with open(workflow_file, 'r') as f:
            workflow = yaml.safe_load(f)
        
        assert 'name' in workflow
        assert 'on' in workflow
        assert 'jobs' in workflow
    
    def test_setup_gitlab_ci(self):
        """Test setting up GitLab CI pipeline."""
        result = self.ci_integration.setup_gitlab_ci()
        
        assert result is True
        ci_file = self.temp_dir / ".gitlab-ci.yml"
        assert ci_file.exists()
        
        # Validate CI content
        with open(ci_file, 'r') as f:
            ci_config = yaml.safe_load(f)
        
        assert 'stages' in ci_config
        assert 'quality-check' in ci_config
    
    def test_setup_jenkins(self):
        """Test setting up Jenkins pipeline."""
        result = self.ci_integration.setup_jenkins()
        
        assert result is True
        jenkins_file = self.temp_dir / "Jenkinsfile"
        assert jenkins_file.exists()
        
        # Validate Jenkins content
        with open(jenkins_file, 'r') as f:
            content = f.read()
        
        assert 'pipeline' in content
        assert 'Quality Check' in content
    
    def test_create_quality_metrics_dashboard(self):
        """Test creating quality metrics dashboard."""
        config = self.ci_integration.create_quality_metrics_dashboard()
        
        assert isinstance(config, dict)
        assert 'metrics' in config
        assert 'reporting' in config
        assert 'alerts' in config
        
        # Check config file was created
        assert self.ci_integration.quality_config.exists()
    
    def test_run_quality_checks(self):
        """Test running quality checks."""
        # Create a test Python file
        test_file = self.temp_dir / "test.py"
        test_file.write_text("print('hello world')")
        
        with patch('tools.code_quality.quality_checker.QualityChecker') as mock_checker:
            mock_report = Mock()
            mock_report.errors = 0
            mock_report.warnings = 1
            mock_report.score = 8.5
            
            mock_checker.return_value.check_quality.return_value = mock_report
            
            result = self.ci_integration.run_quality_checks([test_file])
            
            assert result['success'] is True
            assert 'checks' in result
            assert 'metrics' in result
    
    def test_generate_quality_report(self):
        """Test generating quality report."""
        results = {
            'success': True,
            'checks': {
                'test.py': {'errors': 0, 'warnings': 1, 'score': 8.5}
            },
            'metrics': {
                'overall_score': 8.5,
                'total_errors': 0,
                'total_warnings': 1,
                'files_checked': 1
            }
        }
        
        report = self.ci_integration.generate_quality_report(results)
        
        assert isinstance(report, str)
        assert "✅ PASSED" in report
        assert "8.5/10" in report
        assert "test.py" in report
    
    def test_update_quality_metrics(self):
        """Test updating quality metrics."""
        results = {
            'success': True,
            'metrics': {
                'overall_score': 8.5,
                'total_errors': 0,
                'total_warnings': 1
            }
        }
        
        result = self.ci_integration.update_quality_metrics(results)
        
        assert result is True
        
        metrics_file = self.temp_dir / "quality-metrics.json"
        assert metrics_file.exists()
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        assert 'history' in metrics
        assert len(metrics['history']) == 1
    
    def test_get_ci_status(self):
        """Test getting CI status."""
        status = self.ci_integration.get_ci_status()
        
        assert 'github_actions' in status
        assert 'gitlab_ci' in status
        assert 'jenkins' in status
        assert 'quality_config' in status
        assert 'metrics_tracking' in status


class TestEnforcementCLI:
    """Test enforcement CLI."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cli = EnforcementCLI(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_setup_hooks(self):
        """Test setting up hooks via CLI."""
        with patch.object(self.cli.hook_manager, 'install_hooks', return_value=True):
            result = self.cli.setup_hooks()
            
            assert result is True
    
    def test_setup_ci_github(self):
        """Test setting up GitHub CI via CLI."""
        with patch.object(self.cli.ci_integration, 'setup_github_actions', return_value=True):
            result = self.cli.setup_ci('github')
            
            assert result is True
    
    def test_setup_ci_unsupported(self):
        """Test setting up unsupported CI platform."""
        result = self.cli.setup_ci('unsupported')
        
        assert result is False
    
    def test_run_checks_success(self):
        """Test running checks successfully."""
        with patch.object(self.cli.hook_manager, 'run_hooks') as mock_hooks:
            with patch.object(self.cli.ci_integration, 'run_quality_checks') as mock_ci:
                mock_hooks.return_value = {'success': True, 'failures': []}
                mock_ci.return_value = {'success': True, 'errors': []}
                
                result = self.cli.run_checks()
                
                assert result is True
    
    def test_run_checks_failure(self):
        """Test running checks with failures."""
        with patch.object(self.cli.hook_manager, 'run_hooks') as mock_hooks:
            with patch.object(self.cli.ci_integration, 'run_quality_checks') as mock_ci:
                mock_hooks.return_value = {'success': False, 'failures': ['Hook failed']}
                mock_ci.return_value = {'success': False, 'errors': ['CI failed']}
                
                result = self.cli.run_checks()
                
                assert result is False
    
    def test_create_dashboard(self):
        """Test creating dashboard via CLI."""
        with patch.object(self.cli.ci_integration, 'create_quality_metrics_dashboard') as mock_dashboard:
            mock_dashboard.return_value = {'metrics': {}}
            
            result = self.cli.create_dashboard()
            
            assert result is True
    
    def test_generate_report_console(self):
        """Test generating console report."""
        with patch.object(self.cli.ci_integration, 'run_quality_checks') as mock_checks:
            with patch.object(self.cli.ci_integration, 'generate_quality_report') as mock_report:
                mock_checks.return_value = {'success': True}
                mock_report.return_value = "Quality Report"
                
                # Should not raise exception
                self.cli.generate_report('console')
    
    def test_generate_report_html(self):
        """Test generating HTML report."""
        with patch.object(self.cli.ci_integration, 'run_quality_checks') as mock_checks:
            mock_checks.return_value = {
                'success': True,
                'checks': {'test.py': {'errors': 0, 'warnings': 1, 'score': 8.5}},
                'metrics': {'overall_score': 8.5, 'total_errors': 0, 'total_warnings': 1, 'files_checked': 1}
            }
            
            self.cli.generate_report('html')
            
            report_file = self.temp_dir / "quality-report.html"
            assert report_file.exists()
            
            content = report_file.read_text()
            assert "Code Quality Report" in content
            assert "8.5/10" in content


class TestIntegration:
    """Integration tests for the enforcement system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create a mock Git repository
        git_dir = self.temp_dir / ".git"
        git_dir.mkdir()
        (git_dir / "hooks").mkdir()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_full_enforcement_setup(self):
        """Test complete enforcement system setup."""
        cli = EnforcementCLI(self.temp_dir)
        
        # Setup hooks
        with patch.object(cli.hook_manager, '_is_pre_commit_available', return_value=False):
            hooks_result = cli.setup_hooks()
            assert hooks_result is True
        
        # Setup CI
        ci_result = cli.setup_ci('github')
        assert ci_result is True
        
        # Create dashboard
        dashboard_result = cli.create_dashboard()
        assert dashboard_result is True
        
        # Check status
        hook_status = cli.hook_manager.get_hook_status()
        ci_status = cli.ci_integration.get_ci_status()
        
        assert hook_status['installed'] is True
        assert ci_status['github_actions'] is True
        assert ci_status['quality_config'] is True
    
    def test_enforcement_workflow(self):
        """Test complete enforcement workflow."""
        # Create test files
        test_file = self.temp_dir / "test.py"
        test_file.write_text("""
def hello_world():
    print("Hello, World!")
    return True
""")
        
        cli = EnforcementCLI(self.temp_dir)
        
        # Mock quality checker
        with patch('tools.code_quality.quality_checker.QualityChecker') as mock_checker:
            mock_report = Mock()
            mock_report.errors = 0
            mock_report.warnings = 0
            mock_report.score = 9.0
            
            mock_checker.return_value.check_quality.return_value = mock_report
            
            # Run checks
            result = cli.run_checks([str(test_file)])
            assert result is True
            
            # Generate report
            cli.generate_report('json')
            
            report_file = self.temp_dir / "quality-report.json"
            assert report_file.exists()
            
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            
            assert report_data['success'] is True
            assert 'metrics' in report_data


if __name__ == '__main__':
    pytest.main([__file__])