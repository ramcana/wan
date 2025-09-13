"""
Pre-commit hook management for automated quality enforcement.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import yaml
import json

logger = logging.getLogger(__name__)


class PreCommitHookManager:
    """Manages pre-commit hooks for code quality enforcement."""
    
    def __init__(self, project_root: Path):
        """Initialize hook manager."""
        self.project_root = project_root
        self.hooks_dir = project_root / ".git" / "hooks"
        self.config_file = project_root / ".pre-commit-config.yaml"
        
    def install_hooks(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Install pre-commit hooks for quality enforcement.
        
        Args:
            config: Optional configuration for hooks
            
        Returns:
            True if installation successful
        """
        try:
            # Create default configuration if none provided
            if config is None:
                config = self._get_default_config()
            
            # Write pre-commit configuration
            self._write_pre_commit_config(config)
            
            # Install pre-commit hooks
            if self._is_pre_commit_available():
                result = subprocess.run([
                    sys.executable, '-m', 'pre_commit', 'install'
                ], cwd=self.project_root, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("Pre-commit hooks installed successfully")
                    return True
                else:
                    logger.error(f"Failed to install pre-commit hooks: {result.stderr}")
            else:
                # Install manual hooks if pre-commit not available
                return self._install_manual_hooks()
            
        except Exception as e:
            logger.error(f"Failed to install pre-commit hooks: {e}")
        
        return False
    
    def uninstall_hooks(self) -> bool:
        """Uninstall pre-commit hooks."""
        try:
            if self._is_pre_commit_available():
                result = subprocess.run([
                    sys.executable, '-m', 'pre_commit', 'uninstall'
                ], cwd=self.project_root, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("Pre-commit hooks uninstalled successfully")
                    return True
            else:
                # Remove manual hooks
                return self._remove_manual_hooks()
                
        except Exception as e:
            logger.error(f"Failed to uninstall pre-commit hooks: {e}")
        
        return False
    
    def run_hooks(self, files: Optional[List[Path]] = None) -> Dict[str, Any]:
        """
        Run pre-commit hooks manually.
        
        Args:
            files: Optional list of files to check
            
        Returns:
            Results of hook execution
        """
        results = {
            'success': False,
            'hooks_run': [],
            'failures': [],
            'warnings': []
        }
        
        try:
            if self._is_pre_commit_available():
                cmd = [sys.executable, '-m', 'pre_commit', 'run']
                if files:
                    cmd.extend(['--files'] + [str(f) for f in files])
                else:
                    cmd.append('--all-files')
                
                result = subprocess.run(cmd, cwd=self.project_root, 
                                      capture_output=True, text=True)
                
                results['success'] = result.returncode == 0
                results['output'] = result.stdout
                results['errors'] = result.stderr
                
            else:
                # Run manual hooks
                results = self._run_manual_hooks(files)
            
        except Exception as e:
            logger.error(f"Failed to run pre-commit hooks: {e}")
            results['failures'].append(str(e))
        
        return results
    
    def update_hooks(self) -> bool:
        """Update pre-commit hooks to latest versions."""
        try:
            if self._is_pre_commit_available():
                result = subprocess.run([
                    sys.executable, '-m', 'pre_commit', 'autoupdate'
                ], cwd=self.project_root, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("Pre-commit hooks updated successfully")
                    return True
                else:
                    logger.error(f"Failed to update hooks: {result.stderr}")
            
        except Exception as e:
            logger.error(f"Failed to update pre-commit hooks: {e}")
        
        return False
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate pre-commit configuration."""
        validation_result = {
            'valid': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            if not self.config_file.exists():
                validation_result['errors'].append("Pre-commit config file not found")
                return validation_result
            
            # Load and validate YAML
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Basic validation
            if not isinstance(config, dict):
                validation_result['errors'].append("Config must be a dictionary")
                return validation_result
            
            if 'repos' not in config:
                validation_result['errors'].append("Config must have 'repos' key")
                return validation_result
            
            # Validate repositories
            for i, repo in enumerate(config['repos']):
                if not isinstance(repo, dict):
                    validation_result['errors'].append(f"Repo {i} must be a dictionary")
                    continue
                
                if 'repo' not in repo:
                    validation_result['errors'].append(f"Repo {i} missing 'repo' key")
                
                if 'hooks' not in repo:
                    validation_result['errors'].append(f"Repo {i} missing 'hooks' key")
            
            validation_result['valid'] = len(validation_result['errors']) == 0
            
        except yaml.YAMLError as e:
            validation_result['errors'].append(f"Invalid YAML: {e}")
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {e}")
        
        return validation_result
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default pre-commit configuration."""
        return {
            'repos': [
                {
                    'repo': 'https://github.com/pre-commit/pre-commit-hooks',
                    'rev': 'v4.4.0',
                    'hooks': [
                        {'id': 'trailing-whitespace'},
                        {'id': 'end-of-file-fixer'},
                        {'id': 'check-yaml'},
                        {'id': 'check-json'},
                        {'id': 'check-merge-conflict'},
                        {'id': 'check-added-large-files'},
                    ]
                },
                {
                    'repo': 'https://github.com/psf/black',
                    'rev': '23.3.0',
                    'hooks': [
                        {
                            'id': 'black',
                            'language_version': 'python3',
                            'args': ['--line-length=88']
                        }
                    ]
                },
                {
                    'repo': 'https://github.com/pycqa/isort',
                    'rev': '5.12.0',
                    'hooks': [
                        {
                            'id': 'isort',
                            'args': ['--profile=black']
                        }
                    ]
                },
                {
                    'repo': 'https://github.com/pycqa/flake8',
                    'rev': '6.0.0',
                    'hooks': [
                        {
                            'id': 'flake8',
                            'args': ['--max-line-length=88', '--ignore=E203,W503']
                        }
                    ]
                },
                {
                    'repo': 'local',
                    'hooks': [
                        {
                            'id': 'code-quality-check',
                            'name': 'Code Quality Check',
                            'entry': 'python -m tools.code_quality.cli check',
                            'language': 'system',
                            'files': r'\.py$',
                            'args': ['--fail-on-error']
                        }
                    ]
                }
            ]
        }
    
    def _write_pre_commit_config(self, config: Dict[str, Any]) -> None:
        """Write pre-commit configuration to file."""
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def _is_pre_commit_available(self) -> bool:
        """Check if pre-commit is available."""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pre_commit', '--version'
            ], capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def _install_manual_hooks(self) -> bool:
        """Install manual Git hooks when pre-commit is not available."""
        try:
            if not self.hooks_dir.exists():
                self.hooks_dir.mkdir(parents=True)
            
            # Create pre-commit hook script
            hook_script = self.hooks_dir / "pre-commit"
            hook_content = f'''#!/bin/sh
# Code quality pre-commit hook

echo "Running code quality checks..."

# Get list of Python files being committed
python_files=$(git diff --cached --name-only --diff-filter=ACM | grep '\\.py$')

if [ -z "$python_files" ]; then
    echo "No Python files to check"
    exit 0
fi

# Run code quality check
python -m tools.code_quality.cli check $python_files --fail-on-error

if [ $? -ne 0 ]; then
    echo "Code quality check failed. Commit aborted."
    echo "Run 'python -m tools.code_quality.cli fix <files>' to auto-fix issues"
    exit 1
fi

echo "Code quality check passed"
exit 0
'''
            
            with open(hook_script, 'w') as f:
                f.write(hook_content)
            
            # Make executable
            hook_script.chmod(0o755)
            
            logger.info("Manual pre-commit hook installed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to install manual hooks: {e}")
            return False
    
    def _remove_manual_hooks(self) -> bool:
        """Remove manual Git hooks."""
        try:
            hook_script = self.hooks_dir / "pre-commit"
            if hook_script.exists():
                hook_script.unlink()
                logger.info("Manual pre-commit hook removed")
            return True
        except Exception as e:
            logger.error(f"Failed to remove manual hooks: {e}")
            return False
    
    def _run_manual_hooks(self, files: Optional[List[Path]] = None) -> Dict[str, Any]:
        """Run manual quality checks."""
        results = {
            'success': True,
            'hooks_run': ['manual-quality-check'],
            'failures': [],
            'warnings': []
        }
        
        try:
            # Import quality checker
            from quality_checker import QualityChecker
            
            checker = QualityChecker()
            
            if files:
                # Check specific files
                for file_path in files:
                    if file_path.suffix == '.py':
                        report = checker.check_quality(file_path)
                        if report.errors > 0:
                            results['success'] = False
                            results['failures'].append(f"{file_path}: {report.errors} errors")
                        elif report.warnings > 0:
                            results['warnings'].append(f"{file_path}: {report.warnings} warnings")
            else:
                # Check all Python files in project
                report = checker.check_quality(self.project_root)
                if report.errors > 0:
                    results['success'] = False
                    results['failures'].append(f"Project has {report.errors} errors")
                elif report.warnings > 0:
                    results['warnings'].append(f"Project has {report.warnings} warnings")
            
        except Exception as e:
            results['success'] = False
            results['failures'].append(f"Manual hook execution failed: {e}")
        
        return results
    
    def get_hook_status(self) -> Dict[str, Any]:
        """Get status of pre-commit hooks."""
        status = {
            'installed': False,
            'pre_commit_available': self._is_pre_commit_available(),
            'config_exists': self.config_file.exists(),
            'config_valid': False,
            'hooks': []
        }
        
        # Check if hooks are installed
        if self._is_pre_commit_available():
            try:
                result = subprocess.run([
                    sys.executable, '-m', 'pre_commit', 'run', '--help'
                ], cwd=self.project_root, capture_output=True, text=True)
                status['installed'] = result.returncode == 0
            except Exception:
                pass
        else:
            # Check for manual hooks
            hook_script = self.hooks_dir / "pre-commit"
            status['installed'] = hook_script.exists()
        
        # Validate configuration
        if status['config_exists']:
            validation = self.validate_config()
            status['config_valid'] = validation['valid']
            status['config_errors'] = validation['errors']
        
        return status
