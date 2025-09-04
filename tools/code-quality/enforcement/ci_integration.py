"""
CI/CD integration for automated quality checking.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import yaml
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class CIIntegration:
    """Manages CI/CD integration for code quality enforcement."""
    
    def __init__(self, project_root: Path): 
       """Initialize CI integration manager."""
        self.project_root = project_root
        self.github_workflows_dir = project_root / ".github" / "workflows"
        self.gitlab_ci_file = project_root / ".gitlab-ci.yml"
        self.jenkins_file = project_root / "Jenkinsfile"
        self.quality_config = project_root / "quality-config.yaml"
        
    def setup_github_actions(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Set up GitHub Actions workflow for quality enforcement.
        
        Args:
            config: Optional configuration for the workflow
            
        Returns:
            True if setup successful
        """
        try:
            if config is None:
                config = self._get_default_github_config()
            
            # Create workflows directory
            self.github_workflows_dir.mkdir(parents=True, exist_ok=True)
            
            # Write quality check workflow
            workflow_file = self.github_workflows_dir / "code-quality.yml"
            with open(workflow_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            logger.info("GitHub Actions workflow created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup GitHub Actions: {e}")
            return False
    
    def setup_gitlab_ci(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Set up GitLab CI pipeline for quality enforcement.
        
        Args:
            config: Optional configuration for the pipeline
            
        Returns:
            True if setup successful
        """
        try:
            if config is None:
                config = self._get_default_gitlab_config()
            
            with open(self.gitlab_ci_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            logger.info("GitLab CI pipeline created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup GitLab CI: {e}")
            return False
    
    def setup_jenkins(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Set up Jenkins pipeline for quality enforcement.
        
        Args:
            config: Optional configuration for the pipeline
            
        Returns:
            True if setup successful
        """
        try:
            if config is None:
                config = self._get_default_jenkins_config()
            
            with open(self.jenkins_file, 'w') as f:
                f.write(config)
            
            logger.info("Jenkins pipeline created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Jenkins: {e}")
            return False
    
    def create_quality_metrics_dashboard(self) -> Dict[str, Any]:
        """
        Create quality metrics tracking dashboard configuration.
        
        Returns:
            Dashboard configuration
        """
        dashboard_config = {
            'metrics': {
                'code_coverage': {
                    'threshold': 80,
                    'trend_tracking': True,
                    'alert_on_decrease': True
                },
                'code_quality_score': {
                    'threshold': 8.0,
                    'components': ['complexity', 'maintainability', 'reliability'],
                    'trend_tracking': True
                },
                'test_success_rate': {
                    'threshold': 95,
                    'trend_tracking': True,
                    'alert_on_decrease': True
                },
                'build_success_rate': {
                    'threshold': 90,
                    'trend_tracking': True,
                    'alert_on_decrease': True
                }
            },
            'reporting': {
                'frequency': 'daily',
                'recipients': ['team@example.com'],
                'format': 'html',
                'include_trends': True
            },
            'alerts': {
                'slack_webhook': None,
                'email_notifications': True,
                'threshold_violations': True
            }
        }
        
        try:
            with open(self.quality_config, 'w') as f:
                yaml.dump(dashboard_config, f, default_flow_style=False, sort_keys=False)
            
            logger.info("Quality metrics dashboard configuration created")
            
        except Exception as e:
            logger.error(f"Failed to create dashboard config: {e}")
        
        return dashboard_config
    
    def run_quality_checks(self, files: Optional[List[Path]] = None) -> Dict[str, Any]:
        """
        Run comprehensive quality checks for CI/CD.
        
        Args:
            files: Optional list of files to check
            
        Returns:
            Quality check results
        """
        results = {
            'success': True,
            'checks': {},
            'metrics': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Import quality checker
            from quality_checker import QualityChecker
            
            checker = QualityChecker()
            
            # Run quality checks
            if files:
                for file_path in files:
                    if file_path.suffix == '.py':
                        report = checker.check_quality(file_path)
                        results['checks'][str(file_path)] = {
                            'errors': report.errors,
                            'warnings': report.warnings,
                            'score': report.score
                        }
                        if report.errors > 0:
                            results['success'] = False
            else:
                report = checker.check_quality(self.project_root)
                results['checks']['overall'] = {
                    'errors': report.errors,
                    'warnings': report.warnings,
                    'score': report.score
                }
                if report.errors > 0:
                    results['success'] = False
            
            # Calculate metrics
            results['metrics'] = self._calculate_quality_metrics(results['checks'])
            
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"Quality check failed: {e}")
            logger.error(f"Quality check execution failed: {e}")
        
        return results
    
    def generate_quality_report(self, results: Dict[str, Any]) -> str:
        """
        Generate quality report for CI/CD output.
        
        Args:
            results: Quality check results
            
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("# Code Quality Report")
        report_lines.append("")
        
        # Overall status
        status = "✅ PASSED" if results['success'] else "❌ FAILED"
        report_lines.append(f"**Status:** {status}")
        report_lines.append("")
        
        # Metrics summary
        if 'metrics' in results:
            metrics = results['metrics']
            report_lines.append("## Quality Metrics")
            report_lines.append("")
            report_lines.append(f"- **Overall Score:** {metrics.get('overall_score', 'N/A')}/10")
            report_lines.append(f"- **Total Errors:** {metrics.get('total_errors', 0)}")
            report_lines.append(f"- **Total Warnings:** {metrics.get('total_warnings', 0)}")
            report_lines.append(f"- **Files Checked:** {metrics.get('files_checked', 0)}")
            report_lines.append("")
        
        # Detailed results
        if 'checks' in results:
            report_lines.append("## Detailed Results")
            report_lines.append("")
            
            for file_path, check_result in results['checks'].items():
                if check_result['errors'] > 0 or check_result['warnings'] > 0:
                    report_lines.append(f"### {file_path}")
                    if check_result['errors'] > 0:
                        report_lines.append(f"- ❌ Errors: {check_result['errors']}")
                    if check_result['warnings'] > 0:
                        report_lines.append(f"- ⚠️ Warnings: {check_result['warnings']}")
                    report_lines.append(f"- Score: {check_result['score']}/10")
                    report_lines.append("")
        
        # Errors and warnings
        if results.get('errors'):
            report_lines.append("## Errors")
            report_lines.append("")
            for error in results['errors']:
                report_lines.append(f"- ❌ {error}")
            report_lines.append("")
        
        if results.get('warnings'):
            report_lines.append("## Warnings")
            report_lines.append("")
            for warning in results['warnings']:
                report_lines.append(f"- ⚠️ {warning}")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def update_quality_metrics(self, results: Dict[str, Any]) -> bool:
        """
        Update quality metrics tracking.
        
        Args:
            results: Quality check results
            
        Returns:
            True if update successful
        """
        try:
            metrics_file = self.project_root / "quality-metrics.json"
            
            # Load existing metrics
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics_history = json.load(f)
            else:
                metrics_history = {'history': []}
            
            # Add current metrics
            current_metrics = {
                'timestamp': datetime.now().isoformat(),
                'metrics': results.get('metrics', {}),
                'success': results['success']
            }
            
            metrics_history['history'].append(current_metrics)
            
            # Keep only last 100 entries
            if len(metrics_history['history']) > 100:
                metrics_history['history'] = metrics_history['history'][-100:]
            
            # Save updated metrics
            with open(metrics_file, 'w') as f:
                json.dump(metrics_history, f, indent=2)
            
            logger.info("Quality metrics updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update quality metrics: {e}")
            return False
    
    def _get_default_github_config(self) -> Dict[str, Any]:
        """Get default GitHub Actions workflow configuration."""
        return {
            'name': 'Code Quality Check',
            'on': {
                'push': {
                    'branches': ['main', 'develop']
                },
                'pull_request': {
                    'branches': ['main', 'develop']
                }
            },
            'jobs': {
                'quality-check': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v3'
                        },
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v4',
                            'with': {
                                'python-version': '3.9'
                            }
                        },
                        {
                            'name': 'Install dependencies',
                            'run': 'pip install -r requirements.txt'
                        },
                        {
                            'name': 'Run quality checks',
                            'run': 'python -m tools.code_quality.cli check --fail-on-error'
                        },
                        {
                            'name': 'Generate quality report',
                            'run': 'python -m tools.code_quality.cli report --format=github'
                        }
                    ]
                }
            }
        }
    
    def _get_default_gitlab_config(self) -> Dict[str, Any]:
        """Get default GitLab CI pipeline configuration."""
        return {
            'stages': ['quality'],
            'quality-check': {
                'stage': 'quality',
                'image': 'python:3.9',
                'before_script': [
                    'pip install -r requirements.txt'
                ],
                'script': [
                    'python -m tools.code_quality.cli check --fail-on-error',
                    'python -m tools.code_quality.cli report --format=gitlab'
                ],
                'artifacts': {
                    'reports': {
                        'junit': 'quality-report.xml'
                    },
                    'paths': ['quality-report.html']
                },
                'only': ['main', 'develop', 'merge_requests']
            }
        }
    
    def _get_default_jenkins_config(self) -> str:
        """Get default Jenkins pipeline configuration."""
        return '''pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('Quality Check') {
            steps {
                sh 'python -m tools.code_quality.cli check --fail-on-error'
                sh 'python -m tools.code_quality.cli report --format=jenkins'
            }
            post {
                always {
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: '.',
                        reportFiles: 'quality-report.html',
                        reportName: 'Quality Report'
                    ])
                }
            }
        }
    }
    
    post {
        failure {
            emailext (
                subject: "Quality Check Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Quality check failed. Please check the report for details.",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}'''
    
    def _calculate_quality_metrics(self, checks: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality metrics from check results."""
        total_errors = 0
        total_warnings = 0
        total_score = 0
        files_checked = len(checks)
        
        for check_result in checks.values():
            total_errors += check_result.get('errors', 0)
            total_warnings += check_result.get('warnings', 0)
            total_score += check_result.get('score', 0)
        
        overall_score = total_score / files_checked if files_checked > 0 else 0
        
        return {
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'overall_score': round(overall_score, 2),
            'files_checked': files_checked,
            'quality_grade': self._get_quality_grade(overall_score)
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """Get quality grade based on score."""
        if score >= 9.0:
            return 'A+'
        elif score >= 8.0:
            return 'A'
        elif score >= 7.0:
            return 'B'
        elif score >= 6.0:
            return 'C'
        elif score >= 5.0:
            return 'D'
        else:
            return 'F'
    
    def get_ci_status(self) -> Dict[str, Any]:
        """Get status of CI/CD integration."""
        status = {
            'github_actions': self.github_workflows_dir.exists() and 
                            (self.github_workflows_dir / "code-quality.yml").exists(),
            'gitlab_ci': self.gitlab_ci_file.exists(),
            'jenkins': self.jenkins_file.exists(),
            'quality_config': self.quality_config.exists(),
            'metrics_tracking': (self.project_root / "quality-metrics.json").exists()
        }
        
        return status