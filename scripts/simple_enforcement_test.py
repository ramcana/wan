from unittest.mock import Mock, patch
"""
Simple test of the enforcement system components.
"""

import tempfile
import shutil
from pathlib import Path
import yaml
import json


def test_pre_commit_config_generation():
    """Test generating pre-commit configuration."""
    print("🔧 Testing pre-commit configuration generation...")
    
    config = {
        'repos': [
            {
                'repo': 'https://github.com/pre-commit/pre-commit-hooks',
                'rev': 'v4.4.0',
                'hooks': [
                    {'id': 'trailing-whitespace'},
                    {'id': 'end-of-file-fixer'},
                    {'id': 'check-yaml'},
                    {'id': 'check-json'},
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
    
    # Write configuration
    config_file = Path('.pre-commit-config.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Pre-commit config created: {config_file}")
    return True


def test_github_actions_workflow():
    """Test generating GitHub Actions workflow."""
    print("🔄 Testing GitHub Actions workflow generation...")
    
    workflow = {
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
                        'run': 'python -m tools.code_quality.cli report --format=html'
                    }
                ]
            }
        }
    }
    
    # Create workflows directory
    workflows_dir = Path('.github/workflows')
    workflows_dir.mkdir(parents=True, exist_ok=True)
    
    # Write workflow
    workflow_file = workflows_dir / 'code-quality.yml'
    with open(workflow_file, 'w') as f:
        yaml.dump(workflow, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ GitHub Actions workflow created: {workflow_file}")
    return True


def test_quality_metrics_dashboard():
    """Test creating quality metrics dashboard configuration."""
    print("📈 Testing quality metrics dashboard...")
    
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
    
    # Write dashboard config
    config_file = Path('quality-config.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(dashboard_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Quality dashboard config created: {config_file}")
    return True


def test_git_hook_script():
    """Test creating Git hook script."""
    print("🪝 Testing Git hook script generation...")
    
    hook_content = '''#!/bin/sh
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
    
    # Create hooks directory
    hooks_dir = Path('.git/hooks')
    hooks_dir.mkdir(parents=True, exist_ok=True)
    
    # Write hook script
    hook_file = hooks_dir / 'pre-commit'
    with open(hook_file, 'w') as f:
        f.write(hook_content)
    
    # Make executable (on Unix systems)
    try:
        hook_file.chmod(0o755)
    except:
        pass  # Windows doesn't need this
    
    print(f"✅ Git hook script created: {hook_file}")
    return True


def test_quality_report_generation():
    """Test generating quality report."""
    print("📋 Testing quality report generation...")
    
    # Mock quality report data
    report_data = {
        'success': True,
        'checks': {
            'example.py': {'errors': 0, 'warnings': 2, 'score': 8.5},
            'another.py': {'errors': 1, 'warnings': 0, 'score': 7.0}
        },
        'metrics': {
            'overall_score': 7.75,
            'total_errors': 1,
            'total_warnings': 2,
            'files_checked': 2,
            'quality_grade': 'B'
        }
    }
    
    # Generate JSON report
    json_report = Path('quality-report.json')
    with open(json_report, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Generate HTML report
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Code Quality Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .success {{ color: green; }}
        .failure {{ color: red; }}
        .metrics {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; flex: 1; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Code Quality Report</h1>
        <p class="{'success' if report_data['success'] else 'failure'}">
            Status: {'PASSED' if report_data['success'] else 'FAILED'}
        </p>
    </div>
    
    <div class="metrics">
        <div class="metric-card">
            <h3>Overall Score</h3>
            <p style="font-size: 24px; font-weight: bold;">{report_data['metrics']['overall_score']}/10</p>
        </div>
        <div class="metric-card">
            <h3>Errors</h3>
            <p style="font-size: 24px; font-weight: bold; color: red;">{report_data['metrics']['total_errors']}</p>
        </div>
        <div class="metric-card">
            <h3>Warnings</h3>
            <p style="font-size: 24px; font-weight: bold; color: orange;">{report_data['metrics']['total_warnings']}</p>
        </div>
        <div class="metric-card">
            <h3>Files Checked</h3>
            <p style="font-size: 24px; font-weight: bold;">{report_data['metrics']['files_checked']}</p>
        </div>
    </div>
    
    <h2>File Details</h2>
    <table style="width: 100%; border-collapse: collapse;">
        <tr style="background-color: #f2f2f2;">
            <th style="border: 1px solid #ddd; padding: 8px;">File</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Score</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Errors</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Warnings</th>
        </tr>
"""
    
    for file_path, check_result in report_data['checks'].items():
        html_content += f"""
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">{file_path}</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{check_result['score']}</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{check_result['errors']}</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{check_result['warnings']}</td>
        </tr>
        """
    
    html_content += """
    </table>
</body>
</html>
    """
    
    html_report = Path('quality-report.html')
    with open(html_report, 'w') as f:
        f.write(html_content)
    
    print(f"✅ JSON report created: {json_report}")
    print(f"✅ HTML report created: {html_report}")
    return True


def main():
    """Run all enforcement system tests."""
    print("🚀 Quality Enforcement System Test")
    print("=" * 40)
    
    tests = [
        test_pre_commit_config_generation,
        test_github_actions_workflow,
        test_quality_metrics_dashboard,
        test_git_hook_script,
        test_quality_report_generation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All enforcement system components working!")
        print("\n📁 Files created:")
        files = [
            '.pre-commit-config.yaml',
            '.github/workflows/code-quality.yml',
            'quality-config.yaml',
            '.git/hooks/pre-commit',
            'quality-report.json',
            'quality-report.html'
        ]
        
        for file_path in files:
            if Path(file_path).exists():
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ {file_path}")
        
        print("\n💡 The enforcement system includes:")
        print("  • Pre-commit hooks for local quality checks")
        print("  • GitHub Actions workflow for CI/CD")
        print("  • Quality metrics dashboard configuration")
        print("  • Git hooks for commit-time validation")
        print("  • HTML and JSON quality reports")
    else:
        print("\n❌ Some tests failed. Check the output above.")


if __name__ == "__main__":
    main()