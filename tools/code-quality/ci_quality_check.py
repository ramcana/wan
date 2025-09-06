#!/usr/bin/env python3
"""
Simplified code quality checker for CI environments.
This version has minimal dependencies and runs basic quality checks.
"""

import subprocess
import sys
import json
from pathlib import Path
from typing import List, Dict, Any


def run_command(cmd: List[str], cwd: Path = None) -> Dict[str, Any]:
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=cwd,
            timeout=300,  # 5 minute timeout
            encoding='utf-8',
            errors='replace'  # Replace problematic characters
        )
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout or '',
            'stderr': result.stderr or '',
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': 'Command timed out',
            'returncode': 124
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'returncode': 1
        }


def find_python_files(path: Path) -> List[Path]:
    """Find all Python files in the given path."""
    if path.is_file() and path.suffix == '.py':
        return [path]
    
    python_files = []
    if path.is_dir():
        for py_file in path.rglob('*.py'):
            # Skip common directories that shouldn't be checked
            if any(part.startswith('.') for part in py_file.parts):
                continue
            if any(part in ['__pycache__', 'node_modules', 'venv', 'env'] for part in py_file.parts):
                continue
            python_files.append(py_file)
    
    return python_files


def run_black_check(path: Path) -> Dict[str, Any]:
    """Run black formatting check."""
    print("Running Black formatting check...")
    
    # Check if black is available
    check_result = run_command(['black', '--version'])
    if not check_result['success']:
        return {
            'tool': 'black',
            'passed': False,
            'output': 'Black is not installed or not available',
            'issues_count': 1
        }
    
    result = run_command(['black', '--check', '--diff', str(path)])
    
    return {
        'tool': 'black',
        'passed': result['success'],
        'output': result['stdout'] + result['stderr'],
        'issues_count': 0 if result['success'] else result['stdout'].count('would reformat')
    }


def run_ruff_check(path: Path) -> Dict[str, Any]:
    """Run ruff linting check."""
    print("Running Ruff linting check...")
    
    # Check if ruff is available
    check_result = run_command(['ruff', '--version'])
    if not check_result['success']:
        return {
            'tool': 'ruff',
            'passed': False,
            'output': 'Ruff is not installed or not available',
            'issues_count': 1
        }
    
    result = run_command(['ruff', 'check', str(path), '--output-format=json'])
    
    issues_count = 0
    if not result['success'] and result['stdout']:
        try:
            issues = json.loads(result['stdout'])
            issues_count = len(issues)
        except json.JSONDecodeError:
            # Fallback to counting lines if JSON parsing fails
            issues_count = len([line for line in result['stdout'].split('\n') if line.strip()])
    
    return {
        'tool': 'ruff',
        'passed': result['success'],
        'output': result['stdout'] + result['stderr'],
        'issues_count': issues_count
    }


def run_pytest_check(path: Path) -> Dict[str, Any]:
    """Run pytest to check if tests pass."""
    print("Running pytest...")
    
    # Check if pytest is available
    check_result = run_command(['pytest', '--version'])
    if not check_result['success']:
        return {
            'tool': 'pytest',
            'passed': False,
            'output': 'Pytest is not installed or not available',
            'issues_count': 1
        }
    
    # Look for test directories
    test_dirs = []
    for test_dir in ['tests', 'test']:
        test_path = path / test_dir
        if test_path.exists():
            test_dirs.append(str(test_path))
    
    if not test_dirs:
        # Look for test files in the current directory
        test_files = list(path.rglob('test_*.py')) + list(path.rglob('*_test.py'))
        if test_files:
            test_dirs = [str(path)]
    
    if not test_dirs:
        return {
            'tool': 'pytest',
            'passed': True,
            'output': 'No tests found',
            'issues_count': 0
        }
    
    result = run_command(['pytest', '--tb=short', '-v'] + test_dirs)
    
    return {
        'tool': 'pytest',
        'passed': result['success'],
        'output': result['stdout'] + result['stderr'],
        'issues_count': 0 if result['success'] else 1
    }


def run_basic_syntax_check(python_files: List[Path]) -> Dict[str, Any]:
    """Run basic Python syntax check."""
    print("Running Python syntax check...")
    
    syntax_errors = []
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            compile(content, str(py_file), 'exec')
        except SyntaxError as e:
            syntax_errors.append(f"{py_file}:{e.lineno}: {e.msg}")
        except Exception as e:
            syntax_errors.append(f"{py_file}: {str(e)}")
    
    return {
        'tool': 'syntax_check',
        'passed': len(syntax_errors) == 0,
        'output': '\n'.join(syntax_errors) if syntax_errors else 'All files have valid syntax',
        'issues_count': len(syntax_errors)
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple CI code quality checker')
    parser.add_argument('command', choices=['check', 'report'], help='Command to run')
    parser.add_argument('path', type=Path, help='Path to check')
    parser.add_argument('--fail-on-error', action='store_true', help='Exit with error code if issues found')
    parser.add_argument('--output', type=Path, help='Output file for results')
    parser.add_argument('--format', choices=['json', 'text', 'html'], default='text', help='Output format')
    
    args = parser.parse_args()
    
    if not args.path.exists():
        print(f"Error: Path {args.path} does not exist")
        return 1
    
    print(f"Running code quality checks on {args.path}")
    
    # Find Python files
    python_files = find_python_files(args.path)
    print(f"Found {len(python_files)} Python files")
    
    # Run checks
    results = []
    
    # Basic syntax check
    results.append(run_basic_syntax_check(python_files))
    
    # Black formatting check
    results.append(run_black_check(args.path))
    
    # Ruff linting check
    results.append(run_ruff_check(args.path))
    
    # Pytest check
    results.append(run_pytest_check(args.path))
    
    # Calculate summary
    total_issues = sum(r['issues_count'] for r in results)
    passed_checks = sum(1 for r in results if r['passed'])
    total_checks = len(results)
    
    # Generate output
    if args.format == 'json':
        output_data = {
            'summary': {
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'total_issues': total_issues,
                'files_analyzed': len(python_files)
            },
            'results': results
        }
        output_content = json.dumps(output_data, indent=2)
    
    elif args.format == 'html':
        output_content = generate_html_report(results, total_issues, passed_checks, total_checks, len(python_files))
    
    else:  # text format
        output_lines = [
            f"Code Quality Check Results",
            f"=" * 40,
            f"Files analyzed: {len(python_files)}",
            f"Checks run: {total_checks}",
            f"Checks passed: {passed_checks}",
            f"Total issues: {total_issues}",
            f"",
            f"Detailed Results:",
            f"-" * 20
        ]
        
        for result in results:
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            output_lines.append(f"{result['tool']}: {status} ({result['issues_count']} issues)")
            if result['output'] and not result['passed']:
                # Show first few lines of output for failed checks
                output_preview = '\n'.join(result['output'].split('\n')[:10])
                output_lines.append(f"  {output_preview}")
                if len(result['output'].split('\n')) > 10:
                    output_lines.append("  ...")
            output_lines.append("")
        
        output_content = '\n'.join(output_lines)
    
    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_content)
        print(f"Results written to {args.output}")
    else:
        print(output_content)
    
    # Exit with appropriate code
    if args.fail_on_error and total_issues > 0:
        return 1
    
    return 0


def generate_html_report(results, total_issues, passed_checks, total_checks, files_analyzed):
    """Generate a simple HTML report."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Code Quality Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .result {{ margin: 10px 0; padding: 10px; border-radius: 5px; }}
        .pass {{ background-color: #d4edda; border-left: 4px solid #28a745; }}
        .fail {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}
        .output {{ font-family: monospace; background-color: #f8f9fa; padding: 10px; margin-top: 10px; white-space: pre-wrap; }}
    </style>
</head>
<body>
    <h1>Code Quality Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Files analyzed: {files_analyzed}</p>
        <p>Checks run: {total_checks}</p>
        <p>Checks passed: {passed_checks}</p>
        <p>Total issues: {total_issues}</p>
    </div>
    
    <h2>Detailed Results</h2>
"""
    
    for result in results:
        css_class = "pass" if result['passed'] else "fail"
        status = "PASS" if result['passed'] else "FAIL"
        
        html += f"""
    <div class="result {css_class}">
        <h3>{result['tool']}: {status} ({result['issues_count']} issues)</h3>
        {f'<div class="output">{result["output"]}</div>' if result['output'] and not result['passed'] else ''}
    </div>
"""
    
    html += """
</body>
</html>
"""
    return html


if __name__ == '__main__':
    sys.exit(main())