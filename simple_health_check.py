#!/usr/bin/env python3
"""
Simple health check script for CI/CD workflows
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime


def check_project_health():
    """Perform basic project health checks"""
    score = 0
    max_score = 100
    issues = []
    
    # Check if basic files exist (20 points)
    required_files = ["config.json", "requirements.txt", "README.md"]
    for file_path in required_files:
        if Path(file_path).exists():
            score += 7
        else:
            issues.append(f"Missing required file: {file_path}")
    
    # Check if basic directories exist (20 points)
    required_dirs = ["backend", "frontend", "tests"]
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            score += 7
        else:
            issues.append(f"Missing required directory: {dir_path}")
    
    # Check if config.json is valid JSON (20 points)
    try:
        with open("config.json") as f:
            json.load(f)
        score += 20
    except Exception as e:
        issues.append(f"Invalid config.json: {e}")
    
    # Check if requirements.txt has content (20 points)
    try:
        with open("requirements.txt") as f:
            content = f.read().strip()
            if content and len(content.split('\n')) > 5:
                score += 20
            else:
                issues.append("requirements.txt appears to be empty or too short")
    except Exception as e:
        issues.append(f"Cannot read requirements.txt: {e}")
    
    # Check if tests directory has test files (20 points)
    test_files = list(Path("tests").rglob("test_*.py"))
    if len(test_files) >= 3:
        score += 20
    else:
        issues.append(f"Found only {len(test_files)} test files, expected at least 3")
    
    return {
        "overall_score": min(score, max_score),
        "max_score": max_score,
        "timestamp": datetime.now().isoformat(),
        "issues": issues,
        "checks": {
            "files_exist": len([f for f in required_files if Path(f).exists()]),
            "dirs_exist": len([d for d in required_dirs if Path(d).exists()]),
            "config_valid": Path("config.json").exists(),
            "requirements_valid": Path("requirements.txt").exists(),
            "test_files_count": len(test_files)
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Simple project health check")
    parser.add_argument("--output-format", default="json", choices=["json", "text"])
    parser.add_argument("--output-file", help="Output file path")
    
    args = parser.parse_args()
    
    try:
        health_report = check_project_health()
        
        if args.output_format == "json":
            output = json.dumps(health_report, indent=2)
        else:
            output = f"""
Project Health Report
====================
Overall Score: {health_report['overall_score']}/{health_report['max_score']}
Timestamp: {health_report['timestamp']}

Issues:
{chr(10).join(f"- {issue}" for issue in health_report['issues']) if health_report['issues'] else "None"}

Checks:
- Files exist: {health_report['checks']['files_exist']}/3
- Directories exist: {health_report['checks']['dirs_exist']}/3
- Config valid: {health_report['checks']['config_valid']}
- Requirements valid: {health_report['checks']['requirements_valid']}
- Test files: {health_report['checks']['test_files_count']}
"""
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(output)
            print(f"Health report written to {args.output_file}")
        else:
            print(output)
        
        # Exit with error code if score is too low
        if health_report['overall_score'] < 75:
            print(f"Health score {health_report['overall_score']} is below threshold 75", file=sys.stderr)
            sys.exit(1)
        
        print(f"Health check passed with score {health_report['overall_score']}/100")
        
    except Exception as e:
        print(f"Error running health check: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()