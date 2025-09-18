#!/usr/bin/env python3
"""
Port validation script for WAN2.2 Video Generation System.
This script checks for consistent port usage across the codebase.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Define expected port values
EXPECTED_PORTS = {
    "backend": 8000,
    "frontend_dev": 3000,
    "gradio": 7860,
    "database": 5432,
    "redis": 6379,
}

# File patterns to check
FILE_PATTERNS = [
    "*.py",
    "*.ts",
    "*.tsx",
    "*.js",
    "*.jsx",
    "*.json",
    "*.yml",
    "*.yaml",
    "*.env",
    "*.conf",
    "*.config",
]

# Directories to exclude
EXCLUDE_DIRS = [
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    "coverage",
    ".pytest_cache",
]

# Patterns that indicate actual port usage (more specific than just numbers)
PORT_PATTERNS = [
    r'[\'"](?:localhost|127\.0\.0\.1)[:\s](\d{4,5})[\'"]',
    r'(?:port|PORT)[:\s][\'"]?(\d{4,5})[\'"]?',
    r"http(?:s?)://[\w.-]*[:\s](\d{4,5})",
    r"BACKEND_PORT\s*=\s*(\d+)",
    r"FRONTEND_PORT\s*=\s*(\d+)",
    r"(?:server|host|address).*?(\d{4,5})",
    r"docker.*?(\d{4,5}):(\d{4,5})",
]

# File extensions that are more likely to contain actual port configurations
CONFIG_EXTENSIONS = {
    ".py",
    ".ts",
    ".js",
    ".json",
    ".yml",
    ".yaml",
    ".env",
    ".conf",
    ".config",
}


def find_files(directory: Path) -> List[Path]:
    """Find all files matching the patterns, excluding certain directories."""
    files = []
    for pattern in FILE_PATTERNS:
        for file_path in directory.rglob(pattern):
            # Check if file is in an excluded directory
            excluded = False
            for exclude_dir in EXCLUDE_DIRS:
                if exclude_dir in str(file_path):
                    excluded = True
                    break
            if not excluded:
                files.append(file_path)
    return files


def get_sort_key(item):
    """Get sort key for port grouping."""
    return len(item[1])


def check_port_usage(file_path: Path) -> List[Tuple[int, str, int, str]]:
    """Check a file for port usage and return any non-standard ports found."""
    issues: List[Tuple[int, str, int, str]] = []

    # Only check files with extensions likely to contain configuration
    if file_path.suffix not in CONFIG_EXTENSIONS:
        return issues

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # Skip lines that are clearly not port configurations
                if "version" in line.lower() or "license" in line.lower():
                    continue

                # Check for specific port patterns
                for pattern in PORT_PATTERNS:
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    for match in matches:
                        # For most patterns, the port is in group 1
                        try:
                            port = int(match.group(1))
                        except (IndexError, ValueError):
                            continue

                        # Check if this is a non-standard port
                        is_expected = port in EXPECTED_PORTS.values()
                        is_valid_range = 1024 <= port <= 65535

                        # Only flag unexpected ports in valid range
                        if not is_expected and is_valid_range:
                            if "(" in pattern:
                                pattern_type = pattern.split("(")[0]
                            else:
                                pattern_type = pattern[:20]
                            # Create tuple for the issue
                            issue_data = (port, line.strip(), line_num, pattern_type)
                            issues.append(issue_data)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return issues


def validate_environment_configs() -> Dict[str, List[str]]:
    """Validate environment configuration files."""
    issues: Dict[str, List[str]] = {}

    # Check .env files
    env_files = list(Path(".").rglob(".env*"))
    for env_file in env_files:
        issues[str(env_file)] = []
        try:
            with open(env_file, "r") as f:
                content = f.read()

                # Check for port-related environment variables
                port_vars = re.findall(r"(.*_PORT\s*=\s*)(\d+)", content)
                for var_name, port_value in port_vars:
                    port = int(port_value)
                    expected_name = var_name.split("=")[0].strip().lower()

                    # Check if this matches expected ports
                    is_backend = "backend" in expected_name
                    backend_mismatch = port != EXPECTED_PORTS["backend"]
                    if is_backend and backend_mismatch:
                        msg = (
                            f"Backend port mismatch: expected "
                            f"{EXPECTED_PORTS['backend']}, found {port}"
                        )
                        issues[str(env_file)].append(msg)

                    is_frontend = "frontend" in expected_name
                    frontend_mismatch = port != EXPECTED_PORTS["frontend_dev"]
                    if is_frontend and frontend_mismatch:
                        msg = (
                            f"Frontend port mismatch: expected "
                            f"{EXPECTED_PORTS['frontend_dev']}, found {port}"
                        )
                        issues[str(env_file)].append(msg)
        except Exception as e:
            issues[str(env_file)].append(f"Error reading file: {e}")

    return issues


def main():
    """Main validation function."""
    print("Validating port configurations across the codebase...")

    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Find all files to check
    files = find_files(project_root)
    print(f"Found {len(files)} files to check")

    # Track issues
    port_issues = []
    file_count = 0

    # Check each file
    for file_path in files:
        file_count += 1
        if file_count % 1000 == 0:
            print(f"Checked {file_count}/{len(files)} files...")

        issues = check_port_usage(file_path)
        if issues:
            for port, line, line_num, pattern_type in issues:
                port_issues.append(
                    {
                        "file": str(file_path.relative_to(project_root)),
                        "port": port,
                        "line": line_num,
                        "content": line.strip(),
                        "pattern": pattern_type,
                    }
                )

    # Validate environment configurations
    env_issues = validate_environment_configs()

    # Report findings
    print("\n" + "=" * 50)
    print("PORT VALIDATION REPORT")
    print("=" * 50)

    issue_count = len(port_issues)
    print(f"\nFound {issue_count} potential port configuration issues:")

    if port_issues:
        # Group issues by port number
        port_groups = {}
        for issue in port_issues:
            port = issue["port"]
            if port not in port_groups:
                port_groups[port] = []
            port_groups[port].append(issue)

        # Show top 10 ports with the most issues
        sorted_items = port_groups.items()
        sorted_ports = sorted(sorted_items, key=get_sort_key, reverse=True)
        for port, issues in sorted_ports[:10]:
            print(f"\nPort {port} - {len(issues)} issues:")
            for issue in issues[:3]:  # Show first 3 issues for each port
                file_info = f"{issue['file']}:{issue['line']}"
                print(f"  {file_info} - {issue['content']}")
            if len(issues) > 3:
                print(f"  ... and {len(issues) - 3} more")

        if len(sorted_ports) > 10:
            remaining = len(sorted_ports) - 10
            print(f"\n... and {remaining} more ports with issues")
    else:
        print("No unexpected port configuration issues found.")

    # Report environment issues
    env_issues_found = False
    for file_path, issues in env_issues.items():
        if issues:
            env_issues_found = True
            print(f"\nEnvironment configuration issues in {file_path}:")
            for issue in issues:
                print(f"  {issue}")

    if not env_issues_found:
        print("\nEnvironment configurations appear consistent.")

    # Summary
    print("\nSummary:")
    print(f"  - Checked {len(files)} files")
    print(f"  - Found {len(port_issues)} potential port configuration issues")

    env_issue_count = sum(len(issues) for issues in env_issues.values())
    print(f"  - Found {env_issue_count} environment configuration issues")

    # Return exit code
    has_issues = port_issues or any(env_issues.values())
    if has_issues:
        print("\nValidation completed with issues found.")
        return 1
    else:
        print("\nValidation completed successfully. No issues found.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
