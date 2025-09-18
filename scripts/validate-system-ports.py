#!/usr/bin/env python3
"""
System port validation script for WAN2.2 Video Generation System.
This script checks for consistent usage of system ports (8000, 3000, 7860)
across the codebase.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Define expected system ports
SYSTEM_PORTS = {"backend_api": 8000, "frontend_dev": 3000, "gradio_ui": 7860}

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
    "*.md",
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
    "fresh_env",
]

# Patterns to identify port usage in configuration files
PORT_PATTERNS = [
    # Environment variable assignments
    r"(?:BACKEND|FRONTEND|GRADIO|API|SERVER|PORT).*?[:=\s](\d{4,5})",
    # URL patterns
    r"(?:localhost|127\.0\.0\.1)[:\s](\d{4,5})",
    r"http(?:s?)://.*?:(\d{4,5})",
    # Docker port mappings
    r'"(\d{4,5}):\d{4,5}"',
    r"'(\d{4,5}):\d{4,5}'",
    # Configuration files
    r'"port"\s*:\s*(\d{4,5})',
    r"'port'\s*:\s*(\d{4,5})",
]


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


def check_system_port_usage(
    file_path: Path
) -> List[Tuple[str, int, str, int]]:
    """Check a file for system port usage and return any mismatches."""
    issues = []

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                line_content = line.strip()
                if not line_content:
                    continue

                # Skip lines that are clearly not port configurations
                skip_line = (
                    "version" in line_content.lower()
                    or "license" in line_content.lower()
                    or "copyright" in line_content.lower()
                )
                if skip_line:
                    continue

                # Check for specific port patterns
                for pattern in PORT_PATTERNS:
                    matches = re.finditer(pattern, line_content, re.IGNORECASE)
                    for match in matches:
                        try:
                            port = int(match.group(1))
                        except (IndexError, ValueError):
                            continue

                        # Check if this is one of our system ports
                        if port in SYSTEM_PORTS.values():
                            # Determine which system port this should be
                            port_name = ""
                            for name, expected_port in SYSTEM_PORTS.items():
                                if port == expected_port:
                                    port_name = name
                                    break

                            # Check context to see if this is the right place
                            context_issues = validate_port_context(
                                port_name, port, line_content, file_path
                            )

                            if context_issues:
                                for issue in context_issues:
                                    issues.append(
                                        (port_name, port, issue, line_num)
                                    )

                        # Check for common incorrect ports (like 9000)
                        elif port == 9000:  # Common mistake
                            # Skip documentation mentions that are just 
                            # describing changes
                            doc_mention = "Updated default port from 9000 to 8000"
                            if doc_mention in line_content:
                                continue
                            
                            backend_port = SYSTEM_PORTS["backend_api"]
                            error_msg = (f"Found port 8000, should be "
                                       f"{backend_port} (backend API)")
                            issues.append(
                                ("backend_api", port, error_msg, line_num)
                            )
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return issues


def validate_port_context(
    port_name: str, port: int, line_content: str, file_path: Path
) -> List[str]:
    """Validate that the port is used in the correct context."""
    issues = []

    # Check for backend port in frontend files
    if port_name == "backend_api" and "frontend" in str(file_path).lower():
        # Check if it's correctly configured (should reference VITE_API_URL)
        has_localhost = "localhost:8000" in line_content
        missing_env_var = "VITE_API_URL" not in line_content
        if has_localhost and missing_env_var:
            issues.append(
                "Hardcoded backend port in frontend, should use "
                "VITE_API_URL or relative path"
            )

    # Check for frontend port in backend files
    if port_name == "frontend_dev" and "backend" in str(file_path).lower():
        issues.append("Frontend development port found in backend code")

    # Check for gradio port in non-gradio files
    if port_name == "gradio_ui" and "gradio" not in str(file_path).lower():
        # This might be okay in some config files, but worth noting
        ignore_paths = ["config", "docker-compose"]
        path_str = str(file_path)
        should_ignore = any(ignore in path_str for ignore in ignore_paths)
        if not should_ignore:
            issues.append("Gradio port found outside gradio-related files")

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

                # Check for system port environment variables
                for (
                    port_name, 
                    expected_port
                ) in SYSTEM_PORTS.items():
                    # Look for environment variables that should contain this port
                    env_var_pattern = port_name.upper().replace("_", "")
                    pattern = rf"({env_var_pattern}_PORT\s*=\s*)(\d+)"
                    matches = re.findall(pattern, content)

                    for var_name, port_value in matches:
                        port = int(port_value)
                        if port != expected_port:
                            msg = (f"{port_name} port mismatch: expected "
                                 f"{expected_port}, found {port}")
                            issues[str(env_file)].append(msg)
        except Exception as e:
            issues[str(env_file)].append(f"Error reading file: {e}")

    return issues


def main():
    """Main validation function."""
    print("Validating system port configurations...")

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

        issues = check_system_port_usage(file_path)
        if issues:
            for port_name, port, issue_desc, line_num in issues:
                port_issues.append({
                    "file": str(file_path.relative_to(project_root)),
                    "port_name": port_name,
                    "port": port,
                    "issue": issue_desc,
                    "line": line_num,
                })

    # Validate environment configurations
    env_issues = validate_environment_configs()

    # Report findings
    print("\n" + "=" * 60)
    print("SYSTEM PORT VALIDATION REPORT")
    print("=" * 60)

    if port_issues:
        print(f"\nFound {len(port_issues)} system port configuration issues:")

        # Group issues by file
        file_groups = {}
        for issue in port_issues:
            file_path = issue["file"]
            if file_path not in file_groups:
                file_groups[file_path] = []
            file_groups[file_path].append(issue)

        # Show issues by file
        for file_path, issues in file_groups.items():
            print(f"\n{file_path}:")
            for issue in issues:
                port_info = f"{issue['port_name']} ({issue['port']})"
                line_part1 = f"  Line {issue['line']} - {port_info}:"
                line_part2 = f" {issue['issue']}"
                print(line_part1 + line_part2)
    else:
        print("No system port configuration issues found.")

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
    print(f"  - Found {len(port_issues)} system port configuration issues")

    env_issue_count = sum(len(issues) for issues in env_issues.values())
    print(f"  - Found {env_issue_count} environment configuration issues")

    # Show expected ports
    print("\nExpected system ports:")
    for name, port in SYSTEM_PORTS.items():
        print(f"  - {name}: {port}")

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