#!/usr/bin/env python3
"""
Script to automatically fix common port configuration issues in the
WAN2.2 codebase.
"""

import re
import sys
from pathlib import Path
from typing import List

# Define the fixes to apply
PORT_FIXES = [
    # Fix port 8000 -> 8000 (backend API port) for localhost references
    {
        "pattern": r"(localhost|127\.0\.0\.1):9000",
        "replacement": r"\1:8000",
        "description": "Fix localhost:8000 -> localhost:8000",
    },
    # Fix standalone port 8000 -> 8000 in command line arguments
    {
        "pattern": r"--port\s+9000",
        "replacement": r"--port 8000",
        "description": "Fix --port 8000 -> --port 8000",
    },
    # Fix port mentions in troubleshooting sections
    {
        "pattern": r"port\s+9000",
        "replacement": r"port 8000",
        "description": "Fix 'port 8000' -> 'port 8000'",
    },
    # Fix port mentions in automatic port detection lists
    {
        "pattern": r"\(8000,",
        "replacement": r"(8000,",
        "description": "Fix port detection list (8000, -> (8000,",
    },
]

# Files to exclude from automatic fixes
EXCLUDE_FILES = [
    "package-lock.json",
    "yarn.lock",
    "*.log",
    "*.txt",
    "*.css",  # Add CSS files to avoid regex issues
    "*.min.css",  # Add minified CSS files
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
    ".mypy_cache",  # Add this to avoid the regex errors
    "site",  # Exclude the site directory which contains generated HTML files
]


def should_exclude_file(file_path: Path) -> bool:
    """Check if a file should be excluded from fixes."""
    # Check file extensions
    for exclude_pattern in EXCLUDE_FILES:
        if file_path.match(exclude_pattern):
            return True

    # Check directory path
    for exclude_dir in EXCLUDE_DIRS:
        if exclude_dir in str(file_path):
            return True

    return False


def apply_fixes_to_file(file_path: Path) -> List[str]:
    """Apply port fixes to a single file."""
    fixes_applied: List[str] = []

    if should_exclude_file(file_path):
        return fixes_applied

    try:
        # Read the file
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Apply fixes
        new_content = content
        for fix in PORT_FIXES:
            # Apply the regex substitution
            original_content = new_content
            pattern = fix["pattern"]
            replacement = fix["replacement"]

            # Compile the regex pattern to catch errors early
            try:
                compiled_pattern = re.compile(pattern)
                new_content = compiled_pattern.sub(replacement, new_content)
            except re.error as e:
                error_msg = (
                    f"Regex error in pattern '{pattern}' for {file_path}: {e}"
                )
                print(error_msg)
                continue

            # Check if any changes were made
            if new_content != original_content:
                fixes_applied.append(fix["description"])

        # Write back if changes were made
        if new_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return fixes_applied


def find_files_to_fix(directory: Path) -> List[Path]:
    """Find all files that should be checked for port fixes."""
    files = []

    # Common file patterns to check
    file_patterns = [
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

    for pattern in file_patterns:
        for file_path in directory.rglob(pattern):
            if not should_exclude_file(file_path):
                files.append(file_path)

    return files


def main():
    """Main function to apply port fixes."""
    print("Applying automatic port configuration fixes...")

    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Find all files to check
    files = find_files_to_fix(project_root)
    print(f"Found {len(files)} files to check")

    # Track fixes
    total_fixes = 0
    files_with_fixes = 0

    # Apply fixes to each file
    for file_path in files:
        fixes = apply_fixes_to_file(file_path)
        if fixes:
            files_with_fixes += 1
            total_fixes += len(fixes)
            print(f"Applied fixes to {file_path}:")
            for fix in fixes:
                print(f"  - {fix}")

    # Summary
    print("\nSummary:")
    print(f"  - Checked {len(files)} files")
    print(f"  - Applied fixes to {files_with_fixes} files")
    print(f"  - Total fixes applied: {total_fixes}")

    if total_fixes > 0:
        print("\nPlease review the changes and run validation script again.")
        return 1
    else:
        print("\nNo fixes were needed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
