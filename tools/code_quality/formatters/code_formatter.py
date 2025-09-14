"""
Code formatting checker and fixer.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import logging

from ..models import QualityIssue, QualityIssueType, QualitySeverity, QualityConfig


logger = logging.getLogger(__name__)


class CodeFormatter:
    """Handles code formatting checking and fixing."""

    def __init__(self, config: QualityConfig):
        """Initialize formatter with configuration."""
        self.config = config

    def check_formatting(self, file_path: Path, content: str) -> List[QualityIssue]:
        """Check formatting issues in the given file."""
        issues = []

        # Check with black if enabled
        if self.config.use_black:
            black_issues = self._check_black_formatting(file_path, content)
            issues.extend(black_issues)

        # Check import sorting with isort if enabled
        if self.config.use_isort:
            isort_issues = self._check_isort_formatting(file_path, content)
            issues.extend(isort_issues)

        # Check basic formatting rules
        basic_issues = self._check_basic_formatting(file_path, content)
        issues.extend(basic_issues)

        return issues

    def fix_formatting(self, file_path: Path) -> bool:
        """Fix formatting issues in the given file."""
        try:
            # Fix with black if enabled
            if self.config.use_black:
                self._run_black(file_path)

            # Fix imports with isort if enabled
            if self.config.use_isort:
                self._run_isort(file_path)

            return True
        except Exception as e:
            logger.error(f"Failed to fix formatting in {file_path}: {e}")
            return False

    def fix_imports(self, file_path: Path) -> bool:
        """Fix import issues in the given file."""
        try:
            if self.config.use_isort:
                self._run_isort(file_path)
            return True
        except Exception as e:
            logger.error(f"Failed to fix imports in {file_path}: {e}")
            return False

    def _check_black_formatting(self, file_path: Path, content: str) -> List[QualityIssue]:
        """Check formatting with black."""
        issues = []

        try:
            # Run black in check mode
            result = subprocess.run([
                sys.executable, '-m', 'black',
                '--check',
                '--line-length', str(self.config.line_length),
                str(file_path)
            ], capture_output=True, text=True)

            if result.returncode != 0:
                # Black found formatting issues
                issue = QualityIssue(
                    file_path=file_path,
                    line_number=1,
                    column=1,
                    issue_type=QualityIssueType.FORMATTING,
                    severity=QualitySeverity.WARNING,
                    message="File is not formatted according to black standards",
                    rule_code="BLACK_FORMAT",
                    suggestion="Run black to auto-format this file",
                    auto_fixable=True
                )
                issues.append(issue)

        except FileNotFoundError:
            logger.warning("Black not found, skipping black formatting check")
        except Exception as e:
            logger.error(f"Error running black on {file_path}: {e}")

        return issues

    def _check_isort_formatting(self, file_path: Path, content: str) -> List[QualityIssue]:
        """Check import sorting with isort."""
        issues = []

        try:
            # Run isort in check mode
            result = subprocess.run([
                sys.executable, '-m', 'isort',
                '--check-only',
                '--line-length', str(self.config.line_length),
                str(file_path)
            ], capture_output=True, text=True)

            if result.returncode != 0:
                # isort found import issues
                issue = QualityIssue(
                    file_path=file_path,
                    line_number=1,
                    column=1,
                    issue_type=QualityIssueType.IMPORTS,
                    severity=QualitySeverity.WARNING,
                    message="Imports are not sorted according to isort standards",
                    rule_code="ISORT_IMPORTS",
                    suggestion="Run isort to auto-sort imports",
                    auto_fixable=True
                )
                issues.append(issue)

        except FileNotFoundError:
            logger.warning("isort not found, skipping import sorting check")
        except Exception as e:
            logger.error(f"Error running isort on {file_path}: {e}")

        return issues

    def _check_basic_formatting(self, file_path: Path, content: str) -> List[QualityIssue]:
        """Check basic formatting rules."""
        issues = []
        lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
            # Check line length
            if len(line) > self.config.max_line_length:
                issue = QualityIssue(
                    file_path=file_path,
                    line_number=line_num,
                    column=self.config.max_line_length + 1,
                    issue_type=QualityIssueType.FORMATTING,
                    severity=QualitySeverity.WARNING,
                    message=f"Line too long ({len(line)} > {self.config.max_line_length} characters)",
                    rule_code="LINE_TOO_LONG",
                    suggestion="Break line into multiple lines"
                )
                issues.append(issue)

            # Check trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                issue = QualityIssue(
                    file_path=file_path,
                    line_number=line_num,
                    column=len(line.rstrip()) + 1,
                    issue_type=QualityIssueType.FORMATTING,
                    severity=QualitySeverity.INFO,
                    message="Trailing whitespace",
                    rule_code="TRAILING_WHITESPACE",
                    suggestion="Remove trailing whitespace",
                    auto_fixable=True
                )
                issues.append(issue)

            # Check mixed tabs and spaces
            if '\t' in line and '    ' in line:
                issue = QualityIssue(
                    file_path=file_path,
                    line_number=line_num,
                    column=1,
                    issue_type=QualityIssueType.FORMATTING,
                    severity=QualitySeverity.WARNING,
                    message="Mixed tabs and spaces for indentation",
                    rule_code="MIXED_INDENTATION",
                    suggestion="Use consistent indentation (spaces recommended)"
                )
                issues.append(issue)

        # Check file ending
        if content and not content.endswith('\n'):
            issue = QualityIssue(
                file_path=file_path,
                line_number=len(lines),
                column=len(lines[-1]) + 1,
                issue_type=QualityIssueType.FORMATTING,
                severity=QualitySeverity.INFO,
                message="File does not end with newline",
                rule_code="NO_NEWLINE_EOF",
                suggestion="Add newline at end of file",
                auto_fixable=True
            )
            issues.append(issue)

        return issues

    def _run_black(self, file_path: Path) -> None:
        """Run black formatter on file."""
        subprocess.run([
            sys.executable, '-m', 'black',
            '--line-length', str(self.config.line_length),
            str(file_path)
        ], check=True)

    def _run_isort(self, file_path: Path) -> None:
        """Run isort on file."""
        subprocess.run([
            sys.executable, '-m', 'isort',
            '--line-length', str(self.config.line_length),
            str(file_path)
        ], check=True)
