"""
Style validation using flake8 and custom rules.
"""

import subprocess
import sys
from pathlib import Path
from typing import List
import logging

from models import QualityIssue, QualityIssueType, QualitySeverity, QualityConfig


logger = logging.getLogger(__name__)


class StyleValidator:
    """Validates code style using flake8 and custom rules."""
    
    def __init__(self, config: QualityConfig):
        """Initialize validator with configuration."""
        self.config = config
    
    def validate_style(self, file_path: Path, content: str) -> List[QualityIssue]:
        """Validate code style in the given file."""
        issues = []
        
        # Run flake8 for comprehensive style checking
        flake8_issues = self._run_flake8_check(file_path)
        issues.extend(flake8_issues)
        
        # Run custom style checks
        custom_issues = self._check_custom_style_rules(file_path, content)
        issues.extend(custom_issues)
        
        return issues
    
    def _run_flake8_check(self, file_path: Path) -> List[QualityIssue]:
        """Run flake8 style checker on the file."""
        issues = []
        
        try:
            # Build flake8 command
            cmd = [
                sys.executable, '-m', 'flake8',
                '--max-line-length', str(self.config.max_line_length),
                '--format=%(path)s:%(row)d:%(col)d:%(code)s:%(text)s'
            ]
            
            # Add ignore rules if specified
            if self.config.ignore_rules:
                cmd.extend(['--ignore', ','.join(self.config.ignore_rules)])
            
            cmd.append(str(file_path))
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout:
                # Parse flake8 output
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        flake8_issue = self._parse_flake8_output(file_path, line)
                        if flake8_issue:
                            issues.append(flake8_issue)
        
        except FileNotFoundError:
            logger.debug("flake8 not found, skipping flake8 style checking")
        except Exception as e:
            logger.error(f"Error running flake8 on {file_path}: {e}")
        
        return issues
    
    def _parse_flake8_output(self, file_path: Path, line: str) -> QualityIssue:
        """Parse flake8 output line into QualityIssue."""
        try:
            # Format: file.py:line:col:code:message
            parts = line.split(':', 4)
            if len(parts) >= 5:
                line_num = int(parts[1])
                col_num = int(parts[2])
                rule_code = parts[3]
                message = parts[4].strip()
                
                # Determine severity based on error code
                severity = self._get_severity_from_code(rule_code)
                
                # Check if it's auto-fixable
                auto_fixable = self._is_auto_fixable(rule_code)
                
                return QualityIssue(
                    file_path=file_path,
                    line_number=line_num,
                    column=col_num,
                    issue_type=QualityIssueType.STYLE,
                    severity=severity,
                    message=message,
                    rule_code=rule_code,
                    suggestion=self._get_suggestion_for_code(rule_code),
                    auto_fixable=auto_fixable
                )
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse flake8 output: {line} - {e}")
        
        return None
    
    def _get_severity_from_code(self, code: str) -> QualitySeverity:
        """Determine severity based on flake8 error code."""
        if code.startswith('E'):
            # Error codes
            if code.startswith('E9'):
                return QualitySeverity.ERROR  # Runtime errors
            else:
                return QualitySeverity.WARNING  # Style errors
        elif code.startswith('W'):
            return QualitySeverity.WARNING  # Warnings
        elif code.startswith('F'):
            return QualitySeverity.ERROR  # Pyflakes errors
        elif code.startswith('C'):
            return QualitySeverity.INFO  # Complexity warnings
        elif code.startswith('N'):
            return QualitySeverity.INFO  # Naming conventions
        else:
            return QualitySeverity.WARNING
    
    def _is_auto_fixable(self, code: str) -> bool:
        """Check if error code represents an auto-fixable issue."""
        # Common auto-fixable codes
        auto_fixable_codes = {
            'E101',  # indentation contains mixed spaces and tabs
            'E111',  # indentation is not a multiple of four
            'E112',  # expected an indented block
            'E113',  # unexpected indentation
            'E114',  # indentation is not a multiple of four (comment)
            'E115',  # expected an indented block (comment)
            'E116',  # unexpected indentation (comment)
            'E121',  # continuation line under-indented for hanging indent
            'E122',  # continuation line missing indentation or outdented
            'E123',  # closing bracket does not match indentation of opening bracket's line
            'E124',  # closing bracket does not match visual indentation
            'E125',  # continuation line with same indent as next logical line
            'E126',  # continuation line over-indented for hanging indent
            'E127',  # continuation line over-indented for visual indent
            'E128',  # continuation line under-indented for visual indent
            'E129',  # visually indented line with same indent as next logical line
            'E131',  # continuation line unaligned for hanging indent
            'E133',  # closing bracket is missing indentation
            'E201',  # whitespace after '('
            'E202',  # whitespace before ')'
            'E203',  # whitespace before ':'
            'E211',  # whitespace before '('
            'E221',  # multiple spaces before operator
            'E222',  # multiple spaces after operator
            'E223',  # tab before operator
            'E224',  # tab after operator
            'E225',  # missing whitespace around operator
            'E226',  # missing whitespace around arithmetic operator
            'E227',  # missing whitespace around bitwise or shift operator
            'E228',  # missing whitespace around modulo operator
            'E231',  # missing whitespace after ','
            'E241',  # multiple spaces after ','
            'E242',  # tab after ','
            'E251',  # unexpected spaces around keyword / parameter equals
            'E261',  # at least two spaces before inline comment
            'E262',  # inline comment should start with '# '
            'E265',  # block comment should start with '# '
            'E266',  # too many leading '#' for block comment
            'E271',  # multiple spaces after keyword
            'E272',  # multiple spaces before keyword
            'E273',  # tab after keyword
            'E274',  # tab before keyword
            'E275',  # missing whitespace after keyword
            'W291',  # trailing whitespace
            'W292',  # no newline at end of file
            'W293',  # blank line contains whitespace
            'W391',  # blank line at end of file
        }
        
        return code in auto_fixable_codes
    
    def _get_suggestion_for_code(self, code: str) -> str:
        """Get suggestion for fixing specific error code."""
        suggestions = {
            'E101': 'Use consistent indentation (spaces or tabs, not both)',
            'E111': 'Use 4 spaces for indentation',
            'E201': 'Remove whitespace after opening parenthesis',
            'E202': 'Remove whitespace before closing parenthesis',
            'E203': 'Remove whitespace before colon',
            'E225': 'Add whitespace around operator',
            'E231': 'Add whitespace after comma',
            'E261': 'Add at least two spaces before inline comment',
            'E262': 'Start inline comment with "# "',
            'E265': 'Start block comment with "# "',
            'E501': 'Break long line into multiple lines',
            'W291': 'Remove trailing whitespace',
            'W292': 'Add newline at end of file',
            'W293': 'Remove whitespace from blank line',
            'W391': 'Remove blank line at end of file',
            'F401': 'Remove unused import',
            'F841': 'Remove unused variable',
        }
        
        return suggestions.get(code, 'Fix style issue according to PEP 8')
    
    def _check_custom_style_rules(self, file_path: Path, content: str) -> List[QualityIssue]:
        """Check custom style rules not covered by flake8."""
        issues = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Check for TODO/FIXME comments
            if 'TODO' in line or 'FIXME' in line:
                issue = QualityIssue(
                    file_path=file_path,
                    line_number=line_num,
                    column=line.find('TODO') + 1 if 'TODO' in line else line.find('FIXME') + 1,
                    issue_type=QualityIssueType.STYLE,
                    severity=QualitySeverity.INFO,
                    message="TODO/FIXME comment found",
                    rule_code="TODO_COMMENT",
                    suggestion="Address TODO/FIXME comment or create issue tracker item"
                )
                issues.append(issue)
            
            # Check for print statements (should use logging)
            if line.strip().startswith('print(') and not line.strip().startswith('# print('):
                issue = QualityIssue(
                    file_path=file_path,
                    line_number=line_num,
                    column=line.find('print(') + 1,
                    issue_type=QualityIssueType.STYLE,
                    severity=QualitySeverity.INFO,
                    message="Use logging instead of print statements",
                    rule_code="PRINT_STATEMENT",
                    suggestion="Replace print() with appropriate logging call"
                )
                issues.append(issue)
            
            # Check for hardcoded strings that might be constants
            if self._has_hardcoded_strings(line):
                issue = QualityIssue(
                    file_path=file_path,
                    line_number=line_num,
                    column=1,
                    issue_type=QualityIssueType.STYLE,
                    severity=QualitySeverity.INFO,
                    message="Consider using constants for hardcoded strings",
                    rule_code="HARDCODED_STRING",
                    suggestion="Define string constants at module level"
                )
                issues.append(issue)
        
        return issues
    
    def _has_hardcoded_strings(self, line: str) -> bool:
        """Check if line contains hardcoded strings that should be constants."""
        # Simple heuristic: look for string literals longer than 20 characters
        # that are not in comments or docstrings
        if line.strip().startswith('#') or '"""' in line or "'''" in line:
            return False
        
        # Look for string literals
        import re
        string_pattern = r'["\'][^"\']{20,}["\']'
        matches = re.findall(string_pattern, line)
        
        return len(matches) > 0
