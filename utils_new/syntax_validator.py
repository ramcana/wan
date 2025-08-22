"""
Syntax Validator for WAN22 System Optimization
Detects and repairs syntax errors in critical Python files using AST parsing
"""

import ast
import logging
import os
import shutil
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SyntaxIssue:
    """Represents a syntax error found in a file"""
    file_path: str
    line_number: int
    column: int
    message: str
    error_type: str
    suggested_fix: Optional[str] = None

@dataclass
class ValidationResult:
    """Result of syntax validation"""
    is_valid: bool
    file_path: str
    errors: List[SyntaxIssue]
    warnings: List[str]
    
@dataclass
class RepairResult:
    """Result of syntax repair operation"""
    success: bool
    file_path: str
    backup_path: Optional[str]
    repairs_made: List[str]
    remaining_errors: List[SyntaxIssue]

class SyntaxValidator:
    """
    Advanced syntax validator with AST parsing and automated repair capabilities
    """
    
    def __init__(self, backup_dir: str = "syntax_backups"):
        """
        Initialize the syntax validator
        
        Args:
            backup_dir: Directory to store file backups before repairs
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Common syntax error patterns and their fixes
        self.repair_patterns = {
            'missing_else_in_conditional_expression': {
                'pattern': r'(\w+\s+if\s+[^]]+)\]',
                'description': 'Missing else clause in conditional expression',
                'fix_template': r'\1 else None]'
            },
            'missing_closing_bracket': {
                'pattern': r'(\[[^\]]*$)',
                'description': 'Missing closing bracket',
                'fix_template': r'\1]'
            },
            'missing_closing_paren': {
                'pattern': r'(\([^)]*$)',
                'description': 'Missing closing parenthesis', 
                'fix_template': r'\1)'
            },
            'trailing_comma_in_function_call': {
                'pattern': r'(\w+\([^)]*,)\s*\)',
                'description': 'Trailing comma in function call',
                'fix_template': r'\1)'
            }
        }
        
        # Critical files that should be validated
        self.critical_files = [
            'ui_event_handlers_enhanced.py',
            'ui_event_handlers.py', 
            'main.py',
            'ui.py',
            'wan_pipeline_loader.py',
            'utils.py'
        ]
        
        logger.info(f"SyntaxValidator initialized with backup directory: {self.backup_dir}")
    
    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Validate syntax of a Python file using AST parsing
        
        Args:
            file_path: Path to the Python file to validate
            
        Returns:
            ValidationResult with validation status and any errors found
        """
        file_path = str(file_path)
        errors = []
        warnings = []
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                errors.append(SyntaxIssue(
                    file_path=file_path,
                    line_number=0,
                    column=0,
                    message=f"File not found: {file_path}",
                    error_type="file_not_found"
                ))
                return ValidationResult(False, file_path, errors, warnings)
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse with AST
            try:
                ast.parse(content, filename=file_path)
                logger.info(f"✅ Syntax validation passed for {file_path}")
                return ValidationResult(True, file_path, [], warnings)
                
            except SyntaxError as e:
                # AST parsing failed, analyze the error
                syntax_issue = SyntaxIssue(
                    file_path=file_path,
                    line_number=e.lineno or 0,
                    column=e.offset or 0,
                    message=e.msg or "Unknown syntax error",
                    error_type="syntax_error",
                    suggested_fix=self._suggest_fix(content, e)
                )
                errors.append(syntax_issue)
                
                logger.error(f"❌ Syntax error in {file_path} at line {e.lineno}: {e.msg}")
                
            # Additional validation checks
            additional_errors = self._perform_additional_checks(file_path, content)
            errors.extend(additional_errors)
            
            return ValidationResult(False, file_path, errors, warnings)
            
        except Exception as e:
            logger.error(f"Validation failed for {file_path}: {e}")
            errors.append(SyntaxIssue(
                file_path=file_path,
                line_number=0,
                column=0,
                message=f"Validation error: {str(e)}",
                error_type="validation_error"
            ))
            return ValidationResult(False, file_path, errors, warnings)
    
    def _suggest_fix(self, content: str, syntax_error: Exception) -> Optional[str]:
        """
        Suggest a fix for common syntax errors
        
        Args:
            content: File content
            syntax_error: The syntax error exception
            
        Returns:
            Suggested fix string or None
        """
        error_msg = str(syntax_error.msg).lower()
        
        # Check for missing else in conditional expression
        if "expected 'else'" in error_msg and "if" in error_msg:
            return "Add 'else None' or appropriate else clause to conditional expression"
        
        # Check for missing closing brackets/parentheses
        if "unexpected eof" in error_msg or "expected" in error_msg:
            lines = content.split('\n')
            if syntax_error.lineno and syntax_error.lineno <= len(lines):
                line = lines[syntax_error.lineno - 1]
                
                # Count brackets and parentheses
                open_brackets = line.count('[') - line.count(']')
                open_parens = line.count('(') - line.count(')')
                
                if open_brackets > 0:
                    return f"Add {open_brackets} closing bracket(s) ']'"
                elif open_parens > 0:
                    return f"Add {open_parens} closing parenthesis(es) ')'"
        
        return None
    
    def _perform_additional_checks(self, file_path: str, content: str) -> List[SyntaxIssue]:
        """
        Perform additional syntax checks beyond AST parsing
        
        Args:
            file_path: Path to the file
            content: File content
            
        Returns:
            List of additional syntax errors found
        """
        errors = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for common issues
            
            # Unmatched brackets in list comprehensions
            if 'if' in line and ']' in line and 'else' not in line:
                # Look for pattern like: [item if condition]
                if re.search(r'\[.*\s+if\s+[^]]+\]', line) and 'else' not in line:
                    errors.append(SyntaxIssue(
                        file_path=file_path,
                        line_number=i,
                        column=0,
                        message="Conditional expression in list may be missing 'else' clause",
                        error_type="missing_else_clause",
                        suggested_fix="Add 'else None' or appropriate else value"
                    ))
            
            # Trailing commas in inappropriate places
            if re.search(r'\w+\([^)]*,\s*\)$', line.strip()):
                errors.append(SyntaxIssue(
                    file_path=file_path,
                    line_number=i,
                    column=0,
                    message="Trailing comma in function call",
                    error_type="trailing_comma",
                    suggested_fix="Remove trailing comma"
                ))
        
        return errors
    
    def backup_file(self, file_path: str) -> str:
        """
        Create a backup of the file before making repairs
        
        Args:
            file_path: Path to the file to backup
            
        Returns:
            Path to the backup file
        """
        file_path = Path(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}.backup"
        backup_path = self.backup_dir / backup_name
        
        try:
            shutil.copy2(file_path, backup_path)
            logger.info(f"📁 Created backup: {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.error(f"Failed to create backup for {file_path}: {e}")
            raise
    
    def repair_syntax_errors(self, file_path: str) -> RepairResult:
        """
        Attempt to automatically repair syntax errors in a file
        
        Args:
            file_path: Path to the file to repair
            
        Returns:
            RepairResult with repair status and details
        """
        file_path = str(file_path)
        repairs_made = []
        backup_path = None
        
        try:
            # First validate to identify errors
            validation_result = self.validate_file(file_path)
            
            if validation_result.is_valid:
                logger.info(f"✅ No repairs needed for {file_path}")
                return RepairResult(True, file_path, None, [], [])
            
            # If file doesn't exist, return the validation errors
            if not os.path.exists(file_path):
                return RepairResult(False, file_path, None, [], validation_result.errors)
            
            # Create backup before making changes
            backup_path = self.backup_file(file_path)
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply repairs for each error
            for error in validation_result.errors:
                if error.error_type == "syntax_error":
                    content, repair_made = self._repair_syntax_error(content, error)
                    if repair_made:
                        repairs_made.append(repair_made)
            
            # Apply pattern-based repairs
            content, pattern_repairs = self._apply_pattern_repairs(content)
            repairs_made.extend(pattern_repairs)
            
            # Write repaired content back to file
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"🔧 Applied {len(repairs_made)} repairs to {file_path}")
                
                # Validate the repaired file
                final_validation = self.validate_file(file_path)
                
                return RepairResult(
                    success=final_validation.is_valid,
                    file_path=file_path,
                    backup_path=backup_path,
                    repairs_made=repairs_made,
                    remaining_errors=final_validation.errors
                )
            else:
                logger.warning(f"⚠️ No repairs could be applied to {file_path}")
                return RepairResult(False, file_path, backup_path, [], validation_result.errors)
                
        except Exception as e:
            logger.error(f"Repair failed for {file_path}: {e}")
            return RepairResult(False, file_path, backup_path, repairs_made, [])
    
    def _repair_syntax_error(self, content: str, error: SyntaxIssue) -> Tuple[str, Optional[str]]:
        """
        Repair a specific syntax error
        
        Args:
            content: File content
            error: The syntax error to repair
            
        Returns:
            Tuple of (repaired_content, repair_description)
        """
        lines = content.split('\n')
        
        if error.line_number > 0 and error.line_number <= len(lines):
            line_idx = error.line_number - 1
            line = lines[line_idx]
            
            # Handle missing else clause in conditional expression
            if "expected 'else'" in error.message.lower():
                # Look for pattern: item if condition]
                match = re.search(r'(\w+\s+if\s+[^]]+)\]', line)
                if match:
                    # Replace with: item if condition else None]
                    new_line = line.replace(match.group(1) + ']', match.group(1) + ' else None]')
                    lines[line_idx] = new_line
                    return '\n'.join(lines), f"Added 'else None' clause at line {error.line_number}"
            
            # Handle missing closing brackets
            if "unexpected eof" in error.message.lower():
                open_brackets = line.count('[') - line.count(']')
                if open_brackets > 0:
                    lines[line_idx] = line + ']' * open_brackets
                    return '\n'.join(lines), f"Added {open_brackets} closing bracket(s) at line {error.line_number}"
        
        return content, None
    
    def _apply_pattern_repairs(self, content: str) -> Tuple[str, List[str]]:
        """
        Apply pattern-based repairs to content
        
        Args:
            content: File content
            
        Returns:
            Tuple of (repaired_content, list_of_repairs_made)
        """
        repairs_made = []
        
        for pattern_name, pattern_info in self.repair_patterns.items():
            pattern = pattern_info['pattern']
            fix_template = pattern_info['fix_template']
            description = pattern_info['description']
            
            # Apply the pattern fix
            new_content, count = re.subn(pattern, fix_template, content)
            
            if count > 0:
                content = new_content
                repairs_made.append(f"{description} ({count} instances)")
        
        return content, repairs_made
    
    def validate_enhanced_handlers(self) -> ValidationResult:
        """
        Specifically validate the enhanced event handlers file
        
        Returns:
            ValidationResult for ui_event_handlers_enhanced.py
        """
        return self.validate_file('ui_event_handlers_enhanced.py')
    
    def validate_critical_files(self) -> Dict[str, ValidationResult]:
        """
        Validate all critical Python files
        
        Returns:
            Dictionary mapping file paths to their validation results
        """
        results = {}
        
        for file_path in self.critical_files:
            if os.path.exists(file_path):
                results[file_path] = self.validate_file(file_path)
            else:
                logger.warning(f"Critical file not found: {file_path}")
        
        return results
    
    def repair_critical_files(self) -> Dict[str, RepairResult]:
        """
        Repair syntax errors in all critical files
        
        Returns:
            Dictionary mapping file paths to their repair results
        """
        results = {}
        
        for file_path in self.critical_files:
            if os.path.exists(file_path):
                results[file_path] = self.repair_syntax_errors(file_path)
            else:
                logger.warning(f"Critical file not found for repair: {file_path}")
        
        return results
    
    def get_validation_summary(self, validation_results: Dict[str, ValidationResult]) -> str:
        """
        Generate a summary of validation results
        
        Args:
            validation_results: Dictionary of validation results
            
        Returns:
            Formatted summary string
        """
        total_files = len(validation_results)
        valid_files = sum(1 for result in validation_results.values() if result.is_valid)
        invalid_files = total_files - valid_files
        
        summary = f"""
Syntax Validation Summary:
========================
Total files checked: {total_files}
Valid files: {valid_files}
Files with errors: {invalid_files}

"""
        
        if invalid_files > 0:
            summary += "Files with syntax errors:\n"
            for file_path, result in validation_results.items():
                if not result.is_valid:
                    summary += f"  ❌ {file_path}: {len(result.errors)} error(s)\n"
                    for error in result.errors:
                        summary += f"     Line {error.line_number}: {error.message}\n"
        
        return summary

# Convenience functions for easy usage
def validate_file(file_path: str) -> ValidationResult:
    """Validate a single file"""
    validator = SyntaxValidator()
    return validator.validate_file(file_path)

def repair_file(file_path: str) -> RepairResult:
    """Repair a single file"""
    validator = SyntaxValidator()
    return validator.repair_syntax_errors(file_path)

def validate_all_critical_files() -> Dict[str, ValidationResult]:
    """Validate all critical files"""
    validator = SyntaxValidator()
    return validator.validate_critical_files()