import pytest
"""
Naming Standardization and File Organization System

This module provides comprehensive naming convention analysis and standardization:
- Identifies inconsistent naming patterns
- Supports multiple naming conventions (snake_case, camelCase, PascalCase, kebab-case)
- Safe refactoring with reference updates
- File organization and structure improvements
"""

import os
import ast
import re
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import keyword


@dataclass
class NamingViolation:
    """Represents a naming convention violation"""
    name: str
    current_convention: str
    expected_convention: str
    suggested_name: str
    file_path: str
    line_number: int
    element_type: str  # 'function', 'class', 'variable', 'file', 'directory'
    severity: str  # 'error', 'warning', 'info'


@dataclass
class InconsistentPattern:
    """Represents an inconsistent naming pattern across the codebase"""
    pattern_type: str
    variations: List[str]
    files_affected: List[str]
    recommended_standard: str


@dataclass
class OrganizationSuggestion:
    """Represents a file organization suggestion"""
    current_path: str
    suggested_path: str
    reason: str
    impact_level: str  # 'low', 'medium', 'high'


@dataclass
class NamingReport:
    """Report of naming convention analysis"""
    total_files_analyzed: int
    violations: List[NamingViolation]
    inconsistent_patterns: List[InconsistentPattern]
    organization_suggestions: List[OrganizationSuggestion]
    convention_summary: Dict[str, int]
    recommendations: List[str]
    analysis_timestamp: str


class NamingStandardizer:
    """
    Comprehensive naming standardization system that:
    - Analyzes naming conventions across the codebase
    - Identifies inconsistent patterns
    - Provides safe refactoring capabilities
    - Suggests file organization improvements
    """
    
    def __init__(self, root_path: str, backup_dir: str = "backups/naming"):
        self.root_path = Path(root_path)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # File extensions to analyze
        self.code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx'}
        
        # Directories to exclude
        self.exclude_patterns = {
            '__pycache__', '.git', '.pytest_cache', 'node_modules',
            '.venv', 'venv', 'dist', 'build', '.next'
        }
        
        # Naming convention patterns
        self.naming_patterns = {
            'snake_case': re.compile(r'^[a-z][a-z0-9_]*$'),
            'camelCase': re.compile(r'^[a-z][a-zA-Z0-9]*$'),
            'PascalCase': re.compile(r'^[A-Z][a-zA-Z0-9]*$'),
            'kebab-case': re.compile(r'^[a-z][a-z0-9-]*$'),
            'UPPER_CASE': re.compile(r'^[A-Z][A-Z0-9_]*$')
        }
        
        # Common prefixes/suffixes that indicate purpose
        self.purpose_indicators = {
            'test_': 'test',
            '_test': 'test',
            'demo_': 'demo',
            'example_': 'example',
            'temp_': 'temporary',
            '_temp': 'temporary',
            'backup_': 'backup',
            '_backup': 'backup',
            'old_': 'deprecated',
            '_old': 'deprecated'
        }
    
    def analyze_naming_conventions(self) -> NamingReport:
        """
        Perform comprehensive naming convention analysis
        
        Returns:
            NamingReport with all findings and recommendations
        """
        print("Starting naming convention analysis...")
        
        # Get files to analyze
        files_to_analyze = self._get_files_to_analyze()
        print(f"Analyzing {len(files_to_analyze)} files...")
        
        # Analyze naming violations
        violations = self._find_naming_violations(files_to_analyze)
        
        # Find inconsistent patterns
        inconsistent_patterns = self._find_inconsistent_patterns(files_to_analyze)
        
        # Generate organization suggestions
        organization_suggestions = self._generate_organization_suggestions(files_to_analyze)
        
        # Create convention summary
        convention_summary = self._create_convention_summary(violations)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            violations, inconsistent_patterns, organization_suggestions
        )
        
        report = NamingReport(
            total_files_analyzed=len(files_to_analyze),
            violations=violations,
            inconsistent_patterns=inconsistent_patterns,
            organization_suggestions=organization_suggestions,
            convention_summary=convention_summary,
            recommendations=recommendations,
            analysis_timestamp=datetime.now().isoformat()
        )
        
        print(f"Analysis complete. Found {len(violations)} naming violations, "
              f"{len(inconsistent_patterns)} inconsistent patterns, "
              f"{len(organization_suggestions)} organization suggestions.")
        
        return report
    
    def _get_files_to_analyze(self) -> List[Path]:
        """Get list of files to analyze for naming conventions"""
        files = []
        
        for root, dirs, filenames in os.walk(self.root_path):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.exclude_patterns)]
            
            for filename in filenames:
                file_path = Path(root) / filename
                
                # Skip if file matches exclude patterns
                if any(pattern in str(file_path) for pattern in self.exclude_patterns):
                    continue
                
                # Include all files for naming analysis (not just code files)
                files.append(file_path)
        
        return files
    
    def _find_naming_violations(self, files: List[Path]) -> List[NamingViolation]:
        """Find naming convention violations in files"""
        violations = []
        
        for file_path in files:
            # Check file naming
            violations.extend(self._check_file_naming(file_path))
            
            # Check code element naming (for code files)
            if file_path.suffix in self.code_extensions:
                violations.extend(self._check_code_element_naming(file_path))
        
        return violations
    
    def _check_file_naming(self, file_path: Path) -> List[NamingViolation]:
        """Check file and directory naming conventions"""
        violations = []
        
        # Check filename
        filename = file_path.stem  # Without extension
        if filename and not self._is_valid_name(filename, 'snake_case'):
            # Skip special files
            if filename.startswith('.') or filename in ['__init__', '__main__']:
                pass
            else:
                suggested_name = self._convert_to_convention(filename, 'snake_case')
                violations.append(NamingViolation(
                    name=filename,
                    current_convention=self._detect_convention(filename),
                    expected_convention='snake_case',
                    suggested_name=suggested_name,
                    file_path=str(file_path),
                    line_number=0,
                    element_type='file',
                    severity='warning'
                ))
        
        # Check directory names in path
        for part in file_path.parts[:-1]:  # Exclude filename
            if part and not self._is_valid_name(part, 'snake_case'):
                # Skip special directories
                if part.startswith('.') or part in ['__pycache__']:
                    continue
                
                suggested_name = self._convert_to_convention(part, 'snake_case')
                violations.append(NamingViolation(
                    name=part,
                    current_convention=self._detect_convention(part),
                    expected_convention='snake_case',
                    suggested_name=suggested_name,
                    file_path=str(file_path.parent),
                    line_number=0,
                    element_type='directory',
                    severity='info'
                ))
        
        return violations
    
    def _check_code_element_naming(self, file_path: Path) -> List[NamingViolation]:
        """Check naming conventions in code files"""
        violations = []
        
        if file_path.suffix == '.py':
            violations.extend(self._check_python_naming(file_path))
        elif file_path.suffix in {'.js', '.ts', '.jsx', '.tsx'}:
            violations.extend(self._check_javascript_naming(file_path))
        
        return violations
    
    def _check_python_naming(self, file_path: Path) -> List[NamingViolation]:
        """Check Python naming conventions (PEP 8)"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Functions should be snake_case
                    if not self._is_valid_name(node.name, 'snake_case'):
                        # Skip magic methods
                        if not (node.name.startswith('__') and node.name.endswith('__')):
                            violations.append(NamingViolation(
                                name=node.name,
                                current_convention=self._detect_convention(node.name),
                                expected_convention='snake_case',
                                suggested_name=self._convert_to_convention(node.name, 'snake_case'),
                                file_path=str(file_path),
                                line_number=node.lineno,
                                element_type='function',
                                severity='warning'
                            ))
                
                elif isinstance(node, ast.ClassDef):
                    # Classes should be PascalCase
                    if not self._is_valid_name(node.name, 'PascalCase'):
                        violations.append(NamingViolation(
                            name=node.name,
                            current_convention=self._detect_convention(node.name),
                            expected_convention='PascalCase',
                            suggested_name=self._convert_to_convention(node.name, 'PascalCase'),
                            file_path=str(file_path),
                            line_number=node.lineno,
                            element_type='class',
                            severity='error'
                        ))
                
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    # Variables should be snake_case
                    var_name = node.id
                    if (not self._is_valid_name(var_name, 'snake_case') and 
                        not keyword.iskeyword(var_name) and
                        not var_name.startswith('_') and
                        len(var_name) > 1):
                        
                        violations.append(NamingViolation(
                            name=var_name,
                            current_convention=self._detect_convention(var_name),
                            expected_convention='snake_case',
                            suggested_name=self._convert_to_convention(var_name, 'snake_case'),
                            file_path=str(file_path),
                            line_number=node.lineno,
                            element_type='variable',
                            severity='info'
                        ))
        
        except Exception as e:
            print(f"Error checking Python naming in {file_path}: {e}")
        
        return violations
    
    def _check_javascript_naming(self, file_path: Path) -> List[NamingViolation]:
        """Check JavaScript/TypeScript naming conventions"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple regex-based analysis for JavaScript
            # Function declarations
            func_matches = re.finditer(r'function\s+(\w+)', content)
            for match in func_matches:
                func_name = match.group(1)
                if not self._is_valid_name(func_name, 'camelCase'):
                    line_num = content[:match.start()].count('\n') + 1
                    violations.append(NamingViolation(
                        name=func_name,
                        current_convention=self._detect_convention(func_name),
                        expected_convention='camelCase',
                        suggested_name=self._convert_to_convention(func_name, 'camelCase'),
                        file_path=str(file_path),
                        line_number=line_num,
                        element_type='function',
                        severity='warning'
                    ))
            
            # Class declarations
            class_matches = re.finditer(r'class\s+(\w+)', content)
            for match in class_matches:
                class_name = match.group(1)
                if not self._is_valid_name(class_name, 'PascalCase'):
                    line_num = content[:match.start()].count('\n') + 1
                    violations.append(NamingViolation(
                        name=class_name,
                        current_convention=self._detect_convention(class_name),
                        expected_convention='PascalCase',
                        suggested_name=self._convert_to_convention(class_name, 'PascalCase'),
                        file_path=str(file_path),
                        line_number=line_num,
                        element_type='class',
                        severity='error'
                    ))
        
        except Exception as e:
            print(f"Error checking JavaScript naming in {file_path}: {e}")
        
        return violations
    
    def _is_valid_name(self, name: str, convention: str) -> bool:
        """Check if name follows the specified convention"""
        if not name or convention not in self.naming_patterns:
            return False
        
        return bool(self.naming_patterns[convention].match(name))
    
    def _detect_convention(self, name: str) -> str:
        """Detect the naming convention used by a name"""
        for convention, pattern in self.naming_patterns.items():
            if pattern.match(name):
                return convention
        
        return 'unknown'
    
    def _convert_to_convention(self, name: str, target_convention: str) -> str:
        """Convert name to target naming convention"""
        if not name:
            return name
        
        # Split name into words
        words = self._split_name_into_words(name)
        
        if target_convention == 'snake_case':
            return '_'.join(word.lower() for word in words)
        elif target_convention == 'camelCase':
            if not words:
                return name
            return words[0].lower() + ''.join(word.capitalize() for word in words[1:])
        elif target_convention == 'PascalCase':
            return ''.join(word.capitalize() for word in words)
        elif target_convention == 'kebab-case':
            return '-'.join(word.lower() for word in words)
        elif target_convention == 'UPPER_CASE':
            return '_'.join(word.upper() for word in words)
        
        return name
    
    def _split_name_into_words(self, name: str) -> List[str]:
        """Split a name into constituent words"""
        # Handle different naming conventions
        words = []
        
        # Split on underscores and hyphens
        parts = re.split(r'[_-]', name)
        
        for part in parts:
            if not part:
                continue
            
            # Split camelCase and PascalCase
            camel_words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', part)
            if camel_words:
                words.extend(camel_words)
            else:
                words.append(part)
        
        return [word for word in words if word]
    
    def _find_inconsistent_patterns(self, files: List[Path]) -> List[InconsistentPattern]:
        """Find inconsistent naming patterns across the codebase"""
        patterns = []
        
        # Group files by type and analyze patterns
        file_groups = {
            'python': [f for f in files if f.suffix == '.py'],
            'javascript': [f for f in files if f.suffix in {'.js', '.ts', '.jsx', '.tsx'}],
            'config': [f for f in files if f.suffix in {'.json', '.yaml', '.yml', '.ini'}],
            'docs': [f for f in files if f.suffix in {'.md', '.rst', '.txt'}]
        }
        
        for group_name, group_files in file_groups.items():
            if len(group_files) < 2:
                continue
            
            # Analyze naming patterns within each group
            naming_styles = {}
            for file_path in group_files:
                style = self._detect_convention(file_path.stem)
                if style not in naming_styles:
                    naming_styles[style] = []
                naming_styles[style].append(str(file_path))
            
            # If multiple styles are used, it's inconsistent
            if len(naming_styles) > 1:
                most_common_style = max(naming_styles.keys(), key=lambda k: len(naming_styles[k]))
                patterns.append(InconsistentPattern(
                    pattern_type=f'{group_name}_files',
                    variations=list(naming_styles.keys()),
                    files_affected=[f for files in naming_styles.values() for f in files],
                    recommended_standard=most_common_style
                ))
        
        return patterns
    
    def _generate_organization_suggestions(self, files: List[Path]) -> List[OrganizationSuggestion]:
        """Generate file organization suggestions"""
        suggestions = []
        
        # Group files by purpose based on naming patterns
        purpose_groups = {}
        
        for file_path in files:
            filename = file_path.name.lower()
            purpose = self._detect_file_purpose(filename)
            
            if purpose not in purpose_groups:
                purpose_groups[purpose] = []
            purpose_groups[purpose].append(file_path)
        
        # Suggest organization for scattered files of same purpose
        for purpose, files_list in purpose_groups.items():
            if purpose == 'unknown' or len(files_list) < 2:
                continue
            
            # Check if files are scattered across directories
            directories = set(f.parent for f in files_list)
            if len(directories) > 1:
                # Suggest consolidation
                suggested_dir = self._suggest_directory_for_purpose(purpose)
                
                for file_path in files_list:
                    if file_path.parent.name != suggested_dir:
                        suggested_path = self.root_path / suggested_dir / file_path.name
                        suggestions.append(OrganizationSuggestion(
                            current_path=str(file_path),
                            suggested_path=str(suggested_path),
                            reason=f"Consolidate {purpose} files in {suggested_dir}/ directory",
                            impact_level='medium'
                        ))
        
        return suggestions
    
    def _detect_file_purpose(self, filename: str) -> str:
        """Detect the purpose of a file based on its name"""
        for indicator, purpose in self.purpose_indicators.items():
            if indicator in filename:
                return purpose
        
        # Additional purpose detection based on common patterns
        if 'config' in filename or filename.endswith('.json') or filename.endswith('.yaml'):
            return 'configuration'
        elif 'doc' in filename or filename.endswith('.md') or filename.endswith('.rst'):
            return 'documentation'
        elif 'script' in filename or filename.endswith('.sh') or filename.endswith('.bat'):
            return 'script'
        elif 'util' in filename or 'helper' in filename:
            return 'utility'
        
        return 'unknown'
    
    def _suggest_directory_for_purpose(self, purpose: str) -> str:
        """Suggest appropriate directory name for a file purpose"""
        directory_mapping = {
            'test': 'tests',
            'demo': 'examples',
            'example': 'examples',
            'temporary': 'temp',
            'backup': 'backups',
            'deprecated': 'deprecated',
            'configuration': 'config',
            'documentation': 'docs',
            'script': 'scripts',
            'utility': 'utils'
        }
        
        return directory_mapping.get(purpose, purpose)
    
    def _create_convention_summary(self, violations: List[NamingViolation]) -> Dict[str, int]:
        """Create summary of naming conventions found"""
        summary = {}
        
        for violation in violations:
            conv = violation.current_convention
            if conv not in summary:
                summary[conv] = 0
            summary[conv] += 1
        
        return summary
    
    def _generate_recommendations(self, violations: List[NamingViolation],
                                inconsistent_patterns: List[InconsistentPattern],
                                organization_suggestions: List[OrganizationSuggestion]) -> List[str]:
        """Generate recommendations for naming improvements"""
        recommendations = []
        
        if violations:
            error_violations = [v for v in violations if v.severity == 'error']
            warning_violations = [v for v in violations if v.severity == 'warning']
            
            if error_violations:
                recommendations.append(f"Fix {len(error_violations)} critical naming violations (classes, etc.)")
            
            if warning_violations:
                recommendations.append(f"Address {len(warning_violations)} naming warnings (functions, etc.)")
            
            # Group by type
            by_type = {}
            for violation in violations:
                if violation.element_type not in by_type:
                    by_type[violation.element_type] = 0
                by_type[violation.element_type] += 1
            
            for element_type, count in by_type.items():
                recommendations.append(f"Standardize {count} {element_type} names")
        
        if inconsistent_patterns:
            recommendations.append(f"Resolve {len(inconsistent_patterns)} inconsistent naming patterns")
        
        if organization_suggestions:
            high_impact = [s for s in organization_suggestions if s.impact_level == 'high']
            if high_impact:
                recommendations.append(f"Prioritize {len(high_impact)} high-impact organization changes")
            
            recommendations.append(f"Consider {len(organization_suggestions)} file organization improvements")
        
        return recommendations
    
    def apply_naming_fixes(self, report: NamingReport, target_convention: str) -> Dict[str, str]:
        """
        Apply naming fixes based on the report
        
        Args:
            report: NamingReport from analysis
            target_convention: Target naming convention to apply
            
        Returns:
            Dict mapping operation to result message
        """
        results = {}
        
        # Create backup first
        all_files = set()
        for violation in report.violations:
            all_files.add(violation.file_path)
        
        if all_files:
            backup_path = self.create_backup(list(all_files))
            results['backup'] = f"Created backup at {backup_path}"
            
            # Apply fixes by severity (errors first)
            error_fixes = [v for v in report.violations if v.severity == 'error']
            warning_fixes = [v for v in report.violations if v.severity == 'warning']
            
            if error_fixes:
                fixed_errors = self._apply_naming_fixes_batch(error_fixes)
                results['errors'] = f"Fixed {fixed_errors} critical naming issues"
            
            if warning_fixes:
                fixed_warnings = self._apply_naming_fixes_batch(warning_fixes)
                results['warnings'] = f"Fixed {fixed_warnings} naming warnings"
        
        return results
    
    def _apply_naming_fixes_batch(self, violations: List[NamingViolation]) -> int:
        """Apply a batch of naming fixes"""
        fixed_count = 0
        
        # Group by file to minimize file operations
        files_violations = {}
        for violation in violations:
            if violation.file_path not in files_violations:
                files_violations[violation.file_path] = []
            files_violations[violation.file_path].append(violation)
        
        for file_path, file_violations in files_violations.items():
            try:
                if self._apply_file_naming_fixes(file_path, file_violations):
                    fixed_count += len(file_violations)
            except Exception as e:
                print(f"Error applying fixes to {file_path}: {e}")
        
        return fixed_count
    
    def _apply_file_naming_fixes(self, file_path: str, violations: List[NamingViolation]) -> bool:
        """Apply naming fixes to a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply fixes (simplified - would need more sophisticated refactoring)
            modified_content = content
            
            for violation in violations:
                if violation.element_type in ['function', 'class', 'variable']:
                    # Simple find-and-replace (in real implementation, would use AST)
                    old_name = violation.name
                    new_name = violation.suggested_name
                    
                    # Be careful with replacements to avoid false positives
                    pattern = r'\b' + re.escape(old_name) + r'\b'
                    modified_content = re.sub(pattern, new_name, modified_content)
            
            # Write back if changed
            if modified_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                return True
        
        except Exception as e:
            print(f"Error fixing file {file_path}: {e}")
            return False
        
        return False
    
    def create_backup(self, files_to_backup: List[str]) -> str:
        """Create backup of files before applying fixes"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"naming_fixes_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        for file_path in files_to_backup:
            src_path = Path(file_path)
            if src_path.exists():
                rel_path = src_path.relative_to(self.root_path)
                backup_file_path = backup_path / rel_path
                backup_file_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, backup_file_path)
        
        # Create backup manifest
        manifest = {
            'timestamp': timestamp,
            'files': files_to_backup,
            'backup_path': str(backup_path)
        }
        
        with open(backup_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return str(backup_path)
    
    def save_report(self, report: NamingReport, output_path: str) -> None:
        """Save naming analysis report to file"""
        with open(output_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"Naming report saved to {output_path}")


def main():
    """Example usage of NamingStandardizer"""
    standardizer = NamingStandardizer(".")
    
    # Analyze naming conventions
    report = standardizer.analyze_naming_conventions()
    
    # Save report
    standardizer.save_report(report, "naming_report.json")
    
    # Print summary
    print(f"\nNaming Convention Analysis Summary:")
    print(f"Files analyzed: {report.total_files_analyzed}")
    print(f"Naming violations: {len(report.violations)}")
    print(f"Inconsistent patterns: {len(report.inconsistent_patterns)}")
    print(f"Organization suggestions: {len(report.organization_suggestions)}")
    
    print(f"\nConvention Summary:")
    for convention, count in report.convention_summary.items():
        print(f"  {convention}: {count}")
    
    for rec in report.recommendations:
        print(f"- {rec}")


if __name__ == "__main__":
    main()