"""
Main code quality checking engine.
"""

import ast
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import logging
import yaml
import json

from models import (
    QualityReport, QualityIssue, QualityMetrics, QualityConfig,
    QualityIssueType, QualitySeverity
)
from formatters.code_formatter import CodeFormatter
from validators.documentation_validator import DocumentationValidator
from validators.type_hint_validator import TypeHintValidator
from validators.style_validator import StyleValidator
from analyzers.complexity_analyzer import ComplexityAnalyzer


logger = logging.getLogger(__name__)


class QualityChecker:
    """Main code quality checking engine."""
    
    def __init__(self, config: Optional[QualityConfig] = None):
        """Initialize quality checker with configuration."""
        self.config = config or QualityConfig()
        self.formatter = CodeFormatter(self.config)
        self.doc_validator = DocumentationValidator(self.config)
        self.type_validator = TypeHintValidator(self.config)
        self.style_validator = StyleValidator(self.config)
        self.complexity_analyzer = ComplexityAnalyzer(self.config)
        
    @classmethod
    def from_config_file(cls, config_path: Path) -> 'QualityChecker':
        """Create quality checker from configuration file."""
        if not config_path.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls()
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            config = cls._parse_config(config_data)
            return cls(config)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return cls()
    
    @staticmethod
    def _parse_config(config_data: Dict[str, Any]) -> QualityConfig:
        """Parse configuration data into QualityConfig object."""
        config = QualityConfig()
        
        # Formatting settings
        if 'formatting' in config_data:
            fmt_config = config_data['formatting']
            config.line_length = fmt_config.get('line_length', config.line_length)
            config.use_black = fmt_config.get('use_black', config.use_black)
            config.use_isort = fmt_config.get('use_isort', config.use_isort)
            config.use_autopep8 = fmt_config.get('use_autopep8', config.use_autopep8)
        
        # Documentation settings
        if 'documentation' in config_data:
            doc_config = config_data['documentation']
            config.require_module_docstrings = doc_config.get('require_module_docstrings', config.require_module_docstrings)
            config.require_class_docstrings = doc_config.get('require_class_docstrings', config.require_class_docstrings)
            config.require_function_docstrings = doc_config.get('require_function_docstrings', config.require_function_docstrings)
            config.min_docstring_length = doc_config.get('min_docstring_length', config.min_docstring_length)
        
        # Type hints settings
        if 'type_hints' in config_data:
            type_config = config_data['type_hints']
            config.require_return_types = type_config.get('require_return_types', config.require_return_types)
            config.require_parameter_types = type_config.get('require_parameter_types', config.require_parameter_types)
            config.strict_mode = type_config.get('strict_mode', config.strict_mode)
        
        # Complexity settings
        if 'complexity' in config_data:
            comp_config = config_data['complexity']
            config.max_cyclomatic_complexity = comp_config.get('max_cyclomatic_complexity', config.max_cyclomatic_complexity)
            config.max_function_length = comp_config.get('max_function_length', config.max_function_length)
            config.max_class_length = comp_config.get('max_class_length', config.max_class_length)
            config.max_module_length = comp_config.get('max_module_length', config.max_module_length)
        
        # Style settings
        if 'style' in config_data:
            style_config = config_data['style']
            config.max_line_length = style_config.get('max_line_length', config.max_line_length)
            config.ignore_rules = style_config.get('ignore_rules', config.ignore_rules)
        
        # File patterns
        if 'files' in config_data:
            file_config = config_data['files']
            config.include_patterns = file_config.get('include_patterns', config.include_patterns)
            config.exclude_patterns = file_config.get('exclude_patterns', config.exclude_patterns)
        
        return config
    
    def check_quality(self, path: Path, checks: Optional[List[str]] = None) -> QualityReport:
        """
        Perform comprehensive quality check on the given path.
        
        Args:
            path: Path to check (file or directory)
            checks: List of specific checks to run (None for all)
        
        Returns:
            QualityReport with all issues and metrics
        """
        report = QualityReport(project_path=path)
        
        # Get all Python files to analyze
        python_files = self._get_python_files(path)
        report.files_analyzed = len(python_files)
        
        if not python_files:
            logger.warning(f"No Python files found in {path}")
            return report
        
        # Determine which checks to run
        all_checks = ['formatting', 'style', 'documentation', 'type_hints', 'complexity']
        checks_to_run = checks if checks else all_checks
        
        logger.info(f"Running quality checks: {', '.join(checks_to_run)}")
        
        # Analyze each file
        for file_path in python_files:
            try:
                file_issues, file_metrics = self._analyze_file(file_path, checks_to_run)
                
                # Add issues to report
                for issue in file_issues:
                    report.add_issue(issue)
                
                # Aggregate metrics
                self._aggregate_metrics(report.metrics, file_metrics)
                
            except Exception as e:
                logger.error(f"Failed to analyze {file_path}: {e}")
                # Add error issue
                error_issue = QualityIssue(
                    file_path=file_path,
                    line_number=1,
                    column=1,
                    issue_type=QualityIssueType.STYLE,
                    severity=QualitySeverity.ERROR,
                    message=f"Failed to analyze file: {str(e)}",
                    rule_code="ANALYSIS_ERROR"
                )
                report.add_issue(error_issue)
        
        logger.info(f"Quality check complete. Found {report.total_issues} issues in {report.files_analyzed} files")
        return report
    
    def fix_issues(self, path: Path, auto_fix_only: bool = True) -> QualityReport:
        """
        Automatically fix quality issues where possible.
        
        Args:
            path: Path to fix (file or directory)
            auto_fix_only: Only fix issues marked as auto-fixable
        
        Returns:
            QualityReport showing what was fixed
        """
        logger.info(f"Fixing quality issues in {path}")
        
        # First, run quality check to identify issues
        report = self.check_quality(path)
        
        # Get fixable issues
        fixable_issues = [issue for issue in report.issues if issue.auto_fixable]
        
        if not fixable_issues:
            logger.info("No auto-fixable issues found")
            return report
        
        logger.info(f"Found {len(fixable_issues)} auto-fixable issues")
        
        # Group issues by file
        issues_by_file: Dict[Path, List[QualityIssue]] = {}
        for issue in fixable_issues:
            if issue.file_path not in issues_by_file:
                issues_by_file[issue.file_path] = []
            issues_by_file[issue.file_path].append(issue)
        
        # Fix issues in each file
        fixed_count = 0
        for file_path, file_issues in issues_by_file.items():
            try:
                fixed = self._fix_file_issues(file_path, file_issues)
                fixed_count += fixed
            except Exception as e:
                logger.error(f"Failed to fix issues in {file_path}: {e}")
        
        logger.info(f"Fixed {fixed_count} issues")
        
        # Run quality check again to get updated report
        return self.check_quality(path)
    
    def _get_python_files(self, path: Path) -> List[Path]:
        """Get all Python files in the given path."""
        python_files = []
        
        if path.is_file():
            if path.suffix == '.py':
                python_files.append(path)
        else:
            # Recursively find Python files
            for pattern in self.config.include_patterns:
                python_files.extend(path.rglob(pattern))
        
        # Filter out excluded patterns
        filtered_files = []
        for file_path in python_files:
            exclude = False
            for pattern in self.config.exclude_patterns:
                if pattern in str(file_path):
                    exclude = True
                    break
            if not exclude:
                filtered_files.append(file_path)
        
        return sorted(filtered_files)
    
    def _analyze_file(self, file_path: Path, checks: List[str]) -> tuple[List[QualityIssue], QualityMetrics]:
        """Analyze a single file for quality issues."""
        issues = []
        metrics = QualityMetrics()
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return issues, metrics
        
        # Parse AST for analysis
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            syntax_issue = QualityIssue(
                file_path=file_path,
                line_number=e.lineno or 1,
                column=e.offset or 1,
                issue_type=QualityIssueType.STYLE,
                severity=QualitySeverity.ERROR,
                message=f"Syntax error: {e.msg}",
                rule_code="SYNTAX_ERROR"
            )
            issues.append(syntax_issue)
            return issues, metrics
        
        # Calculate basic metrics
        lines = content.split('\n')
        metrics.total_lines = len(lines)
        metrics.blank_lines = sum(1 for line in lines if not line.strip())
        metrics.comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        metrics.code_lines = metrics.total_lines - metrics.blank_lines - metrics.comment_lines
        
        # Run specific checks
        if 'formatting' in checks:
            formatting_issues = self.formatter.check_formatting(file_path, content)
            issues.extend(formatting_issues)
        
        if 'style' in checks:
            style_issues = self.style_validator.validate_style(file_path, content)
            issues.extend(style_issues)
        
        if 'documentation' in checks:
            doc_issues, doc_metrics = self.doc_validator.validate_documentation(file_path, tree)
            issues.extend(doc_issues)
            metrics.functions_count = doc_metrics.get('functions_count', 0)
            metrics.classes_count = doc_metrics.get('classes_count', 0)
            metrics.modules_count = 1 if doc_metrics.get('has_module_docstring', False) else 0
            metrics.documented_functions = doc_metrics.get('documented_functions', 0)
            metrics.documented_classes = doc_metrics.get('documented_classes', 0)
            metrics.documented_modules = 1 if doc_metrics.get('has_module_docstring', False) else 0
        
        if 'type_hints' in checks:
            type_issues, type_metrics = self.type_validator.validate_type_hints(file_path, tree)
            issues.extend(type_issues)
            metrics.type_annotated_functions = type_metrics.get('annotated_functions', 0)
        
        if 'complexity' in checks:
            complexity_issues, complexity_metrics = self.complexity_analyzer.analyze_complexity(file_path, tree)
            issues.extend(complexity_issues)
            metrics.complexity_score = complexity_metrics.get('average_complexity', 0.0)
            metrics.maintainability_index = complexity_metrics.get('maintainability_index', 0.0)
        
        return issues, metrics
    
    def _aggregate_metrics(self, total_metrics: QualityMetrics, file_metrics: QualityMetrics) -> None:
        """Aggregate file metrics into total metrics."""
        total_metrics.total_lines += file_metrics.total_lines
        total_metrics.code_lines += file_metrics.code_lines
        total_metrics.comment_lines += file_metrics.comment_lines
        total_metrics.blank_lines += file_metrics.blank_lines
        total_metrics.functions_count += file_metrics.functions_count
        total_metrics.classes_count += file_metrics.classes_count
        total_metrics.modules_count += file_metrics.modules_count
        total_metrics.documented_functions += file_metrics.documented_functions
        total_metrics.documented_classes += file_metrics.documented_classes
        total_metrics.documented_modules += file_metrics.documented_modules
        total_metrics.type_annotated_functions += file_metrics.type_annotated_functions
        
        # Average complexity and maintainability
        if file_metrics.complexity_score > 0:
            total_metrics.complexity_score = (
                (total_metrics.complexity_score + file_metrics.complexity_score) / 2
            )
        if file_metrics.maintainability_index > 0:
            total_metrics.maintainability_index = (
                (total_metrics.maintainability_index + file_metrics.maintainability_index) / 2
            )
    
    def _fix_file_issues(self, file_path: Path, issues: List[QualityIssue]) -> int:
        """Fix issues in a specific file."""
        fixed_count = 0
        
        # Group issues by type for efficient fixing
        formatting_issues = [i for i in issues if i.issue_type == QualityIssueType.FORMATTING]
        import_issues = [i for i in issues if i.issue_type == QualityIssueType.IMPORTS]
        
        # Fix formatting issues
        if formatting_issues:
            try:
                if self.formatter.fix_formatting(file_path):
                    fixed_count += len(formatting_issues)
            except Exception as e:
                logger.error(f"Failed to fix formatting in {file_path}: {e}")
        
        # Fix import issues
        if import_issues:
            try:
                if self.formatter.fix_imports(file_path):
                    fixed_count += len(import_issues)
            except Exception as e:
                logger.error(f"Failed to fix imports in {file_path}: {e}")
        
        return fixed_count
    
    def generate_report(self, report: QualityReport, output_format: str = 'json') -> str:
        """Generate formatted report."""
        if output_format.lower() == 'json':
            return json.dumps(report.to_dict(), indent=2)
        elif output_format.lower() == 'yaml':
            return yaml.dump(report.to_dict(), default_flow_style=False)
        else:
            return self._generate_text_report(report)
    
    def _generate_text_report(self, report: QualityReport) -> str:
        """Generate human-readable text report."""
        lines = []
        lines.append("=" * 60)
        lines.append("CODE QUALITY REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Project: {report.project_path}")
        lines.append("")
        
        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 20)
        lines.append(f"Files analyzed: {report.files_analyzed}")
        lines.append(f"Total issues: {report.total_issues}")
        lines.append(f"  Errors: {report.errors}")
        lines.append(f"  Warnings: {report.warnings}")
        lines.append(f"  Info: {report.infos}")
        lines.append(f"Auto-fixable: {report.auto_fixable_issues}")
        lines.append(f"Quality score: {report.quality_score:.1f}/100")
        lines.append("")
        
        # Metrics
        lines.append("METRICS")
        lines.append("-" * 20)
        lines.append(f"Total lines: {report.metrics.total_lines}")
        lines.append(f"Code lines: {report.metrics.code_lines}")
        lines.append(f"Functions: {report.metrics.functions_count}")
        lines.append(f"Classes: {report.metrics.classes_count}")
        lines.append(f"Documentation coverage: {report.metrics.documentation_coverage:.1f}%")
        lines.append(f"Type hint coverage: {report.metrics.type_hint_coverage:.1f}%")
        lines.append(f"Average complexity: {report.metrics.complexity_score:.1f}")
        lines.append("")
        
        # Issues by type
        if report.issues:
            lines.append("ISSUES BY TYPE")
            lines.append("-" * 20)
            issue_types = {}
            for issue in report.issues:
                issue_type = issue.issue_type.value
                if issue_type not in issue_types:
                    issue_types[issue_type] = 0
                issue_types[issue_type] += 1
            
            for issue_type, count in sorted(issue_types.items()):
                lines.append(f"{issue_type}: {count}")
            lines.append("")
        
        # Top issues
        if report.issues:
            lines.append("TOP ISSUES")
            lines.append("-" * 20)
            # Show first 10 errors and warnings
            top_issues = [i for i in report.issues if i.severity in [QualitySeverity.ERROR, QualitySeverity.WARNING]][:10]
            for issue in top_issues:
                lines.append(f"{issue.file_path}:{issue.line_number} [{issue.severity.value}] {issue.message}")
            lines.append("")
        
        return "\n".join(lines)
