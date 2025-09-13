import pytest
"""
Test Coverage Analyzer - Comprehensive code coverage measurement and reporting
"""

import logging
import subprocess
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
import os

from orchestrator import TestCategory, TestConfig

logger = logging.getLogger(__name__)


@dataclass
class FileCoverage:
    """Coverage information for a single file"""
    file_path: Path
    total_lines: int
    covered_lines: int
    missing_lines: List[int]
    excluded_lines: List[int] = field(default_factory=list)
    
    @property
    def coverage_percentage(self) -> float:
        """Calculate coverage percentage for this file"""
        if self.total_lines == 0:
            return 100.0
        return (self.covered_lines / self.total_lines) * 100
    
    @property
    def is_fully_covered(self) -> bool:
        """Check if file has 100% coverage"""
        return len(self.missing_lines) == 0


@dataclass
class ModuleCoverage:
    """Coverage information for a module (directory)"""
    module_name: str
    files: Dict[str, FileCoverage] = field(default_factory=dict)
    
    @property
    def total_lines(self) -> int:
        return sum(f.total_lines for f in self.files.values())
    
    @property
    def covered_lines(self) -> int:
        return sum(f.covered_lines for f in self.files.values())
    
    @property
    def coverage_percentage(self) -> float:
        if self.total_lines == 0:
            return 100.0
        return (self.covered_lines / self.total_lines) * 100


@dataclass
class CoverageReport:
    """Complete coverage report"""
    timestamp: datetime
    total_lines: int
    covered_lines: int
    coverage_percentage: float
    threshold_met: bool
    threshold_value: float
    modules: Dict[str, ModuleCoverage] = field(default_factory=dict)
    files: Dict[str, FileCoverage] = field(default_factory=dict)
    categories_tested: List[TestCategory] = field(default_factory=list)
    
    @property
    def uncovered_lines(self) -> int:
        return self.total_lines - self.covered_lines
    
    @property
    def files_with_low_coverage(self) -> List[FileCoverage]:
        """Get files below the coverage threshold"""
        return [f for f in self.files.values() 
                if f.coverage_percentage < self.threshold_value]
    
    @property
    def fully_covered_files(self) -> List[FileCoverage]:
        """Get files with 100% coverage"""
        return [f for f in self.files.values() if f.is_fully_covered]


@dataclass
class CoverageTrend:
    """Coverage trend analysis over time"""
    current_coverage: float
    previous_coverage: Optional[float]
    trend_direction: str  # 'up', 'down', 'stable'
    change_percentage: float
    
    @property
    def is_improving(self) -> bool:
        return self.trend_direction == 'up'
    
    @property
    def is_declining(self) -> bool:
        return self.trend_direction == 'down'


@dataclass
class CoverageHistory:
    """Historical coverage data"""
    reports: List[Tuple[datetime, float]] = field(default_factory=list)
    
    def add_report(self, timestamp: datetime, coverage: float):
        """Add a coverage report to history"""
        self.reports.append((timestamp, coverage))
        # Keep only last 30 reports
        if len(self.reports) > 30:
            self.reports = self.reports[-30:]
    
    def get_trend(self) -> Optional[CoverageTrend]:
        """Calculate coverage trend"""
        if len(self.reports) < 2:
            return None
        
        current = self.reports[-1][1]
        previous = self.reports[-2][1]
        
        change = current - previous
        change_percentage = (change / previous * 100) if previous > 0 else 0
        
        if abs(change) < 0.1:  # Less than 0.1% change
            trend_direction = 'stable'
        elif change > 0:
            trend_direction = 'up'
        else:
            trend_direction = 'down'
        
        return CoverageTrend(
            current_coverage=current,
            previous_coverage=previous,
            trend_direction=trend_direction,
            change_percentage=change_percentage
        )


class CoverageThresholdValidator:
    """Validates coverage against thresholds and policies"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.global_threshold = config.coverage.get('minimum_threshold', 70)
        self.file_threshold = config.coverage.get('file_threshold', 60)
        self.module_threshold = config.coverage.get('module_threshold', 75)
    
    def validate_coverage(self, report: CoverageReport) -> Dict[str, Any]:
        """
        Validate coverage report against all thresholds
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'global_threshold_met': report.coverage_percentage >= self.global_threshold,
            'global_threshold': self.global_threshold,
            'global_coverage': report.coverage_percentage,
            'violations': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check global threshold
        if not results['global_threshold_met']:
            results['violations'].append({
                'type': 'global_threshold',
                'message': f"Global coverage {report.coverage_percentage:.1f}% below threshold {self.global_threshold}%",
                'severity': 'error'
            })
        
        # Check file-level thresholds
        low_coverage_files = []
        for file_path, file_coverage in report.files.items():
            if file_coverage.coverage_percentage < self.file_threshold:
                low_coverage_files.append({
                    'file': file_path,
                    'coverage': file_coverage.coverage_percentage,
                    'threshold': self.file_threshold
                })
        
        if low_coverage_files:
            results['violations'].append({
                'type': 'file_threshold',
                'message': f"{len(low_coverage_files)} files below {self.file_threshold}% threshold",
                'details': low_coverage_files,
                'severity': 'warning'
            })
        
        # Check module-level thresholds
        low_coverage_modules = []
        for module_name, module_coverage in report.modules.items():
            if module_coverage.coverage_percentage < self.module_threshold:
                low_coverage_modules.append({
                    'module': module_name,
                    'coverage': module_coverage.coverage_percentage,
                    'threshold': self.module_threshold
                })
        
        if low_coverage_modules:
            results['violations'].append({
                'type': 'module_threshold',
                'message': f"{len(low_coverage_modules)} modules below {self.module_threshold}% threshold",
                'details': low_coverage_modules,
                'severity': 'warning'
            })
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(report)
        
        return results
    
    def _generate_recommendations(self, report: CoverageReport) -> List[Dict[str, str]]:
        """Generate actionable recommendations for improving coverage"""
        recommendations = []
        
        # Files with no coverage
        uncovered_files = [f for f in report.files.values() if f.coverage_percentage == 0]
        if uncovered_files:
            recommendations.append({
                'type': 'uncovered_files',
                'priority': 'high',
                'message': f"Add tests for {len(uncovered_files)} completely uncovered files",
                'action': 'Create basic test files for uncovered modules'
            })
        
        # Files with low coverage
        low_coverage = [f for f in report.files.values() 
                       if 0 < f.coverage_percentage < 50]
        if low_coverage:
            recommendations.append({
                'type': 'low_coverage',
                'priority': 'medium',
                'message': f"Improve coverage for {len(low_coverage)} files with <50% coverage",
                'action': 'Focus on testing critical paths and edge cases'
            })
        
        # Missing edge case coverage
        partial_coverage = [f for f in report.files.values() 
                          if 50 <= f.coverage_percentage < 80]
        if partial_coverage:
            recommendations.append({
                'type': 'edge_cases',
                'priority': 'low',
                'message': f"Add edge case tests for {len(partial_coverage)} files",
                'action': 'Review missing lines and add tests for error conditions'
            })
        
        return recommendations


class CoverageAnalyzer:
    """
    Main coverage analyzer with measurement, reporting, and trend analysis
    """
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.threshold_validator = CoverageThresholdValidator(config)
        self.coverage_history = CoverageHistory()
        self.project_root = Path.cwd()
        self.coverage_data_dir = Path("test_results/coverage")
        self.coverage_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load historical data
        self._load_coverage_history()
    
    def measure_coverage(self, test_categories: List[TestCategory], 
                        source_dirs: Optional[List[Path]] = None) -> CoverageReport:
        """
        Measure code coverage for specified test categories
        
        Args:
            test_categories: Categories of tests to run for coverage
            source_dirs: Source directories to measure coverage for
            
        Returns:
            CoverageReport with detailed coverage information
        """
        logger.info(f"Measuring coverage for categories: {[c.value for c in test_categories]}")
        
        if source_dirs is None:
            source_dirs = self._get_default_source_dirs()
        
        # Run tests with coverage
        coverage_data = self._run_tests_with_coverage(test_categories, source_dirs)
        
        # Parse coverage results
        report = self._parse_coverage_data(coverage_data, test_categories)
        
        # Validate against thresholds
        report.threshold_met = report.coverage_percentage >= self.threshold_validator.global_threshold
        report.threshold_value = self.threshold_validator.global_threshold
        
        # Add to history
        self.coverage_history.add_report(report.timestamp, report.coverage_percentage)
        self._save_coverage_history()
        
        logger.info(f"Coverage measurement complete: {report.coverage_percentage:.1f}%")
        return report
    
    def _get_default_source_dirs(self) -> List[Path]:
        """Get default source directories to measure coverage for"""
        potential_dirs = ['backend', 'frontend/src', 'core', 'tools', 'scripts']
        source_dirs = []
        
        for dir_name in potential_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                source_dirs.append(dir_path)
        
        return source_dirs
    
    def _run_tests_with_coverage(self, test_categories: List[TestCategory], 
                               source_dirs: List[Path]) -> Dict[str, Any]:
        """Run tests with coverage measurement"""
        
        # Prepare coverage command
        source_paths = [str(d) for d in source_dirs]
        coverage_file = self.coverage_data_dir / f"coverage_{int(datetime.now().timestamp())}.xml"
        
        # Build pytest command with coverage
        command = [
            sys.executable, '-m', 'pytest',
            '--cov=' + ','.join(source_paths),
            '--cov-report=xml:' + str(coverage_file),
            '--cov-report=json:' + str(coverage_file.with_suffix('.json')),
            '--cov-report=html:' + str(self.coverage_data_dir / 'html'),
            '--cov-fail-under=' + str(self.threshold_validator.global_threshold)
        ]
        
        # Add test paths for categories
        for category in test_categories:
            category_config = self.config.categories.get(category.value, {})
            patterns = category_config.get('patterns', [])
            for pattern in patterns:
                command.append(pattern)
        
        # Execute coverage run
        try:
            logger.debug(f"Running coverage command: {' '.join(command)}")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=600  # 10 minute timeout
            )
            
            return {
                'xml_file': coverage_file,
                'json_file': coverage_file.with_suffix('.json'),
                'html_dir': self.coverage_data_dir / 'html',
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Coverage measurement timed out")
            raise
        except Exception as e:
            logger.error(f"Error running coverage: {e}")
            raise
    
    def _parse_coverage_data(self, coverage_data: Dict[str, Any], 
                           test_categories: List[TestCategory]) -> CoverageReport:
        """Parse coverage data from XML and JSON reports"""
        
        # Try to parse JSON report first (more detailed)
        json_file = coverage_data['json_file']
        if json_file.exists():
            return self._parse_json_coverage(json_file, test_categories)
        
        # Fallback to XML report
        xml_file = coverage_data['xml_file']
        if xml_file.exists():
            return self._parse_xml_coverage(xml_file, test_categories)
        
        # If no coverage files, create empty report
        logger.warning("No coverage data files found, creating empty report")
        return CoverageReport(
            timestamp=datetime.now(),
            total_lines=0,
            covered_lines=0,
            coverage_percentage=0.0,
            threshold_met=False,
            threshold_value=self.threshold_validator.global_threshold,
            categories_tested=test_categories
        )
    
    def _parse_json_coverage(self, json_file: Path, 
                           test_categories: List[TestCategory]) -> CoverageReport:
        """Parse JSON coverage report"""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            files = {}
            modules = {}
            total_lines = 0
            covered_lines = 0
            
            # Parse file coverage
            for file_path, file_data in data.get('files', {}).items():
                summary = file_data.get('summary', {})
                
                file_coverage = FileCoverage(
                    file_path=Path(file_path),
                    total_lines=summary.get('num_statements', 0),
                    covered_lines=summary.get('covered_lines', 0),
                    missing_lines=file_data.get('missing_lines', []),
                    excluded_lines=file_data.get('excluded_lines', [])
                )
                
                files[file_path] = file_coverage
                total_lines += file_coverage.total_lines
                covered_lines += file_coverage.covered_lines
                
                # Group by module
                module_name = str(Path(file_path).parent)
                if module_name not in modules:
                    modules[module_name] = ModuleCoverage(module_name=module_name)
                modules[module_name].files[file_path] = file_coverage
            
            # Calculate overall coverage
            coverage_percentage = (covered_lines / total_lines * 100) if total_lines > 0 else 0.0
            
            return CoverageReport(
                timestamp=datetime.now(),
                total_lines=total_lines,
                covered_lines=covered_lines,
                coverage_percentage=coverage_percentage,
                threshold_met=coverage_percentage >= self.threshold_validator.global_threshold,
                threshold_value=self.threshold_validator.global_threshold,
                files=files,
                modules=modules,
                categories_tested=test_categories
            )
            
        except Exception as e:
            logger.error(f"Error parsing JSON coverage report: {e}")
            raise
    
    def _parse_xml_coverage(self, xml_file: Path, 
                          test_categories: List[TestCategory]) -> CoverageReport:
        """Parse XML coverage report (fallback)"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            files = {}
            modules = {}
            total_lines = 0
            covered_lines = 0
            
            # Parse coverage data from XML
            for package in root.findall('.//package'):
                for class_elem in package.findall('classes/class'):
                    filename = class_elem.get('filename', '')
                    
                    lines_elem = class_elem.find('lines')
                    if lines_elem is not None:
                        line_count = len(lines_elem.findall('line'))
                        covered_count = len(lines_elem.findall('line[@hits>0]'))
                        
                        file_coverage = FileCoverage(
                            file_path=Path(filename),
                            total_lines=line_count,
                            covered_lines=covered_count,
                            missing_lines=[]  # Would need to parse from XML
                        )
                        
                        files[filename] = file_coverage
                        total_lines += line_count
                        covered_lines += covered_count
            
            coverage_percentage = (covered_lines / total_lines * 100) if total_lines > 0 else 0.0
            
            return CoverageReport(
                timestamp=datetime.now(),
                total_lines=total_lines,
                covered_lines=covered_lines,
                coverage_percentage=coverage_percentage,
                threshold_met=coverage_percentage >= self.threshold_validator.global_threshold,
                threshold_value=self.threshold_validator.global_threshold,
                files=files,
                modules=modules,
                categories_tested=test_categories
            )
            
        except Exception as e:
            logger.error(f"Error parsing XML coverage report: {e}")
            raise
    
    def validate_coverage_thresholds(self, report: CoverageReport) -> Dict[str, Any]:
        """Validate coverage report against configured thresholds"""
        return self.threshold_validator.validate_coverage(report)
    
    def get_coverage_trend(self) -> Optional[CoverageTrend]:
        """Get coverage trend analysis"""
        return self.coverage_history.get_trend()
    
    def generate_coverage_report(self, report: CoverageReport, 
                               output_path: Path, format_type: str = 'html') -> Path:
        """
        Generate formatted coverage report
        
        Args:
            report: Coverage report to format
            output_path: Output file path
            format_type: Report format ('html', 'json', 'markdown')
            
        Returns:
            Path to generated report
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == 'json':
            self._generate_json_report(report, output_path)
        elif format_type == 'markdown':
            self._generate_markdown_report(report, output_path)
        elif format_type == 'html':
            self._generate_html_report(report, output_path)
        else:
            raise ValueError(f"Unsupported report format: {format_type}")
        
        logger.info(f"Coverage report generated: {output_path}")
        return output_path
    
    def _generate_json_report(self, report: CoverageReport, output_path: Path):
        """Generate JSON coverage report"""
        report_data = {
            'timestamp': report.timestamp.isoformat(),
            'summary': {
                'total_lines': report.total_lines,
                'covered_lines': report.covered_lines,
                'coverage_percentage': report.coverage_percentage,
                'threshold_met': report.threshold_met,
                'threshold_value': report.threshold_value
            },
            'files': {
                path: {
                    'total_lines': file_cov.total_lines,
                    'covered_lines': file_cov.covered_lines,
                    'coverage_percentage': file_cov.coverage_percentage,
                    'missing_lines': file_cov.missing_lines
                }
                for path, file_cov in report.files.items()
            },
            'validation': self.validate_coverage_thresholds(report),
            'trend': self.get_coverage_trend().__dict__ if self.get_coverage_trend() else None
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
    
    def _generate_markdown_report(self, report: CoverageReport, output_path: Path):
        """Generate Markdown coverage report"""
        trend = self.get_coverage_trend()
        validation = self.validate_coverage_thresholds(report)
        
        content = f"""# Coverage Report

Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Lines**: {report.total_lines:,}
- **Covered Lines**: {report.covered_lines:,}
- **Coverage**: {report.coverage_percentage:.1f}%
- **Threshold**: {report.threshold_value}%
- **Status**: {'✅ PASS' if report.threshold_met else '❌ FAIL'}

"""
        
        if trend:
            trend_icon = '📈' if trend.is_improving else '📉' if trend.is_declining else '➡️'
            content += f"""## Trend Analysis

{trend_icon} **Coverage Trend**: {trend.trend_direction.upper()}
- Current: {trend.current_coverage:.1f}%
- Previous: {trend.previous_coverage:.1f}%
- Change: {trend.change_percentage:+.1f}%

"""
        
        # Add file details
        content += "## File Coverage\n\n"
        content += "| File | Coverage | Lines | Status |\n"
        content += "|------|----------|-------|--------|\n"
        
        for file_path, file_cov in sorted(report.files.items()):
            status = '✅' if file_cov.coverage_percentage >= 80 else '⚠️' if file_cov.coverage_percentage >= 60 else '❌'
            content += f"| {file_path} | {file_cov.coverage_percentage:.1f}% | {file_cov.covered_lines}/{file_cov.total_lines} | {status} |\n"
        
        # Add recommendations
        if validation['recommendations']:
            content += "\n## Recommendations\n\n"
            for rec in validation['recommendations']:
                priority_icon = '🔴' if rec['priority'] == 'high' else '🟡' if rec['priority'] == 'medium' else '🟢'
                content += f"- {priority_icon} **{rec['type'].title()}**: {rec['message']}\n"
                content += f"  - Action: {rec['action']}\n\n"
        
        with open(output_path, 'w') as f:
            f.write(content)
    
    def _generate_html_report(self, report: CoverageReport, output_path: Path):
        """Generate HTML coverage report"""
        # This would generate a comprehensive HTML report
        # For now, create a simple HTML version
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Coverage Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Coverage Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Coverage:</strong> {report.coverage_percentage:.1f}%</p>
        <p><strong>Status:</strong> <span class="{'pass' if report.threshold_met else 'fail'}">
            {'PASS' if report.threshold_met else 'FAIL'}
        </span></p>
        <p><strong>Generated:</strong> {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <h2>File Coverage</h2>
    <table>
        <tr>
            <th>File</th>
            <th>Coverage</th>
            <th>Lines Covered</th>
            <th>Total Lines</th>
        </tr>
"""
        
        for file_path, file_cov in sorted(report.files.items()):
            html_content += f"""
        <tr>
            <td>{file_path}</td>
            <td>{file_cov.coverage_percentage:.1f}%</td>
            <td>{file_cov.covered_lines}</td>
            <td>{file_cov.total_lines}</td>
        </tr>
"""
        
        html_content += """
    </table>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _load_coverage_history(self):
        """Load coverage history from file"""
        history_file = self.coverage_data_dir / 'coverage_history.json'
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                
                for entry in data.get('reports', []):
                    timestamp = datetime.fromisoformat(entry['timestamp'])
                    coverage = entry['coverage']
                    self.coverage_history.add_report(timestamp, coverage)
                    
            except Exception as e:
                logger.warning(f"Could not load coverage history: {e}")
    
    def _save_coverage_history(self):
        """Save coverage history to file"""
        history_file = self.coverage_data_dir / 'coverage_history.json'
        
        data = {
            'reports': [
                {
                    'timestamp': timestamp.isoformat(),
                    'coverage': coverage
                }
                for timestamp, coverage in self.coverage_history.reports
            ]
        }
        
        try:
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save coverage history: {e}")


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    
    # Load config
    config = TestConfig.load_from_file(Path("tests/config/test-config.yaml"))
    
    # Create analyzer
    analyzer = CoverageAnalyzer(config)
    
    # Measure coverage
    categories = [TestCategory.UNIT, TestCategory.INTEGRATION]
    report = analyzer.measure_coverage(categories)
    
    # Generate reports
    analyzer.generate_coverage_report(report, Path("test_results/coverage_report.html"), "html")
    analyzer.generate_coverage_report(report, Path("test_results/coverage_report.md"), "markdown")
    
    print(f"Coverage: {report.coverage_percentage:.1f}%")
    print(f"Threshold met: {report.threshold_met}")
