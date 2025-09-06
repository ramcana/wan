import pytest
#!/usr/bin/env python3
"""
Test Coverage Analyzer

Comprehensive test coverage analysis tool that provides detailed insights into
code coverage, identifies untested code paths, and generates actionable recommendations.
"""

import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import xml.etree.ElementTree as ET


@dataclass
class FileCoverage:
    """Coverage information for a single file"""
    file_path: str
    total_lines: int
    covered_lines: int
    missing_lines: List[int]
    coverage_percentage: float
    functions: Dict[str, Dict[str, Any]]  # function_name -> coverage info
    classes: Dict[str, Dict[str, Any]]    # class_name -> coverage info
    branches: Optional[Dict[str, Any]] = None


@dataclass
class CoverageGap:
    """Represents a gap in test coverage"""
    file_path: str
    gap_type: str  # 'function', 'class', 'branch', 'lines'
    name: str
    line_start: int
    line_end: int
    severity: str  # 'critical', 'high', 'medium', 'low'
    suggestion: str


@dataclass
class CoverageReport:
    """Complete coverage analysis report"""
    total_files: int
    covered_files: int
    total_lines: int
    covered_lines: int
    overall_percentage: float
    file_coverages: List[FileCoverage]
    coverage_gaps: List[CoverageGap]
    recommendations: List[str]
    threshold_violations: List[str]
    trend_analysis: Optional[Dict[str, Any]] = None


class CoverageThresholdManager:
    """Manages coverage thresholds and violations"""
    
    def __init__(self):
        self.thresholds = {
            'overall': 80.0,
            'file': 70.0,
            'function': 75.0,
            'class': 80.0,
            'branch': 70.0
        }
        
        self.critical_files = set()  # Files that must have high coverage
        self.excluded_files = set()  # Files excluded from coverage requirements
    
    def set_threshold(self, threshold_type: str, value: float):
        """Set coverage threshold"""
        self.thresholds[threshold_type] = value
    
    def add_critical_file(self, file_pattern: str):
        """Add file pattern that requires high coverage"""
        self.critical_files.add(file_pattern)
    
    def exclude_file(self, file_pattern: str):
        """Exclude file pattern from coverage requirements"""
        self.excluded_files.add(file_pattern)
    
    def check_violations(self, coverage_report: CoverageReport) -> List[str]:
        """Check for threshold violations"""
        violations = []
        
        # Check overall coverage
        if coverage_report.overall_percentage < self.thresholds['overall']:
            violations.append(
                f"Overall coverage {coverage_report.overall_percentage:.1f}% "
                f"below threshold {self.thresholds['overall']:.1f}%"
            )
        
        # Check file coverage
        for file_cov in coverage_report.file_coverages:
            if self._is_excluded(file_cov.file_path):
                continue
            
            threshold = self.thresholds['file']
            if self._is_critical(file_cov.file_path):
                threshold = max(threshold, 90.0)  # Higher threshold for critical files
            
            if file_cov.coverage_percentage < threshold:
                violations.append(
                    f"File {file_cov.file_path} coverage {file_cov.coverage_percentage:.1f}% "
                    f"below threshold {threshold:.1f}%"
                )
        
        return violations
    
    def _is_excluded(self, file_path: str) -> bool:
        """Check if file is excluded from coverage requirements"""
        for pattern in self.excluded_files:
            if pattern in file_path:
                return True
        return False
    
    def _is_critical(self, file_path: str) -> bool:
        """Check if file is marked as critical"""
        for pattern in self.critical_files:
            if pattern in file_path:
                return True
        return False


class CoverageDataCollector:
    """Collects coverage data from various sources"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.coverage_formats = ['json', 'xml', 'lcov']
    
    def collect_coverage_data(self, test_files: List[Path]) -> Dict[str, Any]:
        """Collect coverage data by running tests with coverage"""
        coverage_dir = self.project_root / '.coverage_analysis'
        coverage_dir.mkdir(exist_ok=True)
        
        try:
            # Run tests with coverage
            coverage_data = self._run_coverage_tests(test_files, coverage_dir)
            
            # Parse coverage data
            parsed_data = self._parse_coverage_data(coverage_dir)
            
            return parsed_data
            
        except Exception as e:
            print(f"Error collecting coverage data: {e}")
            return {}
    
    def _run_coverage_tests(self, test_files: List[Path], coverage_dir: Path) -> bool:
        """Run tests with coverage collection"""
        # Install coverage if not available
        try:
            import coverage
        except ImportError:
            print("Installing coverage package...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'coverage[toml]'], check=True)
        
        # Build coverage command
        test_paths = [str(f) for f in test_files]
        
        cmd = [
            sys.executable, '-m', 'coverage', 'run',
            '--source=.',
            '--omit=*/tests/*,*/test_*,*/.venv/*,*/venv/*,*/__pycache__/*',
            '-m', 'pytest'
        ] + test_paths + [
            '--tb=no',
            '-q'
        ]
        
        # Run coverage
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Coverage run failed: {result.stderr}")
            return False
        
        # Generate reports
        self._generate_coverage_reports(coverage_dir)
        
        return True
    
    def _generate_coverage_reports(self, coverage_dir: Path):
        """Generate coverage reports in multiple formats"""
        formats = {
            'json': ['json', '-o', str(coverage_dir / 'coverage.json')],
            'xml': ['xml', '-o', str(coverage_dir / 'coverage.xml')],
            'html': ['html', '-d', str(coverage_dir / 'html')],
            'report': ['report', '--show-missing']
        }
        
        for format_name, args in formats.items():
            try:
                cmd = [sys.executable, '-m', 'coverage'] + args
                subprocess.run(cmd, cwd=self.project_root, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed to generate {format_name} report: {e}")
    
    def _parse_coverage_data(self, coverage_dir: Path) -> Dict[str, Any]:
        """Parse coverage data from generated reports"""
        coverage_data = {}
        
        # Parse JSON report
        json_file = coverage_dir / 'coverage.json'
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    coverage_data['json'] = json.load(f)
            except Exception as e:
                print(f"Error parsing JSON coverage: {e}")
        
        # Parse XML report
        xml_file = coverage_dir / 'coverage.xml'
        if xml_file.exists():
            try:
                coverage_data['xml'] = self._parse_xml_coverage(xml_file)
            except Exception as e:
                print(f"Error parsing XML coverage: {e}")
        
        return coverage_data


    def _parse_xml_coverage(self, xml_file: Path) -> Dict[str, Any]:
        """Parse XML coverage report"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        coverage_data = {
            'files': {},
            'summary': {}
        }
        
        # Parse summary
        for counter in root.findall('.//counter'):
            counter_type = counter.get('type', '').lower()
            covered = int(counter.get('covered', 0))
            missed = int(counter.get('missed', 0))
            
            coverage_data['summary'][counter_type] = {
                'covered': covered,
                'missed': missed,
                'total': covered + missed,
                'percentage': (covered / (covered + missed) * 100) if (covered + missed) > 0 else 0
            }
        
        # Parse file details
        for package in root.findall('.//package'):
            for class_elem in package.findall('.//class'):
                filename = class_elem.get('filename', '')
                if filename:
                    coverage_data['files'][filename] = self._parse_class_coverage(class_elem)
        
        return coverage_data
    
    def _parse_class_coverage(self, class_elem) -> Dict[str, Any]:
        """Parse coverage for a single class/file"""
        file_data = {
            'lines': {},
            'methods': {},
            'summary': {}
        }
        
        # Parse line coverage
        for line in class_elem.findall('.//line'):
            line_num = int(line.get('nr', 0))
            hits = int(line.get('ci', 0))
            file_data['lines'][line_num] = hits
        
        # Parse method coverage
        for method in class_elem.findall('.//method'):
            method_name = method.get('name', '')
            line_rate = float(method.get('line-rate', 0))
            file_data['methods'][method_name] = {
                'line_rate': line_rate,
                'covered': line_rate > 0
            }
        
        return file_data


class CoverageGapAnalyzer:
    """Analyzes coverage gaps and generates recommendations"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def analyze_gaps(self, coverage_data: Dict[str, Any]) -> List[CoverageGap]:
        """Analyze coverage data to identify gaps"""
        gaps = []
        
        if 'json' in coverage_data:
            gaps.extend(self._analyze_json_gaps(coverage_data['json']))
        
        return gaps
    
    def _analyze_json_gaps(self, json_data: Dict[str, Any]) -> List[CoverageGap]:
        """Analyze gaps from JSON coverage data"""
        gaps = []
        
        files_data = json_data.get('files', {})
        
        for file_path, file_data in files_data.items():
            # Skip test files
            if 'test' in file_path.lower():
                continue
            
            missing_lines = file_data.get('missing_lines', [])
            executed_lines = file_data.get('executed_lines', [])
            
            # Analyze missing functions
            gaps.extend(self._analyze_missing_functions(file_path, missing_lines))
            
            # Analyze missing branches
            if 'missing_branches' in file_data:
                gaps.extend(self._analyze_missing_branches(file_path, file_data['missing_branches']))
            
            # Analyze large uncovered blocks
            gaps.extend(self._analyze_uncovered_blocks(file_path, missing_lines))
        
        return gaps
    
    def _analyze_missing_functions(self, file_path: str, missing_lines: List[int]) -> List[CoverageGap]:
        """Analyze missing function coverage"""
        gaps = []
        
        try:
            # Parse file to find functions
            full_path = self.project_root / file_path
            if not full_path.exists():
                return gaps
            
            with open(full_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_start = node.lineno
                    func_end = node.end_lineno or func_start
                    
                    # Check if function lines are in missing lines
                    func_lines = set(range(func_start, func_end + 1))
                    missing_set = set(missing_lines)
                    
                    if func_lines.intersection(missing_set):
                        severity = self._determine_function_severity(node)
                        
                        gaps.append(CoverageGap(
                            file_path=file_path,
                            gap_type='function',
                            name=node.name,
                            line_start=func_start,
                            line_end=func_end,
                            severity=severity,
                            suggestion=f"Add tests for function '{node.name}'"
                        ))
        
        except Exception as e:
            print(f"Error analyzing functions in {file_path}: {e}")
        
        return gaps
    
    def _analyze_missing_branches(self, file_path: str, missing_branches: List) -> List[CoverageGap]:
        """Analyze missing branch coverage"""
        gaps = []
        
        for branch in missing_branches:
            if isinstance(branch, list) and len(branch) >= 2:
                line_num = branch[0]
                branch_id = branch[1]
                
                gaps.append(CoverageGap(
                    file_path=file_path,
                    gap_type='branch',
                    name=f"Branch {branch_id}",
                    line_start=line_num,
                    line_end=line_num,
                    severity='medium',
                    suggestion=f"Add test case to cover branch at line {line_num}"
                ))
        
        return gaps
    
    def _analyze_uncovered_blocks(self, file_path: str, missing_lines: List[int]) -> List[CoverageGap]:
        """Analyze large blocks of uncovered code"""
        gaps = []
        
        if not missing_lines:
            return gaps
        
        # Find consecutive missing lines (blocks)
        missing_lines.sort()
        blocks = []
        current_block = [missing_lines[0]]
        
        for line in missing_lines[1:]:
            if line == current_block[-1] + 1:
                current_block.append(line)
            else:
                blocks.append(current_block)
                current_block = [line]
        
        blocks.append(current_block)
        
        # Identify significant blocks
        for block in blocks:
            if len(block) >= 5:  # Block of 5+ consecutive lines
                severity = 'high' if len(block) >= 10 else 'medium'
                
                gaps.append(CoverageGap(
                    file_path=file_path,
                    gap_type='lines',
                    name=f"Uncovered block ({len(block)} lines)",
                    line_start=block[0],
                    line_end=block[-1],
                    severity=severity,
                    suggestion=f"Add tests to cover {len(block)} consecutive uncovered lines"
                ))
        
        return gaps
    
    def _determine_function_severity(self, func_node: ast.FunctionDef) -> str:
        """Determine severity of missing function coverage"""
        # Public functions are more critical
        if not func_node.name.startswith('_'):
            return 'high'
        
        # Functions with complex logic are important
        complexity = self._calculate_complexity(func_node)
        if complexity > 5:
            return 'high'
        elif complexity > 2:
            return 'medium'
        
        return 'low'
    
    def _calculate_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of function"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity


class CoverageAnalyzer:
    """Main coverage analyzer that orchestrates all analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.threshold_manager = CoverageThresholdManager()
        self.data_collector = CoverageDataCollector(project_root)
        self.gap_analyzer = CoverageGapAnalyzer(project_root)
    
    def analyze_coverage(self, test_files: List[Path]) -> CoverageReport:
        """Perform comprehensive coverage analysis"""
        print("Collecting coverage data...")
        
        # Collect coverage data
        coverage_data = self.data_collector.collect_coverage_data(test_files)
        
        if not coverage_data:
            return self._create_empty_report()
        
        print("Analyzing coverage gaps...")
        
        # Analyze coverage
        file_coverages = self._extract_file_coverages(coverage_data)
        coverage_gaps = self.gap_analyzer.analyze_gaps(coverage_data)
        
        # Calculate overall statistics
        total_files = len(file_coverages)
        covered_files = len([fc for fc in file_coverages if fc.coverage_percentage > 0])
        total_lines = sum(fc.total_lines for fc in file_coverages)
        covered_lines = sum(fc.covered_lines for fc in file_coverages)
        overall_percentage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(file_coverages, coverage_gaps)
        
        # Check threshold violations
        report = CoverageReport(
            total_files=total_files,
            covered_files=covered_files,
            total_lines=total_lines,
            covered_lines=covered_lines,
            overall_percentage=overall_percentage,
            file_coverages=file_coverages,
            coverage_gaps=coverage_gaps,
            recommendations=recommendations,
            threshold_violations=[]
        )
        
        report.threshold_violations = self.threshold_manager.check_violations(report)
        
        return report
    
    def _extract_file_coverages(self, coverage_data: Dict[str, Any]) -> List[FileCoverage]:
        """Extract file coverage information"""
        file_coverages = []
        
        if 'json' in coverage_data:
            json_data = coverage_data['json']
            files_data = json_data.get('files', {})
            
            for file_path, file_data in files_data.items():
                # Skip test files
                if 'test' in file_path.lower():
                    continue
                
                summary = file_data.get('summary', {})
                total_lines = summary.get('num_statements', 0)
                covered_lines = summary.get('covered_lines', 0)
                missing_lines = file_data.get('missing_lines', [])
                
                coverage_percentage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
                
                # Extract function coverage
                functions = self._extract_function_coverage(file_path, file_data)
                
                file_coverages.append(FileCoverage(
                    file_path=file_path,
                    total_lines=total_lines,
                    covered_lines=covered_lines,
                    missing_lines=missing_lines,
                    coverage_percentage=coverage_percentage,
                    functions=functions,
                    classes={}  # TODO: Extract class coverage
                ))
        
        return file_coverages
    
    def _extract_function_coverage(self, file_path: str, file_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract function-level coverage information"""
        functions = {}
        
        try:
            # Parse file to get function definitions
            full_path = self.project_root / file_path
            if not full_path.exists():
                return functions
            
            with open(full_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            executed_lines = set(file_data.get('executed_lines', []))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_start = node.lineno
                    func_end = node.end_lineno or func_start
                    
                    func_lines = set(range(func_start, func_end + 1))
                    covered_func_lines = func_lines.intersection(executed_lines)
                    
                    coverage_percentage = (len(covered_func_lines) / len(func_lines) * 100) if func_lines else 0
                    
                    functions[node.name] = {
                        'line_start': func_start,
                        'line_end': func_end,
                        'total_lines': len(func_lines),
                        'covered_lines': len(covered_func_lines),
                        'coverage_percentage': coverage_percentage,
                        'is_covered': coverage_percentage > 0
                    }
        
        except Exception as e:
            print(f"Error extracting function coverage for {file_path}: {e}")
        
        return functions
    
    def _generate_recommendations(self, file_coverages: List[FileCoverage], gaps: List[CoverageGap]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Overall coverage recommendations
        low_coverage_files = [fc for fc in file_coverages if fc.coverage_percentage < 50]
        if low_coverage_files:
            recommendations.append(
                f"Prioritize testing for {len(low_coverage_files)} files with less than 50% coverage"
            )
        
        # Function coverage recommendations
        uncovered_functions = []
        for fc in file_coverages:
            for func_name, func_data in fc.functions.items():
                if func_data['coverage_percentage'] == 0:
                    uncovered_functions.append((fc.file_path, func_name))
        
        if uncovered_functions:
            recommendations.append(
                f"Add tests for {len(uncovered_functions)} completely untested functions"
            )
        
        # Gap-specific recommendations
        critical_gaps = [g for g in gaps if g.severity == 'critical']
        if critical_gaps:
            recommendations.append(
                f"Address {len(critical_gaps)} critical coverage gaps immediately"
            )
        
        high_gaps = [g for g in gaps if g.severity == 'high']
        if high_gaps:
            recommendations.append(
                f"Address {len(high_gaps)} high-priority coverage gaps"
            )
        
        # Branch coverage recommendations
        branch_gaps = [g for g in gaps if g.gap_type == 'branch']
        if branch_gaps:
            recommendations.append(
                f"Improve branch coverage by testing {len(branch_gaps)} uncovered code paths"
            )
        
        return recommendations
    
    def _create_empty_report(self) -> CoverageReport:
        """Create empty report when coverage data collection fails"""
        return CoverageReport(
            total_files=0,
            covered_files=0,
            total_lines=0,
            covered_lines=0,
            overall_percentage=0.0,
            file_coverages=[],
            coverage_gaps=[],
            recommendations=["Failed to collect coverage data - ensure tests can run successfully"],
            threshold_violations=[]
        )


def main():
    """Main entry point for coverage analyzer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive test coverage analyzer")
    parser.add_argument('--project-root', type=Path, default=Path.cwd(), help='Project root directory')
    parser.add_argument('--test-files', nargs='*', help='Specific test files to analyze')
    parser.add_argument('--output', type=Path, help='Output file for coverage report')
    parser.add_argument('--threshold', type=float, help='Overall coverage threshold')
    
    args = parser.parse_args()
    
    # Setup analyzer
    analyzer = CoverageAnalyzer(args.project_root)
    
    if args.threshold:
        analyzer.threshold_manager.set_threshold('overall', args.threshold)
    
    # Discover test files if not specified
    if args.test_files:
        test_files = [Path(f) for f in args.test_files]
    else:
        from test_auditor import TestDiscoveryEngine
        discovery = TestDiscoveryEngine(args.project_root)
        test_files = discovery.discover_test_files()
    
    # Run analysis
    report = analyzer.analyze_coverage(test_files)
    
    # Save report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        print(f"Coverage report saved to {args.output}")
    
    # Print summary
    print(f"\nCoverage Analysis Summary:")
    print(f"Overall coverage: {report.overall_percentage:.1f}%")
    print(f"Files covered: {report.covered_files}/{report.total_files}")
    print(f"Lines covered: {report.covered_lines}/{report.total_lines}")
    print(f"Coverage gaps: {len(report.coverage_gaps)}")
    print(f"Threshold violations: {len(report.threshold_violations)}")
    
    if report.recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())