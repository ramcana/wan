---
title: tools.test-auditor.coverage_analyzer
category: api
tags: [api, tools]
---

# tools.test-auditor.coverage_analyzer



## Classes

### FileCoverage

Coverage information for a single file

### CoverageGap

Represents a gap in test coverage

### CoverageReport

Complete coverage analysis report

### CoverageThresholdManager

Manages coverage thresholds and violations

#### Methods

##### __init__(self: Any)



##### set_threshold(self: Any, threshold_type: str, value: float)

Set coverage threshold

##### add_critical_file(self: Any, file_pattern: str)

Add file pattern that requires high coverage

##### exclude_file(self: Any, file_pattern: str)

Exclude file pattern from coverage requirements

##### check_violations(self: Any, coverage_report: CoverageReport) -> <ast.Subscript object at 0x00000194345E0790>

Check for threshold violations

##### _is_excluded(self: Any, file_path: str) -> bool

Check if file is excluded from coverage requirements

##### _is_critical(self: Any, file_path: str) -> bool

Check if file is marked as critical

### CoverageDataCollector

Collects coverage data from various sources

#### Methods

##### __init__(self: Any, project_root: Path)



##### collect_coverage_data(self: Any, test_files: <ast.Subscript object at 0x00000194345E0F40>) -> <ast.Subscript object at 0x00000194345E2B00>

Collect coverage data by running tests with coverage

##### _run_coverage_tests(self: Any, test_files: <ast.Subscript object at 0x00000194345E3790>, coverage_dir: Path) -> bool

Run tests with coverage collection

##### _generate_coverage_reports(self: Any, coverage_dir: Path)

Generate coverage reports in multiple formats

##### _parse_coverage_data(self: Any, coverage_dir: Path) -> <ast.Subscript object at 0x0000019433CBE890>

Parse coverage data from generated reports

##### _parse_xml_coverage(self: Any, xml_file: Path) -> <ast.Subscript object at 0x000001943026AC20>

Parse XML coverage report

##### _parse_class_coverage(self: Any, class_elem: Any) -> <ast.Subscript object at 0x000001943026BE20>

Parse coverage for a single class/file

### CoverageGapAnalyzer

Analyzes coverage gaps and generates recommendations

#### Methods

##### __init__(self: Any, project_root: Path)



##### analyze_gaps(self: Any, coverage_data: <ast.Subscript object at 0x0000019430268F10>) -> <ast.Subscript object at 0x0000019430269B40>

Analyze coverage data to identify gaps

##### _analyze_json_gaps(self: Any, json_data: <ast.Subscript object at 0x00000194302697B0>) -> <ast.Subscript object at 0x000001942EF26980>

Analyze gaps from JSON coverage data

##### _analyze_missing_functions(self: Any, file_path: str, missing_lines: <ast.Subscript object at 0x000001942EF25900>) -> <ast.Subscript object at 0x000001942EF26EC0>

Analyze missing function coverage

##### _analyze_missing_branches(self: Any, file_path: str, missing_branches: List) -> <ast.Subscript object at 0x000001942F2221D0>

Analyze missing branch coverage

##### _analyze_uncovered_blocks(self: Any, file_path: str, missing_lines: <ast.Subscript object at 0x000001942F222FB0>) -> <ast.Subscript object at 0x000001942F222110>

Analyze large blocks of uncovered code

##### _determine_function_severity(self: Any, func_node: ast.FunctionDef) -> str

Determine severity of missing function coverage

##### _calculate_complexity(self: Any, func_node: ast.FunctionDef) -> int

Calculate cyclomatic complexity of function

### CoverageAnalyzer

Main coverage analyzer that orchestrates all analysis

#### Methods

##### __init__(self: Any, project_root: Path)



##### analyze_coverage(self: Any, test_files: <ast.Subscript object at 0x000001942F223430>) -> CoverageReport

Perform comprehensive coverage analysis

##### _extract_file_coverages(self: Any, coverage_data: <ast.Subscript object at 0x000001943193A860>) -> <ast.Subscript object at 0x0000019431939900>

Extract file coverage information

##### _extract_function_coverage(self: Any, file_path: str, file_data: <ast.Subscript object at 0x000001943193BD60>) -> <ast.Subscript object at 0x0000019432E5E470>

Extract function-level coverage information

##### _generate_recommendations(self: Any, file_coverages: <ast.Subscript object at 0x0000019432E5D780>, gaps: <ast.Subscript object at 0x0000019432E5D450>) -> <ast.Subscript object at 0x00000194344A1540>

Generate actionable recommendations

##### _create_empty_report(self: Any) -> CoverageReport

Create empty report when coverage data collection fails

