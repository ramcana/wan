---
title: tools.test-runner.coverage_analyzer
category: api
tags: [api, tools]
---

# tools.test-runner.coverage_analyzer



## Classes

### FileCoverage

Coverage information for a single file

#### Methods

##### coverage_percentage(self: Any) -> float

Calculate coverage percentage for this file

##### is_fully_covered(self: Any) -> bool

Check if file has 100% coverage

### ModuleCoverage

Coverage information for a module (directory)

#### Methods

##### total_lines(self: Any) -> int



##### covered_lines(self: Any) -> int



##### coverage_percentage(self: Any) -> float



### CoverageReport

Complete coverage report

#### Methods

##### uncovered_lines(self: Any) -> int



##### files_with_low_coverage(self: Any) -> <ast.Subscript object at 0x0000019429095FC0>

Get files below the coverage threshold

##### fully_covered_files(self: Any) -> <ast.Subscript object at 0x0000019429095C00>

Get files with 100% coverage

### CoverageTrend

Coverage trend analysis over time

#### Methods

##### is_improving(self: Any) -> bool



##### is_declining(self: Any) -> bool



### CoverageHistory

Historical coverage data

#### Methods

##### add_report(self: Any, timestamp: datetime, coverage: float)

Add a coverage report to history

##### get_trend(self: Any) -> <ast.Subscript object at 0x0000019429013400>

Calculate coverage trend

### CoverageThresholdValidator

Validates coverage against thresholds and policies

#### Methods

##### __init__(self: Any, config: TestConfig)



##### validate_coverage(self: Any, report: CoverageReport) -> <ast.Subscript object at 0x00000194290109D0>

Validate coverage report against all thresholds

Returns:
    Dictionary with validation results

##### _generate_recommendations(self: Any, report: CoverageReport) -> <ast.Subscript object at 0x0000019429029030>

Generate actionable recommendations for improving coverage

### CoverageAnalyzer

Main coverage analyzer with measurement, reporting, and trend analysis

#### Methods

##### __init__(self: Any, config: TestConfig)



##### measure_coverage(self: Any, test_categories: <ast.Subscript object at 0x0000019429028520>, source_dirs: <ast.Subscript object at 0x0000019429028460>) -> CoverageReport

Measure code coverage for specified test categories

Args:
    test_categories: Categories of tests to run for coverage
    source_dirs: Source directories to measure coverage for
    
Returns:
    CoverageReport with detailed coverage information

##### _get_default_source_dirs(self: Any) -> <ast.Subscript object at 0x000001942898EBF0>

Get default source directories to measure coverage for

##### _run_tests_with_coverage(self: Any, test_categories: <ast.Subscript object at 0x000001942898EAA0>, source_dirs: <ast.Subscript object at 0x000001942898E9E0>) -> <ast.Subscript object at 0x000001942898C040>

Run tests with coverage measurement

##### _parse_coverage_data(self: Any, coverage_data: <ast.Subscript object at 0x00000194289ABDC0>, test_categories: <ast.Subscript object at 0x00000194289ABD30>) -> CoverageReport

Parse coverage data from XML and JSON reports

##### _parse_json_coverage(self: Any, json_file: Path, test_categories: <ast.Subscript object at 0x0000019428949CC0>) -> CoverageReport

Parse JSON coverage report

##### _parse_xml_coverage(self: Any, xml_file: Path, test_categories: <ast.Subscript object at 0x00000194289AF9A0>) -> CoverageReport

Parse XML coverage report (fallback)

##### validate_coverage_thresholds(self: Any, report: CoverageReport) -> <ast.Subscript object at 0x000001942832AE30>

Validate coverage report against configured thresholds

##### get_coverage_trend(self: Any) -> <ast.Subscript object at 0x000001942832AB60>

Get coverage trend analysis

##### generate_coverage_report(self: Any, report: CoverageReport, output_path: Path, format_type: str) -> Path

Generate formatted coverage report

Args:
    report: Coverage report to format
    output_path: Output file path
    format_type: Report format ('html', 'json', 'markdown')
    
Returns:
    Path to generated report

##### _generate_json_report(self: Any, report: CoverageReport, output_path: Path)

Generate JSON coverage report

##### _generate_markdown_report(self: Any, report: CoverageReport, output_path: Path)

Generate Markdown coverage report

##### _generate_html_report(self: Any, report: CoverageReport, output_path: Path)

Generate HTML coverage report

##### _load_coverage_history(self: Any)

Load coverage history from file

##### _save_coverage_history(self: Any)

Save coverage history to file

