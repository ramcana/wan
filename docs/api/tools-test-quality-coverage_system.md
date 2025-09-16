---
title: tools.test-quality.coverage_system
category: api
tags: [api, tools]
---

# tools.test-quality.coverage_system

Comprehensive Test Coverage Analysis System

Enhanced coverage reporting system that identifies untested code paths,
enforces coverage thresholds for new code, generates detailed reports
with actionable recommendations, and tracks coverage trends over time.

## Classes

### CoverageTrend

Coverage trend data point

### NewCodeCoverage

Coverage analysis for new/changed code

### CoverageThresholdResult

Result of coverage threshold enforcement

### CoverageTrendTracker

Tracks coverage trends over time

#### Methods

##### __init__(self: Any, project_root: Path)



##### _init_database(self: Any)

Initialize the trends database

##### record_coverage(self: Any, report: CoverageReport) -> int

Record coverage data point

##### get_trends(self: Any, days: int) -> <ast.Subscript object at 0x000001942CBAEF80>

Get coverage trends for the last N days

##### get_file_trends(self: Any, file_path: str, days: int) -> <ast.Subscript object at 0x000001942CBAE290>

Get coverage trends for a specific file

##### _get_current_commit(self: Any) -> <ast.Subscript object at 0x000001942CBAFBE0>

Get current git commit hash

##### _get_current_branch(self: Any) -> <ast.Subscript object at 0x000001942CBAF880>

Get current git branch

### NewCodeCoverageAnalyzer

Analyzes coverage for new/changed code

#### Methods

##### __init__(self: Any, project_root: Path)



##### set_new_code_threshold(self: Any, threshold: float)

Set coverage threshold for new code

##### analyze_new_code_coverage(self: Any, report: CoverageReport, base_branch: str) -> <ast.Subscript object at 0x000001942CAF8AC0>

Analyze coverage for new/changed code compared to base branch

##### _get_changed_files(self: Any, base_branch: str) -> <ast.Subscript object at 0x0000019429CBA080>

Get list of files changed compared to base branch

##### _get_new_lines(self: Any, file_path: str, base_branch: str) -> <ast.Subscript object at 0x00000194287F97E0>

Get line numbers of new/changed lines in a file

### CoverageThresholdEnforcer

Enforces coverage thresholds for new code

#### Methods

##### __init__(self: Any, project_root: Path)



##### set_thresholds(self: Any, overall: float, new_code: float, critical: float)

Set coverage thresholds

##### enforce_thresholds(self: Any, report: CoverageReport, base_branch: str) -> CoverageThresholdResult

Enforce coverage thresholds

##### _is_critical_file(self: Any, file_path: str) -> bool

Check if file is considered critical

### DetailedCoverageReporter

Generates detailed coverage reports with actionable recommendations

#### Methods

##### __init__(self: Any, project_root: Path)



##### generate_detailed_report(self: Any, report: CoverageReport, threshold_result: CoverageThresholdResult) -> <ast.Subscript object at 0x00000194288C5D50>

Generate comprehensive coverage report

##### _generate_summary(self: Any, report: CoverageReport) -> <ast.Subscript object at 0x00000194288C6E90>

Generate coverage summary

##### _generate_file_analysis(self: Any, report: CoverageReport) -> <ast.Subscript object at 0x00000194288A7E50>

Generate per-file analysis

##### _generate_gap_analysis(self: Any, report: CoverageReport) -> <ast.Subscript object at 0x000001942C5A8CD0>

Generate coverage gap analysis

##### _generate_detailed_recommendations(self: Any, report: CoverageReport, threshold_result: CoverageThresholdResult) -> <ast.Subscript object at 0x000001942C5AAB60>

Generate detailed, prioritized recommendations

##### _generate_trend_analysis(self: Any) -> <ast.Subscript object at 0x000001942C555AE0>

Generate coverage trend analysis

##### _generate_actionable_items(self: Any, report: CoverageReport, threshold_result: CoverageThresholdResult) -> <ast.Subscript object at 0x000001942C5579D0>

Generate specific actionable items

##### _group_missing_lines(self: Any, missing_lines: <ast.Subscript object at 0x000001942C557AF0>) -> <ast.Subscript object at 0x000001942C554340>

Group consecutive missing lines into ranges

##### _analyze_function_coverage(self: Any, file_cov: FileCoverage) -> <ast.Subscript object at 0x0000019428934670>

Analyze function-level coverage

##### _determine_file_priority(self: Any, file_cov: FileCoverage) -> str

Determine priority level for file coverage improvement

##### _generate_file_recommendations(self: Any, file_cov: FileCoverage) -> <ast.Subscript object at 0x0000019428937460>

Generate specific recommendations for a file

### ComprehensiveCoverageSystem

Main system that orchestrates all coverage analysis components

#### Methods

##### __init__(self: Any, project_root: Path)



##### run_comprehensive_analysis(self: Any, test_files: <ast.Subscript object at 0x00000194289366E0>, base_branch: str) -> <ast.Subscript object at 0x00000194288F2110>

Run complete coverage analysis with all features

##### save_report(self: Any, analysis_result: <ast.Subscript object at 0x00000194288F2440>, output_path: Path)

Save comprehensive analysis report

