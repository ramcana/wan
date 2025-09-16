---
title: tools.code_quality.quality_checker
category: api
tags: [api, tools]
---

# tools.code_quality.quality_checker

Main code quality checking engine.

## Classes

### QualityChecker

Main code quality checking engine.

#### Methods

##### __init__(self: Any, config: <ast.Subscript object at 0x000001942A2E6DD0>)

Initialize quality checker with configuration.

##### from_config_file(cls: Any, config_path: Path) -> QualityChecker

Create quality checker from configuration file.

##### _parse_config(config_data: <ast.Subscript object at 0x000001942A2E52A0>) -> QualityConfig

Parse configuration data into QualityConfig object.

##### check_quality(self: Any, path: Path, checks: <ast.Subscript object at 0x000001942CB35060>) -> QualityReport

Perform comprehensive quality check on the given path.

Args:
    path: Path to check (file or directory)
    checks: List of specific checks to run (None for all)

Returns:
    QualityReport with all issues and metrics

##### fix_issues(self: Any, path: Path, auto_fix_only: bool) -> QualityReport

Automatically fix quality issues where possible.

Args:
    path: Path to fix (file or directory)
    auto_fix_only: Only fix issues marked as auto-fixable

Returns:
    QualityReport showing what was fixed

##### _get_python_files(self: Any, path: Path) -> <ast.Subscript object at 0x00000194288F4160>

Get all Python files in the given path.

##### _analyze_file(self: Any, file_path: Path, checks: <ast.Subscript object at 0x00000194288F5ED0>) -> <ast.Subscript object at 0x00000194288F4D90>

Analyze a single file for quality issues.

##### _aggregate_metrics(self: Any, total_metrics: QualityMetrics, file_metrics: QualityMetrics) -> None

Aggregate file metrics into total metrics.

##### _fix_file_issues(self: Any, file_path: Path, issues: <ast.Subscript object at 0x000001942A1BD360>) -> int

Fix issues in a specific file.

##### generate_report(self: Any, report: QualityReport, output_format: str) -> str

Generate formatted report.

##### _generate_text_report(self: Any, report: QualityReport) -> str

Generate human-readable text report.

