---
title: tools.code_quality.enforcement.ci_integration
category: api
tags: [api, tools]
---

# tools.code_quality.enforcement.ci_integration

CI/CD integration for automated quality checking.

## Classes

### CIIntegration

Manages CI/CD integration for code quality enforcement.

#### Methods

##### __init__(self: Any, project_root: Path)

Initialize CI integration manager.

##### setup_github_actions(self: Any, config: <ast.Subscript object at 0x0000019427C7F5B0>) -> bool

Set up GitHub Actions workflow for quality enforcement.

Args:
    config: Optional configuration for the workflow

Returns:
    True if setup successful

##### setup_gitlab_ci(self: Any, config: <ast.Subscript object at 0x0000019427C7F340>) -> bool

Set up GitLab CI pipeline for quality enforcement.

Args:
    config: Optional configuration for the pipeline

Returns:
    True if setup successful

##### setup_jenkins(self: Any, config: <ast.Subscript object at 0x0000019427C7ED10>) -> bool

Set up Jenkins pipeline for quality enforcement.

Args:
    config: Optional configuration for the pipeline

Returns:
    True if setup successful

##### create_quality_metrics_dashboard(self: Any) -> <ast.Subscript object at 0x00000194275814B0>

Create quality metrics tracking dashboard configuration.

Returns:
    Dashboard configuration

##### run_quality_checks(self: Any, files: <ast.Subscript object at 0x0000019427581300>) -> <ast.Subscript object at 0x000001942750B7C0>

Run comprehensive quality checks for CI/CD.

Args:
    files: Optional list of files to check

Returns:
    Quality check results

##### generate_quality_report(self: Any, results: <ast.Subscript object at 0x000001942750B610>) -> str

Generate quality report for CI/CD output.

Args:
    results: Quality check results

Returns:
    Formatted report string

##### update_quality_metrics(self: Any, results: <ast.Subscript object at 0x000001942CCB70A0>) -> bool

Update quality metrics tracking.

Args:
    results: Quality check results

Returns:
    True if update successful

##### _get_default_github_config(self: Any) -> <ast.Subscript object at 0x000001942CCE7E80>

Get default GitHub Actions workflow configuration.

##### _get_default_gitlab_config(self: Any) -> <ast.Subscript object at 0x000001942CCE7730>

Get default GitLab CI pipeline configuration.

##### _get_default_jenkins_config(self: Any) -> str

Get default Jenkins pipeline configuration.

##### _calculate_quality_metrics(self: Any, checks: <ast.Subscript object at 0x000001942CCE7310>) -> <ast.Subscript object at 0x000001942CC09BA0>

Calculate overall quality metrics from check results.

##### _get_quality_grade(self: Any, score: float) -> str

Get quality grade based on score.

##### get_ci_status(self: Any) -> <ast.Subscript object at 0x000001942CC089D0>

Get status of CI/CD integration.

