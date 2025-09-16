---
title: tools.test-quality.integration_example
category: api
tags: [api, tools]
---

# tools.test-quality.integration_example

Test Quality Integration Example

Demonstrates how to integrate all test quality tools into a comprehensive
development workflow, including CI/CD integration and automated monitoring.

## Classes

### TestQualityIntegration

Integrated test quality management system

#### Methods

##### __init__(self: Any, project_root: Path)



##### run_pre_commit_checks(self: Any) -> <ast.Subscript object at 0x000001942FC1E0E0>

Run quality checks suitable for pre-commit hooks

##### run_ci_quality_gates(self: Any) -> <ast.Subscript object at 0x000001942FBC18D0>

Run comprehensive quality gates for CI/CD

##### run_nightly_analysis(self: Any) -> <ast.Subscript object at 0x0000019431BEB4C0>

Run comprehensive nightly analysis with detailed reporting

##### generate_quality_dashboard_data(self: Any) -> <ast.Subscript object at 0x0000019431B06FE0>

Generate data for quality dashboard

