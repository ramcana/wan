---
title: core.model_orchestrator.tests.test_runner
category: api
tags: [api, core]
---

# core.model_orchestrator.tests.test_runner

Comprehensive test runner for Model Orchestrator.

This script runs all test suites and generates comprehensive reports
for end-to-end validation of the Model Orchestrator system.

## Classes

### TestRunner

Comprehensive test runner for Model Orchestrator.

#### Methods

##### __init__(self: Any, verbose: bool, coverage: bool)



##### run_test_suite(self: Any, suite_name: str, test_path: str, markers: <ast.Subscript object at 0x000001942F7A3B20>) -> Dict

Run a specific test suite and return results.

##### run_all_tests(self: Any) -> Dict

Run all test suites.

##### generate_report(self: Any, output_file: <ast.Subscript object at 0x0000019431984670>) -> str

Generate comprehensive test report.

##### run_coverage_report(self: Any)

Generate coverage report.

