---
title: tools.test-runner.orchestrator
category: api
tags: [api, tools]
---

# tools.test-runner.orchestrator



## Classes

### TestCategory

Test categories for organization and execution

### TestStatus

Test execution status

### TestDetail

Individual test result details

### CategoryResults

Results for a specific test category

#### Methods

##### success_rate(self: Any) -> float

Calculate success rate as percentage

### TestSummary

Overall test suite summary

### TestResults

Complete test execution results

### TestConfig

Test configuration loaded from YAML

#### Methods

##### load_from_file(cls: Any, config_path: Path) -> TestConfig

Load test configuration from YAML file

### ResourceManager

Manages system resources during test execution

#### Methods

##### __init__(self: Any, max_workers: int)



### TestSuiteOrchestrator

Main orchestrator for test suite execution with category management and parallel execution

#### Methods

##### __init__(self: Any, config_path: <ast.Subscript object at 0x000001942B749F90>)

Initialize the test orchestrator

##### _calculate_overall_summary(self: Any, category_results: <ast.Subscript object at 0x000001942902A620>, total_duration: float, categories_run: <ast.Subscript object at 0x000001942902A7A0>) -> TestSummary

Calculate overall test suite summary from category results

##### _get_config_summary(self: Any) -> <ast.Subscript object at 0x00000194290100A0>

Get summary of configuration used for this run

##### _get_environment_info(self: Any) -> <ast.Subscript object at 0x000001942B784730>

Get environment information for the test run

##### get_results(self: Any, suite_id: str) -> <ast.Subscript object at 0x000001942B7843D0>

Retrieve cached test results by suite ID

##### export_results(self: Any, results: TestResults, output_path: Path, format_type: str)

Export test results to file

## Constants

### UNIT

Type: `str`

Value: `unit`

### INTEGRATION

Type: `str`

Value: `integration`

### PERFORMANCE

Type: `str`

Value: `performance`

### E2E

Type: `str`

Value: `e2e`

### PENDING

Type: `str`

Value: `pending`

### RUNNING

Type: `str`

Value: `running`

### PASSED

Type: `str`

Value: `passed`

### FAILED

Type: `str`

Value: `failed`

### SKIPPED

Type: `str`

Value: `skipped`

### TIMEOUT

Type: `str`

Value: `timeout`

### ERROR

Type: `str`

Value: `error`

