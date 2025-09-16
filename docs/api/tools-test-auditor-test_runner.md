---
title: tools.test-auditor.test_runner
category: api
tags: [api, tools]
---

# tools.test-auditor.test_runner



## Classes

### TestExecutionResult

Result of executing a single test file

### TestSuiteExecutionReport

Complete execution report for test suite

### TestIsolationManager

Manages test isolation and cleanup

#### Methods

##### __init__(self: Any)



##### create_isolated_environment(self: Any) -> Path

Create isolated temporary directory for test execution

##### cleanup_environment(self: Any, temp_dir: Path)

Clean up isolated environment

##### cleanup_all(self: Any)

Clean up all created environments

### TestTimeoutManager

Manages test timeouts with configurable limits

#### Methods

##### __init__(self: Any)



##### get_timeout_for_file(self: Any, test_file: Path) -> int

Get appropriate timeout for test file based on its type

##### set_timeout_override(self: Any, pattern: str, timeout: int)

Set custom timeout for files matching pattern

### TestRetryManager

Manages test retry logic for flaky tests

#### Methods

##### __init__(self: Any)



##### should_retry(self: Any, test_file: Path, attempt: int, result: TestExecutionResult) -> bool

Determine if test should be retried

##### get_retry_delay(self: Any, attempt: int) -> float

Get delay before retry

### TestCoverageCollector

Collects test coverage information

#### Methods

##### __init__(self: Any)



##### setup_coverage(self: Any, test_file: Path, temp_dir: Path) -> <ast.Subscript object at 0x000001942F3913F0>

Setup coverage collection for test execution

##### collect_coverage_data(self: Any, temp_dir: Path) -> <ast.Subscript object at 0x000001942F393CA0>

Collect coverage data from test execution

### TestExecutor

Executes individual test files with full isolation and monitoring

#### Methods

##### __init__(self: Any)



##### execute_test_file(self: Any, test_file: Path, project_root: Path) -> TestExecutionResult

Execute a single test file with full monitoring

##### _execute_single_attempt(self: Any, test_file: Path, project_root: Path, attempt: int) -> TestExecutionResult

Execute single test attempt

##### _parse_test_counts(self: Any, stdout: str, temp_dir: Path) -> <ast.Subscript object at 0x000001942F6C4EE0>

Parse test counts from pytest output

### ParallelTestRunner

Runs tests in parallel with resource management

#### Methods

##### __init__(self: Any, max_workers: <ast.Subscript object at 0x000001942F6C5A80>)



##### run_tests_parallel(self: Any, test_files: <ast.Subscript object at 0x000001942F6C6A70>, project_root: Path) -> TestSuiteExecutionReport

Run tests in parallel

##### _run_single_test(self: Any, test_file: Path, project_root: Path) -> TestExecutionResult

Run single test (for process pool)

##### _generate_performance_summary(self: Any, results: <ast.Subscript object at 0x000001942FB481C0>) -> <ast.Subscript object at 0x000001942FB4BDF0>

Generate performance summary

##### _generate_retry_summary(self: Any, results: <ast.Subscript object at 0x000001942FB4BCA0>) -> <ast.Subscript object at 0x000001942FB48880>

Generate retry summary

##### _generate_timeout_summary(self: Any, results: <ast.Subscript object at 0x000001942FB48700>) -> <ast.Subscript object at 0x000001942FB4B8B0>

Generate timeout summary

