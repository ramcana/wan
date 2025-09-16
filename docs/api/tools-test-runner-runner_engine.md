---
title: tools.test-runner.runner_engine
category: api
tags: [api, tools]
---

# tools.test-runner.runner_engine

Test Runner Engine - Core test execution with timeout handling and discovery

## Classes

### TestDiscoveryMethod

Methods for discovering tests

### TestExecutionContext

Context for test execution

### ExecutionProgress

Progress tracking for test execution

#### Methods

##### progress_percentage(self: Any) -> float



##### elapsed_time(self: Any) -> float



### ProgressMonitor

Monitors and reports test execution progress

#### Methods

##### __init__(self: Any)



##### add_progress_callback(self: Any, callback: <ast.Subscript object at 0x0000019433D60B20>)

Add a callback to be called on progress updates

##### update_progress(self: Any, progress: ExecutionProgress)

Update current progress and notify callbacks

##### start_execution(self: Any, total_tests: int)

Start tracking execution progress

##### complete_test(self: Any, test_name: str)

Mark a test as completed

### TestDiscovery

Discovers and categorizes tests based on patterns and file structure

#### Methods

##### __init__(self: Any, config: TestConfig)



##### discover_tests(self: Any, category: TestCategory) -> <ast.Subscript object at 0x0000019433C75330>

Discover test files for a specific category

Args:
    category: Test category to discover tests for
    
Returns:
    List of test file paths

##### _get_default_patterns(self: Any, category: TestCategory) -> <ast.Subscript object at 0x0000019433C74C40>

Get default file patterns for a category

##### _is_valid_test_file(self: Any, file_path: Path) -> bool

Check if a file is a valid Python test file

##### categorize_test_file(self: Any, file_path: Path) -> <ast.Subscript object at 0x0000019433D2A830>

Automatically categorize a test file based on its path and content

### TimeoutManager

Manages test execution timeouts with graceful handling

#### Methods

##### __init__(self: Any)



##### execute_with_timeout(self: Any, command: <ast.Subscript object at 0x0000019433D2A1D0>, timeout: int, cwd: <ast.Subscript object at 0x0000019433D2A0B0>, env: <ast.Subscript object at 0x0000019433D29FF0>) -> <ast.Subscript object at 0x0000019433D55330>

Execute command with timeout handling

Args:
    command: Command to execute
    timeout: Timeout in seconds
    cwd: Working directory
    env: Environment variables
    
Returns:
    Tuple of (return_code, stdout, stderr)

##### _terminate_process(self: Any, process: subprocess.Popen)

Gracefully terminate a process

### TestRunnerEngine

Core test execution engine with timeout handling and progress monitoring

#### Methods

##### __init__(self: Any, config: TestConfig)



##### _determine_test_runner(self: Any, test_file: Path) -> TestDiscoveryMethod

Determine the appropriate test runner for a file

##### _parse_pytest_output(self: Any, stdout: str, stderr: str, return_code: int, test_file: Path, context: TestExecutionContext) -> <ast.Subscript object at 0x00000194341E5D80>

Parse pytest output to extract test results

##### _parse_unittest_output(self: Any, stdout: str, stderr: str, return_code: int, test_file: Path, context: TestExecutionContext) -> <ast.Subscript object at 0x00000194341E6AD0>

Parse unittest output to extract test results

##### _parse_text_output(self: Any, stdout: str, stderr: str, return_code: int, test_file: Path, context: TestExecutionContext) -> <ast.Subscript object at 0x0000019433C84A60>

Parse text output when structured output is not available

##### add_progress_callback(self: Any, callback: <ast.Subscript object at 0x0000019433C84BB0>)

Add a progress monitoring callback

##### get_current_progress(self: Any) -> <ast.Subscript object at 0x0000019433C85000>

Get current execution progress

## Constants

### PYTEST

Type: `str`

Value: `pytest`

### UNITTEST

Type: `str`

Value: `unittest`

### CUSTOM

Type: `str`

Value: `custom`

