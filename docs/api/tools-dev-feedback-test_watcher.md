---
title: tools.dev-feedback.test_watcher
category: api
tags: [api, tools]
---

# tools.dev-feedback.test_watcher



## Classes

### TestResult

Test execution result

### WatchConfig

Configuration for test watcher

### TestFileHandler

File system event handler for test watching

#### Methods

##### __init__(self: Any, watcher: TestWatcher)



##### on_modified(self: Any, event: Any)



##### on_created(self: Any, event: Any)



### TestWatcher

Watch files and run tests with selective execution

#### Methods

##### __init__(self: Any, config: WatchConfig, project_root: <ast.Subscript object at 0x0000019431ABBE50>)



##### _initialize_file_hashes(self: Any)

Initialize file hashes for change detection

##### _get_file_hash(self: Any, file_path: Path) -> str

Get hash of file content

##### should_trigger_tests(self: Any, file_path: Path) -> bool

Check if file change should trigger test execution

##### schedule_test_run(self: Any, changed_file: Path)

Schedule test run with debouncing

##### _execute_pending_tests(self: Any)

Execute pending test runs

##### _determine_tests_to_run(self: Any, changed_files: <ast.Subscript object at 0x00000194346328F0>) -> <ast.Subscript object at 0x0000019434630A30>

Determine which tests to run based on changed files

##### _is_test_file(self: Any, file_path: Path) -> bool

Check if file is a test file

##### _find_related_tests(self: Any, source_file: Path) -> <ast.Subscript object at 0x00000194300D95A0>

Find test files related to a source file

##### _matches_test_criteria(self: Any, test_file: Path) -> bool

Check if test file matches configured criteria

##### _get_test_category(self: Any, test_file: Path) -> str

Determine test category from file path

##### _is_slow_test(self: Any, test_file: Path) -> bool

Check if test is considered slow

##### _run_tests(self: Any, test_files: <ast.Subscript object at 0x00000194300D9360>)

Run the specified test files

##### _run_test_batch(self: Any, test_files: <ast.Subscript object at 0x00000194300DBFA0>) -> <ast.Subscript object at 0x00000194300DB670>

Run a batch of test files

##### _run_single_test(self: Any, test_file: Path) -> TestResult

Run a single test file

##### _report_test_results(self: Any, results: <ast.Subscript object at 0x000001942EF9EBC0>, total_duration: float)

Report test execution results

##### start_watching(self: Any)

Start watching for file changes

##### stop_watching(self: Any)

Stop watching for file changes

##### run_all_tests(self: Any)

Run all tests matching criteria

