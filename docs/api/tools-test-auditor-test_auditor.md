---
title: tools.test-auditor.test_auditor
category: api
tags: [api, tools]
---

# tools.test-auditor.test_auditor



## Classes

### TestIssue

Represents a specific issue found in a test

### TestFileAnalysis

Analysis results for a single test file

### TestSuiteAuditReport

Complete audit report for the entire test suite

### TestDiscoveryEngine

Discovers and categorizes test files across the project

#### Methods

##### __init__(self: Any, project_root: Path)



##### discover_test_files(self: Any) -> <ast.Subscript object at 0x00000194285DF310>

Discover all test files in the project

##### _scan_directory(self: Any, directory: Path) -> <ast.Subscript object at 0x00000194285DC9D0>

Recursively scan directory for test files

### TestDependencyAnalyzer

Analyzes test dependencies, imports, and fixtures

#### Methods

##### __init__(self: Any)



##### analyze_dependencies(self: Any, test_file: Path) -> <ast.Subscript object at 0x00000194285DC700>

Analyze test file dependencies
Returns: (imports, missing_imports, fixtures_used, missing_fixtures)

##### _extract_imports(self: Any, tree: ast.AST) -> <ast.Subscript object at 0x000001942C6348B0>

Extract all import statements from AST

##### _extract_fixtures(self: Any, tree: ast.AST) -> <ast.Subscript object at 0x000001942C6367A0>

Extract pytest fixtures used in test functions

##### _check_missing_imports(self: Any, imports: <ast.Subscript object at 0x000001942C6342E0>, test_file: Path) -> <ast.Subscript object at 0x000001942C635000>

Check for imports that cannot be resolved

##### _check_missing_fixtures(self: Any, fixtures: <ast.Subscript object at 0x000001942C636260>, test_file: Path) -> <ast.Subscript object at 0x0000019429CAE530>

Check for fixtures that are not defined

### TestPerformanceProfiler

Profiles test execution performance to identify slow tests

#### Methods

##### __init__(self: Any, timeout_seconds: int)



##### profile_test_file(self: Any, test_file: Path) -> <ast.Subscript object at 0x0000019429CAFE80>

Profile a single test file
Returns: (execution_time, slow_tests, timed_out)

##### _parse_slow_tests(self: Any, pytest_output: str) -> <ast.Subscript object at 0x0000019429CAD990>

Parse pytest output to identify slow tests

### TestAuditor

Main test auditor that orchestrates all analysis components

#### Methods

##### __init__(self: Any, project_root: Path)



##### audit_test_suite(self: Any) -> TestSuiteAuditReport

Perform comprehensive audit of the entire test suite

##### _analyze_test_file(self: Any, test_file: Path) -> TestFileAnalysis

Analyze a single test file comprehensively

##### _check_syntax(self: Any, test_file: Path) -> <ast.Subscript object at 0x0000019429C25420>

Check file for syntax errors

##### _analyze_test_structure(self: Any, test_file: Path) -> <ast.Subscript object at 0x0000019427F3AF50>

Analyze test file structure and identify issues

##### _has_assertions(self: Any, func_node: ast.FunctionDef) -> bool

Check if function contains assertions

##### _run_tests(self: Any, test_file: Path) -> <ast.Subscript object at 0x000001942C623280>

Run tests and return pass/fail/skip counts

##### _generate_recommendations(self: Any, file_analyses: <ast.Subscript object at 0x000001942C620910>) -> <ast.Subscript object at 0x0000019428473F40>

Generate actionable recommendations based on analysis

