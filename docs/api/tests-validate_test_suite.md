---
title: tests.validate_test_suite
category: api
tags: [api, tests]
---

# tests.validate_test_suite



## Classes

### TestSuiteValidator

Validator for comprehensive test suite

#### Methods

##### __init__(self: Any)



##### validate_file_structure(self: Any, file_path: Path) -> <ast.Subscript object at 0x000001942813B880>

Validate a test file's structure

##### validate_all_files(self: Any) -> bool

Validate all test files

##### print_summary(self: Any)

Print validation summary

##### check_dependencies(self: Any)

Check if required dependencies are available

##### validate_test_requirements(self: Any)

Validate that tests meet the requirements

