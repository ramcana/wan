---
title: scripts.run-comprehensive-tests
category: api
tags: [api, scripts]
---

# scripts.run-comprehensive-tests



## Classes

### ComprehensiveTestRunner



#### Methods

##### __init__(self: Any, project_root: Path)



##### setup_environment(self: Any)

Setup test environment

##### run_frontend_tests(self: Any) -> <ast.Subscript object at 0x000001942F048F40>

Run frontend tests

##### run_backend_tests(self: Any) -> <ast.Subscript object at 0x0000019433D2DC00>

Run backend tests

##### run_integration_tests(self: Any) -> <ast.Subscript object at 0x000001943448DD80>

Run full-stack integration tests

##### generate_coverage_report(self: Any) -> <ast.Subscript object at 0x000001943448FBB0>

Generate code coverage report

##### run_security_tests(self: Any) -> <ast.Subscript object at 0x0000019434446A10>

Run security tests

##### generate_report(self: Any) -> <ast.Subscript object at 0x00000194344448B0>

Generate comprehensive test report

##### generate_recommendations(self: Any) -> <ast.Subscript object at 0x00000194319062F0>

Generate recommendations based on test results

##### save_report(self: Any, report: <ast.Subscript object at 0x0000019431904040>, output_file: str)

Save report to file

##### print_summary(self: Any, report: <ast.Subscript object at 0x00000194319052D0>)

Print test summary to console

##### run_all_tests(self: Any, categories: <ast.Subscript object at 0x000001942F8432B0>)

Run all test categories

