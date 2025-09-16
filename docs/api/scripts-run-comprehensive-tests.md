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

##### run_frontend_tests(self: Any) -> <ast.Subscript object at 0x0000019427BBB9A0>

Run frontend tests

##### run_backend_tests(self: Any) -> <ast.Subscript object at 0x00000194284AECB0>

Run backend tests

##### run_integration_tests(self: Any) -> <ast.Subscript object at 0x00000194284BE9B0>

Run full-stack integration tests

##### generate_coverage_report(self: Any) -> <ast.Subscript object at 0x000001942756BB80>

Generate code coverage report

##### run_security_tests(self: Any) -> <ast.Subscript object at 0x0000019427569E70>

Run security tests

##### generate_report(self: Any) -> <ast.Subscript object at 0x00000194275CDE70>

Generate comprehensive test report

##### generate_recommendations(self: Any) -> <ast.Subscript object at 0x0000019427BA47C0>

Generate recommendations based on test results

##### save_report(self: Any, report: <ast.Subscript object at 0x0000019427BA5C60>, output_file: str)

Save report to file

##### print_summary(self: Any, report: <ast.Subscript object at 0x000001942CD00E80>)

Print test summary to console

##### run_all_tests(self: Any, categories: <ast.Subscript object at 0x0000019427587700>)

Run all test categories

