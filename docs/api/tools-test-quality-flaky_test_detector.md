---
title: tools.test-quality.flaky_test_detector
category: api
tags: [api, tools]
---

# tools.test-quality.flaky_test_detector



## Classes

### TestExecution

Single test execution record

### FlakyTestPattern

Pattern analysis for a flaky test

### FlakyTestRecommendation

Recommendation for fixing a flaky test

### QuarantineDecision

Decision about quarantining a flaky test

### FlakyTestStatisticalAnalyzer

Performs statistical analysis to identify flaky tests

#### Methods

##### __init__(self: Any)



##### analyze_test_flakiness(self: Any, executions: <ast.Subscript object at 0x000001942CE6D870>) -> <ast.Subscript object at 0x000001942CE6DD80>

Analyze test executions to identify flaky patterns

##### _analyze_single_test(self: Any, test_id: str, executions: <ast.Subscript object at 0x000001942CE6FE50>) -> FlakyTestPattern

Analyze flakiness for a single test

##### _calculate_flakiness_score(self: Any, executions: <ast.Subscript object at 0x000001942CE6D210>) -> float

Calculate flakiness score based on execution patterns

##### _analyze_error_patterns(self: Any, executions: <ast.Subscript object at 0x000001942C5F2C20>) -> <ast.Subscript object at 0x000001942CE2DBD0>

Analyze common error patterns

##### _analyze_failure_patterns(self: Any, executions: <ast.Subscript object at 0x000001942CE2E3E0>) -> <ast.Subscript object at 0x0000019427B2F9D0>

Analyze patterns in test failures

##### _analyze_time_patterns(self: Any, executions: <ast.Subscript object at 0x0000019427B2CB50>) -> <ast.Subscript object at 0x0000019427B2ECB0>

Analyze time-based failure patterns

##### _analyze_environment_patterns(self: Any, executions: <ast.Subscript object at 0x0000019427B2D570>) -> <ast.Subscript object at 0x0000019427B2E320>

Analyze environment-based failure patterns

##### _analyze_duration_patterns(self: Any, executions: <ast.Subscript object at 0x0000019427B2F820>) -> <ast.Subscript object at 0x000001942A2E7460>

Analyze duration-based patterns

##### _analyze_sequence_patterns(self: Any, executions: <ast.Subscript object at 0x000001942A2E7310>) -> <ast.Subscript object at 0x000001942CC897B0>

Analyze sequential failure patterns

##### _calculate_confidence(self: Any, executions: <ast.Subscript object at 0x000001942CC8AA70>, flakiness_score: float) -> float

Calculate confidence in flakiness assessment

### FlakyTestTracker

Tracks flaky test executions and maintains historical data

#### Methods

##### __init__(self: Any, project_root: Path)



##### _init_database(self: Any)

Initialize flaky test tracking database

##### record_test_execution(self: Any, execution: TestExecution)

Record a single test execution

##### record_test_executions(self: Any, executions: <ast.Subscript object at 0x000001942CC8AA10>)

Record multiple test executions

##### get_test_executions(self: Any, test_id: str, days: int) -> <ast.Subscript object at 0x0000019427ADA830>

Get test executions from the database

##### update_flaky_pattern(self: Any, pattern: FlakyTestPattern)

Update or insert flaky test pattern

##### get_flaky_patterns(self: Any, min_flakiness: float) -> <ast.Subscript object at 0x0000019427ADA8F0>

Get flaky test patterns from database

### FlakyTestRecommendationEngine

Generates recommendations for fixing flaky tests

#### Methods

##### __init__(self: Any)



##### generate_recommendations(self: Any, patterns: <ast.Subscript object at 0x00000194281205E0>) -> <ast.Subscript object at 0x0000019428122FE0>

Generate recommendations for fixing flaky tests

##### _analyze_pattern_for_recommendations(self: Any, pattern: FlakyTestPattern) -> <ast.Subscript object at 0x0000019428122DA0>

Analyze a single flaky pattern to generate recommendations

##### _analyze_failure_patterns_for_recommendations(self: Any, pattern: FlakyTestPattern) -> <ast.Subscript object at 0x0000019427A6BFA0>

Generate recommendations based on failure patterns

##### _determine_priority(self: Any, pattern: FlakyTestPattern, error_frequency: float) -> str

Determine priority based on pattern characteristics

### FlakyTestQuarantineManager

Manages quarantine decisions for flaky tests

#### Methods

##### __init__(self: Any, project_root: Path)



##### evaluate_quarantine_decisions(self: Any, patterns: <ast.Subscript object at 0x0000019429CB7220>) -> <ast.Subscript object at 0x0000019429CB6FE0>

Evaluate which tests should be quarantined

##### _evaluate_single_test_quarantine(self: Any, pattern: FlakyTestPattern) -> QuarantineDecision

Evaluate quarantine decision for a single test

##### apply_quarantine(self: Any, decision: QuarantineDecision)

Apply quarantine decision to a test

##### _create_quarantine_marker(self: Any, decision: QuarantineDecision)

Create pytest marker for quarantined test

### FlakyTestDetectionSystem

Main system that orchestrates flaky test detection and management

#### Methods

##### __init__(self: Any, project_root: Path)



##### run_flaky_test_analysis(self: Any, days: int) -> <ast.Subscript object at 0x000001942836DC60>

Run comprehensive flaky test analysis

##### record_test_run_results(self: Any, test_results_file: Path)

Record test run results from pytest output

##### _parse_test_results(self: Any, results_file: Path) -> <ast.Subscript object at 0x00000194282EFAF0>

Parse test results from file

##### _get_current_commit(self: Any) -> <ast.Subscript object at 0x00000194282EF520>

Get current git commit hash

##### _get_current_branch(self: Any) -> <ast.Subscript object at 0x00000194282EDAE0>

Get current git branch

##### save_analysis_report(self: Any, analysis_result: <ast.Subscript object at 0x00000194282EE110>, output_path: Path)

Save flaky test analysis report

