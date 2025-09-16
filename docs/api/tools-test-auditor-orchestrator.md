---
title: tools.test-auditor.orchestrator
category: api
tags: [api, tools]
---

# tools.test-auditor.orchestrator

Test Suite Audit Orchestrator

Main orchestrator that coordinates all test auditing components to provide
a comprehensive analysis of the test suite health and quality.

## Classes

### ComprehensiveTestAnalysis

Complete test suite analysis combining all audit components

### TestSuiteHealthScorer

Calculates overall health score for test suite

#### Methods

##### __init__(self: Any)



##### calculate_health_score(self: Any, analysis: ComprehensiveTestAnalysis) -> float

Calculate overall health score (0-100)

##### _calculate_syntax_health(self: Any, audit_report: TestSuiteAuditReport) -> float

Calculate syntax health score

##### _calculate_completeness_score(self: Any, audit_report: TestSuiteAuditReport) -> float

Calculate test completeness score

##### _calculate_execution_score(self: Any, execution_report: TestSuiteExecutionReport) -> float

Calculate execution success score

##### _calculate_performance_score(self: Any, execution_report: TestSuiteExecutionReport) -> float

Calculate performance score

##### _calculate_coverage_score(self: Any, coverage_report: CoverageReport) -> float

Calculate coverage score

##### _calculate_reliability_score(self: Any, execution_report: TestSuiteExecutionReport) -> float

Calculate reliability score based on retries and timeouts

### ActionPlanGenerator

Generates actionable plans to improve test suite health

#### Methods

##### generate_action_plan(self: Any, analysis: ComprehensiveTestAnalysis) -> <ast.Subscript object at 0x0000019434132740>

Generate prioritized action plan

##### _generate_critical_actions(self: Any, analysis: ComprehensiveTestAnalysis) -> <ast.Subscript object at 0x0000019434130460>

Generate critical actions that must be addressed immediately

##### _generate_high_priority_actions(self: Any, analysis: ComprehensiveTestAnalysis) -> <ast.Subscript object at 0x0000019434130FD0>

Generate high priority actions

##### _generate_medium_priority_actions(self: Any, analysis: ComprehensiveTestAnalysis) -> <ast.Subscript object at 0x0000019431B8E770>

Generate medium priority actions

##### _generate_long_term_actions(self: Any, analysis: ComprehensiveTestAnalysis) -> <ast.Subscript object at 0x0000019431B8DD50>

Generate long-term improvement actions

### TestSuiteOrchestrator

Main orchestrator for comprehensive test suite analysis

#### Methods

##### __init__(self: Any, project_root: Path)



##### run_comprehensive_analysis(self: Any) -> ComprehensiveTestAnalysis

Run complete test suite analysis

##### _generate_analysis_summary(self: Any, audit_report: TestSuiteAuditReport, execution_report: TestSuiteExecutionReport, coverage_report: CoverageReport, total_time: float) -> <ast.Subscript object at 0x000001942FA8DC30>

Generate comprehensive analysis summary

##### _calculate_test_distribution(self: Any, audit_report: TestSuiteAuditReport) -> <ast.Subscript object at 0x000001942FA8CB50>

Calculate distribution of tests across categories

##### _generate_comprehensive_recommendations(self: Any, audit_report: TestSuiteAuditReport, execution_report: TestSuiteExecutionReport, coverage_report: CoverageReport) -> <ast.Subscript object at 0x000001942FA8EEC0>

Generate comprehensive recommendations combining all analyses

##### save_analysis(self: Any, analysis: ComprehensiveTestAnalysis, output_file: Path)

Save comprehensive analysis to file

##### print_summary(self: Any, analysis: ComprehensiveTestAnalysis)

Print analysis summary

