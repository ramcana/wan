---
title: tools.health-checker.recommendation_engine
category: api
tags: [api, tools]
---

# tools.health-checker.recommendation_engine

Actionable recommendations engine for project health improvements

## Classes

### RecommendationRule

Base class for recommendation rules

#### Methods

##### __init__(self: Any, name: str, category: HealthCategory, priority: int)



##### applies_to(self: Any, report: HealthReport) -> bool

Check if this rule applies to the given health report

##### generate_recommendation(self: Any, report: HealthReport) -> <ast.Subscript object at 0x000001942FA91660>

Generate recommendation based on health report

### TestSuiteRecommendationRule

Recommendations for test suite improvements

#### Methods

##### __init__(self: Any)



##### applies_to(self: Any, report: HealthReport) -> bool



##### generate_recommendation(self: Any, report: HealthReport) -> <ast.Subscript object at 0x000001942FA79810>



### DocumentationRecommendationRule

Recommendations for documentation improvements

#### Methods

##### __init__(self: Any)



##### applies_to(self: Any, report: HealthReport) -> bool



##### generate_recommendation(self: Any, report: HealthReport) -> <ast.Subscript object at 0x000001942FB307F0>



### ConfigurationRecommendationRule

Recommendations for configuration improvements

#### Methods

##### __init__(self: Any)



##### applies_to(self: Any, report: HealthReport) -> bool



##### generate_recommendation(self: Any, report: HealthReport) -> <ast.Subscript object at 0x0000019431A5A1A0>



### CodeQualityRecommendationRule

Recommendations for code quality improvements

#### Methods

##### __init__(self: Any)



##### applies_to(self: Any, report: HealthReport) -> bool



##### generate_recommendation(self: Any, report: HealthReport) -> <ast.Subscript object at 0x00000194319EB220>



### CriticalIssueRecommendationRule

Recommendations for addressing critical issues

#### Methods

##### __init__(self: Any)



##### applies_to(self: Any, report: HealthReport) -> bool



##### generate_recommendation(self: Any, report: HealthReport) -> <ast.Subscript object at 0x00000194319E9330>



### TrendBasedRecommendationRule

Recommendations based on health trends

#### Methods

##### __init__(self: Any)



##### applies_to(self: Any, report: HealthReport) -> bool



##### generate_recommendation(self: Any, report: HealthReport) -> <ast.Subscript object at 0x0000019431A72140>



### RecommendationEngine

Main recommendation engine that generates actionable improvement suggestions

#### Methods

##### __init__(self: Any, config: <ast.Subscript object at 0x0000019431A71F60>)



##### generate_recommendations(self: Any, report: HealthReport) -> <ast.Subscript object at 0x0000019432DE51E0>

Generate prioritized recommendations based on health report

Args:
    report: Current health report
    
Returns:
    List of recommendations sorted by priority

##### prioritize_recommendations(self: Any, recommendations: <ast.Subscript object at 0x0000019434630160>) -> <ast.Subscript object at 0x0000019434633820>

Re-prioritize recommendations based on additional criteria

Args:
    recommendations: List of recommendations to prioritize
    
Returns:
    Re-prioritized list of recommendations

##### _calculate_priority_score(self: Any, recommendation: Recommendation) -> float

Calculate dynamic priority score for recommendation

##### track_recommendation_progress(self: Any, recommendation_id: str, status: str, notes: str) -> None

Track progress on a recommendation

Args:
    recommendation_id: ID of the recommendation
    status: Current status (not_started, in_progress, completed, cancelled)
    notes: Optional progress notes

##### get_recommendation_progress(self: Any, recommendation_id: str) -> <ast.Subscript object at 0x0000019431AE7B50>

Get progress information for a recommendation

##### generate_implementation_plan(self: Any, recommendations: <ast.Subscript object at 0x000001942F8400D0>) -> <ast.Subscript object at 0x000001942F7AE2F0>

Generate an implementation plan for recommendations

Args:
    recommendations: List of recommendations to plan
    
Returns:
    Implementation plan with phases and timelines

##### _estimate_total_duration(self: Any, phases: <ast.Subscript object at 0x000001942F7AD360>) -> str

Estimate total implementation duration

##### _define_success_metrics(self: Any, recommendations: <ast.Subscript object at 0x000001942F830460>) -> <ast.Subscript object at 0x000001942F831720>

Define success metrics for implementation plan

##### _recommendation_to_dict(self: Any, recommendation: Recommendation) -> <ast.Subscript object at 0x000001942F831EA0>

Convert recommendation to dictionary format

##### add_custom_rule(self: Any, rule: RecommendationRule) -> None

Add a custom recommendation rule

##### get_recommendation_history(self: Any, limit: int) -> <ast.Subscript object at 0x000001942F8326E0>

Get recent recommendation history

##### export_recommendations(self: Any, recommendations: <ast.Subscript object at 0x000001942F8328F0>, format_type: str) -> str

Export recommendations in specified format

Args:
    recommendations: List of recommendations to export
    format_type: Export format ("json", "markdown", "csv")
    
Returns:
    Formatted string representation

##### _export_markdown(self: Any, recommendations: <ast.Subscript object at 0x000001942F833370>) -> str

Export recommendations as Markdown

##### _export_csv(self: Any, recommendations: <ast.Subscript object at 0x000001942FC64760>) -> str

Export recommendations as CSV

