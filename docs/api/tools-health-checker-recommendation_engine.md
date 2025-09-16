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

##### generate_recommendation(self: Any, report: HealthReport) -> <ast.Subscript object at 0x0000019427F0D570>

Generate recommendation based on health report

### TestSuiteRecommendationRule

Recommendations for test suite improvements

#### Methods

##### __init__(self: Any)



##### applies_to(self: Any, report: HealthReport) -> bool



##### generate_recommendation(self: Any, report: HealthReport) -> <ast.Subscript object at 0x0000019427F7F0A0>



### DocumentationRecommendationRule

Recommendations for documentation improvements

#### Methods

##### __init__(self: Any)



##### applies_to(self: Any, report: HealthReport) -> bool



##### generate_recommendation(self: Any, report: HealthReport) -> <ast.Subscript object at 0x0000019427F5C700>



### ConfigurationRecommendationRule

Recommendations for configuration improvements

#### Methods

##### __init__(self: Any)



##### applies_to(self: Any, report: HealthReport) -> bool



##### generate_recommendation(self: Any, report: HealthReport) -> <ast.Subscript object at 0x000001942A1620B0>



### CodeQualityRecommendationRule

Recommendations for code quality improvements

#### Methods

##### __init__(self: Any)



##### applies_to(self: Any, report: HealthReport) -> bool



##### generate_recommendation(self: Any, report: HealthReport) -> <ast.Subscript object at 0x0000019429C43130>



### CriticalIssueRecommendationRule

Recommendations for addressing critical issues

#### Methods

##### __init__(self: Any)



##### applies_to(self: Any, report: HealthReport) -> bool



##### generate_recommendation(self: Any, report: HealthReport) -> <ast.Subscript object at 0x0000019429C41240>



### TrendBasedRecommendationRule

Recommendations based on health trends

#### Methods

##### __init__(self: Any)



##### applies_to(self: Any, report: HealthReport) -> bool



##### generate_recommendation(self: Any, report: HealthReport) -> <ast.Subscript object at 0x0000019429CB6050>



### RecommendationEngine

Main recommendation engine that generates actionable improvement suggestions

#### Methods

##### __init__(self: Any, config: <ast.Subscript object at 0x0000019429CB5E70>)



##### generate_recommendations(self: Any, report: HealthReport) -> <ast.Subscript object at 0x0000019429C8BD00>

Generate prioritized recommendations based on health report

Args:
    report: Current health report
    
Returns:
    List of recommendations sorted by priority

##### prioritize_recommendations(self: Any, recommendations: <ast.Subscript object at 0x0000019429C8B250>) -> <ast.Subscript object at 0x0000019429C8AB30>

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

##### get_recommendation_progress(self: Any, recommendation_id: str) -> <ast.Subscript object at 0x0000019427BBEC20>

Get progress information for a recommendation

##### generate_implementation_plan(self: Any, recommendations: <ast.Subscript object at 0x0000019427B92830>) -> <ast.Subscript object at 0x0000019427FDFB80>

Generate an implementation plan for recommendations

Args:
    recommendations: List of recommendations to plan
    
Returns:
    Implementation plan with phases and timelines

##### _estimate_total_duration(self: Any, phases: <ast.Subscript object at 0x0000019427FDFD30>) -> str

Estimate total implementation duration

##### _define_success_metrics(self: Any, recommendations: <ast.Subscript object at 0x000001942C8E0370>) -> <ast.Subscript object at 0x000001942C8E1630>

Define success metrics for implementation plan

##### _recommendation_to_dict(self: Any, recommendation: Recommendation) -> <ast.Subscript object at 0x000001942C8E1DB0>

Convert recommendation to dictionary format

##### add_custom_rule(self: Any, rule: RecommendationRule) -> None

Add a custom recommendation rule

##### get_recommendation_history(self: Any, limit: int) -> <ast.Subscript object at 0x000001942C8E25F0>

Get recent recommendation history

##### export_recommendations(self: Any, recommendations: <ast.Subscript object at 0x000001942C8E2800>, format_type: str) -> str

Export recommendations in specified format

Args:
    recommendations: List of recommendations to export
    format_type: Export format ("json", "markdown", "csv")
    
Returns:
    Formatted string representation

##### _export_markdown(self: Any, recommendations: <ast.Subscript object at 0x000001942C8E3280>) -> str

Export recommendations as Markdown

##### _export_csv(self: Any, recommendations: <ast.Subscript object at 0x000001942C8B8670>) -> str

Export recommendations as CSV

