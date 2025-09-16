---
title: tools.health-checker.health_analytics
category: api
tags: [api, tools]
---

# tools.health-checker.health_analytics

Health analytics and trend analysis system

## Classes

### HealthAnalytics

Advanced analytics for health monitoring data

#### Methods

##### __init__(self: Any, history_file: <ast.Subscript object at 0x0000019429C9C880>)



##### analyze_health_trends(self: Any, days: int) -> <ast.Subscript object at 0x0000019429C266B0>

Analyze health trends over specified time period

Args:
    days: Number of days to analyze
    
Returns:
    Comprehensive trend analysis

##### analyze_component_performance(self: Any, component: str, days: int) -> <ast.Subscript object at 0x0000019428985AE0>

Analyze performance of a specific component

##### generate_health_insights(self: Any, report: HealthReport) -> <ast.Subscript object at 0x0000019429CAE470>

Generate actionable insights from current health report

##### calculate_health_metrics(self: Any, report: HealthReport) -> <ast.Subscript object at 0x0000019429CACDF0>

Calculate advanced health metrics

##### _load_history(self: Any) -> <ast.Subscript object at 0x000001942CE6CC40>

Load historical health data

##### _analyze_scores(self: Any, score_history: <ast.Subscript object at 0x000001942CE6FB50>) -> <ast.Subscript object at 0x000001942CE6D870>

Analyze score statistics

##### _analyze_trends(self: Any, score_history: <ast.Subscript object at 0x000001942CE6FA00>) -> <ast.Subscript object at 0x000001942C81EBC0>

Analyze trend direction and strength

##### _analyze_volatility(self: Any, score_history: <ast.Subscript object at 0x00000194288F1600>) -> <ast.Subscript object at 0x00000194288F1060>

Analyze score volatility and stability

##### _analyze_patterns(self: Any, score_history: <ast.Subscript object at 0x00000194288F2080>) -> <ast.Subscript object at 0x00000194288F0DC0>

Analyze patterns in health data

##### _forecast_trends(self: Any, score_history: <ast.Subscript object at 0x00000194288F0430>) -> <ast.Subscript object at 0x0000019427A957E0>

Simple trend forecasting

##### _calculate_short_term_trend(self: Any, recent_scores: <ast.Subscript object at 0x0000019427A979D0>) -> str

Calculate short-term trend from recent scores

##### _detect_cyclical_patterns(self: Any, scores: <ast.Subscript object at 0x0000019427A97F10>) -> <ast.Subscript object at 0x000001942C6361D0>

Detect cyclical patterns in scores

##### _detect_anomalies(self: Any, scores: <ast.Subscript object at 0x000001942C6373D0>) -> <ast.Subscript object at 0x000001942C6358A0>

Detect anomalous scores

##### _detect_recovery_patterns(self: Any, scores: <ast.Subscript object at 0x000001942C6347F0>) -> <ast.Subscript object at 0x000001942C6379D0>

Detect recovery patterns after drops

##### _detect_degradation_patterns(self: Any, scores: <ast.Subscript object at 0x000001942C634190>) -> <ast.Subscript object at 0x000001942C636170>

Detect degradation patterns

##### _calculate_risk_metrics(self: Any, report: HealthReport) -> <ast.Subscript object at 0x0000019428118250>

Calculate risk-related metrics

##### _calculate_quality_metrics(self: Any, report: HealthReport) -> <ast.Subscript object at 0x00000194281182B0>

Calculate quality-related metrics

##### _calculate_stability_metrics(self: Any, report: HealthReport) -> <ast.Subscript object at 0x000001942811B8B0>

Calculate stability-related metrics

##### _calculate_performance_metrics(self: Any, report: HealthReport) -> <ast.Subscript object at 0x000001942811B670>

Calculate performance-related metrics

##### _calculate_score_volatility(self: Any, score_history: <ast.Subscript object at 0x0000019428119B40>) -> float

Calculate volatility from score history

##### _generate_analytics_recommendations(self: Any, analysis: <ast.Subscript object at 0x0000019428D7F8E0>) -> <ast.Subscript object at 0x0000019428D7DD80>

Generate recommendations based on analytics

##### _generate_component_recommendations(self: Any, component: str, data: <ast.Subscript object at 0x0000019428D7F070>) -> <ast.Subscript object at 0x0000019428D7C880>

Generate component-specific recommendations

##### _prioritize_actions(self: Any, insights: <ast.Subscript object at 0x0000019428D7F700>, report: HealthReport) -> <ast.Subscript object at 0x0000019428D7C490>

Prioritize actions based on insights

