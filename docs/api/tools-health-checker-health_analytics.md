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

##### __init__(self: Any, history_file: <ast.Subscript object at 0x0000019431A391B0>)



##### analyze_health_trends(self: Any, days: int) -> <ast.Subscript object at 0x000001942F1CA0B0>

Analyze health trends over specified time period

Args:
    days: Number of days to analyze
    
Returns:
    Comprehensive trend analysis

##### analyze_component_performance(self: Any, component: str, days: int) -> <ast.Subscript object at 0x0000019434544A60>

Analyze performance of a specific component

##### generate_health_insights(self: Any, report: HealthReport) -> <ast.Subscript object at 0x0000019434545E40>

Generate actionable insights from current health report

##### calculate_health_metrics(self: Any, report: HealthReport) -> <ast.Subscript object at 0x000001942EFF9D20>

Calculate advanced health metrics

##### _load_history(self: Any) -> <ast.Subscript object at 0x000001942EFFBA00>

Load historical health data

##### _analyze_scores(self: Any, score_history: <ast.Subscript object at 0x000001942EFFB280>) -> <ast.Subscript object at 0x000001942EFFB4F0>

Analyze score statistics

##### _analyze_trends(self: Any, score_history: <ast.Subscript object at 0x000001942EFF8280>) -> <ast.Subscript object at 0x000001942F049000>

Analyze trend direction and strength

##### _analyze_volatility(self: Any, score_history: <ast.Subscript object at 0x000001942F049390>) -> <ast.Subscript object at 0x000001942F049300>

Analyze score volatility and stability

##### _analyze_patterns(self: Any, score_history: <ast.Subscript object at 0x000001942F048640>) -> <ast.Subscript object at 0x0000019431BDED70>

Analyze patterns in health data

##### _forecast_trends(self: Any, score_history: <ast.Subscript object at 0x0000019431BDE1D0>) -> <ast.Subscript object at 0x0000019431BDC760>

Simple trend forecasting

##### _calculate_short_term_trend(self: Any, recent_scores: <ast.Subscript object at 0x0000019431BDD7E0>) -> str

Calculate short-term trend from recent scores

##### _detect_cyclical_patterns(self: Any, scores: <ast.Subscript object at 0x0000019431BDC5E0>) -> <ast.Subscript object at 0x0000019432E36200>

Detect cyclical patterns in scores

##### _detect_anomalies(self: Any, scores: <ast.Subscript object at 0x0000019432E35ED0>) -> <ast.Subscript object at 0x0000019432E36620>

Detect anomalous scores

##### _detect_recovery_patterns(self: Any, scores: <ast.Subscript object at 0x0000019432E356C0>) -> <ast.Subscript object at 0x0000019432E346D0>

Detect recovery patterns after drops

##### _detect_degradation_patterns(self: Any, scores: <ast.Subscript object at 0x0000019432E37460>) -> <ast.Subscript object at 0x0000019432E34850>

Detect degradation patterns

##### _calculate_risk_metrics(self: Any, report: HealthReport) -> <ast.Subscript object at 0x0000019432DE7970>

Calculate risk-related metrics

##### _calculate_quality_metrics(self: Any, report: HealthReport) -> <ast.Subscript object at 0x0000019432DE53F0>

Calculate quality-related metrics

##### _calculate_stability_metrics(self: Any, report: HealthReport) -> <ast.Subscript object at 0x0000019432DE4AC0>

Calculate stability-related metrics

##### _calculate_performance_metrics(self: Any, report: HealthReport) -> <ast.Subscript object at 0x0000019432DE5DB0>

Calculate performance-related metrics

##### _calculate_score_volatility(self: Any, score_history: <ast.Subscript object at 0x0000019432DE4D30>) -> float

Calculate volatility from score history

##### _generate_analytics_recommendations(self: Any, analysis: <ast.Subscript object at 0x000001942F42EC50>) -> <ast.Subscript object at 0x000001942F42FAF0>

Generate recommendations based on analytics

##### _generate_component_recommendations(self: Any, component: str, data: <ast.Subscript object at 0x000001942F42DF60>) -> <ast.Subscript object at 0x000001942F42E2F0>

Generate component-specific recommendations

##### _prioritize_actions(self: Any, insights: <ast.Subscript object at 0x000001942F42C160>, report: HealthReport) -> <ast.Subscript object at 0x000001942F42C3D0>

Prioritize actions based on insights

