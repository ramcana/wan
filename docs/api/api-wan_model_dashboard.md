---
title: api.wan_model_dashboard
category: api
tags: [api, api]
---

# api.wan_model_dashboard

WAN Model Dashboard Integration
Provides dashboard-specific endpoints and real-time data for WAN model monitoring

## Classes

### DashboardMetrics

Dashboard metrics summary

### ModelStatusSummary

Model status summary for dashboard

### SystemAlert

System alert for dashboard

### WANModelDashboard

WAN Model Dashboard Integration

#### Methods

##### __init__(self: Any)



##### _generate_model_cards_html(self: Any, model_summaries: <ast.Subscript object at 0x000001942CD45A80>) -> str

Generate HTML for model cards

##### _generate_alerts_html(self: Any, alerts: <ast.Subscript object at 0x000001942CD45C90>) -> str

Generate HTML for alerts

##### _is_cache_valid(self: Any, cache_key: str) -> bool

Check if cache entry is still valid

## Constants

### WAN_API_AVAILABLE

Type: `bool`

Value: `True`

### WAN_API_AVAILABLE

Type: `bool`

Value: `False`

