---
title: tools.health-checker.dashboard_server
category: api
tags: [api, tools]
---

# tools.health-checker.dashboard_server

Simple health monitoring dashboard server

## Classes

### HealthDashboard

Real-time health monitoring dashboard

#### Methods

##### __init__(self: Any, config: <ast.Subscript object at 0x00000194283E22F0>)



##### _create_fastapi_app(self: Any) -> FastAPI

Create FastAPI application for dashboard

##### _should_update_data(self: Any) -> bool

Check if dashboard data should be updated

##### _get_dashboard_html(self: Any) -> str

Get dashboard HTML content

##### run_server(self: Any, host: str, port: int)

Run the dashboard server

## Constants

### FASTAPI_AVAILABLE

Type: `bool`

Value: `True`

### FASTAPI_AVAILABLE

Type: `bool`

Value: `False`

