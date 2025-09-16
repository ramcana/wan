---
title: tools.health-checker.production_deployment
category: api
tags: [api, tools]
---

# tools.health-checker.production_deployment

Production Health Monitoring Deployment System

This module handles the deployment and configuration of health monitoring
for production environments with appropriate thresholds and reporting.

## Classes

### ProductionHealthConfig

Production-specific health monitoring configuration

### ProductionHealthMonitor

Production health monitoring system with automated reporting and alerting

#### Methods

##### __init__(self: Any, config_path: <ast.Subscript object at 0x000001942CE129E0>)



##### _load_config(self: Any) -> ProductionHealthConfig

Load production health monitoring configuration

##### _save_config(self: Any, config: ProductionHealthConfig) -> None

Save production health monitoring configuration

##### _setup_logging(self: Any) -> logging.Logger

Set up production logging with appropriate levels and handlers

##### schedule_production_monitoring(self: Any) -> None

Schedule automated production health monitoring

##### _run_daily_health_report(self: Any) -> None

Run daily health report (synchronous wrapper)

##### _run_weekly_health_report(self: Any) -> None

Run weekly health report (synchronous wrapper)

##### _run_critical_health_check(self: Any) -> None

Run critical health check (synchronous wrapper)

##### _cleanup_old_reports(self: Any) -> None

Clean up old health reports

##### start_monitoring(self: Any) -> None

Start the production health monitoring system

##### stop_monitoring(self: Any) -> None

Stop the production health monitoring system

