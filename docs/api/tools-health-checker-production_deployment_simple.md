---
title: tools.health-checker.production_deployment_simple
category: api
tags: [api, tools]
---

# tools.health-checker.production_deployment_simple

Simplified Production Health Monitoring Deployment System

This module handles the deployment and configuration of health monitoring
for production environments with standard library dependencies only.

## Classes

### ProductionHealthConfig

Production-specific health monitoring configuration

### ProductionHealthMonitor

Production health monitoring system with automated reporting and alerting

#### Methods

##### __init__(self: Any, config_path: <ast.Subscript object at 0x000001942F33C430>)



##### _load_config(self: Any) -> ProductionHealthConfig

Load production health monitoring configuration

##### _save_config(self: Any, config: ProductionHealthConfig) -> None

Save production health monitoring configuration

##### _setup_logging(self: Any) -> logging.Logger

Set up production logging with appropriate levels and handlers

