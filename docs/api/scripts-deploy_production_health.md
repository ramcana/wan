---
title: scripts.deploy_production_health
category: api
tags: [api, scripts]
---

# scripts.deploy_production_health

Production Health Monitoring Deployment Script

This script handles the deployment of health monitoring to production environments,
including configuration validation, service setup, and initial health checks.

## Classes

### ProductionDeploymentManager

Manages deployment of health monitoring to production

#### Methods

##### __init__(self: Any, environment: str)



##### log_step(self: Any, message: str, success: bool) -> None

Log deployment step with timestamp

##### validate_environment(self: Any) -> bool

Validate production environment requirements

##### setup_configuration(self: Any) -> bool

Set up production configuration files

##### create_service_directories(self: Any) -> bool

Create necessary directories for production service

##### create_systemd_service(self: Any) -> bool

Create systemd service for production monitoring (Linux only)

##### create_deployment_summary(self: Any) -> None

Create deployment summary report

