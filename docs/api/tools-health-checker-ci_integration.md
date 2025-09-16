---
title: tools.health-checker.ci_integration
category: api
tags: [api, tools]
---

# tools.health-checker.ci_integration

CI/CD integration utilities for health monitoring.

This module provides utilities for integrating health monitoring
with CI/CD pipelines and deployment gates.

## Classes

### CIHealthIntegration

Integrates health monitoring with CI/CD pipelines.

#### Methods

##### __init__(self: Any, config_path: <ast.Subscript object at 0x0000019427B63CA0>)



##### load_config(self: Any)

Load CI health integration configuration.

##### run_health_check_for_ci(self: Any, comprehensive: bool, deployment_gate: bool) -> Dict

Run health check optimized for CI environment.

##### evaluate_deployment_gate(self: Any, health_report: Dict, branch: str) -> <ast.Subscript object at 0x0000019427CA57B0>

Evaluate if deployment gate requirements are met.

##### generate_ci_summary(self: Any, health_report: Dict, deployment_gate_result: <ast.Subscript object at 0x0000019427CA6260>) -> str

Generate CI-friendly summary of health check results.

##### set_github_outputs(self: Any, health_report: Dict, deployment_gate_result: <ast.Subscript object at 0x0000019427CA55D0>)

Set GitHub Actions outputs for use in other steps.

##### create_status_check(self: Any, health_report: Dict, context: str) -> bool

Create GitHub status check for health monitoring.

##### notify_health_issues(self: Any, health_report: Dict)

Send notifications for health issues.

##### _send_slack_notification(self: Any, health_report: Dict, critical_issues: <ast.Subscript object at 0x000001942CC45E70>, webhook_url: str)

Send Slack notification for critical health issues.

