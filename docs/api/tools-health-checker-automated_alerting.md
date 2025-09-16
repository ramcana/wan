---
title: tools.health-checker.automated_alerting
category: api
tags: [api, tools]
---

# tools.health-checker.automated_alerting

Automated Alerting and Notification System

This module handles automated alerting for critical health issues,
escalation policies, and integration with project management tools.

## Classes

### AlertLevel

Alert severity levels

### AlertRule

Configuration for an alert rule

### EscalationPolicy

Escalation policy configuration

### AlertHistory

Track alert history for rate limiting and escalation

### AutomatedAlertingSystem

Automated alerting system with escalation and rate limiting

#### Methods

##### __init__(self: Any, config_path: <ast.Subscript object at 0x0000019428137EE0>)



##### _setup_logging(self: Any) -> logging.Logger

Set up logging for alerting system

##### _load_alert_rules(self: Any) -> <ast.Subscript object at 0x0000019428160250>

Load alert rules from configuration

##### _create_default_alert_rules(self: Any) -> <ast.Subscript object at 0x0000019428133070>

Create default alert rules

##### _load_escalation_policies(self: Any) -> <ast.Subscript object at 0x0000019428131F60>

Load escalation policies from configuration

##### _create_default_escalation_policies(self: Any) -> <ast.Subscript object at 0x0000019428131240>

Create default escalation policies

##### _should_trigger_alert(self: Any, rule: AlertRule, now: datetime) -> bool

Check if alert should be triggered based on rate limiting

##### _create_alert_message(self: Any, rule: AlertRule, health_report: HealthReport, history: AlertHistory) -> <ast.Subscript object at 0x0000019428400EE0>

Create alert message with relevant information

##### _get_escalation_policy(self: Any, alert_level: AlertLevel) -> <ast.Subscript object at 0x000001942C6662C0>

Get appropriate escalation policy for alert level

##### acknowledge_alert(self: Any, rule_name: str, acknowledged_by: str) -> bool

Acknowledge an alert to stop escalation

##### resolve_alert(self: Any, rule_name: str, resolved_by: str, resolution_notes: str) -> bool

Mark an alert as resolved

##### get_active_alerts(self: Any) -> <ast.Subscript object at 0x000001942C629B70>

Get list of active (unresolved) alerts

##### cleanup_old_alerts(self: Any, days: int) -> None

Clean up old resolved alerts

##### save_configuration(self: Any) -> None

Save current alert rules and escalation policies to configuration file

## Constants

### INFO

Type: `str`

Value: `info`

### WARNING

Type: `str`

Value: `warning`

### CRITICAL

Type: `str`

Value: `critical`

### EMERGENCY

Type: `str`

Value: `emergency`

