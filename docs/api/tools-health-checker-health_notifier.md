---
title: tools.health-checker.health_notifier
category: api
tags: [api, tools]
---

# tools.health-checker.health_notifier

Health notification and alerting system

## Classes

### NotificationChannel

Base class for notification channels

#### Methods

##### __init__(self: Any, name: str, config: <ast.Subscript object at 0x000001942C6D3280>)



##### is_enabled(self: Any) -> bool

Check if this channel is enabled

### ConsoleNotificationChannel

Console/stdout notification channel

### EmailNotificationChannel

Email notification channel

#### Methods

##### _create_email_body(self: Any, message: str, severity: Severity, metadata: <ast.Subscript object at 0x000001942CE2FC40>) -> str

Create HTML email body

### SlackNotificationChannel

Slack notification channel

#### Methods

##### _create_slack_message(self: Any, message: str, severity: Severity, metadata: <ast.Subscript object at 0x00000194288F5A20>) -> <ast.Subscript object at 0x00000194288F4A30>

Create Slack message payload

### WebhookNotificationChannel

Generic webhook notification channel

### FileNotificationChannel

File-based notification channel

### HealthNotifier

Main health notification system that manages multiple channels and alert rules

#### Methods

##### __init__(self: Any, config: <ast.Subscript object at 0x0000019427B62890>)



##### _initialize_channels(self: Any)

Initialize notification channels

##### _load_alert_rules(self: Any) -> <ast.Subscript object at 0x0000019427C798D0>

Load alert rules configuration

##### _should_trigger_alert(self: Any, rule: <ast.Subscript object at 0x0000019428DC9150>, report: HealthReport) -> bool

Check if alert rule should trigger

##### _create_alert_message(self: Any, rule: <ast.Subscript object at 0x000001942CCB6B60>, report: HealthReport) -> str

Create alert message based on rule and report

##### get_notification_history(self: Any, limit: int) -> <ast.Subscript object at 0x000001942CE22200>

Get recent notification history

##### add_custom_channel(self: Any, name: str, channel: NotificationChannel)

Add a custom notification channel

##### add_custom_rule(self: Any, rule: <ast.Subscript object at 0x000001942CE22D10>)

Add a custom alert rule

##### test_notifications(self: Any) -> <ast.Subscript object at 0x000001942CE22290>

Test all notification channels

### CIPipelineIntegration

Integration with CI/CD pipelines

#### Methods

##### __init__(self: Any, notifier: HealthNotifier)



##### create_github_status(self: Any, report: HealthReport) -> <ast.Subscript object at 0x0000019428470BE0>

Create GitHub status check payload

##### set_exit_code(self: Any, report: HealthReport, threshold: float) -> int

Set appropriate exit code for CI/CD

## Constants

### REQUESTS_AVAILABLE

Type: `bool`

Value: `True`

### REQUESTS_AVAILABLE

Type: `bool`

Value: `False`

