#!/usr/bin/env python3
"""
Automated Alerting and Notification System

This module handles automated alerting for critical health issues,
escalation policies, and integration with project management tools.
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from health_models import HealthReport, HealthIssue, Severity, HealthCategory
from health_notifier import HealthNotifier


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class AlertRule:
    """Configuration for an alert rule"""
    name: str
    condition: str  # Python expression to evaluate
    alert_level: AlertLevel
    channels: List[str]  # notification channels
    cooldown_minutes: int = 30  # minimum time between alerts
    escalation_minutes: int = 60  # time before escalation
    max_alerts_per_hour: int = 10  # rate limiting
    enabled: bool = True


@dataclass
class EscalationPolicy:
    """Escalation policy configuration"""
    name: str
    levels: List[Dict[str, Any]]  # escalation levels with delays and channels
    max_escalations: int = 3
    reset_after_minutes: int = 240  # reset escalation after 4 hours


@dataclass
class AlertHistory:
    """Track alert history for rate limiting and escalation"""
    rule_name: str
    last_alert_time: datetime
    alert_count: int = 1
    escalation_level: int = 0
    acknowledged: bool = False
    resolved: bool = False


class AutomatedAlertingSystem:
    """Automated alerting system with escalation and rate limiting"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/alerting-config.yaml")
        self.alert_rules = self._load_alert_rules()
        self.escalation_policies = self._load_escalation_policies()
        self.alert_history: Dict[str, AlertHistory] = {}
        self.notifier = HealthNotifier()
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for alerting system"""
        logger = logging.getLogger("automated_alerting")
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        log_dir = Path("logs/alerting")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(
            log_dir / f"alerting_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _load_alert_rules(self) -> List[AlertRule]:
        """Load alert rules from configuration"""
        try:
            if self.config_path.exists():
                import yaml
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                rules = []
                for rule_data in config.get('alert_rules', []):
                    rule = AlertRule(
                        name=rule_data['name'],
                        condition=rule_data['condition'],
                        alert_level=AlertLevel(rule_data['alert_level']),
                        channels=rule_data['channels'],
                        cooldown_minutes=rule_data.get('cooldown_minutes', 30),
                        escalation_minutes=rule_data.get('escalation_minutes', 60),
                        max_alerts_per_hour=rule_data.get('max_alerts_per_hour', 10),
                        enabled=rule_data.get('enabled', True)
                    )
                    rules.append(rule)
                
                return rules
            else:
                # Create default alert rules
                return self._create_default_alert_rules()
                
        except Exception as e:
            self.logger.error(f"Failed to load alert rules: {e}")
            return self._create_default_alert_rules()
    
    def _create_default_alert_rules(self) -> List[AlertRule]:
        """Create default alert rules"""
        return [
            AlertRule(
                name="critical_health_score",
                condition="health_report.overall_score < 70",
                alert_level=AlertLevel.CRITICAL,
                channels=["email", "slack"],
                cooldown_minutes=15,
                escalation_minutes=30
            ),
            AlertRule(
                name="emergency_health_score",
                condition="health_report.overall_score < 50",
                alert_level=AlertLevel.EMERGENCY,
                channels=["email", "slack", "sms"],
                cooldown_minutes=5,
                escalation_minutes=15
            ),
            AlertRule(
                name="test_failures",
                condition="any(issue.category == HealthCategory.TESTS and issue.severity == Severity.CRITICAL for issue in health_report.issues)",
                alert_level=AlertLevel.WARNING,
                channels=["slack"],
                cooldown_minutes=30
            ),
            AlertRule(
                name="security_issues",
                condition="any(issue.category == HealthCategory.SECURITY and issue.severity in [Severity.CRITICAL, Severity.HIGH] for issue in health_report.issues)",
                alert_level=AlertLevel.CRITICAL,
                channels=["email", "slack"],
                cooldown_minutes=10
            ),
            AlertRule(
                name="performance_degradation",
                condition="any(issue.category == HealthCategory.PERFORMANCE and issue.severity == Severity.HIGH for issue in health_report.issues)",
                alert_level=AlertLevel.WARNING,
                channels=["slack"],
                cooldown_minutes=60
            )
        ]
    
    def _load_escalation_policies(self) -> List[EscalationPolicy]:
        """Load escalation policies from configuration"""
        try:
            if self.config_path.exists():
                import yaml
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                policies = []
                for policy_data in config.get('escalation_policies', []):
                    policy = EscalationPolicy(
                        name=policy_data['name'],
                        levels=policy_data['levels'],
                        max_escalations=policy_data.get('max_escalations', 3),
                        reset_after_minutes=policy_data.get('reset_after_minutes', 240)
                    )
                    policies.append(policy)
                
                return policies
            else:
                return self._create_default_escalation_policies()
                
        except Exception as e:
            self.logger.error(f"Failed to load escalation policies: {e}")
            return self._create_default_escalation_policies()
    
    def _create_default_escalation_policies(self) -> List[EscalationPolicy]:
        """Create default escalation policies"""
        return [
            EscalationPolicy(
                name="standard_escalation",
                levels=[
                    {
                        "delay_minutes": 0,
                        "channels": ["slack"],
                        "message": "Initial alert"
                    },
                    {
                        "delay_minutes": 30,
                        "channels": ["email", "slack"],
                        "message": "Escalated: Issue not resolved"
                    },
                    {
                        "delay_minutes": 60,
                        "channels": ["email", "slack", "sms"],
                        "message": "Critical escalation: Immediate attention required"
                    }
                ]
            ),
            EscalationPolicy(
                name="emergency_escalation",
                levels=[
                    {
                        "delay_minutes": 0,
                        "channels": ["email", "slack", "sms"],
                        "message": "EMERGENCY: Critical system issue"
                    },
                    {
                        "delay_minutes": 15,
                        "channels": ["email", "slack", "sms", "phone"],
                        "message": "EMERGENCY ESCALATION: System down"
                    }
                ]
            )
        ]
    
    async def evaluate_health_report(self, health_report: HealthReport) -> None:
        """Evaluate health report against alert rules"""
        self.logger.info("Evaluating health report for alerts")
        
        # Create evaluation context
        context = {
            'health_report': health_report,
            'HealthCategory': HealthCategory,
            'Severity': Severity,
            'datetime': datetime,
            'any': any,
            'all': all,
            'len': len
        }
        
        # Evaluate each alert rule
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
                
            try:
                # Evaluate the condition
                if eval(rule.condition, context):
                    await self._trigger_alert(rule, health_report)
                    
            except Exception as e:
                self.logger.error(f"Failed to evaluate alert rule '{rule.name}': {e}")
    
    async def _trigger_alert(self, rule: AlertRule, health_report: HealthReport) -> None:
        """Trigger an alert if conditions are met"""
        now = datetime.now()
        
        # Check if we should trigger this alert (rate limiting)
        if not self._should_trigger_alert(rule, now):
            return
        
        # Update alert history
        if rule.name in self.alert_history:
            history = self.alert_history[rule.name]
            history.last_alert_time = now
            history.alert_count += 1
        else:
            history = AlertHistory(
                rule_name=rule.name,
                last_alert_time=now,
                alert_count=1
            )
            self.alert_history[rule.name] = history
        
        # Create alert message
        alert_message = self._create_alert_message(rule, health_report, history)
        
        # Send notifications
        await self._send_alert_notifications(rule, alert_message, health_report)
        
        # Schedule escalation if needed
        if rule.escalation_minutes > 0:
            asyncio.create_task(
                self._schedule_escalation(rule, health_report, history)
            )
        
        self.logger.warning(f"Alert triggered: {rule.name} (level: {rule.alert_level.value})")
    
    def _should_trigger_alert(self, rule: AlertRule, now: datetime) -> bool:
        """Check if alert should be triggered based on rate limiting"""
        if rule.name not in self.alert_history:
            return True
        
        history = self.alert_history[rule.name]
        
        # Check cooldown period
        cooldown_delta = timedelta(minutes=rule.cooldown_minutes)
        if now - history.last_alert_time < cooldown_delta:
            return False
        
        # Check rate limiting (alerts per hour)
        hour_ago = now - timedelta(hours=1)
        if history.last_alert_time > hour_ago and history.alert_count >= rule.max_alerts_per_hour:
            return False
        
        # Reset alert count if more than an hour has passed
        if history.last_alert_time <= hour_ago:
            history.alert_count = 0
        
        return True
    
    def _create_alert_message(self, rule: AlertRule, health_report: HealthReport, history: AlertHistory) -> Dict[str, Any]:
        """Create alert message with relevant information"""
        return {
            'rule_name': rule.name,
            'alert_level': rule.alert_level.value,
            'timestamp': datetime.now().isoformat(),
            'health_score': health_report.overall_score,
            'alert_count': history.alert_count,
            'escalation_level': history.escalation_level,
            'critical_issues': [
                {
                    'severity': issue.severity.value,
                    'category': issue.category.value,
                    'description': issue.description
                }
                for issue in health_report.get_critical_issues()
            ],
            'top_recommendations': [
                {
                    'priority': rec.priority,
                    'description': rec.description
                }
                for rec in health_report.recommendations[:3]
            ]
        }
    
    async def _send_alert_notifications(self, rule: AlertRule, alert_message: Dict[str, Any], health_report: HealthReport) -> None:
        """Send alert notifications through configured channels"""
        for channel in rule.channels:
            try:
                if channel == "email":
                    await self.notifier.send_email_alert(alert_message, health_report)
                elif channel == "slack":
                    await self.notifier.send_slack_alert(alert_message, health_report)
                elif channel == "sms":
                    await self.notifier.send_sms_alert(alert_message)
                elif channel == "webhook":
                    await self.notifier.send_webhook_alert(alert_message)
                else:
                    self.logger.warning(f"Unknown notification channel: {channel}")
                    
            except Exception as e:
                self.logger.error(f"Failed to send alert via {channel}: {e}")
    
    async def _schedule_escalation(self, rule: AlertRule, health_report: HealthReport, history: AlertHistory) -> None:
        """Schedule alert escalation"""
        await asyncio.sleep(rule.escalation_minutes * 60)
        
        # Check if alert has been acknowledged or resolved
        if history.acknowledged or history.resolved:
            return
        
        # Find appropriate escalation policy
        escalation_policy = self._get_escalation_policy(rule.alert_level)
        if not escalation_policy:
            return
        
        # Escalate if within limits
        if history.escalation_level < escalation_policy.max_escalations:
            history.escalation_level += 1
            
            # Get escalation level configuration
            if history.escalation_level <= len(escalation_policy.levels):
                level_config = escalation_policy.levels[history.escalation_level - 1]
                
                # Create escalated alert message
                escalated_message = self._create_alert_message(rule, health_report, history)
                escalated_message['escalated'] = True
                escalated_message['escalation_message'] = level_config.get('message', 'Escalated alert')
                
                # Send escalated notifications
                for channel in level_config.get('channels', []):
                    try:
                        if channel == "email":
                            await self.notifier.send_email_alert(escalated_message, health_report)
                        elif channel == "slack":
                            await self.notifier.send_slack_alert(escalated_message, health_report)
                        elif channel == "sms":
                            await self.notifier.send_sms_alert(escalated_message)
                        elif channel == "phone":
                            await self.notifier.send_phone_alert(escalated_message)
                    except Exception as e:
                        self.logger.error(f"Failed to send escalated alert via {channel}: {e}")
                
                self.logger.critical(f"Alert escalated: {rule.name} (level: {history.escalation_level})")
                
                # Schedule next escalation if available
                if history.escalation_level < escalation_policy.max_escalations:
                    asyncio.create_task(
                        self._schedule_escalation(rule, health_report, history)
                    )
    
    def _get_escalation_policy(self, alert_level: AlertLevel) -> Optional[EscalationPolicy]:
        """Get appropriate escalation policy for alert level"""
        if alert_level == AlertLevel.EMERGENCY:
            return next((p for p in self.escalation_policies if p.name == "emergency_escalation"), None)
        else:
            return next((p for p in self.escalation_policies if p.name == "standard_escalation"), None)
    
    def acknowledge_alert(self, rule_name: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert to stop escalation"""
        if rule_name in self.alert_history:
            history = self.alert_history[rule_name]
            history.acknowledged = True
            
            self.logger.info(f"Alert acknowledged: {rule_name} by {acknowledged_by}")
            
            # Log acknowledgment
            ack_log = {
                'rule_name': rule_name,
                'acknowledged_by': acknowledged_by,
                'acknowledged_at': datetime.now().isoformat(),
                'escalation_level': history.escalation_level
            }
            
            ack_file = Path("logs/alerting/acknowledgments.jsonl")
            with open(ack_file, 'a') as f:
                f.write(json.dumps(ack_log) + '\n')
            
            return True
        
        return False
    
    def resolve_alert(self, rule_name: str, resolved_by: str, resolution_notes: str = "") -> bool:
        """Mark an alert as resolved"""
        if rule_name in self.alert_history:
            history = self.alert_history[rule_name]
            history.resolved = True
            
            self.logger.info(f"Alert resolved: {rule_name} by {resolved_by}")
            
            # Log resolution
            resolution_log = {
                'rule_name': rule_name,
                'resolved_by': resolved_by,
                'resolved_at': datetime.now().isoformat(),
                'resolution_notes': resolution_notes,
                'total_escalations': history.escalation_level
            }
            
            resolution_file = Path("logs/alerting/resolutions.jsonl")
            with open(resolution_file, 'a') as f:
                f.write(json.dumps(resolution_log) + '\n')
            
            return True
        
        return False
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active (unresolved) alerts"""
        active_alerts = []
        
        for rule_name, history in self.alert_history.items():
            if not history.resolved:
                active_alerts.append({
                    'rule_name': rule_name,
                    'last_alert_time': history.last_alert_time.isoformat(),
                    'alert_count': history.alert_count,
                    'escalation_level': history.escalation_level,
                    'acknowledged': history.acknowledged
                })
        
        return active_alerts
    
    def cleanup_old_alerts(self, days: int = 7) -> None:
        """Clean up old resolved alerts"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        to_remove = []
        for rule_name, history in self.alert_history.items():
            if history.resolved and history.last_alert_time < cutoff_date:
                to_remove.append(rule_name)
        
        for rule_name in to_remove:
            del self.alert_history[rule_name]
        
        if to_remove:
            self.logger.info(f"Cleaned up {len(to_remove)} old alerts")
    
    def save_configuration(self) -> None:
        """Save current alert rules and escalation policies to configuration file"""
        try:
            config = {
                'alert_rules': [
                    {
                        'name': rule.name,
                        'condition': rule.condition,
                        'alert_level': rule.alert_level.value,
                        'channels': rule.channels,
                        'cooldown_minutes': rule.cooldown_minutes,
                        'escalation_minutes': rule.escalation_minutes,
                        'max_alerts_per_hour': rule.max_alerts_per_hour,
                        'enabled': rule.enabled
                    }
                    for rule in self.alert_rules
                ],
                'escalation_policies': [
                    {
                        'name': policy.name,
                        'levels': policy.levels,
                        'max_escalations': policy.max_escalations,
                        'reset_after_minutes': policy.reset_after_minutes
                    }
                    for policy in self.escalation_policies
                ]
            }
            
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            import yaml
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            self.logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")


# Integration functions for project management tools

async def create_jira_issue(alert_message: Dict[str, Any], jira_config: Dict[str, Any]) -> Optional[str]:
    """Create a Jira issue for critical alerts"""
    try:
        # This would integrate with Jira API
        # For now, return a mock issue ID
        issue_id = f"WAN22-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        print(f"Created Jira issue: {issue_id} for alert: {alert_message['rule_name']}")
        return issue_id
        
    except Exception as e:
        print(f"Failed to create Jira issue: {e}")
        return None


async def update_github_status(alert_message: Dict[str, Any], github_config: Dict[str, Any]) -> bool:
    """Update GitHub commit status based on health alerts"""
    try:
        # This would integrate with GitHub API
        # For now, just log the action
        
        status = "failure" if alert_message['alert_level'] in ['critical', 'emergency'] else "pending"
        
        print(f"Updated GitHub status to {status} for alert: {alert_message['rule_name']}")
        return True
        
    except Exception as e:
        print(f"Failed to update GitHub status: {e}")
        return False


async def send_teams_notification(alert_message: Dict[str, Any], teams_config: Dict[str, Any]) -> bool:
    """Send notification to Microsoft Teams"""
    try:
        # This would integrate with Teams webhook
        # For now, just log the action
        
        print(f"Sent Teams notification for alert: {alert_message['rule_name']}")
        return True
        
    except Exception as e:
        print(f"Failed to send Teams notification: {e}")
        return False


# CLI interface for alert management

def create_alerting_cli():
    """Create CLI interface for managing alerts"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage automated alerting system")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List active alerts
    list_parser = subparsers.add_parser('list', help='List active alerts')
    
    # Acknowledge alert
    ack_parser = subparsers.add_parser('acknowledge', help='Acknowledge an alert')
    ack_parser.add_argument('rule_name', help='Name of the alert rule')
    ack_parser.add_argument('--by', required=True, help='Person acknowledging the alert')
    
    # Resolve alert
    resolve_parser = subparsers.add_parser('resolve', help='Resolve an alert')
    resolve_parser.add_argument('rule_name', help='Name of the alert rule')
    resolve_parser.add_argument('--by', required=True, help='Person resolving the alert')
    resolve_parser.add_argument('--notes', help='Resolution notes')
    
    # Test alert
    test_parser = subparsers.add_parser('test', help='Test alert system')
    test_parser.add_argument('rule_name', help='Name of the alert rule to test')
    
    return parser


async def main():
    """Main function for CLI usage"""
    parser = create_alerting_cli()
    args = parser.parse_args()
    
    alerting_system = AutomatedAlertingSystem()
    
    if args.command == 'list':
        active_alerts = alerting_system.get_active_alerts()
        if active_alerts:
            print("Active alerts:")
            for alert in active_alerts:
                print(f"  - {alert['rule_name']} (escalation level: {alert['escalation_level']})")
        else:
            print("No active alerts")
    
    elif args.command == 'acknowledge':
        success = alerting_system.acknowledge_alert(args.rule_name, args.by)
        if success:
            print(f"Alert '{args.rule_name}' acknowledged by {args.by}")
        else:
            print(f"Alert '{args.rule_name}' not found")
    
    elif args.command == 'resolve':
        success = alerting_system.resolve_alert(args.rule_name, args.by, args.notes or "")
        if success:
            print(f"Alert '{args.rule_name}' resolved by {args.by}")
        else:
            print(f"Alert '{args.rule_name}' not found")
    
    elif args.command == 'test':
        # Create a mock health report for testing
        from health_models import HealthReport, HealthTrends
        
        test_report = HealthReport(
            timestamp=datetime.now(),
            overall_score=60.0,  # Low score to trigger alerts
            component_scores={},
            issues=[],
            recommendations=[],
            trends=HealthTrends()
        )
        
        await alerting_system.evaluate_health_report(test_report)
        print(f"Test alert sent for rule: {args.rule_name}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())