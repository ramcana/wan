#!/usr/bin/env python3
"""
Automated health monitoring and alerting system.

This module provides automated health monitoring with trend tracking,
alerting, and continuous improvement recommendations.
"""

import asyncio
import json
import logging
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

from .health_checker import ProjectHealthChecker
from .establish_baseline import BaselineEstablisher, ContinuousImprovementTracker
from .health_models import HealthReport, Severity
from .health_notifier import HealthNotifier


class AutomatedHealthMonitor:
    """
    Automated health monitoring system with continuous improvement tracking.
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path("tools/health-checker/monitoring_config.json")
        self.config = self._load_config()
        
        # Initialize components
        self.health_checker = ProjectHealthChecker()
        self.baseline_establisher = BaselineEstablisher()
        self.improvement_tracker = ContinuousImprovementTracker(self.baseline_establisher)
        self.notifier = HealthNotifier(self.config.get("notifications", {}))
        
        # Monitoring state
        self.monitoring_active = False
        self.last_check_time = None
        self.alert_history = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _load_config(self) -> Dict:
        """Load monitoring configuration."""
        default_config = {
            "monitoring": {
                "enabled": True,
                "check_interval_minutes": 60,
                "baseline_update_interval_hours": 24,
                "trend_analysis_days": 7,
                "max_alert_frequency_minutes": 30
            },
            "thresholds": {
                "critical_score": 50,
                "warning_score": 70,
                "execution_time_warning": 300,
                "execution_time_critical": 600
            },
            "notifications": {
                "enabled": True,
                "channels": ["console", "file"],
                "email": {
                    "enabled": False,
                    "smtp_server": "",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "recipients": []
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": ""
                }
            },
            "improvement_tracking": {
                "enabled": True,
                "auto_create_initiatives": True,
                "target_improvement_percent": 10
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    default_config.update(user_config)
            except Exception as e:
                print(f"⚠️ Error loading config, using defaults: {e}")
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging for monitoring."""
        log_file = Path("tools/health-checker/monitoring.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def start_monitoring(self):
        """Start automated health monitoring."""
        if not self.config["monitoring"]["enabled"]:
            self.logger.info("Monitoring is disabled in configuration")
            return
        
        self.monitoring_active = True
        self.logger.info("Starting automated health monitoring")
        
        # Schedule regular health checks
        interval = self.config["monitoring"]["check_interval_minutes"]
        schedule.every(interval).minutes.do(self._run_scheduled_check)
        
        # Schedule baseline updates
        baseline_interval = self.config["monitoring"]["baseline_update_interval_hours"]
        schedule.every(baseline_interval).hours.do(self._update_baseline)
        
        # Schedule trend analysis
        schedule.every().day.at("09:00").do(self._analyze_trends)
        
        # Run initial check
        self._run_scheduled_check()
        
        # Start monitoring loop
        try:
            while self.monitoring_active:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        finally:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop automated health monitoring."""
        self.monitoring_active = False
        schedule.clear()
        self.logger.info("Automated health monitoring stopped")
    
    def _run_scheduled_check(self):
        """Run scheduled health check."""
        try:
            self.logger.info("Running scheduled health check")
            
            # Run health check
            report = asyncio.run(self.health_checker.run_optimized_health_check(
                lightweight=True,  # Use lightweight for frequent checks
                use_cache=True,
                parallel=True
            ))
            
            # Update baseline with new data
            self.baseline_establisher.update_baseline_with_new_data(report)
            
            # Check against baseline and generate alerts
            baseline_check = self.baseline_establisher.check_against_baseline(report)
            
            # Process alerts
            self._process_alerts(report, baseline_check)
            
            # Update improvement tracking
            self._update_improvement_tracking(report)
            
            self.last_check_time = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error in scheduled health check: {e}")
    
    def _process_alerts(self, report: HealthReport, baseline_check: Dict):
        """Process and send alerts based on health check results."""
        
        alerts = baseline_check.get("alerts", [])
        
        if not alerts:
            return
        
        # Filter alerts based on frequency limits
        filtered_alerts = self._filter_alert_frequency(alerts)
        
        if not filtered_alerts:
            return
        
        # Send notifications
        for alert in filtered_alerts:
            self._send_alert_notification(alert, report)
            
            # Add to alert history
            self.alert_history.append({
                "timestamp": datetime.now().isoformat(),
                "alert": alert,
                "report_id": id(report)
            })
        
        # Keep only recent alert history
        cutoff_time = datetime.now() - timedelta(days=7)
        self.alert_history = [
            entry for entry in self.alert_history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]
    
    def _filter_alert_frequency(self, alerts: List[Dict]) -> List[Dict]:
        """Filter alerts based on frequency limits to avoid spam."""
        
        max_frequency_minutes = self.config["monitoring"]["max_alert_frequency_minutes"]
        cutoff_time = datetime.now() - timedelta(minutes=max_frequency_minutes)
        
        # Get recent alerts of same type
        recent_alert_types = set()
        for entry in self.alert_history:
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time:
                recent_alert_types.add(entry["alert"]["metric"])
        
        # Filter out alerts that were recently sent
        filtered_alerts = [
            alert for alert in alerts
            if alert["metric"] not in recent_alert_types
        ]
        
        return filtered_alerts
    
    def _send_alert_notification(self, alert: Dict, report: HealthReport):
        """Send alert notification through configured channels."""
        
        message = self._format_alert_message(alert, report)
        
        # Send through notifier
        if alert["type"] == "critical":
            self.notifier.send_critical_alert(
                title=f"Critical Health Alert: {alert['metric']}",
                message=message,
                details=alert
            )
        else:
            self.notifier.send_warning_alert(
                title=f"Health Warning: {alert['metric']}",
                message=message,
                details=alert
            )
    
    def _format_alert_message(self, alert: Dict, report: HealthReport) -> str:
        """Format alert message for notifications."""
        
        message = f"""
Health Alert: {alert['type'].upper()}

Metric: {alert['metric']}
Current Value: {alert['value']}
Threshold: {alert['threshold']}
Message: {alert['message']}

Overall Health Score: {report.overall_score:.1f}
Check Time: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Recommendations:
- Review recent changes that might have impacted health
- Check system logs for errors
- Run detailed health analysis for more information
        """.strip()
        
        return message
    
    def _update_baseline(self):
        """Update baseline metrics with recent data."""
        try:
            self.logger.info("Updating baseline metrics")
            
            # Re-establish baseline with recent data
            self.baseline_establisher.establish_comprehensive_baseline(num_runs=3)
            
            self.logger.info("Baseline metrics updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating baseline: {e}")
    
    def _analyze_trends(self):
        """Analyze health trends and generate improvement recommendations."""
        try:
            self.logger.info("Analyzing health trends")
            
            # Analyze trends
            trend_days = self.config["monitoring"]["trend_analysis_days"]
            trends = self.improvement_tracker.analyze_trends(days=trend_days)
            
            # Generate improvement report
            improvement_report = self.improvement_tracker.generate_improvement_report()
            
            # Check if new improvement initiatives should be created
            if self.config["improvement_tracking"]["auto_create_initiatives"]:
                self._auto_create_improvement_initiatives(trends, improvement_report)
            
            # Send trend report
            self._send_trend_report(trends, improvement_report)
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends: {e}")
    
    def _auto_create_improvement_initiatives(self, trends: Dict, improvement_report: Dict):
        """Automatically create improvement initiatives based on trends."""
        
        # Check for declining trends that need attention
        declining_metrics = []
        
        overall_trend = trends.get("overall_score", {})
        if overall_trend.get("trend") == "declining":
            declining_metrics.append({
                "metric": "overall_score",
                "trend": overall_trend,
                "priority": "high"
            })
        
        # Check component trends
        for component, trend in trends.get("component_scores", {}).items():
            if trend.get("trend") == "declining":
                declining_metrics.append({
                    "metric": f"component_{component}",
                    "trend": trend,
                    "priority": "medium"
                })
        
        # Create initiatives for declining metrics
        for metric_info in declining_metrics:
            initiative_name = f"Improve {metric_info['metric']}"
            description = f"Address declining trend in {metric_info['metric']} (change: {metric_info['trend'].get('change_percent', 0):.1f}%)"
            
            target_improvement = self.config["improvement_tracking"]["target_improvement_percent"]
            current_value = metric_info['trend'].get('end_value', 0)
            target_value = current_value * (1 + target_improvement / 100)
            
            target_metrics = {
                metric_info['metric']: {
                    "current": current_value,
                    "target": target_value,
                    "improvement_percent": target_improvement
                }
            }
            
            # Check if similar initiative already exists
            existing_initiatives = improvement_report.get("active_initiatives", [])
            similar_exists = any(
                metric_info['metric'] in init['name'].lower()
                for init in existing_initiatives
            )
            
            if not similar_exists:
                initiative_id = self.improvement_tracker.track_improvement_initiative(
                    name=initiative_name,
                    description=description,
                    target_metrics=target_metrics,
                    timeline="30 days"
                )
                
                self.logger.info(f"Auto-created improvement initiative: {initiative_name} (ID: {initiative_id})")
    
    def _send_trend_report(self, trends: Dict, improvement_report: Dict):
        """Send trend analysis report."""
        
        # Format trend report message
        message = self._format_trend_report(trends, improvement_report)
        
        # Send through notifier
        self.notifier.send_info_notification(
            title="Weekly Health Trend Report",
            message=message,
            details={
                "trends": trends,
                "improvement_report": improvement_report
            }
        )
    
    def _format_trend_report(self, trends: Dict, improvement_report: Dict) -> str:
        """Format trend report message."""
        
        overall_trend = trends.get("overall_score", {})
        trend_direction = overall_trend.get("trend", "unknown")
        trend_change = overall_trend.get("change_percent", 0)
        
        active_initiatives = len(improvement_report.get("active_initiatives", []))
        completed_initiatives = improvement_report["summary"].get("completed_initiatives", 0)
        
        message = f"""
Weekly Health Trend Report

Overall Health Trend: {trend_direction.upper()} ({trend_change:+.1f}%)

Improvement Initiatives:
- Active: {active_initiatives}
- Completed: {completed_initiatives}

Key Trends:
        """.strip()
        
        # Add component trends
        for component, trend in trends.get("component_scores", {}).items():
            direction = trend.get("trend", "unknown")
            change = trend.get("change_percent", 0)
            message += f"\n- {component}: {direction} ({change:+.1f}%)"
        
        # Add recommendations
        recommendations = improvement_report.get("recommendations", [])
        if recommendations:
            message += "\n\nRecommendations:"
            for rec in recommendations[:3]:  # Top 3 recommendations
                message += f"\n- {rec}"
        
        return message
    
    def _update_improvement_tracking(self, report: HealthReport):
        """Update improvement tracking with current health data."""
        
        # Check progress on active initiatives
        improvement_report = self.improvement_tracker.generate_improvement_report()
        
        for initiative in improvement_report.get("active_initiatives", []):
            # Check if initiative targets are met
            targets_met = self._check_initiative_targets(initiative, report)
            
            if targets_met:
                progress_update = {
                    "targets_met": True,
                    "metrics": {
                        "overall_score": report.overall_score,
                        "component_scores": report.component_scores
                    },
                    "completion_note": "Targets achieved through automated monitoring"
                }
                
                self.improvement_tracker.update_initiative_progress(
                    initiative["id"],
                    progress_update
                )
    
    def _check_initiative_targets(self, initiative: Dict, report: HealthReport) -> bool:
        """Check if initiative targets are met."""
        
        target_metrics = initiative.get("target_metrics", {})
        
        for metric_name, target_info in target_metrics.items():
            target_value = target_info.get("target", 0)
            
            if metric_name == "overall_score":
                current_value = report.overall_score
            elif metric_name.startswith("component_"):
                component_name = metric_name.replace("component_", "")
                current_value = report.component_scores.get(component_name, 0)
            else:
                continue
            
            # Check if target is met (assuming higher is better for scores)
            if current_value < target_value:
                return False
        
        return True
    
    def get_monitoring_status(self) -> Dict:
        """Get current monitoring status."""
        
        return {
            "monitoring_active": self.monitoring_active,
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "recent_alerts": len([
                entry for entry in self.alert_history
                if datetime.fromisoformat(entry["timestamp"]) > datetime.now() - timedelta(hours=24)
            ]),
            "config": self.config,
            "next_scheduled_check": self._get_next_scheduled_check()
        }
    
    def _get_next_scheduled_check(self) -> Optional[str]:
        """Get next scheduled check time."""
        
        if not schedule.jobs:
            return None
        
        next_run = min(job.next_run for job in schedule.jobs)
        return next_run.isoformat() if next_run else None
    
    def run_manual_check(self) -> Dict:
        """Run manual health check and return results."""
        
        try:
            # Run comprehensive health check
            report = asyncio.run(self.health_checker.run_optimized_health_check(
                lightweight=False,
                use_cache=False,
                parallel=True
            ))
            
            # Check against baseline
            baseline_check = self.baseline_establisher.check_against_baseline(report)
            
            # Generate improvement recommendations
            improvement_report = self.improvement_tracker.generate_improvement_report()
            
            return {
                "success": True,
                "report": {
                    "timestamp": report.timestamp.isoformat(),
                    "overall_score": report.overall_score,
                    "component_scores": report.component_scores,
                    "issues_count": len(report.issues),
                    "execution_time": report.execution_time
                },
                "baseline_check": baseline_check,
                "improvement_report": improvement_report
            }
            
        except Exception as e:
            self.logger.error(f"Error in manual health check: {e}")
            return {
                "success": False,
                "error": str(e)
            }


def main():
    """Main function for running automated monitoring."""
    
    monitor = AutomatedHealthMonitor()
    
    try:
        print("🔍 Starting automated health monitoring...")
        print("Press Ctrl+C to stop monitoring")
        
        monitor.start_monitoring()
        
    except KeyboardInterrupt:
        print("\n⏹️ Stopping monitoring...")
        monitor.stop_monitoring()
        print("✅ Monitoring stopped")


if __name__ == "__main__":
    main()