#!/usr/bin/env python3
"""
Simplified Production Health Monitoring Deployment System

This module handles the deployment and configuration of health monitoring
for production environments with standard library dependencies only.
"""

import os
import json
import yaml
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from health_checker import ProjectHealthChecker
from health_reporter import HealthReporter
from health_notifier import HealthNotifier
from health_models import HealthReport


@dataclass
class ProductionHealthConfig:
    """Production-specific health monitoring configuration"""
    
    # Monitoring intervals
    daily_report_time: str = "08:00"  # UTC time for daily reports
    weekly_report_day: str = "monday"  # Day for weekly reports
    critical_check_interval: int = 15  # Minutes between critical checks
    
    # Thresholds (more conservative for production)
    min_health_score: float = 85.0  # Minimum acceptable health score
    critical_health_score: float = 70.0  # Score that triggers immediate alerts
    test_pass_rate_threshold: float = 95.0  # Minimum test pass rate
    coverage_threshold: float = 80.0  # Minimum code coverage
    
    # Notification settings
    enable_email_notifications: bool = True
    enable_slack_notifications: bool = True
    enable_dashboard_updates: bool = True
    
    # Production-specific checks
    enable_performance_monitoring: bool = True
    enable_security_scanning: bool = True
    enable_dependency_auditing: bool = True
    
    # Resource limits
    max_check_duration: int = 300  # Maximum time for health checks (seconds)
    max_concurrent_checks: int = 3  # Maximum concurrent health checks
    
    # Reporting
    report_retention_days: int = 90  # How long to keep health reports
    detailed_reporting: bool = True  # Include detailed metrics in reports


class ProductionHealthMonitor:
    """Production health monitoring system with automated reporting and alerting"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/production-health.yaml")
        self.config = self._load_config()
        self.health_checker = ProjectHealthChecker()
        self.reporter = HealthReporter()
        self.notifier = HealthNotifier()
        self.logger = self._setup_logging()
        
    def _load_config(self) -> ProductionHealthConfig:
        """Load production health monitoring configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                return ProductionHealthConfig(**config_data)
            else:
                # Create default config if none exists
                default_config = ProductionHealthConfig()
                self._save_config(default_config)
                return default_config
        except Exception as e:
            print(f"Failed to load production config: {e}")
            return ProductionHealthConfig()
    
    def _save_config(self, config: ProductionHealthConfig) -> None:
        """Save production health monitoring configuration"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(asdict(config), f, default_flow_style=False)
        except Exception as e:
            print(f"Failed to save production config: {e}")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up production logging with appropriate levels and handlers"""
        logger = logging.getLogger("production_health_monitor")
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs/health-monitoring")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler for production logs
        file_handler = logging.FileHandler(
            log_dir / f"health_monitor_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.INFO)
        
        # Console handler for immediate feedback
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
    
    async def run_production_health_check(self) -> HealthReport:
        """Run comprehensive health check with production-specific validations"""
        self.logger.info("Starting production health check")
        start_time = datetime.now()
        
        try:
            # Run health check with timeout
            health_report = await asyncio.wait_for(
                self.health_checker.run_health_check(),
                timeout=self.config.max_check_duration
            )
            
            # Add production-specific validations
            await self._validate_production_requirements(health_report)
            
            # Log results
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Production health check completed in {duration:.2f}s. "
                f"Overall score: {health_report.overall_score:.1f}"
            )
            
            # Check if immediate action is needed
            if health_report.overall_score < self.config.critical_health_score:
                await self._handle_critical_health_issue(health_report)
            
            return health_report
            
        except asyncio.TimeoutError:
            self.logger.error(f"Health check timed out after {self.config.max_check_duration}s")
            raise
        except Exception as e:
            self.logger.error(f"Production health check failed: {e}")
            raise
    
    async def _validate_production_requirements(self, report: HealthReport) -> None:
        """Validate production-specific requirements"""
        issues = []
        
        # Check overall health score
        if report.overall_score < self.config.min_health_score:
            issues.append(
                f"Overall health score ({report.overall_score:.1f}) below "
                f"production threshold ({self.config.min_health_score})"
            )
        
        # Check test pass rate
        test_component = report.component_scores.get('tests')
        if test_component and test_component.score < self.config.test_pass_rate_threshold:
            issues.append(
                f"Test pass rate ({test_component.score:.1f}%) below "
                f"production threshold ({self.config.test_pass_rate_threshold}%)"
            )
        
        # Check code coverage
        coverage_component = report.component_scores.get('coverage')
        if coverage_component and coverage_component.score < self.config.coverage_threshold:
            issues.append(
                f"Code coverage ({coverage_component.score:.1f}%) below "
                f"production threshold ({self.config.coverage_threshold}%)"
            )
        
        if issues:
            self.logger.warning(f"Production validation issues: {'; '.join(issues)}")
    
    async def _handle_critical_health_issue(self, report: HealthReport) -> None:
        """Handle critical health issues that require immediate attention"""
        self.logger.critical(
            f"Critical health issue detected! Score: {report.overall_score:.1f}"
        )
        
        # Create incident report
        await self._create_incident_report(report)
    
    async def _create_incident_report(self, report: HealthReport) -> None:
        """Create detailed incident report for critical health issues"""
        incident_dir = Path("logs/incidents")
        incident_dir.mkdir(parents=True, exist_ok=True)
        
        incident_file = incident_dir / f"incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        incident_data = {
            'timestamp': datetime.now().isoformat(),
            'severity': 'critical',
            'health_score': report.overall_score,
            'component_scores': {k: v.score for k, v in report.component_scores.items()},
            'critical_issues': [
                {
                    'severity': issue.severity.value,
                    'category': issue.category.value,
                    'description': issue.description
                }
                for issue in report.get_critical_issues()
            ],
            'recommendations': [
                {
                    'priority': rec.priority,
                    'category': rec.category.value,
                    'description': rec.description
                }
                for rec in report.recommendations[:5]  # Top 5 recommendations
            ],
            'system_info': await self._collect_system_info()
        }
        
        with open(incident_file, 'w') as f:
            json.dump(incident_data, f, indent=2)
        
        self.logger.info(f"Incident report created: {incident_file}")
    
    async def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for incident reports"""
        try:
            import psutil
            import platform
            
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_usage': psutil.cpu_percent(interval=1),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'timestamp': datetime.now().isoformat()
            }
        except ImportError:
            return {
                'platform': 'unknown',
                'python_version': 'unknown',
                'timestamp': datetime.now().isoformat()
            }
    
    async def generate_daily_report(self) -> None:
        """Generate and distribute daily health report"""
        try:
            self.logger.info("Generating daily health report")
            
            # Run health check
            health_report = await self.run_production_health_check()
            
            # Generate report
            report_content = await self.reporter.generate_daily_report(health_report)
            
            # Save report
            report_dir = Path("reports/daily")
            report_dir.mkdir(parents=True, exist_ok=True)
            report_file = report_dir / f"health_report_{datetime.now().strftime('%Y%m%d')}.html"
            
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            self.logger.info(f"Daily health report generated: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate daily report: {e}")
    
    async def cleanup_old_reports(self) -> None:
        """Clean up old health reports based on retention policy"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.report_retention_days)
            
            # Clean up daily reports
            daily_reports = Path("reports/daily")
            if daily_reports.exists():
                for report_file in daily_reports.glob("*.html"):
                    if report_file.stat().st_mtime < cutoff_date.timestamp():
                        report_file.unlink()
                        self.logger.info(f"Cleaned up old report: {report_file}")
            
            # Clean up old logs
            log_dir = Path("logs/health-monitoring")
            if log_dir.exists():
                for log_file in log_dir.glob("*.log"):
                    if log_file.stat().st_mtime < cutoff_date.timestamp():
                        log_file.unlink()
                        self.logger.info(f"Cleaned up old log: {log_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old reports: {e}")


async def deploy_production_monitoring() -> None:
    """Deploy production health monitoring system"""
    print("Deploying production health monitoring system...")
    
    # Create production monitor
    monitor = ProductionHealthMonitor()
    
    # Run initial setup and validation
    try:
        # Test health check functionality
        print("Running initial health check...")
        initial_report = await monitor.run_production_health_check()
        print(f"Initial health score: {initial_report.overall_score:.1f}")
        
        # Generate initial reports
        print("Generating initial reports...")
        await monitor.generate_daily_report()
        
        print("Production health monitoring deployed successfully!")
        print(f"Configuration saved to: {monitor.config_path}")
        print("To start monitoring, run: python -m tools.health-checker.production_deployment_simple")
        
    except Exception as e:
        print(f"Failed to deploy production monitoring: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "deploy":
        # Deploy production monitoring
        asyncio.run(deploy_production_monitoring())
    else:
        # Start monitoring (simplified version)
        monitor = ProductionHealthMonitor()
        try:
            print("Starting production health monitoring...")
            asyncio.run(monitor.run_production_health_check())
            print("Health check completed. Check logs for details.")
        except KeyboardInterrupt:
            print("\nProduction health monitoring stopped")