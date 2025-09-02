#!/usr/bin/env python3
"""
Production Health Monitoring Deployment System

This module handles the deployment and configuration of health monitoring
for production environments with appropriate thresholds and reporting.
"""

import os
import json
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import asyncio
# import aiofiles  # Will use regular file operations for now
import time
from concurrent.futures import ThreadPoolExecutor

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
        self.health_checker = ProjectHealthChecker(self.config)
        self.reporter = HealthReporter()
        self.notifier = HealthNotifier()
        self.logger = self._setup_logging()
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_checks)
        
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
            self.logger.error(f"Failed to load production config: {e}")
            return ProductionHealthConfig()
    
    def _save_config(self, config: ProductionHealthConfig) -> None:
        """Save production health monitoring configuration"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(asdict(config), f, default_flow_style=False)
        except Exception as e:
            self.logger.error(f"Failed to save production config: {e}")
    
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
        test_score = report.component_scores.get('tests', 0)
        if test_score < self.config.test_pass_rate_threshold:
            issues.append(
                f"Test pass rate ({test_score:.1f}%) below "
                f"production threshold ({self.config.test_pass_rate_threshold}%)"
            )
        
        # Check code coverage
        coverage_score = report.component_scores.get('coverage', 0)
        if coverage_score < self.config.coverage_threshold:
            issues.append(
                f"Code coverage ({coverage_score:.1f}%) below "
                f"production threshold ({self.config.coverage_threshold}%)"
            )
        
        if issues:
            self.logger.warning(f"Production validation issues: {'; '.join(issues)}")
            # Add issues to the report
            for issue in issues:
                report.issues.append({
                    'severity': 'high',
                    'category': 'production_validation',
                    'description': issue,
                    'timestamp': datetime.now().isoformat()
                })
    
    async def _handle_critical_health_issue(self, report: HealthReport) -> None:
        """Handle critical health issues that require immediate attention"""
        self.logger.critical(
            f"Critical health issue detected! Score: {report.overall_score:.1f}"
        )
        
        # Send immediate notifications
        if self.config.enable_email_notifications:
            await self.notifier.send_critical_alert(report)
        
        if self.config.enable_slack_notifications:
            await self.notifier.send_slack_alert(report)
        
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
            'component_scores': report.component_scores,
            'critical_issues': [
                issue for issue in report.issues 
                if issue.get('severity') in ['critical', 'high']
            ],
            'recommendations': report.recommendations[:5],  # Top 5 recommendations
            'system_info': await self._collect_system_info()
        }
        
        async with aiofiles.open(incident_file, 'w') as f:
            await f.write(json.dumps(incident_data, indent=2))
        
        self.logger.info(f"Incident report created: {incident_file}")
    
    async def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for incident reports"""
        import psutil
        import platform
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None,
            'timestamp': datetime.now().isoformat()
        }
    
    def schedule_production_monitoring(self) -> None:
        """Schedule automated production health monitoring"""
        self.logger.info("Setting up production health monitoring schedule")
        
        # Daily health reports
        schedule.every().day.at(self.config.daily_report_time).do(
            self._run_daily_health_report
        )
        
        # Weekly comprehensive reports
        getattr(schedule.every(), self.config.weekly_report_day).at(
            self.config.daily_report_time
        ).do(self._run_weekly_health_report)
        
        # Critical health checks (every 15 minutes by default)
        schedule.every(self.config.critical_check_interval).minutes.do(
            self._run_critical_health_check
        )
        
        # Cleanup old reports
        schedule.every().day.at("02:00").do(self._cleanup_old_reports)
        
        self.logger.info("Production monitoring schedule configured")
    
    def _run_daily_health_report(self) -> None:
        """Run daily health report (synchronous wrapper)"""
        asyncio.run(self.generate_daily_report())
    
    def _run_weekly_health_report(self) -> None:
        """Run weekly health report (synchronous wrapper)"""
        asyncio.run(self.generate_weekly_report())
    
    def _run_critical_health_check(self) -> None:
        """Run critical health check (synchronous wrapper)"""
        asyncio.run(self.run_critical_health_check())
    
    def _cleanup_old_reports(self) -> None:
        """Clean up old health reports"""
        asyncio.run(self.cleanup_old_reports())
    
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
            
            async with aiofiles.open(report_file, 'w') as f:
                await f.write(report_content)
            
            # Send notifications if configured
            if self.config.enable_email_notifications:
                await self.notifier.send_daily_report(health_report, report_file)
            
            self.logger.info(f"Daily health report generated: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate daily report: {e}")
    
    async def generate_weekly_report(self) -> None:
        """Generate comprehensive weekly health report with trends"""
        try:
            self.logger.info("Generating weekly health report")
            
            # Collect health data from the past week
            weekly_data = await self._collect_weekly_health_data()
            
            # Generate comprehensive report
            report_content = await self.reporter.generate_weekly_report(weekly_data)
            
            # Save report
            report_dir = Path("reports/weekly")
            report_dir.mkdir(parents=True, exist_ok=True)
            report_file = report_dir / f"weekly_health_report_{datetime.now().strftime('%Y%m%d')}.html"
            
            async with aiofiles.open(report_file, 'w') as f:
                await f.write(report_content)
            
            # Send notifications
            if self.config.enable_email_notifications:
                await self.notifier.send_weekly_report(weekly_data, report_file)
            
            self.logger.info(f"Weekly health report generated: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate weekly report: {e}")
    
    async def run_critical_health_check(self) -> None:
        """Run lightweight critical health check"""
        try:
            # Run only critical health checks (faster subset)
            critical_report = await self.health_checker.run_critical_checks()
            
            # Check if immediate action is needed
            if critical_report.overall_score < self.config.critical_health_score:
                await self._handle_critical_health_issue(critical_report)
            
        except Exception as e:
            self.logger.error(f"Critical health check failed: {e}")
    
    async def _collect_weekly_health_data(self) -> Dict[str, Any]:
        """Collect health data from the past week for trend analysis"""
        # This would collect data from daily reports and logs
        # For now, return current health data
        current_report = await self.run_production_health_check()
        
        return {
            'current_report': current_report,
            'trend_data': {},  # Would be populated with historical data
            'week_start': (datetime.now() - timedelta(days=7)).isoformat(),
            'week_end': datetime.now().isoformat()
        }
    
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
            
            # Clean up weekly reports
            weekly_reports = Path("reports/weekly")
            if weekly_reports.exists():
                for report_file in weekly_reports.glob("*.html"):
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
    
    def start_monitoring(self) -> None:
        """Start the production health monitoring system"""
        self.logger.info("Starting production health monitoring system")
        
        # Set up scheduled monitoring
        self.schedule_production_monitoring()
        
        # Run initial health check
        asyncio.run(self.run_production_health_check())
        
        # Start scheduler loop
        self.logger.info("Production health monitoring is now active")
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def stop_monitoring(self) -> None:
        """Stop the production health monitoring system"""
        self.logger.info("Stopping production health monitoring system")
        schedule.clear()
        self.executor.shutdown(wait=True)


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
        print("To start monitoring, run: python -m tools.health-checker.production_deployment")
        
    except Exception as e:
        print(f"Failed to deploy production monitoring: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "deploy":
        # Deploy production monitoring
        asyncio.run(deploy_production_monitoring())
    else:
        # Start monitoring
        monitor = ProductionHealthMonitor()
        try:
            monitor.start_monitoring()
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("\nProduction health monitoring stopped")