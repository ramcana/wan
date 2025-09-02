#!/usr/bin/env python3
"""
Production Health Monitoring Deployment Script

This script handles the deployment of health monitoring to production environments,
including configuration validation, service setup, and initial health checks.
"""

import os
import sys
import json
import yaml
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import with correct path structure
sys.path.insert(0, str(project_root / "tools" / "health-checker"))
from production_deployment_simple import ProductionHealthMonitor
from health_checker import ProjectHealthChecker


class ProductionDeploymentManager:
    """Manages deployment of health monitoring to production"""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.project_root = Path(__file__).parent.parent
        self.config_dir = self.project_root / "config"
        self.deployment_log = []
        
    def log_step(self, message: str, success: bool = True) -> None:
        """Log deployment step with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "✓" if success else "✗"
        log_entry = f"[{timestamp}] {status} {message}"
        print(log_entry)
        self.deployment_log.append(log_entry)
    
    def validate_environment(self) -> bool:
        """Validate production environment requirements"""
        self.log_step("Validating production environment...")
        
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 8):
                self.log_step(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}", False)
                return False
            
            # Check required directories exist
            required_dirs = [
                "tools/health-checker",
                "config",
                "tests",
                "docs"
            ]
            
            for dir_path in required_dirs:
                full_path = self.project_root / dir_path
                if not full_path.exists():
                    self.log_step(f"Required directory missing: {dir_path}", False)
                    return False
            
            # Check required Python packages
            required_packages = [
                "yaml",
                "psutil"
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                self.log_step(f"Missing required packages: {', '.join(missing_packages)}", False)
                return False
            
            # Check system resources
            import psutil
            
            # Check available memory (minimum 2GB)
            available_memory = psutil.virtual_memory().available / (1024**3)
            if available_memory < 2.0:
                self.log_step(f"Insufficient memory: {available_memory:.1f}GB available, 2GB required", False)
                return False
            
            # Check available disk space (minimum 5GB)
            available_disk = psutil.disk_usage('/').free / (1024**3)
            if available_disk < 5.0:
                self.log_step(f"Insufficient disk space: {available_disk:.1f}GB available, 5GB required", False)
                return False
            
            self.log_step("Environment validation passed")
            return True
            
        except Exception as e:
            self.log_step(f"Environment validation failed: {e}", False)
            return False
    
    def setup_configuration(self) -> bool:
        """Set up production configuration files"""
        self.log_step("Setting up production configuration...")
        
        try:
            # Ensure config directory exists
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if production config exists
            prod_config_path = self.config_dir / "production-health.yaml"
            if not prod_config_path.exists():
                self.log_step("Production health config not found, using default", False)
                return False
            
            # Validate configuration
            with open(prod_config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Check required configuration sections
            required_sections = [
                "daily_report_time",
                "min_health_score",
                "notifications",
                "production_checks"
            ]
            
            for section in required_sections:
                if section not in config_data:
                    self.log_step(f"Missing required config section: {section}", False)
                    return False
            
            # Validate environment variables
            env_vars_needed = []
            
            # Check for Slack webhook if enabled
            if config_data.get("notifications", {}).get("slack", {}).get("enabled"):
                if not os.getenv("SLACK_WEBHOOK_URL"):
                    self.log_step("Warning: SLACK_WEBHOOK_URL not set, Slack notifications will be disabled")
            
            # Check for GitHub token if enabled
            if config_data.get("integrations", {}).get("github", {}).get("enabled"):
                if not os.getenv("GITHUB_TOKEN"):
                    self.log_step("Warning: GITHUB_TOKEN not set, GitHub integration will be disabled")
            
            if env_vars_needed:
                self.log_step(f"Missing environment variables: {', '.join(env_vars_needed)}", False)
                return False
            
            self.log_step("Configuration setup completed")
            return True
            
        except Exception as e:
            self.log_step(f"Configuration setup failed: {e}", False)
            return False
    
    def create_service_directories(self) -> bool:
        """Create necessary directories for production service"""
        self.log_step("Creating service directories...")
        
        try:
            directories = [
                "logs/health-monitoring",
                "logs/incidents",
                "reports/daily",
                "reports/weekly",
                "data/health-metrics",
                "backups/health-config"
            ]
            
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                self.log_step(f"Created directory: {directory}")
            
            # Set appropriate permissions
            if os.name != 'nt':  # Unix-like systems
                for directory in directories:
                    dir_path = self.project_root / directory
                    os.chmod(dir_path, 0o755)
            
            self.log_step("Service directories created")
            return True
            
        except Exception as e:
            self.log_step(f"Failed to create service directories: {e}", False)
            return False
    
    async def run_initial_health_check(self) -> bool:
        """Run initial health check to validate system"""
        self.log_step("Running initial health check...")
        
        try:
            # Create production monitor
            monitor = ProductionHealthMonitor()
            
            # Run comprehensive health check
            health_report = await monitor.run_production_health_check()
            
            # Log results
            self.log_step(f"Initial health score: {health_report.overall_score:.1f}")
            
            # Check if system meets production requirements
            min_score = monitor.config.min_health_score
            if health_report.overall_score < min_score:
                self.log_step(
                    f"Health score ({health_report.overall_score:.1f}) below "
                    f"production threshold ({min_score})", False
                )
                
                # Log top issues
                for issue in health_report.issues[:3]:
                    self.log_step(f"  Issue: {issue.get('description', 'Unknown issue')}")
                
                return False
            
            # Generate initial report
            await monitor.generate_daily_report()
            self.log_step("Initial health report generated")
            
            self.log_step("Initial health check passed")
            return True
            
        except Exception as e:
            self.log_step(f"Initial health check failed: {e}", False)
            return False
    
    def create_systemd_service(self) -> bool:
        """Create systemd service for production monitoring (Linux only)"""
        if os.name == 'nt':
            self.log_step("Skipping systemd service creation on Windows")
            return True
        
        self.log_step("Creating systemd service...")
        
        try:
            service_content = f"""[Unit]
Description=WAN22 Project Health Monitor
After=network.target
Wants=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory={self.project_root}
Environment=PYTHONPATH={self.project_root}
ExecStart=/usr/bin/python3 -m tools.health_checker.production_deployment
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
            
            service_file = Path("/tmp/wan22-health-monitor.service")
            with open(service_file, 'w') as f:
                f.write(service_content)
            
            self.log_step(f"Systemd service file created: {service_file}")
            self.log_step("To install: sudo cp /tmp/wan22-health-monitor.service /etc/systemd/system/")
            self.log_step("To enable: sudo systemctl enable wan22-health-monitor")
            self.log_step("To start: sudo systemctl start wan22-health-monitor")
            
            return True
            
        except Exception as e:
            self.log_step(f"Failed to create systemd service: {e}", False)
            return False
    
    def create_deployment_summary(self) -> None:
        """Create deployment summary report"""
        summary_file = self.project_root / "PRODUCTION_HEALTH_DEPLOYMENT_SUMMARY.md"
        
        summary_content = f"""# Production Health Monitoring Deployment Summary

**Deployment Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}
**Environment:** {self.environment}

## Deployment Steps

"""
        
        for log_entry in self.deployment_log:
            summary_content += f"- {log_entry}\n"
        
        summary_content += f"""

## Configuration

- **Config File:** `config/production-health.yaml`
- **Log Directory:** `logs/health-monitoring/`
- **Reports Directory:** `reports/`
- **Service Script:** `tools/health-checker/production_deployment.py`

## Next Steps

1. **Start Monitoring Service:**
   ```bash
   python -m tools.health_checker.production_deployment
   ```

2. **Enable Systemd Service (Linux):**
   ```bash
   sudo cp /tmp/wan22-health-monitor.service /etc/systemd/system/
   sudo systemctl enable wan22-health-monitor
   sudo systemctl start wan22-health-monitor
   ```

3. **Verify Service Status:**
   ```bash
   sudo systemctl status wan22-health-monitor
   ```

4. **Monitor Logs:**
   ```bash
   tail -f logs/health-monitoring/health_monitor_$(date +%Y%m%d).log
   ```

## Environment Variables Required

- `SLACK_WEBHOOK_URL` - For Slack notifications (if enabled)
- `GITHUB_TOKEN` - For GitHub integration (if enabled)

## Monitoring Schedule

- **Daily Reports:** {datetime.now().strftime("%H:%M")} UTC
- **Weekly Reports:** Monday {datetime.now().strftime("%H:%M")} UTC
- **Critical Checks:** Every 15 minutes

## Health Thresholds

- **Minimum Health Score:** 85.0
- **Critical Alert Threshold:** 70.0
- **Test Pass Rate:** 95.0%
- **Code Coverage:** 80.0%

## Support

For issues with health monitoring, check:
1. Service logs: `logs/health-monitoring/`
2. Incident reports: `logs/incidents/`
3. Configuration: `config/production-health.yaml`
"""
        
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        
        self.log_step(f"Deployment summary created: {summary_file}")
    
    async def deploy(self) -> bool:
        """Execute complete production deployment"""
        self.log_step(f"Starting production health monitoring deployment for {self.environment}")
        
        # Validation steps
        if not self.validate_environment():
            return False
        
        if not self.setup_configuration():
            return False
        
        if not self.create_service_directories():
            return False
        
        # Health validation
        if not await self.run_initial_health_check():
            return False
        
        # Service setup
        if not self.create_systemd_service():
            return False
        
        # Create summary
        self.create_deployment_summary()
        
        self.log_step("Production health monitoring deployment completed successfully!")
        return True


async def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy production health monitoring")
    parser.add_argument(
        "--environment", 
        default="production",
        help="Target environment (default: production)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate environment, don't deploy"
    )
    
    args = parser.parse_args()
    
    # Create deployment manager
    deployer = ProductionDeploymentManager(args.environment)
    
    if args.validate_only:
        # Only run validation
        if deployer.validate_environment():
            print("✓ Environment validation passed")
            return 0
        else:
            print("✗ Environment validation failed")
            return 1
    else:
        # Run full deployment
        success = await deployer.deploy()
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))