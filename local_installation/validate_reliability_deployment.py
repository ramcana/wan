#!/usr/bin/env python3
"""
Reliability System Deployment Validation Script

This script validates that the reliability system has been properly deployed
and configured. It performs comprehensive checks to ensure all components
are working correctly.

Requirements addressed:
- 1.4: User configurable retry limits and feature control validation
- 8.1: Health report generation and deployment integration validation
- 8.5: Cross-instance monitoring configuration validation
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add scripts directory to path
script_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(script_dir))


class DeploymentValidator:
    """Validates reliability system deployment."""
    
    def __init__(self):
        """Initialize deployment validator."""
        self.script_dir = Path(__file__).parent / "scripts"
        self.project_root = Path(__file__).parent
        self.validation_results = []
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup validation logging."""
        logger = logging.getLogger("deployment_validator")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def validate_deployment(self) -> bool:
        """Run complete deployment validation."""
        self.logger.info("Starting reliability system deployment validation")
        
        validation_checks = [
            ("Configuration Files", self._validate_configuration_files),
            ("Component Integration", self._validate_component_integration),
            ("Feature Flags", self._validate_feature_flags),
            ("Monitoring System", self._validate_monitoring_system),
            ("Deployment Artifacts", self._validate_deployment_artifacts),
            ("System Health", self._validate_system_health),
            ("Documentation", self._validate_documentation)
        ]
        
        all_passed = True
        
        for check_name, check_function in validation_checks:
            self.logger.info(f"Running validation: {check_name}")
            try:
                passed, details = check_function()
                self.validation_results.append({
                    "check": check_name,
                    "passed": passed,
                    "details": details,
                    "timestamp": datetime.now().isoformat()
                })
                
                if passed:
                    self.logger.info(f"✓ {check_name}: PASSED")
                else:
                    self.logger.error(f"✗ {check_name}: FAILED - {details}")
                    all_passed = False
                    
            except Exception as e:
                error_details = f"Exception during validation: {str(e)}"
                self.logger.error(f"✗ {check_name}: ERROR - {error_details}")
                self.validation_results.append({
                    "check": check_name,
                    "passed": False,
                    "details": error_details,
                    "timestamp": datetime.now().isoformat()
                })
                all_passed = False
        
        # Generate validation report
        self._generate_validation_report()
        
        if all_passed:
            self.logger.info("🎉 All validation checks passed! Reliability system is properly deployed.")
        else:
            self.logger.error("❌ Some validation checks failed. Please review the issues above.")
        
        return all_passed
    
    def _validate_configuration_files(self) -> Tuple[bool, str]:
        """Validate configuration files exist and are valid."""
        required_configs = [
            "reliability_config.json",
            "feature_flags.json",
            "production_monitoring_config.json"
        ]
        
        issues = []
        
        for config_file in required_configs:
            config_path = self.script_dir / config_file
            
            if not config_path.exists():
                issues.append(f"Missing configuration file: {config_file}")
                continue
            
            try:
                with open(config_path, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                issues.append(f"Invalid JSON in {config_file}: {str(e)}")
        
        # Validate reliability configuration specifically
        try:
            from reliability_config import get_reliability_config, get_config_manager
            config = get_reliability_config()
            config_manager = get_config_manager()
            validation_issues = config_manager.validate_config()
            
            if validation_issues:
                issues.extend([f"Reliability config: {issue}" for issue in validation_issues])
                
        except Exception as e:
            issues.append(f"Failed to load reliability configuration: {str(e)}")
        
        if issues:
            return False, "; ".join(issues)
        return True, "All configuration files are valid"
    
    def _validate_component_integration(self) -> Tuple[bool, str]:
        """Validate component integration."""
        issues = []
        
        # Check reliability integration
        try:
            from reliability_integration import get_reliability_integration
            integration = get_reliability_integration()
            
            if not integration.is_available():
                issues.append("Reliability integration is not available")
            else:
                # Test basic functionality
                health_status = integration.get_health_status()
                if not isinstance(health_status, dict):
                    issues.append("Health status check failed")
                    
        except ImportError as e:
            issues.append(f"Cannot import reliability integration: {str(e)}")
        except Exception as e:
            issues.append(f"Reliability integration error: {str(e)}")
        
        # Check main installer integration
        main_installer_path = self.script_dir / "main_installer.py"
        if main_installer_path.exists():
            try:
                with open(main_installer_path, 'r') as f:
                    content = f.read()
                
                if "ReliabilityManager" not in content:
                    issues.append("Main installer not integrated with reliability system")
                    
            except Exception as e:
                issues.append(f"Failed to check main installer integration: {str(e)}")
        else:
            issues.append("Main installer file not found")
        
        if issues:
            return False, "; ".join(issues)
        return True, "Component integration is working correctly"
    
    def _validate_feature_flags(self) -> Tuple[bool, str]:
        """Validate feature flags system."""
        issues = []
        
        try:
            from feature_flags import get_feature_flag_manager, is_feature_enabled
            flag_manager = get_feature_flag_manager()
            
            # Check that default flags exist
            expected_flags = [
                "enhanced_error_context",
                "missing_method_recovery",
                "model_validation_recovery",
                "network_failure_recovery",
                "dependency_recovery",
                "pre_installation_validation",
                "diagnostic_monitoring",
                "health_reporting",
                "timeout_management",
                "user_guidance_enhancements",
                "intelligent_retry_system"
            ]
            
            for flag_name in expected_flags:
                if flag_name not in flag_manager.flags:
                    issues.append(f"Missing feature flag: {flag_name}")
                else:
                    # Test flag evaluation
                    try:
                        is_enabled = is_feature_enabled(flag_name)
                        # Should return boolean without error
                        if not isinstance(is_enabled, bool):
                            issues.append(f"Feature flag {flag_name} evaluation returned non-boolean")
                    except Exception as e:
                        issues.append(f"Feature flag {flag_name} evaluation failed: {str(e)}")
            
            # Test flag status retrieval
            try:
                all_status = flag_manager.get_all_flags_status()
                if not isinstance(all_status, dict) or "flags" not in all_status:
                    issues.append("Feature flag status retrieval failed")
            except Exception as e:
                issues.append(f"Feature flag status retrieval error: {str(e)}")
                
        except ImportError as e:
            issues.append(f"Cannot import feature flags system: {str(e)}")
        except Exception as e:
            issues.append(f"Feature flags system error: {str(e)}")
        
        if issues:
            return False, "; ".join(issues)
        return True, "Feature flags system is working correctly"
    
    def _validate_monitoring_system(self) -> Tuple[bool, str]:
        """Validate monitoring system."""
        issues = []
        
        try:
            from production_monitoring import ProductionMonitor, MetricsCollector
            
            # Test monitor initialization
            monitor = ProductionMonitor()
            if not hasattr(monitor, 'config') or not hasattr(monitor, 'instance_id'):
                issues.append("Production monitor initialization failed")
            
            # Test metrics collection
            collector = MetricsCollector("test_instance")
            
            try:
                system_metrics = collector.collect_system_metrics()
                if not isinstance(system_metrics, list):
                    issues.append("System metrics collection failed")
                elif len(system_metrics) == 0:
                    issues.append("No system metrics collected")
            except Exception as e:
                issues.append(f"System metrics collection error: {str(e)}")
            
            try:
                app_metrics = collector.collect_application_metrics()
                if not isinstance(app_metrics, list):
                    issues.append("Application metrics collection failed")
            except Exception as e:
                issues.append(f"Application metrics collection error: {str(e)}")
                
        except ImportError as e:
            issues.append(f"Cannot import monitoring system: {str(e)}")
        except Exception as e:
            issues.append(f"Monitoring system error: {str(e)}")
        
        if issues:
            return False, "; ".join(issues)
        return True, "Monitoring system is working correctly"
    
    def _validate_deployment_artifacts(self) -> Tuple[bool, str]:
        """Validate deployment artifacts."""
        issues = []
        
        # Check deployment marker
        marker_file = self.project_root / ".reliability_deployed"
        if not marker_file.exists():
            issues.append("Deployment marker file not found")
        else:
            try:
                with open(marker_file, 'r') as f:
                    marker_data = json.load(f)
                
                required_fields = ["deployment_timestamp", "environment", "version"]
                for field in required_fields:
                    if field not in marker_data:
                        issues.append(f"Deployment marker missing field: {field}")
                        
            except json.JSONDecodeError:
                issues.append("Deployment marker file is invalid JSON")
            except Exception as e:
                issues.append(f"Failed to read deployment marker: {str(e)}")
        
        # Check integration wrapper
        wrapper_file = self.script_dir / "reliability_integration.py"
        if not wrapper_file.exists():
            issues.append("Reliability integration wrapper not found")
        
        # Check monitoring script
        monitor_script = self.script_dir / "monitor_reliability.py"
        if not monitor_script.exists():
            issues.append("Monitoring script not found")
        
        # Check log directories
        log_dir = self.project_root / "logs"
        if not log_dir.exists():
            issues.append("Logs directory not found")
        
        if issues:
            return False, "; ".join(issues)
        return True, "All deployment artifacts are present"
    
    def _validate_system_health(self) -> Tuple[bool, str]:
        """Validate overall system health."""
        issues = []
        
        try:
            # Test reliability system health
            from reliability_integration import get_reliability_integration
            integration = get_reliability_integration()
            
            if integration.is_available():
                health_status = integration.get_health_status()
                
                # Check for critical health indicators
                if isinstance(health_status, dict):
                    if health_status.get("status") == "error":
                        issues.append("Reliability system reports error status")
                    
                    # Check error rates if available
                    error_rate = health_status.get("error_rate", 0)
                    if isinstance(error_rate, (int, float)) and error_rate > 0.1:
                        issues.append(f"High error rate detected: {error_rate}")
                else:
                    issues.append("Invalid health status format")
            else:
                issues.append("Reliability system is not available")
                
        except Exception as e:
            issues.append(f"Health check failed: {str(e)}")
        
        # Check disk space
        try:
            import shutil
            free_space = shutil.disk_usage(self.project_root).free
            if free_space < 100 * 1024 * 1024:  # 100MB
                issues.append("Low disk space detected")
        except Exception as e:
            issues.append(f"Disk space check failed: {str(e)}")
        
        if issues:
            return False, "; ".join(issues)
        return True, "System health is good"
    
    def _validate_documentation(self) -> Tuple[bool, str]:
        """Validate documentation exists."""
        issues = []
        
        # Check configuration guide
        config_guide = self.project_root / "RELIABILITY_SYSTEM_CONFIGURATION_GUIDE.md"
        if not config_guide.exists():
            issues.append("Configuration guide not found")
        
        # Check if configuration templates can be generated
        try:
            from reliability_config import ReliabilityConfigManager
            manager = ReliabilityConfigManager()
            
            template_path = self.project_root / "temp_template_test.json"
            if manager.export_config_template(str(template_path)):
                template_path.unlink()  # Clean up
            else:
                issues.append("Cannot generate configuration template")
                
        except Exception as e:
            issues.append(f"Configuration template generation failed: {str(e)}")
        
        if issues:
            return False, "; ".join(issues)
        return True, "Documentation is available"
    
    def _generate_validation_report(self):
        """Generate validation report."""
        report_data = {
            "validation_timestamp": datetime.now().isoformat(),
            "total_checks": len(self.validation_results),
            "passed_checks": len([r for r in self.validation_results if r["passed"]]),
            "failed_checks": len([r for r in self.validation_results if not r["passed"]]),
            "results": self.validation_results,
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "script_directory": str(self.script_dir),
                "project_root": str(self.project_root)
            }
        }
        
        # Save report
        report_path = self.project_root / "logs" / f"deployment_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            self.logger.info(f"Validation report saved to: {report_path}")
        except Exception as e:
            self.logger.error(f"Failed to save validation report: {e}")


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate reliability system deployment")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--report-only", action="store_true", help="Generate report without console output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    validator = DeploymentValidator()
    
    if args.report_only:
        # Suppress console output for report-only mode
        validator.logger.handlers = []
    
    success = validator.validate_deployment()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()