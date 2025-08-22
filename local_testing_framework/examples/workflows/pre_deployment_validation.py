#!/usr/bin/env python3
"""
Pre-Deployment Validation Workflow

This script performs comprehensive validation before deploying to production.
It runs all tests, validates performance targets, and generates deployment
readiness reports.

Usage:
    python pre_deployment_validation.py [--environment prod|staging] [--strict]
"""

import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime

class DeploymentValidator:
    def __init__(self, environment="staging", strict_mode=False):
        self.environment = environment
        self.strict_mode = strict_mode
        self.results = {
            "environment_validation": None,
            "performance_tests": None,
            "integration_tests": None,
            "security_validation": None,
            "production_readiness": None,
            "overall_status": "PENDING"
        }
        
    def run_command(self, command, description, required=True):
        """Run a command and track results"""
        print(f"\n{'='*70}")
        print(f"🔄 Running: {description}")
        print(f"Command: {command}")
        print(f"{'='*70}")
        
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ SUCCESS: {description}")
                return True, result.stdout
            else:
                print(f"❌ FAILED: {description}")
                print(f"Error: {result.stderr}")
                if required and self.strict_mode:
                    print("🛑 STRICT MODE: Stopping due to failure")
                    sys.exit(1)
                return False, result.stderr
        except Exception as e:
            print(f"💥 ERROR: {description} - {str(e)}")
            if required and self.strict_mode:
                sys.exit(1)
            return False, str(e)

    def validate_environment(self):
        """Comprehensive environment validation"""
        print("\n🌍 PHASE 1: Environment Validation")
        
        # Basic environment check
        success, output = self.run_command(
            "python -m local_testing_framework validate-env --fix --report",
            "Environment validation with auto-fix"
        )
        self.results["environment_validation"] = success
        
        if not success:
            print("⚠️  Environment validation failed. Manual intervention required.")
            return False
            
        # Verify configuration for target environment
        config_file = f"examples/configurations/{self.environment}_config.json"
        if Path(config_file).exists():
            print(f"✅ Using {self.environment} configuration")
        else:
            print(f"⚠️  No specific configuration for {self.environment}")
            
        return success

    def run_performance_tests(self):
        """Comprehensive performance testing"""
        print("\n⚡ PHASE 2: Performance Testing")
        
        # Run comprehensive benchmarks
        success, output = self.run_command(
            "python -m local_testing_framework test-performance --benchmark --targets",
            "Comprehensive performance benchmarks"
        )
        
        if success:
            # Validate VRAM optimization
            vram_success, vram_output = self.run_command(
                "python -m local_testing_framework test-performance --vram-test",
                "VRAM optimization validation"
            )
            success = success and vram_success
            
        self.results["performance_tests"] = success
        
        if not success:
            print("⚠️  Performance tests failed. Check optimization settings.")
            
        return success

    def run_integration_tests(self):
        """Full integration test suite"""
        print("\n🔗 PHASE 3: Integration Testing")
        
        # Full integration test suite
        success, output = self.run_command(
            "python -m local_testing_framework test-integration --full",
            "Complete integration test suite"
        )
        
        self.results["integration_tests"] = success
        
        if not success:
            print("⚠️  Integration tests failed. Check system components.")
            
        return success

    def validate_security(self):
        """Security validation for production deployment"""
        print("\n🔒 PHASE 4: Security Validation")
        
        if self.environment == "production":
            # Check HTTPS configuration
            https_success, _ = self.run_command(
                "openssl s_client -connect localhost:8080 -servername localhost < /dev/null",
                "HTTPS configuration validation",
                required=False
            )
            
            # Check authentication
            auth_success, _ = self.run_command(
                "python -m local_testing_framework test-integration --api --auth",
                "Authentication validation",
                required=False
            )
            
            success = https_success and auth_success
        else:
            print("ℹ️  Skipping security validation for non-production environment")
            success = True
            
        self.results["security_validation"] = success
        return success

    def validate_production_readiness(self):
        """Production readiness validation"""
        print("\n🚀 PHASE 5: Production Readiness")
        
        # Run production validator
        success, output = self.run_command(
            "python -m local_testing_framework run-all --report-format json",
            "Production readiness validation"
        )
        
        self.results["production_readiness"] = success
        
        if success:
            # Generate deployment certificate
            cert_success, _ = self.run_command(
                f"python -m local_testing_framework generate-deployment-cert --environment {self.environment}",
                "Deployment certificate generation",
                required=False
            )
            
        return success

    def generate_deployment_report(self):
        """Generate comprehensive deployment report"""
        print("\n📊 PHASE 6: Report Generation")
        
        # Generate HTML report
        self.run_command(
            "python -m local_testing_framework run-all --report-format html --output reports/deployment",
            "HTML deployment report generation",
            required=False
        )
        
        # Generate JSON report for automation
        self.run_command(
            "python -m local_testing_framework run-all --report-format json --output reports/deployment",
            "JSON deployment report generation",
            required=False
        )
        
        # Create deployment summary
        summary = {
            "deployment_validation": {
                "timestamp": datetime.now().isoformat(),
                "environment": self.environment,
                "strict_mode": self.strict_mode,
                "results": self.results,
                "overall_status": self.results["overall_status"]
            }
        }
        
        with open(f"reports/deployment_summary_{self.environment}.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"📄 Deployment summary saved to reports/deployment_summary_{self.environment}.json")

    def run_validation(self):
        """Run complete validation workflow"""
        print("🚀 STARTING PRE-DEPLOYMENT VALIDATION")
        print(f"Environment: {self.environment}")
        print(f"Strict Mode: {self.strict_mode}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create reports directory
        Path("reports").mkdir(exist_ok=True)
        
        # Run validation phases
        phases = [
            ("Environment", self.validate_environment),
            ("Performance", self.run_performance_tests),
            ("Integration", self.run_integration_tests),
            ("Security", self.validate_security),
            ("Production Readiness", self.validate_production_readiness)
        ]
        
        all_passed = True
        for phase_name, phase_func in phases:
            try:
                success = phase_func()
                if not success:
                    all_passed = False
                    if self.strict_mode:
                        print(f"🛑 STRICT MODE: Stopping at {phase_name} phase")
                        break
            except KeyboardInterrupt:
                print("\n⏹️  Validation interrupted by user")
                return 1
            except Exception as e:
                print(f"💥 Unexpected error in {phase_name} phase: {e}")
                all_passed = False
                if self.strict_mode:
                    break
        
        # Determine overall status
        if all_passed:
            self.results["overall_status"] = "READY_FOR_DEPLOYMENT"
        elif self.strict_mode:
            self.results["overall_status"] = "DEPLOYMENT_BLOCKED"
        else:
            self.results["overall_status"] = "DEPLOYMENT_WITH_WARNINGS"
        
        # Generate reports
        self.generate_deployment_report()
        
        # Print final summary
        self.print_final_summary()
        
        return 0 if all_passed else 1

    def print_final_summary(self):
        """Print final validation summary"""
        print("\n" + "="*70)
        print("📋 DEPLOYMENT VALIDATION SUMMARY")
        print("="*70)
        
        status_emoji = {
            "READY_FOR_DEPLOYMENT": "✅",
            "DEPLOYMENT_WITH_WARNINGS": "⚠️",
            "DEPLOYMENT_BLOCKED": "❌"
        }
        
        print(f"Overall Status: {status_emoji.get(self.results['overall_status'], '❓')} {self.results['overall_status']}")
        print(f"Environment: {self.environment}")
        print(f"Validation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\nPhase Results:")
        for phase, result in self.results.items():
            if phase != "overall_status" and result is not None:
                emoji = "✅" if result else "❌"
                print(f"  {emoji} {phase.replace('_', ' ').title()}: {'PASSED' if result else 'FAILED'}")
        
        if self.results["overall_status"] == "READY_FOR_DEPLOYMENT":
            print("\n🎉 System is ready for deployment!")
        elif self.results["overall_status"] == "DEPLOYMENT_WITH_WARNINGS":
            print("\n⚠️  System can be deployed but has warnings. Review issues carefully.")
        else:
            print("\n🛑 System is NOT ready for deployment. Fix issues before proceeding.")

def main():
    parser = argparse.ArgumentParser(description="Pre-Deployment Validation")
    parser.add_argument(
        "--environment", 
        choices=["staging", "production"], 
        default="staging",
        help="Target deployment environment"
    )
    parser.add_argument(
        "--strict", 
        action="store_true", 
        help="Stop on first failure (strict mode)"
    )
    args = parser.parse_args()

    validator = DeploymentValidator(args.environment, args.strict)
    return validator.run_validation()

if __name__ == "__main__":
    sys.exit(main())