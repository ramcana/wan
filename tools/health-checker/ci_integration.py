#!/usr/bin/env python3
"""
CI/CD integration utilities for health monitoring.

This module provides utilities for integrating health monitoring
with CI/CD pipelines and deployment gates.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess


class CIHealthIntegration:
    """Integrates health monitoring with CI/CD pipelines."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/ci-health-config.yaml")
        self.load_config()
    
    def load_config(self):
        """Load CI health integration configuration."""
        if self.config_path.exists():
            import yaml
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                "thresholds": {
                    "health_score": 80,
                    "critical_issues": 0,
                    "coverage": 70,
                    "test_pass_rate": 95
                },
                "deployment_gates": {
                    "main": {
                        "health_score": 85,
                        "critical_issues": 0,
                        "coverage": 80
                    },
                    "develop": {
                        "health_score": 75,
                        "critical_issues": 2,
                        "coverage": 70
                    }
                },
                "notifications": {
                    "slack_webhook": os.getenv("SLACK_WEBHOOK_URL"),
                    "email_recipients": [],
                    "github_issues": True
                }
            }
    
    def run_health_check_for_ci(self, comprehensive: bool = False, deployment_gate: bool = False) -> Dict:
        """Run health check optimized for CI environment."""
        
        # Determine check level based on context
        if deployment_gate:
            check_level = "deployment"
        elif comprehensive:
            check_level = "comprehensive"
        else:
            check_level = "standard"
        
        print(f"🔍 Running {check_level} health check for CI...")
        
        # Run health checker
        cmd = [
            sys.executable, "tools/health-checker/run_health_check.py",
            f"--comprehensive={comprehensive}",
            "--output-format=json",
            "--output-file=health-report.json",
            "--include-trends=true"
        ]
        
        if deployment_gate:
            cmd.append("--deployment-gate=true")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ Health check completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Health check failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise
        
        # Load and return results
        with open("health-report.json", 'r') as f:
            health_report = json.load(f)
        
        return health_report
    
    def evaluate_deployment_gate(self, health_report: Dict, branch: str = "main") -> Tuple[bool, List[str]]:
        """Evaluate if deployment gate requirements are met."""
        
        gate_config = self.config["deployment_gates"].get(branch, self.config["deployment_gates"]["main"])
        
        issues = []
        gate_passed = True
        
        # Check health score
        health_score = health_report.get("overall_score", 0)
        required_score = gate_config["health_score"]
        if health_score < required_score:
            issues.append(f"Health score {health_score}% below required {required_score}%")
            gate_passed = False
        
        # Check critical issues
        critical_issues = len([
            issue for issue in health_report.get("issues", [])
            if issue.get("severity") == "critical"
        ])
        max_critical = gate_config["critical_issues"]
        if critical_issues > max_critical:
            issues.append(f"Found {critical_issues} critical issues, maximum allowed: {max_critical}")
            gate_passed = False
        
        # Check coverage if available
        if "test_results" in health_report:
            coverage = health_report["test_results"].get("coverage_percentage", 0)
            required_coverage = gate_config["coverage"]
            if coverage < required_coverage:
                issues.append(f"Coverage {coverage}% below required {required_coverage}%")
                gate_passed = False
        
        return gate_passed, issues
    
    def generate_ci_summary(self, health_report: Dict, deployment_gate_result: Optional[Tuple[bool, List[str]]] = None) -> str:
        """Generate CI-friendly summary of health check results."""
        
        summary = []
        
        # Overall status
        health_score = health_report.get("overall_score", 0)
        summary.append(f"🏥 Health Score: {health_score}%")
        
        # Component scores
        component_scores = health_report.get("component_scores", {})
        for component, score in component_scores.items():
            emoji = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
            summary.append(f"{emoji} {component}: {score}%")
        
        # Issues summary
        issues = health_report.get("issues", [])
        if issues:
            critical = len([i for i in issues if i.get("severity") == "critical"])
            high = len([i for i in issues if i.get("severity") == "high"])
            medium = len([i for i in issues if i.get("severity") == "medium"])
            
            summary.append(f"🚨 Issues: {critical} critical, {high} high, {medium} medium")
        else:
            summary.append("✅ No issues found")
        
        # Deployment gate result
        if deployment_gate_result:
            gate_passed, gate_issues = deployment_gate_result
            if gate_passed:
                summary.append("🚀 Deployment gate: PASSED")
            else:
                summary.append("🚫 Deployment gate: FAILED")
                for issue in gate_issues:
                    summary.append(f"   - {issue}")
        
        return "\n".join(summary)
    
    def set_github_outputs(self, health_report: Dict, deployment_gate_result: Optional[Tuple[bool, List[str]]] = None):
        """Set GitHub Actions outputs for use in other steps."""
        
        github_output = os.getenv("GITHUB_OUTPUT")
        if not github_output:
            print("⚠️ GITHUB_OUTPUT not set, skipping output generation")
            return
        
        outputs = {
            "health-score": health_report.get("overall_score", 0),
            "critical-issues": len([
                issue for issue in health_report.get("issues", [])
                if issue.get("severity") == "critical"
            ]),
            "total-issues": len(health_report.get("issues", [])),
            "deployment-ready": "false"
        }
        
        # Add test results if available
        if "test_results" in health_report:
            test_results = health_report["test_results"]
            outputs.update({
                "tests-passed": test_results.get("tests_passed", 0),
                "tests-total": test_results.get("tests_total", 0),
                "coverage": test_results.get("coverage_percentage", 0)
            })
        
        # Add deployment gate result
        if deployment_gate_result:
            gate_passed, _ = deployment_gate_result
            outputs["deployment-ready"] = "true" if gate_passed else "false"
        
        # Write outputs
        with open(github_output, 'a') as f:
            for key, value in outputs.items():
                f.write(f"{key}={value}\n")
        
        print("✅ GitHub Actions outputs set:")
        for key, value in outputs.items():
            print(f"   {key}={value}")
    
    def create_status_check(self, health_report: Dict, context: str = "health-check") -> bool:
        """Create GitHub status check for health monitoring."""
        
        github_token = os.getenv("GITHUB_TOKEN")
        github_repo = os.getenv("GITHUB_REPOSITORY")
        github_sha = os.getenv("GITHUB_SHA")
        
        if not all([github_token, github_repo, github_sha]):
            print("⚠️ GitHub environment variables not set, skipping status check")
            return False
        
        import requests
        
        health_score = health_report.get("overall_score", 0)
        critical_issues = len([
            issue for issue in health_report.get("issues", [])
            if issue.get("severity") == "critical"
        ])
        
        # Determine status
        if critical_issues > 0:
            state = "failure"
            description = f"Health: {health_score}%, {critical_issues} critical issues"
        elif health_score >= 80:
            state = "success"
            description = f"Health: {health_score}%, no critical issues"
        else:
            state = "failure"
            description = f"Health: {health_score}%, below threshold"
        
        # Create status check
        url = f"https://api.github.com/repos/{github_repo}/statuses/{github_sha}"
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        data = {
            "state": state,
            "description": description,
            "context": context,
            "target_url": f"https://github.com/{github_repo}/actions/runs/{os.getenv('GITHUB_RUN_ID', '')}"
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 201:
            print(f"✅ Created status check: {context} - {state}")
            return True
        else:
            print(f"❌ Failed to create status check: {response.status_code}")
            return False
    
    def notify_health_issues(self, health_report: Dict):
        """Send notifications for health issues."""
        
        critical_issues = [
            issue for issue in health_report.get("issues", [])
            if issue.get("severity") == "critical"
        ]
        
        if not critical_issues:
            return
        
        # Slack notification
        slack_webhook = self.config["notifications"]["slack_webhook"]
        if slack_webhook:
            self._send_slack_notification(health_report, critical_issues, slack_webhook)
        
        # Email notification (if configured)
        email_recipients = self.config["notifications"]["email_recipients"]
        if email_recipients:
            self._send_email_notification(health_report, critical_issues, email_recipients)
    
    def _send_slack_notification(self, health_report: Dict, critical_issues: List[Dict], webhook_url: str):
        """Send Slack notification for critical health issues."""
        
        import requests
        
        health_score = health_report.get("overall_score", 0)
        
        message = {
            "text": f"🚨 Critical Project Health Issues Detected",
            "attachments": [
                {
                    "color": "danger",
                    "fields": [
                        {
                            "title": "Health Score",
                            "value": f"{health_score}%",
                            "short": True
                        },
                        {
                            "title": "Critical Issues",
                            "value": str(len(critical_issues)),
                            "short": True
                        }
                    ]
                }
            ]
        }
        
        # Add issue details
        if critical_issues:
            issue_text = "\n".join([
                f"• {issue.get('description', 'Unknown issue')}"
                for issue in critical_issues[:5]  # Limit to first 5
            ])
            
            if len(critical_issues) > 5:
                issue_text += f"\n... and {len(critical_issues) - 5} more issues"
            
            message["attachments"][0]["fields"].append({
                "title": "Issues",
                "value": issue_text,
                "short": False
            })
        
        try:
            response = requests.post(webhook_url, json=message)
            if response.status_code == 200:
                print("✅ Slack notification sent")
            else:
                print(f"❌ Failed to send Slack notification: {response.status_code}")
        except Exception as e:
            print(f"❌ Error sending Slack notification: {e}")


def main():
    """Main function for CI integration."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="CI/CD health monitoring integration")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive health check")
    parser.add_argument("--deployment-gate", action="store_true", help="Evaluate deployment gate")
    parser.add_argument("--branch", default="main", help="Branch for deployment gate evaluation")
    parser.add_argument("--create-status-check", action="store_true", help="Create GitHub status check")
    parser.add_argument("--notify", action="store_true", help="Send notifications for critical issues")
    
    args = parser.parse_args()
    
    # Create CI integration
    ci_integration = CIHealthIntegration()
    
    # Run health check
    health_report = ci_integration.run_health_check_for_ci(
        comprehensive=args.comprehensive,
        deployment_gate=args.deployment_gate
    )
    
    # Evaluate deployment gate if requested
    deployment_gate_result = None
    if args.deployment_gate:
        deployment_gate_result = ci_integration.evaluate_deployment_gate(health_report, args.branch)
        gate_passed, gate_issues = deployment_gate_result
        
        if not gate_passed:
            print(f"\n❌ Deployment gate failed for branch '{args.branch}':")
            for issue in gate_issues:
                print(f"   - {issue}")
            sys.exit(1)
        else:
            print(f"\n✅ Deployment gate passed for branch '{args.branch}'")
    
    # Generate and print summary
    summary = ci_integration.generate_ci_summary(health_report, deployment_gate_result)
    print(f"\n📊 Health Check Summary:\n{summary}")
    
    # Set GitHub Actions outputs
    ci_integration.set_github_outputs(health_report, deployment_gate_result)
    
    # Create status check if requested
    if args.create_status_check:
        ci_integration.create_status_check(health_report)
    
    # Send notifications if requested
    if args.notify:
        ci_integration.notify_health_issues(health_report)
    
    # Exit with appropriate code
    critical_issues = len([
        issue for issue in health_report.get("issues", [])
        if issue.get("severity") == "critical"
    ])
    
    if critical_issues > 0:
        print(f"\n❌ Exiting with error due to {critical_issues} critical issues")
        sys.exit(1)
    else:
        print("\n✅ Health check completed successfully")


if __name__ == "__main__":
    main()