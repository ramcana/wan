#!/usr/bin/env python3
"""
Health status badge generator for project visibility.

This module generates dynamic badges for health metrics that can be
displayed in README files and project dashboards.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests


class HealthBadgeGenerator:
    """Generates health status badges for project visibility."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(".github/badges")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Badge color thresholds
        self.health_thresholds = [
            (90, "brightgreen"),
            (80, "green"), 
            (70, "yellow"),
            (60, "orange"),
            (0, "red")
        ]
        
        self.coverage_thresholds = [
            (90, "brightgreen"),
            (80, "green"),
            (70, "yellow"),
            (60, "orange"),
            (0, "red")
        ]
    
    def get_badge_color(self, value: float, thresholds: List[Tuple[float, str]]) -> str:
        """Get badge color based on value and thresholds."""
        for threshold, color in thresholds:
            if value >= threshold:
                return color
        return "red"
    
    def create_shield_badge(self, label: str, message: str, color: str) -> Dict:
        """Create a shields.io compatible badge configuration."""
        return {
            "schemaVersion": 1,
            "label": label,
            "message": message,
            "color": color,
            "style": "flat-square"
        }
    
    def generate_health_badge(self, health_score: float) -> Dict:
        """Generate health score badge."""
        color = self.get_badge_color(health_score, self.health_thresholds)
        return self.create_shield_badge(
            label="health",
            message=f"{health_score:.1f}%",
            color=color
        )
    
    def generate_coverage_badge(self, coverage: float) -> Dict:
        """Generate test coverage badge."""
        color = self.get_badge_color(coverage, self.coverage_thresholds)
        return self.create_shield_badge(
            label="coverage",
            message=f"{coverage:.1f}%",
            color=color
        )
    
    def generate_test_status_badge(self, tests_passed: int, tests_total: int) -> Dict:
        """Generate test status badge."""
        if tests_total == 0:
            return self.create_shield_badge("tests", "no tests", "lightgrey")
        
        pass_rate = (tests_passed / tests_total) * 100
        color = "brightgreen" if pass_rate == 100 else "yellow" if pass_rate >= 80 else "red"
        
        return self.create_shield_badge(
            label="tests",
            message=f"{tests_passed}/{tests_total}",
            color=color
        )
    
    def generate_issues_badge(self, critical_issues: int, total_issues: int) -> Dict:
        """Generate issues status badge."""
        if critical_issues > 0:
            color = "red"
            message = f"{critical_issues} critical"
        elif total_issues > 10:
            color = "yellow"
            message = f"{total_issues} issues"
        elif total_issues > 0:
            color = "green"
            message = f"{total_issues} issues"
        else:
            color = "brightgreen"
            message = "no issues"
        
        return self.create_shield_badge(
            label="issues",
            message=message,
            color=color
        )
    
    def generate_deployment_status_badge(self, deployment_ready: bool, last_deployment: Optional[datetime] = None) -> Dict:
        """Generate deployment status badge."""
        if deployment_ready:
            color = "brightgreen"
            message = "ready"
        else:
            color = "red"
            message = "blocked"
        
        return self.create_shield_badge(
            label="deployment",
            message=message,
            color=color
        )
    
    def generate_all_badges(self, health_report: Dict) -> Dict[str, Dict]:
        """Generate all badges from health report."""
        badges = {}
        
        # Health score badge
        health_score = health_report.get("overall_score", 0)
        badges["health"] = self.generate_health_badge(health_score)
        
        # Coverage badge (if available)
        if "test_results" in health_report:
            coverage = health_report["test_results"].get("coverage_percentage", 0)
            badges["coverage"] = self.generate_coverage_badge(coverage)
            
            # Test status badge
            tests_passed = health_report["test_results"].get("tests_passed", 0)
            tests_total = health_report["test_results"].get("tests_total", 0)
            badges["tests"] = self.generate_test_status_badge(tests_passed, tests_total)
        
        # Issues badge
        issues = health_report.get("issues", [])
        critical_issues = len([i for i in issues if i.get("severity") == "critical"])
        badges["issues"] = self.generate_issues_badge(critical_issues, len(issues))
        
        # Deployment status badge
        deployment_ready = health_report.get("deployment_ready", False)
        badges["deployment"] = self.generate_deployment_status_badge(deployment_ready)
        
        return badges
    
    def save_badges(self, badges: Dict[str, Dict]) -> List[Path]:
        """Save badge configurations to files."""
        saved_files = []
        
        for badge_name, badge_config in badges.items():
            badge_file = self.output_dir / f"{badge_name}.json"
            
            with open(badge_file, 'w') as f:
                json.dump(badge_config, f, indent=2)
            
            saved_files.append(badge_file)
            print(f"✅ Saved {badge_name} badge to {badge_file}")
        
        return saved_files
    
    def generate_badge_urls(self, badges: Dict[str, Dict], base_url: str = None) -> Dict[str, str]:
        """Generate badge URLs for use in markdown."""
        if base_url is None:
            base_url = "https://img.shields.io/endpoint"
        
        urls = {}
        
        for badge_name, badge_config in badges.items():
            # Create URL for shields.io endpoint
            badge_url = f"{base_url}?url=https://raw.githubusercontent.com/{{owner}}/{{repo}}/main/.github/badges/{badge_name}.json"
            urls[badge_name] = badge_url
        
        return urls
    
    def generate_readme_badges_section(self, badges: Dict[str, Dict], repo_info: Dict = None) -> str:
        """Generate markdown section with badges for README."""
        if repo_info is None:
            repo_info = {"owner": "{owner}", "repo": "{repo}"}
        
        badge_urls = self.generate_badge_urls(badges)
        
        # Replace placeholders with actual repo info
        for badge_name, url in badge_urls.items():
            badge_urls[badge_name] = url.format(
                owner=repo_info["owner"],
                repo=repo_info["repo"]
            )
        
        markdown = "## Project Health Status\n\n"
        
        # Add badges in a logical order
        badge_order = ["health", "tests", "coverage", "issues", "deployment"]
        
        for badge_name in badge_order:
            if badge_name in badge_urls:
                badge_config = badges[badge_name]
                alt_text = f"{badge_config['label']} {badge_config['message']}"
                markdown += f"![{alt_text}]({badge_urls[badge_name]}) "
        
        markdown += "\n\n"
        
        # Add descriptions
        markdown += "### Badge Descriptions\n\n"
        markdown += "- **Health**: Overall project health score (0-100%)\n"
        markdown += "- **Tests**: Test suite status (passed/total)\n"
        markdown += "- **Coverage**: Code coverage percentage\n"
        markdown += "- **Issues**: Number of health issues found\n"
        markdown += "- **Deployment**: Deployment readiness status\n\n"
        
        return markdown
    
    def update_readme_badges(self, readme_path: Path, badges: Dict[str, Dict], repo_info: Dict = None) -> bool:
        """Update README file with current badges."""
        if not readme_path.exists():
            print(f"❌ README file not found: {readme_path}")
            return False
        
        # Read current README
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Generate new badges section
        badges_section = self.generate_readme_badges_section(badges, repo_info)
        
        # Look for existing badges section
        start_marker = "## Project Health Status"
        end_marker = "##"
        
        start_idx = content.find(start_marker)
        if start_idx != -1:
            # Find end of section
            end_idx = content.find(end_marker, start_idx + len(start_marker))
            if end_idx == -1:
                end_idx = len(content)
            
            # Replace existing section
            new_content = content[:start_idx] + badges_section + content[end_idx:]
        else:
            # Add badges section at the top (after title if present)
            lines = content.split('\n')
            insert_idx = 0
            
            # Skip title and description
            for i, line in enumerate(lines):
                if line.startswith('#') and i == 0:
                    insert_idx = i + 1
                    break
                elif line.strip() == '' and i > 0:
                    insert_idx = i + 1
                    break
            
            lines.insert(insert_idx, badges_section.rstrip())
            new_content = '\n'.join(lines)
        
        # Write updated README
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ Updated README badges in {readme_path}")
        return True


def main():
    """Main function for badge generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate health status badges")
    parser.add_argument("--health-report", required=True, help="Path to health report JSON file")
    parser.add_argument("--output-dir", default=".github/badges", help="Output directory for badges")
    parser.add_argument("--update-readme", help="Path to README file to update with badges")
    parser.add_argument("--repo-owner", help="Repository owner for badge URLs")
    parser.add_argument("--repo-name", help="Repository name for badge URLs")
    
    args = parser.parse_args()
    
    # Load health report
    with open(args.health_report, 'r') as f:
        health_report = json.load(f)
    
    # Create badge generator
    generator = HealthBadgeGenerator(Path(args.output_dir))
    
    # Generate badges
    badges = generator.generate_all_badges(health_report)
    
    # Save badge files
    saved_files = generator.save_badges(badges)
    
    print(f"\n✅ Generated {len(badges)} badges:")
    for badge_name, badge_config in badges.items():
        print(f"   - {badge_name}: {badge_config['message']} ({badge_config['color']})")
    
    # Update README if requested
    if args.update_readme:
        repo_info = None
        if args.repo_owner and args.repo_name:
            repo_info = {"owner": args.repo_owner, "repo": args.repo_name}
        
        generator.update_readme_badges(Path(args.update_readme), badges, repo_info)
    
    # Print badge URLs for manual use
    print(f"\n📋 Badge URLs (replace {{owner}}/{{repo}} with your repository):")
    badge_urls = generator.generate_badge_urls(badges)
    for badge_name, url in badge_urls.items():
        print(f"   - {badge_name}: {url}")


if __name__ == "__main__":
    main()
