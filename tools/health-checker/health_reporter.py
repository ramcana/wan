"""
Health reporting and analytics system
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from health_models import (
    HealthReport, HealthIssue, ComponentHealth, HealthTrends,
    HealthCategory, Severity, HealthConfig
)


class HealthReporter:
    """
    Generates comprehensive health reports with analytics and visualizations
    """
    
    def __init__(self, config: Optional[HealthConfig] = None):
        self.config = config or HealthConfig()
        self.logger = logging.getLogger(__name__)
        
        # Report output directory
        self.reports_dir = Path("tools/health-checker/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, health_report: HealthReport, format_type: str = "html") -> Path:
        """
        Generate a formatted health report
        
        Args:
            health_report: The health report data
            format_type: Output format ("html", "json", "markdown", "console")
            
        Returns:
            Path to the generated report file
        """
        timestamp = health_report.timestamp.strftime("%Y%m%d_%H%M%S")
        
        if format_type == "html":
            return self._generate_html_report(health_report, timestamp)
        elif format_type == "json":
            return self._generate_json_report(health_report, timestamp)
        elif format_type == "markdown":
            return self._generate_markdown_report(health_report, timestamp)
        elif format_type == "console":
            self._print_console_report(health_report)
            return Path("console")
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def _generate_html_report(self, report: HealthReport, timestamp: str) -> Path:
        """Generate HTML report with dashboard-style layout"""
        output_file = self.reports_dir / f"health_report_{timestamp}.html"
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Health Report - {report.timestamp.strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        {self._get_html_styles()}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>Project Health Report</h1>
            <p class="timestamp">Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        
        <div class="summary-section">
            <div class="score-card overall-score">
                <h2>Overall Health Score</h2>
                <div class="score {self._get_score_class(report.overall_score)}">{report.overall_score:.1f}</div>
                <div class="score-label">{self._get_score_label(report.overall_score)}</div>
            </div>
            
            <div class="metrics-grid">
                {self._generate_component_cards(report)}
            </div>
        </div>
        
        <div class="charts-section">
            <div class="chart-container">
                <canvas id="scoreChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="issuesChart"></canvas>
            </div>
        </div>
        
        <div class="issues-section">
            <h2>Issues by Severity</h2>
            {self._generate_issues_html(report)}
        </div>
        
        <div class="trends-section">
            <h2>Health Trends</h2>
            {self._generate_trends_html(report.trends)}
        </div>
        
        <div class="details-section">
            <h2>Component Details</h2>
            {self._generate_component_details_html(report)}
        </div>
    </div>
    
    <script>
        {self._generate_chart_scripts(report)}
    </script>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {output_file}")
        return output_file
    
    def _generate_json_report(self, report: HealthReport, timestamp: str) -> Path:
        """Generate JSON report for programmatic access"""
        output_file = self.reports_dir / f"health_report_{timestamp}.json"
        
        # Convert report to JSON-serializable format
        report_data = {
            "timestamp": report.timestamp.isoformat(),
            "overall_score": report.overall_score,
            "component_scores": {
                name: {
                    "component_name": comp.component_name,
                    "category": comp.category.value,
                    "score": comp.score,
                    "status": comp.status,
                    "issues_count": len(comp.issues),
                    "metrics": comp.metrics,
                    "last_checked": comp.last_checked.isoformat()
                }
                for name, comp in report.component_scores.items()
            },
            "issues": [
                {
                    "severity": issue.severity.value,
                    "category": issue.category.value,
                    "title": issue.title,
                    "description": issue.description,
                    "affected_components": issue.affected_components,
                    "remediation_steps": issue.remediation_steps,
                    "file_path": str(issue.file_path) if issue.file_path else None,
                    "line_number": issue.line_number,
                    "metadata": issue.metadata
                }
                for issue in report.issues
            ],
            "trends": {
                "score_history": [
                    [ts.isoformat(), score] for ts, score in report.trends.score_history
                ],
                "issue_trends": {
                    category: [[ts.isoformat(), count] for ts, count in trend_data]
                    for category, trend_data in report.trends.issue_trends.items()
                },
                "improvement_rate": report.trends.improvement_rate,
                "degradation_alerts": report.trends.degradation_alerts
            },
            "metadata": report.metadata
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"JSON report generated: {output_file}")
        return output_file
    
    def _generate_markdown_report(self, report: HealthReport, timestamp: str) -> Path:
        """Generate Markdown report for documentation"""
        output_file = self.reports_dir / f"health_report_{timestamp}.md"
        
        markdown_content = f"""# Project Health Report

**Generated:** {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Overall Health Score: {report.overall_score:.1f}/100

{self._get_score_emoji(report.overall_score)} **{self._get_score_label(report.overall_score)}**

## Component Scores

| Component | Score | Status | Issues |
|-----------|-------|--------|--------|
{self._generate_component_table_rows(report)}

## Issues Summary

### Critical Issues ({len(report.get_critical_issues())})
{self._generate_issues_markdown(report.get_issues_by_severity(Severity.CRITICAL))}

### High Priority Issues ({len(report.get_issues_by_severity(Severity.HIGH))})
{self._generate_issues_markdown(report.get_issues_by_severity(Severity.HIGH))}

### Medium Priority Issues ({len(report.get_issues_by_severity(Severity.MEDIUM))})
{self._generate_issues_markdown(report.get_issues_by_severity(Severity.MEDIUM))}

## Health Trends

- **Improvement Rate:** {report.trends.improvement_rate:.2f} points per check
- **Recent Score History:** {len(report.trends.score_history)} data points
{self._generate_trends_markdown(report.trends)}

## Component Details

{self._generate_component_details_markdown(report)}

---
*Report generated by Project Health Monitoring System*
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        self.logger.info(f"Markdown report generated: {output_file}")
        return output_file
    
    def _print_console_report(self, report: HealthReport) -> None:
        """Print a formatted console report"""
        print("\n" + "="*60)
        print(f"PROJECT HEALTH REPORT - {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Overall score
        score_color = self._get_console_color(report.overall_score)
        print(f"\nOverall Health Score: {score_color}{report.overall_score:.1f}/100\\033[0m")
        print(f"Status: {self._get_score_label(report.overall_score)}")
        
        # Component scores
        print("\nComponent Scores:")
        print("-" * 40)
        for name, component in report.component_scores.items():
            color = self._get_console_color(component.score)
            print(f"{name:20} {color}{component.score:5.1f}\\033[0m  {component.status}")
        
        # Critical issues
        critical_issues = report.get_critical_issues()
        if critical_issues:
            print(f"\n\\033[91mCRITICAL ISSUES ({len(critical_issues)}):\\033[0m")
            for issue in critical_issues[:5]:  # Show first 5
                print(f"  • {issue.title}")
        
        # High priority issues
        high_issues = report.get_issues_by_severity(Severity.HIGH)
        if high_issues:
            print(f"\n\\033[93mHIGH PRIORITY ISSUES ({len(high_issues)}):\\033[0m")
            for issue in high_issues[:3]:  # Show first 3
                print(f"  • {issue.title}")
        
        # Trends
        if report.trends.improvement_rate != 0:
            trend_color = "\\033[92m" if report.trends.improvement_rate > 0 else "\\033[91m"
            print(f"\nTrend: {trend_color}{report.trends.improvement_rate:+.2f}\\033[0m points per check")
        
        print("\n" + "="*60 + "\n")
    
    def generate_dashboard_data(self, report: HealthReport) -> Dict[str, Any]:
        """Generate data for real-time dashboard"""
        return {
            "timestamp": report.timestamp.isoformat(),
            "overall_score": report.overall_score,
            "status": self._get_score_label(report.overall_score),
            "components": {
                name: {
                    "score": comp.score,
                    "status": comp.status,
                    "issues_count": len(comp.issues)
                }
                for name, comp in report.component_scores.items()
            },
            "issues_by_severity": {
                severity.value: len(report.get_issues_by_severity(severity))
                for severity in Severity
            },
            "trends": {
                "recent_scores": [
                    {"timestamp": ts.isoformat(), "score": score}
                    for ts, score in report.trends.score_history[-10:]
                ],
                "improvement_rate": report.trends.improvement_rate
            },
            "alerts": report.trends.degradation_alerts
        }
    
    def analyze_trends(self, history_file: Path) -> Dict[str, Any]:
        """Analyze health trends from historical data"""
        try:
            with open(history_file, 'r') as f:
                history_data = json.load(f)
            
            score_history = history_data.get('score_history', [])
            if len(score_history) < 2:
                return {"error": "Insufficient historical data"}
            
            # Convert timestamps and calculate trends
            scores = [(datetime.fromisoformat(ts), score) for ts, score in score_history]
            scores.sort(key=lambda x: x[0])
            
            # Calculate various trend metrics
            recent_scores = [score for _, score in scores[-10:]]
            older_scores = [score for _, score in scores[-20:-10]] if len(scores) >= 20 else []
            
            analysis = {
                "total_data_points": len(scores),
                "current_score": scores[-1][1] if scores else 0,
                "previous_score": scores[-2][1] if len(scores) >= 2 else 0,
                "score_change": scores[-1][1] - scores[-2][1] if len(scores) >= 2 else 0,
                "recent_average": sum(recent_scores) / len(recent_scores) if recent_scores else 0,
                "older_average": sum(older_scores) / len(older_scores) if older_scores else 0,
                "trend_direction": "improving" if len(recent_scores) >= 2 and recent_scores[-1] > recent_scores[0] else "declining",
                "volatility": self._calculate_volatility(recent_scores),
                "time_range": {
                    "start": scores[0][0].isoformat(),
                    "end": scores[-1][0].isoformat()
                }
            }
            
            # Add recommendations based on trends
            analysis["recommendations"] = self._generate_trend_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze trends: {e}")
            return {"error": str(e)}
    
    def _calculate_volatility(self, scores: List[float]) -> float:
        """Calculate score volatility (standard deviation)"""
        if len(scores) < 2:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        return variance ** 0.5
    
    def _generate_trend_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on trend analysis"""
        recommendations = []
        
        if analysis["score_change"] < -5:
            recommendations.append("Health score has declined significantly. Review recent changes.")
        
        if analysis["volatility"] > 10:
            recommendations.append("Health score is highly volatile. Investigate unstable components.")
        
        if analysis["current_score"] < 50:
            recommendations.append("Health score is critically low. Immediate action required.")
        
        if analysis["trend_direction"] == "declining":
            recommendations.append("Health trend is declining. Focus on addressing root causes.")
        
        return recommendations
    
    # Helper methods for HTML generation
    def _get_html_styles(self) -> str:
        """Get CSS styles for HTML report"""
        return """
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px 8px 0 0; }
        header h1 { margin: 0; font-size: 2em; }
        .timestamp { margin: 5px 0 0 0; opacity: 0.8; }
        .summary-section { padding: 20px; }
        .score-card { text-align: center; background: #ecf0f1; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .score { font-size: 3em; font-weight: bold; margin: 10px 0; }
        .score.healthy { color: #27ae60; }
        .score.warning { color: #f39c12; }
        .score.critical { color: #e74c3c; }
        .score-label { font-size: 1.2em; text-transform: uppercase; letter-spacing: 1px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }
        .component-card { background: white; border: 1px solid #ddd; border-radius: 6px; padding: 15px; }
        .component-card h3 { margin: 0 0 10px 0; color: #2c3e50; }
        .charts-section { padding: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .chart-container { background: white; padding: 15px; border-radius: 6px; border: 1px solid #ddd; }
        .issues-section, .trends-section, .details-section { padding: 20px; border-top: 1px solid #eee; }
        .issue-item { background: #f8f9fa; border-left: 4px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 0 4px 4px 0; }
        .issue-item.critical { border-left-color: #e74c3c; }
        .issue-item.high { border-left-color: #f39c12; }
        .issue-item.medium { border-left-color: #3498db; }
        .issue-item.low { border-left-color: #95a5a6; }
        .issue-title { font-weight: bold; margin-bottom: 5px; }
        .issue-description { color: #666; margin-bottom: 10px; }
        .remediation-steps { font-size: 0.9em; }
        .remediation-steps li { margin: 3px 0; }
        """
    
    def _get_score_class(self, score: float) -> str:
        """Get CSS class for score"""
        if score >= self.config.warning_threshold:
            return "healthy"
        elif score >= self.config.critical_threshold:
            return "warning"
        else:
            return "critical"
    
    def _get_score_label(self, score: float) -> str:
        """Get human-readable score label"""
        if score >= self.config.warning_threshold:
            return "Healthy"
        elif score >= self.config.critical_threshold:
            return "Warning"
        else:
            return "Critical"
    
    def _get_score_emoji(self, score: float) -> str:
        """Get emoji for score"""
        if score >= self.config.warning_threshold:
            return "✅"
        elif score >= self.config.critical_threshold:
            return "⚠️"
        else:
            return "❌"
    
    def _get_console_color(self, score: float) -> str:
        """Get console color code for score"""
        if score >= self.config.warning_threshold:
            return "\\033[92m"  # Green
        elif score >= self.config.critical_threshold:
            return "\\033[93m"  # Yellow
        else:
            return "\\033[91m"  # Red
    
    def _generate_component_cards(self, report: HealthReport) -> str:
        """Generate HTML for component cards"""
        cards = []
        for name, component in report.component_scores.items():
            score_class = self._get_score_class(component.score)
            cards.append(f"""
            <div class="component-card">
                <h3>{name.replace('_', ' ').title()}</h3>
                <div class="score {score_class}">{component.score:.1f}</div>
                <div>Status: {component.status}</div>
                <div>Issues: {len(component.issues)}</div>
            </div>
            """)
        return "".join(cards)
    
    def _generate_issues_html(self, report: HealthReport) -> str:
        """Generate HTML for issues section"""
        html_parts = []
        
        for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
            issues = report.get_issues_by_severity(severity)
            if issues:
                html_parts.append(f"<h3>{severity.value.title()} Issues ({len(issues)})</h3>")
                for issue in issues:
                    html_parts.append(f"""
                    <div class="issue-item {severity.value}">
                        <div class="issue-title">{issue.title}</div>
                        <div class="issue-description">{issue.description}</div>
                        <div class="remediation-steps">
                            <strong>Remediation:</strong>
                            <ul>
                                {"".join(f"<li>{step}</li>" for step in issue.remediation_steps)}
                            </ul>
                        </div>
                    </div>
                    """)
        
        return "".join(html_parts)
    
    def _generate_trends_html(self, trends: HealthTrends) -> str:
        """Generate HTML for trends section"""
        return f"""
        <p><strong>Improvement Rate:</strong> {trends.improvement_rate:+.2f} points per check</p>
        <p><strong>Data Points:</strong> {len(trends.score_history)} historical records</p>
        {"<p><strong>Alerts:</strong> " + ", ".join(trends.degradation_alerts) + "</p>" if trends.degradation_alerts else ""}
        """
    
    def _generate_component_details_html(self, report: HealthReport) -> str:
        """Generate HTML for component details"""
        details = []
        for name, component in report.component_scores.items():
            details.append(f"""
            <div class="component-detail">
                <h3>{name.replace('_', ' ').title()}</h3>
                <p><strong>Score:</strong> {component.score:.1f}/100</p>
                <p><strong>Status:</strong> {component.status}</p>
                <p><strong>Last Checked:</strong> {component.last_checked.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Metrics:</strong></p>
                <ul>
                    {"".join(f"<li>{k}: {v}</li>" for k, v in component.metrics.items())}
                </ul>
            </div>
            """)
        return "".join(details)
    
    def _generate_chart_scripts(self, report: HealthReport) -> str:
        """Generate JavaScript for charts"""
        # Prepare data for charts
        component_names = list(report.component_scores.keys())
        component_scores = [comp.score for comp in report.component_scores.values()]
        
        issue_counts = {
            severity.value: len(report.get_issues_by_severity(severity))
            for severity in Severity
        }
        
        return f"""
        // Component Scores Chart
        const scoreCtx = document.getElementById('scoreChart').getContext('2d');
        new Chart(scoreCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(component_names)},
                datasets: [{{
                    label: 'Health Score',
                    data: {json.dumps(component_scores)},
                    backgroundColor: 'rgba(52, 152, 219, 0.8)',
                    borderColor: 'rgba(52, 152, 219, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Component Health Scores'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }}
            }}
        }});
        
        // Issues Chart
        const issuesCtx = document.getElementById('issuesChart').getContext('2d');
        new Chart(issuesCtx, {{
            type: 'doughnut',
            data: {{
                labels: {json.dumps(list(issue_counts.keys()))},
                datasets: [{{
                    data: {json.dumps(list(issue_counts.values()))},
                    backgroundColor: [
                        'rgba(231, 76, 60, 0.8)',
                        'rgba(243, 156, 18, 0.8)',
                        'rgba(52, 152, 219, 0.8)',
                        'rgba(149, 165, 166, 0.8)',
                        'rgba(46, 204, 113, 0.8)'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Issues by Severity'
                    }}
                }}
            }}
        }});
        """
    
    # Helper methods for Markdown generation
    def _generate_component_table_rows(self, report: HealthReport) -> str:
        """Generate table rows for component scores"""
        rows = []
        for name, component in report.component_scores.items():
            emoji = self._get_score_emoji(component.score)
            rows.append(f"| {name.replace('_', ' ').title()} | {component.score:.1f} | {emoji} {component.status} | {len(component.issues)} |")
        return "\n".join(rows)
    
    def _generate_issues_markdown(self, issues: List[HealthIssue]) -> str:
        """Generate markdown for issues list"""
        if not issues:
            return "\n*No issues found.*\n"
        
        markdown_parts = []
        for issue in issues[:5]:  # Show first 5 issues
            markdown_parts.append(f"""
#### {issue.title}

{issue.description}

**Affected Components:** {', '.join(issue.affected_components)}

**Remediation Steps:**
{chr(10).join(f"- {step}" for step in issue.remediation_steps)}
""")
        
        if len(issues) > 5:
            markdown_parts.append(f"\n*... and {len(issues) - 5} more issues*\n")
        
        return "\n".join(markdown_parts)
    
    def _generate_trends_markdown(self, trends: HealthTrends) -> str:
        """Generate markdown for trends"""
        parts = []
        
        if trends.score_history:
            recent_scores = [score for _, score in trends.score_history[-5:]]
            parts.append(f"- **Recent Scores:** {', '.join(f'{score:.1f}' for score in recent_scores)}")
        
        if trends.degradation_alerts:
            parts.append(f"- **Alerts:** {', '.join(trends.degradation_alerts)}")
        
        return "\n".join(parts)
    
    def _generate_component_details_markdown(self, report: HealthReport) -> str:
        """Generate markdown for component details"""
        details = []
        
        for name, component in report.component_scores.items():
            details.append(f"""
### {name.replace('_', ' ').title()}

- **Score:** {component.score:.1f}/100
- **Status:** {self._get_score_emoji(component.score)} {component.status}
- **Issues:** {len(component.issues)}
- **Last Checked:** {component.last_checked.strftime('%Y-%m-%d %H:%M:%S')}

**Metrics:**
{chr(10).join(f"- {k}: {v}" for k, v in component.metrics.items())}
""")
        
        return "\n".join(details)
