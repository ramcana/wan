#!/usr/bin/env python3
"""
Generate HTML dashboard from health report
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


def generate_html_dashboard(health_data, output_file):
    """Generate HTML dashboard from health report data"""
    
    # Extract key metrics
    overall_score = health_data.get('overall_score', 0)
    timestamp = health_data.get('timestamp', datetime.now().isoformat())
    categories = health_data.get('categories', {})
    issues = health_data.get('issues', [])
    recommendations = health_data.get('recommendations', [])
    
    # Determine overall status color
    if overall_score >= 90:
        status_color = "#28a745"  # Green
        status_text = "Excellent"
    elif overall_score >= 75:
        status_color = "#ffc107"  # Yellow
        status_text = "Good"
    elif overall_score >= 50:
        status_color = "#fd7e14"  # Orange
        status_text = "Needs Attention"
    else:
        status_color = "#dc3545"  # Red
        status_text = "Critical"
    
    # Generate category cards
    category_cards = ""
    for category, data in categories.items():
        score = data.get('score', 0)
        status = data.get('status', 'unknown')
        
        if score >= 90:
            card_color = "#d4edda"
        elif score >= 75:
            card_color = "#fff3cd"
        else:
            card_color = "#f8d7da"
        
        category_cards += f"""
        <div class="category-card" style="background-color: {card_color};">
            <h3>{category.replace('_', ' ').title()}</h3>
            <div class="score">{score:.1f}/100</div>
            <div class="status">{status.title()}</div>
        </div>
        """
    
    # Generate issues list
    issues_html = ""
    if issues:
        for issue in issues:
            severity = issue.get('severity', 'info')
            description = issue.get('description', 'No description')
            
            if severity == 'critical':
                issue_color = "#dc3545"
            elif severity == 'warning':
                issue_color = "#ffc107"
            else:
                issue_color = "#17a2b8"
            
            issues_html += f"""
            <div class="issue" style="border-left: 4px solid {issue_color};">
                <span class="severity">{severity.upper()}</span>
                <span class="description">{description}</span>
            </div>
            """
    else:
        issues_html = "<p>No issues found.</p>"
    
    # Generate recommendations list
    recommendations_html = ""
    if recommendations:
        for rec in recommendations:
            recommendations_html += f"<li>{rec}</li>"
    else:
        recommendations_html = "<li>No specific recommendations at this time.</li>"
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Health Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .overall-score {{
            font-size: 3em;
            font-weight: bold;
            color: {status_color};
        }}
        .status-text {{
            font-size: 1.5em;
            color: {status_color};
            margin-top: 10px;
        }}
        .timestamp {{
            color: #6c757d;
            margin-top: 10px;
        }}
        .categories {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .category-card {{
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            text-align: center;
        }}
        .category-card h3 {{
            margin: 0 0 10px 0;
            color: #495057;
        }}
        .category-card .score {{
            font-size: 2em;
            font-weight: bold;
            color: #495057;
        }}
        .category-card .status {{
            color: #6c757d;
            margin-top: 5px;
        }}
        .section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            margin-bottom: 20px;
        }}
        .section h2 {{
            margin-top: 0;
            color: #495057;
        }}
        .issue {{
            padding: 10px;
            margin-bottom: 10px;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        .issue .severity {{
            font-weight: bold;
            margin-right: 10px;
        }}
        .recommendations ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .recommendations li {{
            margin-bottom: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Project Health Dashboard</h1>
            <div class="overall-score">{overall_score:.1f}/100</div>
            <div class="status-text">{status_text}</div>
            <div class="timestamp">Last updated: {timestamp}</div>
        </div>
        
        <div class="categories">
            {category_cards}
        </div>
        
        <div class="section">
            <h2>Issues</h2>
            {issues_html}
        </div>
        
        <div class="section recommendations">
            <h2>Recommendations</h2>
            <ul>
                {recommendations_html}
            </ul>
        </div>
    </div>
</body>
</html>"""
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Dashboard generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate health dashboard")
    parser.add_argument('--input', default='health-report.json', help='Input health report JSON file')
    parser.add_argument('--output', default='health-dashboard.html', help='Output HTML file')
    
    args = parser.parse_args()
    
    try:
        # Load health report
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Health report not found: {input_path}")
            # Create a basic report
            health_data = {
                "timestamp": datetime.now().isoformat(),
                "overall_score": 75.0,
                "categories": {
                    "tests": {"score": 75.0, "status": "good"},
                    "documentation": {"score": 80.0, "status": "good"},
                    "configuration": {"score": 75.0, "status": "good"},
                    "code_quality": {"score": 75.0, "status": "good"}
                },
                "issues": [{"severity": "info", "description": "Health report not found, using default values"}],
                "recommendations": ["Run a proper health check to get accurate data"]
            }
        else:
            with open(input_path, 'r') as f:
                health_data = json.load(f)
        
        # Generate dashboard
        generate_html_dashboard(health_data, args.output)
        
        return 0
        
    except Exception as e:
        print(f"Error generating dashboard: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())