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
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
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
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .score-circle {{
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: conic-gradient({status_color} {overall_score * 3.6}deg, #e9ecef 0deg);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 20px auto;
            position: relative;
        }}
        .score-circle::before {{
            content: '';
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: white;
            position: absolute;
        }}
        .score-text {{
            position: relative;
            z-index: 1;
            font-size: 24px;
            font-weight: bold;
            color: {status_color};
        }}
        .status-badge {{
            background: {status_color};
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
            margin-top: 10px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card h3 {{
            margin-top: 0;
            color: #495057;
        }}
        .category-score {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #e9ecef;
        }}
        .category-score:last-child {{
            border-bottom: none;
        }}
        .score-bar {{
            width: 100px;
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
        }}
        .score-fill {{
            height: 100%;
            border-radius: 4px;
        }}
        .issue {{
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
            border-left: 4px solid;
        }}
        .issue.critical {{
            background: #f8d7da;
            border-color: #dc3545;
        }}
        .issue.warning {{
            background: #fff3cd;
            border-color: #ffc107;
        }}
        .issue.info {{
            background: #d1ecf1;
            border-color: #17a2b8;
        }}
        .recommendation {{
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 4px;
            padding: 10px;
            margin: 5px 0;
        }}
        .timestamp {{
            color: #6c757d;
            font-size: 14px;
            text-align: center;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Project Health Dashboard</h1>
            <div class="score-circle">
                <div class="score-text">{overall_score:.0f}</div>
            </div>
            <div class="status-badge">{status_text}</div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>Category Scores</h3>
    """
    
    # Add category scores
    for category, data in categories.items():
        score = data.get('score', 0)
        status = data.get('status', 'unknown')
        
        # Determine color based on score
        if score >= 90:
            color = "#28a745"
        elif score >= 75:
            color = "#ffc107"
        elif score >= 50:
            color = "#fd7e14"
        else:
            color = "#dc3545"
        
        html_content += f"""
                <div class="category-score">
                    <span>{category.replace('_', ' ').title()}</span>
                    <div>
                        <div class="score-bar">
                            <div class="score-fill" style="width: {score}%; background: {color};"></div>
                        </div>
                        <small>{score:.0f}%</small>
                    </div>
                </div>
        """
    
    html_content += """
            </div>
            
            <div class="card">
                <h3>Issues</h3>
    """
    
    # Add issues
    if issues:
        for issue in issues[:10]:  # Limit to first 10 issues
            severity = issue.get('severity', 'info')
            description = issue.get('description', 'Unknown issue')
            html_content += f"""
                <div class="issue {severity}">
                    <strong>{severity.upper()}:</strong> {description}
                </div>
            """
    else:
        html_content += "<p>No issues found! 🎉</p>"
    
    html_content += """
            </div>
            
            <div class="card">
                <h3>Recommendations</h3>
    """
    
    # Add recommendations
    if recommendations:
        for rec in recommendations[:10]:  # Limit to first 10 recommendations
            if isinstance(rec, dict):
                title = rec.get('title', 'Recommendation')
                description = rec.get('description', '')
            else:
                title = "Recommendation"
                description = str(rec)
            
            html_content += f"""
                <div class="recommendation">
                    <strong>{title}</strong><br>
                    {description}
                </div>
            """
    else:
        html_content += "<p>No specific recommendations at this time.</p>"
    
    html_content += f"""
            </div>
        </div>
        
        <div class="timestamp">
            Last updated: {timestamp}
        </div>
    </div>
</body>
</html>
    """
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML dashboard generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML dashboard from health report")
    parser.add_argument('--input', required=True, help='Input JSON health report file')
    parser.add_argument('--output', required=True, help='Output HTML file')
    
    args = parser.parse_args()
    
    try:
        # Load health report
        with open(args.input, 'r') as f:
            health_data = json.load(f)
        
        # Generate dashboard
        generate_html_dashboard(health_data, args.output)
        
        return 0
        
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found")
        return 1
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{args.input}'")
        return 1
    except Exception as e:
        print(f"Error generating dashboard: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
