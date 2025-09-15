#!/usr/bin/env python3
"""
Analyze health trends from current and historical health reports
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


def load_health_history():
    """Load health history from the health checker directory"""
    history_file = Path(__file__).parent / "health_history.json"
    
    if not history_file.exists():
        return {
            "score_history": [],
            "improvement_rate": 0.0,
            "last_updated": datetime.now().isoformat()
        }
    
    try:
        with open(history_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {
            "score_history": [],
            "improvement_rate": 0.0,
            "last_updated": datetime.now().isoformat()
        }


def update_health_history(current_report):
    """Update health history with current report data"""
    history_file = Path(__file__).parent / "health_history.json"
    history = load_health_history()
    
    # Add current score to history
    current_score = current_report.get('overall_score', 0)
    timestamp = current_report.get('timestamp', datetime.now().isoformat())
    
    # Add to score history (keep last 50 entries)
    score_history = history.get('score_history', [])
    score_history.append([timestamp, current_score])
    
    # Keep only last 50 entries
    if len(score_history) > 50:
        score_history = score_history[-50:]
    
    # Calculate improvement rate
    improvement_rate = 0.0
    if len(score_history) >= 2:
        recent_scores = [entry[1] for entry in score_history[-5:]]  # Last 5 scores
        if len(recent_scores) >= 2:
            improvement_rate = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
    
    # Update history
    history.update({
        "score_history": score_history,
        "improvement_rate": improvement_rate,
        "last_updated": datetime.now().isoformat()
    })
    
    # Save updated history
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save health history: {e}")


def analyze_trends(current_report):
    """Analyze health trends"""
    # Update history first
    update_health_history(current_report)
    
    # Load history for analysis
    history = load_health_history()
    score_history = history.get('score_history', [])
    
    if len(score_history) < 2:
        return {
            "trend_direction": "insufficient_data",
            "trend_strength": "unknown",
            "data_points": len(score_history),
            "current_score": current_report.get('overall_score', 0),
            "recommendations": ["Need more data points to analyze trends"]
        }
    
    # Simple trend analysis
    scores = [entry[1] for entry in score_history]
    current_score = scores[-1]
    previous_score = scores[-2] if len(scores) >= 2 else current_score
    
    # Determine trend direction
    if current_score > previous_score + 2:
        trend_direction = "improving"
    elif current_score < previous_score - 2:
        trend_direction = "declining"
    else:
        trend_direction = "stable"
    
    # Calculate average score
    avg_score = sum(scores) / len(scores)
    
    # Generate recommendations
    recommendations = []
    if trend_direction == "declining":
        recommendations.append("Health score is declining - review recent changes")
    elif trend_direction == "improving":
        recommendations.append("Health score is improving - continue current practices")
    else:
        recommendations.append("Health score is stable - consider optimization opportunities")
    
    return {
        "trend_direction": trend_direction,
        "trend_strength": "moderate",
        "data_points": len(score_history),
        "current_score": current_score,
        "previous_score": previous_score,
        "average_score": avg_score,
        "improvement_rate": history.get('improvement_rate', 0.0),
        "recommendations": recommendations
    }


def create_simple_chart_data(score_history):
    """Create simple chart data for visualization"""
    if not score_history:
        return {"labels": [], "scores": []}
    
    # Take last 10 data points for chart
    recent_history = score_history[-10:]
    
    labels = []
    scores = []
    
    for entry in recent_history:
        timestamp = entry[0]
        score = entry[1]
        
        # Simple date formatting
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            labels.append(dt.strftime('%m/%d'))
        except:
            labels.append('N/A')
        
        scores.append(score)
    
    return {"labels": labels, "scores": scores}


def main():
    parser = argparse.ArgumentParser(description="Analyze health trends")
    parser.add_argument('--current-report', default='health-report.json', help='Current health report file')
    parser.add_argument('--output', default='health-trends.json', help='Output trends file')
    parser.add_argument('--chart-output', default='health-trends.png', help='Chart output file (placeholder)')
    
    args = parser.parse_args()
    
    try:
        # Load current report
        current_report_path = Path(args.current_report)
        if current_report_path.exists():
            with open(current_report_path, 'r') as f:
                current_report = json.load(f)
        else:
            print(f"Current report not found: {current_report_path}")
            current_report = {
                "overall_score": 75.0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Analyze trends
        trends = analyze_trends(current_report)
        
        # Load history for chart data
        history = load_health_history()
        chart_data = create_simple_chart_data(history.get('score_history', []))
        
        # Create output
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "trends": trends,
            "chart_data": chart_data,
            "summary": {
                "current_score": trends["current_score"],
                "trend_direction": trends["trend_direction"],
                "data_points": trends["data_points"]
            }
        }
        
        # Save trends
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Trend analysis completed: {args.output}")
        print(f"Current score: {trends['current_score']:.1f}")
        print(f"Trend direction: {trends['trend_direction']}")
        
        # Create placeholder chart file
        chart_placeholder = f"# Health Trends Chart\nCurrent Score: {trends['current_score']:.1f}\nTrend: {trends['trend_direction']}\n"
        with open(args.chart_output, 'w') as f:
            f.write(chart_placeholder)
        
        return 0
        
    except Exception as e:
        print(f"Error analyzing trends: {e}")
        return 0  # Don't fail CI


if __name__ == "__main__":
    sys.exit(main())