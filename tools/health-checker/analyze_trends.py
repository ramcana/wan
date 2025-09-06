#!/usr/bin/env python3
"""
Analyze health trends from current and historical health reports
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any


def load_health_history() -> Dict[str, Any]:
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


def update_health_history(current_report: Dict[str, Any]) -> None:
    """Update health history with current report data"""
    history_file = Path(__file__).parent / "health_history.json"
    history = load_health_history()
    
    # Add current score to history
    current_score = current_report.get('overall_score', 0)
    timestamp = current_report.get('timestamp', datetime.now().isoformat())
    
    # Add to score history (keep last 100 entries)
    score_history = history.get('score_history', [])
    score_history.append([timestamp, current_score])
    
    # Keep only last 100 entries
    if len(score_history) > 100:
        score_history = score_history[-100:]
    
    # Calculate improvement rate
    if len(score_history) >= 2:
        recent_scores = [entry[1] for entry in score_history[-10:]]  # Last 10 scores
        older_scores = [entry[1] for entry in score_history[-20:-10]] if len(score_history) >= 20 else []
        
        if older_scores:
            recent_avg = sum(recent_scores) / len(recent_scores)
            older_avg = sum(older_scores) / len(older_scores)
            improvement_rate = recent_avg - older_avg
        else:
            improvement_rate = 0.0
    else:
        improvement_rate = 0.0
    
    # Update history
    updated_history = {
        "score_history": score_history,
        "improvement_rate": improvement_rate,
        "last_updated": datetime.now().isoformat()
    }
    
    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(updated_history, f, indent=2)


def analyze_trends(current_report: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze trends from current report and history"""
    
    # Update history with current report
    update_health_history(current_report)
    
    # Load updated history
    history = load_health_history()
    score_history = history.get('score_history', [])
    
    if len(score_history) < 2:
        return {
            "error": "Insufficient data for trend analysis",
            "data_points": len(score_history),
            "recommendation": "Run more health checks to build trend data"
        }
    
    # Extract scores and timestamps
    scores = [entry[1] for entry in score_history]
    timestamps = [entry[0] for entry in score_history]
    
    # Basic statistics
    current_score = scores[-1] if scores else 0
    average_score = sum(scores) / len(scores) if scores else 0
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 0
    score_range = max_score - min_score
    
    # Trend analysis
    if len(scores) >= 3:
        # Simple linear trend (last 3 vs previous 3)
        recent_scores = scores[-3:]
        if len(scores) >= 6:
            previous_scores = scores[-6:-3]
        else:
            previous_scores = scores[:-3]
        
        recent_avg = sum(recent_scores) / len(recent_scores)
        previous_avg = sum(previous_scores) / len(previous_scores) if previous_scores else recent_avg
        
        trend_change = recent_avg - previous_avg
        
        if trend_change > 2:
            trend_direction = "improving"
            trend_strength = "strong" if abs(trend_change) > 5 else "moderate"
        elif trend_change < -2:
            trend_direction = "declining"
            trend_strength = "strong" if abs(trend_change) > 5 else "moderate"
        else:
            trend_direction = "stable"
            trend_strength = "stable"
    else:
        trend_direction = "unknown"
        trend_strength = "insufficient_data"
        trend_change = 0
    
    # Volatility analysis
    if len(scores) >= 5:
        # Calculate standard deviation of recent scores
        recent_scores = scores[-10:] if len(scores) >= 10 else scores
        mean = sum(recent_scores) / len(recent_scores)
        variance = sum((x - mean) ** 2 for x in recent_scores) / len(recent_scores)
        std_dev = variance ** 0.5
        
        if std_dev < 3:
            volatility_level = "low"
        elif std_dev < 8:
            volatility_level = "moderate"
        else:
            volatility_level = "high"
    else:
        volatility_level = "unknown"
        std_dev = 0
    
    # Generate recommendations
    recommendations = []
    
    if trend_direction == "declining":
        recommendations.append("Health score is declining - investigate recent changes")
        recommendations.append("Review failed checks and address critical issues")
    elif trend_direction == "improving":
        recommendations.append("Health score is improving - maintain current practices")
    elif trend_direction == "stable" and current_score < 75:
        recommendations.append("Health score is stable but below target - focus on improvements")
    
    if volatility_level == "high":
        recommendations.append("High volatility detected - stabilize development practices")
    
    if current_score < 50:
        recommendations.append("Critical health score - immediate action required")
    elif current_score < 75:
        recommendations.append("Health score needs improvement - review recommendations")
    
    # Time period analysis
    if timestamps:
        try:
            # Handle different timestamp formats
            first_ts = timestamps[0]
            last_ts = timestamps[-1]
            
            # Remove Z and add timezone if needed
            if first_ts.endswith('Z'):
                first_ts = first_ts.replace('Z', '+00:00')
            if last_ts.endswith('Z'):
                last_ts = last_ts.replace('Z', '+00:00')
            
            # Parse timestamps
            first_timestamp = datetime.fromisoformat(first_ts)
            last_timestamp = datetime.fromisoformat(last_ts)
            
            # Remove timezone info for calculation if both have it
            if first_timestamp.tzinfo and last_timestamp.tzinfo:
                first_timestamp = first_timestamp.replace(tzinfo=None)
                last_timestamp = last_timestamp.replace(tzinfo=None)
            
            time_span = (last_timestamp - first_timestamp).days
        except Exception as e:
            print(f"Warning: Could not parse timestamps for time span calculation: {e}")
            time_span = 0
    else:
        time_span = 0
    
    return {
        "analysis_timestamp": datetime.now().isoformat(),
        "time_period": {
            "data_points": len(scores),
            "time_span_days": time_span,
            "first_check": timestamps[0] if timestamps else None,
            "last_check": timestamps[-1] if timestamps else None
        },
        "score_analysis": {
            "current_score": current_score,
            "average_score": round(average_score, 1),
            "min_score": min_score,
            "max_score": max_score,
            "score_range": round(score_range, 1)
        },
        "trend_analysis": {
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "trend_change": round(trend_change, 1),
            "improvement_rate": round(history.get('improvement_rate', 0), 1)
        },
        "volatility_analysis": {
            "volatility_level": volatility_level,
            "standard_deviation": round(std_dev, 1)
        },
        "recommendations": recommendations,
        "score_history": score_history[-20:]  # Include last 20 data points
    }


def generate_trend_chart(trend_data: Dict[str, Any], output_file: str) -> None:
    """Generate a simple trend chart (requires matplotlib)"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        
        score_history = trend_data.get('score_history', [])
        if not score_history:
            print("No data available for chart generation")
            return
        
        # Extract data
        timestamps = []
        scores = [entry[1] for entry in score_history]
        
        for entry in score_history:
            ts = entry[0]
            if ts.endswith('Z'):
                ts = ts.replace('Z', '+00:00')
            try:
                parsed_ts = datetime.fromisoformat(ts)
                # Remove timezone for plotting
                if parsed_ts.tzinfo:
                    parsed_ts = parsed_ts.replace(tzinfo=None)
                timestamps.append(parsed_ts)
            except Exception:
                # Fallback to current time if parsing fails
                timestamps.append(datetime.now())
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, scores, marker='o', linewidth=2, markersize=4)
        plt.title('Project Health Score Trend', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Health Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(timestamps)//10)))
        plt.xticks(rotation=45)
        
        # Add trend line if enough data
        if len(scores) >= 3:
            # Simple linear regression
            x_numeric = list(range(len(scores)))
            n = len(scores)
            sum_x = sum(x_numeric)
            sum_y = sum(scores)
            sum_xy = sum(x * y for x, y in zip(x_numeric, scores))
            sum_x2 = sum(x * x for x in x_numeric)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
            
            trend_line = [slope * x + intercept for x in x_numeric]
            plt.plot(timestamps, trend_line, '--', color='red', alpha=0.7, label='Trend')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Trend chart saved to {output_file}")
        
    except ImportError:
        print("matplotlib not available - skipping chart generation")
    except Exception as e:
        print(f"Error generating chart: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze health trends")
    parser.add_argument('--current-report', required=True, help='Current health report JSON file')
    parser.add_argument('--output', required=True, help='Output trends JSON file')
    parser.add_argument('--chart-output', help='Output chart file (PNG)')
    
    args = parser.parse_args()
    
    try:
        # Load current report
        with open(args.current_report, 'r') as f:
            current_report = json.load(f)
        
        # Analyze trends
        trend_analysis = analyze_trends(current_report)
        
        # Save trend analysis
        with open(args.output, 'w') as f:
            json.dump(trend_analysis, f, indent=2)
        
        print(f"Trend analysis saved to {args.output}")
        
        # Generate chart if requested
        if args.chart_output:
            generate_trend_chart(trend_analysis, args.chart_output)
        
        # Print summary
        if "error" not in trend_analysis:
            score_analysis = trend_analysis.get('score_analysis', {})
            trend_info = trend_analysis.get('trend_analysis', {})
            
            print(f"\nTrend Analysis Summary:")
            print(f"Current Score: {score_analysis.get('current_score', 0):.1f}")
            print(f"Average Score: {score_analysis.get('average_score', 0):.1f}")
            print(f"Trend: {trend_info.get('trend_direction', 'unknown')} ({trend_info.get('trend_strength', 'unknown')})")
            
            recommendations = trend_analysis.get('recommendations', [])
            if recommendations:
                print(f"\nKey Recommendations:")
                for rec in recommendations[:3]:
                    print(f"- {rec}")
        
        return 0
        
    except FileNotFoundError:
        print(f"Error: Input file '{args.current_report}' not found")
        return 1
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{args.current_report}'")
        return 1
    except Exception as e:
        print(f"Error analyzing trends: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())