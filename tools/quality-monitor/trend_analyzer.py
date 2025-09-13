"""
Quality trend analysis system.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import statistics

try:
    from tools.quality-monitor.models import QualityMetric, QualityTrend, MetricType, TrendDirection
except ImportError:
    from models import QualityMetric, QualityTrend, MetricType, TrendDirection


class TrendAnalyzer:
    """Analyzes quality metric trends over time."""
    
    def __init__(self, data_dir: str = "data/quality-metrics"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def store_metrics(self, metrics: List[QualityMetric]) -> None:
        """Store metrics data for trend analysis."""
        timestamp = datetime.now()
        filename = f"metrics_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.data_dir / filename
        
        data = {
            'timestamp': timestamp.isoformat(),
            'metrics': [metric.to_dict() for metric in metrics]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_historical_metrics(self, days: int = 30) -> Dict[MetricType, List[QualityMetric]]:
        """Load historical metrics for the specified number of days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        metrics_by_type = {}
        
        for file_path in self.data_dir.glob("metrics_*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                
                file_timestamp = datetime.fromisoformat(data['timestamp'])
                if file_timestamp < cutoff_date:
                    continue
                
                for metric_data in data['metrics']:
                    metric_type = MetricType(metric_data['metric_type'])
                    
                    metric = QualityMetric(
                        metric_type=metric_type,
                        value=metric_data['value'],
                        timestamp=datetime.fromisoformat(metric_data['timestamp']),
                        component=metric_data.get('component'),
                        details=metric_data.get('details', {})
                    )
                    
                    if metric_type not in metrics_by_type:
                        metrics_by_type[metric_type] = []
                    
                    metrics_by_type[metric_type].append(metric)
            
            except Exception as e:
                print(f"Error loading metrics from {file_path}: {e}")
        
        return metrics_by_type
    
    def analyze_trend(self, metric_type: MetricType, days: int = 30) -> Optional[QualityTrend]:
        """Analyze trend for a specific metric type."""
        historical_metrics = self.load_historical_metrics(days)
        
        if metric_type not in historical_metrics:
            return None
        
        metrics = sorted(historical_metrics[metric_type], key=lambda m: m.timestamp)
        
        if len(metrics) < 2:
            return None
        
        # Calculate trend direction and change rate
        values = [m.value for m in metrics]
        timestamps = [m.timestamp for m in metrics]
        
        # Simple linear regression for trend
        n = len(values)
        if n < 2:
            return None
        
        # Calculate slope (change rate)
        x_values = [(ts - timestamps[0]).total_seconds() / 86400 for ts in timestamps]  # Days
        
        if len(set(x_values)) < 2:  # All same timestamp
            return None
        
        # Linear regression
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Determine trend direction
        if abs(slope) < 0.1:  # Threshold for stable
            direction = TrendDirection.STABLE
        elif slope > 0:
            # For some metrics, higher is better (coverage), for others lower is better (complexity)
            if metric_type in [MetricType.TEST_COVERAGE, MetricType.DOCUMENTATION_COVERAGE, MetricType.TYPE_HINT_COVERAGE]:
                direction = TrendDirection.IMPROVING
            else:
                direction = TrendDirection.DEGRADING
        else:
            if metric_type in [MetricType.TEST_COVERAGE, MetricType.DOCUMENTATION_COVERAGE, MetricType.TYPE_HINT_COVERAGE]:
                direction = TrendDirection.DEGRADING
            else:
                direction = TrendDirection.IMPROVING
        
        # Calculate confidence based on data points and variance
        confidence = min(1.0, len(metrics) / 10.0)  # More data points = higher confidence
        if len(values) > 1:
            variance = statistics.variance(values)
            if variance > 0:
                cv = statistics.stdev(values) / statistics.mean(values) if statistics.mean(values) != 0 else 1
                confidence *= max(0.1, 1.0 - cv)  # Lower variance = higher confidence
        
        # Calculate percentage change rate per day
        if len(values) >= 2 and values[0] != 0:
            total_change = (values[-1] - values[0]) / values[0] * 100
            time_span = (timestamps[-1] - timestamps[0]).total_seconds() / 86400  # Days
            change_rate = total_change / time_span if time_span > 0 else 0
        else:
            change_rate = 0
        
        return QualityTrend(
            metric_type=metric_type,
            direction=direction,
            change_rate=change_rate,
            confidence=confidence,
            time_period_days=days,
            current_value=values[-1],
            previous_value=values[0]
        )
    
    def analyze_all_trends(self, days: int = 30) -> List[QualityTrend]:
        """Analyze trends for all available metric types."""
        trends = []
        
        for metric_type in MetricType:
            trend = self.analyze_trend(metric_type, days)
            if trend:
                trends.append(trend)
        
        return trends
    
    def get_trend_summary(self, days: int = 30) -> Dict[str, any]:
        """Get a summary of all trends."""
        trends = self.analyze_all_trends(days)
        
        summary = {
            'total_metrics': len(trends),
            'improving': len([t for t in trends if t.direction == TrendDirection.IMPROVING]),
            'stable': len([t for t in trends if t.direction == TrendDirection.STABLE]),
            'degrading': len([t for t in trends if t.direction == TrendDirection.DEGRADING]),
            'trends': [t.to_dict() for t in trends],
            'analysis_period_days': days,
            'generated_at': datetime.now().isoformat()
        }
        
        return summary
    
    def cleanup_old_data(self, keep_days: int = 90) -> int:
        """Clean up old metric data files."""
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        deleted_count = 0
        
        for file_path in self.data_dir.glob("metrics_*.json"):
            try:
                # Extract timestamp from filename
                timestamp_str = file_path.stem.replace('metrics_', '')
                file_timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                
                if file_timestamp < cutoff_date:
                    file_path.unlink()
                    deleted_count += 1
            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return deleted_count
