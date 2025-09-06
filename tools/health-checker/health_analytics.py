"""
Health analytics and trend analysis system
"""

import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

from health_models import HealthReport, HealthTrends, HealthCategory, Severity


class HealthAnalytics:
    """
    Advanced analytics for health monitoring data
    """
    
    def __init__(self, history_file: Optional[Path] = None):
        self.history_file = history_file or Path("tools/health-checker/health_history.json")
        self.logger = logging.getLogger(__name__)
    
    def analyze_health_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze health trends over specified time period
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Comprehensive trend analysis
        """
        try:
            history_data = self._load_history()
            if not history_data:
                return {"error": "No historical data available"}
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Filter data to specified time period
            score_history = [
                (datetime.fromisoformat(ts), score)
                for ts, score in history_data.get('score_history', [])
                if datetime.fromisoformat(ts) >= cutoff_date
            ]
            
            if len(score_history) < 2:
                return {"error": "Insufficient data for trend analysis"}
            
            analysis = {
                "time_period": {
                    "days": days,
                    "start_date": score_history[0][0].isoformat(),
                    "end_date": score_history[-1][0].isoformat(),
                    "data_points": len(score_history)
                },
                "score_analysis": self._analyze_scores(score_history),
                "trend_analysis": self._analyze_trends(score_history),
                "volatility_analysis": self._analyze_volatility(score_history),
                "pattern_analysis": self._analyze_patterns(score_history),
                "forecasting": self._forecast_trends(score_history),
                "recommendations": []
            }
            
            # Generate recommendations based on analysis
            analysis["recommendations"] = self._generate_analytics_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze health trends: {e}")
            return {"error": str(e)}
    
    def analyze_component_performance(self, component: str, days: int = 30) -> Dict[str, Any]:
        """Analyze performance of a specific component"""
        try:
            history_data = self._load_history()
            issue_trends = history_data.get('issue_trends', {})
            
            if component not in issue_trends:
                return {"error": f"No data available for component: {component}"}
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Filter component data
            component_data = [
                (datetime.fromisoformat(ts), count)
                for ts, count in issue_trends[component]
                if datetime.fromisoformat(ts) >= cutoff_date
            ]
            
            if not component_data:
                return {"error": "No recent data for component"}
            
            return {
                "component": component,
                "time_period": days,
                "issue_trend": self._analyze_issue_trend(component_data),
                "stability": self._calculate_stability(component_data),
                "improvement_rate": self._calculate_improvement_rate(component_data),
                "recommendations": self._generate_component_recommendations(component, component_data)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze component {component}: {e}")
            return {"error": str(e)}
    
    def generate_health_insights(self, report: HealthReport) -> Dict[str, Any]:
        """Generate actionable insights from current health report"""
        insights = {
            "critical_insights": [],
            "improvement_opportunities": [],
            "stability_concerns": [],
            "positive_trends": [],
            "action_priorities": []
        }
        
        # Analyze critical issues
        critical_issues = report.get_critical_issues()
        if critical_issues:
            insights["critical_insights"].append({
                "type": "critical_issues",
                "message": f"Found {len(critical_issues)} critical issues requiring immediate attention",
                "details": [issue.title for issue in critical_issues[:3]],
                "priority": 1
            })
        
        # Analyze component performance
        for name, component in report.component_scores.items():
            if component.score < 50:
                insights["critical_insights"].append({
                    "type": "component_failure",
                    "message": f"{name} component is in critical state ({component.score:.1f})",
                    "component": name,
                    "priority": 1
                })
            elif component.score < 75:
                insights["improvement_opportunities"].append({
                    "type": "component_improvement",
                    "message": f"{name} component needs attention ({component.score:.1f})",
                    "component": name,
                    "priority": 2
                })
        
        # Analyze trends
        if report.trends.improvement_rate < -2:
            insights["stability_concerns"].append({
                "type": "declining_trend",
                "message": f"Health score declining at {report.trends.improvement_rate:.2f} points per check",
                "priority": 2
            })
        elif report.trends.improvement_rate > 2:
            insights["positive_trends"].append({
                "type": "improving_trend",
                "message": f"Health score improving at {report.trends.improvement_rate:.2f} points per check"
            })
        
        # Generate action priorities
        insights["action_priorities"] = self._prioritize_actions(insights, report)
        
        return insights
    
    def calculate_health_metrics(self, report: HealthReport) -> Dict[str, Any]:
        """Calculate advanced health metrics"""
        metrics = {
            "overall_metrics": {
                "health_score": report.overall_score,
                "total_issues": len(report.issues),
                "critical_issues": len(report.get_critical_issues()),
                "components_healthy": sum(1 for comp in report.component_scores.values() if comp.score >= 75),
                "components_total": len(report.component_scores)
            },
            "risk_metrics": self._calculate_risk_metrics(report),
            "quality_metrics": self._calculate_quality_metrics(report),
            "stability_metrics": self._calculate_stability_metrics(report),
            "performance_metrics": self._calculate_performance_metrics(report)
        }
        
        return metrics
    
    def _load_history(self) -> Dict[str, Any]:
        """Load historical health data"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load history: {e}")
        return {}
    
    def _analyze_scores(self, score_history: List[Tuple[datetime, float]]) -> Dict[str, Any]:
        """Analyze score statistics"""
        scores = [score for _, score in score_history]
        
        return {
            "current_score": scores[-1],
            "average_score": statistics.mean(scores),
            "median_score": statistics.median(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_range": max(scores) - min(scores),
            "standard_deviation": statistics.stdev(scores) if len(scores) > 1 else 0
        }
    
    def _analyze_trends(self, score_history: List[Tuple[datetime, float]]) -> Dict[str, Any]:
        """Analyze trend direction and strength"""
        if len(score_history) < 2:
            return {"trend": "insufficient_data"}
        
        scores = [score for _, score in score_history]
        
        # Calculate linear trend
        n = len(scores)
        x_values = list(range(n))
        
        # Simple linear regression
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(scores)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, scores))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        slope = numerator / denominator if denominator != 0 else 0
        
        # Determine trend strength
        if abs(slope) < 0.1:
            trend_strength = "stable"
        elif abs(slope) < 0.5:
            trend_strength = "weak"
        elif abs(slope) < 1.0:
            trend_strength = "moderate"
        else:
            trend_strength = "strong"
        
        trend_direction = "improving" if slope > 0 else "declining" if slope < 0 else "stable"
        
        return {
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "slope": slope,
            "recent_change": scores[-1] - scores[-2] if len(scores) >= 2 else 0,
            "short_term_trend": self._calculate_short_term_trend(scores[-5:]) if len(scores) >= 5 else "insufficient_data"
        }
    
    def _analyze_volatility(self, score_history: List[Tuple[datetime, float]]) -> Dict[str, Any]:
        """Analyze score volatility and stability"""
        scores = [score for _, score in score_history]
        
        if len(scores) < 3:
            return {"volatility": "insufficient_data"}
        
        # Calculate rolling volatility
        rolling_std = []
        window_size = min(5, len(scores) // 2)
        
        for i in range(window_size, len(scores)):
            window_scores = scores[i-window_size:i]
            rolling_std.append(statistics.stdev(window_scores))
        
        avg_volatility = statistics.mean(rolling_std) if rolling_std else 0
        
        # Classify volatility
        if avg_volatility < 2:
            volatility_level = "low"
        elif avg_volatility < 5:
            volatility_level = "moderate"
        else:
            volatility_level = "high"
        
        return {
            "volatility_level": volatility_level,
            "average_volatility": avg_volatility,
            "current_volatility": rolling_std[-1] if rolling_std else 0,
            "volatility_trend": "increasing" if len(rolling_std) >= 2 and rolling_std[-1] > rolling_std[0] else "decreasing"
        }
    
    def _analyze_patterns(self, score_history: List[Tuple[datetime, float]]) -> Dict[str, Any]:
        """Analyze patterns in health data"""
        scores = [score for _, score in score_history]
        timestamps = [ts for ts, _ in score_history]
        
        patterns = {
            "cyclical_patterns": self._detect_cyclical_patterns(scores),
            "anomalies": self._detect_anomalies(scores),
            "recovery_patterns": self._detect_recovery_patterns(scores),
            "degradation_patterns": self._detect_degradation_patterns(scores)
        }
        
        return patterns
    
    def _forecast_trends(self, score_history: List[Tuple[datetime, float]]) -> Dict[str, Any]:
        """Simple trend forecasting"""
        if len(score_history) < 3:
            return {"forecast": "insufficient_data"}
        
        scores = [score for _, score in score_history]
        
        # Simple linear extrapolation
        recent_scores = scores[-5:]  # Use last 5 data points
        n = len(recent_scores)
        
        if n < 2:
            return {"forecast": "insufficient_data"}
        
        # Calculate trend from recent data
        x_values = list(range(n))
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(recent_scores)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, recent_scores))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        slope = numerator / denominator if denominator != 0 else 0
        intercept = y_mean - slope * x_mean
        
        # Forecast next few points
        forecast_points = []
        for i in range(1, 4):  # Forecast next 3 points
            forecast_score = intercept + slope * (n + i - 1)
            forecast_score = max(0, min(100, forecast_score))  # Clamp to valid range
            forecast_points.append(forecast_score)
        
        return {
            "method": "linear_extrapolation",
            "forecast_points": forecast_points,
            "confidence": "low" if abs(slope) > 2 else "medium",
            "trend_continuation": slope > 0
        }
    
    def _calculate_short_term_trend(self, recent_scores: List[float]) -> str:
        """Calculate short-term trend from recent scores"""
        if len(recent_scores) < 2:
            return "insufficient_data"
        
        first_half = recent_scores[:len(recent_scores)//2]
        second_half = recent_scores[len(recent_scores)//2:]
        
        if not first_half or not second_half:
            return "insufficient_data"
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        diff = second_avg - first_avg
        
        if abs(diff) < 1:
            return "stable"
        elif diff > 0:
            return "improving"
        else:
            return "declining"
    
    def _detect_cyclical_patterns(self, scores: List[float]) -> Dict[str, Any]:
        """Detect cyclical patterns in scores"""
        # Simple pattern detection - look for regular ups and downs
        if len(scores) < 6:
            return {"detected": False}
        
        # Look for alternating patterns
        direction_changes = 0
        for i in range(1, len(scores) - 1):
            if ((scores[i] > scores[i-1] and scores[i] > scores[i+1]) or
                (scores[i] < scores[i-1] and scores[i] < scores[i+1])):
                direction_changes += 1
        
        # If more than 30% of points are direction changes, consider it cyclical
        cyclical_ratio = direction_changes / (len(scores) - 2)
        
        return {
            "detected": cyclical_ratio > 0.3,
            "cyclical_ratio": cyclical_ratio,
            "pattern_strength": "strong" if cyclical_ratio > 0.5 else "weak"
        }
    
    def _detect_anomalies(self, scores: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalous scores"""
        if len(scores) < 5:
            return []
        
        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores)
        
        anomalies = []
        for i, score in enumerate(scores):
            z_score = abs(score - mean_score) / std_score if std_score > 0 else 0
            if z_score > 2:  # More than 2 standard deviations
                anomalies.append({
                    "index": i,
                    "score": score,
                    "z_score": z_score,
                    "type": "high" if score > mean_score else "low"
                })
        
        return anomalies
    
    def _detect_recovery_patterns(self, scores: List[float]) -> Dict[str, Any]:
        """Detect recovery patterns after drops"""
        recoveries = []
        
        for i in range(2, len(scores)):
            # Look for pattern: drop then recovery
            if (scores[i-2] > scores[i-1] and scores[i] > scores[i-1] and
                scores[i] - scores[i-1] > 5):  # Significant recovery
                recoveries.append({
                    "start_index": i-1,
                    "recovery_magnitude": scores[i] - scores[i-1]
                })
        
        return {
            "recovery_count": len(recoveries),
            "average_recovery": statistics.mean([r["recovery_magnitude"] for r in recoveries]) if recoveries else 0,
            "resilience_score": min(100, len(recoveries) * 10)  # Simple resilience metric
        }
    
    def _detect_degradation_patterns(self, scores: List[float]) -> Dict[str, Any]:
        """Detect degradation patterns"""
        degradations = []
        
        for i in range(1, len(scores)):
            if scores[i] < scores[i-1] - 5:  # Significant drop
                degradations.append({
                    "index": i,
                    "magnitude": scores[i-1] - scores[i]
                })
        
        return {
            "degradation_count": len(degradations),
            "average_degradation": statistics.mean([d["magnitude"] for d in degradations]) if degradations else 0,
            "stability_risk": min(100, len(degradations) * 15)  # Risk metric
        }
    
    def _calculate_risk_metrics(self, report: HealthReport) -> Dict[str, Any]:
        """Calculate risk-related metrics"""
        critical_issues = len(report.get_critical_issues())
        high_issues = len(report.get_issues_by_severity(Severity.HIGH))
        
        # Risk score based on issues and component health
        risk_score = 0
        risk_score += critical_issues * 25
        risk_score += high_issues * 15
        
        # Add risk from low-scoring components
        for component in report.component_scores.values():
            if component.score < 50:
                risk_score += 20
            elif component.score < 75:
                risk_score += 10
        
        risk_level = "low"
        if risk_score > 50:
            risk_level = "high"
        elif risk_score > 25:
            risk_level = "medium"
        
        return {
            "risk_score": min(100, risk_score),
            "risk_level": risk_level,
            "critical_issues": critical_issues,
            "high_issues": high_issues,
            "components_at_risk": sum(1 for comp in report.component_scores.values() if comp.score < 75)
        }
    
    def _calculate_quality_metrics(self, report: HealthReport) -> Dict[str, Any]:
        """Calculate quality-related metrics"""
        # Extract quality-related metrics from components
        test_component = report.component_scores.get("tests")
        code_quality_component = report.component_scores.get("code_quality")
        
        quality_metrics = {
            "test_health": test_component.score if test_component else 0,
            "code_quality": code_quality_component.score if code_quality_component else 0
        }
        
        # Add specific metrics if available
        if test_component:
            quality_metrics.update({
                "test_coverage": test_component.metrics.get("coverage_percent", 0),
                "test_pass_rate": test_component.metrics.get("pass_rate", 0)
            })
        
        if code_quality_component:
            quality_metrics.update({
                "syntax_errors": code_quality_component.metrics.get("syntax_errors", 0),
                "code_smells": code_quality_component.metrics.get("code_smells", 0)
            })
        
        return quality_metrics
    
    def _calculate_stability_metrics(self, report: HealthReport) -> Dict[str, Any]:
        """Calculate stability-related metrics"""
        return {
            "improvement_rate": report.trends.improvement_rate,
            "score_volatility": self._calculate_score_volatility(report.trends.score_history),
            "degradation_alerts": len(report.trends.degradation_alerts),
            "stability_score": max(0, 100 - abs(report.trends.improvement_rate) * 10)
        }
    
    def _calculate_performance_metrics(self, report: HealthReport) -> Dict[str, Any]:
        """Calculate performance-related metrics"""
        check_duration = report.metadata.get("check_duration", 0)
        
        return {
            "check_duration": check_duration,
            "performance_score": max(0, 100 - check_duration * 2),  # Penalty for slow checks
            "components_checked": len(report.component_scores),
            "issues_per_component": len(report.issues) / len(report.component_scores) if report.component_scores else 0
        }
    
    def _calculate_score_volatility(self, score_history: List[Tuple[datetime, float]]) -> float:
        """Calculate volatility from score history"""
        if len(score_history) < 2:
            return 0.0
        
        scores = [score for _, score in score_history[-10:]]  # Last 10 scores
        return statistics.stdev(scores) if len(scores) > 1 else 0.0
    
    def _generate_analytics_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analytics"""
        recommendations = []
        
        score_analysis = analysis.get("score_analysis", {})
        trend_analysis = analysis.get("trend_analysis", {})
        volatility_analysis = analysis.get("volatility_analysis", {})
        
        # Score-based recommendations
        if score_analysis.get("current_score", 0) < 50:
            recommendations.append("Critical: Health score is below 50. Immediate action required.")
        
        # Trend-based recommendations
        if trend_analysis.get("trend_direction") == "declining":
            if trend_analysis.get("trend_strength") == "strong":
                recommendations.append("Strong declining trend detected. Investigate root causes immediately.")
            else:
                recommendations.append("Declining trend detected. Monitor closely and address issues.")
        
        # Volatility-based recommendations
        if volatility_analysis.get("volatility_level") == "high":
            recommendations.append("High volatility detected. Focus on stabilizing the system.")
        
        # Pattern-based recommendations
        pattern_analysis = analysis.get("pattern_analysis", {})
        if pattern_analysis.get("cyclical_patterns", {}).get("detected"):
            recommendations.append("Cyclical patterns detected. Investigate recurring issues.")
        
        return recommendations
    
    def _generate_component_recommendations(self, component: str, data: List[Tuple[datetime, int]]) -> List[str]:
        """Generate component-specific recommendations"""
        recommendations = []
        
        issue_counts = [count for _, count in data]
        
        if not issue_counts:
            return recommendations
        
        avg_issues = statistics.mean(issue_counts)
        recent_issues = issue_counts[-1] if issue_counts else 0
        
        if recent_issues > avg_issues * 1.5:
            recommendations.append(f"{component} component showing increased issues. Investigate recent changes.")
        
        if avg_issues > 5:
            recommendations.append(f"{component} component consistently has many issues. Consider refactoring.")
        
        return recommendations
    
    def _prioritize_actions(self, insights: Dict[str, Any], report: HealthReport) -> List[Dict[str, Any]]:
        """Prioritize actions based on insights"""
        actions = []
        
        # Add critical insights as high priority actions
        for insight in insights["critical_insights"]:
            actions.append({
                "priority": 1,
                "action": insight["message"],
                "type": insight["type"],
                "urgency": "immediate"
            })
        
        # Add improvement opportunities as medium priority
        for insight in insights["improvement_opportunities"]:
            actions.append({
                "priority": 2,
                "action": insight["message"],
                "type": insight["type"],
                "urgency": "soon"
            })
        
        # Add stability concerns
        for insight in insights["stability_concerns"]:
            actions.append({
                "priority": 2,
                "action": insight["message"],
                "type": insight["type"],
                "urgency": "monitor"
            })
        
        # Sort by priority
        actions.sort(key=lambda x: x["priority"])
        
        return actions[:10]  # Return top 10 actions