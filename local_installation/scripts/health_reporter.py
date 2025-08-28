"""
Health Reporting and Analytics System for Installation Reliability

This module provides comprehensive installation reporting, error pattern tracking,
trend analysis across installations, success/failure metrics collection,
centralized dashboard for multiple installation monitoring, and effective
recovery method logging for future optimization.

Requirements addressed: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import json
import logging
import sqlite3
import threading
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import hashlib

from interfaces import InstallationError, ErrorCategory, HardwareProfile, InstallationPhase
from base_classes import BaseInstallationComponent
from diagnostic_monitor import (
    ResourceMetrics, ComponentHealth, Alert, ErrorPattern, 
    PotentialIssue, HealthReport, HealthStatus, AlertLevel
)


class InstallationStatus(Enum):
    """Installation completion status."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    IN_PROGRESS = "in_progress"
    CANCELLED = "cancelled"


class RecoveryEffectiveness(Enum):
    """Effectiveness rating for recovery methods."""
    HIGHLY_EFFECTIVE = "highly_effective"  # >90% success rate
    EFFECTIVE = "effective"  # 70-90% success rate
    MODERATELY_EFFECTIVE = "moderately_effective"  # 50-70% success rate
    INEFFECTIVE = "ineffective"  # <50% success rate
    UNKNOWN = "unknown"


@dataclass
class InstallationReport:
    """Comprehensive installation report."""
    installation_id: str
    timestamp: datetime
    status: InstallationStatus
    duration_seconds: float
    hardware_profile: Optional[HardwareProfile]
    phases_completed: List[InstallationPhase]
    errors_encountered: List[Dict[str, Any]]
    warnings_generated: List[str]
    recovery_attempts: List[Dict[str, Any]]
    successful_recoveries: List[Dict[str, Any]]
    final_health_score: float  # 0-100 scale
    resource_usage_peak: Dict[str, float]
    performance_metrics: Dict[str, Any]
    user_interventions: int
    configuration_used: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['status'] = self.status.value
        result['phases_completed'] = [phase.value for phase in self.phases_completed]
        if self.hardware_profile:
            result['hardware_profile'] = asdict(self.hardware_profile)
        return result


@dataclass
class ErrorTrend:
    """Error trend analysis data."""
    error_type: str
    category: ErrorCategory
    frequency_trend: str  # "increasing", "decreasing", "stable"
    occurrence_count: int
    first_seen: datetime
    last_seen: datetime
    affected_installations: int
    average_resolution_time: float
    most_effective_recovery: Optional[str]
    confidence_score: float  # 0-1 scale
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        result['category'] = self.category.value
        result['first_seen'] = self.first_seen.isoformat()
        result['last_seen'] = self.last_seen.isoformat()
        return result


@dataclass
class TrendAnalysis:
    """Comprehensive trend analysis across installations."""
    analysis_period: timedelta
    total_installations: int
    success_rate: float
    average_installation_time: float
    most_common_errors: List[ErrorTrend]
    performance_trends: Dict[str, List[float]]
    hardware_correlation: Dict[str, float]
    recovery_effectiveness: Dict[str, RecoveryEffectiveness]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        result['analysis_period_days'] = self.analysis_period.days
        result['most_common_errors'] = [error.to_dict() for error in self.most_common_errors]
        result['recovery_effectiveness'] = {k: v.value for k, v in self.recovery_effectiveness.items()}
        result['generated_at'] = self.generated_at.isoformat()
        return result


@dataclass
class RecoveryMethodStats:
    """Statistics for recovery method effectiveness."""
    method_name: str
    total_attempts: int
    successful_attempts: int
    success_rate: float
    average_execution_time: float
    error_types_handled: List[str]
    last_used: datetime
    effectiveness_rating: RecoveryEffectiveness
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        result['last_used'] = self.last_used.isoformat()
        result['effectiveness_rating'] = self.effectiveness_rating.value
        return result


class HealthReporter(BaseInstallationComponent):
    """
    Comprehensive health reporting and analytics system for installation reliability.
    
    Provides installation reporting, error pattern tracking, trend analysis,
    success/failure metrics collection, centralized monitoring, and recovery
    method effectiveness logging.
    """
    
    def __init__(self, installation_path: str, config_path: str = "config.json"):
        """Initialize health reporter."""
        super().__init__(installation_path)
        self.config_path = config_path
        self.config = self._load_config()
        
        # Database setup
        self.db_path = Path(installation_path) / "logs" / "health_analytics.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Configuration
        self.retention_days = self.config.get("health_reporting", {}).get("retention_days", 90)
        self.analysis_window_days = self.config.get("health_reporting", {}).get("analysis_window_days", 30)
        self.min_data_points = self.config.get("health_reporting", {}).get("min_data_points", 5)
        
        # In-memory caches for performance
        self.installation_cache: deque = deque(maxlen=1000)
        self.error_pattern_cache: Dict[str, List[datetime]] = defaultdict(list)
        self.recovery_stats_cache: Dict[str, RecoveryMethodStats] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Callbacks for real-time monitoring
        self.report_callbacks: List[Callable[[InstallationReport], None]] = []
        self.trend_callbacks: List[Callable[[TrendAnalysis], None]] = []
        
        self.logger.info("HealthReporter initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load config from {self.config_path}: {e}")
            return {}
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Installation reports table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS installation_reports (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        status TEXT NOT NULL,
                        duration_seconds REAL NOT NULL,
                        hardware_profile TEXT,
                        phases_completed TEXT,
                        errors_encountered TEXT,
                        warnings_generated TEXT,
                        recovery_attempts TEXT,
                        successful_recoveries TEXT,
                        final_health_score REAL,
                        resource_usage_peak TEXT,
                        performance_metrics TEXT,
                        user_interventions INTEGER,
                        configuration_used TEXT
                    )
                ''')
                
                # Error patterns table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS error_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        error_type TEXT NOT NULL,
                        category TEXT NOT NULL,
                        installation_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        context TEXT,
                        recovery_method TEXT,
                        recovery_success BOOLEAN,
                        resolution_time REAL,
                        FOREIGN KEY (installation_id) REFERENCES installation_reports (id)
                    )
                ''')
                
                # Recovery method statistics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS recovery_stats (
                        method_name TEXT PRIMARY KEY,
                        total_attempts INTEGER DEFAULT 0,
                        successful_attempts INTEGER DEFAULT 0,
                        success_rate REAL DEFAULT 0.0,
                        average_execution_time REAL DEFAULT 0.0,
                        error_types_handled TEXT,
                        last_used TEXT,
                        effectiveness_rating TEXT
                    )
                ''')
                
                # Performance metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        installation_id TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (installation_id) REFERENCES installation_reports (id)
                    )
                ''')
                
                # Create indexes for better query performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_installation_timestamp ON installation_reports(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_error_timestamp ON error_patterns(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_error_type ON error_patterns(error_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)')
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def generate_installation_report(self, 
                                   installation_id: str,
                                   status: InstallationStatus,
                                   duration_seconds: float,
                                   hardware_profile: Optional[HardwareProfile] = None,
                                   phases_completed: Optional[List[InstallationPhase]] = None,
                                   errors_encountered: Optional[List[Dict[str, Any]]] = None,
                                   warnings_generated: Optional[List[str]] = None,
                                   recovery_attempts: Optional[List[Dict[str, Any]]] = None,
                                   successful_recoveries: Optional[List[Dict[str, Any]]] = None,
                                   resource_usage_peak: Optional[Dict[str, float]] = None,
                                   performance_metrics: Optional[Dict[str, Any]] = None,
                                   user_interventions: int = 0,
                                   configuration_used: Optional[Dict[str, Any]] = None) -> InstallationReport:
        """Generate comprehensive installation report."""
        
        with self.lock:
            # Calculate health score based on various factors
            health_score = self._calculate_health_score(
                status, len(errors_encountered or []), len(warnings_generated or []),
                len(recovery_attempts or []), len(successful_recoveries or []),
                user_interventions, duration_seconds
            )
            
            report = InstallationReport(
                installation_id=installation_id,
                timestamp=datetime.now(),
                status=status,
                duration_seconds=duration_seconds,
                hardware_profile=hardware_profile,
                phases_completed=phases_completed or [],
                errors_encountered=errors_encountered or [],
                warnings_generated=warnings_generated or [],
                recovery_attempts=recovery_attempts or [],
                successful_recoveries=successful_recoveries or [],
                final_health_score=health_score,
                resource_usage_peak=resource_usage_peak or {},
                performance_metrics=performance_metrics or {},
                user_interventions=user_interventions,
                configuration_used=configuration_used or {}
            )
            
            # Store in database
            self._store_installation_report(report)
            
            # Update caches
            self.installation_cache.append(report)
            
            # Update error patterns
            for error in errors_encountered or []:
                self._track_error_pattern(error, installation_id)
            
            # Update recovery statistics
            for recovery in recovery_attempts or []:
                self._update_recovery_stats(recovery)
            
            # Trigger callbacks
            for callback in self.report_callbacks:
                try:
                    callback(report)
                except Exception as e:
                    self.logger.error(f"Error in report callback: {e}")
            
            self.logger.info(f"Generated installation report for {installation_id}")
            return report
    
    def _calculate_health_score(self, status: InstallationStatus, error_count: int,
                              warning_count: int, recovery_attempts: int,
                              successful_recoveries: int, user_interventions: int,
                              duration_seconds: float) -> float:
        """Calculate overall health score for installation (0-100 scale)."""
        base_score = 100.0
        
        # Status impact
        if status == InstallationStatus.SUCCESS:
            status_penalty = 0
        elif status == InstallationStatus.PARTIAL:
            status_penalty = 20
        elif status == InstallationStatus.FAILURE:
            status_penalty = 50
        elif status == InstallationStatus.CANCELLED:
            status_penalty = 30
        else:  # IN_PROGRESS
            status_penalty = 10
        
        # Error impact
        error_penalty = min(30, error_count * 5)
        
        # Warning impact
        warning_penalty = min(10, warning_count * 1)
        
        # Recovery impact (positive for successful recoveries)
        recovery_bonus = min(20, successful_recoveries * 5)
        recovery_penalty = min(15, (recovery_attempts - successful_recoveries) * 3)
        
        # User intervention impact
        intervention_penalty = min(15, user_interventions * 5)
        
        # Duration impact (penalize very long installations)
        expected_duration = 1800  # 30 minutes baseline
        if duration_seconds > expected_duration * 2:
            duration_penalty = min(10, (duration_seconds - expected_duration) / expected_duration * 5)
        else:
            duration_penalty = 0
        
        final_score = (base_score - status_penalty - error_penalty - warning_penalty 
                      - recovery_penalty - intervention_penalty - duration_penalty + recovery_bonus)
        
        return max(0.0, min(100.0, final_score))
    
    def _store_installation_report(self, report: InstallationReport):
        """Store installation report in database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO installation_reports 
                    (id, timestamp, status, duration_seconds, hardware_profile, 
                     phases_completed, errors_encountered, warnings_generated,
                     recovery_attempts, successful_recoveries, final_health_score,
                     resource_usage_peak, performance_metrics, user_interventions,
                     configuration_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    report.installation_id,
                    report.timestamp.isoformat(),
                    report.status.value,
                    report.duration_seconds,
                    json.dumps(asdict(report.hardware_profile)) if report.hardware_profile else None,
                    json.dumps([phase.value for phase in report.phases_completed]),
                    json.dumps(report.errors_encountered),
                    json.dumps(report.warnings_generated),
                    json.dumps(report.recovery_attempts),
                    json.dumps(report.successful_recoveries),
                    report.final_health_score,
                    json.dumps(report.resource_usage_peak),
                    json.dumps(report.performance_metrics),
                    report.user_interventions,
                    json.dumps(report.configuration_used)
                ))
                
                # Store performance metrics separately for easier querying
                for metric_name, metric_value in report.performance_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        cursor.execute('''
                            INSERT INTO performance_metrics 
                            (installation_id, metric_name, metric_value, timestamp)
                            VALUES (?, ?, ?, ?)
                        ''', (
                            report.installation_id,
                            metric_name,
                            float(metric_value),
                            report.timestamp.isoformat()
                        ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store installation report: {e}")
    
    def track_error_patterns(self, errors: List[Dict[str, Any]]):
        """Track error patterns for trend analysis."""
        with self.lock:
            for error in errors:
                self._track_error_pattern(error, error.get('installation_id', 'unknown'))
    
    def _track_error_pattern(self, error: Dict[str, Any], installation_id: str):
        """Track individual error pattern."""
        try:
            error_type = error.get('type', 'unknown')
            category = error.get('category', 'unknown')
            timestamp = datetime.now()
            
            # Update in-memory cache
            self.error_pattern_cache[error_type].append(timestamp)
            
            # Store in database
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO error_patterns 
                    (error_type, category, installation_id, timestamp, context,
                     recovery_method, recovery_success, resolution_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    error_type,
                    category,
                    installation_id,
                    timestamp.isoformat(),
                    json.dumps(error.get('context', {})),
                    error.get('recovery_method'),
                    error.get('recovery_success', False),
                    error.get('resolution_time', 0.0)
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to track error pattern: {e}")
    
    def _update_recovery_stats(self, recovery: Dict[str, Any]):
        """Update recovery method statistics."""
        try:
            method_name = recovery.get('method', 'unknown')
            success = recovery.get('success', False)
            execution_time = recovery.get('execution_time', 0.0)
            error_types = recovery.get('error_types_handled', [])
            
            # Update in-memory cache
            if method_name not in self.recovery_stats_cache:
                self.recovery_stats_cache[method_name] = RecoveryMethodStats(
                    method_name=method_name,
                    total_attempts=0,
                    successful_attempts=0,
                    success_rate=0.0,
                    average_execution_time=0.0,
                    error_types_handled=[],
                    last_used=datetime.now(),
                    effectiveness_rating=RecoveryEffectiveness.UNKNOWN
                )
            
            stats = self.recovery_stats_cache[method_name]
            stats.total_attempts += 1
            if success:
                stats.successful_attempts += 1
            stats.success_rate = stats.successful_attempts / stats.total_attempts
            
            # Update average execution time
            stats.average_execution_time = (
                (stats.average_execution_time * (stats.total_attempts - 1) + execution_time) 
                / stats.total_attempts
            )
            
            # Update error types handled
            for error_type in error_types:
                if error_type not in stats.error_types_handled:
                    stats.error_types_handled.append(error_type)
            
            stats.last_used = datetime.now()
            
            # Update effectiveness rating
            if stats.success_rate >= 0.9:
                stats.effectiveness_rating = RecoveryEffectiveness.HIGHLY_EFFECTIVE
            elif stats.success_rate >= 0.7:
                stats.effectiveness_rating = RecoveryEffectiveness.EFFECTIVE
            elif stats.success_rate >= 0.5:
                stats.effectiveness_rating = RecoveryEffectiveness.MODERATELY_EFFECTIVE
            else:
                stats.effectiveness_rating = RecoveryEffectiveness.INEFFECTIVE
            
            # Store in database
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO recovery_stats 
                    (method_name, total_attempts, successful_attempts, success_rate,
                     average_execution_time, error_types_handled, last_used, effectiveness_rating)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    method_name,
                    stats.total_attempts,
                    stats.successful_attempts,
                    stats.success_rate,
                    stats.average_execution_time,
                    json.dumps(stats.error_types_handled),
                    stats.last_used.isoformat(),
                    stats.effectiveness_rating.value
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to update recovery stats: {e}")   
 
    def generate_trend_analysis(self, analysis_period_days: Optional[int] = None) -> TrendAnalysis:
        """Generate comprehensive trend analysis across installations."""
        
        if analysis_period_days is None:
            analysis_period_days = self.analysis_window_days
        
        analysis_period = timedelta(days=analysis_period_days)
        cutoff_time = datetime.now() - analysis_period
        
        with self.lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.cursor()
                    
                    # Get installation data within analysis period
                    cursor.execute('''
                        SELECT * FROM installation_reports 
                        WHERE timestamp >= ? 
                        ORDER BY timestamp DESC
                    ''', (cutoff_time.isoformat(),))
                    
                    installations = cursor.fetchall()
                    
                    if len(installations) < self.min_data_points:
                        self.logger.warning(f"Insufficient data for trend analysis: {len(installations)} installations")
                        return self._create_empty_trend_analysis(analysis_period)
                    
                    # Calculate basic metrics
                    total_installations = len(installations)
                    successful_installations = sum(1 for inst in installations if inst[2] == 'success')
                    success_rate = successful_installations / total_installations if total_installations > 0 else 0.0
                    
                    # Calculate average installation time
                    durations = [inst[3] for inst in installations]  # duration_seconds column
                    average_installation_time = statistics.mean(durations) if durations else 0.0
                    
                    # Analyze error trends
                    most_common_errors = self._analyze_error_trends(cursor, cutoff_time)
                    
                    # Analyze performance trends
                    performance_trends = self._analyze_performance_trends(cursor, cutoff_time)
                    
                    # Analyze hardware correlation
                    hardware_correlation = self._analyze_hardware_correlation(installations)
                    
                    # Analyze recovery effectiveness
                    recovery_effectiveness = self._analyze_recovery_effectiveness()
                    
                    # Generate recommendations
                    recommendations = self._generate_recommendations(
                        success_rate, most_common_errors, recovery_effectiveness
                    )
                    
                    trend_analysis = TrendAnalysis(
                        analysis_period=analysis_period,
                        total_installations=total_installations,
                        success_rate=success_rate,
                        average_installation_time=average_installation_time,
                        most_common_errors=most_common_errors,
                        performance_trends=performance_trends,
                        hardware_correlation=hardware_correlation,
                        recovery_effectiveness=recovery_effectiveness,
                        recommendations=recommendations
                    )
                    
                    # Trigger callbacks
                    for callback in self.trend_callbacks:
                        try:
                            callback(trend_analysis)
                        except Exception as e:
                            self.logger.error(f"Error in trend callback: {e}")
                    
                    self.logger.info(f"Generated trend analysis for {analysis_period_days} days")
                    return trend_analysis
                    
            except Exception as e:
                self.logger.error(f"Failed to generate trend analysis: {e}")
                return self._create_empty_trend_analysis(analysis_period)
    
    def _create_empty_trend_analysis(self, analysis_period: timedelta) -> TrendAnalysis:
        """Create empty trend analysis when insufficient data."""
        return TrendAnalysis(
            analysis_period=analysis_period,
            total_installations=0,
            success_rate=0.0,
            average_installation_time=0.0,
            most_common_errors=[],
            performance_trends={},
            hardware_correlation={},
            recovery_effectiveness={},
            recommendations=["Insufficient data for analysis. Need more installation reports."]
        )
    
    def _analyze_error_trends(self, cursor, cutoff_time: datetime) -> List[ErrorTrend]:
        """Analyze error trends from database."""
        try:
            cursor.execute('''
                SELECT error_type, category, COUNT(*) as count,
                       MIN(timestamp) as first_seen, MAX(timestamp) as last_seen,
                       COUNT(DISTINCT installation_id) as affected_installations,
                       AVG(resolution_time) as avg_resolution_time,
                       recovery_method
                FROM error_patterns 
                WHERE timestamp >= ?
                GROUP BY error_type, category
                ORDER BY count DESC
                LIMIT 10
            ''', (cutoff_time.isoformat(),))
            
            error_data = cursor.fetchall()
            error_trends = []
            
            for row in error_data:
                error_type, category, count, first_seen, last_seen, affected_installations, avg_resolution_time, recovery_method = row
                
                # Determine trend direction (simplified)
                trend_direction = self._calculate_error_trend_direction(cursor, error_type, cutoff_time)
                
                # Calculate confidence score
                confidence_score = min(1.0, count / 10.0)  # Simple confidence based on frequency
                
                error_trend = ErrorTrend(
                    error_type=error_type,
                    category=ErrorCategory(category) if category in [e.value for e in ErrorCategory] else ErrorCategory.SYSTEM,
                    frequency_trend=trend_direction,
                    occurrence_count=count,
                    first_seen=datetime.fromisoformat(first_seen),
                    last_seen=datetime.fromisoformat(last_seen),
                    affected_installations=affected_installations,
                    average_resolution_time=avg_resolution_time or 0.0,
                    most_effective_recovery=recovery_method,
                    confidence_score=confidence_score
                )
                error_trends.append(error_trend)
            
            return error_trends
            
        except Exception as e:
            self.logger.error(f"Failed to analyze error trends: {e}")
            return []
    
    def _calculate_error_trend_direction(self, cursor, error_type: str, cutoff_time: datetime) -> str:
        """Calculate trend direction for specific error type."""
        try:
            # Get error occurrences in two halves of the analysis period
            mid_time = cutoff_time + (datetime.now() - cutoff_time) / 2
            
            cursor.execute('''
                SELECT COUNT(*) FROM error_patterns 
                WHERE error_type = ? AND timestamp >= ? AND timestamp < ?
            ''', (error_type, cutoff_time.isoformat(), mid_time.isoformat()))
            first_half_count = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT COUNT(*) FROM error_patterns 
                WHERE error_type = ? AND timestamp >= ?
            ''', (error_type, mid_time.isoformat()))
            second_half_count = cursor.fetchone()[0]
            
            if second_half_count > first_half_count * 1.2:
                return "increasing"
            elif second_half_count < first_half_count * 0.8:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            self.logger.error(f"Failed to calculate trend direction: {e}")
            return "stable"
    
    def _analyze_performance_trends(self, cursor, cutoff_time: datetime) -> Dict[str, List[float]]:
        """Analyze performance trends from metrics data."""
        try:
            cursor.execute('''
                SELECT metric_name, metric_value, timestamp 
                FROM performance_metrics 
                WHERE timestamp >= ?
                ORDER BY metric_name, timestamp
            ''', (cutoff_time.isoformat(),))
            
            metrics_data = cursor.fetchall()
            performance_trends = defaultdict(list)
            
            for metric_name, metric_value, timestamp in metrics_data:
                performance_trends[metric_name].append(metric_value)
            
            # Limit to last 50 data points per metric for visualization
            for metric_name in performance_trends:
                performance_trends[metric_name] = performance_trends[metric_name][-50:]
            
            return dict(performance_trends)
            
        except Exception as e:
            self.logger.error(f"Failed to analyze performance trends: {e}")
            return {}
    
    def _analyze_hardware_correlation(self, installations: List[Tuple]) -> Dict[str, float]:
        """Analyze correlation between hardware profiles and success rates."""
        try:
            hardware_success = defaultdict(list)
            
            for installation in installations:
                hardware_profile_json = installation[4]  # hardware_profile column
                status = installation[2]  # status column
                
                if hardware_profile_json:
                    try:
                        hardware_profile = json.loads(hardware_profile_json)
                        
                        # Extract key hardware characteristics
                        gpu_model = hardware_profile.get('gpu', {}).get('model', 'unknown')
                        memory_gb = hardware_profile.get('memory', {}).get('total_gb', 0)
                        cpu_cores = hardware_profile.get('cpu', {}).get('cores', 0)
                        
                        success = 1.0 if status == 'success' else 0.0
                        
                        hardware_success[f'gpu_{gpu_model}'].append(success)
                        hardware_success[f'memory_{memory_gb}gb'].append(success)
                        hardware_success[f'cpu_{cpu_cores}cores'].append(success)
                        
                    except json.JSONDecodeError:
                        continue
            
            # Calculate success rates for each hardware characteristic
            correlations = {}
            for hw_key, success_values in hardware_success.items():
                if len(success_values) >= 3:  # Minimum sample size
                    correlations[hw_key] = statistics.mean(success_values)
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Failed to analyze hardware correlation: {e}")
            return {}
    
    def _analyze_recovery_effectiveness(self) -> Dict[str, RecoveryEffectiveness]:
        """Analyze effectiveness of recovery methods."""
        recovery_effectiveness = {}
        
        for method_name, stats in self.recovery_stats_cache.items():
            recovery_effectiveness[method_name] = stats.effectiveness_rating
        
        # Also load from database for methods not in cache
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT method_name, effectiveness_rating FROM recovery_stats')
                db_stats = cursor.fetchall()
                
                for method_name, effectiveness_str in db_stats:
                    if method_name not in recovery_effectiveness:
                        try:
                            recovery_effectiveness[method_name] = RecoveryEffectiveness(effectiveness_str)
                        except ValueError:
                            recovery_effectiveness[method_name] = RecoveryEffectiveness.UNKNOWN
        
        except Exception as e:
            self.logger.error(f"Failed to analyze recovery effectiveness: {e}")
        
        return recovery_effectiveness
    
    def _generate_recommendations(self, success_rate: float, 
                                error_trends: List[ErrorTrend],
                                recovery_effectiveness: Dict[str, RecoveryEffectiveness]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Success rate recommendations
        if success_rate < 0.7:
            recommendations.append("Critical: Installation success rate is below 70%. Review most common errors and improve recovery mechanisms.")
        elif success_rate < 0.9:
            recommendations.append("Warning: Installation success rate could be improved. Focus on top error patterns.")
        
        # Error pattern recommendations
        if error_trends:
            top_error = error_trends[0]
            if top_error.frequency_trend == "increasing":
                recommendations.append(f"Alert: '{top_error.error_type}' errors are increasing. Investigate root cause immediately.")
            
            if top_error.average_resolution_time > 300:  # 5 minutes
                recommendations.append(f"Optimize recovery for '{top_error.error_type}' - current resolution time is {top_error.average_resolution_time:.1f}s.")
        
        # Recovery effectiveness recommendations
        ineffective_methods = [
            method for method, effectiveness in recovery_effectiveness.items()
            if effectiveness == RecoveryEffectiveness.INEFFECTIVE
        ]
        
        if ineffective_methods:
            recommendations.append(f"Review ineffective recovery methods: {', '.join(ineffective_methods[:3])}")
        
        # General recommendations
        if len(error_trends) > 5:
            recommendations.append("High error diversity detected. Consider implementing more robust pre-installation validation.")
        
        if not recommendations:
            recommendations.append("System health is good. Continue monitoring for potential issues.")
        
        return recommendations
    
    def export_metrics(self, format: str = "json", 
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> str:
        """Export metrics in specified format."""
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=self.analysis_window_days)
        if end_date is None:
            end_date = datetime.now()
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Export installation reports
                cursor.execute('''
                    SELECT * FROM installation_reports 
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp DESC
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                installations = cursor.fetchall()
                
                # Export error patterns
                cursor.execute('''
                    SELECT * FROM error_patterns 
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp DESC
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                errors = cursor.fetchall()
                
                # Export recovery stats
                cursor.execute('SELECT * FROM recovery_stats')
                recovery_stats = cursor.fetchall()
                
                # Export performance metrics
                cursor.execute('''
                    SELECT * FROM performance_metrics 
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp DESC
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                performance_metrics = cursor.fetchall()
                
                if format.lower() == "json":
                    export_data = {
                        "export_timestamp": datetime.now().isoformat(),
                        "period": {
                            "start": start_date.isoformat(),
                            "end": end_date.isoformat()
                        },
                        "installation_reports": [
                            {
                                "id": row[0], "timestamp": row[1], "status": row[2],
                                "duration_seconds": row[3], "final_health_score": row[10],
                                "user_interventions": row[13]
                            } for row in installations
                        ],
                        "error_patterns": [
                            {
                                "error_type": row[1], "category": row[2], 
                                "timestamp": row[4], "recovery_success": row[6]
                            } for row in errors
                        ],
                        "recovery_stats": [
                            {
                                "method_name": row[0], "total_attempts": row[1],
                                "successful_attempts": row[2], "success_rate": row[3],
                                "effectiveness_rating": row[7]
                            } for row in recovery_stats
                        ],
                        "performance_metrics": [
                            {
                                "metric_name": row[2], "metric_value": row[3],
                                "timestamp": row[4]
                            } for row in performance_metrics
                        ]
                    }
                    
                    return json.dumps(export_data, indent=2)
                
                elif format.lower() == "csv":
                    # Simple CSV export for installations
                    csv_lines = ["installation_id,timestamp,status,duration_seconds,health_score,user_interventions"]
                    for row in installations:
                        csv_lines.append(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[10]},{row[13]}")
                    
                    return "\n".join(csv_lines)
                
                else:
                    raise ValueError(f"Unsupported export format: {format}")
                    
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return f"Export failed: {str(e)}"
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for centralized dashboard monitoring."""
        
        try:
            # Get recent trend analysis
            trend_analysis = self.generate_trend_analysis(7)  # Last 7 days
            
            # Get current recovery method effectiveness
            recovery_stats = list(self.recovery_stats_cache.values())
            
            # Get recent installation reports
            recent_reports = list(self.installation_cache)[-10:]  # Last 10 installations
            
            # Calculate real-time metrics
            current_time = datetime.now()
            recent_success_rate = 0.0
            if recent_reports:
                successful = sum(1 for r in recent_reports if r.status == InstallationStatus.SUCCESS)
                recent_success_rate = successful / len(recent_reports)
            
            dashboard_data = {
                "timestamp": current_time.isoformat(),
                "summary": {
                    "total_installations_7d": trend_analysis.total_installations,
                    "success_rate_7d": trend_analysis.success_rate,
                    "recent_success_rate": recent_success_rate,
                    "average_installation_time": trend_analysis.average_installation_time,
                    "active_error_patterns": len(trend_analysis.most_common_errors)
                },
                "trend_analysis": trend_analysis.to_dict(),
                "recovery_effectiveness": [stats.to_dict() for stats in recovery_stats],
                "recent_installations": [report.to_dict() for report in recent_reports],
                "recommendations": trend_analysis.recommendations,
                "health_status": self._calculate_overall_health_status(trend_analysis)
            }
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Failed to get dashboard data: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": f"Failed to generate dashboard data: {str(e)}"
            }
    
    def _calculate_overall_health_status(self, trend_analysis: TrendAnalysis) -> str:
        """Calculate overall system health status."""
        if trend_analysis.success_rate >= 0.95:
            return "excellent"
        elif trend_analysis.success_rate >= 0.85:
            return "good"
        elif trend_analysis.success_rate >= 0.70:
            return "fair"
        else:
            return "poor"
    
    def add_report_callback(self, callback: Callable[[InstallationReport], None]):
        """Add callback for installation report events."""
        self.report_callbacks.append(callback)
    
    def add_trend_callback(self, callback: Callable[[TrendAnalysis], None]):
        """Add callback for trend analysis events."""
        self.trend_callbacks.append(callback)
    
    def cleanup_old_data(self):
        """Clean up old data based on retention policy."""
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Clean up old installation reports
                cursor.execute('DELETE FROM installation_reports WHERE timestamp < ?', 
                             (cutoff_time.isoformat(),))
                
                # Clean up old error patterns
                cursor.execute('DELETE FROM error_patterns WHERE timestamp < ?', 
                             (cutoff_time.isoformat(),))
                
                # Clean up old performance metrics
                cursor.execute('DELETE FROM performance_metrics WHERE timestamp < ?', 
                             (cutoff_time.isoformat(),))
                
                conn.commit()
                
                self.logger.info(f"Cleaned up data older than {self.retention_days} days")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
    
    def get_recovery_method_stats(self) -> List[RecoveryMethodStats]:
        """Get statistics for all recovery methods."""
        return list(self.recovery_stats_cache.values())
    
    def get_installation_history(self, limit: int = 100) -> List[InstallationReport]:
        """Get recent installation history."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM installation_reports 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                
                rows = cursor.fetchall()
                reports = []
                
                for row in rows:
                    # Reconstruct InstallationReport from database row
                    hardware_profile = None
                    if row[4]:  # hardware_profile column
                        try:
                            hw_data = json.loads(row[4])
                            # Would need to reconstruct HardwareProfile object here
                            # For now, we'll skip this complex reconstruction
                        except json.JSONDecodeError:
                            pass
                    
                    report = InstallationReport(
                        installation_id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        status=InstallationStatus(row[2]),
                        duration_seconds=row[3],
                        hardware_profile=hardware_profile,
                        phases_completed=[InstallationPhase(phase) for phase in json.loads(row[5] or '[]')],
                        errors_encountered=json.loads(row[6] or '[]'),
                        warnings_generated=json.loads(row[7] or '[]'),
                        recovery_attempts=json.loads(row[8] or '[]'),
                        successful_recoveries=json.loads(row[9] or '[]'),
                        final_health_score=row[10] or 0.0,
                        resource_usage_peak=json.loads(row[11] or '{}'),
                        performance_metrics=json.loads(row[12] or '{}'),
                        user_interventions=row[13] or 0,
                        configuration_used=json.loads(row[14] or '{}')
                    )
                    reports.append(report)
                
                return reports
                
        except Exception as e:
            self.logger.error(f"Failed to get installation history: {e}")
            return []


# Utility functions for dashboard and reporting
def create_health_dashboard_html(health_reporter: HealthReporter) -> str:
    """Create HTML dashboard for health monitoring."""
    
    dashboard_data = health_reporter.get_dashboard_data()
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Installation Health Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .metric-card { 
                border: 1px solid #ddd; 
                border-radius: 8px; 
                padding: 15px; 
                margin: 10px; 
                display: inline-block; 
                min-width: 200px;
            }
            .success { background-color: #d4edda; }
            .warning { background-color: #fff3cd; }
            .danger { background-color: #f8d7da; }
            .metric-value { font-size: 2em; font-weight: bold; }
            .metric-label { color: #666; }
            .recommendations { background-color: #e7f3ff; padding: 15px; border-radius: 8px; }
        </style>
    </head>
    <body>
        <h1>Installation Health Dashboard</h1>
        <p>Last updated: {timestamp}</p>
        
        <div class="metrics">
            <div class="metric-card {success_class}">
                <div class="metric-value">{success_rate:.1%}</div>
                <div class="metric-label">Success Rate (7d)</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">{total_installations}</div>
                <div class="metric-label">Total Installations (7d)</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">{avg_time:.0f}s</div>
                <div class="metric-label">Avg Installation Time</div>
            </div>
            
            <div class="metric-card {error_class}">
                <div class="metric-value">{error_patterns}</div>
                <div class="metric-label">Active Error Patterns</div>
            </div>
        </div>
        
        <div class="recommendations">
            <h3>Recommendations</h3>
            <ul>
                {recommendations_html}
            </ul>
        </div>
        
        <div class="recent-installations">
            <h3>Recent Installations</h3>
            <table border="1" style="width: 100%; border-collapse: collapse;">
                <tr>
                    <th>ID</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Health Score</th>
                    <th>Timestamp</th>
                </tr>
                {installations_html}
            </table>
        </div>
    </body>
    </html>
    """
    
    # Determine CSS classes based on metrics
    success_rate = dashboard_data.get('summary', {}).get('success_rate_7d', 0)
    success_class = 'success' if success_rate >= 0.9 else 'warning' if success_rate >= 0.7 else 'danger'
    
    error_patterns = dashboard_data.get('summary', {}).get('active_error_patterns', 0)
    error_class = 'success' if error_patterns <= 2 else 'warning' if error_patterns <= 5 else 'danger'
    
    # Generate recommendations HTML
    recommendations = dashboard_data.get('recommendations', [])
    recommendations_html = '\n'.join([f'<li>{rec}</li>' for rec in recommendations])
    
    # Generate recent installations HTML
    recent_installations = dashboard_data.get('recent_installations', [])
    installations_html = ''
    for installation in recent_installations[:10]:  # Show last 10
        status_class = 'success' if installation['status'] == 'success' else 'danger'
        installations_html += f"""
        <tr class="{status_class}">
            <td>{installation['installation_id'][:8]}...</td>
            <td>{installation['status']}</td>
            <td>{installation['duration_seconds']:.0f}s</td>
            <td>{installation['final_health_score']:.1f}</td>
            <td>{installation['timestamp'][:19]}</td>
        </tr>
        """
    
    return html_template.format(
        timestamp=dashboard_data.get('timestamp', ''),
        success_rate=success_rate,
        success_class=success_class,
        total_installations=dashboard_data.get('summary', {}).get('total_installations_7d', 0),
        avg_time=dashboard_data.get('summary', {}).get('average_installation_time', 0),
        error_patterns=error_patterns,
        error_class=error_class,
        recommendations_html=recommendations_html,
        installations_html=installations_html
    )