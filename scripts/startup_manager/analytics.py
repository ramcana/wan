"""
Usage analytics and optimization system for startup manager.

This module provides anonymous usage analytics to identify common failure patterns,
optimization suggestions based on system configuration and usage patterns,
and performance benchmarking against baseline startup times.
"""

import hashlib
import platform
import psutil
import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
from enum import Enum

from .performance_monitor import PerformanceMonitor, StartupSession, PerformanceStats


class OptimizationCategory(Enum):
    """Categories of optimization suggestions."""
    SYSTEM_RESOURCES = "system_resources"
    CONFIGURATION = "configuration"
    ENVIRONMENT = "environment"
    PROCESS_MANAGEMENT = "process_management"
    NETWORK = "network"


class OptimizationPriority(Enum):
    """Priority levels for optimization suggestions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SystemProfile:
    """Anonymous system profile for analytics."""
    os_type: str
    os_version: str
    cpu_count: int
    cpu_freq_mhz: Optional[float]
    memory_gb: float
    disk_type: str  # "SSD" or "HDD"
    python_version: str
    node_version: Optional[str]
    profile_hash: str  # Anonymous identifier
    
    @classmethod
    def create_anonymous_profile(cls) -> 'SystemProfile':
        """Create anonymous system profile."""
        # Get system information
        os_info = platform.system()
        os_version = platform.version()
        cpu_count = psutil.cpu_count()
        
        # Get CPU frequency (may not be available on all systems)
        cpu_freq = None
        try:
            freq_info = psutil.cpu_freq()
            if freq_info:
                cpu_freq = freq_info.max
        except (AttributeError, OSError):
            pass
        
        # Get memory info
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024 ** 3)
        
        # Detect disk type (simplified heuristic)
        disk_type = "Unknown"
        try:
            # This is a simplified detection - in reality, disk type detection is complex
            disk_usage = psutil.disk_usage('/')
            # Assume SSD if available space operations are fast (this is very simplified)
            disk_type = "SSD"  # Default assumption for modern systems
        except Exception:
            disk_type = "Unknown"
        
        # Get Python version
        python_version = platform.python_version()
        
        # Try to get Node.js version (if available)
        node_version = None
        try:
            import subprocess
            result = subprocess.run(['node', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                node_version = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        # Create anonymous hash
        profile_data = f"{os_info}_{cpu_count}_{memory_gb:.1f}_{python_version}"
        profile_hash = hashlib.sha256(profile_data.encode()).hexdigest()[:16]
        
        return cls(
            os_type=os_info,
            os_version=os_version,
            cpu_count=cpu_count,
            cpu_freq_mhz=cpu_freq,
            memory_gb=memory_gb,
            disk_type=disk_type,
            python_version=python_version,
            node_version=node_version,
            profile_hash=profile_hash
        )


@dataclass
class FailurePattern:
    """Identified failure pattern from analytics."""
    pattern_id: str
    error_type: str
    frequency: int
    affected_systems: List[str]  # System profile hashes
    common_conditions: Dict[str, Any]
    suggested_fixes: List[str]
    confidence_score: float  # 0.0 to 1.0


@dataclass
class OptimizationSuggestion:
    """Performance optimization suggestion."""
    suggestion_id: str
    category: OptimizationCategory
    priority: OptimizationPriority
    title: str
    description: str
    expected_improvement: str
    implementation_steps: List[str]
    applicable_systems: List[str]  # System profile patterns
    confidence_score: float


@dataclass
class BenchmarkResult:
    """Performance benchmark result."""
    benchmark_id: str
    system_profile: SystemProfile
    baseline_duration: float
    current_duration: float
    improvement_percentage: float
    phase_comparisons: Dict[str, Dict[str, float]]
    timestamp: str


@dataclass
class UsageAnalytics:
    """Anonymous usage analytics data."""
    total_sessions: int
    success_rate: float
    average_duration: float
    common_errors: Dict[str, int]
    system_profiles: Dict[str, int]  # profile_hash -> count
    performance_trends: Dict[str, List[float]]  # phase -> durations
    optimization_impact: Dict[str, float]  # suggestion_id -> improvement


class AnalyticsEngine:
    """
    Usage analytics and optimization engine.
    
    Features:
    - Anonymous usage analytics collection
    - Failure pattern identification
    - System-specific optimization suggestions
    - Performance benchmarking against baselines
    - Trend analysis and recommendations
    """
    
    def __init__(
        self,
        performance_monitor: PerformanceMonitor,
        data_dir: Union[str, Path] = "logs/analytics",
        enable_analytics: bool = True
    ):
        """
        Initialize analytics engine.
        
        Args:
            performance_monitor: Performance monitor instance
            data_dir: Directory to store analytics data
            enable_analytics: Whether to enable analytics collection
        """
        self.performance_monitor = performance_monitor
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.enable_analytics = enable_analytics
        
        # System profile
        self.system_profile = SystemProfile.create_anonymous_profile()
        
        # Analytics data
        self.failure_patterns: Dict[str, FailurePattern] = {}
        self.optimization_suggestions: Dict[str, OptimizationSuggestion] = {}
        self.benchmark_results: List[BenchmarkResult] = []
        
        # Load existing data
        self._load_analytics_data()
        
        # Initialize baseline optimizations
        self._initialize_baseline_optimizations()
    
    def collect_session_analytics(self, session: StartupSession):
        """
        Collect analytics from a startup session.
        
        Args:
            session: Completed startup session
        """
        if not self.enable_analytics:
            return
        
        # Analyze errors for patterns
        self._analyze_error_patterns(session)
        
        # Update system performance data
        self._update_performance_data(session)
        
        # Generate optimization suggestions
        self._generate_optimization_suggestions(session)
        
        # Save analytics data
        self._save_analytics_data()
    
    def get_optimization_suggestions(
        self,
        max_suggestions: int = 5,
        min_priority: OptimizationPriority = OptimizationPriority.LOW
    ) -> List[OptimizationSuggestion]:
        """
        Get optimization suggestions for the current system.
        
        Args:
            max_suggestions: Maximum number of suggestions to return
            min_priority: Minimum priority level to include
            
        Returns:
            List of optimization suggestions
        """
        # Filter suggestions by priority and system compatibility
        priority_order = {
            OptimizationPriority.CRITICAL: 4,
            OptimizationPriority.HIGH: 3,
            OptimizationPriority.MEDIUM: 2,
            OptimizationPriority.LOW: 1
        }
        
        min_priority_value = priority_order[min_priority]
        
        applicable_suggestions = []
        for suggestion in self.optimization_suggestions.values():
            # Check priority
            if priority_order[suggestion.priority] < min_priority_value:
                continue
            
            # Check system compatibility
            if self._is_suggestion_applicable(suggestion):
                applicable_suggestions.append(suggestion)
        
        # Sort by priority and confidence
        applicable_suggestions.sort(
            key=lambda s: (priority_order[s.priority], s.confidence_score),
            reverse=True
        )
        
        return applicable_suggestions[:max_suggestions]
    
    def run_performance_benchmark(self) -> BenchmarkResult:
        """
        Run performance benchmark against baseline.
        
        Returns:
            Benchmark result
        """
        # Get recent performance stats
        stats = self.performance_monitor.get_performance_stats(days=7)
        
        if stats.total_sessions == 0:
            # No data available for benchmarking
            return BenchmarkResult(
                benchmark_id=f"benchmark_{int(datetime.now().timestamp())}",
                system_profile=self.system_profile,
                baseline_duration=0.0,
                current_duration=0.0,
                improvement_percentage=0.0,
                phase_comparisons={},
                timestamp=datetime.now().isoformat()
            )
        
        # Get baseline duration for similar systems
        baseline_duration = self._get_baseline_duration()
        current_duration = stats.average_duration
        
        # Calculate improvement percentage
        if baseline_duration > 0:
            improvement_percentage = ((baseline_duration - current_duration) / baseline_duration) * 100
        else:
            improvement_percentage = 0.0
        
        # Compare phase durations
        phase_comparisons = {}
        baseline_phases = self._get_baseline_phase_durations()
        
        for phase, current_avg in stats.phase_averages.items():
            baseline_avg = baseline_phases.get(phase, current_avg)
            phase_improvement = 0.0
            if baseline_avg > 0:
                phase_improvement = ((baseline_avg - current_avg) / baseline_avg) * 100
            
            phase_comparisons[phase] = {
                "baseline": baseline_avg,
                "current": current_avg,
                "improvement_percentage": phase_improvement
            }
        
        benchmark_result = BenchmarkResult(
            benchmark_id=f"benchmark_{int(datetime.now().timestamp())}",
            system_profile=self.system_profile,
            baseline_duration=baseline_duration,
            current_duration=current_duration,
            improvement_percentage=improvement_percentage,
            phase_comparisons=phase_comparisons,
            timestamp=datetime.now().isoformat()
        )
        
        # Store benchmark result
        self.benchmark_results.append(benchmark_result)
        
        # Keep only recent benchmarks
        cutoff_time = datetime.now() - timedelta(days=30)
        self.benchmark_results = [
            result for result in self.benchmark_results
            if datetime.fromisoformat(result.timestamp) > cutoff_time
        ]
        
        return benchmark_result
    
    def get_usage_analytics(self) -> UsageAnalytics:
        """
        Get anonymous usage analytics summary.
        
        Returns:
            Usage analytics data
        """
        # Get performance stats
        stats = self.performance_monitor.get_performance_stats(days=30)
        
        # Analyze error patterns
        common_errors = {}
        for pattern in self.failure_patterns.values():
            common_errors[pattern.error_type] = pattern.frequency
        
        # System profile distribution (anonymized)
        system_profiles = {self.system_profile.profile_hash: 1}
        
        # Performance trends by phase
        performance_trends = {}
        sessions = list(self.performance_monitor.sessions)
        
        for phase in ["environment_validation", "port_management", "process_startup", "health_verification"]:
            phase_durations = []
            for session in sessions[-20:]:  # Last 20 sessions
                if phase in session.phase_timings:
                    timing = session.phase_timings[phase]
                    if timing.duration is not None:
                        phase_durations.append(timing.duration)
            performance_trends[phase] = phase_durations
        
        # Optimization impact (placeholder - would be measured over time)
        optimization_impact = {}
        for suggestion in self.optimization_suggestions.values():
            # Estimate impact based on confidence and priority
            priority_multiplier = {
                OptimizationPriority.CRITICAL: 0.3,
                OptimizationPriority.HIGH: 0.2,
                OptimizationPriority.MEDIUM: 0.1,
                OptimizationPriority.LOW: 0.05
            }
            estimated_impact = suggestion.confidence_score * priority_multiplier[suggestion.priority]
            optimization_impact[suggestion.suggestion_id] = estimated_impact
        
        return UsageAnalytics(
            total_sessions=stats.total_sessions,
            success_rate=stats.success_rate,
            average_duration=stats.average_duration,
            common_errors=common_errors,
            system_profiles=system_profiles,
            performance_trends=performance_trends,
            optimization_impact=optimization_impact
        )
    
    def _analyze_error_patterns(self, session: StartupSession):
        """Analyze session errors for patterns."""
        if not session.errors:
            return
        
        for error in session.errors:
            # Extract error type
            error_type = error.split(":")[0] if ":" in error else error
            
            # Create or update failure pattern
            pattern_id = hashlib.sha256(error_type.encode()).hexdigest()[:12]
            
            if pattern_id in self.failure_patterns:
                pattern = self.failure_patterns[pattern_id]
                pattern.frequency += 1
                if self.system_profile.profile_hash not in pattern.affected_systems:
                    pattern.affected_systems.append(self.system_profile.profile_hash)
            else:
                # Create new failure pattern
                suggested_fixes = self._generate_error_fixes(error_type)
                
                pattern = FailurePattern(
                    pattern_id=pattern_id,
                    error_type=error_type,
                    frequency=1,
                    affected_systems=[self.system_profile.profile_hash],
                    common_conditions=self._extract_error_conditions(session),
                    suggested_fixes=suggested_fixes,
                    confidence_score=0.5  # Initial confidence
                )
                
                self.failure_patterns[pattern_id] = pattern
    
    def _update_performance_data(self, session: StartupSession):
        """Update performance data from session."""
        # This would update internal performance tracking
        # For now, we rely on the performance monitor's data
        pass
    
    def _generate_optimization_suggestions(self, session: StartupSession):
        """Generate optimization suggestions based on session data."""
        # Analyze session for optimization opportunities
        
        # Check for slow phases
        for phase_name, timing in session.phase_timings.items():
            if timing.duration and timing.duration > 2.0:  # Slow phase threshold
                self._suggest_phase_optimization(phase_name, timing.duration)
        
        # Check resource usage
        if session.resource_snapshots:
            avg_cpu = statistics.mean([s.cpu_percent for s in session.resource_snapshots])
            avg_memory = statistics.mean([s.memory_percent for s in session.resource_snapshots])
            
            if avg_cpu > 80:
                self._suggest_cpu_optimization(avg_cpu)
            
            if avg_memory > 85:
                self._suggest_memory_optimization(avg_memory)
        
        # Check for repeated errors
        error_counts = Counter(session.errors)
        for error, count in error_counts.items():
            if count > 1:
                self._suggest_error_prevention(error, count)
    
    def _suggest_phase_optimization(self, phase_name: str, duration: float):
        """Suggest optimization for slow startup phase."""
        suggestion_id = f"optimize_{phase_name}"
        
        if suggestion_id in self.optimization_suggestions:
            return  # Already exists
        
        phase_optimizations = {
            "environment_validation": {
                "title": "Optimize Environment Validation",
                "description": f"Environment validation is taking {duration:.2f}s. Consider caching validation results.",
                "steps": [
                    "Enable dependency caching",
                    "Skip redundant version checks",
                    "Use faster validation methods"
                ],
                "improvement": "30-50% faster validation"
            },
            "port_management": {
                "title": "Optimize Port Management",
                "description": f"Port management is taking {duration:.2f}s. Consider using faster port detection.",
                "steps": [
                    "Use parallel port checking",
                    "Cache port availability results",
                    "Optimize port conflict resolution"
                ],
                "improvement": "20-40% faster port allocation"
            },
            "process_startup": {
                "title": "Optimize Process Startup",
                "description": f"Process startup is taking {duration:.2f}s. Consider process optimization.",
                "steps": [
                    "Use process pooling",
                    "Optimize startup scripts",
                    "Reduce initialization overhead"
                ],
                "improvement": "15-30% faster process startup"
            }
        }
        
        if phase_name in phase_optimizations:
            opt = phase_optimizations[phase_name]
            
            suggestion = OptimizationSuggestion(
                suggestion_id=suggestion_id,
                category=OptimizationCategory.PROCESS_MANAGEMENT,
                priority=OptimizationPriority.MEDIUM if duration > 3.0 else OptimizationPriority.LOW,
                title=opt["title"],
                description=opt["description"],
                expected_improvement=opt["improvement"],
                implementation_steps=opt["steps"],
                applicable_systems=[self.system_profile.profile_hash],
                confidence_score=0.7
            )
            
            self.optimization_suggestions[suggestion_id] = suggestion
    
    def _suggest_cpu_optimization(self, avg_cpu: float):
        """Suggest CPU optimization."""
        suggestion_id = "optimize_cpu_usage"
        
        if suggestion_id in self.optimization_suggestions:
            return
        
        suggestion = OptimizationSuggestion(
            suggestion_id=suggestion_id,
            category=OptimizationCategory.SYSTEM_RESOURCES,
            priority=OptimizationPriority.HIGH if avg_cpu > 90 else OptimizationPriority.MEDIUM,
            title="Reduce CPU Usage During Startup",
            description=f"High CPU usage detected ({avg_cpu:.1f}%). Consider reducing concurrent operations.",
            expected_improvement="10-25% faster startup",
            implementation_steps=[
                "Close unnecessary applications before startup",
                "Reduce parallel operations",
                "Use CPU affinity settings",
                "Consider upgrading hardware"
            ],
            applicable_systems=[self.system_profile.profile_hash],
            confidence_score=0.8
        )
        
        self.optimization_suggestions[suggestion_id] = suggestion
    
    def _suggest_memory_optimization(self, avg_memory: float):
        """Suggest memory optimization."""
        suggestion_id = "optimize_memory_usage"
        
        if suggestion_id in self.optimization_suggestions:
            return
        
        suggestion = OptimizationSuggestion(
            suggestion_id=suggestion_id,
            category=OptimizationCategory.SYSTEM_RESOURCES,
            priority=OptimizationPriority.HIGH if avg_memory > 95 else OptimizationPriority.MEDIUM,
            title="Reduce Memory Usage During Startup",
            description=f"High memory usage detected ({avg_memory:.1f}%). Consider memory optimization.",
            expected_improvement="15-30% faster startup",
            implementation_steps=[
                "Close memory-intensive applications",
                "Increase virtual memory",
                "Use memory-efficient startup options",
                "Consider adding more RAM"
            ],
            applicable_systems=[self.system_profile.profile_hash],
            confidence_score=0.8
        )
        
        self.optimization_suggestions[suggestion_id] = suggestion
    
    def _suggest_error_prevention(self, error: str, count: int):
        """Suggest error prevention measures."""
        suggestion_id = f"prevent_{hashlib.sha256(error.encode()).hexdigest()[:8]}"
        
        if suggestion_id in self.optimization_suggestions:
            return
        
        suggestion = OptimizationSuggestion(
            suggestion_id=suggestion_id,
            category=OptimizationCategory.CONFIGURATION,
            priority=OptimizationPriority.HIGH if count > 3 else OptimizationPriority.MEDIUM,
            title=f"Prevent Recurring Error",
            description=f"Error '{error}' occurred {count} times. Consider preventive measures.",
            expected_improvement="Eliminate startup failures",
            implementation_steps=[
                "Review error logs for root cause",
                "Update configuration to prevent error",
                "Add error prevention checks",
                "Consider environment changes"
            ],
            applicable_systems=[self.system_profile.profile_hash],
            confidence_score=0.6
        )
        
        self.optimization_suggestions[suggestion_id] = suggestion
    
    def _initialize_baseline_optimizations(self):
        """Initialize baseline optimization suggestions."""
        # System-specific optimizations
        if self.system_profile.memory_gb < 8:
            self._add_low_memory_optimizations()
        
        if self.system_profile.cpu_count < 4:
            self._add_low_cpu_optimizations()
        
        if self.system_profile.os_type == "Windows":
            self._add_windows_optimizations()
    
    def _add_low_memory_optimizations(self):
        """Add optimizations for low-memory systems."""
        suggestion = OptimizationSuggestion(
            suggestion_id="low_memory_optimization",
            category=OptimizationCategory.SYSTEM_RESOURCES,
            priority=OptimizationPriority.MEDIUM,
            title="Low Memory System Optimization",
            description=f"System has {self.system_profile.memory_gb:.1f}GB RAM. Consider memory optimizations.",
            expected_improvement="20-40% better performance",
            implementation_steps=[
                "Enable memory-efficient startup mode",
                "Reduce concurrent processes",
                "Use swap file optimization",
                "Close unnecessary applications"
            ],
            applicable_systems=[f"memory_lt_8gb"],
            confidence_score=0.8
        )
        
        self.optimization_suggestions[suggestion.suggestion_id] = suggestion
    
    def _add_low_cpu_optimizations(self):
        """Add optimizations for low-CPU systems."""
        suggestion = OptimizationSuggestion(
            suggestion_id="low_cpu_optimization",
            category=OptimizationCategory.SYSTEM_RESOURCES,
            priority=OptimizationPriority.MEDIUM,
            title="Low CPU System Optimization",
            description=f"System has {self.system_profile.cpu_count} CPU cores. Consider CPU optimizations.",
            expected_improvement="15-30% better performance",
            implementation_steps=[
                "Reduce parallel operations",
                "Use sequential startup mode",
                "Optimize process priorities",
                "Consider CPU upgrade"
            ],
            applicable_systems=[f"cpu_lt_4cores"],
            confidence_score=0.7
        )
        
        self.optimization_suggestions[suggestion.suggestion_id] = suggestion
    
    def _add_windows_optimizations(self):
        """Add Windows-specific optimizations."""
        suggestion = OptimizationSuggestion(
            suggestion_id="windows_optimization",
            category=OptimizationCategory.CONFIGURATION,
            priority=OptimizationPriority.LOW,
            title="Windows System Optimization",
            description="Windows-specific optimizations for better startup performance.",
            expected_improvement="10-20% better performance",
            implementation_steps=[
                "Disable Windows Defender real-time scanning for project folder",
                "Add firewall exceptions for development ports",
                "Use Windows performance mode",
                "Optimize Windows startup programs"
            ],
            applicable_systems=["windows"],
            confidence_score=0.6
        )
        
        self.optimization_suggestions[suggestion.suggestion_id] = suggestion
    
    def _is_suggestion_applicable(self, suggestion: OptimizationSuggestion) -> bool:
        """Check if suggestion is applicable to current system."""
        # Check system profile compatibility
        for pattern in suggestion.applicable_systems:
            if pattern == self.system_profile.profile_hash:
                return True
            elif pattern == "windows" and self.system_profile.os_type == "Windows":
                return True
            elif pattern == "memory_lt_8gb" and self.system_profile.memory_gb < 8:
                return True
            elif pattern == "cpu_lt_4cores" and self.system_profile.cpu_count < 4:
                return True
        
        return False
    
    def _get_baseline_duration(self) -> float:
        """Get baseline startup duration for similar systems."""
        # This would ideally come from a database of system performance
        # For now, use heuristics based on system specs
        
        base_duration = 3.0  # Base startup time in seconds
        
        # Adjust based on system specs
        if self.system_profile.memory_gb < 4:
            base_duration += 1.0
        elif self.system_profile.memory_gb > 16:
            base_duration -= 0.5
        
        if self.system_profile.cpu_count < 4:
            base_duration += 0.5
        elif self.system_profile.cpu_count > 8:
            base_duration -= 0.3
        
        if self.system_profile.disk_type == "SSD":
            base_duration -= 0.5
        
        return max(base_duration, 1.0)  # Minimum 1 second
    
    def _get_baseline_phase_durations(self) -> Dict[str, float]:
        """Get baseline phase durations."""
        baseline_total = self._get_baseline_duration()
        
        # Typical phase distribution
        return {
            "environment_validation": baseline_total * 0.3,
            "port_management": baseline_total * 0.2,
            "process_startup": baseline_total * 0.4,
            "health_verification": baseline_total * 0.1
        }
    
    def _generate_error_fixes(self, error_type: str) -> List[str]:
        """Generate suggested fixes for error type."""
        error_fixes = {
            "Port conflict": [
                "Kill processes using the port",
                "Use alternative port ranges",
                "Configure automatic port allocation"
            ],
            "Permission denied": [
                "Run as administrator",
                "Add firewall exceptions",
                "Check file permissions"
            ],
            "Module not found": [
                "Install missing dependencies",
                "Activate virtual environment",
                "Check Python path"
            ],
            "Connection refused": [
                "Check network connectivity",
                "Verify server is running",
                "Check firewall settings"
            ]
        }
        
        # Find matching error type
        for pattern, fixes in error_fixes.items():
            if pattern.lower() in error_type.lower():
                return fixes
        
        # Default fixes
        return [
            "Check error logs for details",
            "Restart the application",
            "Verify system configuration"
        ]
    
    def _extract_error_conditions(self, session: StartupSession) -> Dict[str, Any]:
        """Extract conditions that may have contributed to errors."""
        conditions = {}
        
        # System resource conditions
        if session.resource_snapshots:
            avg_cpu = statistics.mean([s.cpu_percent for s in session.resource_snapshots])
            avg_memory = statistics.mean([s.memory_percent for s in session.resource_snapshots])
            
            conditions["high_cpu"] = avg_cpu > 80
            conditions["high_memory"] = avg_memory > 85
        
        # Timing conditions
        if session.total_duration:
            conditions["slow_startup"] = session.total_duration > 5.0
        
        # Phase conditions
        for phase_name, timing in session.phase_timings.items():
            if timing.duration and timing.duration > 2.0:
                conditions[f"slow_{phase_name}"] = True
        
        return conditions
    
    def _load_analytics_data(self):
        """Load analytics data from disk."""
        data_file = self.data_dir / "analytics_data.json"
        
        if not data_file.exists():
            return
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load failure patterns
            for pattern_data in data.get('failure_patterns', []):
                pattern = FailurePattern(**pattern_data)
                self.failure_patterns[pattern.pattern_id] = pattern
            
            # Load optimization suggestions
            for suggestion_data in data.get('optimization_suggestions', []):
                # Convert enum strings back to enums
                suggestion_data['category'] = OptimizationCategory(suggestion_data['category'])
                suggestion_data['priority'] = OptimizationPriority(suggestion_data['priority'])
                
                suggestion = OptimizationSuggestion(**suggestion_data)
                self.optimization_suggestions[suggestion.suggestion_id] = suggestion
            
            # Load benchmark results
            for benchmark_data in data.get('benchmark_results', []):
                # Reconstruct system profile
                profile_data = benchmark_data.pop('system_profile')
                system_profile = SystemProfile(**profile_data)
                
                benchmark = BenchmarkResult(
                    system_profile=system_profile,
                    **benchmark_data
                )
                self.benchmark_results.append(benchmark)
        
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Ignore corrupted data
            pass
    
    def _save_analytics_data(self):
        """Save analytics data to disk."""
        data_file = self.data_dir / "analytics_data.json"
        
        try:
            data = {
                'failure_patterns': [asdict(pattern) for pattern in self.failure_patterns.values()],
                'optimization_suggestions': [
                    {
                        **asdict(suggestion),
                        'category': suggestion.category.value,
                        'priority': suggestion.priority.value
                    }
                    for suggestion in self.optimization_suggestions.values()
                ],
                'benchmark_results': [asdict(result) for result in self.benchmark_results[-10:]],  # Keep last 10
                'last_updated': datetime.now().isoformat()
            }
            
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        
        except Exception:
            # Ignore save errors
            pass


# Global analytics engine instance
_global_analytics: Optional[AnalyticsEngine] = None


def get_analytics_engine(performance_monitor: PerformanceMonitor, **kwargs) -> AnalyticsEngine:
    """
    Get or create global analytics engine instance.
    
    Args:
        performance_monitor: Performance monitor instance
        **kwargs: Additional arguments for analytics engine initialization
    
    Returns:
        AnalyticsEngine instance
    """
    global _global_analytics
    
    if _global_analytics is None:
        _global_analytics = AnalyticsEngine(performance_monitor, **kwargs)
    
    return _global_analytics