# Startup Manager package for WAN22 server management
from .config import StartupConfig, BackendConfig, FrontendConfig
from .utils import SystemDetector, PathManager
from .environment_validator import (
    EnvironmentValidator, 
    DependencyValidator, 
    ConfigurationValidator,
    ValidationIssue,
    ValidationResult,
    ValidationStatus
)
from .process_manager import (
    ProcessManager,
    ProcessInfo,
    ProcessResult,
    ProcessStatus,
    HealthMonitor
)
from .port_manager import PortManager, PortStatus, PortAllocation
from .recovery_engine import RecoveryEngine, ErrorType, StartupError
from .performance_monitor import (
    PerformanceMonitor,
    TimingMetric,
    ResourceSnapshot,
    StartupSession,
    PerformanceStats,
    StartupPhase,
    MetricType,
    get_performance_monitor
)
from .analytics import (
    AnalyticsEngine,
    SystemProfile,
    FailurePattern,
    OptimizationSuggestion,
    BenchmarkResult,
    UsageAnalytics,
    OptimizationCategory,
    OptimizationPriority,
    get_analytics_engine
)

__version__ = "1.0.0"
__all__ = [
    "StartupConfig", "BackendConfig", "FrontendConfig", 
    "SystemDetector", "PathManager",
    "EnvironmentValidator", "DependencyValidator", "ConfigurationValidator",
    "ValidationIssue", "ValidationResult", "ValidationStatus",
    "ProcessManager", "ProcessInfo", "ProcessResult", "ProcessStatus", "HealthMonitor",
    "PortManager", "PortStatus", "PortAllocation",
    "RecoveryEngine", "ErrorType", "StartupError",
    "PerformanceMonitor", "TimingMetric", "ResourceSnapshot", "StartupSession",
    "PerformanceStats", "StartupPhase", "MetricType", "get_performance_monitor",
    "AnalyticsEngine", "SystemProfile", "FailurePattern", "OptimizationSuggestion",
    "BenchmarkResult", "UsageAnalytics", "OptimizationCategory", "OptimizationPriority",
    "get_analytics_engine"
]
