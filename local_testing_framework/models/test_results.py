"""
Data models for test results and validation outcomes
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional


class ValidationStatus(Enum):
    """Status of validation checks"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class TestStatus(Enum):
    """Overall test execution status"""
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class ValidationResult:
    """Result of a single validation check"""
    component: str
    status: ValidationStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    remediation_steps: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "remediation_steps": self.remediation_steps,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class EnvironmentValidationResults:
    """Results from environment validation"""
    python_version: ValidationResult
    cuda_availability: ValidationResult
    dependencies: ValidationResult
    configuration: ValidationResult
    environment_variables: ValidationResult
    overall_status: ValidationStatus
    remediation_steps: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "python_version": self.python_version.to_dict(),
            "cuda_availability": self.cuda_availability.to_dict(),
            "dependencies": self.dependencies.to_dict(),
            "configuration": self.configuration.to_dict(),
            "environment_variables": self.environment_variables.to_dict(),
            "overall_status": self.overall_status.value,
            "remediation_steps": self.remediation_steps,
            "timestamp": self.timestamp.isoformat()
        }

    def get_failed_validations(self) -> List[ValidationResult]:
        """Get list of failed validation results"""
        failed = []
        for result in [self.python_version, self.cuda_availability, self.dependencies, 
                      self.configuration, self.environment_variables]:
            if result.status == ValidationStatus.FAILED:
                failed.append(result)
        return failed


@dataclass
class ResourceMetrics:
    """Resource usage metrics snapshot"""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpu_percent: float
    vram_used_mb: int
    vram_total_mb: int
    vram_percent: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_gb": self.memory_used_gb,
            "memory_total_gb": self.memory_total_gb,
            "gpu_percent": self.gpu_percent,
            "vram_used_mb": self.vram_used_mb,
            "vram_total_mb": self.vram_total_mb,
            "vram_percent": self.vram_percent,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class IntegrationTestResults:
    """Results from integration testing"""
    start_time: datetime
    end_time: datetime
    generation_results: List[Any] = field(default_factory=list)
    error_handling_result: Optional[ValidationResult] = None
    ui_results: Optional['UITestResults'] = None
    api_results: Optional['APITestResults'] = None
    resource_monitoring_result: Optional[ValidationResult] = None
    overall_status: TestStatus = TestStatus.ERROR

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "generation_results": [r.__dict__ if hasattr(r, '__dict__') else str(r) for r in self.generation_results],
            "overall_status": self.overall_status.value
        }
        
        if self.error_handling_result:
            result["error_handling_result"] = self.error_handling_result.to_dict()
        if self.ui_results:
            result["ui_results"] = self.ui_results.to_dict()
        if self.api_results:
            result["api_results"] = self.api_results.to_dict()
        if self.resource_monitoring_result:
            result["resource_monitoring_result"] = self.resource_monitoring_result.to_dict()
            
        return result


@dataclass
class UITestResults:
    """Results from UI testing"""
    overall_status: TestStatus
    browser_access_result: ValidationResult
    component_test_results: List[ValidationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "overall_status": self.overall_status.value,
            "browser_access_result": self.browser_access_result.to_dict(),
            "component_test_results": [result.to_dict() for result in self.component_test_results]
        }


@dataclass
class APITestResults:
    """Results from API testing"""
    overall_status: TestStatus
    endpoint_test_results: List[ValidationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "overall_status": self.overall_status.value,
            "endpoint_test_results": [result.to_dict() for result in self.endpoint_test_results]
        }


@dataclass
class BenchmarkResult:
    """Individual benchmark result"""
    resolution: str
    generation_time: float
    target_time: float
    meets_target: bool
    vram_usage: float
    cpu_usage: float
    memory_usage: float
    optimization_level: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "resolution": self.resolution,
            "generation_time": self.generation_time,
            "target_time": self.target_time,
            "meets_target": self.meets_target,
            "vram_usage": self.vram_usage,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "optimization_level": self.optimization_level,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class OptimizationResult:
    """VRAM optimization test result"""
    baseline_vram_mb: int
    optimized_vram_mb: int
    reduction_percent: float
    target_reduction_percent: float
    meets_target: bool
    optimizations_applied: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "baseline_vram_mb": self.baseline_vram_mb,
            "optimized_vram_mb": self.optimized_vram_mb,
            "reduction_percent": self.reduction_percent,
            "target_reduction_percent": self.target_reduction_percent,
            "meets_target": self.meets_target,
            "optimizations_applied": self.optimizations_applied,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PerformanceTestResults:
    """Performance test results"""
    benchmark_720p: Optional[BenchmarkResult] = None
    benchmark_1080p: Optional[BenchmarkResult] = None
    vram_optimization: Optional[OptimizationResult] = None
    overall_status: TestStatus = TestStatus.ERROR
    recommendations: List[str] = field(default_factory=list)
    error_logs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "overall_status": self.overall_status.value,
            "recommendations": self.recommendations,
            "error_logs": self.error_logs
        }
        
        if self.benchmark_720p:
            result["benchmark_720p"] = self.benchmark_720p.to_dict()
        if self.benchmark_1080p:
            result["benchmark_1080p"] = self.benchmark_1080p.to_dict()
        if self.vram_optimization:
            result["vram_optimization"] = self.vram_optimization.to_dict()
            
        return result

    def get_failed_benchmarks(self) -> List[BenchmarkResult]:
        """Get list of failed benchmark results"""
        failed = []
        if self.benchmark_720p and not self.benchmark_720p.meets_target:
            failed.append(self.benchmark_720p)
        if self.benchmark_1080p and not self.benchmark_1080p.meets_target:
            failed.append(self.benchmark_1080p)
        return failed


@dataclass
class TestResults:
    """Comprehensive test results container"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    environment_results: Optional[EnvironmentValidationResults] = None
    performance_results: Optional[PerformanceTestResults] = None
    integration_results: Optional[IntegrationTestResults] = None
    overall_status: TestStatus = TestStatus.ERROR
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "overall_status": self.overall_status.value,
            "recommendations": self.recommendations
        }
        
        if self.end_time:
            result["end_time"] = self.end_time.isoformat()
            
        if self.environment_results:
            result["environment_results"] = self.environment_results.to_dict()
            
        if self.performance_results:
            result["performance_results"] = self.performance_results.to_dict()
            
        if self.integration_results:
            result["integration_results"] = self.integration_results.to_dict()
            
        return result