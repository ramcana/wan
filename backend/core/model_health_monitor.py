"""
Model Health Monitor
Provides integrity checking, performance monitoring, corruption detection,
and automated health checks for WAN2.2 models.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
import threading
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import os
import psutil

logger = logging.getLogger(__name__)

# Import performance monitoring (with fallback if not available)
try:
    from .performance_monitoring_system import get_performance_monitor
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False
    logger.warning("Performance monitoring not available")


class HealthStatus(Enum):
    """Model health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CORRUPTED = "corrupted"
    MISSING = "missing"
    UNKNOWN = "unknown"


class CorruptionType(Enum):
    """Types of corruption that can be detected"""
    FILE_MISSING = "file_missing"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    INCOMPLETE_DOWNLOAD = "incomplete_download"
    INVALID_FORMAT = "invalid_format"
    PERMISSION_ERROR = "permission_error"
    DISK_ERROR = "disk_error"


@dataclass
class IntegrityResult:
    """Result of model integrity check"""
    model_id: str
    is_healthy: bool
    health_status: HealthStatus
    integrity_score: float = 1.0
    issues: List[str] = field(default_factory=list)
    corruption_types: List[CorruptionType] = field(default_factory=list)
    last_check: Optional[datetime] = None
    repair_suggestions: List[str] = field(default_factory=list)
    can_auto_repair: bool = False
    issues: List[str] = field(default_factory=list)
    corruption_types: List[CorruptionType] = field(default_factory=list)
    file_count: int = 0
    total_size_mb: float = 0.0
    checksum: Optional[str] = None
    last_checked: Optional[datetime] = None
    repair_suggestions: List[str] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Performance metrics for model operations"""
    model_id: str
    load_time_seconds: float
    generation_time_seconds: float
    memory_usage_mb: float
    vram_usage_mb: float
    cpu_usage_percent: float
    throughput_fps: float
    quality_score: Optional[float] = None
    error_rate: float = 0.0
    timestamp: Optional[datetime] = None


@dataclass
class PerformanceHealth:
    """Health assessment based on performance metrics"""
    model_id: str
    overall_score: float  # 0.0 to 1.0
    performance_trend: str  # "improving", "stable", "degrading"
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    baseline_comparison: Dict[str, float] = field(default_factory=dict)
    last_assessment: Optional[datetime] = None


@dataclass
class CorruptionReport:
    """Detailed corruption analysis report"""
    model_id: str
    corruption_detected: bool
    corruption_types: List[CorruptionType] = field(default_factory=list)
    affected_files: List[str] = field(default_factory=list)
    severity: str = "low"  # "low", "medium", "high", "critical"
    repair_possible: bool = True
    repair_actions: List[str] = field(default_factory=list)
    detection_time: Optional[datetime] = None


@dataclass
class SystemHealthReport:
    """Overall system health report"""
    overall_health_score: float  # 0.0 to 1.0
    models_healthy: int
    models_degraded: int
    models_corrupted: int
    models_missing: int
    storage_usage_percent: float
    recommendations: List[str] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    detailed_reports: Dict[str, IntegrityResult] = field(default_factory=dict)


@dataclass
class HealthCheckConfig:
    """Configuration for health monitoring"""
    check_interval_hours: int = 24
    performance_monitoring_enabled: bool = True
    automatic_repair_enabled: bool = True
    corruption_detection_enabled: bool = True
    baseline_performance_days: int = 7
    performance_degradation_threshold: float = 0.2  # 20% degradation triggers alert
    storage_warning_threshold: float = 0.8  # 80% storage usage warning
    storage_critical_threshold: float = 0.95  # 95% storage usage critical


class ModelHealthMonitor:
    """
    Comprehensive model health monitoring system with integrity checking,
    performance monitoring, corruption detection, and automated health checks.
    """
    
    def __init__(self, models_dir: Optional[str] = None, config: Optional[HealthCheckConfig] = None):
        """
        Initialize the model health monitor.
        
        Args:
            models_dir: Directory containing models
            config: Health check configuration
        """
        self.models_dir = Path(models_dir) if models_dir else Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or HealthCheckConfig()
        
        # Health data storage
        self.health_data_dir = self.models_dir / ".health"
        self.health_data_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self.performance_baselines: Dict[str, PerformanceMetrics] = {}
        
        # Health check scheduling
        self._health_check_task: Optional[asyncio.Task] = None
        self._monitoring_active = False
        
        # Callbacks for health events
        self._health_callbacks: List[Callable] = []
        self._corruption_callbacks: List[Callable] = []
        
        # Thread pool for intensive operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="health_monitor")
        
        logger.info(f"Model Health Monitor initialized with models_dir: {self.models_dir}")
    
    async def initialize(self) -> bool:
        """Initialize the model health monitor"""
        try:
            # Ensure directories exist
            self.health_data_dir.mkdir(exist_ok=True)
            
            # Start monitoring if configured
            if self.config.check_interval_hours > 0:
                await self._start_monitoring()
            
            logger.info("Model Health Monitor initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Model Health Monitor: {e}")
            return False
    
    async def _start_monitoring(self):
        """Start background health monitoring"""
        try:
            self._monitoring_active = True
            logger.info("Health monitoring started")
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
    
    def add_health_callback(self, callback: Callable[[IntegrityResult], None]):
        """Add a callback for health check results"""
        self._health_callbacks.append(callback)
    
    def add_corruption_callback(self, callback: Callable[[CorruptionReport], None]):
        """Add a callback for corruption detection"""
        self._corruption_callbacks.append(callback)
    
    async def _notify_health_callbacks(self, result: IntegrityResult):
        """Notify all health callbacks"""
        for callback in self._health_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.warning(f"Health callback error: {e}")
    
    async def _notify_corruption_callbacks(self, report: CorruptionReport):
        """Notify all corruption callbacks"""
        for callback in self._corruption_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(report)
                else:
                    callback(report)
            except Exception as e:
                logger.warning(f"Corruption callback error: {e}")
    
    async def check_model_integrity(self, model_id: str) -> IntegrityResult:
        """
        Perform comprehensive integrity check on a model.
        
        Args:
            model_id: Model identifier to check
            
        Returns:
            IntegrityResult with detailed integrity information
        """
        logger.info(f"Starting integrity check for model: {model_id}")
        start_time = time.time()
        
        # Start performance monitoring
        performance_id = None
        if PERFORMANCE_MONITORING_AVAILABLE:
            try:
                monitor = get_performance_monitor()
                performance_id = monitor.track_health_check(
                    f"integrity_check_{model_id}",
                    {
                        "model_id": model_id,
                        "check_type": "full_integrity",
                        "check_level": "comprehensive"
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to start performance monitoring: {e}")
        
        result = IntegrityResult(
            model_id=model_id,
            is_healthy=False,
            health_status=HealthStatus.UNKNOWN,
            last_checked=datetime.now()
        )
        
        try:
            # Check if model directory exists
            model_path = self._get_model_path(model_id)
            if not model_path.exists():
                result.health_status = HealthStatus.MISSING
                result.issues.append("Model directory not found")
                result.repair_suggestions.append("Re-download the model")
                return result
            
            # Check essential files
            essential_files = await self._get_essential_files(model_id)
            missing_files = []
            
            for file_name in essential_files:
                file_path = model_path / file_name
                if not file_path.exists():
                    missing_files.append(file_name)
                    result.issues.append(f"Missing essential file: {file_name}")
                    result.corruption_types.append(CorruptionType.FILE_MISSING)
            
            if missing_files:
                result.health_status = HealthStatus.CORRUPTED
                result.repair_suggestions.extend([
                    "Re-download missing files",
                    "Verify download completion",
                    "Check available disk space"
                ])
            
            # Calculate total size and file count
            total_size = 0
            file_count = 0
            
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    file_count += 1
                    try:
                        total_size += file_path.stat().st_size
                    except (OSError, PermissionError) as e:
                        result.issues.append(f"Cannot access file {file_path.name}: {e}")
                        result.corruption_types.append(CorruptionType.PERMISSION_ERROR)
            
            result.file_count = file_count
            result.total_size_mb = total_size / (1024 * 1024)
            
            # Perform checksum verification for critical files
            checksum_issues = await self._verify_checksums(model_id, model_path)
            if checksum_issues:
                result.issues.extend(checksum_issues)
                result.corruption_types.append(CorruptionType.CHECKSUM_MISMATCH)
                result.repair_suggestions.append("Re-download corrupted files")
            
            # Check file format validity
            format_issues = await self._verify_file_formats(model_id, model_path)
            if format_issues:
                result.issues.extend(format_issues)
                result.corruption_types.append(CorruptionType.INVALID_FORMAT)
                result.repair_suggestions.append("Verify model compatibility")
            
            # Calculate overall checksum
            result.checksum = await self._calculate_model_checksum(model_path)
            
            # Determine final health status
            if not result.issues:
                result.is_healthy = True
                result.health_status = HealthStatus.HEALTHY
            elif len(result.issues) <= 2 and CorruptionType.FILE_MISSING not in result.corruption_types:
                result.health_status = HealthStatus.DEGRADED
                result.repair_suggestions.append("Monitor for further degradation")
            else:
                result.health_status = HealthStatus.CORRUPTED
            
            # Save health check result
            await self._save_health_result(result)
            
            # Notify callbacks
            await self._notify_health_callbacks(result)
            
            check_time = time.time() - start_time
            logger.info(f"Integrity check completed for {model_id} in {check_time:.2f}s: {result.health_status.value}")
            
            # End performance monitoring
            if PERFORMANCE_MONITORING_AVAILABLE and performance_id:
                try:
                    monitor = get_performance_monitor()
                    monitor.end_tracking(
                        performance_id,
                        success=result.health_status != HealthStatus.UNKNOWN,
                        error_message=None if result.is_healthy else f"Health issues detected: {len(result.issues)} issues",
                        additional_metadata={
                            "health_status": result.health_status.value,
                            "is_healthy": result.is_healthy,
                            "issues_count": len(result.issues),
                            "corruption_types": [ct.value for ct in result.corruption_types],
                            "files_checked": getattr(result, 'files_checked', 0),
                            "integrity_score": result.integrity_score
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to end performance monitoring: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during integrity check for {model_id}: {e}")
            result.issues.append(f"Integrity check error: {str(e)}")
            result.health_status = HealthStatus.UNKNOWN
            
            # End performance monitoring with error
            if PERFORMANCE_MONITORING_AVAILABLE and performance_id:
                try:
                    monitor = get_performance_monitor()
                    monitor.end_tracking(
                        performance_id,
                        success=False,
                        error_message=str(e),
                        additional_metadata={
                            "health_status": "error",
                            "error_type": type(e).__name__
                        }
                    )
                except Exception as monitor_e:
                    logger.warning(f"Failed to end performance monitoring: {monitor_e}")
            
            return result
    
    def _get_model_path(self, model_id: str) -> Path:
        """Get the path to a model directory"""
        # Handle different model ID formats
        safe_id = model_id.replace("/", "_").replace("\\", "_")
        return self.models_dir / safe_id
    
    async def _get_essential_files(self, model_id: str) -> List[str]:
        """Get list of essential files for a model type"""
        # Basic essential files for most models
        essential_files = [
            "config.json",
            "model_index.json"
        ]
        
        # Add model-specific essential files based on model type
        model_type = self._detect_model_type(model_id)
        
        if "t2v" in model_type.lower():
            essential_files.extend([
                "unet/config.json",
                "text_encoder/config.json",
                "vae/config.json"
            ])
        elif "i2v" in model_type.lower():
            essential_files.extend([
                "unet/config.json",
                "image_encoder/config.json",
                "vae/config.json"
            ])
        elif "ti2v" in model_type.lower():
            essential_files.extend([
                "unet/config.json",
                "text_encoder/config.json",
                "image_encoder/config.json",
                "vae/config.json"
            ])
        
        return essential_files
    
    def _detect_model_type(self, model_id: str) -> str:
        """Detect model type from ID"""
        model_id_lower = model_id.lower()
        
        if "t2v" in model_id_lower:
            return "text-to-video"
        elif "i2v" in model_id_lower:
            return "image-to-video"
        elif "ti2v" in model_id_lower:
            return "text-image-to-video"
        else:
            return "unknown"
    
    async def _verify_checksums(self, model_id: str, model_path: Path) -> List[str]:
        """Verify checksums for critical model files"""
        issues = []
        
        try:
            # Look for checksum files
            checksum_files = list(model_path.glob("*.sha256")) + list(model_path.glob("*.md5"))
            
            if not checksum_files:
                # No checksum files found - not necessarily an issue
                return issues
            
            for checksum_file in checksum_files:
                try:
                    async with aiofiles.open(checksum_file, 'r') as f:
                        checksum_data = await f.read()
                    
                    # Parse checksum file (format: checksum filename)
                    for line in checksum_data.strip().split('\n'):
                        if not line.strip():
                            continue
                        
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            expected_checksum = parts[0]
                            filename = ' '.join(parts[1:]).strip('*')
                            
                            file_path = model_path / filename
                            if file_path.exists():
                                actual_checksum = await self._calculate_file_checksum(file_path)
                                
                                if actual_checksum != expected_checksum:
                                    issues.append(f"Checksum mismatch for {filename}")
                
                except Exception as e:
                    issues.append(f"Error verifying checksum file {checksum_file.name}: {e}")
        
        except Exception as e:
            logger.warning(f"Error during checksum verification: {e}")
        
        return issues
    
    async def _verify_file_formats(self, model_id: str, model_path: Path) -> List[str]:
        """Verify file formats are valid"""
        issues = []
        
        try:
            # Check JSON files
            json_files = list(model_path.rglob("*.json"))
            for json_file in json_files:
                try:
                    async with aiofiles.open(json_file, 'r') as f:
                        content = await f.read()
                    json.loads(content)  # Validate JSON format
                except json.JSONDecodeError:
                    issues.append(f"Invalid JSON format: {json_file.relative_to(model_path)}")
                except Exception as e:
                    issues.append(f"Cannot read JSON file {json_file.relative_to(model_path)}: {e}")
            
            # Check for common model file extensions
            model_files = (
                list(model_path.rglob("*.bin")) +
                list(model_path.rglob("*.safetensors")) +
                list(model_path.rglob("*.pt")) +
                list(model_path.rglob("*.pth"))
            )
            
            for model_file in model_files:
                try:
                    # Basic file size check
                    size = model_file.stat().st_size
                    if size == 0:
                        issues.append(f"Empty model file: {model_file.relative_to(model_path)}")
                    elif size < 1024:  # Less than 1KB is suspicious for model files
                        issues.append(f"Suspiciously small model file: {model_file.relative_to(model_path)}")
                
                except Exception as e:
                    issues.append(f"Cannot access model file {model_file.relative_to(model_path)}: {e}")
        
        except Exception as e:
            logger.warning(f"Error during file format verification: {e}")
        
        return issues
    
    async def _calculate_model_checksum(self, model_path: Path) -> str:
        """Calculate overall checksum for the model"""
        try:
            # Calculate checksum based on all files in the model directory
            sha256_hash = hashlib.sha256()
            
            # Sort files for consistent checksum
            all_files = sorted(model_path.rglob("*"))
            
            for file_path in all_files:
                if file_path.is_file():
                    # Add file path to hash for structure verification
                    relative_path = file_path.relative_to(model_path)
                    sha256_hash.update(str(relative_path).encode())
                    
                    # Add file size to hash
                    try:
                        size = file_path.stat().st_size
                        sha256_hash.update(size.to_bytes(8, 'big'))
                    except Exception:
                        continue
            
            return sha256_hash.hexdigest()
        
        except Exception as e:
            logger.warning(f"Error calculating model checksum: {e}")
            return "unknown"
    
    async def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a single file"""
        sha256_hash = hashlib.sha256()
        
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                while chunk := await f.read(8192):
                    sha256_hash.update(chunk)
            
            return sha256_hash.hexdigest()
        
        except Exception as e:
            logger.warning(f"Error calculating checksum for {file_path}: {e}")
            return "error"
    
    async def monitor_model_performance(self, model_id: str, generation_metrics: Dict[str, Any]) -> PerformanceHealth:
        """
        Monitor and analyze model performance metrics.
        
        Args:
            model_id: Model identifier
            generation_metrics: Metrics from generation process
            
        Returns:
            PerformanceHealth assessment
        """
        try:
            # Create performance metrics object
            metrics = PerformanceMetrics(
                model_id=model_id,
                load_time_seconds=generation_metrics.get('load_time_seconds', 0.0),
                generation_time_seconds=generation_metrics.get('generation_time_seconds', 0.0),
                memory_usage_mb=generation_metrics.get('memory_usage_mb', 0.0),
                vram_usage_mb=generation_metrics.get('vram_usage_mb', 0.0),
                cpu_usage_percent=generation_metrics.get('cpu_usage_percent', 0.0),
                throughput_fps=generation_metrics.get('throughput_fps', 0.0),
                quality_score=generation_metrics.get('quality_score'),
                error_rate=generation_metrics.get('error_rate', 0.0),
                timestamp=datetime.now()
            )
            
            # Store performance history
            if model_id not in self.performance_history:
                self.performance_history[model_id] = []
            
            self.performance_history[model_id].append(metrics)
            
            # Keep only recent history (last 100 entries)
            if len(self.performance_history[model_id]) > 100:
                self.performance_history[model_id] = self.performance_history[model_id][-100:]
            
            # Calculate performance health
            health = await self._assess_performance_health(model_id, metrics)
            
            # Save performance data
            await self._save_performance_data(model_id, metrics, health)
            
            return health
        
        except Exception as e:
            logger.error(f"Error monitoring performance for {model_id}: {e}")
            return PerformanceHealth(
                model_id=model_id,
                overall_score=0.5,
                performance_trend="unknown",
                recommendations=["Performance monitoring error - check logs"]
            )
    
    async def _assess_performance_health(self, model_id: str, current_metrics: PerformanceMetrics) -> PerformanceHealth:
        """Assess performance health based on current and historical metrics"""
        try:
            history = self.performance_history.get(model_id, [])
            
            # Calculate baseline if we have enough data
            if len(history) >= 5:
                baseline = await self._calculate_performance_baseline(model_id)
                self.performance_baselines[model_id] = baseline
            else:
                baseline = self.performance_baselines.get(model_id)
            
            health = PerformanceHealth(
                model_id=model_id,
                overall_score=1.0,
                performance_trend="stable",
                last_assessment=datetime.now()
            )
            
            bottlenecks = []
            recommendations = []
            
            # Analyze current performance
            if current_metrics.generation_time_seconds > 300:  # 5 minutes
                bottlenecks.append("Slow generation time")
                recommendations.append("Consider reducing inference steps or resolution")
                health.overall_score *= 0.8
            
            if current_metrics.vram_usage_mb > 14000:  # 14GB
                bottlenecks.append("High VRAM usage")
                recommendations.append("Enable model offloading or reduce batch size")
                health.overall_score *= 0.9
            
            if current_metrics.error_rate > 0.1:  # 10% error rate
                bottlenecks.append("High error rate")
                recommendations.append("Check model integrity and system resources")
                health.overall_score *= 0.7
            
            # Compare with baseline if available
            if baseline:
                baseline_comparison = {}
                
                # Generation time comparison
                time_ratio = current_metrics.generation_time_seconds / baseline.generation_time_seconds
                baseline_comparison['generation_time_ratio'] = time_ratio
                
                if time_ratio > 1.2:  # 20% slower
                    bottlenecks.append("Performance degradation detected")
                    recommendations.append("Consider model re-download or system optimization")
                    health.overall_score *= 0.8
                    health.performance_trend = "degrading"
                elif time_ratio < 0.9:  # 10% faster
                    health.performance_trend = "improving"
                
                # Memory usage comparison
                memory_ratio = current_metrics.vram_usage_mb / baseline.vram_usage_mb
                baseline_comparison['vram_usage_ratio'] = memory_ratio
                
                if memory_ratio > 1.3:  # 30% more memory
                    bottlenecks.append("Increased memory usage")
                    recommendations.append("Check for memory leaks or system changes")
                    health.overall_score *= 0.9
                
                health.baseline_comparison = baseline_comparison
            
            # Analyze trend from recent history
            if len(history) >= 10:
                recent_times = [m.generation_time_seconds for m in history[-10:]]
                if len(set(recent_times)) > 1:  # Not all the same
                    trend_slope = self._calculate_trend_slope(recent_times)
                    
                    if trend_slope > 0.1:  # Increasing trend
                        health.performance_trend = "degrading"
                        health.overall_score *= 0.9
                    elif trend_slope < -0.1:  # Decreasing trend (improvement)
                        health.performance_trend = "improving"
            
            health.bottlenecks = bottlenecks
            health.recommendations = recommendations
            
            return health
        
        except Exception as e:
            logger.error(f"Error assessing performance health: {e}")
            return PerformanceHealth(
                model_id=model_id,
                overall_score=0.5,
                performance_trend="unknown"
            )
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope using simple linear regression"""
        try:
            n = len(values)
            if n < 2:
                return 0.0
            
            x_values = list(range(n))
            
            # Calculate means
            x_mean = sum(x_values) / n
            y_mean = sum(values) / n
            
            # Calculate slope
            numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
        
        except Exception:
            return 0.0
    
    async def _calculate_performance_baseline(self, model_id: str) -> PerformanceMetrics:
        """Calculate performance baseline from historical data"""
        history = self.performance_history.get(model_id, [])
        
        if not history:
            return None
        
        # Use median values for baseline (more robust than mean)
        load_times = sorted([m.load_time_seconds for m in history])
        generation_times = sorted([m.generation_time_seconds for m in history])
        memory_usage = sorted([m.memory_usage_mb for m in history])
        vram_usage = sorted([m.vram_usage_mb for m in history])
        cpu_usage = sorted([m.cpu_usage_percent for m in history])
        throughput = sorted([m.throughput_fps for m in history])
        
        def median(values):
            n = len(values)
            if n == 0:
                return 0.0
            return values[n // 2] if n % 2 == 1 else (values[n // 2 - 1] + values[n // 2]) / 2
        
        return PerformanceMetrics(
            model_id=model_id,
            load_time_seconds=median(load_times),
            generation_time_seconds=median(generation_times),
            memory_usage_mb=median(memory_usage),
            vram_usage_mb=median(vram_usage),
            cpu_usage_percent=median(cpu_usage),
            throughput_fps=median(throughput),
            timestamp=datetime.now()
        )
    
    async def detect_corruption(self, model_id: str) -> CorruptionReport:
        """
        Detect and analyze model corruption.
        
        Args:
            model_id: Model identifier to check
            
        Returns:
            CorruptionReport with detailed analysis
        """
        logger.info(f"Starting corruption detection for model: {model_id}")
        
        report = CorruptionReport(
            model_id=model_id,
            corruption_detected=False,
            detection_time=datetime.now()
        )
        
        try:
            # Run integrity check first
            integrity_result = await self.check_model_integrity(model_id)
            
            if integrity_result.corruption_types:
                report.corruption_detected = True
                report.corruption_types = integrity_result.corruption_types
                report.affected_files = [
                    issue.split(": ")[1] if ": " in issue else issue
                    for issue in integrity_result.issues
                    if "file" in issue.lower()
                ]
                
                # Determine severity
                if CorruptionType.FILE_MISSING in report.corruption_types:
                    report.severity = "critical"
                elif CorruptionType.CHECKSUM_MISMATCH in report.corruption_types:
                    report.severity = "high"
                elif CorruptionType.INVALID_FORMAT in report.corruption_types:
                    report.severity = "medium"
                else:
                    report.severity = "low"
                
                # Determine repair actions
                repair_actions = []
                
                if CorruptionType.FILE_MISSING in report.corruption_types:
                    repair_actions.extend([
                        "Re-download missing files",
                        "Verify download completion",
                        "Check available disk space"
                    ])
                    report.repair_possible = True
                
                if CorruptionType.CHECKSUM_MISMATCH in report.corruption_types:
                    repair_actions.extend([
                        "Re-download corrupted files",
                        "Verify network connection stability",
                        "Check disk health"
                    ])
                    report.repair_possible = True
                
                if CorruptionType.PERMISSION_ERROR in report.corruption_types:
                    repair_actions.extend([
                        "Check file permissions",
                        "Run as administrator if needed",
                        "Verify disk access rights"
                    ])
                    report.repair_possible = True
                
                if CorruptionType.DISK_ERROR in report.corruption_types:
                    repair_actions.extend([
                        "Check disk health",
                        "Run disk error checking",
                        "Consider moving to different storage"
                    ])
                    report.repair_possible = False  # May require manual intervention
                
                report.repair_actions = repair_actions
            
            # Additional corruption checks
            await self._perform_advanced_corruption_detection(model_id, report)
            
            # Notify callbacks if corruption detected
            if report.corruption_detected:
                await self._notify_corruption_callbacks(report)
            
            logger.info(f"Corruption detection completed for {model_id}: {'DETECTED' if report.corruption_detected else 'CLEAN'}")
            
            return report
        
        except Exception as e:
            logger.error(f"Error during corruption detection for {model_id}: {e}")
            report.corruption_detected = True
            report.severity = "unknown"
            report.repair_possible = False
            report.repair_actions = [f"Corruption detection error: {str(e)}"]
            return report
    
    async def _perform_advanced_corruption_detection(self, model_id: str, report: CorruptionReport):
        """Perform advanced corruption detection algorithms"""
        try:
            model_path = self._get_model_path(model_id)
            
            if not model_path.exists():
                return
            
            # Check for incomplete downloads (temporary files)
            temp_files = list(model_path.rglob("*.tmp")) + list(model_path.rglob("*.partial"))
            if temp_files:
                report.corruption_detected = True
                report.corruption_types.append(CorruptionType.INCOMPLETE_DOWNLOAD)
                report.affected_files.extend([f.name for f in temp_files])
                report.repair_actions.append("Complete interrupted downloads")
            
            # Check for zero-byte files
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    try:
                        if file_path.stat().st_size == 0:
                            report.corruption_detected = True
                            if CorruptionType.INCOMPLETE_DOWNLOAD not in report.corruption_types:
                                report.corruption_types.append(CorruptionType.INCOMPLETE_DOWNLOAD)
                            report.affected_files.append(str(file_path.relative_to(model_path)))
                    except Exception:
                        continue
            
            # Check disk space and health
            try:
                disk_usage = psutil.disk_usage(str(model_path))
                if disk_usage.free < 1024 * 1024 * 1024:  # Less than 1GB free
                    report.repair_actions.append("Free up disk space")
            except Exception:
                pass
        
        except Exception as e:
            logger.warning(f"Error in advanced corruption detection: {e}")
    
    async def schedule_health_checks(self):
        """Start scheduled health checks for all models"""
        if self._monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self._monitoring_active = True
        
        async def health_check_loop():
            while self._monitoring_active:
                try:
                    logger.info("Starting scheduled health checks")
                    
                    # Get all models in the directory
                    model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                    
                    for model_dir in model_dirs:
                        if not self._monitoring_active:
                            break
                        
                        model_id = model_dir.name
                        
                        try:
                            # Perform integrity check
                            integrity_result = await self.check_model_integrity(model_id)
                            
                            # Perform corruption detection if issues found
                            if integrity_result.issues:
                                corruption_report = await self.detect_corruption(model_id)
                                
                                # Trigger automatic repair if enabled and possible
                                if (self.config.automatic_repair_enabled and 
                                    corruption_report.repair_possible and 
                                    corruption_report.severity in ["low", "medium"]):
                                    
                                    await self._attempt_automatic_repair(model_id, corruption_report)
                        
                        except Exception as e:
                            logger.error(f"Error in scheduled health check for {model_id}: {e}")
                    
                    logger.info("Scheduled health checks completed")
                    
                    # Wait for next check interval
                    await asyncio.sleep(self.config.check_interval_hours * 3600)
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in health check loop: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes before retrying
        
        self._health_check_task = asyncio.create_task(health_check_loop())
        logger.info(f"Scheduled health checks started (interval: {self.config.check_interval_hours} hours)")
    
    async def stop_health_checks(self):
        """Stop scheduled health checks"""
        self._monitoring_active = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
        
        logger.info("Scheduled health checks stopped")
    
    async def _attempt_automatic_repair(self, model_id: str, corruption_report: CorruptionReport):
        """Attempt automatic repair of detected issues"""
        try:
            logger.info(f"Attempting automatic repair for {model_id}")
            
            repair_success = False
            
            # Handle missing files
            if CorruptionType.FILE_MISSING in corruption_report.corruption_types:
                # This would typically trigger a re-download
                logger.info(f"Automatic repair needed: re-download missing files for {model_id}")
                # Note: Actual re-download would be handled by the enhanced downloader
                repair_success = True
            
            # Handle permission errors
            if CorruptionType.PERMISSION_ERROR in corruption_report.corruption_types:
                model_path = self._get_model_path(model_id)
                try:
                    # Attempt to fix permissions
                    for file_path in model_path.rglob("*"):
                        if file_path.is_file():
                            file_path.chmod(0o644)
                    repair_success = True
                    logger.info(f"Fixed file permissions for {model_id}")
                except Exception as e:
                    logger.warning(f"Could not fix permissions for {model_id}: {e}")
            
            # Clean up temporary files
            if CorruptionType.INCOMPLETE_DOWNLOAD in corruption_report.corruption_types:
                model_path = self._get_model_path(model_id)
                temp_files = list(model_path.rglob("*.tmp")) + list(model_path.rglob("*.partial"))
                
                for temp_file in temp_files:
                    try:
                        temp_file.unlink()
                        logger.info(f"Removed temporary file: {temp_file.name}")
                        repair_success = True
                    except Exception as e:
                        logger.warning(f"Could not remove temporary file {temp_file.name}: {e}")
            
            if repair_success:
                logger.info(f"Automatic repair completed for {model_id}")
                
                # Re-check integrity after repair
                await asyncio.sleep(1)  # Brief delay
                integrity_result = await self.check_model_integrity(model_id)
                
                if integrity_result.is_healthy:
                    logger.info(f"Model {model_id} successfully repaired")
                else:
                    logger.warning(f"Model {model_id} still has issues after repair attempt")
            
        except Exception as e:
            logger.error(f"Error during automatic repair for {model_id}: {e}")
    
    async def get_health_report(self) -> SystemHealthReport:
        """
        Generate comprehensive system health report.
        
        Returns:
            SystemHealthReport with overall system status
        """
        logger.info("Generating system health report")
        
        report = SystemHealthReport(
            overall_health_score=1.0,
            models_healthy=0,
            models_degraded=0,
            models_corrupted=0,
            models_missing=0,
            storage_usage_percent=0.0,
            last_updated=datetime.now()
        )
        
        try:
            # Get all models
            model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            
            total_models = len(model_dirs)
            if total_models == 0:
                report.recommendations.append("No models found - consider downloading models")
                return report
            
            # Check each model
            for model_dir in model_dirs:
                model_id = model_dir.name
                
                try:
                    integrity_result = await self.check_model_integrity(model_id)
                    report.detailed_reports[model_id] = integrity_result
                    
                    if integrity_result.health_status == HealthStatus.HEALTHY:
                        report.models_healthy += 1
                    elif integrity_result.health_status == HealthStatus.DEGRADED:
                        report.models_degraded += 1
                    elif integrity_result.health_status == HealthStatus.CORRUPTED:
                        report.models_corrupted += 1
                    elif integrity_result.health_status == HealthStatus.MISSING:
                        report.models_missing += 1
                
                except Exception as e:
                    logger.error(f"Error checking model {model_id} for health report: {e}")
                    report.models_corrupted += 1
            
            # Calculate overall health score
            if total_models > 0:
                health_weight = report.models_healthy / total_models
                degraded_weight = (report.models_degraded / total_models) * 0.7
                corrupted_weight = (report.models_corrupted / total_models) * 0.3
                missing_weight = (report.models_missing / total_models) * 0.1
                
                report.overall_health_score = health_weight + degraded_weight + corrupted_weight + missing_weight
                report.overall_health_score = max(0.0, min(1.0, report.overall_health_score))
            
            # Check storage usage
            try:
                disk_usage = psutil.disk_usage(str(self.models_dir))
                report.storage_usage_percent = (disk_usage.used / disk_usage.total) * 100
                
                if report.storage_usage_percent > self.config.storage_critical_threshold * 100:
                    report.recommendations.append("Critical: Storage usage above 95% - immediate cleanup required")
                elif report.storage_usage_percent > self.config.storage_warning_threshold * 100:
                    report.recommendations.append("Warning: Storage usage above 80% - consider cleanup")
            
            except Exception as e:
                logger.warning(f"Could not check storage usage: {e}")
            
            # Generate recommendations
            if report.models_corrupted > 0:
                report.recommendations.append(f"{report.models_corrupted} corrupted models detected - repair or re-download recommended")
            
            if report.models_degraded > 0:
                report.recommendations.append(f"{report.models_degraded} degraded models detected - monitor closely")
            
            if report.models_missing > 0:
                report.recommendations.append(f"{report.models_missing} missing models detected - download required")
            
            if report.overall_health_score < 0.8:
                report.recommendations.append("Overall system health below 80% - immediate attention required")
            elif report.overall_health_score < 0.9:
                report.recommendations.append("System health could be improved - review model status")
            
            logger.info(f"Health report generated: {report.models_healthy} healthy, {report.models_degraded} degraded, {report.models_corrupted} corrupted")
            
            return report
        
        except Exception as e:
            logger.error(f"Error generating health report: {e}")
            report.recommendations.append(f"Error generating health report: {str(e)}")
            return report
    
    async def _save_health_result(self, result: IntegrityResult):
        """Save health check result to disk"""
        try:
            result_file = self.health_data_dir / f"{result.model_id}_integrity.json"
            
            result_data = {
                "model_id": result.model_id,
                "is_healthy": result.is_healthy,
                "health_status": result.health_status.value,
                "issues": result.issues,
                "corruption_types": [ct.value for ct in result.corruption_types],
                "file_count": result.file_count,
                "total_size_mb": result.total_size_mb,
                "checksum": result.checksum,
                "last_checked": result.last_checked.isoformat() if result.last_checked else None,
                "repair_suggestions": result.repair_suggestions
            }
            
            async with aiofiles.open(result_file, 'w') as f:
                await f.write(json.dumps(result_data, indent=2))
        
        except Exception as e:
            logger.warning(f"Could not save health result for {result.model_id}: {e}")
    
    async def _save_performance_data(self, model_id: str, metrics: PerformanceMetrics, health: PerformanceHealth):
        """Save performance data to disk"""
        try:
            perf_file = self.health_data_dir / f"{model_id}_performance.json"
            
            # Load existing data
            existing_data = []
            if perf_file.exists():
                try:
                    async with aiofiles.open(perf_file, 'r') as f:
                        content = await f.read()
                    existing_data = json.loads(content)
                except Exception:
                    existing_data = []
            
            # Add new metrics
            metrics_data = {
                "timestamp": metrics.timestamp.isoformat() if metrics.timestamp else None,
                "load_time_seconds": metrics.load_time_seconds,
                "generation_time_seconds": metrics.generation_time_seconds,
                "memory_usage_mb": metrics.memory_usage_mb,
                "vram_usage_mb": metrics.vram_usage_mb,
                "cpu_usage_percent": metrics.cpu_usage_percent,
                "throughput_fps": metrics.throughput_fps,
                "quality_score": metrics.quality_score,
                "error_rate": metrics.error_rate,
                "health_score": health.overall_score,
                "performance_trend": health.performance_trend,
                "bottlenecks": health.bottlenecks,
                "recommendations": health.recommendations
            }
            
            existing_data.append(metrics_data)
            
            # Keep only recent data (last 100 entries)
            if len(existing_data) > 100:
                existing_data = existing_data[-100:]
            
            async with aiofiles.open(perf_file, 'w') as f:
                await f.write(json.dumps(existing_data, indent=2))
        
        except Exception as e:
            logger.warning(f"Could not save performance data for {model_id}: {e}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.stop_health_checks()
        
        if self._executor:
            self._executor.shutdown(wait=True)
        
        logger.info("Model Health Monitor cleanup completed")


# Global health monitor instance
_health_monitor = None

def get_model_health_monitor(models_dir: Optional[str] = None, config: Optional[HealthCheckConfig] = None) -> ModelHealthMonitor:
    """Get the global model health monitor instance"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = ModelHealthMonitor(models_dir, config)
    return _health_monitor

# Convenience functions
async def check_model_integrity(model_id: str) -> IntegrityResult:
    """Check integrity of a specific model"""
    monitor = get_model_health_monitor()
    return await monitor.check_model_integrity(model_id)

async def monitor_model_performance(model_id: str, generation_metrics: Dict[str, Any]) -> PerformanceHealth:
    """Monitor performance of a specific model"""
    monitor = get_model_health_monitor()
    return await monitor.monitor_model_performance(model_id, generation_metrics)

async def detect_model_corruption(model_id: str) -> CorruptionReport:
    """Detect corruption in a specific model"""
    monitor = get_model_health_monitor()
    return await monitor.detect_corruption(model_id)

async def get_system_health_report() -> SystemHealthReport:
    """Get comprehensive system health report"""
    monitor = get_model_health_monitor()
    return await monitor.get_health_report()

async def start_health_monitoring():
    """Start scheduled health monitoring"""
    monitor = get_model_health_monitor()
    await monitor.schedule_health_checks()

async def stop_health_monitoring():
    """Stop scheduled health monitoring"""
    monitor = get_model_health_monitor()
    await monitor.stop_health_checks()