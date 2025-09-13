"""
Diagnostic Tool for Wan2.2 UI Variant Local Testing Framework
Provides automated troubleshooting, system analysis, and recovery capabilities
"""

import os
import re
import json
import time
import psutil
import logging
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# Import existing modules
from .models.test_results import ValidationResult, ValidationStatus, ResourceMetrics
from error_handler import (
    ErrorCategory, ErrorInfo, ErrorRecoveryManager, 
    get_error_recovery_manager, create_error_info
)

# Configure logging
logger = logging.getLogger(__name__)


class DiagnosticCategory(Enum):
    """Categories of diagnostic issues"""
    CUDA_ERROR = "cuda_error"
    MEMORY_ERROR = "memory_error"
    MODEL_DOWNLOAD_ERROR = "model_download_error"
    CONFIGURATION_ERROR = "configuration_error"
    SYSTEM_RESOURCE_ERROR = "system_resource_error"
    NETWORK_ERROR = "network_error"
    FILE_PERMISSION_ERROR = "file_permission_error"
    DEPENDENCY_ERROR = "dependency_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class DiagnosticIssue:
    """Represents a diagnostic issue found during analysis"""
    category: DiagnosticCategory
    severity: str  # "critical", "warning", "info"
    title: str
    description: str
    affected_components: List[str]
    symptoms: List[str]
    root_cause: str
    remediation_steps: List[str]
    auto_recoverable: bool
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "category": self.category.value,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "affected_components": self.affected_components,
            "symptoms": self.symptoms,
            "root_cause": self.root_cause,
            "remediation_steps": self.remediation_steps,
            "auto_recoverable": self.auto_recoverable,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class SystemAnalysis:
    """Results of system analysis"""
    timestamp: datetime
    system_health_score: float  # 0-100
    resource_status: ResourceMetrics
    configuration_issues: List[DiagnosticIssue]
    performance_issues: List[DiagnosticIssue]
    dependency_issues: List[DiagnosticIssue]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "system_health_score": self.system_health_score,
            "resource_status": self.resource_status.to_dict(),
            "configuration_issues": [issue.to_dict() for issue in self.configuration_issues],
            "performance_issues": [issue.to_dict() for issue in self.performance_issues],
            "dependency_issues": [issue.to_dict() for issue in self.dependency_issues],
            "recommendations": self.recommendations
        }


@dataclass
class DiagnosticResults:
    """Comprehensive diagnostic results"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    system_analysis: Optional[SystemAnalysis] = None
    error_log_analysis: Optional[Dict[str, Any]] = None
    issues_found: List[DiagnosticIssue] = field(default_factory=list)
    recovery_attempts: List[Dict[str, Any]] = field(default_factory=list)
    overall_status: str = "unknown"  # "healthy", "warning", "critical"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "issues_found": [issue.to_dict() for issue in self.issues_found],
            "recovery_attempts": self.recovery_attempts,
            "overall_status": self.overall_status
        }
        
        if self.end_time:
            result["end_time"] = self.end_time.isoformat()
        if self.system_analysis:
            result["system_analysis"] = self.system_analysis.to_dict()
        if self.error_log_analysis:
            result["error_log_analysis"] = self.error_log_analysis
            
        return result


class SystemAnalyzer:
    """Analyzes system resources and configuration for issues"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_path}: {e}")
            return {}
    
    def analyze_system_resources(self) -> Tuple[ResourceMetrics, List[DiagnosticIssue]]:
        """Analyze current system resource usage and identify issues"""
        issues = []
        
        # Collect current resource metrics
        metrics = ResourceMetrics(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            memory_used_gb=psutil.virtual_memory().used / (1024**3),
            memory_total_gb=psutil.virtual_memory().total / (1024**3),
            gpu_percent=0.0,
            vram_used_mb=0,
            vram_total_mb=0,
            vram_percent=0.0
        )
        
        # Try to get GPU metrics
        try:
            import torch
            if torch.cuda.is_available():
                metrics.vram_used_mb = torch.cuda.memory_allocated(0) / (1024**2)
                metrics.vram_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                metrics.vram_percent = (metrics.vram_used_mb / metrics.vram_total_mb) * 100
                
                # Try to get GPU utilization
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        metrics.gpu_percent = gpus[0].load * 100
                except ImportError:
                    pass
        except Exception as e:
            logger.debug(f"Failed to collect GPU metrics: {e}")
        
        # Analyze resource usage for issues
        if metrics.cpu_percent > 90:
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.SYSTEM_RESOURCE_ERROR,
                severity="critical",
                title="High CPU Usage",
                description=f"CPU usage is at {metrics.cpu_percent:.1f}%, which may cause performance issues",
                affected_components=["system_performance", "generation_speed"],
                symptoms=["Slow response times", "System lag", "High fan noise"],
                root_cause="CPU is overloaded with processes",
                remediation_steps=[
                    "Close unnecessary applications",
                    "Check for background processes consuming CPU",
                    "Consider upgrading CPU or reducing workload",
                    "Enable CPU optimization in config"
                ],
                auto_recoverable=False
            ))
        
        if metrics.memory_percent > 85:
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.MEMORY_ERROR,
                severity="critical" if metrics.memory_percent > 95 else "warning",
                title="High Memory Usage",
                description=f"Memory usage is at {metrics.memory_percent:.1f}% ({metrics.memory_used_gb:.1f}GB/{metrics.memory_total_gb:.1f}GB)",
                affected_components=["system_stability", "model_loading"],
                symptoms=["System slowdown", "Application crashes", "Swap usage"],
                root_cause="Insufficient available RAM for current workload",
                remediation_steps=[
                    "Close unnecessary applications",
                    "Enable model offloading to reduce RAM usage",
                    "Clear system cache",
                    "Consider adding more RAM"
                ],
                auto_recoverable=True
            ))
        
        if metrics.vram_percent > 90:
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.CUDA_ERROR,
                severity="critical",
                title="High VRAM Usage",
                description=f"VRAM usage is at {metrics.vram_percent:.1f}% ({metrics.vram_used_mb:.0f}MB/{metrics.vram_total_mb:.0f}MB)",
                affected_components=["gpu_performance", "model_inference"],
                symptoms=["CUDA out of memory errors", "Generation failures", "Model loading issues"],
                root_cause="GPU memory is nearly exhausted",
                remediation_steps=[
                    "Enable attention slicing in optimizations",
                    "Reduce VAE tile size",
                    "Enable model offloading",
                    "Use lower precision (fp16/int8)",
                    "Reduce batch size or resolution"
                ],
                auto_recoverable=True
            ))
        
        # Check disk space
        disk_usage = psutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        
        if free_gb < 5:
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.SYSTEM_RESOURCE_ERROR,
                severity="critical",
                title="Low Disk Space",
                description=f"Only {free_gb:.1f}GB of disk space remaining",
                affected_components=["model_downloads", "output_generation", "cache_storage"],
                symptoms=["Download failures", "Cache errors", "Unable to save outputs"],
                root_cause="Insufficient disk space for operations",
                remediation_steps=[
                    "Clear model cache directory",
                    "Remove old output files",
                    "Clean temporary files",
                    "Move files to external storage"
                ],
                auto_recoverable=True
            ))
        
        return metrics, issues
    
    def analyze_configuration(self) -> List[DiagnosticIssue]:
        """Analyze configuration files for issues"""
        issues = []
        
        # Check config.json
        if not self.config:
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.CONFIGURATION_ERROR,
                severity="critical",
                title="Missing Configuration File",
                description="config.json file is missing or invalid",
                affected_components=["system_initialization", "optimization_settings"],
                symptoms=["Application startup failures", "Default settings used"],
                root_cause="Configuration file not found or corrupted",
                remediation_steps=[
                    "Create config.json from template",
                    "Restore from backup",
                    "Run configuration wizard"
                ],
                auto_recoverable=True
            ))
        else:
            # Check required configuration sections
            required_sections = ["system", "directories", "optimization", "performance"]
            for section in required_sections:
                if section not in self.config:
                    issues.append(DiagnosticIssue(
                        category=DiagnosticCategory.CONFIGURATION_ERROR,
                        severity="warning",
                        title=f"Missing Configuration Section: {section}",
                        description=f"Required configuration section '{section}' is missing",
                        affected_components=[f"{section}_functionality"],
                        symptoms=["Default settings used", "Suboptimal performance"],
                        root_cause=f"Configuration section '{section}' not defined",
                        remediation_steps=[
                            f"Add '{section}' section to config.json",
                            "Use configuration template",
                            "Reset to default configuration"
                        ],
                        auto_recoverable=True
                    ))
        
        # Check .env file
        env_file = Path(".env")
        if not env_file.exists():
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.CONFIGURATION_ERROR,
                severity="warning",
                title="Missing Environment File",
                description=".env file not found",
                affected_components=["model_downloads", "authentication"],
                symptoms=["Model download failures", "Authentication errors"],
                root_cause="Environment variables not configured",
                remediation_steps=[
                    "Create .env file with required variables",
                    "Set HF_TOKEN for model downloads",
                    "Configure CUDA environment variables"
                ],
                auto_recoverable=True
            ))
        else:
            # Check for required environment variables
            try:
                with open(env_file, 'r') as f:
                    env_content = f.read()
                
                if "HF_TOKEN" not in env_content:
                    issues.append(DiagnosticIssue(
                        category=DiagnosticCategory.CONFIGURATION_ERROR,
                        severity="warning",
                        title="Missing HF_TOKEN",
                        description="HF_TOKEN not found in .env file",
                        affected_components=["model_downloads"],
                        symptoms=["Model download failures", "Authentication errors"],
                        root_cause="Hugging Face token not configured",
                        remediation_steps=[
                            "Add HF_TOKEN=your_token to .env file",
                            "Get token from huggingface.co/settings/tokens",
                            "Ensure token has read permissions"
                        ],
                        auto_recoverable=False
                    ))
            except Exception as e:
                logger.debug(f"Failed to read .env file: {e}")
        
        return issues
    
    def analyze_dependencies(self) -> List[DiagnosticIssue]:
        """Analyze Python dependencies for issues"""
        issues = []
        
        try:
            # Check critical dependencies
            critical_deps = {
                "torch": "PyTorch for GPU operations",
                "diffusers": "Diffusion models",
                "transformers": "Model transformers",
                "gradio": "Web interface"
            }
            
            for dep_name, description in critical_deps.items():
                try:
                    __import__(dep_name)
                except ImportError as e:
                    issues.append(DiagnosticIssue(
                        category=DiagnosticCategory.DEPENDENCY_ERROR,
                        severity="critical",
                        title=f"Missing Dependency: {dep_name}",
                        description=f"Required dependency '{dep_name}' ({description}) is not installed",
                        affected_components=["core_functionality"],
                        symptoms=["Import errors", "Application startup failures"],
                        root_cause=f"Python package '{dep_name}' not installed",
                        remediation_steps=[
                            f"Install {dep_name}: pip install {dep_name}",
                            "Install from requirements.txt: pip install -r requirements.txt",
                            "Check virtual environment activation"
                        ],
                        auto_recoverable=True
                    ))
            
            # Check CUDA availability if torch is available
            try:
                import torch
                if not torch.cuda.is_available():
                    issues.append(DiagnosticIssue(
                        category=DiagnosticCategory.CUDA_ERROR,
                        severity="warning",
                        title="CUDA Not Available",
                        description="PyTorch cannot access CUDA, falling back to CPU",
                        affected_components=["gpu_acceleration", "performance"],
                        symptoms=["Slow generation", "CPU-only processing"],
                        root_cause="CUDA drivers or PyTorch CUDA support not properly installed",
                        remediation_steps=[
                            "Install CUDA drivers from NVIDIA",
                            "Install PyTorch with CUDA support",
                            "Check CUDA_VISIBLE_DEVICES environment variable",
                            "Verify GPU compatibility"
                        ],
                        auto_recoverable=False
                    ))
            except ImportError:
                pass
                
        except Exception as e:
            logger.error(f"Failed to analyze dependencies: {e}")
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.DEPENDENCY_ERROR,
                severity="warning",
                title="Dependency Analysis Failed",
                description=f"Could not analyze dependencies: {e}",
                affected_components=["dependency_validation"],
                symptoms=["Unknown dependency status"],
                root_cause="Dependency analysis system error",
                remediation_steps=[
                    "Check Python environment",
                    "Verify package installation",
                    "Run manual dependency check"
                ],
                auto_recoverable=False
            ))
        
        return issues


class ErrorLogAnalyzer:
    """Analyzes error logs to identify patterns and issues"""
    
    def __init__(self, log_file_path: str = "wan22_errors.log"):
        self.log_file_path = Path(log_file_path)
        
    def analyze_error_logs(self, hours_back: int = 24) -> Dict[str, Any]:
        """Analyze error logs for patterns and recent issues"""
        if not self.log_file_path.exists():
            return {
                "status": "no_logs",
                "message": "Error log file not found",
                "recent_errors": [],
                "error_patterns": {},
                "recommendations": ["Enable error logging", "Check log file permissions"]
            }
        
        try:
            # Read log file
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # Parse log entries
            log_entries = self._parse_log_entries(log_content)
            
            # Filter recent entries
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_entries = [
                entry for entry in log_entries 
                if entry.get('timestamp', datetime.min) > cutoff_time
            ]
            
            # Analyze error patterns
            error_patterns = self._analyze_error_patterns(log_entries)
            
            # Generate recommendations
            recommendations = self._generate_error_recommendations(error_patterns, recent_entries)
            
            return {
                "status": "analyzed",
                "total_entries": len(log_entries),
                "recent_entries": len(recent_entries),
                "recent_errors": recent_entries[-10:],  # Last 10 errors
                "error_patterns": error_patterns,
                "recommendations": recommendations,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze error logs: {e}")
            return {
                "status": "analysis_failed",
                "error": str(e),
                "recommendations": ["Check log file permissions", "Verify log file format"]
            }
    
    def _parse_log_entries(self, log_content: str) -> List[Dict[str, Any]]:
        """Parse log entries from log content"""
        entries = []
        
        # Split by log entry pattern (timestamp at start of line)
        log_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})'
        parts = re.split(log_pattern, log_content)
        
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                timestamp_str = parts[i]
                content = parts[i + 1]
                
                try:
                    # Parse timestamp
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                    
                    # Extract error information
                    entry = {
                        'timestamp': timestamp,
                        'raw_content': content,
                        'error_type': self._extract_error_type(content),
                        'error_message': self._extract_error_message(content),
                        'function_name': self._extract_function_name(content),
                        'severity': self._extract_severity(content)
                    }
                    
                    entries.append(entry)
                    
                except Exception as e:
                    logger.debug(f"Failed to parse log entry: {e}")
        
        return entries
    
    def _extract_error_type(self, content: str) -> str:
        """Extract error type from log content"""
        # Look for exception types
        exception_match = re.search(r'(\w+Error|\w+Exception):', content)
        if exception_match:
            return exception_match.group(1)
        
        # Look for other error indicators
        if 'CUDA out of memory' in content:
            return 'CUDAOutOfMemoryError'
        elif 'Failed to import' in content:
            return 'ImportError'
        elif 'Connection' in content and ('timeout' in content.lower() or 'failed' in content.lower()):
            return 'ConnectionError'
        
        return 'UnknownError'
    
    def _extract_error_message(self, content: str) -> str:
        """Extract error message from log content"""
        lines = content.strip().split('\n')
        if lines:
            # First line usually contains the main error message
            first_line = lines[0].strip()
            # Remove log formatting
            if ' - ' in first_line:
                parts = first_line.split(' - ')
                if len(parts) > 3:
                    return parts[-1]
            return first_line
        return "Unknown error"
    
    def _extract_function_name(self, content: str) -> str:
        """Extract function name from log content"""
        # Look for function name in log format
        func_match = re.search(r' - (\w+):\d+ - ', content)
        if func_match:
            return func_match.group(1)
        return "unknown"
    
    def _extract_severity(self, content: str) -> str:
        """Extract severity level from log content"""
        if 'CRITICAL' in content:
            return 'critical'
        elif 'ERROR' in content:
            return 'error'
        elif 'WARNING' in content:
            return 'warning'
        return 'info'
    
    def _analyze_error_patterns(self, log_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error patterns from log entries"""
        patterns = {
            'error_types': {},
            'frequent_functions': {},
            'time_patterns': {},
            'severity_distribution': {}
        }
        
        for entry in log_entries:
            # Count error types
            error_type = entry.get('error_type', 'Unknown')
            patterns['error_types'][error_type] = patterns['error_types'].get(error_type, 0) + 1
            
            # Count function occurrences
            func_name = entry.get('function_name', 'unknown')
            patterns['frequent_functions'][func_name] = patterns['frequent_functions'].get(func_name, 0) + 1
            
            # Count severity levels
            severity = entry.get('severity', 'info')
            patterns['severity_distribution'][severity] = patterns['severity_distribution'].get(severity, 0) + 1
            
            # Analyze time patterns (hour of day)
            if 'timestamp' in entry:
                hour = entry['timestamp'].hour
                patterns['time_patterns'][hour] = patterns['time_patterns'].get(hour, 0) + 1
        
        # Sort by frequency
        for key in patterns:
            if isinstance(patterns[key], dict):
                patterns[key] = dict(sorted(patterns[key].items(), key=lambda x: x[1], reverse=True))
        
        return patterns
    
    def _generate_error_recommendations(self, error_patterns: Dict[str, Any], recent_entries: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on error analysis"""
        recommendations = []
        
        # Analyze most common error types
        error_types = error_patterns.get('error_types', {})
        
        if 'CUDAOutOfMemoryError' in error_types:
            recommendations.extend([
                "Enable attention slicing to reduce VRAM usage",
                "Reduce VAE tile size in optimization settings",
                "Enable model offloading to CPU",
                "Use lower precision (fp16) for models",
                "Reduce batch size or generation resolution"
            ])
        
        if 'ImportError' in error_types:
            recommendations.extend([
                "Check Python environment and package installations",
                "Reinstall dependencies: pip install -r requirements.txt",
                "Verify CUDA installation if GPU-related imports fail",
                "Check for conflicting package versions"
            ])
        
        if 'ConnectionError' in error_types:
            recommendations.extend([
                "Check internet connection stability",
                "Verify Hugging Face Hub accessibility",
                "Configure proxy settings if behind firewall",
                "Try downloading models manually"
            ])
        
        # Check for recent error spikes
        if len(recent_entries) > 10:
            recommendations.append("High error frequency detected - consider system restart")
        
        # Check severity distribution
        severity_dist = error_patterns.get('severity_distribution', {})
        if severity_dist.get('critical', 0) > 0:
            recommendations.append("Critical errors detected - immediate attention required")
        
        return recommendations


class DiagnosticTool:
    """Main diagnostic orchestrator with real-time monitoring and analysis"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.system_analyzer = SystemAnalyzer(config_path)
        self.error_log_analyzer = ErrorLogAnalyzer()
        self.recovery_manager = RecoveryManager()
        self.report_generator = DiagnosticReportGenerator()
        self.error_recovery_manager = get_error_recovery_manager()
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Load performance profiler for real-time monitoring
        try:
            from performance_profiler import get_performance_profiler
            self.performance_profiler = get_performance_profiler()
        except ImportError:
            logger.warning("Performance profiler not available")
            self.performance_profiler = None
    
    def start_real_time_monitoring(self):
        """Start real-time system monitoring using performance_profiler.py --monitor"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start performance profiler monitoring if available
        if self.performance_profiler:
            self.performance_profiler.start_monitoring()
        
        # Start diagnostic monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Real-time diagnostic monitoring started")
    
    def stop_real_time_monitoring(self):
        """Stop real-time monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        # Stop performance profiler monitoring
        if self.performance_profiler:
            self.performance_profiler.stop_monitoring()
        
        # Stop monitoring thread
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        
        logger.info("Real-time diagnostic monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop for continuous diagnostics"""
        while self.monitoring_active:
            try:
                # Perform lightweight system checks
                metrics, issues = self.system_analyzer.analyze_system_resources()
                
                # Log critical issues immediately
                for issue in issues:
                    if issue.severity == "critical":
                        logger.warning(f"Critical diagnostic issue: {issue.title}")
                
                # Sleep before next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in diagnostic monitoring loop: {e}")
                time.sleep(60)  # Longer sleep on error
    
    def run_comprehensive_diagnosis(self) -> DiagnosticResults:
        """Run comprehensive system diagnosis"""
        session_id = f"diag_{int(time.time())}"
        start_time = datetime.now()
        
        logger.info("Starting comprehensive system diagnosis")
        
        results = DiagnosticResults(
            session_id=session_id,
            start_time=start_time
        )
        
        try:
            # Analyze system resources and configuration
            metrics, resource_issues = self.system_analyzer.analyze_system_resources()
            config_issues = self.system_analyzer.analyze_configuration()
            dependency_issues = self.system_analyzer.analyze_dependencies()
            
            # Analyze error logs
            error_log_analysis = self.error_log_analyzer.analyze_error_logs()
            
            # Combine all issues
            all_issues = resource_issues + config_issues + dependency_issues
            
            # Calculate system health score
            health_score = self._calculate_health_score(all_issues, metrics)
            
            # Generate recommendations
            recommendations = self._generate_system_recommendations(all_issues, error_log_analysis)
            
            # Create system analysis
            system_analysis = SystemAnalysis(
                timestamp=datetime.now(),
                system_health_score=health_score,
                resource_status=metrics,
                configuration_issues=config_issues,
                performance_issues=resource_issues,
                dependency_issues=dependency_issues,
                recommendations=recommendations
            )
            
            # Determine overall status
            overall_status = self._determine_overall_status(all_issues)
            
            # Update results
            results.end_time = datetime.now()
            results.system_analysis = system_analysis
            results.error_log_analysis = error_log_analysis
            results.issues_found = all_issues
            results.overall_status = overall_status
            
            logger.info(f"Diagnosis completed. Health score: {health_score:.1f}, Status: {overall_status}")
            
        except Exception as e:
            logger.error(f"Failed to complete diagnosis: {e}")
            results.overall_status = "error"
            results.end_time = datetime.now()
        
        return results
    
    def _calculate_health_score(self, issues: List[DiagnosticIssue], metrics: ResourceMetrics) -> float:
        """Calculate system health score (0-100)"""
        base_score = 100.0
        
        # Deduct points for issues
        for issue in issues:
            if issue.severity == "critical":
                base_score -= 20
            elif issue.severity == "warning":
                base_score -= 10
            else:
                base_score -= 5
        
        # Deduct points for resource usage
        if metrics.cpu_percent > 80:
            base_score -= (metrics.cpu_percent - 80) * 0.5
        
        if metrics.memory_percent > 80:
            base_score -= (metrics.memory_percent - 80) * 0.5
        
        if metrics.vram_percent > 80:
            base_score -= (metrics.vram_percent - 80) * 0.5
        
        return max(0.0, min(100.0, base_score))
    
    def _generate_system_recommendations(self, issues: List[DiagnosticIssue], error_analysis: Dict[str, Any]) -> List[str]:
        """Generate system-wide recommendations"""
        recommendations = []
        
        # Add issue-specific recommendations
        for issue in issues:
            if issue.auto_recoverable:
                recommendations.extend(issue.remediation_steps[:2])  # Top 2 steps
        
        # Add error log recommendations
        if error_analysis.get('recommendations'):
            recommendations.extend(error_analysis['recommendations'][:3])  # Top 3
        
        # Add general recommendations
        recommendations.extend([
            "Regularly monitor system resources",
            "Keep dependencies updated",
            "Maintain adequate free disk space",
            "Monitor error logs for patterns"
        ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]  # Top 10 recommendations
    
    def _determine_overall_status(self, issues: List[DiagnosticIssue]) -> str:
        """Determine overall system status"""
        critical_count = sum(1 for issue in issues if issue.severity == "critical")
        warning_count = sum(1 for issue in issues if issue.severity == "warning")
        
        if critical_count > 0:
            return "critical"
        elif warning_count > 2:
            return "warning"
        elif warning_count > 0:
            return "warning"
        else:
            return "healthy"
    
    def run_specialized_diagnosis(self, diagnosis_type: str) -> List[DiagnosticIssue]:
        """Run specialized diagnosis for specific issue types"""
        if diagnosis_type == "cuda":
            return self.diagnose_cuda_issues()
        elif diagnosis_type == "memory":
            return self.diagnose_memory_issues()
        elif diagnosis_type == "model_download":
            return self.diagnose_model_download_issues()
        else:
            logger.warning(f"Unknown diagnosis type: {diagnosis_type}")
            return []
    
    def attempt_issue_recovery(self, issues: List[DiagnosticIssue]) -> List[Dict[str, Any]]:
        """Attempt automatic recovery for a list of issues"""
        recovery_attempts = []
        
        for issue in issues:
            if issue.auto_recoverable:
                logger.info(f"Attempting recovery for: {issue.title}")
                recovery_result = self.recovery_manager.attempt_automatic_recovery(issue)
                recovery_attempts.append(recovery_result)
                
                if recovery_result["success"]:
                    logger.info(f"Successfully recovered from: {issue.title}")
                else:
                    logger.warning(f"Failed to recover from: {issue.title}")
        
        return recovery_attempts
    
    def generate_diagnostic_report(self, results: DiagnosticResults, output_path: Optional[str] = None) -> str:
        """Generate comprehensive diagnostic report"""
        report = self.report_generator.generate_comprehensive_report(results)
        
        if output_path:
            return self.report_generator.save_report_to_file(report, output_path)
        else:
            return json.dumps(report, indent=2, default=str)
    
    def integrate_with_error_handler(self, error_info: ErrorInfo) -> DiagnosticResults:
        """Integrate diagnostic analysis with existing error handler system"""
        session_id = f"error_diag_{int(time.time())}"
        start_time = datetime.now()
        
        results = DiagnosticResults(
            session_id=session_id,
            start_time=start_time
        )
        
        try:
            # Convert error info to diagnostic issue
            diagnostic_issue = self._convert_error_to_diagnostic_issue(error_info)
            results.issues_found.append(diagnostic_issue)
            
            # Run targeted diagnosis based on error category
            additional_issues = []
            if error_info.category == ErrorCategory.VRAM_ERROR:
                additional_issues = self.diagnose_cuda_issues()
            elif error_info.category == ErrorCategory.MODEL_LOADING_ERROR:
                additional_issues = self.diagnose_model_download_issues()
            elif error_info.category in [ErrorCategory.SYSTEM_ERROR, ErrorCategory.UNKNOWN_ERROR]:
                additional_issues = self.diagnose_memory_issues()
            
            results.issues_found.extend(additional_issues)
            
            # Attempt recovery
            recovery_attempts = self.attempt_issue_recovery(results.issues_found)
            results.recovery_attempts = recovery_attempts
            
            # Determine overall status
            results.overall_status = self._determine_overall_status(results.issues_found)
            results.end_time = datetime.now()
            
            logger.info(f"Error-triggered diagnosis completed for {error_info.category.value}")
            
        except Exception as e:
            logger.error(f"Failed to complete error-triggered diagnosis: {e}")
            results.overall_status = "error"
            results.end_time = datetime.now()
        
        return results
    
    def _convert_error_to_diagnostic_issue(self, error_info: ErrorInfo) -> DiagnosticIssue:
        """Convert ErrorInfo to DiagnosticIssue"""
        # Map error categories to diagnostic categories
        category_mapping = {
            ErrorCategory.VRAM_ERROR: DiagnosticCategory.CUDA_ERROR,
            ErrorCategory.MODEL_LOADING_ERROR: DiagnosticCategory.MODEL_DOWNLOAD_ERROR,
            ErrorCategory.GENERATION_ERROR: DiagnosticCategory.SYSTEM_RESOURCE_ERROR,
            ErrorCategory.FILE_IO_ERROR: DiagnosticCategory.FILE_PERMISSION_ERROR,
            ErrorCategory.NETWORK_ERROR: DiagnosticCategory.NETWORK_ERROR,
            ErrorCategory.VALIDATION_ERROR: DiagnosticCategory.CONFIGURATION_ERROR,
            ErrorCategory.SYSTEM_ERROR: DiagnosticCategory.SYSTEM_RESOURCE_ERROR,
            ErrorCategory.UI_ERROR: DiagnosticCategory.CONFIGURATION_ERROR,
            ErrorCategory.UNKNOWN_ERROR: DiagnosticCategory.UNKNOWN_ERROR
        }
        
        diagnostic_category = category_mapping.get(error_info.category, DiagnosticCategory.UNKNOWN_ERROR)
        
        return DiagnosticIssue(
            category=diagnostic_category,
            severity="critical" if error_info.retry_count >= error_info.max_retries else "warning",
            title=f"Error: {error_info.error_type}",
            description=error_info.user_message,
            affected_components=[error_info.function_name] if error_info.function_name else ["unknown"],
            symptoms=[error_info.message],
            root_cause=error_info.message,
            remediation_steps=error_info.recovery_suggestions,
            auto_recoverable=error_info.is_recoverable,
            timestamp=error_info.timestamp
        )
    
    def diagnose_cuda_issues(self) -> List[DiagnosticIssue]:
        """Specialized CUDA error diagnosis with specific recommendations"""
        issues = []
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                issues.append(DiagnosticIssue(
                    category=DiagnosticCategory.CUDA_ERROR,
                    severity="critical",
                    title="CUDA Not Available",
                    description="CUDA is not available for PyTorch operations",
                    affected_components=["gpu_acceleration", "model_inference"],
                    symptoms=["CPU-only processing", "Slow generation times", "No GPU utilization"],
                    root_cause="CUDA drivers not installed or PyTorch compiled without CUDA support",
                    remediation_steps=[
                        "Install NVIDIA CUDA drivers from nvidia.com",
                        "Install PyTorch with CUDA support: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118",
                        "Verify GPU is detected: nvidia-smi",
                        "Check CUDA_VISIBLE_DEVICES environment variable"
                    ],
                    auto_recoverable=False
                ))
                return issues
            
            # Check for CUDA out of memory patterns in error logs
            cuda_memory_errors = self._check_cuda_memory_errors()
            if cuda_memory_errors:
                issues.append(DiagnosticIssue(
                    category=DiagnosticCategory.CUDA_ERROR,
                    severity="critical",
                    title="CUDA Out of Memory Detected",
                    description=f"Found {cuda_memory_errors} CUDA out of memory errors in recent logs",
                    affected_components=["model_loading", "inference", "generation"],
                    symptoms=["Generation failures", "Model loading errors", "Application crashes"],
                    root_cause="GPU memory exhausted during operations",
                    remediation_steps=[
                        "Enable attention slicing: config['optimization']['enable_attention_slicing'] = True",
                        "Reduce VAE tile size: config['optimization']['vae_tile_size'] = 128",
                        "Enable model offloading: config['optimization']['enable_cpu_offload'] = True",
                        "Use lower precision: config['optimization']['use_fp16'] = True",
                        "Reduce generation resolution or batch size"
                    ],
                    auto_recoverable=True
                ))
            
            # Check current VRAM usage
            if torch.cuda.is_available():
                current_vram = torch.cuda.memory_allocated(0) / (1024**2)  # MB
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB
                vram_percent = (current_vram / total_vram) * 100
                
                if vram_percent > 85:
                    issues.append(DiagnosticIssue(
                        category=DiagnosticCategory.CUDA_ERROR,
                        severity="warning",
                        title="High VRAM Usage",
                        description=f"VRAM usage at {vram_percent:.1f}% ({current_vram:.0f}MB/{total_vram:.0f}MB)",
                        affected_components=["gpu_performance"],
                        symptoms=["Potential memory errors", "Reduced performance"],
                        root_cause="High GPU memory utilization",
                        remediation_steps=[
                            "Clear GPU cache: torch.cuda.empty_cache()",
                            "Enable attention slicing",
                            "Reduce model precision to fp16",
                            "Enable gradient checkpointing"
                        ],
                        auto_recoverable=True
                    ))
            
            # Check for GPU fragmentation
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                
                if reserved > allocated * 1.5:  # More than 50% fragmentation
                    issues.append(DiagnosticIssue(
                        category=DiagnosticCategory.CUDA_ERROR,
                        severity="warning",
                        title="GPU Memory Fragmentation",
                        description=f"GPU memory fragmented: {allocated/(1024**2):.0f}MB allocated, {reserved/(1024**2):.0f}MB reserved",
                        affected_components=["memory_efficiency"],
                        symptoms=["Inefficient memory usage", "Potential allocation failures"],
                        root_cause="GPU memory fragmentation from repeated allocations",
                        remediation_steps=[
                            "Clear GPU cache: torch.cuda.empty_cache()",
                            "Restart application to defragment memory",
                            "Use memory pooling strategies",
                            "Enable memory-efficient attention"
                        ],
                        auto_recoverable=True
                    ))
                    
        except ImportError:
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.DEPENDENCY_ERROR,
                severity="critical",
                title="PyTorch Not Available",
                description="PyTorch is not installed or not importable",
                affected_components=["core_functionality"],
                symptoms=["Import errors", "Application startup failures"],
                root_cause="PyTorch package not installed",
                remediation_steps=[
                    "Install PyTorch: pip install torch torchvision",
                    "Check virtual environment activation",
                    "Verify Python version compatibility"
                ],
                auto_recoverable=True
            ))
        except Exception as e:
            logger.error(f"Error in CUDA diagnosis: {e}")
        
        return issues
    
    def diagnose_model_download_issues(self) -> List[DiagnosticIssue]:
        """Specialized model download failure diagnosis with HF_TOKEN validation"""
        issues = []
        
        # Check HF_TOKEN availability
        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token:
            # Check .env file
            env_file = Path('.env')
            if env_file.exists():
                try:
                    with open(env_file, 'r') as f:
                        env_content = f.read()
                    if 'HF_TOKEN' not in env_content:
                        hf_token_missing = True
                    else:
                        hf_token_missing = False
                except Exception:
                    hf_token_missing = True
            else:
                hf_token_missing = True
            
            if hf_token_missing:
                issues.append(DiagnosticIssue(
                    category=DiagnosticCategory.MODEL_DOWNLOAD_ERROR,
                    severity="critical",
                    title="Missing HF_TOKEN",
                    description="Hugging Face token not found in environment or .env file",
                    affected_components=["model_downloads", "authentication"],
                    symptoms=["Model download failures", "Authentication errors", "403 Forbidden errors"],
                    root_cause="Hugging Face authentication token not configured",
                    remediation_steps=[
                        "Get token from https://huggingface.co/settings/tokens",
                        "Add HF_TOKEN=your_token to .env file",
                        "Ensure token has 'read' permissions",
                        "Restart application to load new environment variables"
                    ],
                    auto_recoverable=False
                ))
        
        # Check for model download errors in logs
        download_errors = self._check_model_download_errors()
        if download_errors > 0:
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.MODEL_DOWNLOAD_ERROR,
                severity="warning",
                title="Model Download Failures Detected",
                description=f"Found {download_errors} model download failures in recent logs",
                affected_components=["model_loading"],
                symptoms=["Model loading failures", "Network timeouts", "Incomplete downloads"],
                root_cause="Network issues or authentication problems",
                remediation_steps=[
                    "Check internet connection stability",
                    "Verify HF_TOKEN is valid and has permissions",
                    "Clear model cache and retry: get_model_manager().clear_cache()",
                    "Try downloading models manually",
                    "Check firewall/proxy settings"
                ],
                auto_recoverable=True
            ))
        
        # Check disk space for model storage
        models_dir = Path("models")
        if models_dir.exists():
            try:
                disk_usage = psutil.disk_usage(str(models_dir))
                free_gb = disk_usage.free / (1024**3)
                
                if free_gb < 20:  # Less than 20GB free
                    issues.append(DiagnosticIssue(
                        category=DiagnosticCategory.MODEL_DOWNLOAD_ERROR,
                        severity="warning",
                        title="Low Disk Space for Models",
                        description=f"Only {free_gb:.1f}GB free space available for model storage",
                        affected_components=["model_downloads", "model_caching"],
                        symptoms=["Download failures", "Incomplete model files", "Cache errors"],
                        root_cause="Insufficient disk space for model files",
                        remediation_steps=[
                            "Clear old model cache files",
                            "Move models to external storage",
                            "Free up disk space",
                            "Configure model cache location"
                        ],
                        auto_recoverable=True
                    ))
            except Exception as e:
                logger.debug(f"Failed to check disk space: {e}")
        
        # Check network connectivity to Hugging Face
        try:
            import requests
            try:
                response = requests.get("https://huggingface.co", timeout=10)
                if response.status_code != 200:
                    issues.append(DiagnosticIssue(
                        category=DiagnosticCategory.NETWORK_ERROR,
                        severity="warning",
                        title="Hugging Face Hub Connectivity Issue",
                        description=f"Cannot reach Hugging Face Hub (status: {response.status_code})",
                        affected_components=["model_downloads"],
                        symptoms=["Download timeouts", "Connection errors"],
                        root_cause="Network connectivity issues to Hugging Face Hub",
                        remediation_steps=[
                            "Check internet connection",
                            "Verify DNS resolution",
                            "Check firewall settings",
                            "Try using VPN if blocked regionally"
                        ],
                        auto_recoverable=False
                    ))
            except requests.exceptions.RequestException:
                issues.append(DiagnosticIssue(
                    category=DiagnosticCategory.NETWORK_ERROR,
                    severity="warning",
                    title="Network Connectivity Issue",
                    description="Cannot connect to Hugging Face Hub",
                    affected_components=["model_downloads"],
                    symptoms=["Download failures", "Timeout errors"],
                    root_cause="Network connectivity problems",
                    remediation_steps=[
                        "Check internet connection",
                        "Verify network settings",
                        "Check proxy configuration",
                        "Try downloading from different network"
                    ],
                    auto_recoverable=False
                ))
        except ImportError:
            logger.debug("Requests library not available for network check")
        
        return issues
    
    def diagnose_memory_issues(self) -> List[DiagnosticIssue]:
        """Specialized memory issue analyzer with optimization recommendations"""
        issues = []
        
        # Check system memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        
        if memory_percent > 90:
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.MEMORY_ERROR,
                severity="critical",
                title="Critical Memory Usage",
                description=f"System memory at {memory_percent:.1f}% usage",
                affected_components=["system_stability", "model_loading"],
                symptoms=["System slowdown", "Application crashes", "Swap usage"],
                root_cause="Insufficient available system memory",
                remediation_steps=[
                    "Close unnecessary applications",
                    "Enable model offloading: config['optimization']['enable_cpu_offload'] = True",
                    "Use memory-efficient attention: config['optimization']['enable_memory_efficient_attention'] = True",
                    "Reduce model precision to int8",
                    "Restart system to clear memory leaks"
                ],
                auto_recoverable=True
            ))
        elif memory_percent > 80:
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.MEMORY_ERROR,
                severity="warning",
                title="High Memory Usage",
                description=f"System memory at {memory_percent:.1f}% usage",
                affected_components=["performance"],
                symptoms=["Slower performance", "Increased swap usage"],
                root_cause="High memory utilization",
                remediation_steps=[
                    "Monitor memory usage patterns",
                    "Enable memory optimizations",
                    "Consider model offloading",
                    "Close unused applications"
                ],
                auto_recoverable=True
            ))
        
        if memory_available_gb < 2:
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.MEMORY_ERROR,
                severity="critical",
                title="Low Available Memory",
                description=f"Only {memory_available_gb:.1f}GB memory available",
                affected_components=["model_loading", "generation"],
                symptoms=["Out of memory errors", "Model loading failures"],
                root_cause="Insufficient free memory for operations",
                remediation_steps=[
                    "Free up system memory immediately",
                    "Enable aggressive memory optimizations",
                    "Use smaller models or lower precision",
                    "Increase virtual memory/swap space"
                ],
                auto_recoverable=True
            ))
        
        # Check for memory leaks by analyzing process memory growth
        try:
            current_process = psutil.Process()
            process_memory_mb = current_process.memory_info().rss / (1024**2)
            
            if process_memory_mb > 8000:  # More than 8GB for this process
                issues.append(DiagnosticIssue(
                    category=DiagnosticCategory.MEMORY_ERROR,
                    severity="warning",
                    title="High Process Memory Usage",
                    description=f"Application using {process_memory_mb:.0f}MB of memory",
                    affected_components=["application_performance"],
                    symptoms=["Slow response", "Memory pressure"],
                    root_cause="High memory usage by application process",
                    remediation_steps=[
                        "Restart application to clear memory",
                        "Enable memory-efficient settings",
                        "Check for memory leaks",
                        "Use model offloading"
                    ],
                    auto_recoverable=True
                ))
        except Exception as e:
            logger.debug(f"Failed to check process memory: {e}")
        
        # Check swap usage
        try:
            swap = psutil.swap_memory()
            if swap.percent > 50:
                issues.append(DiagnosticIssue(
                    category=DiagnosticCategory.MEMORY_ERROR,
                    severity="warning",
                    title="High Swap Usage",
                    description=f"Swap usage at {swap.percent:.1f}%",
                    affected_components=["system_performance"],
                    symptoms=["Very slow performance", "Disk thrashing"],
                    root_cause="System using swap memory due to RAM shortage",
                    remediation_steps=[
                        "Free up RAM to reduce swap usage",
                        "Add more physical RAM",
                        "Enable memory optimizations",
                        "Restart system to clear swap"
                    ],
                    auto_recoverable=True
                ))
        except Exception as e:
            logger.debug(f"Failed to check swap usage: {e}")
        
        return issues
    
    def _check_cuda_memory_errors(self) -> int:
        """Check error logs for CUDA out of memory errors"""
        try:
            if not self.error_log_analyzer.log_file_path.exists():
                return 0
            
            with open(self.error_log_analyzer.log_file_path, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # Count CUDA out of memory occurrences
            cuda_oom_patterns = [
                r'CUDA out of memory',
                r'OutOfMemoryError',
                r'cuda.*memory.*error',
                r'GPU memory.*exhausted'
            ]
            
            count = 0
            for pattern in cuda_oom_patterns:
                matches = re.findall(pattern, log_content, re.IGNORECASE)
                count += len(matches)
            
            return count
            
        except Exception as e:
            logger.debug(f"Failed to check CUDA memory errors: {e}")
            return 0
    
    def _check_model_download_errors(self) -> int:
        """Check error logs for model download errors"""
        try:
            if not self.error_log_analyzer.log_file_path.exists():
                return 0
            
            with open(self.error_log_analyzer.log_file_path, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # Count model download error occurrences
            download_error_patterns = [
                r'Failed to.*download',
                r'ConnectionError',
                r'TimeoutError',
                r'403.*Forbidden',
                r'401.*Unauthorized',
                r'model.*download.*failed',
                r'huggingface.*error'
            ]
            
            count = 0
            for pattern in download_error_patterns:
                matches = re.findall(pattern, log_content, re.IGNORECASE)
                count += len(matches)
            
            return count
            
        except Exception as e:
            logger.debug(f"Failed to check model download errors: {e}")
            return 0


class RecoveryManager:
    """Automated recovery system for diagnostic issues"""
    
    def __init__(self):
        self.recovery_history: List[Dict[str, Any]] = []
        self.error_recovery_manager = get_error_recovery_manager()
    
    def attempt_automatic_recovery(self, issue: DiagnosticIssue) -> Dict[str, Any]:
        """Attempt automatic recovery for a diagnostic issue"""
        recovery_attempt = {
            "issue_title": issue.title,
            "issue_category": issue.category.value,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "actions_taken": [],
            "error_message": None
        }
        
        try:
            if not issue.auto_recoverable:
                recovery_attempt["error_message"] = "Issue marked as not auto-recoverable"
                return recovery_attempt
            
            # Attempt recovery based on issue category
            if issue.category == DiagnosticCategory.CUDA_ERROR:
                success = self._recover_cuda_issue(issue, recovery_attempt)
            elif issue.category == DiagnosticCategory.MEMORY_ERROR:
                success = self._recover_memory_issue(issue, recovery_attempt)
            elif issue.category == DiagnosticCategory.MODEL_DOWNLOAD_ERROR:
                success = self._recover_model_download_issue(issue, recovery_attempt)
            elif issue.category == DiagnosticCategory.CONFIGURATION_ERROR:
                success = self._recover_configuration_issue(issue, recovery_attempt)
            elif issue.category == DiagnosticCategory.SYSTEM_RESOURCE_ERROR:
                success = self._recover_system_resource_issue(issue, recovery_attempt)
            else:
                recovery_attempt["error_message"] = f"No recovery strategy for category: {issue.category.value}"
                success = False
            
            recovery_attempt["success"] = success
            
        except Exception as e:
            recovery_attempt["error_message"] = str(e)
            recovery_attempt["success"] = False
            logger.error(f"Recovery attempt failed for {issue.title}: {e}")
        
        # Store recovery attempt
        self.recovery_history.append(recovery_attempt)
        
        # Limit history size
        if len(self.recovery_history) > 100:
            self.recovery_history = self.recovery_history[-100:]
        
        return recovery_attempt
    
    def _recover_cuda_issue(self, issue: DiagnosticIssue, recovery_attempt: Dict[str, Any]) -> bool:
        """Recover from CUDA-related issues"""
        actions_taken = []
        
        try:
            import torch
            
            if "memory" in issue.title.lower() or "vram" in issue.title.lower():
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    actions_taken.append("Cleared GPU cache")
                
                # Force garbage collection
                import gc
                gc.collect()
                actions_taken.append("Forced garbage collection")
                
                # Wait for memory to be freed
                time.sleep(2)
                actions_taken.append("Waited for memory cleanup")
                
                recovery_attempt["actions_taken"] = actions_taken
                return True
            
        except ImportError:
            actions_taken.append("PyTorch not available for CUDA recovery")
        except Exception as e:
            actions_taken.append(f"CUDA recovery failed: {e}")
        
        recovery_attempt["actions_taken"] = actions_taken
        return False
    
    def _recover_memory_issue(self, issue: DiagnosticIssue, recovery_attempt: Dict[str, Any]) -> bool:
        """Recover from memory-related issues"""
        actions_taken = []
        
        try:
            # Force garbage collection
            import gc
            gc.collect()
            actions_taken.append("Forced garbage collection")
            
            # Clear GPU cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    actions_taken.append("Cleared GPU cache")
            except ImportError:
                pass
            
            # Try to free up some system memory
            try:
                # Clear any large variables from global scope
                import sys
                if hasattr(sys, '_clear_type_cache'):
                    sys._clear_type_cache()
                    actions_taken.append("Cleared type cache")
            except Exception:
                pass
            
            recovery_attempt["actions_taken"] = actions_taken
            return True
            
        except Exception as e:
            actions_taken.append(f"Memory recovery failed: {e}")
            recovery_attempt["actions_taken"] = actions_taken
            return False
    
    def _recover_model_download_issue(self, issue: DiagnosticIssue, recovery_attempt: Dict[str, Any]) -> bool:
        """Recover from model download issues"""
        actions_taken = []
        
        try:
            # Clear model cache if possible
            try:
                # Try to import and use model manager
                from utils import get_model_manager
                model_manager = get_model_manager()
                if hasattr(model_manager, 'clear_cache'):
                    model_manager.clear_cache()
                    actions_taken.append("Cleared model cache")
            except ImportError:
                # Fallback: clear common cache directories
                cache_dirs = [
                    Path.home() / ".cache" / "huggingface",
                    Path("models"),
                    Path(".cache")
                ]
                
                for cache_dir in cache_dirs:
                    if cache_dir.exists():
                        try:
                            # Only clear if it's safe to do so
                            temp_files = list(cache_dir.glob("*.tmp"))
                            for temp_file in temp_files:
                                temp_file.unlink()
                            actions_taken.append(f"Cleared temporary files from {cache_dir}")
                        except Exception:
                            pass
            
            # Wait a moment before retry
            time.sleep(1)
            actions_taken.append("Waited before retry")
            
            recovery_attempt["actions_taken"] = actions_taken
            return True
            
        except Exception as e:
            actions_taken.append(f"Model download recovery failed: {e}")
            recovery_attempt["actions_taken"] = actions_taken
            return False
    
    def _recover_configuration_issue(self, issue: DiagnosticIssue, recovery_attempt: Dict[str, Any]) -> bool:
        """Recover from configuration issues"""
        actions_taken = []
        
        try:
            # Create missing directories
            required_dirs = ["models", "outputs", "loras"]
            for dir_name in required_dirs:
                dir_path = Path(dir_name)
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    actions_taken.append(f"Created directory: {dir_name}")
            
            # Create basic .env file if missing
            env_file = Path(".env")
            if not env_file.exists() and "env" in issue.title.lower():
                with open(env_file, 'w') as f:
                    f.write("# Environment variables for Wan2.2 UI\n")
                    f.write("# HF_TOKEN=your_huggingface_token_here\n")
                    f.write("# CUDA_VISIBLE_DEVICES=0\n")
                actions_taken.append("Created basic .env template")
            
            recovery_attempt["actions_taken"] = actions_taken
            return len(actions_taken) > 0
            
        except Exception as e:
            actions_taken.append(f"Configuration recovery failed: {e}")
            recovery_attempt["actions_taken"] = actions_taken
            return False
    
    def _recover_system_resource_issue(self, issue: DiagnosticIssue, recovery_attempt: Dict[str, Any]) -> bool:
        """Recover from system resource issues"""
        actions_taken = []
        
        try:
            # Clean up temporary files
            temp_dirs = [Path("outputs"), Path(".cache"), Path("temp")]
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    try:
                        # Remove old temporary files (older than 1 day)
                        cutoff_time = time.time() - (24 * 60 * 60)  # 24 hours ago
                        for file_path in temp_dir.rglob("*"):
                            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                                if file_path.suffix in ['.tmp', '.temp', '.log']:
                                    file_path.unlink()
                                    actions_taken.append(f"Removed old temp file: {file_path.name}")
                    except Exception:
                        pass
            
            # Force garbage collection
            import gc
            gc.collect()
            actions_taken.append("Forced garbage collection")
            
            recovery_attempt["actions_taken"] = actions_taken
            return len(actions_taken) > 0
            
        except Exception as e:
            actions_taken.append(f"System resource recovery failed: {e}")
            recovery_attempt["actions_taken"] = actions_taken
            return False
    
    def get_recovery_history(self) -> List[Dict[str, Any]]:
        """Get history of recovery attempts"""
        return self.recovery_history.copy()


class DiagnosticReportGenerator:
    """Generates comprehensive diagnostic reports with issue categorization"""
    
    def __init__(self):
        pass
    
    def generate_comprehensive_report(self, diagnostic_results: DiagnosticResults) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report"""
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "session_id": diagnostic_results.session_id,
                "duration_seconds": self._calculate_duration(diagnostic_results),
                "report_version": "1.0"
            },
            "executive_summary": self._generate_executive_summary(diagnostic_results),
            "system_health": self._generate_system_health_section(diagnostic_results),
            "issues_analysis": self._generate_issues_analysis(diagnostic_results),
            "error_log_summary": diagnostic_results.error_log_analysis or {},
            "recovery_attempts": diagnostic_results.recovery_attempts,
            "recommendations": self._generate_prioritized_recommendations(diagnostic_results),
            "technical_details": self._generate_technical_details(diagnostic_results)
        }
        
        return report
    
    def _calculate_duration(self, results: DiagnosticResults) -> float:
        """Calculate diagnosis duration in seconds"""
        if results.end_time and results.start_time:
            return (results.end_time - results.start_time).total_seconds()
        return 0.0
    
    def _generate_executive_summary(self, results: DiagnosticResults) -> Dict[str, Any]:
        """Generate executive summary of diagnostic results"""
        critical_issues = [issue for issue in results.issues_found if issue.severity == "critical"]
        warning_issues = [issue for issue in results.issues_found if issue.severity == "warning"]
        
        health_score = 100.0
        if results.system_analysis:
            health_score = results.system_analysis.system_health_score
        
        return {
            "overall_status": results.overall_status,
            "health_score": health_score,
            "total_issues": len(results.issues_found),
            "critical_issues": len(critical_issues),
            "warning_issues": len(warning_issues),
            "recovery_attempts": len(results.recovery_attempts),
            "key_findings": [
                issue.title for issue in critical_issues[:3]  # Top 3 critical issues
            ],
            "immediate_actions_required": len(critical_issues) > 0
        }
    
    def _generate_system_health_section(self, results: DiagnosticResults) -> Dict[str, Any]:
        """Generate system health section"""
        if not results.system_analysis:
            return {"status": "not_analyzed"}
        
        analysis = results.system_analysis
        
        return {
            "health_score": analysis.system_health_score,
            "resource_status": {
                "cpu_percent": analysis.resource_status.cpu_percent,
                "memory_percent": analysis.resource_status.memory_percent,
                "memory_used_gb": analysis.resource_status.memory_used_gb,
                "vram_percent": analysis.resource_status.vram_percent,
                "vram_used_mb": analysis.resource_status.vram_used_mb
            },
            "health_indicators": {
                "cpu_healthy": analysis.resource_status.cpu_percent < 80,
                "memory_healthy": analysis.resource_status.memory_percent < 85,
                "vram_healthy": analysis.resource_status.vram_percent < 90,
                "configuration_healthy": len(analysis.configuration_issues) == 0,
                "dependencies_healthy": len(analysis.dependency_issues) == 0
            }
        }
    
    def _generate_issues_analysis(self, results: DiagnosticResults) -> Dict[str, Any]:
        """Generate detailed issues analysis"""
        issues_by_category = {}
        issues_by_severity = {"critical": [], "warning": [], "info": []}
        
        for issue in results.issues_found:
            # Group by category
            category = issue.category.value
            if category not in issues_by_category:
                issues_by_category[category] = []
            issues_by_category[category].append(issue.to_dict())
            
            # Group by severity
            if issue.severity in issues_by_severity:
                issues_by_severity[issue.severity].append(issue.to_dict())
        
        return {
            "by_category": issues_by_category,
            "by_severity": issues_by_severity,
            "most_common_categories": self._get_most_common_categories(results.issues_found),
            "auto_recoverable_count": sum(1 for issue in results.issues_found if issue.auto_recoverable)
        }
    
    def _get_most_common_categories(self, issues: List[DiagnosticIssue]) -> List[Tuple[str, int]]:
        """Get most common issue categories"""
        category_counts = {}
        for issue in issues:
            category = issue.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    
    def _generate_prioritized_recommendations(self, results: DiagnosticResults) -> List[Dict[str, Any]]:
        """Generate prioritized recommendations"""
        recommendations = []
        
        # Add critical issue recommendations first
        critical_issues = [issue for issue in results.issues_found if issue.severity == "critical"]
        for issue in critical_issues[:3]:  # Top 3 critical
            recommendations.append({
                "priority": "critical",
                "category": issue.category.value,
                "title": f"Address {issue.title}",
                "actions": issue.remediation_steps[:3],  # Top 3 actions
                "auto_recoverable": issue.auto_recoverable
            })
        
        # Add system-wide recommendations
        if results.system_analysis:
            for rec in results.system_analysis.recommendations[:5]:  # Top 5
                recommendations.append({
                    "priority": "high",
                    "category": "system_optimization",
                    "title": "System Optimization",
                    "actions": [rec],
                    "auto_recoverable": True
                })
        
        return recommendations
    
    def _generate_technical_details(self, results: DiagnosticResults) -> Dict[str, Any]:
        """Generate technical details section"""
        return {
            "diagnosis_timestamp": results.start_time.isoformat(),
            "diagnosis_duration_seconds": self._calculate_duration(results),
            "system_analysis_available": results.system_analysis is not None,
            "error_log_analysis_available": results.error_log_analysis is not None,
            "total_recovery_attempts": len(results.recovery_attempts),
            "successful_recoveries": sum(1 for attempt in results.recovery_attempts if attempt.get("success", False))
        }
    
    def save_report_to_file(self, report: Dict[str, Any], output_path: str) -> str:
        """Save diagnostic report to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Diagnostic report saved to {output_file}")
        return str(output_file)
