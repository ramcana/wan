"""
WAN22 Recovery Workflows and Advanced Logging

This module implements comprehensive logging with stack traces, system state capture,
log rotation, and user-guided recovery workflows for complex issues.

Requirements addressed:
- 6.4: Create comprehensive error logging with stack traces and system state
- 6.5: Implement log rotation system to prevent disk space issues
- 6.6: Add user-guided recovery workflows for complex issues
- 6.7: Provide clear instructions for manual resolution
"""

import json
import logging
import logging.handlers
import os
import shutil
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
import threading
import zipfile
import psutil
import platform


class LogLevel(Enum):
    """Enhanced log levels for recovery workflows"""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    RECOVERY = "RECOVERY"
    USER_ACTION = "USER_ACTION"


class WorkflowStep(Enum):
    """User-guided recovery workflow steps"""
    DIAGNOSIS = "diagnosis"
    PREPARATION = "preparation"
    EXECUTION = "execution"
    VALIDATION = "validation"
    COMPLETION = "completion"


@dataclass
class LogEntry:
    """Structured log entry with comprehensive context"""
    timestamp: datetime
    level: LogLevel
    component: str
    message: str
    stack_trace: Optional[str]
    system_state: Dict[str, Any]
    user_context: Dict[str, Any]
    recovery_context: Dict[str, Any]
    session_id: str
    thread_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "component": self.component,
            "message": self.message,
            "stack_trace": self.stack_trace,
            "system_state": self.system_state,
            "user_context": self.user_context,
            "recovery_context": self.recovery_context,
            "session_id": self.session_id,
            "thread_id": self.thread_id
        }


@dataclass
class RecoveryWorkflow:
    """User-guided recovery workflow definition"""
    workflow_id: str
    title: str
    description: str
    error_patterns: List[str]
    steps: List['WorkflowStepDefinition']
    estimated_time: int  # minutes
    difficulty_level: str  # "beginner", "intermediate", "advanced"
    prerequisites: List[str]
    success_criteria: List[str]


@dataclass
class WorkflowStepDefinition:
    """Individual step in a recovery workflow"""
    step_id: str
    title: str
    description: str
    step_type: WorkflowStep
    instructions: List[str]
    validation_checks: List[str]
    expected_outcomes: List[str]
    troubleshooting_tips: List[str]
    automated_actions: Optional[Callable] = None
    user_input_required: bool = False
    critical_step: bool = False


@dataclass
class WorkflowExecution:
    """Tracking of workflow execution progress"""
    workflow_id: str
    execution_id: str
    start_time: datetime
    current_step: int
    completed_steps: List[str]
    failed_steps: List[str]
    user_responses: Dict[str, Any]
    system_changes: List[str]
    status: str  # "running", "completed", "failed", "paused"
    end_time: Optional[datetime] = None


class LogRotationManager:
    """Advanced log rotation with compression and cleanup"""
    
    def __init__(self, 
                 log_dir: Path,
                 max_file_size: int = 50 * 1024 * 1024,  # 50MB
                 max_files: int = 10,
                 compress_old_logs: bool = True,
                 cleanup_after_days: int = 30):
        """
        Initialize log rotation manager.
        
        Args:
            log_dir: Directory containing log files
            max_file_size: Maximum size per log file in bytes
            max_files: Maximum number of log files to keep
            compress_old_logs: Whether to compress rotated logs
            cleanup_after_days: Days after which to delete old logs
        """
        self.log_dir = log_dir
        self.max_file_size = max_file_size
        self.max_files = max_files
        self.compress_old_logs = compress_old_logs
        self.cleanup_after_days = cleanup_after_days
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup rotation handler
        self._setup_rotation_handler()
    
    def _setup_rotation_handler(self):
        """Setup rotating file handler with compression"""
        log_file = self.log_dir / "wan22_recovery.log"
        
        # Create custom rotating handler
        self.handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file),
            maxBytes=self.max_file_size,
            backupCount=self.max_files,
            encoding='utf-8'
        )
        
        # Custom rotation with compression
        if self.compress_old_logs:
            self.handler.rotator = self._compress_rotated_log
        
        # Setup formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.handler.setFormatter(formatter)
    
    def _compress_rotated_log(self, source: str, dest: str):
        """Compress rotated log file"""
        try:
            with open(source, 'rb') as f_in:
                with zipfile.ZipFile(f"{dest}.zip", 'w', zipfile.ZIP_DEFLATED) as f_out:
                    f_out.write(source, os.path.basename(source))
            
            # Remove uncompressed file
            os.remove(source)
            
        except Exception as e:
            # If compression fails, just rename the file
            shutil.move(source, dest)
    
    def cleanup_old_logs(self):
        """Clean up old log files based on age"""
        cutoff_time = datetime.now() - timedelta(days=self.cleanup_after_days)
        cleaned_count = 0
        
        for log_file in self.log_dir.glob("*.log*"):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_time:
                    log_file.unlink()
                    cleaned_count += 1
            except Exception:
                continue
        
        return cleaned_count
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get log file statistics"""
        log_files = list(self.log_dir.glob("*.log*"))
        total_size = sum(f.stat().st_size for f in log_files)
        
        return {
            "total_files": len(log_files),
            "total_size_mb": total_size / (1024 * 1024),
            "oldest_log": min((f.stat().st_mtime for f in log_files), default=0),
            "newest_log": max((f.stat().st_mtime for f in log_files), default=0),
            "compressed_files": len(list(self.log_dir.glob("*.zip")))
        }


class SystemStateCapture:
    """Comprehensive system state capture for logging"""
    
    @staticmethod
    def capture_full_system_state() -> Dict[str, Any]:
        """Capture comprehensive system state information"""
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "system_info": SystemStateCapture._get_system_info(),
                "process_info": SystemStateCapture._get_process_info(),
                "memory_info": SystemStateCapture._get_memory_info(),
                "gpu_info": SystemStateCapture._get_gpu_info(),
                "disk_info": SystemStateCapture._get_disk_info(),
                "network_info": SystemStateCapture._get_network_info(),
                "environment_vars": SystemStateCapture._get_relevant_env_vars(),
                "python_info": SystemStateCapture._get_python_info(),
                "thread_info": SystemStateCapture._get_thread_info()
            }
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "error": f"Failed to capture system state: {e}",
                "partial_info": SystemStateCapture._get_minimal_info()
            }
    
    @staticmethod
    def _get_system_info() -> Dict[str, Any]:
        """Get basic system information"""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "boot_time": psutil.boot_time()
        }
    
    @staticmethod
    def _get_process_info() -> Dict[str, Any]:
        """Get current process information"""
        try:
            process = psutil.Process()
            return {
                "pid": process.pid,
                "name": process.name(),
                "status": process.status(),
                "create_time": process.create_time(),
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "num_threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.connections())
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def _get_memory_info() -> Dict[str, Any]:
        """Get memory usage information"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            return {
                "virtual_memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free
                },
                "swap_memory": {
                    "total": swap.total,
                    "used": swap.used,
                    "free": swap.free,
                    "percent": swap.percent
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def _get_gpu_info() -> Dict[str, Any]:
        """Get GPU information if available"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = {}
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_info = torch.cuda.mem_get_info(i)
                    gpu_info[f"gpu_{i}"] = {
                        "name": props.name,
                        "total_memory": props.total_memory,
                        "major": props.major,
                        "minor": props.minor,
                        "multi_processor_count": props.multi_processor_count,
                        "memory_free": memory_info[0],
                        "memory_total": memory_info[1],
                        "memory_used": memory_info[1] - memory_info[0]
                    }
                return gpu_info
            else:
                return {"cuda_available": False}
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def _get_disk_info() -> Dict[str, Any]:
        """Get disk usage information"""
        try:
            disk_usage = psutil.disk_usage('/')
            return {
                "total": disk_usage.total,
                "used": disk_usage.used,
                "free": disk_usage.free,
                "percent": (disk_usage.used / disk_usage.total) * 100
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def _get_network_info() -> Dict[str, Any]:
        """Get network information"""
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "errin": net_io.errin,
                "errout": net_io.errout,
                "dropin": net_io.dropin,
                "dropout": net_io.dropout
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def _get_relevant_env_vars() -> Dict[str, str]:
        """Get relevant environment variables"""
        relevant_vars = [
            'CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF', 'CUDA_LAUNCH_BLOCKING',
            'PYTHONPATH', 'PATH', 'TORCH_HOME', 'HF_HOME', 'TRANSFORMERS_CACHE'
        ]
        return {var: os.environ.get(var, 'Not set') for var in relevant_vars}
    
    @staticmethod
    def _get_python_info() -> Dict[str, Any]:
        """Get Python environment information"""
        import sys
        return {
            "version": sys.version,
            "executable": sys.executable,
            "path": sys.path[:5],  # First 5 paths only
            "modules": list(sys.modules.keys())[:20]  # First 20 modules only
        }
    
    @staticmethod
    def _get_thread_info() -> Dict[str, Any]:
        """Get threading information"""
        return {
            "active_count": threading.active_count(),
            "current_thread": threading.current_thread().name,
            "main_thread": threading.main_thread().name
        }
    
    @staticmethod
    def _get_minimal_info() -> Dict[str, Any]:
        """Get minimal system info when full capture fails"""
        return {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "timestamp": datetime.now().isoformat()
        }


class RecoveryWorkflowManager:
    """Manager for user-guided recovery workflows"""
    
    def __init__(self, workflows_dir: str = "recovery_workflows"):
        """
        Initialize recovery workflow manager.
        
        Args:
            workflows_dir: Directory containing workflow definitions
        """
        self.workflows_dir = Path(workflows_dir)
        self.workflows_dir.mkdir(exist_ok=True)
        
        self.workflows: Dict[str, RecoveryWorkflow] = {}
        self.active_executions: Dict[str, WorkflowExecution] = {}
        
        # Load built-in workflows
        self._load_builtin_workflows()
    
    def _load_builtin_workflows(self):
        """Load built-in recovery workflows"""
        
        # Workflow for VRAM detection issues
        vram_workflow = RecoveryWorkflow(
            workflow_id="vram_detection_failure",
            title="VRAM Detection Failure Recovery",
            description="Guide user through resolving VRAM detection issues",
            error_patterns=["VRAM detection failed", "CUDA out of memory", "GPU not detected"],
            steps=[
                WorkflowStepDefinition(
                    step_id="diagnose_gpu",
                    title="Diagnose GPU Status",
                    description="Check if GPU is properly detected by the system",
                    step_type=WorkflowStep.DIAGNOSIS,
                    instructions=[
                        "Open Command Prompt as Administrator",
                        "Run 'nvidia-smi' command",
                        "Check if your RTX 4080 is listed",
                        "Note the VRAM amount shown"
                    ],
                    validation_checks=[
                        "GPU appears in nvidia-smi output",
                        "VRAM amount is approximately 16GB",
                        "No error messages in nvidia-smi"
                    ],
                    expected_outcomes=[
                        "GPU is detected and shows correct VRAM",
                        "Driver version is displayed",
                        "Temperature and utilization are shown"
                    ],
                    troubleshooting_tips=[
                        "If nvidia-smi fails, reinstall GPU drivers",
                        "If VRAM shows less than 16GB, check for hardware issues",
                        "Restart system if GPU is not detected"
                    ]
                ),
                WorkflowStepDefinition(
                    step_id="check_drivers",
                    title="Verify GPU Drivers",
                    description="Ensure GPU drivers are properly installed and up to date",
                    step_type=WorkflowStep.DIAGNOSIS,
                    instructions=[
                        "Open Device Manager",
                        "Expand 'Display adapters'",
                        "Right-click on RTX 4080",
                        "Select 'Properties' and check driver version"
                    ],
                    validation_checks=[
                        "Driver version is recent (within 6 months)",
                        "No warning icons on GPU device",
                        "Driver is digitally signed by NVIDIA"
                    ],
                    expected_outcomes=[
                        "Driver version 535.xx or newer",
                        "Device status shows 'working properly'",
                        "No driver conflicts reported"
                    ],
                    troubleshooting_tips=[
                        "Download latest drivers from NVIDIA website",
                        "Use DDU (Display Driver Uninstaller) if needed",
                        "Disable Windows automatic driver updates"
                    ]
                ),
                WorkflowStepDefinition(
                    step_id="configure_fallback",
                    title="Configure Manual VRAM Settings",
                    description="Set up manual VRAM configuration as fallback",
                    step_type=WorkflowStep.EXECUTION,
                    instructions=[
                        "Open WAN22 configuration file",
                        "Add manual VRAM setting: 'vram_gb': 16",
                        "Set detection method to 'manual'",
                        "Save configuration file"
                    ],
                    validation_checks=[
                        "Configuration file is valid JSON",
                        "VRAM setting is correctly specified",
                        "Backup of original config exists"
                    ],
                    expected_outcomes=[
                        "Manual VRAM setting is applied",
                        "System uses 16GB VRAM limit",
                        "No more VRAM detection errors"
                    ],
                    troubleshooting_tips=[
                        "Use JSON validator to check config syntax",
                        "Restart application after config changes",
                        "Monitor VRAM usage in task manager"
                    ],
                    user_input_required=True,
                    critical_step=True
                )
            ],
            estimated_time=15,
            difficulty_level="intermediate",
            prerequisites=["Administrator access", "Basic command line knowledge"],
            success_criteria=[
                "VRAM is properly detected or manually configured",
                "No VRAM-related errors during model loading",
                "System can load TI2V-5B model successfully"
            ]
        )
        
        self.workflows["vram_detection_failure"] = vram_workflow
        
        # Workflow for quantization timeout issues
        quantization_workflow = RecoveryWorkflow(
            workflow_id="quantization_timeout",
            title="Quantization Timeout Recovery",
            description="Resolve quantization timeout and hanging issues",
            error_patterns=["quantization timeout", "quantization hanging", "bf16 timeout"],
            steps=[
                WorkflowStepDefinition(
                    step_id="disable_quantization",
                    title="Disable Quantization Temporarily",
                    description="Turn off quantization to test if it resolves the issue",
                    step_type=WorkflowStep.EXECUTION,
                    instructions=[
                        "Open WAN22 settings",
                        "Navigate to quantization settings",
                        "Set quantization to 'none'",
                        "Save settings and restart application"
                    ],
                    validation_checks=[
                        "Quantization is set to 'none'",
                        "Settings are saved successfully",
                        "Application restarts without errors"
                    ],
                    expected_outcomes=[
                        "Model loads without quantization",
                        "No timeout errors occur",
                        "Generation works but may use more VRAM"
                    ],
                    troubleshooting_tips=[
                        "Monitor VRAM usage without quantization",
                        "If VRAM is insufficient, reduce batch size",
                        "Consider using CPU offloading"
                    ],
                    critical_step=True
                ),
                WorkflowStepDefinition(
                    step_id="test_alternative_quantization",
                    title="Test Alternative Quantization Methods",
                    description="Try different quantization methods to find working option",
                    step_type=WorkflowStep.EXECUTION,
                    instructions=[
                        "Try int8 quantization instead of bf16",
                        "Test with increased timeout values",
                        "Monitor system resources during quantization",
                        "Document which methods work"
                    ],
                    validation_checks=[
                        "Alternative quantization completes successfully",
                        "Output quality is acceptable",
                        "No system instability occurs"
                    ],
                    expected_outcomes=[
                        "Working quantization method identified",
                        "Reasonable generation speed achieved",
                        "System remains stable"
                    ],
                    troubleshooting_tips=[
                        "int8 is more stable than bf16 on some systems",
                        "Increase timeout to 600 seconds for large models",
                        "Close other applications to free resources"
                    ]
                )
            ],
            estimated_time=20,
            difficulty_level="beginner",
            prerequisites=["Access to WAN22 settings"],
            success_criteria=[
                "Quantization completes without timeout",
                "Model generates output successfully",
                "System performance is acceptable"
            ]
        )
        
        self.workflows["quantization_timeout"] = quantization_workflow
    
    def start_workflow(self, workflow_id: str, error_context: Dict[str, Any]) -> str:
        """
        Start a recovery workflow execution.
        
        Args:
            workflow_id: ID of the workflow to start
            error_context: Context information about the error
            
        Returns:
            Execution ID for tracking progress
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        execution_id = f"{workflow_id}_{int(time.time())}"
        
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            execution_id=execution_id,
            start_time=datetime.now(),
            current_step=0,
            completed_steps=[],
            failed_steps=[],
            user_responses={},
            system_changes=[],
            status="running"
        )
        
        self.active_executions[execution_id] = execution
        return execution_id
    
    def get_current_step(self, execution_id: str) -> Optional[WorkflowStepDefinition]:
        """Get the current step for a workflow execution"""
        if execution_id not in self.active_executions:
            return None
        
        execution = self.active_executions[execution_id]
        workflow = self.workflows[execution.workflow_id]
        
        if execution.current_step >= len(workflow.steps):
            return None
        
        return workflow.steps[execution.current_step]
    
    def complete_step(self, 
                     execution_id: str, 
                     success: bool, 
                     user_response: Optional[Dict[str, Any]] = None,
                     notes: Optional[str] = None) -> bool:
        """
        Mark a workflow step as completed.
        
        Args:
            execution_id: Execution ID
            success: Whether the step completed successfully
            user_response: User's response/input for the step
            notes: Additional notes about the step completion
            
        Returns:
            True if workflow continues, False if completed or failed
        """
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        workflow = self.workflows[execution.workflow_id]
        
        current_step = workflow.steps[execution.current_step]
        
        if success:
            execution.completed_steps.append(current_step.step_id)
            if user_response:
                execution.user_responses[current_step.step_id] = user_response
            if notes:
                execution.system_changes.append(f"{current_step.step_id}: {notes}")
        else:
            execution.failed_steps.append(current_step.step_id)
            if current_step.critical_step:
                execution.status = "failed"
                execution.end_time = datetime.now()
                return False
        
        # Move to next step
        execution.current_step += 1
        
        # Check if workflow is complete
        if execution.current_step >= len(workflow.steps):
            execution.status = "completed"
            execution.end_time = datetime.now()
            return False
        
        return True
    
    def get_workflow_progress(self, execution_id: str) -> Dict[str, Any]:
        """Get progress information for a workflow execution"""
        if execution_id not in self.active_executions:
            return {"error": "Execution not found"}
        
        execution = self.active_executions[execution_id]
        workflow = self.workflows[execution.workflow_id]
        
        return {
            "workflow_title": workflow.title,
            "execution_id": execution_id,
            "status": execution.status,
            "current_step": execution.current_step,
            "total_steps": len(workflow.steps),
            "completed_steps": len(execution.completed_steps),
            "failed_steps": len(execution.failed_steps),
            "progress_percent": (len(execution.completed_steps) / len(workflow.steps)) * 100,
            "estimated_time_remaining": max(0, workflow.estimated_time - 
                                          (datetime.now() - execution.start_time).seconds // 60),
            "current_step_info": self.get_current_step(execution_id).title if self.get_current_step(execution_id) else None
        }
    
    def find_applicable_workflows(self, error_message: str) -> List[str]:
        """Find workflows applicable to a given error message"""
        applicable = []
        
        for workflow_id, workflow in self.workflows.items():
            for pattern in workflow.error_patterns:
                if pattern.lower() in error_message.lower():
                    applicable.append(workflow_id)
                    break
        
        return applicable


class AdvancedLogger:
    """Advanced logging system with comprehensive context capture"""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 session_id: Optional[str] = None,
                 enable_system_state_capture: bool = True):
        """
        Initialize advanced logger.
        
        Args:
            log_dir: Directory for log files
            session_id: Unique session identifier
            enable_system_state_capture: Whether to capture system state with logs
        """
        self.log_dir = Path(log_dir)
        self.session_id = session_id or f"session_{int(time.time())}"
        self.enable_system_state_capture = enable_system_state_capture
        
        # Setup log rotation
        self.rotation_manager = LogRotationManager(self.log_dir)
        
        # Setup logger
        self.logger = logging.getLogger('WAN22_Advanced')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.rotation_manager.handler)
        
        # Thread-local storage for context
        self._local = threading.local()
    
    def log_with_context(self, 
                        level: LogLevel,
                        component: str,
                        message: str,
                        error: Optional[Exception] = None,
                        user_context: Optional[Dict[str, Any]] = None,
                        recovery_context: Optional[Dict[str, Any]] = None):
        """
        Log message with comprehensive context information.
        
        Args:
            level: Log level
            component: Component generating the log
            message: Log message
            error: Exception if applicable
            user_context: User-specific context
            recovery_context: Recovery-specific context
        """
        # Capture stack trace if error provided
        stack_trace = None
        if error:
            stack_trace = traceback.format_exception(type(error), error, error.__traceback__)
            stack_trace = ''.join(stack_trace)
        
        # Capture system state if enabled
        system_state = {}
        if self.enable_system_state_capture and level in [LogLevel.ERROR, LogLevel.CRITICAL, LogLevel.RECOVERY]:
            system_state = SystemStateCapture.capture_full_system_state()
        
        # Create log entry
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            component=component,
            message=message,
            stack_trace=stack_trace,
            system_state=system_state,
            user_context=user_context or {},
            recovery_context=recovery_context or {},
            session_id=self.session_id,
            thread_id=threading.current_thread().name
        )
        
        # Log as structured JSON
        json_log = json.dumps(log_entry.to_dict(), indent=2, default=str)
        
        # Map to standard logging levels
        if level == LogLevel.CRITICAL:
            self.logger.critical(json_log)
        elif level == LogLevel.ERROR:
            self.logger.error(json_log)
        elif level == LogLevel.WARNING:
            self.logger.warning(json_log)
        elif level == LogLevel.INFO:
            self.logger.info(json_log)
        else:
            self.logger.debug(json_log)
    
    def log_recovery_attempt(self, 
                           component: str,
                           error: Exception,
                           recovery_actions: List[str],
                           success: bool):
        """Log a recovery attempt with full context"""
        recovery_context = {
            "recovery_actions": recovery_actions,
            "recovery_success": success,
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        
        message = f"Recovery attempt for {type(error).__name__}: {'SUCCESS' if success else 'FAILED'}"
        
        self.log_with_context(
            level=LogLevel.RECOVERY,
            component=component,
            message=message,
            error=error,
            recovery_context=recovery_context
        )
    
    def log_user_action(self, 
                       component: str,
                       action: str,
                       context: Dict[str, Any]):
        """Log user actions for audit trail"""
        self.log_with_context(
            level=LogLevel.USER_ACTION,
            component=component,
            message=f"User action: {action}",
            user_context=context
        )
    
    def cleanup_logs(self) -> int:
        """Clean up old log files"""
        return self.rotation_manager.cleanup_old_logs()
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return self.rotation_manager.get_log_statistics()


# Global instances for easy access
_advanced_logger = None
_workflow_manager = None


def get_advanced_logger() -> AdvancedLogger:
    """Get global advanced logger instance"""
    global _advanced_logger
    if _advanced_logger is None:
        _advanced_logger = AdvancedLogger()
    return _advanced_logger


def get_workflow_manager() -> RecoveryWorkflowManager:
    """Get global workflow manager instance"""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = RecoveryWorkflowManager()
    return _workflow_manager