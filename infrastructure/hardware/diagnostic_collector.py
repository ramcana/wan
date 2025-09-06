"""
Diagnostic information collector for video generation troubleshooting.

This module provides comprehensive diagnostic data collection capabilities
to help troubleshoot generation failures and performance issues.
"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import psutil
import torch
from dataclasses import dataclass, asdict

from generation_logger import get_logger, SystemDiagnostics


@dataclass
class ModelDiagnostics:
    """Model-specific diagnostic information."""
    model_type: str
    model_path: str
    model_exists: bool
    model_size_gb: Optional[float]
    model_accessible: bool
    model_format: Optional[str]
    last_modified: Optional[str]
    
    @classmethod
    def collect(cls, model_type: str, model_path: str) -> 'ModelDiagnostics':
        """Collect model diagnostic information."""
        path_obj = Path(model_path)
        
        model_exists = path_obj.exists()
        model_size_gb = None
        model_accessible = False
        model_format = None
        last_modified = None
        
        if model_exists:
            try:
                # Get model size
                if path_obj.is_file():
                    model_size_gb = path_obj.stat().st_size / (1024**3)
                    last_modified = datetime.fromtimestamp(path_obj.stat().st_mtime).isoformat()
                elif path_obj.is_dir():
                    # Calculate directory size
                    total_size = sum(f.stat().st_size for f in path_obj.rglob('*') if f.is_file())
                    model_size_gb = total_size / (1024**3)
                    last_modified = datetime.fromtimestamp(path_obj.stat().st_mtime).isoformat()
                
                # Check accessibility
                model_accessible = os.access(model_path, os.R_OK)
                
                # Determine format
                if path_obj.suffix in ['.pt', '.pth']:
                    model_format = 'pytorch'
                elif path_obj.suffix in ['.safetensors']:
                    model_format = 'safetensors'
                elif path_obj.is_dir():
                    if (path_obj / 'config.json').exists():
                        model_format = 'diffusers'
                    else:
                        model_format = 'directory'
                else:
                    model_format = 'unknown'
                    
            except Exception:
                pass
        
        return cls(
            model_type=model_type,
            model_path=model_path,
            model_exists=model_exists,
            model_size_gb=model_size_gb,
            model_accessible=model_accessible,
            model_format=model_format,
            last_modified=last_modified
        )


@dataclass
class EnvironmentDiagnostics:
    """Environment and dependency diagnostic information."""
    python_version: str
    python_executable: str
    virtual_env: Optional[str]
    cuda_available: bool
    cuda_version: Optional[str]
    torch_version: str
    gpu_count: int
    gpu_names: List[str]
    installed_packages: Dict[str, str]
    environment_variables: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def collect(cls) -> 'EnvironmentDiagnostics':
        """Collect environment diagnostic information."""
        # Python info
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        python_executable = sys.executable
        virtual_env = os.environ.get('VIRTUAL_ENV')
        
        # CUDA and GPU info
        cuda_available = False
        cuda_version = None
        gpu_count = 0
        gpu_names = []
        
        try:
            if hasattr(torch, 'cuda'):
                cuda_available = torch.cuda.is_available()
                cuda_version = torch.version.cuda if cuda_available else None
                gpu_count = torch.cuda.device_count() if cuda_available else 0
                
                if cuda_available:
                    for i in range(gpu_count):
                        try:
                            gpu_names.append(torch.cuda.get_device_name(i))
                        except Exception:
                            gpu_names.append(f"GPU {i} (name unavailable)")
        except Exception:
            # Handle cases where torch.cuda is not available
            pass
        
        # Package versions
        installed_packages = cls._get_installed_packages()
        
        # Environment variables (filtered for security)
        env_vars = cls._get_relevant_env_vars()
        
        return cls(
            python_version=python_version,
            python_executable=python_executable,
            virtual_env=virtual_env,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            torch_version=getattr(torch, '__version__', 'unknown'),
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            installed_packages=installed_packages,
            environment_variables=env_vars
        )
    
    @staticmethod
    def _get_installed_packages() -> Dict[str, str]:
        """Get versions of key installed packages."""
        key_packages = [
            'torch', 'torchvision', 'transformers', 'diffusers',
            'accelerate', 'xformers', 'gradio', 'pillow', 'numpy'
        ]
        
        packages = {}
        for package in key_packages:
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'show', package],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if line.startswith('Version:'):
                            packages[package] = line.split(':', 1)[1].strip()
                            break
                else:
                    packages[package] = 'not installed'
            except Exception:
                packages[package] = 'unknown'
        
        return packages
    
    @staticmethod
    def _get_relevant_env_vars() -> Dict[str, str]:
        """Get relevant environment variables (filtered for security)."""
        relevant_vars = [
            'CUDA_VISIBLE_DEVICES',
            'PYTORCH_CUDA_ALLOC_CONF',
            'TORCH_HOME',
            'HF_HOME',
            'TRANSFORMERS_CACHE',
            'PATH'
        ]
        
        env_vars = {}
        for var in relevant_vars:
            value = os.environ.get(var)
            if value:
                # Truncate PATH for readability
                if var == 'PATH':
                    paths = value.split(os.pathsep)
                    env_vars[var] = f"{len(paths)} paths (truncated)"
                else:
                    env_vars[var] = value
        
        return env_vars


@dataclass
class GenerationDiagnostics:
    """Generation-specific diagnostic information."""
    session_id: str
    model_type: str
    generation_mode: str
    input_parameters: Dict[str, Any]
    validation_results: Dict[str, Any]
    resource_usage: Dict[str, Any]
    error_history: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class DiagnosticCollector:
    """Comprehensive diagnostic information collector."""
    
    def __init__(self):
        """Initialize the diagnostic collector."""
        self.logger = get_logger()
    
    def collect_full_diagnostics(self, 
                                session_id: Optional[str] = None,
                                include_logs: bool = True,
                                include_models: bool = True) -> Dict[str, Any]:
        """
        Collect comprehensive diagnostic information.
        
        Args:
            session_id: Specific session to collect diagnostics for
            include_logs: Whether to include log analysis
            include_models: Whether to include model diagnostics
            
        Returns:
            Dictionary containing all diagnostic information
        """
        diagnostics = {
            'collection_timestamp': datetime.now().isoformat(),
            'system': SystemDiagnostics.collect().to_dict(),
            'environment': EnvironmentDiagnostics.collect().to_dict(),
            'platform': self._collect_platform_info()
        }
        
        if include_models:
            diagnostics['models'] = self._collect_model_diagnostics()
        
        if include_logs:
            diagnostics['logs'] = self._collect_log_analysis(session_id)
        
        if session_id:
            diagnostics['session'] = self._collect_session_diagnostics(session_id)
        
        return diagnostics
    
    def _collect_platform_info(self) -> Dict[str, Any]:
        """Collect platform-specific information."""
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'hostname': platform.node()
        }
    
    def _collect_model_diagnostics(self) -> Dict[str, Any]:
        """Collect diagnostics for all configured models."""
        model_diagnostics = {}
        
        # Check common model locations
        model_paths = {
            'wan22_t2v': 'models/Wan-AI_Wan2.2-T2V-A14B-Diffusers',
            'wan22_huggingface': 'models/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers'
        }
        
        for model_name, model_path in model_paths.items():
            try:
                model_diag = ModelDiagnostics.collect(model_name, model_path)
                model_diagnostics[model_name] = asdict(model_diag)
            except Exception as e:
                model_diagnostics[model_name] = {
                    'error': str(e),
                    'model_type': model_name,
                    'model_path': model_path
                }
        
        return model_diagnostics
    
    def _collect_log_analysis(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Collect and analyze log information."""
        log_analysis = {
            'summary': self.logger._get_log_summary(),
            'recent_errors': self._get_recent_errors(),
            'performance_trends': self._analyze_performance_trends()
        }
        
        if session_id:
            log_analysis['session_logs'] = self.logger.get_session_logs(session_id)
        
        return log_analysis
    
    def _collect_session_diagnostics(self, session_id: str) -> Dict[str, Any]:
        """Collect diagnostics for a specific session."""
        session_logs = self.logger.get_session_logs(session_id)
        
        # Extract session information from logs
        session_info = {
            'session_id': session_id,
            'log_entries': len(sum(session_logs.values(), [])),
            'has_errors': len(session_logs.get('errors', [])) > 0,
            'has_performance_data': len(session_logs.get('performance', [])) > 0
        }
        
        # Try to extract generation parameters from logs
        for log_entry in session_logs.get('generation', []):
            if 'parameters:' in log_entry:
                try:
                    # Extract JSON from log entry
                    json_start = log_entry.find('{')
                    if json_start != -1:
                        params = json.loads(log_entry[json_start:])
                        session_info['parameters'] = params
                        break
                except json.JSONDecodeError:
                    continue
        
        return session_info
    
    def _get_recent_errors(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent errors from error logs."""
        recent_errors = []
        error_log = Path(self.logger.log_dir) / 'errors.log'
        
        if not error_log.exists():
            return recent_errors
        
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        try:
            with open(error_log, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # Parse timestamp from log line
                        timestamp_str = line.split(' - ')[0]
                        timestamp = datetime.fromisoformat(timestamp_str.replace(',', '.')).timestamp()
                        
                        if timestamp > cutoff_time:
                            recent_errors.append({
                                'timestamp': timestamp_str,
                                'message': line.strip()
                            })
                    except Exception:
                        continue
        except Exception as e:
            self.logger.error_logger.error(f"Error reading recent errors: {e}")
        
        return recent_errors[-50:]  # Return last 50 errors
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends from logs."""
        trends = {
            'average_duration_by_model': {},
            'success_rate_by_model': {},
            'common_failure_points': {}
        }
        
        performance_log = Path(self.logger.log_dir) / 'performance.log'
        
        if not performance_log.exists():
            return trends
        
        model_data = {}
        
        try:
            with open(performance_log, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # Extract JSON from log line
                        json_start = line.find('{')
                        if json_start != -1:
                            data = json.loads(line[json_start:])
                            
                            model_type = data.get('model_type', 'unknown')
                            if model_type not in model_data:
                                model_data[model_type] = {
                                    'durations': [],
                                    'successes': 0,
                                    'failures': 0,
                                    'error_types': {}
                                }
                            
                            if 'duration' in data:
                                model_data[model_type]['durations'].append(data['duration'])
                            
                            if data.get('status') == 'success':
                                model_data[model_type]['successes'] += 1
                            elif data.get('status') == 'error':
                                model_data[model_type]['failures'] += 1
                                error_type = data.get('error_type', 'unknown')
                                model_data[model_type]['error_types'][error_type] = \
                                    model_data[model_type]['error_types'].get(error_type, 0) + 1
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            self.logger.error_logger.error(f"Error analyzing performance trends: {e}")
        
        # Calculate trends
        for model_type, data in model_data.items():
            if data['durations']:
                trends['average_duration_by_model'][model_type] = \
                    sum(data['durations']) / len(data['durations'])
            
            total_attempts = data['successes'] + data['failures']
            if total_attempts > 0:
                trends['success_rate_by_model'][model_type] = \
                    data['successes'] / total_attempts
            
            if data['error_types']:
                trends['common_failure_points'][model_type] = data['error_types']
        
        return trends
    
    def export_diagnostics(self, 
                          output_path: str,
                          session_id: Optional[str] = None,
                          format: str = 'json') -> str:
        """
        Export diagnostic information to file.
        
        Args:
            output_path: Path to save the diagnostic report
            session_id: Specific session to include
            format: Output format ('json' or 'txt')
            
        Returns:
            Path to the exported file
        """
        diagnostics = self.collect_full_diagnostics(session_id=session_id)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(diagnostics, f, indent=2, default=str)
        else:
            # Text format
            with open(output_file, 'w', encoding='utf-8') as f:
                self._write_text_report(f, diagnostics)
        
        return str(output_file)
    
    def _write_text_report(self, file, diagnostics: Dict[str, Any]):
        """Write diagnostics in human-readable text format."""
        file.write("WAN2.2 Video Generation Diagnostic Report\n")
        file.write("=" * 50 + "\n\n")
        
        file.write(f"Generated: {diagnostics['collection_timestamp']}\n\n")
        
        # System information
        file.write("SYSTEM INFORMATION\n")
        file.write("-" * 20 + "\n")
        system = diagnostics['system']
        file.write(f"CPU Usage: {system['cpu_usage']:.1f}%\n")
        file.write(f"Memory Usage: {system['memory_usage']:.1f}%\n")
        file.write(f"CUDA Available: {system['cuda_available']}\n")
        if system['gpu_memory_used']:
            file.write(f"GPU Memory: {system['gpu_memory_used']:.1f}GB / {system['gpu_memory_total']:.1f}GB\n")
        file.write("\n")
        
        # Environment information
        file.write("ENVIRONMENT INFORMATION\n")
        file.write("-" * 25 + "\n")
        env = diagnostics['environment']
        file.write(f"Python Version: {env['python_version']}\n")
        file.write(f"PyTorch Version: {env['torch_version']}\n")
        file.write(f"GPU Count: {env['gpu_count']}\n")
        for i, gpu_name in enumerate(env['gpu_names']):
            file.write(f"GPU {i}: {gpu_name}\n")
        file.write("\n")
        
        # Model information
        if 'models' in diagnostics:
            file.write("MODEL INFORMATION\n")
            file.write("-" * 18 + "\n")
            for model_name, model_info in diagnostics['models'].items():
                file.write(f"{model_name}:\n")
                file.write(f"  Path: {model_info.get('model_path', 'N/A')}\n")
                file.write(f"  Exists: {model_info.get('model_exists', False)}\n")
                file.write(f"  Size: {model_info.get('model_size_gb', 'N/A')} GB\n")
                file.write(f"  Format: {model_info.get('model_format', 'N/A')}\n")
                file.write("\n")
        
        # Log summary
        if 'logs' in diagnostics:
            file.write("LOG SUMMARY\n")
            file.write("-" * 12 + "\n")
            summary = diagnostics['logs']['summary']
            file.write(f"Total Sessions: {summary['total_sessions']}\n")
            file.write(f"Successful: {summary['successful_sessions']}\n")
            file.write(f"Failed: {summary['failed_sessions']}\n")
            file.write(f"Average Duration: {summary['average_duration']:.2f}s\n")
            
            if summary['error_types']:
                file.write("\nCommon Errors:\n")
                for error_type, count in summary['error_types'].items():
                    file.write(f"  {error_type}: {count}\n")
            file.write("\n")


# Global diagnostic collector instance
_collector_instance = None


def get_diagnostic_collector() -> DiagnosticCollector:
    """Get the global diagnostic collector instance."""
    global _collector_instance
    
    if _collector_instance is None:
        _collector_instance = DiagnosticCollector()
    
    return _collector_instance