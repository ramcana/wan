"""
Diagnostic and reporting system for Wan model compatibility.

This module provides comprehensive diagnostic collection and reporting capabilities
specifically for Wan model compatibility issues, including model analysis,
pipeline compatibility, and system requirements validation.
"""

import json
import logging
import platform
import sys
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import torch
import psutil

from infrastructure.hardware.architecture_detector import ArchitectureDetector, ArchitectureType, ComponentInfo

logger = logging.getLogger(__name__)


@dataclass
class ModelDiagnostics:
    """Comprehensive model diagnostic information for compatibility analysis."""
    model_path: str
    model_name: str
    timestamp: str
    system_info: Dict[str, Any]
    model_analysis: Dict[str, Any]
    pipeline_attempt: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]

    def to_json(self) -> str:
        """Serialize diagnostics to JSON format."""
        return json.dumps(asdict(self), indent=2, default=str)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class SystemInfo:
    """System information relevant to model compatibility."""
    gpu_name: Optional[str]
    vram_total: Optional[int]  # MB
    vram_available: Optional[int]  # MB
    python_version: str
    torch_version: str
    diffusers_version: Optional[str]
    cuda_available: bool
    cuda_version: Optional[str]
    platform_info: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def collect(cls) -> 'SystemInfo':
        """Collect current system information."""
        # GPU information
        gpu_name = None
        vram_total = None
        vram_available = None
        cuda_available = False
        cuda_version = None

        try:
            if torch.cuda.is_available():
                cuda_available = True
                cuda_version = torch.version.cuda
                if torch.cuda.device_count() > 0:
                    gpu_name = torch.cuda.get_device_name(0)
                    # Get VRAM info
                    vram_total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                    vram_available = (torch.cuda.get_device_properties(0).total_memory - 
                                    torch.cuda.memory_allocated(0)) // (1024 * 1024)
        except Exception as e:
            logger.warning(f"Failed to collect GPU information: {e}")

        # Package versions
        diffusers_version = None
        try:
            import diffusers
            diffusers_version = diffusers.__version__
        except ImportError:
            pass

        # Platform information
        platform_info = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }

        return cls(
            gpu_name=gpu_name,
            vram_total=vram_total,
            vram_available=vram_available,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            torch_version=torch.__version__,
            diffusers_version=diffusers_version,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            platform_info=platform_info
        )


@dataclass
class ModelAnalysis:
    """Analysis results for a specific model."""
    architecture_detected: str
    has_model_index: bool
    model_index_valid: bool
    components_found: List[str]
    vae_dimensions: Optional[int]
    custom_attributes: List[str]
    pipeline_class_detected: Optional[str]
    is_wan_architecture: bool
    component_details: Dict[str, Dict[str, Any]]

    @classmethod
    def analyze_model(cls, model_path: str, detector: ArchitectureDetector) -> 'ModelAnalysis':
        """Analyze a model using the architecture detector."""
        try:
            # Detect architecture
            architecture = detector.detect_model_architecture(model_path)
            
            # Check model_index.json
            model_index_path = Path(model_path) / "model_index.json"
            has_model_index = model_index_path.exists()
            model_index_valid = False
            custom_attributes = []
            pipeline_class_detected = None
            
            if has_model_index:
                try:
                    with open(model_index_path, 'r') as f:
                        model_index = json.load(f)
                    model_index_valid = True
                    
                    # Extract custom attributes
                    standard_keys = {'_class_name', '_diffusers_version', 'scheduler', 'text_encoder', 
                                   'tokenizer', 'vae', 'unet', 'transformer'}
                    custom_attributes = [k for k in model_index.keys() if k not in standard_keys]
                    
                    # Get pipeline class
                    pipeline_class_detected = model_index.get('_class_name')
                    
                except Exception as e:
                    logger.warning(f"Failed to parse model_index.json: {e}")
            
            # Get component details
            component_details = {}
            for comp_name, comp_info in architecture.components.items():
                component_details[comp_name] = {
                    'class_name': comp_info.class_name,
                    'is_custom': comp_info.is_custom,
                    'dependencies': comp_info.dependencies,
                    'config_exists': Path(model_path, comp_info.config_path).exists() if comp_info.config_path else False,
                    'weights_exist': Path(model_path, comp_info.weight_path).exists() if comp_info.weight_path else False
                }
            
            # Determine VAE dimensions
            vae_dimensions = None
            if 'vae' in architecture.components:
                vae_config_path = Path(model_path) / architecture.components['vae'].config_path
                if vae_config_path.exists():
                    try:
                        with open(vae_config_path, 'r') as f:
                            vae_config = json.load(f)
                        # Check for 3D VAE indicators
                        if any(key in vae_config for key in ['in_channels', 'out_channels']):
                            # This is a simplified check - in reality, we'd need more sophisticated detection
                            vae_dimensions = 3 if architecture.architecture_type == ArchitectureType.WAN_T2V else 2
                    except Exception:
                        pass
            
            return cls(
                architecture_detected=architecture.architecture_type.value,
                has_model_index=has_model_index,
                model_index_valid=model_index_valid,
                components_found=list(architecture.components.keys()),
                vae_dimensions=vae_dimensions,
                custom_attributes=custom_attributes,
                pipeline_class_detected=pipeline_class_detected,
                is_wan_architecture=architecture.architecture_type in [ArchitectureType.WAN_T2V, ArchitectureType.WAN_T2I, ArchitectureType.WAN_I2V],
                component_details=component_details
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze model {model_path}: {e}")
            return cls(
                architecture_detected="error",
                has_model_index=False,
                model_index_valid=False,
                components_found=[],
                vae_dimensions=None,
                custom_attributes=[],
                pipeline_class_detected=None,
                is_wan_architecture=False,
                component_details={}
            )


@dataclass
class PipelineAttempt:
    """Information about a pipeline loading attempt."""
    attempted_pipeline: str
    pipeline_available: bool
    trust_remote_code: bool
    remote_code_fetched: bool
    load_success: bool
    error_type: Optional[str]
    error_message: Optional[str]
    load_time_seconds: Optional[float]

    @classmethod
    def create_failed_attempt(cls, pipeline_name: str, error: Exception, 
                            trust_remote_code: bool = False) -> 'PipelineAttempt':
        """Create a failed pipeline attempt record."""
        error_type = type(error).__name__
        return cls(
            attempted_pipeline=pipeline_name,
            pipeline_available=False,
            trust_remote_code=trust_remote_code,
            remote_code_fetched=False,
            load_success=False,
            error_type=error_type,
            error_message=str(error),
            load_time_seconds=None
        )


class DiagnosticCollector:
    """Collect and write comprehensive diagnostic information for Wan model compatibility."""

    def __init__(self, diagnostics_dir: str = "diagnostics"):
        """Initialize the diagnostic collector.
        
        Args:
            diagnostics_dir: Directory to store diagnostic reports
        """
        self.diagnostics_dir = Path(diagnostics_dir)
        self.diagnostics_dir.mkdir(exist_ok=True)
        self.detector = ArchitectureDetector()

    def collect_model_diagnostics(self, model_path: str, 
                                load_attempt_result: Optional[Any] = None) -> ModelDiagnostics:
        """Collect comprehensive diagnostic information for model load attempt.
        
        Args:
            model_path: Path to the model directory
            load_attempt_result: Result of pipeline loading attempt (if any)
            
        Returns:
            ModelDiagnostics object containing all collected information
        """
        timestamp = datetime.now().isoformat()
        model_name = Path(model_path).name
        
        # Collect system information
        system_info = SystemInfo.collect().to_dict()
        
        # Analyze the model
        model_analysis = ModelAnalysis.analyze_model(model_path, self.detector)
        
        # Create pipeline attempt information
        pipeline_attempt = self._analyze_pipeline_attempt(load_attempt_result, model_analysis)
        
        # Generate errors, warnings, and recommendations
        errors, warnings, recommendations = self._generate_diagnostics(
            model_analysis, pipeline_attempt, system_info
        )
        
        return ModelDiagnostics(
            model_path=model_path,
            model_name=model_name,
            timestamp=timestamp,
            system_info=system_info,
            model_analysis=asdict(model_analysis),
            pipeline_attempt=asdict(pipeline_attempt),
            errors=errors,
            warnings=warnings,
            recommendations=recommendations
        )

    def write_compatibility_report(self, model_name: str, 
                                 diagnostics: ModelDiagnostics) -> str:
        """Write diagnostics to <model_name>_compat.json file.
        
        Args:
            model_name: Name of the model for the report filename
            diagnostics: ModelDiagnostics object to write
            
        Returns:
            Path to the written report file
        """
        # Sanitize model name for filename
        safe_name = "".join(c for c in model_name if c.isalnum() or c in ('-', '_', '.'))
        report_path = self.diagnostics_dir / f"{safe_name}_compat.json"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(diagnostics.to_json())
            
            logger.info(f"Compatibility report written to {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Failed to write compatibility report: {e}")
            raise

    def generate_diagnostic_summary(self, diagnostics: ModelDiagnostics) -> str:
        """Generate human-readable diagnostic summary.
        
        Args:
            diagnostics: ModelDiagnostics object to summarize
            
        Returns:
            Human-readable diagnostic summary string
        """
        summary_lines = []
        
        # Header
        summary_lines.append("=" * 60)
        summary_lines.append(f"WAN MODEL COMPATIBILITY DIAGNOSTIC REPORT")
        summary_lines.append("=" * 60)
        summary_lines.append(f"Model: {diagnostics.model_name}")
        summary_lines.append(f"Path: {diagnostics.model_path}")
        summary_lines.append(f"Generated: {diagnostics.timestamp}")
        summary_lines.append("")
        
        # System Information
        summary_lines.append("SYSTEM INFORMATION")
        summary_lines.append("-" * 20)
        sys_info = diagnostics.system_info
        summary_lines.append(f"GPU: {sys_info.get('gpu_name', 'N/A')}")
        summary_lines.append(f"VRAM: {sys_info.get('vram_available', 'N/A')}MB / {sys_info.get('vram_total', 'N/A')}MB")
        summary_lines.append(f"CUDA: {sys_info.get('cuda_available', False)} (v{sys_info.get('cuda_version', 'N/A')})")
        summary_lines.append(f"Python: {sys_info.get('python_version', 'N/A')}")
        summary_lines.append(f"PyTorch: {sys_info.get('torch_version', 'N/A')}")
        summary_lines.append(f"Diffusers: {sys_info.get('diffusers_version', 'N/A')}")
        summary_lines.append("")
        
        # Model Analysis
        summary_lines.append("MODEL ANALYSIS")
        summary_lines.append("-" * 15)
        model_analysis = diagnostics.model_analysis
        summary_lines.append(f"Architecture: {model_analysis.get('architecture_detected', 'unknown')}")
        summary_lines.append(f"Is Wan Model: {model_analysis.get('is_wan_architecture', False)}")
        summary_lines.append(f"Pipeline Class: {model_analysis.get('pipeline_class_detected', 'N/A')}")
        summary_lines.append(f"Model Index Valid: {model_analysis.get('model_index_valid', False)}")
        summary_lines.append(f"Components: {', '.join(model_analysis.get('components_found', []))}")
        summary_lines.append(f"VAE Dimensions: {model_analysis.get('vae_dimensions', 'N/A')}")
        if model_analysis.get('custom_attributes'):
            summary_lines.append(f"Custom Attributes: {', '.join(model_analysis.get('custom_attributes', []))}")
        summary_lines.append("")
        
        # Pipeline Attempt
        summary_lines.append("PIPELINE LOADING")
        summary_lines.append("-" * 16)
        pipeline_attempt = diagnostics.pipeline_attempt
        summary_lines.append(f"Attempted Pipeline: {pipeline_attempt.get('attempted_pipeline', 'N/A')}")
        summary_lines.append(f"Load Success: {pipeline_attempt.get('load_success', False)}")
        summary_lines.append(f"Pipeline Available: {pipeline_attempt.get('pipeline_available', False)}")
        summary_lines.append(f"Trust Remote Code: {pipeline_attempt.get('trust_remote_code', False)}")
        if pipeline_attempt.get('error_type'):
            summary_lines.append(f"Error Type: {pipeline_attempt.get('error_type')}")
        summary_lines.append("")
        
        # Errors
        if diagnostics.errors:
            summary_lines.append("ERRORS")
            summary_lines.append("-" * 6)
            for error in diagnostics.errors:
                summary_lines.append(f"• {error}")
            summary_lines.append("")
        
        # Warnings
        if diagnostics.warnings:
            summary_lines.append("WARNINGS")
            summary_lines.append("-" * 8)
            for warning in diagnostics.warnings:
                summary_lines.append(f"• {warning}")
            summary_lines.append("")
        
        # Recommendations
        if diagnostics.recommendations:
            summary_lines.append("RECOMMENDATIONS")
            summary_lines.append("-" * 15)
            for i, recommendation in enumerate(diagnostics.recommendations, 1):
                summary_lines.append(f"{i}. {recommendation}")
            summary_lines.append("")
        
        summary_lines.append("=" * 60)
        
        return "\n".join(summary_lines)

    def _analyze_pipeline_attempt(self, load_attempt_result: Optional[Any], 
                                model_analysis: ModelAnalysis) -> PipelineAttempt:
        """Analyze pipeline loading attempt results."""
        if load_attempt_result is None:
            # No attempt was made
            pipeline_class = model_analysis.pipeline_class_detected or "WanPipeline"
            return PipelineAttempt(
                attempted_pipeline=pipeline_class,
                pipeline_available=False,
                trust_remote_code=False,
                remote_code_fetched=False,
                load_success=False,
                error_type="no_attempt",
                error_message="No pipeline loading attempt was made",
                load_time_seconds=None
            )
        
        # If load_attempt_result is an exception
        if isinstance(load_attempt_result, Exception):
            pipeline_class = model_analysis.pipeline_class_detected or "WanPipeline"
            return PipelineAttempt.create_failed_attempt(pipeline_class, load_attempt_result)
        
        # If load_attempt_result is a success result (pipeline object)
        return PipelineAttempt(
            attempted_pipeline=type(load_attempt_result).__name__,
            pipeline_available=True,
            trust_remote_code=True,
            remote_code_fetched=True,
            load_success=True,
            error_type=None,
            error_message=None,
            load_time_seconds=None
        )

    def _generate_diagnostics(self, model_analysis: ModelAnalysis, 
                            pipeline_attempt: PipelineAttempt,
                            system_info: Dict[str, Any]) -> tuple[List[str], List[str], List[str]]:
        """Generate errors, warnings, and recommendations based on analysis."""
        errors = []
        warnings = []
        recommendations = []
        
        # Check for critical errors
        if not model_analysis.has_model_index:
            errors.append("model_index.json file not found in model directory")
            recommendations.append("Ensure the model directory contains a valid model_index.json file")
        
        if not model_analysis.model_index_valid:
            errors.append("model_index.json file is corrupted or invalid")
            recommendations.append("Re-download the model or check model_index.json syntax")
        
        if not pipeline_attempt.load_success:
            if pipeline_attempt.error_type == "missing_pipeline_class":
                errors.append(f"{pipeline_attempt.attempted_pipeline} class not found in local environment")
                recommendations.append(f"Install the required pipeline: pip install {pipeline_attempt.attempted_pipeline.lower()}")
                recommendations.append("Enable trust_remote_code=True to fetch pipeline code automatically")
            elif pipeline_attempt.error_type:
                errors.append(f"Pipeline loading failed: {pipeline_attempt.error_message}")
        
        # Check for warnings
        if model_analysis.is_wan_architecture and not pipeline_attempt.trust_remote_code:
            warnings.append("Wan models require trust_remote_code=True for proper loading")
            recommendations.append("Enable trust_remote_code=True in your loading configuration")
        
        # VRAM warnings
        vram_available = system_info.get('vram_available', 0)
        if vram_available and vram_available < 8192:  # Less than 8GB
            warnings.append(f"Available VRAM ({vram_available}MB) may be insufficient for Wan models")
            recommendations.append("Consider enabling CPU offloading to reduce VRAM usage")
            recommendations.append("Use mixed precision (fp16) to reduce memory requirements")
        
        if vram_available and vram_available < 12288:  # Less than 12GB
            warnings.append("VRAM below recommended 12GB for optimal Wan model performance")
            recommendations.append("Enable chunked processing for large videos")
        
        # Component warnings
        missing_components = []
        for comp_name, comp_details in model_analysis.component_details.items():
            if not comp_details.get('config_exists') and not comp_details.get('weights_exist'):
                missing_components.append(comp_name)
        
        if missing_components:
            warnings.append(f"Missing component files: {', '.join(missing_components)}")
            recommendations.append("Re-download the model to ensure all components are present")
        
        # Custom component warnings
        custom_components = [name for name, details in model_analysis.component_details.items() 
                           if details.get('is_custom')]
        if custom_components:
            warnings.append(f"Model uses custom components: {', '.join(custom_components)}")
            recommendations.append("Ensure all custom dependencies are installed")
        
        # System compatibility
        if not system_info.get('cuda_available'):
            warnings.append("CUDA not available - model will run on CPU (very slow)")
            recommendations.append("Install CUDA-compatible PyTorch for GPU acceleration")
        
        # Diffusers version check
        diffusers_version = system_info.get('diffusers_version')
        if not diffusers_version:
            errors.append("Diffusers library not installed")
            recommendations.append("Install diffusers: pip install diffusers")
        elif diffusers_version and diffusers_version < "0.21.0":
            warnings.append(f"Diffusers version {diffusers_version} may be too old for Wan models")
            recommendations.append("Update diffusers: pip install --upgrade diffusers")
        
        return errors, warnings, recommendations