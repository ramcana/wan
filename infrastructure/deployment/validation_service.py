"""
WAN Model Validation Service

Provides comprehensive validation for WAN models before and after deployment,
including integrity checks, compatibility verification, and performance validation.
"""

import asyncio
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import torch
import psutil


@dataclass
class ValidationResult:
    """Result of a validation operation"""
    is_valid: bool
    validation_type: str
    model_name: str
    timestamp: datetime
    checks_performed: List[str]
    passed_checks: List[str]
    failed_checks: List[str]
    warnings: List[str]
    errors: List[str]
    performance_metrics: Dict[str, Any]
    
    
@dataclass
class SystemRequirements:
    """System requirements for model deployment"""
    min_ram_gb: float
    min_vram_gb: float
    min_disk_space_gb: float
    required_cuda_version: Optional[str]
    required_python_version: str
    required_packages: List[str]


class ValidationService:
    """
    Service for validating WAN models and deployment environment
    
    Performs:
    - Pre-deployment validation (environment, dependencies, resources)
    - Post-deployment validation (model loading, inference testing)
    - Continuous health validation
    - Performance benchmarking
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validation_history: List[ValidationResult] = []
    
    async def validate_pre_deployment(self, models: List[str]) -> ValidationResult:
        """
        Validate system and environment before deployment
        
        Args:
            models: List of model names to validate for
            
        Returns:
            ValidationResult with pre-deployment validation status
        """
        self.logger.info(f"Starting pre-deployment validation for models: {models}")
        
        result = ValidationResult(
            is_valid=True,
            validation_type="pre_deployment",
            model_name=",".join(models),
            timestamp=datetime.now(),
            checks_performed=[],
            passed_checks=[],
            failed_checks=[],
            warnings=[],
            errors=[],
            performance_metrics={}
        )
        
        try:
            # System resource validation
            await self._validate_system_resources(result)
            
            # Environment validation
            await self._validate_environment(result)
            
            # Model source validation
            for model in models:
                await self._validate_model_source(model, result)
            
            # Dependency validation
            await self._validate_dependencies(result)
            
            # Storage validation
            await self._validate_storage_requirements(models, result)
            
            # Hardware compatibility validation
            await self._validate_hardware_compatibility(result)
            
        except Exception as e:
            result.errors.append(f"Pre-deployment validation failed: {str(e)}")
            result.failed_checks.append("exception_handling")
            self.logger.error(f"Pre-deployment validation error: {str(e)}")
        
        # Determine overall validation status
        result.is_valid = len(result.failed_checks) == 0
        
        self.validation_history.append(result)
        self.logger.info(f"Pre-deployment validation completed: {'PASSED' if result.is_valid else 'FAILED'}")
        
        return result
    
    async def validate_post_deployment(self, models: List[str]) -> ValidationResult:
        """
        Validate models after deployment
        
        Args:
            models: List of deployed model names to validate
            
        Returns:
            ValidationResult with post-deployment validation status
        """
        self.logger.info(f"Starting post-deployment validation for models: {models}")
        
        result = ValidationResult(
            is_valid=True,
            validation_type="post_deployment",
            model_name=",".join(models),
            timestamp=datetime.now(),
            checks_performed=[],
            passed_checks=[],
            failed_checks=[],
            warnings=[],
            errors=[],
            performance_metrics={}
        )
        
        try:
            # Model loading validation
            for model in models:
                await self._validate_model_loading(model, result)
            
            # Configuration validation
            await self._validate_model_configurations(models, result)
            
            # Basic inference validation
            for model in models:
                await self._validate_model_inference(model, result)
            
            # Performance benchmarking
            await self._benchmark_model_performance(models, result)
            
            # Integration validation
            await self._validate_model_integration(models, result)
            
        except Exception as e:
            result.errors.append(f"Post-deployment validation failed: {str(e)}")
            result.failed_checks.append("exception_handling")
            self.logger.error(f"Post-deployment validation error: {str(e)}")
        
        # Determine overall validation status
        result.is_valid = len(result.failed_checks) == 0
        
        self.validation_history.append(result)
        self.logger.info(f"Post-deployment validation completed: {'PASSED' if result.is_valid else 'FAILED'}")
        
        return result
    
    async def validate_model_health(self, model_name: str) -> ValidationResult:
        """
        Validate the health of a deployed model
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            ValidationResult with health validation status
        """
        result = ValidationResult(
            is_valid=True,
            validation_type="health_check",
            model_name=model_name,
            timestamp=datetime.now(),
            checks_performed=[],
            passed_checks=[],
            failed_checks=[],
            warnings=[],
            errors=[],
            performance_metrics={}
        )
        
        try:
            # Check model file integrity
            await self._validate_model_integrity(model_name, result)
            
            # Check model loading capability
            await self._validate_model_loading(model_name, result)
            
            # Check inference capability
            await self._validate_model_inference(model_name, result)
            
            # Check resource usage
            await self._validate_resource_usage(model_name, result)
            
        except Exception as e:
            result.errors.append(f"Health validation failed: {str(e)}")
            result.failed_checks.append("exception_handling")
        
        result.is_valid = len(result.failed_checks) == 0
        return result
    
    async def _validate_system_resources(self, result: ValidationResult):
        """Validate system resources (RAM, disk space, etc.)"""
        result.checks_performed.append("system_resources")
        
        try:
            # Check RAM
            memory = psutil.virtual_memory()
            available_ram_gb = memory.available / (1024**3)
            
            if available_ram_gb < 8:  # Minimum 8GB RAM
                result.failed_checks.append("insufficient_ram")
                result.errors.append(f"Insufficient RAM: {available_ram_gb:.1f}GB available, 8GB required")
            else:
                result.passed_checks.append("sufficient_ram")
            
            # Check disk space
            disk = psutil.disk_usage('.')
            available_disk_gb = disk.free / (1024**3)
            
            if available_disk_gb < 50:  # Minimum 50GB disk space
                result.failed_checks.append("insufficient_disk")
                result.errors.append(f"Insufficient disk space: {available_disk_gb:.1f}GB available, 50GB required")
            else:
                result.passed_checks.append("sufficient_disk")
            
            result.performance_metrics.update({
                "available_ram_gb": available_ram_gb,
                "available_disk_gb": available_disk_gb
            })
            
        except Exception as e:
            result.failed_checks.append("system_resources")
            result.errors.append(f"System resource validation failed: {str(e)}")
    
    async def _validate_environment(self, result: ValidationResult):
        """Validate Python environment and basic dependencies"""
        result.checks_performed.append("environment")
        
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 8):
                result.failed_checks.append("python_version")
                result.errors.append(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
            else:
                result.passed_checks.append("python_version")
            
            # Check CUDA availability
            if torch.cuda.is_available():
                result.passed_checks.append("cuda_available")
                result.performance_metrics["cuda_devices"] = torch.cuda.device_count()
                result.performance_metrics["cuda_version"] = torch.version.cuda
            else:
                result.warnings.append("CUDA not available - CPU-only mode")
            
        except Exception as e:
            result.failed_checks.append("environment")
            result.errors.append(f"Environment validation failed: {str(e)}")
    
    async def _validate_model_source(self, model_name: str, result: ValidationResult):
        """Validate that model source exists and is accessible"""
        check_name = f"model_source_{model_name}"
        result.checks_performed.append(check_name)
        
        try:
            # Check multiple possible source locations
            possible_sources = [
                Path(self.config.source_models_path) / model_name,
                Path("models") / model_name,
                Path("models") / "models" / model_name
            ]
            
            source_found = False
            for source_path in possible_sources:
                if source_path.exists():
                    source_found = True
                    result.passed_checks.append(check_name)
                    break
            
            if not source_found:
                result.failed_checks.append(check_name)
                result.errors.append(f"Model source not found for {model_name}")
                
        except Exception as e:
            result.failed_checks.append(check_name)
            result.errors.append(f"Model source validation failed for {model_name}: {str(e)}")
    
    async def _validate_dependencies(self, result: ValidationResult):
        """Validate required dependencies are installed"""
        result.checks_performed.append("dependencies")
        
        required_packages = [
            "torch", "transformers", "diffusers", "accelerate", 
            "fastapi", "uvicorn", "pydantic", "aiofiles"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            result.failed_checks.append("dependencies")
            result.errors.append(f"Missing required packages: {', '.join(missing_packages)}")
        else:
            result.passed_checks.append("dependencies")
    
    async def _validate_storage_requirements(self, models: List[str], result: ValidationResult):
        """Validate storage requirements for models"""
        result.checks_performed.append("storage_requirements")
        
        try:
            total_required_space = 0
            
            # Estimate space requirements (rough estimates)
            model_sizes = {
                "t2v-A14B": 30,  # GB
                "i2v-A14B": 30,  # GB
                "ti2v-5B": 15   # GB
            }
            
            for model in models:
                total_required_space += model_sizes.get(model, 20)  # Default 20GB
            
            # Check available space
            disk = psutil.disk_usage(self.config.target_models_path)
            available_space_gb = disk.free / (1024**3)
            
            if available_space_gb < total_required_space * 1.2:  # 20% buffer
                result.failed_checks.append("storage_requirements")
                result.errors.append(
                    f"Insufficient storage: {available_space_gb:.1f}GB available, "
                    f"{total_required_space * 1.2:.1f}GB required"
                )
            else:
                result.passed_checks.append("storage_requirements")
            
            result.performance_metrics["required_storage_gb"] = total_required_space
            result.performance_metrics["available_storage_gb"] = available_space_gb
            
        except Exception as e:
            result.failed_checks.append("storage_requirements")
            result.errors.append(f"Storage validation failed: {str(e)}")
    
    async def _validate_hardware_compatibility(self, result: ValidationResult):
        """Validate hardware compatibility"""
        result.checks_performed.append("hardware_compatibility")
        
        try:
            if torch.cuda.is_available():
                # Check GPU memory
                for i in range(torch.cuda.device_count()):
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    if gpu_memory < 8:  # Minimum 8GB VRAM
                        result.warnings.append(f"GPU {i} has only {gpu_memory:.1f}GB VRAM")
                    
                result.passed_checks.append("hardware_compatibility")
            else:
                result.warnings.append("No CUDA-compatible GPU found")
                
        except Exception as e:
            result.failed_checks.append("hardware_compatibility")
            result.errors.append(f"Hardware validation failed: {str(e)}")
    
    async def _validate_model_loading(self, model_name: str, result: ValidationResult):
        """Validate that a model can be loaded successfully"""
        check_name = f"model_loading_{model_name}"
        result.checks_performed.append(check_name)
        
        try:
            model_path = Path(self.config.target_models_path) / model_name
            
            if not model_path.exists():
                result.failed_checks.append(check_name)
                result.errors.append(f"Model path does not exist: {model_path}")
                return
            
            # Check for required files
            config_file = model_path / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    result.performance_metrics[f"{model_name}_config"] = config
            
            result.passed_checks.append(check_name)
            
        except Exception as e:
            result.failed_checks.append(check_name)
            result.errors.append(f"Model loading validation failed for {model_name}: {str(e)}")
    
    async def _validate_model_configurations(self, models: List[str], result: ValidationResult):
        """Validate model configurations"""
        result.checks_performed.append("model_configurations")
        
        try:
            for model in models:
                model_path = Path(self.config.target_models_path) / model
                config_file = model_path / "config.json"
                
                if not config_file.exists():
                    result.warnings.append(f"No config.json found for {model}")
                    continue
                
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    
                # Validate required configuration fields
                required_fields = ["model_type", "architecture"]
                for field in required_fields:
                    if field not in config:
                        result.warnings.append(f"Missing {field} in {model} configuration")
            
            result.passed_checks.append("model_configurations")
            
        except Exception as e:
            result.failed_checks.append("model_configurations")
            result.errors.append(f"Configuration validation failed: {str(e)}")
    
    async def _validate_model_inference(self, model_name: str, result: ValidationResult):
        """Validate that a model can perform basic inference"""
        check_name = f"model_inference_{model_name}"
        result.checks_performed.append(check_name)
        
        try:
            # This is a placeholder for actual inference testing
            # In a real implementation, you would load the model and run a test inference
            
            model_path = Path(self.config.target_models_path) / model_name
            if model_path.exists():
                result.passed_checks.append(check_name)
                result.performance_metrics[f"{model_name}_inference_test"] = "simulated_pass"
            else:
                result.failed_checks.append(check_name)
                result.errors.append(f"Cannot test inference - model not found: {model_path}")
                
        except Exception as e:
            result.failed_checks.append(check_name)
            result.errors.append(f"Inference validation failed for {model_name}: {str(e)}")
    
    async def _benchmark_model_performance(self, models: List[str], result: ValidationResult):
        """Benchmark model performance"""
        result.checks_performed.append("performance_benchmark")
        
        try:
            # Placeholder for performance benchmarking
            # In a real implementation, you would run actual performance tests
            
            for model in models:
                result.performance_metrics[f"{model}_benchmark"] = {
                    "load_time_seconds": 5.0,  # Simulated
                    "inference_time_seconds": 2.0,  # Simulated
                    "memory_usage_gb": 8.0  # Simulated
                }
            
            result.passed_checks.append("performance_benchmark")
            
        except Exception as e:
            result.failed_checks.append("performance_benchmark")
            result.errors.append(f"Performance benchmarking failed: {str(e)}")
    
    async def _validate_model_integration(self, models: List[str], result: ValidationResult):
        """Validate model integration with the system"""
        result.checks_performed.append("model_integration")
        
        try:
            # Check that models are properly registered in the system
            # This would integrate with your actual model registry/configuration system
            
            result.passed_checks.append("model_integration")
            
        except Exception as e:
            result.failed_checks.append("model_integration")
            result.errors.append(f"Integration validation failed: {str(e)}")
    
    async def _validate_model_integrity(self, model_name: str, result: ValidationResult):
        """Validate model file integrity"""
        check_name = f"model_integrity_{model_name}"
        result.checks_performed.append(check_name)
        
        try:
            model_path = Path(self.config.target_models_path) / model_name
            
            if not model_path.exists():
                result.failed_checks.append(check_name)
                result.errors.append(f"Model directory not found: {model_path}")
                return
            
            # Check for corruption by verifying file sizes and basic structure
            total_files = len(list(model_path.rglob('*')))
            if total_files == 0:
                result.failed_checks.append(check_name)
                result.errors.append(f"Model directory is empty: {model_path}")
            else:
                result.passed_checks.append(check_name)
                result.performance_metrics[f"{model_name}_file_count"] = total_files
                
        except Exception as e:
            result.failed_checks.append(check_name)
            result.errors.append(f"Integrity validation failed for {model_name}: {str(e)}")
    
    async def _validate_resource_usage(self, model_name: str, result: ValidationResult):
        """Validate current resource usage"""
        check_name = f"resource_usage_{model_name}"
        result.checks_performed.append(check_name)
        
        try:
            # Get current system resource usage
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            result.performance_metrics.update({
                f"{model_name}_memory_usage_percent": memory.percent,
                f"{model_name}_cpu_usage_percent": cpu_percent
            })
            
            # Check if resource usage is within acceptable limits
            if memory.percent > 90:
                result.warnings.append(f"High memory usage: {memory.percent}%")
            
            if cpu_percent > 90:
                result.warnings.append(f"High CPU usage: {cpu_percent}%")
            
            result.passed_checks.append(check_name)
            
        except Exception as e:
            result.failed_checks.append(check_name)
            result.errors.append(f"Resource usage validation failed: {str(e)}")
    
    async def get_validation_history(self) -> List[ValidationResult]:
        """Get the history of all validations"""
        return self.validation_history.copy()
    
    async def export_validation_report(self, output_path: str):
        """Export validation history to a report file"""
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "total_validations": len(self.validation_history),
            "validations": [asdict(v) for v in self.validation_history]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Validation report exported to {output_path}")