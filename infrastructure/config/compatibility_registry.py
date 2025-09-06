"""
Compatibility Registry System for Wan Model Pipeline Management

This module provides a registry system for mapping model names to their required
pipeline configurations, dependencies, and compatibility information.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PipelineRequirements:
    """Requirements for a specific pipeline configuration"""
    pipeline_class: str
    min_diffusers_version: str
    required_dependencies: List[str]
    pipeline_source: str
    vram_requirements: Dict[str, int]  # {"min_mb": 8192, "recommended_mb": 12288}
    supported_optimizations: List[str]
    trust_remote_code: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineRequirements':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class CompatibilityCheck:
    """Result of compatibility validation"""
    is_compatible: bool
    compatibility_score: float  # 0.0 to 1.0
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class CompatibilityRegistry:
    """Registry mapping model names to required pipeline versions and configurations"""
    
    def __init__(self, registry_path: str = "compatibility_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry: Dict[str, PipelineRequirements] = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load registry from file or create default"""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Convert dict data back to PipelineRequirements objects
                for model_name, req_data in data.items():
                    self.registry[model_name] = PipelineRequirements.from_dict(req_data)
                    
                logger.info(f"Loaded {len(self.registry)} entries from registry")
            else:
                # Create default registry with known Wan models
                self._create_default_registry()
                self._save_registry()
                logger.info("Created default compatibility registry")
                
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self._create_default_registry()
    
    def _create_default_registry(self) -> None:
        """Create default registry with known Wan model configurations"""
        default_entries = {
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers": PipelineRequirements(
                pipeline_class="WanPipeline",
                min_diffusers_version="0.21.0",
                required_dependencies=[
                    "transformers>=4.25.0",
                    "torch>=2.0.0",
                    "accelerate>=0.20.0"
                ],
                pipeline_source="https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                vram_requirements={"min_mb": 8192, "recommended_mb": 12288},
                supported_optimizations=[
                    "cpu_offload", "mixed_precision", "chunked_processing"
                ],
                trust_remote_code=True
            ),
            "Wan-AI/Wan2.2-T2I-A14B-Diffusers": PipelineRequirements(
                pipeline_class="WanPipeline",
                min_diffusers_version="0.21.0",
                required_dependencies=[
                    "transformers>=4.25.0",
                    "torch>=2.0.0",
                    "accelerate>=0.20.0"
                ],
                pipeline_source="https://huggingface.co/Wan-AI/Wan2.2-T2I-A14B-Diffusers",
                vram_requirements={"min_mb": 6144, "recommended_mb": 8192},
                supported_optimizations=[
                    "cpu_offload", "mixed_precision"
                ],
                trust_remote_code=True
            ),
            "Wan-AI/Wan2.2-Mini-T2V": PipelineRequirements(
                pipeline_class="WanPipeline",
                min_diffusers_version="0.21.0",
                required_dependencies=[
                    "transformers>=4.25.0",
                    "torch>=2.0.0"
                ],
                pipeline_source="https://huggingface.co/Wan-AI/Wan2.2-Mini-T2V",
                vram_requirements={"min_mb": 4096, "recommended_mb": 6144},
                supported_optimizations=[
                    "cpu_offload", "mixed_precision", "chunked_processing"
                ],
                trust_remote_code=True
            )
        }
        
        self.registry.update(default_entries)
    
    def _save_registry(self) -> None:
        """Save registry to file"""
        try:
            # Convert PipelineRequirements objects to dicts for JSON serialization
            registry_data = {
                model_name: req.to_dict() 
                for model_name, req in self.registry.items()
            }
            
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved registry with {len(self.registry)} entries")
            
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            raise
    
    def get_pipeline_requirements(self, model_name: str) -> Optional[PipelineRequirements]:
        """Get pipeline requirements for specific model"""
        # Direct lookup first
        if model_name in self.registry:
            return self.registry[model_name]
        
        # Try partial matching for model variants
        for registered_name in self.registry.keys():
            if self._is_model_variant(model_name, registered_name):
                logger.info(f"Found variant match: {model_name} -> {registered_name}")
                return self.registry[registered_name]
        
        logger.warning(f"No pipeline requirements found for model: {model_name}")
        return None
    
    def _is_model_variant(self, model_name: str, registered_name: str) -> bool:
        """Check if model_name is a variant of registered_name"""
        # Simple heuristic: check if base model name matches
        model_base = model_name.split('/')[-1].lower()
        registered_base = registered_name.split('/')[-1].lower()
        
        # Remove common suffixes for comparison
        suffixes = ['-diffusers', '-pytorch', '-safetensors', '-fp16']
        for suffix in suffixes:
            model_base = model_base.replace(suffix, '')
            registered_base = registered_base.replace(suffix, '')
        
        return model_base in registered_base or registered_base in model_base
    
    def register_model_compatibility(self, model_name: str, requirements: PipelineRequirements) -> None:
        """Register new model compatibility information"""
        self.registry[model_name] = requirements
        self._save_registry()
        logger.info(f"Registered compatibility for model: {model_name}")
    
    def update_registry(self, updates: Dict[str, PipelineRequirements]) -> None:
        """Batch update registry with new compatibility information"""
        self.registry.update(updates)
        self._save_registry()
        logger.info(f"Updated registry with {len(updates)} new entries")
    
    def validate_model_pipeline_compatibility(self, model_name: str, 
                                            available_pipeline: str) -> CompatibilityCheck:
        """Check if available pipeline is compatible with model"""
        requirements = self.get_pipeline_requirements(model_name)
        
        if not requirements:
            return CompatibilityCheck(
                is_compatible=False,
                compatibility_score=0.0,
                issues=[f"No compatibility information found for model: {model_name}"],
                warnings=[],
                recommendations=[
                    "Register model in compatibility registry",
                    "Use trust_remote_code=True for custom models"
                ]
            )
        
        issues = []
        warnings = []
        recommendations = []
        score = 1.0
        
        # Check pipeline class compatibility
        if available_pipeline != requirements.pipeline_class:
            issues.append(
                f"Pipeline mismatch: expected {requirements.pipeline_class}, "
                f"got {available_pipeline}"
            )
            score -= 0.5
        
        # Check if trust_remote_code is required
        if requirements.trust_remote_code and available_pipeline == "StableDiffusionPipeline":
            issues.append("Model requires trust_remote_code=True but using standard pipeline")
            score -= 0.3
        
        # Add recommendations based on requirements
        if requirements.supported_optimizations:
            recommendations.extend([
                f"Consider using optimization: {opt}" 
                for opt in requirements.supported_optimizations
            ])
        
        # Check VRAM requirements (if we can detect available VRAM)
        min_vram = requirements.vram_requirements.get("min_mb", 0)
        if min_vram > 0:
            warnings.append(f"Model requires minimum {min_vram}MB VRAM")
        
        is_compatible = len(issues) == 0
        
        return CompatibilityCheck(
            is_compatible=is_compatible,
            compatibility_score=max(0.0, score),
            issues=issues,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def list_registered_models(self) -> List[str]:
        """Get list of all registered model names"""
        return list(self.registry.keys())
    
    def get_models_by_pipeline(self, pipeline_class: str) -> List[str]:
        """Get all models that use a specific pipeline class"""
        return [
            model_name for model_name, req in self.registry.items()
            if req.pipeline_class == pipeline_class
        ]
    
    def export_registry(self, export_path: str) -> None:
        """Export registry to specified path"""
        export_file = Path(export_path)
        registry_data = {
            "export_timestamp": datetime.now().isoformat(),
            "registry_version": "1.0",
            "models": {
                model_name: req.to_dict() 
                for model_name, req in self.registry.items()
            }
        }
        
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(registry_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported registry to {export_path}")
    
    def import_registry(self, import_path: str, merge: bool = True) -> None:
        """Import registry from specified path"""
        import_file = Path(import_path)
        
        if not import_file.exists():
            raise FileNotFoundError(f"Import file not found: {import_path}")
        
        with open(import_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "models" not in data:
            raise ValueError("Invalid registry format: missing 'models' section")
        
        imported_models = {}
        for model_name, req_data in data["models"].items():
            imported_models[model_name] = PipelineRequirements.from_dict(req_data)
        
        if merge:
            self.registry.update(imported_models)
        else:
            self.registry = imported_models
        
        self._save_registry()
        logger.info(f"Imported {len(imported_models)} models from {import_path}")
    
    def validate_registry_integrity(self) -> Dict[str, Any]:
        """Validate registry integrity and return validation report"""
        report = {
            "total_models": len(self.registry),
            "validation_errors": [],
            "validation_warnings": [],
            "pipeline_classes": {},
            "dependency_analysis": {}
        }
        
        # Count pipeline classes
        for model_name, req in self.registry.items():
            pipeline_class = req.pipeline_class
            if pipeline_class not in report["pipeline_classes"]:
                report["pipeline_classes"][pipeline_class] = []
            report["pipeline_classes"][pipeline_class].append(model_name)
        
        # Validate each entry
        for model_name, req in self.registry.items():
            # Check required fields
            if not req.pipeline_class:
                report["validation_errors"].append(
                    f"{model_name}: Missing pipeline_class"
                )
            
            if not req.pipeline_source:
                report["validation_warnings"].append(
                    f"{model_name}: Missing pipeline_source"
                )
            
            # Check VRAM requirements format
            if not isinstance(req.vram_requirements, dict):
                report["validation_errors"].append(
                    f"{model_name}: Invalid vram_requirements format"
                )
        
        return report


# Convenience function for global registry access
_global_registry: Optional[CompatibilityRegistry] = None

def get_compatibility_registry() -> CompatibilityRegistry:
    """Get global compatibility registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = CompatibilityRegistry()
    return _global_registry