"""
WAN Model Compatibility Fix

This module provides a comprehensive fix for WAN model loading issues,
ensuring proper pipeline selection and fallback handling.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class WanModelCompatibilityFix:
    """
    Comprehensive fix for WAN model compatibility issues.
    
    Addresses the core problem where WAN models are detected correctly
    but fall back to StableDiffusionPipeline instead of loading properly.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".WanModelCompatibilityFix")
    
    def fix_wan_model_loading(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Apply comprehensive fix for WAN model loading.
        
        Args:
            model_path: Path to the WAN model
            
        Returns:
            Dictionary with fix results and recommendations
        """
        model_path = Path(model_path)
        fix_results = {
            "model_path": str(model_path),
            "fixes_applied": [],
            "issues_found": [],
            "recommendations": [],
            "success": False
        }
        
        self.logger.info(f"Applying WAN model compatibility fix for: {model_path}")
        
        try:
            # Step 1: Validate model structure
            structure_issues = self._validate_model_structure(model_path)
            if structure_issues:
                fix_results["issues_found"].extend(structure_issues)
            
            # Step 2: Fix model_index.json if needed
            index_fixes = self._fix_model_index(model_path)
            if index_fixes:
                fix_results["fixes_applied"].extend(index_fixes)
            
            # Step 3: Ensure proper component configurations
            component_fixes = self._fix_component_configs(model_path)
            if component_fixes:
                fix_results["fixes_applied"].extend(component_fixes)
            
            # Step 4: Create pipeline loading wrapper
            wrapper_created = self._create_pipeline_wrapper(model_path)
            if wrapper_created:
                fix_results["fixes_applied"].append("Created pipeline loading wrapper")
            
            # Step 5: Generate recommendations
            recommendations = self._generate_recommendations(model_path, fix_results)
            fix_results["recommendations"] = recommendations
            
            fix_results["success"] = len(fix_results["fixes_applied"]) > 0 or len(fix_results["issues_found"]) == 0
            
            self.logger.info(f"Fix completed. Applied {len(fix_results['fixes_applied'])} fixes, "
                           f"found {len(fix_results['issues_found'])} issues")
            
            return fix_results
            
        except Exception as e:
            self.logger.error(f"Fix failed: {e}")
            fix_results["issues_found"].append(f"Fix process failed: {str(e)}")
            return fix_results
    
    def _validate_model_structure(self, model_path: Path) -> list:
        """Validate WAN model directory structure"""
        issues = []
        
        if not model_path.exists():
            issues.append(f"Model path does not exist: {model_path}")
            return issues
        
        # Check for essential files
        model_index_path = model_path / "model_index.json"
        if not model_index_path.exists():
            issues.append("Missing model_index.json")
        
        # Check for WAN-specific components
        expected_components = ["scheduler", "vae"]
        wan_components = ["transformer", "transformer_2"]
        
        found_components = []
        for component in expected_components + wan_components:
            component_path = model_path / component
            if component_path.exists():
                found_components.append(component)
        
        if not any(comp in found_components for comp in wan_components):
            issues.append("No WAN-specific components (transformer/transformer_2) found")
        
        # Check for WAN model files
        wan_files = [
            "pytorch_model.bin",
            "Wan2.1_VAE.pth", 
            "models_t5_umt5-xxl-enc-bf16.pth"
        ]
        
        found_wan_files = []
        for wan_file in wan_files:
            if (model_path / wan_file).exists():
                found_wan_files.append(wan_file)
        
        if found_wan_files:
            self.logger.info(f"Found WAN-specific files: {found_wan_files}")
        
        return issues
    
    def _fix_model_index(self, model_path: Path) -> list:
        """Fix model_index.json for proper WAN model detection"""
        fixes = []
        model_index_path = model_path / "model_index.json"
        
        if not model_index_path.exists():
            return fixes
        
        try:
            with open(model_index_path, 'r') as f:
                model_index = json.load(f)
            
            original_index = model_index.copy()
            modified = False
            
            # Ensure proper pipeline class for WAN models
            if "_class_name" not in model_index or not model_index["_class_name"].startswith("Wan"):
                # Check if this looks like a WAN model
                if any(key in model_index for key in ["transformer", "transformer_2", "boundary_ratio"]):
                    model_index["_class_name"] = "WanPipeline"
                    modified = True
                    fixes.append("Set pipeline class to WanPipeline")
            
            # Ensure diffusers version is compatible
            if "_diffusers_version" not in model_index:
                model_index["_diffusers_version"] = "0.21.0"
                modified = True
                fixes.append("Added diffusers version")
            
            # Add boundary_ratio if missing for T2V models
            if "transformer_2" in model_index and "boundary_ratio" not in model_index:
                model_index["boundary_ratio"] = 0.5  # Default value
                modified = True
                fixes.append("Added default boundary_ratio")
            
            # Save modified index
            if modified:
                # Backup original
                backup_path = model_index_path.with_suffix('.json.backup')
                with open(backup_path, 'w') as f:
                    json.dump(original_index, f, indent=2)
                
                # Save modified
                with open(model_index_path, 'w') as f:
                    json.dump(model_index, f, indent=2)
                
                fixes.append("Updated model_index.json (backup created)")
                self.logger.info(f"Updated model_index.json, backup saved to {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to fix model_index.json: {e}")
        
        return fixes
    
    def _fix_component_configs(self, model_path: Path) -> list:
        """Fix component configurations for WAN models"""
        fixes = []
        
        # Check transformer components
        for transformer_name in ["transformer", "transformer_2"]:
            transformer_path = model_path / transformer_name
            if transformer_path.exists():
                config_path = transformer_path / "config.json"
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        
                        # Ensure proper class name
                        if "_class_name" not in config or not config["_class_name"].startswith("Wan"):
                            config["_class_name"] = f"Wan{transformer_name.title()}"
                            
                            with open(config_path, 'w') as f:
                                json.dump(config, f, indent=2)
                            
                            fixes.append(f"Fixed {transformer_name} config class name")
                    
                    except Exception as e:
                        self.logger.error(f"Failed to fix {transformer_name} config: {e}")
        
        return fixes
    
    def _create_pipeline_wrapper(self, model_path: Path) -> bool:
        """Create a pipeline loading wrapper for WAN models"""
        try:
            wrapper_path = model_path / "wan_pipeline_wrapper.py"
            
            if wrapper_path.exists():
                return False  # Already exists
            
            wrapper_code = '''"""
WAN Pipeline Loading Wrapper

This wrapper ensures proper loading of WAN models with trust_remote_code=True
and handles the specific requirements of WAN model architecture.
"""

from diffusers import DiffusionPipeline
import logging

logger = logging.getLogger(__name__)

def load_wan_pipeline(model_path, **kwargs):
    """
    Load WAN pipeline with proper configuration.
    
    Args:
        model_path: Path to the WAN model
        **kwargs: Additional arguments for pipeline loading
        
    Returns:
        Loaded WAN pipeline
    """
    # Ensure trust_remote_code is enabled
    kwargs["trust_remote_code"] = True
    
    # Set default torch_dtype if not specified
    if "torch_dtype" not in kwargs:
        import torch
        kwargs["torch_dtype"] = torch.float16
    
    logger.info(f"Loading WAN pipeline from {model_path}")
    
    try:
        pipeline = DiffusionPipeline.from_pretrained(model_path, **kwargs)
        logger.info(f"Successfully loaded WAN pipeline: {type(pipeline).__name__}")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load WAN pipeline: {e}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        pipeline = load_wan_pipeline(model_path)
        print(f"Loaded pipeline: {type(pipeline).__name__}")
'''
            
            with open(wrapper_path, 'w') as f:
                f.write(wrapper_code)
            
            self.logger.info(f"Created pipeline wrapper at {wrapper_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create pipeline wrapper: {e}")
            return False
    
    def _generate_recommendations(self, model_path: Path, fix_results: Dict[str, Any]) -> list:
        """Generate recommendations based on fix results"""
        recommendations = []
        
        # Always recommend trust_remote_code
        recommendations.append("Always use trust_remote_code=True when loading WAN models")
        
        # Check if model has proper structure
        if "Missing model_index.json" in fix_results["issues_found"]:
            recommendations.append("Ensure model has proper model_index.json file")
        
        # Memory recommendations
        recommendations.append("Use torch.float16 or torch.bfloat16 for memory efficiency")
        recommendations.append("Consider CPU offloading for large models (>12GB VRAM required)")
        
        # Pipeline loading recommendations
        recommendations.append("Use DiffusionPipeline.from_pretrained() with trust_remote_code=True")
        recommendations.append("Ensure diffusers>=0.21.0 for WAN model support")
        
        # Specific WAN model recommendations
        model_index_path = model_path / "model_index.json"
        if model_index_path.exists():
            try:
                with open(model_index_path, 'r') as f:
                    model_index = json.load(f)
                
                if "transformer_2" in model_index:
                    recommendations.append("This appears to be a T2V model - ensure sufficient VRAM (12GB+)")
                
                if "boundary_ratio" in model_index:
                    recommendations.append("Model supports boundary ratio control for video generation")
                    
            except:
                pass
        
        return recommendations


def apply_wan_compatibility_fix(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to apply WAN model compatibility fix.
    
    Args:
        model_path: Path to the WAN model
        
    Returns:
        Dictionary with fix results
    """
    fixer = WanModelCompatibilityFix()
    return fixer.fix_wan_model_loading(model_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python wan_model_compatibility_fix.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    results = apply_wan_compatibility_fix(model_path)
    
    print("WAN Model Compatibility Fix Results")
    print("=" * 40)
    print(f"Model: {results['model_path']}")
    print(f"Success: {results['success']}")
    
    if results['fixes_applied']:
        print(f"\nFixes Applied ({len(results['fixes_applied'])}):")
        for fix in results['fixes_applied']:
            print(f"  ✅ {fix}")
    
    if results['issues_found']:
        print(f"\nIssues Found ({len(results['issues_found'])}):")
        for issue in results['issues_found']:
            print(f"  ❌ {issue}")
    
    if results['recommendations']:
        print(f"\nRecommendations ({len(results['recommendations'])}):")
        for rec in results['recommendations']:
            print(f"  💡 {rec}")