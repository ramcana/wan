"""
WAN Model LoRA Integration Manager
Extends existing LoRAManager to work with WAN model architectures
"""

import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Import existing LoRAManager
try:
    from .utils import LoRAManager
    LORA_MANAGER_AVAILABLE = True
except ImportError:
    LORA_MANAGER_AVAILABLE = False
    # Create fallback LoRAManager
    class LoRAManager:
        def __init__(self, config):
            self.config = config
            self.loaded_loras = {}
            self.applied_loras = {}

logger = logging.getLogger(__name__)

class WANModelType(Enum):
    """WAN model types for LoRA compatibility"""
    T2V_A14B = "t2v-A14B"
    I2V_A14B = "i2v-A14B"
    TI2V_5B = "ti2v-5B"
    UNKNOWN = "unknown"

@dataclass
class WANLoRACompatibility:
    """LoRA compatibility information for WAN models"""
    model_type: WANModelType
    supports_lora: bool
    max_lora_count: int
    supported_lora_types: List[str]
    target_modules: List[str]
    architecture_specific_notes: str
    memory_overhead_factor: float = 1.2

@dataclass
class WANLoRAStatus:
    """Status of LoRA application to WAN models"""
    lora_name: str
    model_type: WANModelType
    is_compatible: bool
    is_applied: bool
    current_strength: float
    target_modules_affected: List[str]
    memory_usage_mb: float
    application_method: str
    error_message: Optional[str] = None

class WANLoRAManager(LoRAManager):
    """
    Enhanced LoRA Manager for WAN model architectures
    Extends existing LoRAManager with WAN-specific compatibility and optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # WAN model specific configuration
        self.wan_model_compatibility = self._initialize_wan_compatibility()
        self.wan_applied_loras: Dict[str, Dict[str, WANLoRAStatus]] = {}  # model_id -> {lora_name -> status}
        
        # LoRA blending capabilities
        self.lora_blending_enabled = config.get("lora_blending_enabled", True)
        self.max_blended_loras = config.get("max_blended_loras", 3)
        
        logger.info("WAN LoRA Manager initialized with WAN model support")
    
    def _initialize_wan_compatibility(self) -> Dict[WANModelType, WANLoRACompatibility]:
        """Initialize WAN model LoRA compatibility matrix"""
        return {
            WANModelType.T2V_A14B: WANLoRACompatibility(
                model_type=WANModelType.T2V_A14B,
                supports_lora=True,
                max_lora_count=2,
                supported_lora_types=["style", "character", "concept"],
                target_modules=[
                    "transformer.transformer_blocks.*.attn1.to_q",
                    "transformer.transformer_blocks.*.attn1.to_k", 
                    "transformer.transformer_blocks.*.attn1.to_v",
                    "transformer.transformer_blocks.*.attn2.to_q",
                    "transformer.transformer_blocks.*.attn2.to_k",
                    "transformer.transformer_blocks.*.attn2.to_v",
                    "transformer.transformer_blocks.*.ff.net.0.proj",
                    "transformer.transformer_blocks.*.ff.net.2"
                ],
                architecture_specific_notes="WAN T2V uses transformer architecture, LoRA applied to attention and feed-forward layers",
                memory_overhead_factor=1.3
            ),
            WANModelType.I2V_A14B: WANLoRACompatibility(
                model_type=WANModelType.I2V_A14B,
                supports_lora=True,
                max_lora_count=2,
                supported_lora_types=["style", "motion", "character"],
                target_modules=[
                    "transformer.transformer_blocks.*.attn1.to_q",
                    "transformer.transformer_blocks.*.attn1.to_k",
                    "transformer.transformer_blocks.*.attn1.to_v",
                    "transformer.transformer_blocks.*.attn2.to_q",
                    "transformer.transformer_blocks.*.attn2.to_k", 
                    "transformer.transformer_blocks.*.attn2.to_v",
                    "transformer.image_proj_layers.*.proj"
                ],
                architecture_specific_notes="WAN I2V includes image conditioning layers, LoRA can affect image-to-video translation",
                memory_overhead_factor=1.4
            ),
            WANModelType.TI2V_5B: WANLoRACompatibility(
                model_type=WANModelType.TI2V_5B,
                supports_lora=True,
                max_lora_count=3,
                supported_lora_types=["style", "character", "concept", "motion"],
                target_modules=[
                    "transformer.transformer_blocks.*.attn1.to_q",
                    "transformer.transformer_blocks.*.attn1.to_k",
                    "transformer.transformer_blocks.*.attn1.to_v", 
                    "transformer.transformer_blocks.*.attn2.to_q",
                    "transformer.transformer_blocks.*.attn2.to_k",
                    "transformer.transformer_blocks.*.attn2.to_v",
                    "transformer.transformer_blocks.*.ff.net.0.proj",
                    "transformer.transformer_blocks.*.ff.net.2",
                    "transformer.image_proj_layers.*.proj",
                    "transformer.text_proj_layers.*.proj"
                ],
                architecture_specific_notes="WAN TI2V 5B model supports both text and image conditioning with enhanced LoRA capacity",
                memory_overhead_factor=1.5
            )
        }
    
    def check_wan_model_compatibility(self, model, lora_name: str) -> Tuple[bool, WANLoRACompatibility, str]:
        """
        Check LoRA compatibility with WAN model architecture
        
        Args:
            model: WAN model instance
            lora_name: Name of LoRA to check
            
        Returns:
            Tuple of (is_compatible, compatibility_info, reason)
        """
        try:
            # Detect WAN model type
            model_type = self._detect_wan_model_type(model)
            
            if model_type == WANModelType.UNKNOWN:
                return False, None, "Unknown WAN model type - cannot determine LoRA compatibility"
            
            compatibility = self.wan_model_compatibility[model_type]
            
            if not compatibility.supports_lora:
                return False, compatibility, f"WAN model type {model_type.value} does not support LoRA"
            
            # Check if LoRA is loaded
            if lora_name not in self.loaded_loras:
                return False, compatibility, f"LoRA {lora_name} is not loaded"
            
            # Check LoRA structure compatibility with WAN architecture
            lora_info = self.loaded_loras[lora_name]
            lora_weights = lora_info["weights"]
            
            # Validate LoRA keys match WAN model target modules
            compatible_keys = self._validate_wan_lora_keys(lora_weights, compatibility.target_modules)
            
            if not compatible_keys:
                return False, compatibility, f"LoRA {lora_name} keys do not match WAN model architecture"
            
            # Check current LoRA count limit
            model_id = self._get_model_id(model)
            current_loras = len(self.wan_applied_loras.get(model_id, {}))
            
            if current_loras >= compatibility.max_lora_count:
                return False, compatibility, f"Maximum LoRA count ({compatibility.max_lora_count}) reached for this WAN model"
            
            return True, compatibility, "LoRA is compatible with WAN model"
            
        except Exception as e:
            logger.error(f"Error checking WAN model LoRA compatibility: {e}")
            return False, None, f"Compatibility check failed: {str(e)}"
    
    def _detect_wan_model_type(self, model) -> WANModelType:
        """Detect WAN model type from model instance"""
        try:
            # Check model class name
            model_class = model.__class__.__name__
            
            if "T2V" in model_class or "Text2Video" in model_class:
                if "A14B" in str(model) or "14B" in model_class:
                    return WANModelType.T2V_A14B
            elif "I2V" in model_class or "Image2Video" in model_class:
                if "A14B" in str(model) or "14B" in model_class:
                    return WANModelType.I2V_A14B
            elif "TI2V" in model_class or "TextImage2Video" in model_class:
                if "5B" in str(model) or "5B" in model_class:
                    return WANModelType.TI2V_5B
            
            # Check model configuration if available
            if hasattr(model, 'config'):
                config = model.config
                if hasattr(config, 'model_type'):
                    model_type_str = config.model_type.lower()
                    if "t2v" in model_type_str and "a14b" in model_type_str:
                        return WANModelType.T2V_A14B
                    elif "i2v" in model_type_str and "a14b" in model_type_str:
                        return WANModelType.I2V_A14B
                    elif "ti2v" in model_type_str and "5b" in model_type_str:
                        return WANModelType.TI2V_5B
            
            # Check for WAN-specific components
            if hasattr(model, 'transformer'):
                # WAN models use transformer architecture
                transformer = model.transformer
                if hasattr(transformer, 'transformer_blocks'):
                    # Count parameters to estimate model size
                    param_count = sum(p.numel() for p in transformer.parameters())
                    
                    if param_count > 4e9:  # > 4B parameters
                        return WANModelType.TI2V_5B
                    else:
                        # Check for image conditioning to distinguish T2V vs I2V
                        if hasattr(transformer, 'image_proj_layers') or hasattr(model, 'image_encoder'):
                            return WANModelType.I2V_A14B
                        else:
                            return WANModelType.T2V_A14B
            
            logger.warning(f"Could not detect WAN model type for {model_class}")
            return WANModelType.UNKNOWN
            
        except Exception as e:
            logger.error(f"Error detecting WAN model type: {e}")
            return WANModelType.UNKNOWN
    
    def _validate_wan_lora_keys(self, lora_weights: Dict[str, torch.Tensor], target_modules: List[str]) -> bool:
        """Validate that LoRA keys match WAN model target modules"""
        try:
            lora_keys = list(lora_weights.keys())
            
            # Extract base module names from LoRA keys
            lora_modules = set()
            for key in lora_keys:
                if 'lora_up' in key or 'lora_down' in key:
                    base_key = key.replace('.lora_up.weight', '').replace('.lora_down.weight', '')
                    lora_modules.add(base_key)
            
            # Check if any LoRA modules match target modules (with wildcard support)
            compatible_modules = 0
            for lora_module in lora_modules:
                for target_pattern in target_modules:
                    if self._match_module_pattern(lora_module, target_pattern):
                        compatible_modules += 1
                        break
            
            # Require at least some compatible modules
            return compatible_modules > 0
            
        except Exception as e:
            logger.error(f"Error validating WAN LoRA keys: {e}")
            return False
    
    def _match_module_pattern(self, module_name: str, pattern: str) -> bool:
        """Match module name against pattern with wildcard support"""
        try:
            # Simple wildcard matching for transformer blocks
            if '*' in pattern:
                pattern_parts = pattern.split('*')
                if len(pattern_parts) == 2:
                    prefix, suffix = pattern_parts
                    return module_name.startswith(prefix) and module_name.endswith(suffix)
            else:
                return module_name == pattern
            
            return False
            
        except Exception as e:
            logger.error(f"Error matching module pattern: {e}")
            return False
    
    def apply_wan_lora(self, model, lora_name: str, strength: float = 1.0) -> WANLoRAStatus:
        """
        Apply LoRA to WAN model with architecture-specific handling
        
        Args:
            model: WAN model instance
            lora_name: Name of LoRA to apply
            strength: LoRA strength (0.0 to 2.0)
            
        Returns:
            WANLoRAStatus with application results
        """
        try:
            # Check compatibility first
            is_compatible, compatibility, reason = self.check_wan_model_compatibility(model, lora_name)
            
            if not is_compatible:
                return WANLoRAStatus(
                    lora_name=lora_name,
                    model_type=WANModelType.UNKNOWN,
                    is_compatible=False,
                    is_applied=False,
                    current_strength=0.0,
                    target_modules_affected=[],
                    memory_usage_mb=0.0,
                    application_method="none",
                    error_message=reason
                )
            
            model_type = self._detect_wan_model_type(model)
            model_id = self._get_model_id(model)
            
            logger.info(f"Applying LoRA {lora_name} to WAN model {model_type.value} with strength {strength}")
            
            # Apply LoRA using WAN-specific method
            application_method, affected_modules, memory_usage = self._apply_wan_lora_internal(
                model, lora_name, strength, compatibility
            )
            
            # Track applied LoRA
            if model_id not in self.wan_applied_loras:
                self.wan_applied_loras[model_id] = {}
            
            status = WANLoRAStatus(
                lora_name=lora_name,
                model_type=model_type,
                is_compatible=True,
                is_applied=True,
                current_strength=strength,
                target_modules_affected=affected_modules,
                memory_usage_mb=memory_usage,
                application_method=application_method
            )
            
            self.wan_applied_loras[model_id][lora_name] = status
            
            # Also track in parent class
            self.applied_loras[lora_name] = strength
            
            logger.info(f"Successfully applied WAN LoRA {lora_name} using {application_method} method")
            return status
            
        except Exception as e:
            logger.error(f"Failed to apply WAN LoRA {lora_name}: {e}")
            return WANLoRAStatus(
                lora_name=lora_name,
                model_type=self._detect_wan_model_type(model),
                is_compatible=False,
                is_applied=False,
                current_strength=0.0,
                target_modules_affected=[],
                memory_usage_mb=0.0,
                application_method="failed",
                error_message=str(e)
            )
    
    def _apply_wan_lora_internal(self, model, lora_name: str, strength: float, 
                                compatibility: WANLoRACompatibility) -> Tuple[str, List[str], float]:
        """Internal method to apply LoRA to WAN model"""
        try:
            lora_info = self.loaded_loras[lora_name]
            lora_weights = lora_info["weights"]
            
            # Try diffusers built-in method first (if available)
            if hasattr(model, 'load_lora_weights') and hasattr(model, 'set_adapters'):
                try:
                    model.load_lora_weights(lora_info["path"])
                    model.set_adapters([lora_name], adapter_weights=[strength])
                    
                    # Estimate memory usage
                    memory_usage = lora_info["size_mb"] * compatibility.memory_overhead_factor
                    
                    return "diffusers_builtin", compatibility.target_modules, memory_usage
                    
                except Exception as e:
                    logger.warning(f"Diffusers built-in LoRA loading failed: {e}, falling back to manual method")
            
            # Manual WAN-specific LoRA application
            affected_modules = self._apply_wan_lora_manual(model, lora_weights, strength, compatibility)
            
            # Estimate memory usage
            memory_usage = lora_info["size_mb"] * compatibility.memory_overhead_factor
            
            return "wan_manual", affected_modules, memory_usage
            
        except Exception as e:
            logger.error(f"Error in WAN LoRA internal application: {e}")
            raise
    
    def _apply_wan_lora_manual(self, model, lora_weights: Dict[str, torch.Tensor], 
                              strength: float, compatibility: WANLoRACompatibility) -> List[str]:
        """Manually apply LoRA weights to WAN model transformer architecture"""
        affected_modules = []
        
        try:
            # Get the transformer component (main target for WAN models)
            if hasattr(model, 'transformer'):
                target_model = model.transformer
            else:
                target_model = model
            
            # Apply LoRA weights to transformer blocks
            for key, weight in lora_weights.items():
                if 'lora_up' in key:
                    # Extract the base layer name
                    base_key = key.replace('.lora_up.weight', '')
                    down_key = key.replace('lora_up', 'lora_down')
                    
                    if down_key in lora_weights:
                        try:
                            # Navigate to the parameter in WAN transformer
                            param_path = base_key.split('.')
                            current_module = target_model
                            
                            # Handle WAN-specific path navigation
                            for i, attr in enumerate(param_path):
                                if attr.isdigit():
                                    # Handle transformer block indexing
                                    current_module = current_module[int(attr)]
                                else:
                                    if hasattr(current_module, attr):
                                        current_module = getattr(current_module, attr)
                                    else:
                                        # Skip if module doesn't exist in this model variant
                                        break
                            else:
                                # Successfully navigated to parameter
                                if hasattr(current_module, 'weight'):
                                    original_param = current_module.weight
                                    
                                    # Calculate LoRA delta: strength * (up @ down)
                                    up_weight = weight
                                    down_weight = lora_weights[down_key]
                                    
                                    lora_delta = strength * torch.mm(up_weight, down_weight)
                                    
                                    # Add to original parameter
                                    with torch.no_grad():
                                        original_param.data += lora_delta.to(
                                            original_param.device, 
                                            original_param.dtype
                                        )
                                    
                                    affected_modules.append(base_key)
                                    
                        except Exception as e:
                            logger.warning(f"Failed to apply WAN LoRA weight {key}: {e}")
                            continue
            
            logger.info(f"Applied WAN LoRA to {len(affected_modules)} modules")
            return affected_modules
            
        except Exception as e:
            logger.error(f"Error in manual WAN LoRA application: {e}")
            return affected_modules
    
    def adjust_wan_lora_strength(self, model, lora_name: str, new_strength: float) -> WANLoRAStatus:
        """Adjust LoRA strength for WAN model"""
        try:
            model_id = self._get_model_id(model)
            
            if (model_id not in self.wan_applied_loras or 
                lora_name not in self.wan_applied_loras[model_id]):
                raise ValueError(f"LoRA {lora_name} is not applied to this WAN model")
            
            if new_strength < 0.0 or new_strength > 2.0:
                raise ValueError("LoRA strength must be between 0.0 and 2.0")
            
            current_status = self.wan_applied_loras[model_id][lora_name]
            
            logger.info(f"Adjusting WAN LoRA {lora_name} strength from {current_status.current_strength} to {new_strength}")
            
            # For diffusers built-in method
            if current_status.application_method == "diffusers_builtin":
                if hasattr(model, 'set_adapters'):
                    model.set_adapters([lora_name], adapter_weights=[new_strength])
                    current_status.current_strength = new_strength
                    self.applied_loras[lora_name] = new_strength
                    return current_status
            
            # For manual method, need to remove and reapply
            self.remove_wan_lora(model, lora_name)
            return self.apply_wan_lora(model, lora_name, new_strength)
            
        except Exception as e:
            logger.error(f"Failed to adjust WAN LoRA strength: {e}")
            raise
    
    def remove_wan_lora(self, model, lora_name: str) -> bool:
        """Remove LoRA from WAN model"""
        try:
            model_id = self._get_model_id(model)
            
            if (model_id not in self.wan_applied_loras or 
                lora_name not in self.wan_applied_loras[model_id]):
                logger.warning(f"LoRA {lora_name} is not applied to this WAN model")
                return False
            
            current_status = self.wan_applied_loras[model_id][lora_name]
            
            logger.info(f"Removing WAN LoRA: {lora_name}")
            
            # For diffusers built-in method
            if current_status.application_method == "diffusers_builtin":
                if hasattr(model, 'unfuse_lora'):
                    model.unfuse_lora()
                elif hasattr(model, 'unload_lora_weights'):
                    model.unload_lora_weights()
            else:
                # Manual removal would require storing original weights
                logger.warning("Manual WAN LoRA removal requires model reload")
            
            # Remove from tracking
            del self.wan_applied_loras[model_id][lora_name]
            if lora_name in self.applied_loras:
                del self.applied_loras[lora_name]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove WAN LoRA {lora_name}: {e}")
            return False
    
    def blend_wan_loras(self, model, lora_configs: List[Dict[str, Any]]) -> List[WANLoRAStatus]:
        """
        Blend multiple LoRAs on WAN model
        
        Args:
            model: WAN model instance
            lora_configs: List of {"name": str, "strength": float} configs
            
        Returns:
            List of WANLoRAStatus for each LoRA
        """
        try:
            if not self.lora_blending_enabled:
                raise ValueError("LoRA blending is disabled")
            
            if len(lora_configs) > self.max_blended_loras:
                raise ValueError(f"Cannot blend more than {self.max_blended_loras} LoRAs")
            
            model_type = self._detect_wan_model_type(model)
            compatibility = self.wan_model_compatibility.get(model_type)
            
            if not compatibility:
                raise ValueError(f"Unknown WAN model type: {model_type}")
            
            if len(lora_configs) > compatibility.max_lora_count:
                raise ValueError(f"WAN model {model_type.value} supports maximum {compatibility.max_lora_count} LoRAs")
            
            results = []
            
            # Apply each LoRA in sequence
            for lora_config in lora_configs:
                lora_name = lora_config["name"]
                strength = lora_config.get("strength", 1.0)
                
                status = self.apply_wan_lora(model, lora_name, strength)
                results.append(status)
                
                if not status.is_applied:
                    logger.warning(f"Failed to apply LoRA {lora_name} in blend: {status.error_message}")
            
            successful_loras = [r for r in results if r.is_applied]
            logger.info(f"Successfully blended {len(successful_loras)}/{len(lora_configs)} LoRAs on WAN model")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to blend WAN LoRAs: {e}")
            raise
    
    def get_wan_lora_status(self, model, lora_name: Optional[str] = None) -> Union[WANLoRAStatus, Dict[str, WANLoRAStatus]]:
        """Get LoRA status for WAN model"""
        try:
            model_id = self._get_model_id(model)
            
            if model_id not in self.wan_applied_loras:
                if lora_name:
                    return WANLoRAStatus(
                        lora_name=lora_name,
                        model_type=self._detect_wan_model_type(model),
                        is_compatible=False,
                        is_applied=False,
                        current_strength=0.0,
                        target_modules_affected=[],
                        memory_usage_mb=0.0,
                        application_method="none",
                        error_message="No LoRAs applied to this model"
                    )
                else:
                    return {}
            
            applied_loras = self.wan_applied_loras[model_id]
            
            if lora_name:
                return applied_loras.get(lora_name, WANLoRAStatus(
                    lora_name=lora_name,
                    model_type=self._detect_wan_model_type(model),
                    is_compatible=False,
                    is_applied=False,
                    current_strength=0.0,
                    target_modules_affected=[],
                    memory_usage_mb=0.0,
                    application_method="none",
                    error_message="LoRA not applied to this model"
                ))
            else:
                return applied_loras.copy()
                
        except Exception as e:
            logger.error(f"Error getting WAN LoRA status: {e}")
            if lora_name:
                return WANLoRAStatus(
                    lora_name=lora_name,
                    model_type=WANModelType.UNKNOWN,
                    is_compatible=False,
                    is_applied=False,
                    current_strength=0.0,
                    target_modules_affected=[],
                    memory_usage_mb=0.0,
                    application_method="error",
                    error_message=str(e)
                )
            else:
                return {}
    
    def _get_model_id(self, model) -> str:
        """Get unique identifier for model instance"""
        try:
            # Try to get model name or ID
            if hasattr(model, 'config') and hasattr(model.config, 'name_or_path'):
                return model.config.name_or_path
            elif hasattr(model, '_name_or_path'):
                return model._name_or_path
            else:
                # Fallback to object ID
                return f"model_{id(model)}"
        except Exception:
            return f"model_{id(model)}"
    
    def validate_wan_lora_loading(self, lora_name: str, model_type: str) -> Dict[str, Any]:
        """
        Validate LoRA loading for specific WAN model type
        
        Args:
            lora_name: Name of LoRA to validate
            model_type: WAN model type string
            
        Returns:
            Validation results dictionary
        """
        try:
            # Map model type string to enum
            wan_model_type = WANModelType.UNKNOWN
            for wmt in WANModelType:
                if wmt.value.lower() == model_type.lower():
                    wan_model_type = wmt
                    break
            
            if wan_model_type == WANModelType.UNKNOWN:
                return {
                    "valid": False,
                    "error": f"Unknown WAN model type: {model_type}",
                    "compatibility": None
                }
            
            # Check if LoRA exists and is loaded
            if lora_name not in self.loaded_loras:
                try:
                    self.load_lora(lora_name)
                except Exception as e:
                    return {
                        "valid": False,
                        "error": f"Failed to load LoRA {lora_name}: {str(e)}",
                        "compatibility": None
                    }
            
            # Get compatibility info
            compatibility = self.wan_model_compatibility.get(wan_model_type)
            if not compatibility:
                return {
                    "valid": False,
                    "error": f"No compatibility info for WAN model type: {wan_model_type.value}",
                    "compatibility": None
                }
            
            # Validate LoRA structure
            lora_info = self.loaded_loras[lora_name]
            lora_weights = lora_info["weights"]
            
            compatible_keys = self._validate_wan_lora_keys(lora_weights, compatibility.target_modules)
            
            return {
                "valid": compatible_keys,
                "error": None if compatible_keys else "LoRA keys do not match WAN model architecture",
                "compatibility": {
                    "model_type": wan_model_type.value,
                    "supports_lora": compatibility.supports_lora,
                    "max_lora_count": compatibility.max_lora_count,
                    "supported_types": compatibility.supported_lora_types,
                    "memory_overhead": compatibility.memory_overhead_factor,
                    "notes": compatibility.architecture_specific_notes
                },
                "lora_info": {
                    "size_mb": lora_info["size_mb"],
                    "num_layers": lora_info["num_layers"],
                    "loaded_at": lora_info["loaded_at"].isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error validating WAN LoRA loading: {e}")
            return {
                "valid": False,
                "error": str(e),
                "compatibility": None
            }


# Global WAN LoRA manager instance
_wan_lora_manager = None

def get_wan_lora_manager(config: Optional[Dict[str, Any]] = None) -> WANLoRAManager:
    """Get the global WAN LoRA manager instance"""
    global _wan_lora_manager
    if _wan_lora_manager is None:
        if config is None:
            # Try to get config from existing model manager
            try:
                from .model_manager import get_model_manager
                manager = get_model_manager()
                config = manager.config
            except ImportError:
                # Fallback config
                config = {
                    "directories": {
                        "loras_directory": "loras"
                    },
                    "lora_blending_enabled": True,
                    "max_blended_loras": 3
                }
        
        _wan_lora_manager = WANLoRAManager(config)
    return _wan_lora_manager