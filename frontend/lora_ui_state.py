"""
LoRA UI State Management
Handles LoRA selection, strength values, validation, and state persistence for the UI
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
try:
    from core.services.utils import LoRAManager
except ImportError:
    # For testing purposes, create a mock LoRAManager
    class LoRAManager:
        def __init__(self, config):
            self.config = config
        
        def list_available_loras(self):
            return {}

logger = logging.getLogger(__name__)

@dataclass
class LoRASelection:
    """Represents a single LoRA selection with metadata"""
    name: str
    strength: float
    selected_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "strength": self.strength,
            "selected_at": self.selected_at.isoformat(),
            "last_modified": self.last_modified.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoRASelection':
        """Create from dictionary"""
        return cls(
            name=data["name"],
            strength=data["strength"],
            selected_at=datetime.fromisoformat(data["selected_at"]),
            last_modified=datetime.fromisoformat(data["last_modified"])
        )

class LoRAUIState:
    """Manages LoRA UI state including selection, validation, and persistence"""
    
    # Constants for validation
    MAX_LORAS = 5
    MIN_STRENGTH = 0.0
    MAX_STRENGTH = 2.0
    STRENGTH_STEP = 0.1
    
    def __init__(self, config: Dict[str, Any], state_file: str = ".lora_ui_state.json"):
        self.config = config
        self.state_file = Path(state_file)
        
        # Initialize LoRA manager for validation
        self.lora_manager = LoRAManager(config)
        
        # State tracking
        self.selected_loras: Dict[str, LoRASelection] = {}
        self.upload_progress: Dict[str, float] = {}
        self.last_refresh: datetime = datetime.now()
        self.validation_errors: List[str] = []
        
        # Load persisted state
        self._load_state()
        
        logger.info(f"LoRAUIState initialized with {len(self.selected_loras)} selected LoRAs")
    
    def update_selection(self, lora_name: str, strength: float) -> Tuple[bool, str]:
        """
        Update LoRA selection with validation
        
        Args:
            lora_name: Name of the LoRA to select/update
            strength: Strength value (0.0-2.0)
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Validate strength value
            if not self._validate_strength(strength):
                return False, f"Invalid strength {strength}. Must be between {self.MIN_STRENGTH} and {self.MAX_STRENGTH}"
            
            # Check if LoRA exists
            available_loras = self.lora_manager.list_available_loras()
            if lora_name not in available_loras:
                return False, f"LoRA '{lora_name}' not found in available LoRAs"
            
            # Check maximum LoRA limit (only if adding new)
            if lora_name not in self.selected_loras and len(self.selected_loras) >= self.MAX_LORAS:
                return False, f"Maximum {self.MAX_LORAS} LoRAs can be selected simultaneously"
            
            # Update or add selection
            now = datetime.now()
            if lora_name in self.selected_loras:
                # Update existing selection
                self.selected_loras[lora_name].strength = strength
                self.selected_loras[lora_name].last_modified = now
                action = "updated"
            else:
                # Add new selection
                self.selected_loras[lora_name] = LoRASelection(
                    name=lora_name,
                    strength=strength,
                    selected_at=now,
                    last_modified=now
                )
                action = "selected"
            
            # Clear any previous validation errors for this LoRA
            self._clear_validation_error(lora_name)
            
            # Save state
            self._save_state()
            
            logger.info(f"LoRA {action}: {lora_name} with strength {strength}")
            return True, f"LoRA '{lora_name}' {action} with strength {strength}"
            
        except Exception as e:
            error_msg = f"Failed to update LoRA selection: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def remove_selection(self, lora_name: str) -> Tuple[bool, str]:
        """
        Remove a LoRA from selection
        
        Args:
            lora_name: Name of the LoRA to remove
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if lora_name not in self.selected_loras:
                return False, f"LoRA '{lora_name}' is not currently selected"
            
            # Remove selection
            del self.selected_loras[lora_name]
            
            # Clear validation errors for this LoRA
            self._clear_validation_error(lora_name)
            
            # Save state
            self._save_state()
            
            logger.info(f"LoRA removed from selection: {lora_name}")
            return True, f"LoRA '{lora_name}' removed from selection"
            
        except Exception as e:
            error_msg = f"Failed to remove LoRA selection: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def clear_selection(self) -> Tuple[bool, str]:
        """
        Clear all LoRA selections
        
        Returns:
            Tuple of (success, message)
        """
        try:
            count = len(self.selected_loras)
            self.selected_loras.clear()
            self.validation_errors.clear()
            
            # Save state
            self._save_state()
            
            logger.info(f"Cleared all LoRA selections ({count} LoRAs)")
            return True, f"Cleared {count} LoRA selections"
            
        except Exception as e:
            error_msg = f"Failed to clear LoRA selections: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def validate_selection(self) -> Tuple[bool, List[str]]:
        """
        Validate current LoRA selection
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Check LoRA count
            if len(self.selected_loras) > self.MAX_LORAS:
                errors.append(f"Too many LoRAs selected ({len(self.selected_loras)}/{self.MAX_LORAS})")
            
            # Get available LoRAs for validation
            available_loras = self.lora_manager.list_available_loras()
            
            # Validate each selected LoRA
            for lora_name, selection in self.selected_loras.items():
                # Check if LoRA still exists
                if lora_name not in available_loras:
                    errors.append(f"LoRA '{lora_name}' no longer available")
                    continue
                
                # Validate strength
                if not self._validate_strength(selection.strength):
                    errors.append(f"Invalid strength for '{lora_name}': {selection.strength}")
                
                # Check file integrity (optional, can be expensive)
                lora_info = available_loras[lora_name]
                if lora_info.get("size_mb", 0) == 0:
                    errors.append(f"LoRA '{lora_name}' appears to be corrupted (0 MB)")
            
            # Update validation errors
            self.validation_errors = errors
            
            is_valid = len(errors) == 0
            logger.debug(f"LoRA selection validation: {'valid' if is_valid else 'invalid'} ({len(errors)} errors)")
            
            return is_valid, errors
            
        except Exception as e:
            error_msg = f"Failed to validate LoRA selection: {str(e)}"
            logger.error(error_msg)
            return False, [error_msg]
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of current LoRA selection
        
        Returns:
            Dictionary with selection summary and metadata
        """
        try:
            # Validate current selection
            is_valid, errors = self.validate_selection()
            
            # Calculate total estimated memory impact
            total_memory_mb = 0.0
            available_loras = self.lora_manager.list_available_loras()
            
            for lora_name in self.selected_loras.keys():
                if lora_name in available_loras:
                    lora_size = available_loras[lora_name].get("size_mb", 0)
                    total_memory_mb += lora_size
            
            # Get selection details
            selections = []
            for lora_name, selection in self.selected_loras.items():
                lora_info = available_loras.get(lora_name, {})
                selections.append({
                    "name": lora_name,
                    "strength": selection.strength,
                    "size_mb": lora_info.get("size_mb", 0),
                    "selected_at": selection.selected_at.isoformat(),
                    "last_modified": selection.last_modified.isoformat(),
                    "exists": lora_name in available_loras
                })
            
            # Sort by selection time (most recent first)
            selections.sort(key=lambda x: x["last_modified"], reverse=True)
            
            summary = {
                "count": len(self.selected_loras),
                "max_count": self.MAX_LORAS,
                "is_valid": is_valid,
                "validation_errors": errors,
                "total_memory_mb": total_memory_mb,
                "selections": selections,
                "last_refresh": self.last_refresh.isoformat(),
                "has_selections": len(self.selected_loras) > 0
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get selection summary: {str(e)}")
            return {
                "count": 0,
                "max_count": self.MAX_LORAS,
                "is_valid": False,
                "validation_errors": [f"Error getting summary: {str(e)}"],
                "total_memory_mb": 0.0,
                "selections": [],
                "last_refresh": datetime.now().isoformat(),
                "has_selections": False
            }
    
    def get_display_data(self) -> Dict[str, Any]:
        """
        Get formatted data for UI display
        
        Returns:
            Dictionary optimized for UI rendering
        """
        try:
            summary = self.get_selection_summary()
            available_loras = self.lora_manager.list_available_loras()
            
            # Format for UI display
            display_data = {
                "selection_status": {
                    "count": summary["count"],
                    "max_count": summary["max_count"],
                    "remaining_slots": summary["max_count"] - summary["count"],
                    "is_full": summary["count"] >= summary["max_count"],
                    "is_valid": summary["is_valid"]
                },
                "memory_info": {
                    "total_mb": summary["total_memory_mb"],
                    "total_gb": summary["total_memory_mb"] / 1024,
                    "formatted": f"{summary['total_memory_mb']:.1f} MB"
                },
                "selected_loras": [],
                "available_loras": [],
                "validation_errors": summary["validation_errors"],
                "last_updated": summary["last_refresh"]
            }
            
            # Format selected LoRAs for display
            for selection_info in summary["selections"]:
                display_data["selected_loras"].append({
                    "name": selection_info["name"],
                    "strength": selection_info["strength"],
                    "strength_percent": int(selection_info["strength"] * 100),
                    "size_formatted": f"{selection_info['size_mb']:.1f} MB",
                    "selected_time": selection_info["selected_at"],
                    "is_valid": selection_info["exists"]
                })
            
            # Format available LoRAs (excluding selected ones)
            for lora_name, lora_info in available_loras.items():
                if lora_name not in self.selected_loras:
                    display_data["available_loras"].append({
                        "name": lora_name,
                        "filename": lora_info.get("filename", lora_name),
                        "size_formatted": f"{lora_info.get('size_mb', 0):.1f} MB",
                        "modified_time": lora_info.get("modified_time", ""),
                        "can_select": summary["count"] < summary["max_count"]
                    })
            
            # Sort available LoRAs by name
            display_data["available_loras"].sort(key=lambda x: x["name"].lower())
            
            return display_data
            
        except Exception as e:
            logger.error(f"Failed to get display data: {str(e)}")
            return {
                "selection_status": {"count": 0, "max_count": self.MAX_LORAS, "is_valid": False},
                "memory_info": {"total_mb": 0, "formatted": "0 MB"},
                "selected_loras": [],
                "available_loras": [],
                "validation_errors": [f"Error getting display data: {str(e)}"],
                "last_updated": datetime.now().isoformat()
            }
    
    def refresh_state(self) -> Tuple[bool, str]:
        """
        Refresh state by re-validating against current LoRA files
        
        Returns:
            Tuple of (success, message)
        """
        try:
            self.last_refresh = datetime.now()
            
            # Re-validate current selection
            is_valid, errors = self.validate_selection()
            
            # Remove invalid selections
            invalid_loras = []
            available_loras = self.lora_manager.list_available_loras()
            
            for lora_name in list(self.selected_loras.keys()):
                if lora_name not in available_loras:
                    invalid_loras.append(lora_name)
                    del self.selected_loras[lora_name]
            
            # Save updated state
            if invalid_loras:
                self._save_state()
            
            message_parts = [f"State refreshed at {self.last_refresh.strftime('%H:%M:%S')}"]
            if invalid_loras:
                message_parts.append(f"Removed {len(invalid_loras)} invalid LoRAs: {', '.join(invalid_loras)}")
            if errors:
                message_parts.append(f"{len(errors)} validation errors found")
            
            message = ". ".join(message_parts)
            logger.info(f"LoRA UI state refreshed: {message}")
            
            return True, message
            
        except Exception as e:
            error_msg = f"Failed to refresh state: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _validate_strength(self, strength: float) -> bool:
        """Validate LoRA strength value"""
        return (
            isinstance(strength, (int, float)) and
            self.MIN_STRENGTH <= strength <= self.MAX_STRENGTH
        )
    
    def _clear_validation_error(self, lora_name: str):
        """Clear validation errors related to a specific LoRA"""
        self.validation_errors = [
            error for error in self.validation_errors
            if lora_name not in error
        ]
    
    def _save_state(self):
        """Save current state to file"""
        try:
            state_data = {
                "selected_loras": {
                    name: selection.to_dict()
                    for name, selection in self.selected_loras.items()
                },
                "last_refresh": self.last_refresh.isoformat(),
                "validation_errors": self.validation_errors,
                "saved_at": datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.debug(f"LoRA UI state saved to {self.state_file}")
            
        except Exception as e:
            logger.error(f"Failed to save LoRA UI state: {str(e)}")
    
    def _load_state(self):
        """Load state from file"""
        try:
            if not self.state_file.exists():
                logger.info("No existing LoRA UI state file found, starting fresh")
                return
            
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
            
            # Load selected LoRAs
            if "selected_loras" in state_data:
                for name, selection_data in state_data["selected_loras"].items():
                    try:
                        self.selected_loras[name] = LoRASelection.from_dict(selection_data)
                    except Exception as e:
                        logger.warning(f"Failed to load LoRA selection {name}: {str(e)}")
            
            # Load other state data
            if "last_refresh" in state_data:
                try:
                    self.last_refresh = datetime.fromisoformat(state_data["last_refresh"])
                except Exception:
                    self.last_refresh = datetime.now()
            
            if "validation_errors" in state_data:
                self.validation_errors = state_data["validation_errors"]
            
            logger.info(f"Loaded LoRA UI state: {len(self.selected_loras)} selected LoRAs")
            
            # Validate loaded state
            self.validate_selection()
            
        except Exception as e:
            logger.error(f"Failed to load LoRA UI state: {str(e)}")
            # Reset to clean state on load failure
            self.selected_loras.clear()
            self.validation_errors.clear()
            self.last_refresh = datetime.now()

    def get_selection_for_generation(self) -> Dict[str, float]:
        """
        Get LoRA selection formatted for generation pipeline
        
        Returns:
            Dictionary mapping LoRA names to strength values
        """
        try:
            # Validate before returning
            is_valid, errors = self.validate_selection()
            
            if not is_valid:
                logger.warning(f"Invalid LoRA selection for generation: {errors}")
                # Return empty selection if invalid
                return {}
            
            # Return name -> strength mapping
            return {
                name: selection.strength
                for name, selection in self.selected_loras.items()
            }
            
        except Exception as e:
            logger.error(f"Failed to get selection for generation: {str(e)}")
            return {}
    
    def estimate_memory_impact(self) -> Dict[str, float]:
        """
        Estimate memory impact of current LoRA selection
        
        Returns:
            Dictionary with memory estimates
        """
        try:
            available_loras = self.lora_manager.list_available_loras()
            
            total_size_mb = 0.0
            individual_sizes = {}
            
            for lora_name, selection in self.selected_loras.items():
                if lora_name in available_loras:
                    size_mb = available_loras[lora_name].get("size_mb", 0)
                    # Estimate actual memory usage (LoRAs are typically loaded in full)
                    estimated_memory = size_mb * 1.2  # Add 20% overhead
                    individual_sizes[lora_name] = estimated_memory
                    total_size_mb += estimated_memory
            
            return {
                "total_mb": total_size_mb,
                "total_gb": total_size_mb / 1024,
                "individual_mb": individual_sizes,
                "overhead_factor": 1.2,
                "estimated_load_time_seconds": total_size_mb / 100  # Rough estimate: 100MB/sec
            }
            
        except Exception as e:
            logger.error(f"Failed to estimate memory impact: {str(e)}")
            return {
                "total_mb": 0.0,
                "total_gb": 0.0,
                "individual_mb": {},
                "overhead_factor": 1.0,
                "estimated_load_time_seconds": 0.0
            }