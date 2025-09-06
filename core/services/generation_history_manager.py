"""
Generation History Manager for Wan2.2 Video Generation

This module provides generation history tracking, retry capabilities, and
user experience enhancements for the video generation system.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

class GenerationStatus(Enum):
    """Status of generation requests"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

@dataclass
class GenerationHistoryEntry:
    """Represents a single generation history entry"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    model_type: str = ""
    prompt: str = ""
    image_path: Optional[str] = None
    resolution: str = "1280x720"
    steps: int = 50
    lora_config: Dict[str, float] = field(default_factory=dict)
    status: GenerationStatus = GenerationStatus.PENDING
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    generation_time: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    user_rating: Optional[int] = None  # 1-5 stars
    user_notes: str = ""
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationHistoryEntry':
        """Create from dictionary"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['status'] = GenerationStatus(data['status'])
        return cls(**data)
    
    def can_retry(self) -> bool:
        """Check if this entry can be retried"""
        return (self.status == GenerationStatus.FAILED and 
                self.retry_count < self.max_retries)
    
    def get_display_name(self) -> str:
        """Get display name for UI"""
        prompt_preview = self.prompt[:50] + "..." if len(self.prompt) > 50 else self.prompt
        return f"{self.model_type.upper()} - {prompt_preview}"
    
    def get_status_emoji(self) -> str:
        """Get emoji for status"""
        status_emojis = {
            GenerationStatus.PENDING: "⏳",
            GenerationStatus.IN_PROGRESS: "🔄",
            GenerationStatus.COMPLETED: "✅",
            GenerationStatus.FAILED: "❌",
            GenerationStatus.CANCELLED: "🚫",
            GenerationStatus.RETRYING: "🔁"
        }
        return status_emojis.get(self.status, "❓")

class GenerationHistoryManager:
    """Manages generation history and retry capabilities"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.history_file = Path(self.config.get("history_file", "generation_history.json"))
        self.max_history_entries = self.config.get("max_history_entries", 1000)
        self.auto_cleanup_days = self.config.get("auto_cleanup_days", 30)
        
        # In-memory history cache
        self.history: List[GenerationHistoryEntry] = []
        self.history_lock = threading.RLock()
        
        # Load existing history
        self._load_history()
        
        # Auto-cleanup old entries
        self._cleanup_old_entries()
    
    def _load_history(self):
        """Load history from file"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                with self.history_lock:
                    self.history = [
                        GenerationHistoryEntry.from_dict(entry) 
                        for entry in data.get('entries', [])
                    ]
                
                logger.info(f"Loaded {len(self.history)} history entries")
            else:
                logger.info("No existing history file found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            self.history = []
    
    def _save_history(self):
        """Save history to file"""
        try:
            # Ensure directory exists
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            
            with self.history_lock:
                data = {
                    'version': '1.0',
                    'last_updated': datetime.now().isoformat(),
                    'entries': [entry.to_dict() for entry in self.history]
                }
            
            # Write to temporary file first, then rename for atomicity
            temp_file = self.history_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            temp_file.replace(self.history_file)
            logger.debug("History saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def _cleanup_old_entries(self):
        """Remove old entries based on age and count limits"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.auto_cleanup_days)
            
            with self.history_lock:
                # Remove entries older than cutoff date
                self.history = [
                    entry for entry in self.history 
                    if entry.timestamp > cutoff_date
                ]
                
                # Keep only the most recent entries if we exceed max count
                if len(self.history) > self.max_history_entries:
                    self.history.sort(key=lambda x: x.timestamp, reverse=True)
                    self.history = self.history[:self.max_history_entries]
                
                logger.info(f"Cleaned up history, {len(self.history)} entries remaining")
            
            self._save_history()
            
        except Exception as e:
            logger.error(f"Failed to cleanup history: {e}")
    
    def add_entry(self, model_type: str, prompt: str, image_path: Optional[str] = None,
                  resolution: str = "1280x720", steps: int = 50, 
                  lora_config: Optional[Dict[str, float]] = None,
                  hardware_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Add new generation entry to history
        Returns: entry ID
        """
        try:
            entry = GenerationHistoryEntry(
                model_type=model_type,
                prompt=prompt,
                image_path=image_path,
                resolution=resolution,
                steps=steps,
                lora_config=lora_config or {},
                hardware_info=hardware_info or {},
                status=GenerationStatus.PENDING
            )
            
            with self.history_lock:
                self.history.insert(0, entry)  # Add to beginning for recent-first order
            
            self._save_history()
            logger.info(f"Added new history entry: {entry.id}")
            return entry.id
            
        except Exception as e:
            logger.error(f"Failed to add history entry: {e}")
            return ""
    
    def update_entry_status(self, entry_id: str, status: GenerationStatus,
                           output_path: Optional[str] = None,
                           error_message: Optional[str] = None,
                           generation_time: Optional[float] = None):
        """Update entry status and related information"""
        try:
            with self.history_lock:
                for entry in self.history:
                    if entry.id == entry_id:
                        entry.status = status
                        if output_path:
                            entry.output_path = output_path
                        if error_message:
                            entry.error_message = error_message
                        if generation_time:
                            entry.generation_time = generation_time
                        break
                else:
                    logger.warning(f"Entry not found for update: {entry_id}")
                    return
            
            self._save_history()
            logger.debug(f"Updated entry {entry_id} status to {status.value}")
            
        except Exception as e:
            logger.error(f"Failed to update entry status: {e}")
    
    def get_entry(self, entry_id: str) -> Optional[GenerationHistoryEntry]:
        """Get specific history entry by ID"""
        with self.history_lock:
            for entry in self.history:
                if entry.id == entry_id:
                    return entry
        return None
    
    def get_recent_entries(self, limit: int = 20) -> List[GenerationHistoryEntry]:
        """Get recent history entries"""
        with self.history_lock:
            return self.history[:limit]
    
    def get_entries_by_status(self, status: GenerationStatus) -> List[GenerationHistoryEntry]:
        """Get entries filtered by status"""
        with self.history_lock:
            return [entry for entry in self.history if entry.status == status]
    
    def get_failed_entries(self) -> List[GenerationHistoryEntry]:
        """Get failed entries that can be retried"""
        with self.history_lock:
            return [entry for entry in self.history if entry.can_retry()]
    
    def get_successful_entries(self, limit: int = 50) -> List[GenerationHistoryEntry]:
        """Get successful entries for analysis"""
        with self.history_lock:
            successful = [
                entry for entry in self.history 
                if entry.status == GenerationStatus.COMPLETED
            ]
            return successful[:limit]
    
    def retry_entry(self, entry_id: str) -> Optional[str]:
        """
        Create retry entry from failed generation
        Returns: new entry ID if successful
        """
        try:
            original_entry = self.get_entry(entry_id)
            if not original_entry or not original_entry.can_retry():
                logger.warning(f"Cannot retry entry {entry_id}")
                return None
            
            # Create new entry with same parameters
            new_entry_id = self.add_entry(
                model_type=original_entry.model_type,
                prompt=original_entry.prompt,
                image_path=original_entry.image_path,
                resolution=original_entry.resolution,
                steps=original_entry.steps,
                lora_config=original_entry.lora_config,
                hardware_info=original_entry.hardware_info
            )
            
            # Update original entry retry count
            with self.history_lock:
                original_entry.retry_count += 1
                original_entry.status = GenerationStatus.RETRYING
            
            self._save_history()
            logger.info(f"Created retry entry {new_entry_id} from {entry_id}")
            return new_entry_id
            
        except Exception as e:
            logger.error(f"Failed to retry entry: {e}")
            return None
    
    def add_user_rating(self, entry_id: str, rating: int, notes: str = ""):
        """Add user rating and notes to entry"""
        try:
            if not (1 <= rating <= 5):
                raise ValueError("Rating must be between 1 and 5")
            
            with self.history_lock:
                for entry in self.history:
                    if entry.id == entry_id:
                        entry.user_rating = rating
                        entry.user_notes = notes
                        break
                else:
                    logger.warning(f"Entry not found for rating: {entry_id}")
                    return
            
            self._save_history()
            logger.info(f"Added rating {rating} to entry {entry_id}")
            
        except Exception as e:
            logger.error(f"Failed to add rating: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        try:
            with self.history_lock:
                total_entries = len(self.history)
                if total_entries == 0:
                    return {"total": 0}
                
                status_counts = defaultdict(int)
                model_counts = defaultdict(int)
                resolution_counts = defaultdict(int)
                total_time = 0
                successful_count = 0
                rated_entries = []
                
                for entry in self.history:
                    status_counts[entry.status.value] += 1
                    model_counts[entry.model_type] += 1
                    resolution_counts[entry.resolution] += 1
                    
                    if entry.generation_time:
                        total_time += entry.generation_time
                        successful_count += 1
                    
                    if entry.user_rating:
                        rated_entries.append(entry.user_rating)
                
                avg_time = total_time / successful_count if successful_count > 0 else 0
                avg_rating = sum(rated_entries) / len(rated_entries) if rated_entries else 0
                
                return {
                    "total": total_entries,
                    "by_status": dict(status_counts),
                    "by_model": dict(model_counts),
                    "by_resolution": dict(resolution_counts),
                    "success_rate": status_counts["completed"] / total_entries * 100,
                    "average_generation_time": avg_time,
                    "average_rating": avg_rating,
                    "total_rated": len(rated_entries)
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}
    
    def search_entries(self, query: str, limit: int = 20) -> List[GenerationHistoryEntry]:
        """Search entries by prompt text"""
        try:
            query_lower = query.lower()
            with self.history_lock:
                matches = [
                    entry for entry in self.history
                    if query_lower in entry.prompt.lower() or 
                       query_lower in entry.model_type.lower()
                ]
                return matches[:limit]
                
        except Exception as e:
            logger.error(f"Failed to search entries: {e}")
            return []
    
    def export_history(self, output_path: str, include_images: bool = False) -> bool:
        """Export history to JSON file"""
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_entries": len(self.history),
                "statistics": self.get_statistics(),
                "entries": []
            }
            
            with self.history_lock:
                for entry in self.history:
                    entry_data = entry.to_dict()
                    if not include_images:
                        entry_data.pop("image_path", None)
                    export_data["entries"].append(entry_data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported history to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export history: {e}")
            return False
    
    def clear_history(self, keep_successful: bool = False):
        """Clear history with option to keep successful entries"""
        try:
            with self.history_lock:
                if keep_successful:
                    self.history = [
                        entry for entry in self.history
                        if entry.status == GenerationStatus.COMPLETED
                    ]
                else:
                    self.history = []
            
            self._save_history()
            logger.info(f"Cleared history, {len(self.history)} entries remaining")
            
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")

# Global history manager instance
_history_manager = None

def get_history_manager(config: Optional[Dict[str, Any]] = None) -> GenerationHistoryManager:
    """Get or create global history manager instance"""
    global _history_manager
    if _history_manager is None:
        _history_manager = GenerationHistoryManager(config)
    return _history_manager