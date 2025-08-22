"""
Image Session State Manager for Wan2.2 UI
Handles persistent storage and retrieval of uploaded images during UI sessions
"""

import os
import json
import tempfile
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from PIL import Image
import hashlib
import shutil

logger = logging.getLogger(__name__)

class ImageSessionManager:
    """Manages session state for uploaded images with persistence and cleanup"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session_dir = Path(tempfile.gettempdir()) / "wan22_sessions"
        self.session_id = self._generate_session_id()
        self.current_session_dir = self.session_dir / self.session_id
        
        # Session state storage
        self.session_state = {
            "start_image": None,
            "end_image": None,
            "start_image_metadata": None,
            "end_image_metadata": None,
            "start_image_path": None,
            "end_image_path": None,
            "last_updated": datetime.now().isoformat(),
            "session_created": datetime.now().isoformat()
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Cleanup settings
        self.max_session_age_hours = config.get("session", {}).get("max_age_hours", 24)
        self.cleanup_interval_minutes = config.get("session", {}).get("cleanup_interval_minutes", 30)
        self.enable_cleanup_thread = config.get("session", {}).get("enable_cleanup_thread", True)
        
        # Initialize session
        self._initialize_session()
        
        # Start cleanup thread (unless disabled for testing)
        if self.enable_cleanup_thread:
            self._start_cleanup_thread()
        
        logger.info(f"Image session manager initialized with session ID: {self.session_id}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = str(int(time.time()))
        random_part = os.urandom(8).hex()
        return f"wan22_session_{timestamp}_{random_part}"
    
    def _initialize_session(self):
        """Initialize session directory and state file"""
        try:
            # Create session directory
            self.current_session_dir.mkdir(parents=True, exist_ok=True)
            
            # Create session state file
            self._save_session_state()
            
            logger.debug(f"Session initialized at: {self.current_session_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize session: {e}")
            raise
    
    def _save_session_state(self):
        """Save current session state to file"""
        try:
            # Create a JSON-serializable copy of session state
            serializable_state = {}
            for key, value in self.session_state.items():
                if key.endswith('_image'):
                    # Don't serialize PIL Image objects
                    serializable_state[key] = None
                else:
                    serializable_state[key] = value
            
            state_file = self.current_session_dir / "session_state.json"
            with open(state_file, 'w') as f:
                json.dump(serializable_state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session state: {e}")
    
    def _load_session_state(self) -> bool:
        """Load session state from file"""
        try:
            state_file = self.current_session_dir / "session_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    loaded_state = json.load(f)
                
                # Merge loaded state with current state, ensuring image objects are None
                for key, value in loaded_state.items():
                    if key.endswith('_image'):
                        self.session_state[key] = None  # Images will be loaded on demand
                    else:
                        self.session_state[key] = value
                
                return True
        except Exception as e:
            logger.error(f"Failed to load session state: {e}")
        return False
    
    def store_image(self, image: Image.Image, image_type: str) -> Tuple[bool, str]:
        """
        Store image in session with metadata
        
        Args:
            image: PIL Image object
            image_type: "start" or "end"
            
        Returns:
            Tuple of (success, message)
        """
        if image_type not in ["start", "end"]:
            return False, "Invalid image type. Must be 'start' or 'end'"
        
        with self._lock:
            try:
                # Ensure session directory exists
                self.current_session_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate image metadata
                metadata = self._generate_image_metadata(image, image_type)
                
                # Save image to session directory
                image_filename = f"{image_type}_image_{int(time.time())}.png"
                image_path = self.current_session_dir / image_filename
                
                # Save with optimization
                image.save(image_path, "PNG", optimize=True)
                
                # Verify saved file
                if not image_path.exists() or image_path.stat().st_size == 0:
                    return False, "Failed to save image file"
                
                # Update session state
                self.session_state[f"{image_type}_image"] = image
                self.session_state[f"{image_type}_image_metadata"] = metadata
                self.session_state[f"{image_type}_image_path"] = str(image_path)
                self.session_state["last_updated"] = datetime.now().isoformat()
                
                # Save session state
                self._save_session_state()
                
                logger.debug(f"Stored {image_type} image: {image_path}")
                return True, f"Successfully stored {image_type} image"
                
            except Exception as e:
                logger.error(f"Failed to store {image_type} image: {e}")
                return False, f"Failed to store image: {str(e)}"
    
    def retrieve_image(self, image_type: str) -> Tuple[Optional[Image.Image], Optional[Dict[str, Any]]]:
        """
        Retrieve image and metadata from session
        
        Args:
            image_type: "start" or "end"
            
        Returns:
            Tuple of (image, metadata) or (None, None) if not found
        """
        if image_type not in ["start", "end"]:
            return None, None
        
        with self._lock:
            try:
                # Check if image exists in memory
                image = self.session_state.get(f"{image_type}_image")
                metadata = self.session_state.get(f"{image_type}_image_metadata")
                
                if image is not None:
                    return image, metadata
                
                # Try to load from file
                image_path = self.session_state.get(f"{image_type}_image_path")
                if image_path and os.path.exists(image_path):
                    try:
                        image = Image.open(image_path)
                        image.load()  # Ensure image is fully loaded
                        
                        # Store back in memory for faster access
                        self.session_state[f"{image_type}_image"] = image
                        
                        return image, metadata
                    except Exception as e:
                        logger.error(f"Failed to load {image_type} image from {image_path}: {e}")
                        # Clean up invalid path
                        self.session_state[f"{image_type}_image_path"] = None
                        self._save_session_state()
                
                return None, None
                
            except Exception as e:
                logger.error(f"Failed to retrieve {image_type} image: {e}")
                return None, None
    
    def clear_image(self, image_type: str) -> bool:
        """
        Clear image from session
        
        Args:
            image_type: "start" or "end"
            
        Returns:
            Success status
        """
        if image_type not in ["start", "end"]:
            return False
        
        with self._lock:
            try:
                # Remove from memory
                self.session_state[f"{image_type}_image"] = None
                self.session_state[f"{image_type}_image_metadata"] = None
                
                # Remove file if exists
                image_path = self.session_state.get(f"{image_type}_image_path")
                if image_path and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                        logger.debug(f"Removed {image_type} image file: {image_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {image_type} image file: {e}")
                
                self.session_state[f"{image_type}_image_path"] = None
                self.session_state["last_updated"] = datetime.now().isoformat()
                
                # Save session state
                self._save_session_state()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to clear {image_type} image: {e}")
                return False
    
    def clear_all_images(self) -> bool:
        """Clear all images from session"""
        try:
            success_start = self.clear_image("start")
            success_end = self.clear_image("end")
            return success_start and success_end
        except Exception as e:
            logger.error(f"Failed to clear all images: {e}")
            return False
    
    def _has_image_unlocked(self, image_type: str) -> bool:
        """Check if image exists in session (internal method, assumes lock is held)"""
        if image_type not in ["start", "end"]:
            return False
        
        # Check memory first
        if self.session_state.get(f"{image_type}_image") is not None:
            return True
        
        # Check file existence
        image_path = self.session_state.get(f"{image_type}_image_path")
        return image_path is not None and os.path.exists(image_path)
    
    def has_image(self, image_type: str) -> bool:
        """Check if image exists in session"""
        if image_type not in ["start", "end"]:
            return False
        
        with self._lock:
            return self._has_image_unlocked(image_type)
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information"""
        with self._lock:
            return {
                "session_id": self.session_id,
                "session_dir": str(self.current_session_dir),
                "has_start_image": self._has_image_unlocked("start"),
                "has_end_image": self._has_image_unlocked("end"),
                "last_updated": self.session_state.get("last_updated"),
                "session_created": self.session_state.get("session_created"),
                "start_image_metadata": self.session_state.get("start_image_metadata"),
                "end_image_metadata": self.session_state.get("end_image_metadata")
            }
    
    def restore_from_session_id(self, session_id: str) -> bool:
        """
        Restore session from existing session ID
        
        Args:
            session_id: Existing session ID to restore
            
        Returns:
            Success status
        """
        try:
            old_session_dir = self.session_dir / session_id
            if not old_session_dir.exists():
                logger.warning(f"Session directory not found: {old_session_dir}")
                return False
            
            # Load session state
            state_file = old_session_dir / "session_state.json"
            if not state_file.exists():
                logger.warning(f"Session state file not found: {state_file}")
                return False
            
            with open(state_file, 'r') as f:
                restored_state = json.load(f)
            
            # Update current session
            with self._lock:
                self.session_id = session_id
                self.current_session_dir = old_session_dir
                
                # Merge restored state with current state
                for key, value in restored_state.items():
                    if key.endswith('_image'):
                        self.session_state[key] = None  # Images will be loaded on demand
                    else:
                        self.session_state[key] = value
            
            logger.info(f"Restored session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore session {session_id}: {e}")
            return False
    
    def _generate_image_metadata(self, image: Image.Image, image_type: str) -> Dict[str, Any]:
        """Generate comprehensive metadata for image"""
        try:
            # Calculate image hash for integrity checking
            image_bytes = image.tobytes()
            image_hash = hashlib.md5(image_bytes).hexdigest()
            
            return {
                "type": image_type,
                "format": image.format or "PNG",
                "size": image.size,
                "mode": image.mode,
                "has_transparency": image.mode in ("RGBA", "LA") or "transparency" in image.info,
                "aspect_ratio": image.size[0] / image.size[1] if image.size[1] > 0 else 1.0,
                "pixel_count": image.size[0] * image.size[1],
                "stored_at": datetime.now().isoformat(),
                "image_hash": image_hash,
                "file_size_estimate": len(image_bytes)
            }
        except Exception as e:
            logger.error(f"Failed to generate metadata for {image_type} image: {e}")
            return {
                "type": image_type,
                "stored_at": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_worker():
            while not self._stop_cleanup.is_set():
                try:
                    self._cleanup_old_sessions()
                    # Use wait instead of sleep to allow interruption
                    if self._stop_cleanup.wait(self.cleanup_interval_minutes * 60):
                        break
                except Exception as e:
                    logger.error(f"Session cleanup error: {e}")
                    # Wait 5 minutes on error, but allow interruption
                    if self._stop_cleanup.wait(300):
                        break
        
        self._stop_cleanup = threading.Event()
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        logger.debug("Session cleanup thread started")
    
    def _cleanup_old_sessions(self):
        """Clean up old session directories"""
        try:
            if not self.session_dir.exists():
                return
            
            cutoff_time = datetime.now() - timedelta(hours=self.max_session_age_hours)
            cleaned_count = 0
            
            for session_path in self.session_dir.iterdir():
                if not session_path.is_dir():
                    continue
                
                # Skip current session
                if session_path.name == self.session_id:
                    continue
                
                try:
                    # Check session age
                    state_file = session_path / "session_state.json"
                    if state_file.exists():
                        with open(state_file, 'r') as f:
                            state = json.load(f)
                        
                        created_time = datetime.fromisoformat(state.get("session_created", "1970-01-01T00:00:00"))
                        if created_time < cutoff_time:
                            # Remove old session
                            shutil.rmtree(session_path)
                            cleaned_count += 1
                            logger.debug(f"Cleaned up old session: {session_path.name}")
                    else:
                        # Remove sessions without state files
                        shutil.rmtree(session_path)
                        cleaned_count += 1
                        logger.debug(f"Cleaned up invalid session: {session_path.name}")
                        
                except Exception as e:
                    logger.warning(f"Failed to clean up session {session_path.name}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old sessions")
                
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
    
    def cleanup_current_session(self):
        """Clean up current session (call on UI shutdown)"""
        try:
            # Stop cleanup thread first
            if hasattr(self, '_stop_cleanup'):
                self._stop_cleanup.set()
            if hasattr(self, '_cleanup_thread') and self._cleanup_thread.is_alive():
                self._cleanup_thread.join(timeout=1.0)
            
            with self._lock:
                if self.current_session_dir.exists():
                    shutil.rmtree(self.current_session_dir)
                    logger.info(f"Cleaned up current session: {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup current session: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.cleanup_current_session()
        except Exception:
            pass


# Global session manager instance
_session_manager = None

def get_image_session_manager(config: Dict[str, Any]) -> ImageSessionManager:
    """Get or create global image session manager"""
    global _session_manager
    if _session_manager is None:
        _session_manager = ImageSessionManager(config)
    return _session_manager

def cleanup_session_manager():
    """Cleanup global session manager"""
    global _session_manager
    if _session_manager is not None:
        _session_manager.cleanup_current_session()
        _session_manager = None