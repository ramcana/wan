"""
Installation Flow Controller
Manages installation state, progress tracking, logging, and rollback capabilities.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable

from interfaces import InstallationPhase, InstallationState, InstallationError, ErrorCategory
from rollback_manager import RollbackManager
from logging_system import setup_installation_logging, LogCategory


class InstallationFlowController:
    """
    Controls the installation flow with state management, progress tracking,
    logging, and rollback capabilities.
    """
    
    def __init__(self, installation_path: str, dry_run: bool = False, log_level: str = "INFO"):
        self.installation_path = Path(installation_path)
        self.dry_run = dry_run
        self.logs_dir = self.installation_path / "logs"
        self.backup_dir = self.installation_path / ".wan22_backup"
        
        # State management
        self.state_file = self.logs_dir / "installation_state.json"
        self.current_state = None
        
        # Initialize enhanced logging system
        self.logging_system = setup_installation_logging(
            installation_path=str(installation_path),
            log_level=log_level,
            enable_console=not dry_run,  # Disable console in dry run mode
            enable_structured=True
        )
        self.logger = self.logging_system.get_logger(__name__)
        
        # Initialize rollback manager
        self.rollback_manager = RollbackManager(installation_path, dry_run)
        
        # Progress tracking
        self.progress_callbacks: List[Callable] = []
        self.phase_weights = {
            InstallationPhase.DETECTION: 0.1,
            InstallationPhase.DEPENDENCIES: 0.3,
            InstallationPhase.MODELS: 0.4,
            InstallationPhase.CONFIGURATION: 0.1,
            InstallationPhase.VALIDATION: 0.1,
            InstallationPhase.COMPLETE: 0.0
        }
        
        # Initialize directories
        self._initialize_directories()
        
        self.logger.info(f"Installation flow controller initialized (dry_run={dry_run})")
        self.logging_system.log_user_action("flow_controller_initialized", {
            "installation_path": str(installation_path),
            "dry_run": dry_run,
            "log_level": log_level
        })
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging system."""
        # Ensure logs directory exists
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger('InstallationFlow')
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(
            self.logs_dir / "installation_flow.log", 
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # Error log handler
        error_handler = logging.FileHandler(
            self.logs_dir / "installation_errors.log",
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
        
        return logger
    
    def _initialize_directories(self) -> None:
        """Initialize required directories."""
        directories = [
            self.logs_dir,
            self.backup_dir,
            self.backup_dir / "snapshots"
        ]
        
        for directory in directories:
            if not self.dry_run:
                directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"{'Would create' if self.dry_run else 'Created'} directory: {directory}")
    
    def add_progress_callback(self, callback: Callable[[InstallationPhase, float, str], None]) -> None:
        """Add a progress callback function."""
        self.progress_callbacks.append(callback)
    
    def update_progress(self, phase: InstallationPhase, phase_progress: float, task: str) -> None:
        """Update installation progress and notify callbacks."""
        # Calculate overall progress
        phase_order = [
            InstallationPhase.DETECTION,
            InstallationPhase.DEPENDENCIES,
            InstallationPhase.MODELS,
            InstallationPhase.CONFIGURATION,
            InstallationPhase.VALIDATION,
            InstallationPhase.COMPLETE
        ]
        
        try:
            current_index = phase_order.index(phase)
            completed_weight = sum(
                self.phase_weights[p] for p in phase_order[:current_index]
            )
        except ValueError:
            # If phase not in order, default to 0
            completed_weight = 0
        current_weight = self.phase_weights[phase] * phase_progress
        overall_progress = completed_weight + current_weight
        
        # Update state
        if self.current_state:
            self.current_state.phase = phase
            self.current_state.progress = overall_progress
            self.current_state.current_task = task
            self.save_state()
        
        # Log progress with structured logging
        self.logging_system.log_progress(phase.value, overall_progress, task)
        
        # Notify callbacks
        for callback in self.progress_callbacks:
            try:
                callback(phase, overall_progress, task)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")
    
    def initialize_state(self, installation_path: str) -> InstallationState:
        """Initialize a new installation state."""
        self.current_state = InstallationState(
            phase=InstallationPhase.DETECTION,
            progress=0.0,
            current_task="Initializing installation",
            errors=[],
            warnings=[],
            hardware_profile=None,
            installation_path=installation_path
        )
        
        self.save_state()
        self.logger.info("Installation state initialized")
        return self.current_state
    
    def load_state(self) -> Optional[InstallationState]:
        """Load existing installation state."""
        try:
            if not self.state_file.exists():
                return None
            
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            self.current_state = InstallationState(
                phase=InstallationPhase(state_data['phase']),
                progress=state_data['progress'],
                current_task=state_data['current_task'],
                errors=state_data['errors'],
                warnings=state_data['warnings'],
                hardware_profile=state_data.get('hardware_profile'),
                installation_path=state_data['installation_path']
            )
            
            self.logger.info(f"Loaded installation state: {self.current_state.phase.value}")
            return self.current_state
            
        except Exception as e:
            self.logger.error(f"Failed to load installation state: {e}")
            return None
    
    def save_state(self) -> None:
        """Save current installation state."""
        if not self.current_state or self.dry_run:
            return
        
        try:
            state_data = {
                'phase': self.current_state.phase.value,
                'progress': self.current_state.progress,
                'current_task': self.current_state.current_task,
                'errors': self.current_state.errors,
                'warnings': self.current_state.warnings,
                'hardware_profile': self.current_state.hardware_profile,
                'installation_path': self.current_state.installation_path,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug("Installation state saved")
            
        except Exception as e:
            self.logger.error(f"Failed to save installation state: {e}")
    
    def clear_state(self) -> None:
        """Clear installation state."""
        try:
            if self.state_file.exists() and not self.dry_run:
                self.state_file.unlink()
            self.current_state = None
            self.logger.info("Installation state cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear installation state: {e}")
    
    def add_error(self, error: str, context: Dict[str, Any] = None) -> None:
        """Add an error to the current state."""
        if self.current_state:
            self.current_state.errors.append(error)
            self.save_state()
        
        self.logging_system.log_structured(
            'ERROR',
            error,
            category=LogCategory.INSTALLATION,
            phase=self.current_state.phase.value if self.current_state else "unknown",
            context=context or {}
        )
    
    def add_warning(self, warning: str, context: Dict[str, Any] = None) -> None:
        """Add a warning to the current state."""
        if self.current_state:
            self.current_state.warnings.append(warning)
            self.save_state()
        
        self.logging_system.log_structured(
            'WARNING',
            warning,
            category=LogCategory.INSTALLATION,
            phase=self.current_state.phase.value if self.current_state else "unknown",
            context=context or {}
        )
    
    def create_snapshot(self, description: str, files_to_backup: List[str] = None,
                       dirs_to_backup: List[str] = None) -> str:
        """Create a snapshot for rollback purposes."""
        phase = self.current_state.phase.value if self.current_state else "unknown"
        
        try:
            snapshot_id = self.rollback_manager.create_snapshot(
                description=description,
                phase=phase,
                files_to_backup=files_to_backup,
                dirs_to_backup=dirs_to_backup
            )
            
            self.logging_system.log_structured(
                'INFO',
                f"Created snapshot: {snapshot_id} - {description}",
                category=LogCategory.SYSTEM,
                phase=phase,
                snapshot_id=snapshot_id
            )
            
            return snapshot_id
            
        except Exception as e:
            self.logger.error(f"Failed to create snapshot: {e}")
            raise InstallationError(
                f"Failed to create backup snapshot: {str(e)}",
                ErrorCategory.SYSTEM,
                ["Check disk space", "Verify file permissions"]
            )
    
    def restore_snapshot(self, snapshot_id: str) -> bool:
        """Restore from a snapshot."""
        try:
            success = self.rollback_manager.restore_snapshot(snapshot_id)
            
            if success:
                self.logging_system.log_structured(
                    'INFO',
                    f"Successfully restored snapshot: {snapshot_id}",
                    category=LogCategory.SYSTEM,
                    snapshot_id=snapshot_id
                )
                self.logging_system.log_user_action("snapshot_restored", {
                    "snapshot_id": snapshot_id
                })
            else:
                self.logging_system.log_structured(
                    'ERROR',
                    f"Failed to restore snapshot: {snapshot_id}",
                    category=LogCategory.SYSTEM,
                    snapshot_id=snapshot_id
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to restore snapshot {snapshot_id}: {e}")
            return False
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List available snapshots."""
        try:
            snapshots = self.rollback_manager.list_snapshots()
            return [
                {
                    'id': s.id,
                    'timestamp': s.timestamp,
                    'description': s.description,
                    'phase': s.phase
                }
                for s in snapshots
            ]
        except Exception as e:
            self.logger.error(f"Failed to list snapshots: {e}")
            return []
    
    def cleanup_old_snapshots(self, keep_count: int = 5, keep_days: int = 30) -> int:
        """Clean up old snapshots, keeping only the most recent ones."""
        try:
            deleted_count = self.rollback_manager.cleanup_old_snapshots(
                keep_count=keep_count,
                keep_days=keep_days
            )
            
            if deleted_count > 0:
                self.logging_system.log_structured(
                    'INFO',
                    f"Cleaned up {deleted_count} old snapshots",
                    category=LogCategory.SYSTEM,
                    deleted_count=deleted_count
                )
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old snapshots: {e}")
            return 0
    

    
    def get_installation_summary(self) -> Dict[str, Any]:
        """Get a summary of the installation status."""
        summary = {
            'installation_path': str(self.installation_path),
            'dry_run': self.dry_run,
            'current_state': None,
            'snapshots_count': len(self.list_snapshots()),
            'logging_summary': self.logging_system.create_log_summary()
        }
        
        if self.current_state:
            summary['current_state'] = {
                'phase': self.current_state.phase.value,
                'progress': self.current_state.progress,
                'current_task': self.current_state.current_task,
                'errors_count': len(self.current_state.errors),
                'warnings_count': len(self.current_state.warnings)
            }
        
        return summary
    
    def validate_installation_integrity(self) -> Dict[str, Any]:
        """Validate the integrity of the installation."""
        validation_result = {
            'valid': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check required directories
        required_dirs = ['scripts', 'logs', 'models', 'application']
        for dir_name in required_dirs:
            dir_path = self.installation_path / dir_name
            if not dir_path.exists():
                validation_result['valid'] = False
                validation_result['issues'].append(f"Missing required directory: {dir_name}")
        
        # Check configuration file
        config_file = self.installation_path / "config.json"
        if not config_file.exists():
            validation_result['issues'].append("Configuration file missing")
            validation_result['recommendations'].append("Run configuration generation")
        
        # Check log files
        if not self.logs_dir.exists() or not list(self.logs_dir.glob("*.log")):
            validation_result['issues'].append("No log files found")
        
        # Check for error indicators
        if self.current_state and self.current_state.errors:
            validation_result['valid'] = False
            validation_result['issues'].append(f"{len(self.current_state.errors)} errors in installation state")
        
        return validation_result