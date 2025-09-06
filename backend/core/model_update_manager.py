from unittest.mock import Mock, patch
"""
Model Update Management System
Provides model version checking, update detection, safe update processes with rollback capability,
update scheduling, user approval workflows, and update validation for WAN2.2 models.
"""

import asyncio
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
import threading
import aiohttp
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import hashlib
import tempfile

logger = logging.getLogger(__name__)


class UpdateStatus(Enum):
    """Update status enumeration"""
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    VALIDATING = "validating"
    INSTALLING = "installing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLBACK_REQUIRED = "rollback_required"
    ROLLBACK_IN_PROGRESS = "rollback_in_progress"
    ROLLBACK_COMPLETED = "rollback_completed"


class UpdatePriority(Enum):
    """Update priority levels"""
    CRITICAL = "critical"  # Security fixes, major bugs
    HIGH = "high"         # Performance improvements, important features
    MEDIUM = "medium"     # Minor improvements, optimizations
    LOW = "low"          # Optional updates, experimental features


class UpdateType(Enum):
    """Types of updates"""
    MAJOR = "major"       # Breaking changes, new architecture
    MINOR = "minor"       # New features, improvements
    PATCH = "patch"       # Bug fixes, small improvements
    HOTFIX = "hotfix"     # Critical security or stability fixes


@dataclass
class ModelVersion:
    """Model version information"""
    version: str
    release_date: datetime
    size_mb: float
    checksum: str
    download_url: str
    changelog: List[str] = field(default_factory=list)
    compatibility_notes: List[str] = field(default_factory=list)
    minimum_system_requirements: Dict[str, Any] = field(default_factory=dict)
    is_stable: bool = True
    is_beta: bool = False


@dataclass
class UpdateInfo:
    """Information about an available update"""
    model_id: str
    current_version: str
    latest_version: str
    update_type: UpdateType
    priority: UpdatePriority
    size_mb: float
    changelog: List[str] = field(default_factory=list)
    compatibility_notes: List[str] = field(default_factory=list)
    release_date: datetime = field(default_factory=datetime.now)
    download_url: str = ""
    checksum: str = ""
    requires_user_approval: bool = True
    estimated_download_time: Optional[timedelta] = None
    backup_required: bool = True


@dataclass
class UpdateProgress:
    """Update progress tracking"""
    model_id: str
    status: UpdateStatus
    progress_percent: float
    current_step: str
    total_steps: int
    current_step_number: int
    downloaded_mb: float = 0.0
    total_mb: float = 0.0
    speed_mbps: float = 0.0
    eta_seconds: Optional[float] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_update: Optional[datetime] = None
    can_cancel: bool = True
    can_rollback: bool = False


@dataclass
class UpdateResult:
    """Result of an update operation"""
    success: bool
    model_id: str
    old_version: str
    new_version: str
    final_status: UpdateStatus
    total_time_seconds: float
    backup_path: Optional[str] = None
    error_message: Optional[str] = None
    validation_passed: bool = False
    rollback_available: bool = False


@dataclass
class UpdateSchedule:
    """Update scheduling configuration"""
    model_id: str
    scheduled_time: datetime
    auto_approve: bool = False
    backup_before_update: bool = True
    validate_after_update: bool = True
    rollback_on_failure: bool = True
    notification_enabled: bool = True
    max_retry_attempts: int = 3


@dataclass
class RollbackInfo:
    """Information about a rollback operation"""
    model_id: str
    backup_version: str
    backup_path: str
    backup_date: datetime
    backup_size_mb: float
    is_valid: bool = True
    can_restore: bool = True


class ModelUpdateManager:
    """
    Comprehensive model update management system with version checking,
    update detection, safe update processes, rollback capability, and scheduling.
    """
    
    def __init__(self, models_dir: Optional[str] = None, 
                 downloader=None, health_monitor=None):
        """
        Initialize the model update manager.
        
        Args:
            models_dir: Directory containing models
            downloader: Enhanced model downloader instance
            health_monitor: Model health monitor instance
        """
        self.models_dir = Path(models_dir) if models_dir else Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.downloader = downloader
        self.health_monitor = health_monitor
        
        # Update management directories
        self.updates_dir = self.models_dir / ".updates"
        self.backups_dir = self.models_dir / ".backups"
        self.temp_dir = self.models_dir / ".temp"
        
        # Create directories
        for directory in [self.updates_dir, self.backups_dir, self.temp_dir]:
            directory.mkdir(exist_ok=True)
        
        # Version tracking
        self.version_cache: Dict[str, ModelVersion] = {}
        self.update_cache: Dict[str, UpdateInfo] = {}
        
        # Update tracking
        self._active_updates: Dict[str, UpdateProgress] = {}
        self._update_tasks: Dict[str, asyncio.Task] = {}
        self._update_lock = asyncio.Lock()
        
        # Scheduling
        self._scheduled_updates: Dict[str, UpdateSchedule] = {}
        self._scheduler_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self._update_callbacks: List[Callable] = []
        self._notification_callbacks: List[Callable] = []
        
        # Configuration
        self.auto_check_enabled = True
        self.auto_check_interval_hours = 24
        self.backup_retention_days = 30
        self.max_concurrent_updates = 2
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        
        logger.info(f"Model Update Manager initialized with models_dir: {self.models_dir}")
    
    async def initialize(self) -> bool:
        """Initialize the update manager"""
        try:
            # Ensure directories exist
            for directory in [self.updates_dir, self.backups_dir, self.temp_dir]:
                directory.mkdir(exist_ok=True)
            
            # Load existing version information
            await self._load_version_cache()
            
            # Load scheduled updates
            await self._load_scheduled_updates()
            
            # Start automatic checking if enabled
            if self.auto_check_enabled:
                await self._start_auto_checking()
            
            # Start update scheduler
            await self._start_scheduler()
            
            logger.info("Model Update Manager initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Model Update Manager: {e}")
            return False
    
    async def _ensure_session(self):
        """Ensure aiohttp session is available"""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=300, connect=30)
                connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
                self._session = aiohttp.ClientSession(
                    timeout=timeout,
                    connector=connector,
                    headers={'User-Agent': 'WAN22-Update-Manager/1.0'}
                )
    
    async def _close_session(self):
        """Close aiohttp session"""
        async with self._session_lock:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None
    
    def add_update_callback(self, callback: Callable[[UpdateProgress], None]):
        """Add a callback for update progress"""
        self._update_callbacks.append(callback)
    
    def add_notification_callback(self, callback: Callable[[UpdateInfo], None]):
        """Add a callback for update notifications"""
        self._notification_callbacks.append(callback)
    
    async def _notify_update_callbacks(self, progress: UpdateProgress):
        """Notify all update callbacks"""
        progress.last_update = datetime.now()
        for callback in self._update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress)
                else:
                    callback(progress)
            except Exception as e:
                logger.warning(f"Update callback error: {e}")
    
    async def _notify_notification_callbacks(self, update_info: UpdateInfo):
        """Notify all notification callbacks"""
        for callback in self._notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(update_info)
                else:
                    callback(update_info)
            except Exception as e:
                logger.warning(f"Notification callback error: {e}")
    
    async def check_for_updates(self, model_id: Optional[str] = None) -> Dict[str, UpdateInfo]:
        """
        Check for available updates for models.
        
        Args:
            model_id: Specific model to check, or None for all models
            
        Returns:
            Dictionary mapping model IDs to their update information
        """
        logger.info(f"Checking for updates: {model_id or 'all models'}")
        
        try:
            await self._ensure_session()
            
            models_to_check = [model_id] if model_id else self._get_installed_models()
            available_updates = {}
            
            for model in models_to_check:
                try:
                    current_version = await self._get_current_version(model)
                    if not current_version:
                        logger.warning(f"Could not determine current version for {model}")
                        continue
                    
                    latest_version = await self._fetch_latest_version(model)
                    if not latest_version:
                        logger.warning(f"Could not fetch latest version for {model}")
                        continue
                    
                    if self._is_update_available(current_version, latest_version.version):
                        update_info = UpdateInfo(
                            model_id=model,
                            current_version=current_version,
                            latest_version=latest_version.version,
                            update_type=self._determine_update_type(current_version, latest_version.version),
                            priority=self._determine_update_priority(latest_version),
                            size_mb=latest_version.size_mb,
                            changelog=latest_version.changelog,
                            compatibility_notes=latest_version.compatibility_notes,
                            release_date=latest_version.release_date,
                            download_url=latest_version.download_url,
                            checksum=latest_version.checksum
                        )
                        
                        # Estimate download time
                        if latest_version.size_mb > 0:
                            # Assume 10 Mbps average download speed
                            estimated_seconds = (latest_version.size_mb * 8) / 10
                            update_info.estimated_download_time = timedelta(seconds=estimated_seconds)
                        
                        available_updates[model] = update_info
                        self.update_cache[model] = update_info
                        
                        # Notify about available update
                        await self._notify_notification_callbacks(update_info)
                        
                        logger.info(f"Update available for {model}: {current_version} -> {latest_version.version}")
                
                except Exception as e:
                    logger.error(f"Error checking updates for {model}: {e}")
            
            # Save update cache
            await self._save_update_cache()
            
            return available_updates
            
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            return {}
    
    def _get_installed_models(self) -> List[str]:
        """Get list of installed models"""
        models = []
        
        try:
            for item in self.models_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # Check if it looks like a model directory
                    if any(item.glob("*.json")) or any(item.glob("**/config.json")):
                        models.append(item.name)
        except Exception as e:
            logger.error(f"Error getting installed models: {e}")
        
        return models
    
    async def _get_current_version(self, model_id: str) -> Optional[str]:
        """Get current version of an installed model"""
        try:
            model_path = self.models_dir / model_id
            
            # Check for version file
            version_file = model_path / "version.json"
            if version_file.exists():
                async with aiofiles.open(version_file, 'r') as f:
                    version_data = json.loads(await f.read())
                return version_data.get("version", "unknown")
            
            # Check model_index.json for version info
            index_file = model_path / "model_index.json"
            if index_file.exists():
                async with aiofiles.open(index_file, 'r') as f:
                    index_data = json.loads(await f.read())
                return index_data.get("version", index_data.get("_version", "1.0.0"))
            
            # Fallback to directory modification time as version indicator
            stat = model_path.stat()
            return f"local-{int(stat.st_mtime)}"
            
        except Exception as e:
            logger.error(f"Error getting current version for {model_id}: {e}")
            return None
    
    async def _fetch_latest_version(self, model_id: str) -> Optional[ModelVersion]:
        """Fetch latest version information from remote source"""
        try:
            # This would normally fetch from a real API endpoint
            # For now, we'll simulate version information
            
            # Mock version data based on model type
            version_info = {
                "t2v-A14B": {
                    "version": "2.1.0",
                    "size_mb": 8500.0,
                    "changelog": [
                        "Improved video quality and consistency",
                        "Reduced generation time by 15%",
                        "Fixed memory leak issues",
                        "Enhanced text understanding"
                    ],
                    "compatibility_notes": [
                        "Requires CUDA 11.8 or higher",
                        "Minimum 12GB VRAM recommended"
                    ]
                },
                "i2v-A14B": {
                    "version": "1.8.2",
                    "size_mb": 7200.0,
                    "changelog": [
                        "Better image-to-video transitions",
                        "Improved motion consistency",
                        "Bug fixes for edge cases"
                    ],
                    "compatibility_notes": [
                        "Compatible with all previous versions"
                    ]
                },
                "ti2v-5B": {
                    "version": "1.5.1",
                    "size_mb": 5800.0,
                    "changelog": [
                        "Hotfix for text encoding issues",
                        "Performance optimizations"
                    ],
                    "compatibility_notes": [
                        "Backward compatible with v1.4.x"
                    ]
                }
            }
            
            if model_id not in version_info:
                return None
            
            info = version_info[model_id]
            
            return ModelVersion(
                version=info["version"],
                release_date=datetime.now() - timedelta(days=7),  # Released a week ago
                size_mb=info["size_mb"],
                checksum=hashlib.sha256(f"{model_id}-{info['version']}".encode()).hexdigest(),
                download_url=f"https://models.wan22.ai/{model_id}/v{info['version']}/model.tar.gz",
                changelog=info["changelog"],
                compatibility_notes=info["compatibility_notes"],
                minimum_system_requirements={
                    "vram_gb": 12,
                    "cuda_version": "11.8",
                    "python_version": "3.8"
                }
            )
            
        except Exception as e:
            logger.error(f"Error fetching latest version for {model_id}: {e}")
            return None
    
    def _is_update_available(self, current_version: str, latest_version: str) -> bool:
        """Check if an update is available"""
        try:
            # Simple version comparison
            if current_version.startswith("local-"):
                return True  # Local versions should always be updatable
            
            # Parse semantic versions
            current_parts = self._parse_version(current_version)
            latest_parts = self._parse_version(latest_version)
            
            return latest_parts > current_parts
            
        except Exception as e:
            logger.warning(f"Error comparing versions {current_version} vs {latest_version}: {e}")
            return False
    
    def _parse_version(self, version: str) -> Tuple[int, int, int]:
        """Parse semantic version string"""
        try:
            # Remove any prefix/suffix and split by dots
            clean_version = version.split('-')[0].split('+')[0]
            parts = clean_version.split('.')
            
            # Pad with zeros if needed
            while len(parts) < 3:
                parts.append('0')
            
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        except:
            return (0, 0, 0)
    
    def _determine_update_type(self, current_version: str, latest_version: str) -> UpdateType:
        """Determine the type of update"""
        try:
            current_parts = self._parse_version(current_version)
            latest_parts = self._parse_version(latest_version)
            
            if latest_parts[0] > current_parts[0]:
                return UpdateType.MAJOR
            elif latest_parts[1] > current_parts[1]:
                return UpdateType.MINOR
            else:
                return UpdateType.PATCH
                
        except:
            return UpdateType.MINOR
    
    def _determine_update_priority(self, version_info: ModelVersion) -> UpdatePriority:
        """Determine update priority based on changelog"""
        changelog_text = ' '.join(version_info.changelog).lower()
        
        if any(keyword in changelog_text for keyword in ['security', 'critical', 'hotfix', 'urgent']):
            return UpdatePriority.CRITICAL
        elif any(keyword in changelog_text for keyword in ['performance', 'speed', 'memory', 'optimization']):
            return UpdatePriority.HIGH
        elif any(keyword in changelog_text for keyword in ['improvement', 'enhancement', 'feature']):
            return UpdatePriority.MEDIUM
        else:
            return UpdatePriority.LOW
    
    async def schedule_update(self, model_id: str, scheduled_time: datetime, 
                            auto_approve: bool = False) -> bool:
        """
        Schedule an update for a specific time.
        
        Args:
            model_id: Model to update
            scheduled_time: When to perform the update
            auto_approve: Whether to auto-approve the update
            
        Returns:
            True if successfully scheduled
        """
        try:
            schedule = UpdateSchedule(
                model_id=model_id,
                scheduled_time=scheduled_time,
                auto_approve=auto_approve
            )
            
            self._scheduled_updates[model_id] = schedule
            
            # Save scheduled updates
            await self._save_scheduled_updates()
            
            logger.info(f"Scheduled update for {model_id} at {scheduled_time}")
            return True
            
        except Exception as e:
            logger.error(f"Error scheduling update for {model_id}: {e}")
            return False
    
    async def perform_update(self, model_id: str, user_approved: bool = False) -> UpdateResult:
        """
        Perform a model update with backup and validation.
        
        Args:
            model_id: Model to update
            user_approved: Whether user has approved the update
            
        Returns:
            UpdateResult with operation details
        """
        logger.info(f"Starting update for model: {model_id}")
        start_time = time.time()
        
        # Check if update is available
        if model_id not in self.update_cache:
            await self.check_for_updates(model_id)
        
        if model_id not in self.update_cache:
            return UpdateResult(
                success=False,
                model_id=model_id,
                old_version="unknown",
                new_version="unknown",
                final_status=UpdateStatus.FAILED,
                total_time_seconds=0.0,
                error_message="No update available"
            )
        
        update_info = self.update_cache[model_id]
        
        # Check user approval if required
        if update_info.requires_user_approval and not user_approved:
            return UpdateResult(
                success=False,
                model_id=model_id,
                old_version=update_info.current_version,
                new_version=update_info.latest_version,
                final_status=UpdateStatus.FAILED,
                total_time_seconds=0.0,
                error_message="User approval required"
            )
        
        # Initialize progress tracking
        progress = UpdateProgress(
            model_id=model_id,
            status=UpdateStatus.DOWNLOADING,
            progress_percent=0.0,
            current_step="Preparing update",
            total_steps=6,
            current_step_number=1,
            total_mb=update_info.size_mb,
            started_at=datetime.now()
        )
        
        async with self._update_lock:
            self._active_updates[model_id] = progress
        
        await self._notify_update_callbacks(progress)
        
        try:
            # Step 1: Create backup
            progress.current_step = "Creating backup"
            progress.current_step_number = 1
            await self._notify_update_callbacks(progress)
            
            backup_path = await self._create_backup(model_id)
            if not backup_path:
                raise Exception("Failed to create backup")
            
            progress.progress_percent = 16.7
            await self._notify_update_callbacks(progress)
            
            # Step 2: Download new version
            progress.current_step = "Downloading update"
            progress.current_step_number = 2
            progress.status = UpdateStatus.DOWNLOADING
            await self._notify_update_callbacks(progress)
            
            download_path = await self._download_update(model_id, update_info, progress)
            if not download_path:
                raise Exception("Failed to download update")
            
            progress.progress_percent = 50.0
            await self._notify_update_callbacks(progress)
            
            # Step 3: Validate download
            progress.current_step = "Validating download"
            progress.current_step_number = 3
            progress.status = UpdateStatus.VALIDATING
            await self._notify_update_callbacks(progress)
            
            if not await self._validate_download(download_path, update_info.checksum):
                raise Exception("Download validation failed")
            
            progress.progress_percent = 66.7
            await self._notify_update_callbacks(progress)
            
            # Step 4: Install update
            progress.current_step = "Installing update"
            progress.current_step_number = 4
            progress.status = UpdateStatus.INSTALLING
            await self._notify_update_callbacks(progress)
            
            if not await self._install_update(model_id, download_path):
                raise Exception("Failed to install update")
            
            progress.progress_percent = 83.3
            await self._notify_update_callbacks(progress)
            
            # Step 5: Validate installation
            progress.current_step = "Validating installation"
            progress.current_step_number = 5
            await self._notify_update_callbacks(progress)
            
            validation_passed = await self._validate_installation(model_id)
            
            progress.progress_percent = 95.0
            await self._notify_update_callbacks(progress)
            
            # Step 6: Finalize
            progress.current_step = "Finalizing update"
            progress.current_step_number = 6
            await self._notify_update_callbacks(progress)
            
            if validation_passed:
                # Update version information
                await self._update_version_info(model_id, update_info.latest_version)
                
                # Clean up temporary files
                await self._cleanup_temp_files(model_id)
                
                progress.status = UpdateStatus.COMPLETED
                progress.progress_percent = 100.0
                progress.completed_at = datetime.now()
                progress.can_rollback = True
                await self._notify_update_callbacks(progress)
                
                logger.info(f"Update completed successfully for {model_id}")
                
                return UpdateResult(
                    success=True,
                    model_id=model_id,
                    old_version=update_info.current_version,
                    new_version=update_info.latest_version,
                    final_status=UpdateStatus.COMPLETED,
                    total_time_seconds=time.time() - start_time,
                    backup_path=backup_path,
                    validation_passed=True,
                    rollback_available=True
                )
            else:
                # Validation failed, rollback
                logger.warning(f"Update validation failed for {model_id}, initiating rollback")
                
                rollback_success = await self._perform_rollback(model_id, backup_path)
                
                return UpdateResult(
                    success=False,
                    model_id=model_id,
                    old_version=update_info.current_version,
                    new_version=update_info.latest_version,
                    final_status=UpdateStatus.ROLLBACK_COMPLETED if rollback_success else UpdateStatus.FAILED,
                    total_time_seconds=time.time() - start_time,
                    backup_path=backup_path,
                    error_message="Installation validation failed, rollback performed" if rollback_success else "Installation and rollback both failed",
                    validation_passed=False,
                    rollback_available=not rollback_success
                )
        
        except Exception as e:
            logger.error(f"Update failed for {model_id}: {e}")
            
            progress.status = UpdateStatus.FAILED
            progress.error_message = str(e)
            await self._notify_update_callbacks(progress)
            
            # Attempt rollback if backup exists
            rollback_success = False
            if 'backup_path' in locals() and backup_path:
                rollback_success = await self._perform_rollback(model_id, backup_path)
            
            return UpdateResult(
                success=False,
                model_id=model_id,
                old_version=update_info.current_version,
                new_version=update_info.latest_version,
                final_status=UpdateStatus.ROLLBACK_COMPLETED if rollback_success else UpdateStatus.FAILED,
                total_time_seconds=time.time() - start_time,
                backup_path=backup_path if 'backup_path' in locals() else None,
                error_message=str(e),
                validation_passed=False,
                rollback_available=not rollback_success
            )
        
        finally:
            # Clean up tracking
            async with self._update_lock:
                if model_id in self._active_updates:
                    del self._active_updates[model_id]
    
    async def _create_backup(self, model_id: str) -> Optional[str]:
        """Create a backup of the current model"""
        try:
            model_path = self.models_dir / model_id
            if not model_path.exists():
                logger.error(f"Model path does not exist: {model_path}")
                return None
            
            # Create backup directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{model_id}_backup_{timestamp}"
            backup_path = self.backups_dir / backup_name
            
            # Copy model directory to backup location
            shutil.copytree(model_path, backup_path)
            
            # Create backup metadata
            backup_info = {
                "model_id": model_id,
                "backup_date": datetime.now().isoformat(),
                "original_path": str(model_path),
                "backup_size_mb": self._calculate_directory_size(backup_path) / (1024 * 1024)
            }
            
            backup_info_file = backup_path / "backup_info.json"
            async with aiofiles.open(backup_info_file, 'w') as f:
                await f.write(json.dumps(backup_info, indent=2))
            
            logger.info(f"Created backup for {model_id} at {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Error creating backup for {model_id}: {e}")
            return None
    
    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of a directory"""
        total_size = 0
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Error calculating directory size: {e}")
        return total_size
    
    async def _download_update(self, model_id: str, update_info: UpdateInfo, 
                             progress: UpdateProgress) -> Optional[str]:
        """Download the update file"""
        try:
            # Create temporary download path
            temp_file = self.temp_dir / f"{model_id}_update.tar.gz"
            
            # Use enhanced downloader if available
            if self.downloader:
                # Set up progress callback
                def download_progress_callback(download_progress):
                    progress.downloaded_mb = download_progress.downloaded_mb
                    progress.speed_mbps = download_progress.speed_mbps
                    progress.eta_seconds = download_progress.eta_seconds
                    
                    # Update overall progress (download is 16.7% to 50% of total)
                    download_percent = download_progress.progress_percent
                    overall_percent = 16.7 + (download_percent * 0.333)
                    progress.progress_percent = overall_percent
                
                self.downloader.add_progress_callback(download_progress_callback)
                
                try:
                    result = await self.downloader.download_with_retry(
                        model_id, update_info.download_url
                    )
                    
                    if result.success:
                        return result.download_path
                    else:
                        logger.error(f"Download failed: {result.error_message}")
                        return None
                        
                finally:
                    self.downloader.remove_progress_callback(download_progress_callback)
            
            else:
                # Fallback to basic download
                await self._ensure_session()
                
                async with self._session.get(update_info.download_url) as response:
                    if response.status != 200:
                        logger.error(f"Download failed with status {response.status}")
                        return None
                    
                    total_size = int(response.headers.get('Content-Length', 0))
                    downloaded = 0
                    
                    async with aiofiles.open(temp_file, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Update progress
                            if total_size > 0:
                                download_percent = (downloaded / total_size) * 100
                                overall_percent = 16.7 + (download_percent * 0.333)
                                progress.progress_percent = overall_percent
                                progress.downloaded_mb = downloaded / (1024 * 1024)
                
                return str(temp_file)
            
        except Exception as e:
            logger.error(f"Error downloading update for {model_id}: {e}")
            return None
    
    async def _validate_download(self, download_path: str, expected_checksum: str) -> bool:
        """Validate downloaded file integrity"""
        try:
            # Calculate checksum
            sha256_hash = hashlib.sha256()
            
            async with aiofiles.open(download_path, 'rb') as f:
                while chunk := await f.read(8192):
                    sha256_hash.update(chunk)
            
            actual_checksum = sha256_hash.hexdigest()
            
            if actual_checksum == expected_checksum:
                logger.info("Download validation passed")
                return True
            else:
                logger.error(f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}")
                return False
                
        except Exception as e:
            logger.error(f"Error validating download: {e}")
            return False
    
    async def _install_update(self, model_id: str, download_path: str) -> bool:
        """Install the downloaded update"""
        try:
            model_path = self.models_dir / model_id
            
            # Extract update (assuming it's a tar.gz file)
            import tarfile
            
            with tarfile.open(download_path, 'r:gz') as tar:
                # Extract to temporary location first
                temp_extract_path = self.temp_dir / f"{model_id}_extract"
                tar.extractall(temp_extract_path)
            
            # Remove old model directory
            if model_path.exists():
                shutil.rmtree(model_path)
            
            # Move extracted files to model directory
            extracted_model_path = temp_extract_path / model_id
            if not extracted_model_path.exists():
                # Look for the first directory in extracted content
                extracted_dirs = [d for d in temp_extract_path.iterdir() if d.is_dir()]
                if extracted_dirs:
                    extracted_model_path = extracted_dirs[0]
                else:
                    raise Exception("No model directory found in extracted content")
            
            shutil.move(str(extracted_model_path), str(model_path))
            
            # Clean up temporary extraction directory
            if temp_extract_path.exists():
                shutil.rmtree(temp_extract_path)
            
            logger.info(f"Successfully installed update for {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error installing update for {model_id}: {e}")
            return False
    
    async def _validate_installation(self, model_id: str) -> bool:
        """Validate the installed model"""
        try:
            # Use health monitor if available
            if self.health_monitor:
                integrity_result = await self.health_monitor.check_model_integrity(model_id)
                return integrity_result.is_healthy
            
            # Basic validation - check if essential files exist
            model_path = self.models_dir / model_id
            
            essential_files = ["config.json", "model_index.json"]
            for file_name in essential_files:
                file_path = model_path / file_name
                if not file_path.exists():
                    logger.error(f"Essential file missing after installation: {file_name}")
                    return False
            
            logger.info(f"Installation validation passed for {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating installation for {model_id}: {e}")
            return False
    
    async def _perform_rollback(self, model_id: str, backup_path: str) -> bool:
        """Perform rollback to previous version"""
        try:
            logger.info(f"Starting rollback for {model_id}")
            
            model_path = self.models_dir / model_id
            backup_dir = Path(backup_path)
            
            if not backup_dir.exists():
                logger.error(f"Backup directory does not exist: {backup_path}")
                return False
            
            # Remove current (failed) installation
            if model_path.exists():
                shutil.rmtree(model_path)
            
            # Restore from backup
            shutil.copytree(backup_dir, model_path)
            
            logger.info(f"Rollback completed successfully for {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error during rollback for {model_id}: {e}")
            return False
    
    async def _update_version_info(self, model_id: str, new_version: str):
        """Update version information for a model"""
        try:
            model_path = self.models_dir / model_id
            version_file = model_path / "version.json"
            
            version_info = {
                "version": new_version,
                "updated_at": datetime.now().isoformat(),
                "update_manager_version": "1.0.0"
            }
            
            async with aiofiles.open(version_file, 'w') as f:
                await f.write(json.dumps(version_info, indent=2))
            
            # Update cache
            if model_id in self.version_cache:
                self.version_cache[model_id].version = new_version
            
        except Exception as e:
            logger.error(f"Error updating version info for {model_id}: {e}")
    
    async def _cleanup_temp_files(self, model_id: str):
        """Clean up temporary files after update"""
        try:
            # Remove temporary download files
            temp_files = list(self.temp_dir.glob(f"{model_id}_*"))
            for temp_file in temp_files:
                if temp_file.is_file():
                    temp_file.unlink()
                elif temp_file.is_dir():
                    shutil.rmtree(temp_file)
            
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {e}")
    
    async def get_update_progress(self, model_id: str) -> Optional[UpdateProgress]:
        """Get update progress for a model"""
        try:
            async with self._update_lock:
                return self._active_updates.get(model_id)
        except Exception as e:
            logger.error(f"Error getting update progress for {model_id}: {e}")
            return None
    
    async def cancel_update(self, model_id: str) -> bool:
        """Cancel an active update"""
        try:
            async with self._update_lock:
                if model_id in self._active_updates:
                    progress = self._active_updates[model_id]
                    if progress.can_cancel:
                        progress.status = UpdateStatus.CANCELLED
                        await self._notify_update_callbacks(progress)
                        
                        # Cancel download if using enhanced downloader
                        if self.downloader:
                            await self.downloader.cancel_download(model_id)
                        
                        logger.info(f"Cancelled update for {model_id}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling update for {model_id}: {e}")
            return False
    
    async def get_rollback_info(self, model_id: str) -> List[RollbackInfo]:
        """Get available rollback options for a model"""
        try:
            rollback_options = []
            
            # Find backup directories for this model
            backup_pattern = f"{model_id}_backup_*"
            backup_dirs = list(self.backups_dir.glob(backup_pattern))
            
            for backup_dir in backup_dirs:
                try:
                    backup_info_file = backup_dir / "backup_info.json"
                    if backup_info_file.exists():
                        async with aiofiles.open(backup_info_file, 'r') as f:
                            backup_data = json.loads(await f.read())
                        
                        rollback_info = RollbackInfo(
                            model_id=model_id,
                            backup_version=backup_data.get("version", "unknown"),
                            backup_path=str(backup_dir),
                            backup_date=datetime.fromisoformat(backup_data["backup_date"]),
                            backup_size_mb=backup_data.get("backup_size_mb", 0.0),
                            is_valid=backup_dir.exists() and any(backup_dir.iterdir()),
                            can_restore=True
                        )
                        
                        rollback_options.append(rollback_info)
                
                except Exception as e:
                    logger.warning(f"Error reading backup info for {backup_dir}: {e}")
            
            # Sort by backup date (newest first)
            rollback_options.sort(key=lambda x: x.backup_date, reverse=True)
            
            return rollback_options
            
        except Exception as e:
            logger.error(f"Error getting rollback info for {model_id}: {e}")
            return []
    
    async def perform_rollback(self, model_id: str, backup_path: str) -> bool:
        """Perform rollback to a specific backup"""
        try:
            return await self._perform_rollback(model_id, backup_path)
        except Exception as e:
            logger.error(f"Error performing rollback for {model_id}: {e}")
            return False
    
    async def _start_auto_checking(self):
        """Start automatic update checking"""
        try:
            async def auto_check_loop():
                while self.auto_check_enabled:
                    try:
                        await self.check_for_updates()
                        await asyncio.sleep(self.auto_check_interval_hours * 3600)
                    except Exception as e:
                        logger.error(f"Error in auto-check loop: {e}")
                        await asyncio.sleep(3600)  # Wait 1 hour before retry
            
            asyncio.create_task(auto_check_loop())
            logger.info("Started automatic update checking")
            
        except Exception as e:
            logger.error(f"Error starting auto-checking: {e}")
    
    async def _start_scheduler(self):
        """Start update scheduler"""
        try:
            async def scheduler_loop():
                while True:
                    try:
                        current_time = datetime.now()
                        
                        # Check for scheduled updates
                        for model_id, schedule in list(self._scheduled_updates.items()):
                            if current_time >= schedule.scheduled_time:
                                logger.info(f"Executing scheduled update for {model_id}")
                                
                                # Perform update
                                result = await self.perform_update(model_id, schedule.auto_approve)
                                
                                # Remove from schedule
                                del self._scheduled_updates[model_id]
                                await self._save_scheduled_updates()
                                
                                logger.info(f"Scheduled update completed for {model_id}: {result.success}")
                        
                        # Check every 5 minutes
                        await asyncio.sleep(300)
                        
                    except Exception as e:
                        logger.error(f"Error in scheduler loop: {e}")
                        await asyncio.sleep(300)
            
            self._scheduler_task = asyncio.create_task(scheduler_loop())
            logger.info("Started update scheduler")
            
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
    
    async def _load_version_cache(self):
        """Load version cache from disk"""
        try:
            cache_file = self.updates_dir / "version_cache.json"
            if cache_file.exists():
                async with aiofiles.open(cache_file, 'r') as f:
                    cache_data = json.loads(await f.read())
                
                for model_id, version_data in cache_data.items():
                    self.version_cache[model_id] = ModelVersion(**version_data)
        
        except Exception as e:
            logger.warning(f"Error loading version cache: {e}")
    
    async def _save_version_cache(self):
        """Save version cache to disk"""
        try:
            cache_file = self.updates_dir / "version_cache.json"
            cache_data = {}
            
            for model_id, version in self.version_cache.items():
                cache_data[model_id] = {
                    "version": version.version,
                    "release_date": version.release_date.isoformat(),
                    "size_mb": version.size_mb,
                    "checksum": version.checksum,
                    "download_url": version.download_url,
                    "changelog": version.changelog,
                    "compatibility_notes": version.compatibility_notes,
                    "is_stable": version.is_stable,
                    "is_beta": version.is_beta
                }
            
            async with aiofiles.open(cache_file, 'w') as f:
                await f.write(json.dumps(cache_data, indent=2))
        
        except Exception as e:
            logger.warning(f"Error saving version cache: {e}")
    
    async def _save_update_cache(self):
        """Save update cache to disk"""
        try:
            cache_file = self.updates_dir / "update_cache.json"
            cache_data = {}
            
            for model_id, update_info in self.update_cache.items():
                cache_data[model_id] = {
                    "model_id": update_info.model_id,
                    "current_version": update_info.current_version,
                    "latest_version": update_info.latest_version,
                    "update_type": update_info.update_type.value,
                    "priority": update_info.priority.value,
                    "size_mb": update_info.size_mb,
                    "changelog": update_info.changelog,
                    "compatibility_notes": update_info.compatibility_notes,
                    "release_date": update_info.release_date.isoformat(),
                    "download_url": update_info.download_url,
                    "checksum": update_info.checksum,
                    "requires_user_approval": update_info.requires_user_approval
                }
            
            async with aiofiles.open(cache_file, 'w') as f:
                await f.write(json.dumps(cache_data, indent=2))
        
        except Exception as e:
            logger.warning(f"Error saving update cache: {e}")
    
    async def _load_scheduled_updates(self):
        """Load scheduled updates from disk"""
        try:
            schedule_file = self.updates_dir / "scheduled_updates.json"
            if schedule_file.exists():
                async with aiofiles.open(schedule_file, 'r') as f:
                    schedule_data = json.loads(await f.read())
                
                for model_id, schedule_info in schedule_data.items():
                    self._scheduled_updates[model_id] = UpdateSchedule(
                        model_id=model_id,
                        scheduled_time=datetime.fromisoformat(schedule_info["scheduled_time"]),
                        auto_approve=schedule_info.get("auto_approve", False),
                        backup_before_update=schedule_info.get("backup_before_update", True),
                        validate_after_update=schedule_info.get("validate_after_update", True),
                        rollback_on_failure=schedule_info.get("rollback_on_failure", True),
                        notification_enabled=schedule_info.get("notification_enabled", True),
                        max_retry_attempts=schedule_info.get("max_retry_attempts", 3)
                    )
        
        except Exception as e:
            logger.warning(f"Error loading scheduled updates: {e}")
    
    async def _save_scheduled_updates(self):
        """Save scheduled updates to disk"""
        try:
            schedule_file = self.updates_dir / "scheduled_updates.json"
            schedule_data = {}
            
            for model_id, schedule in self._scheduled_updates.items():
                schedule_data[model_id] = {
                    "scheduled_time": schedule.scheduled_time.isoformat(),
                    "auto_approve": schedule.auto_approve,
                    "backup_before_update": schedule.backup_before_update,
                    "validate_after_update": schedule.validate_after_update,
                    "rollback_on_failure": schedule.rollback_on_failure,
                    "notification_enabled": schedule.notification_enabled,
                    "max_retry_attempts": schedule.max_retry_attempts
                }
            
            async with aiofiles.open(schedule_file, 'w') as f:
                await f.write(json.dumps(schedule_data, indent=2))
        
        except Exception as e:
            logger.warning(f"Error saving scheduled updates: {e}")
    
    async def cleanup_old_backups(self):
        """Clean up old backups based on retention policy"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.backup_retention_days)
            
            for backup_dir in self.backups_dir.iterdir():
                if backup_dir.is_dir():
                    try:
                        backup_info_file = backup_dir / "backup_info.json"
                        if backup_info_file.exists():
                            async with aiofiles.open(backup_info_file, 'r') as f:
                                backup_data = json.loads(await f.read())
                            
                            backup_date = datetime.fromisoformat(backup_data["backup_date"])
                            
                            if backup_date < cutoff_date:
                                shutil.rmtree(backup_dir)
                                logger.info(f"Removed old backup: {backup_dir.name}")
                    
                    except Exception as e:
                        logger.warning(f"Error processing backup {backup_dir.name}: {e}")
        
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
    
    async def shutdown(self):
        """Shutdown the update manager"""
        try:
            # Cancel scheduler task
            if self._scheduler_task:
                self._scheduler_task.cancel()
            
            # Close session
            await self._close_session()
            
            # Save caches
            await self._save_version_cache()
            await self._save_update_cache()
            await self._save_scheduled_updates()
            
            logger.info("Model Update Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")