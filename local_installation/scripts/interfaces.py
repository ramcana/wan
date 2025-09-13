"""
Core interfaces for the WAN2.2 installation system.
Defines abstract base classes and protocols for system detection,
dependency management, and configuration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from enum import Enum


class InstallationPhase(Enum):
    """Installation phases for progress tracking."""
    DETECTION = "detection"
    DEPENDENCIES = "dependencies"
    MODELS = "models"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    COMPLETE = "complete"


class ErrorCategory(Enum):
    """Categories of installation errors."""
    SYSTEM = "system"
    NETWORK = "network"
    PERMISSION = "permission"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    VALIDATION = "validation"


@dataclass
class CPUInfo:
    """CPU hardware information."""
    model: str
    cores: int
    threads: int
    base_clock: float
    boost_clock: float
    architecture: str


@dataclass
class MemoryInfo:
    """Memory hardware information."""
    total_gb: int
    available_gb: int
    type: str
    speed: int


@dataclass
class GPUInfo:
    """GPU hardware information."""
    model: str
    vram_gb: int
    cuda_version: str
    driver_version: str
    compute_capability: str


@dataclass
class StorageInfo:
    """Storage hardware information."""
    available_gb: int
    type: str  # "SSD", "HDD", "NVMe SSD"


@dataclass
class OSInfo:
    """Operating system information."""
    name: str
    version: str
    architecture: str


@dataclass
class HardwareProfile:
    """Complete hardware profile for the system."""
    cpu: CPUInfo
    memory: MemoryInfo
    gpu: Optional[GPUInfo]
    storage: StorageInfo
    os: OSInfo


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None


@dataclass
class InstallationState:
    """Current state of the installation process."""
    phase: InstallationPhase
    progress: float  # 0.0 to 1.0
    current_task: str
    errors: List[str]
    warnings: List[str]
    hardware_profile: Optional[HardwareProfile]
    installation_path: str


class InstallationError(Exception):
    """Custom exception for installation errors."""
    
    def __init__(self, message: str, category: ErrorCategory, 
                 recovery_suggestions: Optional[List[str]] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.recovery_suggestions = recovery_suggestions or []


class ISystemDetector(ABC):
    """Interface for system hardware detection."""
    
    @abstractmethod
    def detect_hardware(self) -> HardwareProfile:
        """Detect and return complete hardware profile."""
        pass
    
    @abstractmethod
    def get_optimal_settings(self, profile: HardwareProfile) -> Dict[str, Any]:
        """Generate optimal settings for the given hardware profile."""
        pass
    
    @abstractmethod
    def validate_requirements(self, profile: HardwareProfile) -> ValidationResult:
        """Validate that hardware meets minimum requirements."""
        pass


class IDependencyManager(ABC):
    """Interface for dependency management."""
    
    @abstractmethod
    def check_python_installation(self) -> Dict[str, Any]:
        """Check for existing Python installation."""
        pass
    
    @abstractmethod
    def install_python(self, target_dir: str) -> bool:
        """Install Python to the target directory."""
        pass
    
    @abstractmethod
    def create_virtual_environment(self, venv_path: str, 
                                 hardware_profile: HardwareProfile) -> bool:
        """Create virtual environment with hardware-optimized settings."""
        pass
    
    @abstractmethod
    def install_packages(self, requirements_file: str, 
                        hardware_profile: HardwareProfile) -> bool:
        """Install packages with hardware-specific optimizations."""
        pass


class IModelDownloader(ABC):
    """Interface for model downloading and management."""
    
    @abstractmethod
    def check_existing_models(self) -> List[str]:
        """Check which models are already downloaded."""
        pass
    
    @abstractmethod
    def download_wan22_models(self, progress_callback: Optional[Callable] = None) -> bool:
        """Download WAN2.2 models with progress tracking."""
        pass
    
    @abstractmethod
    def verify_model_integrity(self, model_path: str) -> bool:
        """Verify model file integrity."""
        pass
    
    @abstractmethod
    def configure_model_paths(self, config_path: str) -> bool:
        """Configure model paths in application config."""
        pass


class IConfigurationEngine(ABC):
    """Interface for configuration generation."""
    
    @abstractmethod
    def generate_config(self, hardware_profile: HardwareProfile) -> Dict[str, Any]:
        """Generate configuration based on hardware profile."""
        pass
    
    @abstractmethod
    def optimize_for_hardware(self, base_config: Dict[str, Any], 
                            hardware: HardwareProfile) -> Dict[str, Any]:
        """Optimize configuration for specific hardware."""
        pass
    
    @abstractmethod
    def save_config(self, config: Dict[str, Any], config_path: str) -> bool:
        """Save configuration to file."""
        pass


class IInstallationValidator(ABC):
    """Interface for installation validation."""
    
    @abstractmethod
    def validate_dependencies(self) -> ValidationResult:
        """Validate that all dependencies are correctly installed."""
        pass
    
    @abstractmethod
    def validate_models(self) -> ValidationResult:
        """Validate that all models are present and accessible."""
        pass
    
    @abstractmethod
    def validate_hardware_integration(self) -> ValidationResult:
        """Validate hardware integration (GPU acceleration, etc.)."""
        pass
    
    @abstractmethod
    def run_functionality_test(self) -> ValidationResult:
        """Run basic functionality test."""
        pass
    
    @abstractmethod
    def run_performance_baseline(self) -> ValidationResult:
        """Run performance baseline test."""
        pass


class IErrorHandler(ABC):
    """Interface for error handling and recovery."""
    
    @abstractmethod
    def handle_error(self, error: InstallationError) -> str:
        """Handle installation error and return recovery action."""
        pass
    
    @abstractmethod
    def log_error(self, error: InstallationError, context: Dict[str, Any]) -> None:
        """Log error with context information."""
        pass
    
    @abstractmethod
    def suggest_recovery(self, error: InstallationError) -> List[str]:
        """Suggest recovery actions for the error."""
        pass


class IProgressReporter(ABC):
    """Interface for progress reporting."""
    
    @abstractmethod
    def update_progress(self, phase: InstallationPhase, progress: float, 
                       task: str) -> None:
        """Update installation progress."""
        pass
    
    @abstractmethod
    def report_error(self, error: InstallationError) -> None:
        """Report an error to the user."""
        pass
    
    @abstractmethod
    def report_warning(self, message: str) -> None:
        """Report a warning to the user."""
        pass
    
    @abstractmethod
    def report_success(self, message: str) -> None:
        """Report successful completion."""
        pass


class PackagingInterface(ABC):
    """Interface for installer packaging and distribution."""
    
    @abstractmethod
    def create_package(self, version: str, package_name: str = "WAN22-Installer") -> str:
        """Create a complete installation package."""
        pass
    
    @abstractmethod
    def verify_package_integrity(self, package_path: str) -> bool:
        """Verify the integrity of a package."""
        pass
    
    @abstractmethod
    def extract_package(self, package_path: str, extract_dir: str) -> bool:
        """Extract a package to the specified directory."""
        pass
