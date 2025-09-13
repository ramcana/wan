"""
Dependency management system for WAN2.2 installation.
Handles Python installation, virtual environment creation, and package management
with hardware-specific optimizations.
"""

import os
import sys
import json
import shutil
import urllib.request
import zipfile
import subprocess
import tempfile
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urlparse

from interfaces import (
    IDependencyManager, HardwareProfile, InstallationError, 
    ErrorCategory, ValidationResult, IProgressReporter
)
from base_classes import BaseInstallationComponent
from package_resolver import PackageInstallationOrchestrator


class PythonInstallationHandler(BaseInstallationComponent):
    """Handles Python detection, download, and installation."""
    
    # Python embedded download URLs for Windows
    PYTHON_URLS = {
        "3.11": {
            "x64": "https://www.python.org/ftp/python/3.11.7/python-3.11.7-embed-amd64.zip",
            "x86": "https://www.python.org/ftp/python/3.11.7/python-3.11.7-embed-win32.zip"
        },
        "3.10": {
            "x64": "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip",
            "x86": "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-win32.zip"
        }
    }
    
    def __init__(self, installation_path: str, progress_reporter: Optional[IProgressReporter] = None):
        super().__init__(installation_path)
        self.progress_reporter = progress_reporter
        self.python_dir = self.installation_path / "python"
        self.venv_dir = self.installation_path / "venv"
        
    def check_python_installation(self) -> Dict[str, Any]:
        """Check for existing Python installation."""
        self.logger.info("Checking for existing Python installation...")
        
        python_info = {
            "system_python": None,
            "embedded_python": None,
            "recommended_action": "install_embedded"
        }
        
        # Check system Python
        try:
            result = subprocess.run([sys.executable, "--version"], 
                                  capture_output=True, text=True, check=True)
            version_str = result.stdout.strip()
            version_parts = version_str.split()[1].split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            
            python_info["system_python"] = {
                "path": sys.executable,
                "version": version_str,
                "major": major,
                "minor": minor,
                "suitable": major == 3 and minor >= 9
            }
            
            self.logger.info(f"Found system Python: {version_str}")
            
        except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
            self.logger.info("No suitable system Python found")
        
        # Check embedded Python
        embedded_python = self.python_dir / "python.exe"
        if embedded_python.exists():
            try:
                result = subprocess.run([str(embedded_python), "--version"], 
                                      capture_output=True, text=True, check=True)
                version_str = result.stdout.strip()
                python_info["embedded_python"] = {
                    "path": str(embedded_python),
                    "version": version_str,
                    "suitable": True
                }
                python_info["recommended_action"] = "use_embedded"
                self.logger.info(f"Found embedded Python: {version_str}")
                
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.logger.info("Embedded Python exists but not functional")
        
        # Determine recommended action
        if python_info["embedded_python"] and python_info["embedded_python"]["suitable"]:
            python_info["recommended_action"] = "use_embedded"
        elif python_info["system_python"] and python_info["system_python"]["suitable"]:
            python_info["recommended_action"] = "use_system"
        else:
            python_info["recommended_action"] = "install_embedded"
        
        return python_info
    
    def install_python(self, target_dir: Optional[str] = None) -> bool:
        """Install embedded Python for portable deployment."""
        if target_dir:
            self.python_dir = Path(target_dir)
        
        self.logger.info("Installing embedded Python...")
        
        if self.progress_reporter:
            self.progress_reporter.update_progress(
                phase=None, progress=0.1, 
                task="Preparing Python installation"
            )
        
        try:
            # Determine architecture and Python version
            arch = "x64" if platform.machine().endswith('64') else "x86"
            python_version = "3.11"  # Default to 3.11
            
            if python_version not in self.PYTHON_URLS:
                raise InstallationError(
                    f"Python version {python_version} not supported",
                    ErrorCategory.SYSTEM,
                    ["Use Python 3.10 or 3.11"]
                )
            
            download_url = self.PYTHON_URLS[python_version][arch]
            self.logger.info(f"Downloading Python {python_version} ({arch}) from {download_url}")
            
            # Download Python embedded
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                zip_path = temp_path / "python.zip"
                
                self._download_with_progress(download_url, zip_path, "Downloading Python")
                
                if self.progress_reporter:
                    self.progress_reporter.update_progress(
                        phase=None, progress=0.7, 
                        task="Extracting Python"
                    )
                
                # Extract Python
                self.ensure_directory(self.python_dir)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.python_dir)
                
                self.logger.info(f"Python extracted to {self.python_dir}")
                
                # Configure Python for package installation
                self._configure_embedded_python()
                
                if self.progress_reporter:
                    self.progress_reporter.update_progress(
                        phase=None, progress=1.0, 
                        task="Python installation complete"
                    )
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to install Python: {e}")
            raise InstallationError(
                f"Python installation failed: {str(e)}",
                ErrorCategory.NETWORK,
                ["Check internet connection", "Try again later", "Use system Python"]
            )
    
    def _download_with_progress(self, url: str, target_path: Path, task_name: str) -> None:
        """Download file with progress reporting."""
        def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0 and self.progress_reporter:
                downloaded = block_num * block_size
                progress = min(downloaded / total_size, 1.0)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                
                self.progress_reporter.update_progress(
                    phase=None, 
                    progress=0.1 + (progress * 0.5),  # 10% to 60% of total progress
                    task=f"{task_name} ({mb_downloaded:.1f}/{mb_total:.1f} MB)"
                )
        
        urllib.request.urlretrieve(url, target_path, progress_hook)
    
    def _configure_embedded_python(self) -> None:
        """Configure embedded Python for package installation."""
        # Create pth file to enable site-packages
        pth_file = self.python_dir / "python311._pth"  # Adjust version as needed
        if not pth_file.exists():
            # Find the actual pth file
            pth_files = list(self.python_dir.glob("python*._pth"))
            if pth_files:
                pth_file = pth_files[0]
        
        if pth_file.exists():
            # Read current content
            content = pth_file.read_text()
            
            # Add site-packages if not present
            if "import site" not in content:
                content += "\nimport site\n"
                pth_file.write_text(content)
                self.logger.info("Enabled site-packages for embedded Python")
        
        # Install pip if not present
        pip_path = self.python_dir / "Scripts" / "pip.exe"
        if not pip_path.exists():
            self._install_pip()
    
    def _install_pip(self) -> None:
        """Install pip in embedded Python."""
        self.logger.info("Installing pip...")
        
        # Download get-pip.py
        get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            get_pip_path = Path(temp_dir) / "get-pip.py"
            urllib.request.urlretrieve(get_pip_url, get_pip_path)
            
            # Run get-pip.py
            python_exe = self.python_dir / "python.exe"
            subprocess.run([str(python_exe), str(get_pip_path)], check=True)
            
            self.logger.info("Pip installation complete")
    
    def create_virtual_environment(self, venv_path: Optional[str] = None, 
                                 hardware_profile: Optional[HardwareProfile] = None) -> bool:
        """Create virtual environment with hardware-optimized settings."""
        if venv_path:
            self.venv_dir = Path(venv_path)
        
        self.logger.info(f"Creating virtual environment at {self.venv_dir}")
        
        try:
            # Determine Python executable
            python_info = self.check_python_installation()
            
            if python_info["embedded_python"] and python_info["embedded_python"]["suitable"]:
                python_exe = python_info["embedded_python"]["path"]
            elif python_info["system_python"] and python_info["system_python"]["suitable"]:
                python_exe = python_info["system_python"]["path"]
            else:
                raise InstallationError(
                    "No suitable Python installation found",
                    ErrorCategory.SYSTEM,
                    ["Install Python first", "Check Python installation"]
                )
            
            # Create virtual environment
            self.ensure_directory(self.venv_dir.parent)
            
            # Use venv module to create virtual environment
            subprocess.run([
                python_exe, "-m", "venv", str(self.venv_dir)
            ], check=True)
            
            self.logger.info("Virtual environment created successfully")
            
            # Configure virtual environment for hardware
            if hardware_profile:
                self._configure_venv_for_hardware(hardware_profile)
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create virtual environment: {e}")
            raise InstallationError(
                f"Virtual environment creation failed: {str(e)}",
                ErrorCategory.SYSTEM,
                ["Check Python installation", "Ensure sufficient disk space"]
            )
    
    def _configure_venv_for_hardware(self, hardware_profile: HardwareProfile) -> None:
        """Configure virtual environment with hardware-specific settings."""
        # Create activation script with hardware-optimized environment variables
        if platform.system() == "Windows":
            activate_script = self.venv_dir / "Scripts" / "activate.bat"
            env_vars = self._get_hardware_env_vars(hardware_profile)
            
            if activate_script.exists():
                # Read existing activation script
                content = activate_script.read_text()
                
                # Add hardware-specific environment variables
                env_section = "\nREM Hardware-optimized environment variables\n"
                for var, value in env_vars.items():
                    env_section += f"set {var}={value}\n"
                
                # Insert before the final label
                if ":end" in content:
                    content = content.replace(":end", env_section + ":end")
                else:
                    content += env_section
                
                activate_script.write_text(content)
                self.logger.info("Configured virtual environment for hardware optimization")
    
    def _get_hardware_env_vars(self, hardware_profile: HardwareProfile) -> Dict[str, str]:
        """Get hardware-specific environment variables."""
        env_vars = {}
        
        # CPU optimization
        if hardware_profile.cpu.threads >= 32:
            env_vars["OMP_NUM_THREADS"] = str(min(hardware_profile.cpu.threads // 2, 32))
            env_vars["MKL_NUM_THREADS"] = str(min(hardware_profile.cpu.threads // 2, 32))
        else:
            env_vars["OMP_NUM_THREADS"] = str(hardware_profile.cpu.threads)
            env_vars["MKL_NUM_THREADS"] = str(hardware_profile.cpu.threads)
        
        # Memory optimization
        if hardware_profile.memory.total_gb >= 64:
            env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        elif hardware_profile.memory.total_gb >= 32:
            env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
        else:
            env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # GPU optimization
        if hardware_profile.gpu and "RTX" in hardware_profile.gpu.model:
            env_vars["CUDA_VISIBLE_DEVICES"] = "0"
            if hardware_profile.gpu.vram_gb >= 16:
                env_vars["PYTORCH_CUDA_ALLOC_CONF"] += ",expandable_segments:True"
        
        return env_vars
    
    def get_python_executable(self) -> str:
        """Get the path to the Python executable to use."""
        python_info = self.check_python_installation()
        
        if python_info["embedded_python"] and python_info["embedded_python"]["suitable"]:
            return python_info["embedded_python"]["path"]
        elif python_info["system_python"] and python_info["system_python"]["suitable"]:
            return python_info["system_python"]["path"]
        else:
            raise InstallationError(
                "No suitable Python installation found",
                ErrorCategory.SYSTEM,
                ["Install Python first", "Check Python installation"]
            )
    
    def get_venv_python_executable(self) -> str:
        """Get the path to the virtual environment Python executable."""
        if platform.system() == "Windows":
            venv_python = self.venv_dir / "Scripts" / "python.exe"
        else:
            venv_python = self.venv_dir / "bin" / "python"
        
        if not venv_python.exists():
            raise InstallationError(
                "Virtual environment not found or not properly created",
                ErrorCategory.SYSTEM,
                ["Create virtual environment first", "Check virtual environment path"]
            )
        
        return str(venv_python)


class PackageInstallationSystem(BaseInstallationComponent):
    """Handles package installation with hardware-specific optimizations."""
    
    def __init__(self, installation_path: str, python_handler: PythonInstallationHandler,
                 progress_reporter: Optional[IProgressReporter] = None):
        super().__init__(installation_path)
        self.python_handler = python_handler
        self.progress_reporter = progress_reporter
        self.requirements_file = self.installation_path / "resources" / "requirements.txt"
        self.orchestrator = None
        
    def install_packages(self, requirements_file: Optional[str] = None, 
                        hardware_profile: Optional[HardwareProfile] = None,
                        use_advanced_resolver: bool = True) -> bool:
        """Install packages with hardware-specific optimizations."""
        if requirements_file:
            self.requirements_file = Path(requirements_file)
        
        self.logger.info(f"Installing packages from {self.requirements_file}")
        
        if self.progress_reporter:
            self.progress_reporter.update_progress(
                phase=None, progress=0.0, 
                task="Preparing package installation"
            )
        
        try:
            # Get virtual environment Python
            venv_python = self.python_handler.get_venv_python_executable()
            
            # Upgrade pip first
            self.logger.info("Upgrading pip...")
            subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], 
                          check=True, capture_output=True)
            
            if self.progress_reporter:
                self.progress_reporter.update_progress(
                    phase=None, progress=0.1, 
                    task="Pip upgraded"
                )
            
            # Process requirements with hardware-specific modifications
            modified_requirements = self._process_requirements_for_hardware(hardware_profile)
            
            # Choose installation method
            if use_advanced_resolver and hardware_profile:
                # Use advanced package resolver
                self.orchestrator = PackageInstallationOrchestrator(
                    str(self.installation_path), venv_python
                )
                success = self.orchestrator.install_packages_with_resolution(
                    modified_requirements, hardware_profile
                )
            else:
                # Use basic batch installation
                self._install_packages_in_batches(venv_python, modified_requirements)
                success = True
            
            if self.progress_reporter:
                self.progress_reporter.update_progress(
                    phase=None, progress=1.0, 
                    task="Package installation complete"
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Package installation failed: {e}")
            
            # Try fallback installation if advanced resolver failed
            if use_advanced_resolver:
                self.logger.info("Advanced resolver failed, trying basic installation...")
                return self.install_packages(requirements_file, hardware_profile, use_advanced_resolver=False)
            
            raise InstallationError(
                f"Package installation failed: {str(e)}",
                ErrorCategory.SYSTEM,
                ["Check internet connection", "Try installing packages individually", "Check disk space"]
            )
    
    def _process_requirements_for_hardware(self, hardware_profile: Optional[HardwareProfile]) -> List[str]:
        """Process requirements file with hardware-specific package selection."""
        if not self.requirements_file.exists():
            raise InstallationError(
                f"Requirements file not found: {self.requirements_file}",
                ErrorCategory.CONFIGURATION,
                ["Check requirements file path", "Restore requirements file"]
            )
        
        requirements = []
        with open(self.requirements_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
        
        # Add hardware-specific packages
        if hardware_profile and hardware_profile.gpu:
            gpu_requirements = self._get_gpu_specific_packages(hardware_profile.gpu)
            requirements.extend(gpu_requirements)
        
        return requirements
    
    def _get_gpu_specific_packages(self, gpu_info) -> List[str]:
        """Get GPU-specific package requirements."""
        gpu_packages = []
        
        if "NVIDIA" in gpu_info.model or "RTX" in gpu_info.model or "GTX" in gpu_info.model:
            # NVIDIA GPU detected - add CUDA-specific packages
            if gpu_info.cuda_version:
                cuda_major = gpu_info.cuda_version.split('.')[0]
                if cuda_major in ['11', '12']:
                    # Add CUDA-specific torch packages
                    gpu_packages.extend([
                        f"torch --index-url https://download.pytorch.org/whl/cu{cuda_major}8",
                        f"torchvision --index-url https://download.pytorch.org/whl/cu{cuda_major}8",
                        f"torchaudio --index-url https://download.pytorch.org/whl/cu{cuda_major}8"
                    ])
                    
                    # Add xformers for optimization if high-end GPU
                    if gpu_info.vram_gb >= 8:
                        gpu_packages.append("xformers")
        
        elif "AMD" in gpu_info.model or "Radeon" in gpu_info.model:
            # AMD GPU detected - add ROCm packages if available
            gpu_packages.extend([
                "torch --index-url https://download.pytorch.org/whl/rocm5.4.2",
                "torchvision --index-url https://download.pytorch.org/whl/rocm5.4.2"
            ])
        
        return gpu_packages
    
    def _install_packages_in_batches(self, python_exe: str, requirements: List[str]) -> None:
        """Install packages in batches to handle dependencies properly."""
        # Define installation batches
        batches = [
            # Core dependencies first
            ["numpy", "scipy", "pillow"],
            # PyTorch ecosystem
            [req for req in requirements if any(pkg in req.lower() for pkg in ["torch", "torchvision", "torchaudio"])],
            # Hugging Face ecosystem
            [req for req in requirements if any(pkg in req.lower() for pkg in ["transformers", "diffusers", "accelerate", "huggingface-hub"])],
            # Computer vision and utilities
            [req for req in requirements if req not in [req for batch in [
                ["numpy", "scipy", "pillow"],
                [req for req in requirements if any(pkg in req.lower() for pkg in ["torch", "torchvision", "torchaudio"])],
                [req for req in requirements if any(pkg in req.lower() for pkg in ["transformers", "diffusers", "accelerate", "huggingface-hub"])]
            ] for req in batch]]
        ]
        
        total_batches = len([batch for batch in batches if batch])
        current_batch = 0
        
        for batch in batches:
            if not batch:
                continue
                
            current_batch += 1
            self.logger.info(f"Installing batch {current_batch}/{total_batches}: {len(batch)} packages")
            
            if self.progress_reporter:
                progress = 0.1 + (current_batch / total_batches) * 0.8
                self.progress_reporter.update_progress(
                    phase=None, progress=progress,
                    task=f"Installing batch {current_batch}/{total_batches}"
                )
            
            # Install batch with retry mechanism
            self._install_batch_with_retry(python_exe, batch)
    
    def _install_batch_with_retry(self, python_exe: str, batch: List[str], max_retries: int = 3) -> None:
        """Install a batch of packages with retry mechanism."""
        for attempt in range(max_retries):
            try:
                # Prepare pip command
                cmd = [python_exe, "-m", "pip", "install"] + batch
                
                # Add additional flags for better compatibility
                cmd.extend(["--no-cache-dir", "--timeout", "300"])
                
                self.logger.debug(f"Running: {' '.join(cmd)}")
                
                # Run installation
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                self.logger.info(f"Batch installed successfully on attempt {attempt + 1}")
                return
                
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"Batch installation attempt {attempt + 1} failed: {e.stderr}")
                
                if attempt == max_retries - 1:
                    # Last attempt failed, try individual packages
                    self.logger.info("Trying individual package installation...")
                    self._install_packages_individually(python_exe, batch)
                else:
                    # Wait before retry
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
    
    def _install_packages_individually(self, python_exe: str, packages: List[str]) -> None:
        """Install packages individually as fallback."""
        failed_packages = []
        
        for package in packages:
            try:
                cmd = [python_exe, "-m", "pip", "install", package, "--no-cache-dir"]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                self.logger.info(f"Successfully installed: {package}")
                
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to install {package}: {e.stderr}")
                failed_packages.append(package)
        
        if failed_packages:
            self.logger.warning(f"Failed to install packages: {failed_packages}")
            # Don't raise exception for individual failures - log and continue


class DependencyManager(IDependencyManager, BaseInstallationComponent):
    """Main dependency manager that orchestrates Python and package installation."""
    
    def __init__(self, installation_path: str, progress_reporter: Optional[IProgressReporter] = None):
        super().__init__(installation_path)
        self.progress_reporter = progress_reporter
        self.python_handler = PythonInstallationHandler(installation_path, progress_reporter)
        self.package_system = PackageInstallationSystem(installation_path, self.python_handler, progress_reporter)
    
    def check_python_installation(self) -> Dict[str, Any]:
        """Check for existing Python installation."""
        return self.python_handler.check_python_installation()
    
    def install_python(self, target_dir: str) -> bool:
        """Install Python to the target directory."""
        return self.python_handler.install_python(target_dir)
    
    def create_virtual_environment(self, venv_path: str, 
                                 hardware_profile: HardwareProfile) -> bool:
        """Create virtual environment with hardware-optimized settings."""
        return self.python_handler.create_virtual_environment(venv_path, hardware_profile)
    
    def install_packages(self, requirements_file: str, 
                        hardware_profile: HardwareProfile) -> bool:
        """Install packages with hardware-specific optimizations."""
        return self.package_system.install_packages(requirements_file, hardware_profile)
    
    def validate_installation(self) -> ValidationResult:
        """Validate that dependencies are properly installed."""
        try:
            # Check virtual environment
            venv_python = self.python_handler.get_venv_python_executable()
            
            # Test basic imports
            test_imports = [
                "import torch",
                "import torchvision", 
                "import transformers",
                "import diffusers",
                "import numpy",
                "import PIL"
            ]
            
            failed_imports = []
            warnings = []
            
            for import_test in test_imports:
                result = subprocess.run([venv_python, "-c", import_test], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    failed_imports.append(import_test.split()[-1])
            
            # Test CUDA availability if GPU is present
            cuda_test = """
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
"""
            
            try:
                result = subprocess.run([venv_python, "-c", cuda_test], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    cuda_info = result.stdout.strip()
                    if "CUDA available: False" in cuda_info:
                        warnings.append("CUDA not available - using CPU mode")
                    else:
                        self.logger.info(f"CUDA validation: {cuda_info}")
                else:
                    warnings.append("Could not validate CUDA installation")
            except subprocess.TimeoutExpired:
                warnings.append("CUDA validation timed out")
            
            if failed_imports:
                return ValidationResult(
                    success=False,
                    message=f"Failed to import required packages: {', '.join(failed_imports)}",
                    warnings=warnings
                )
            
            return ValidationResult(
                success=True,
                message="All dependencies validated successfully",
                warnings=warnings if warnings else None
            )
            
        except Exception as e:
            return ValidationResult(
                success=False,
                message=f"Dependency validation failed: {str(e)}"
            )
    
    def get_installation_summary(self) -> Dict[str, Any]:
        """Get summary of installed packages and their versions."""
        try:
            venv_python = self.python_handler.get_venv_python_executable()
            
            # Get list of installed packages
            result = subprocess.run([venv_python, "-m", "pip", "list", "--format=json"], 
                                  capture_output=True, text=True, check=True)
            
            import json
            packages = json.loads(result.stdout)
            
            # Categorize packages
            summary = {
                "total_packages": len(packages),
                "pytorch_packages": [],
                "huggingface_packages": [],
                "vision_packages": [],
                "other_packages": []
            }
            
            pytorch_names = {"torch", "torchvision", "torchaudio", "xformers"}
            hf_names = {"transformers", "diffusers", "accelerate", "huggingface-hub"}
            vision_names = {"opencv-python", "pillow", "imageio"}
            
            for pkg in packages:
                name = pkg["name"].lower()
                if name in pytorch_names:
                    summary["pytorch_packages"].append(pkg)
                elif name in hf_names:
                    summary["huggingface_packages"].append(pkg)
                elif name in vision_names:
                    summary["vision_packages"].append(pkg)
                else:
                    summary["other_packages"].append(pkg)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get installation summary: {e}")
            return {"error": str(e)}
