"""
Advanced package resolution and conflict management for WAN2.2 installation.
Handles CUDA-aware package selection, dependency conflicts, and automatic retry mechanisms.
"""

import re
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from packaging import version
from packaging.requirements import Requirement

from interfaces import HardwareProfile, GPUInfo, InstallationError, ErrorCategory
from base_classes import BaseInstallationComponent


@dataclass
class PackageInfo:
    """Information about a package and its requirements."""
    name: str
    version: Optional[str]
    source: Optional[str]  # PyPI, custom index, etc.
    dependencies: List[str]
    conflicts: List[str]
    hardware_requirements: Optional[Dict[str, str]]


@dataclass
class ConflictResolution:
    """Resolution for a package conflict."""
    conflicting_packages: List[str]
    resolution_strategy: str  # "upgrade", "downgrade", "exclude", "alternative"
    recommended_versions: Dict[str, str]
    reason: str


class CUDAPackageSelector:
    """Selects appropriate CUDA packages based on hardware detection."""
    
    # CUDA compatibility matrix
    CUDA_COMPATIBILITY = {
        "11.8": {
            "torch_index": "https://download.pytorch.org/whl/cu118",
            "supported_gpus": ["RTX 30", "RTX 40", "GTX 16", "Tesla", "Quadro"],
            "min_driver": "450.80.02"
        },
        "12.1": {
            "torch_index": "https://download.pytorch.org/whl/cu121", 
            "supported_gpus": ["RTX 40", "RTX 30", "Tesla H100", "A100"],
            "min_driver": "527.41"
        },
        "12.4": {
            "torch_index": "https://download.pytorch.org/whl/cu124",
            "supported_gpus": ["RTX 40", "RTX 50", "H100", "A100"],
            "min_driver": "550.54.15"
        }
    }
    
    # Package alternatives for different CUDA versions
    CUDA_PACKAGES = {
        "torch": {
            "11.8": "torch==2.1.0+cu118",
            "12.1": "torch==2.1.0+cu121", 
            "12.4": "torch==2.4.0+cu124"
        },
        "torchvision": {
            "11.8": "torchvision==0.16.0+cu118",
            "12.1": "torchvision==0.16.0+cu121",
            "12.4": "torchvision==0.19.0+cu124"
        },
        "torchaudio": {
            "11.8": "torchaudio==2.1.0+cu118",
            "12.1": "torchaudio==2.1.0+cu121",
            "12.4": "torchaudio==2.4.0+cu124"
        }
    }
    
    def __init__(self, gpu_info: Optional[GPUInfo] = None):
        self.gpu_info = gpu_info
        
    def select_cuda_version(self) -> Optional[str]:
        """Select the best CUDA version for the detected GPU."""
        if not self.gpu_info or not self.gpu_info.cuda_version:
            return None
            
        detected_cuda = self.gpu_info.cuda_version
        gpu_model = self.gpu_info.model
        
        # Parse detected CUDA version
        try:
            detected_major_minor = '.'.join(detected_cuda.split('.')[:2])
        except (AttributeError, IndexError):
            return None
        
        # Find best compatible CUDA version
        compatible_versions = []
        for cuda_ver, info in self.CUDA_COMPATIBILITY.items():
            if any(gpu_series in gpu_model for gpu_series in info["supported_gpus"]):
                if version.parse(cuda_ver) <= version.parse(detected_cuda):
                    compatible_versions.append(cuda_ver)
        
        # Return the highest compatible version
        if compatible_versions:
            return max(compatible_versions, key=lambda x: version.parse(x))
        
        # Fallback to most common version
        return "11.8"
    
    def get_cuda_packages(self, cuda_version: str) -> Dict[str, str]:
        """Get CUDA-specific package specifications."""
        packages = {}
        
        for package_name, versions in self.CUDA_PACKAGES.items():
            if cuda_version in versions:
                packages[package_name] = versions[cuda_version]
            else:
                # Fallback to closest version
                available_versions = list(versions.keys())
                closest_version = min(available_versions, 
                                    key=lambda x: abs(version.parse(x).major - version.parse(cuda_version).major))
                packages[package_name] = versions[closest_version]
        
        return packages
    
    def get_torch_index_url(self, cuda_version: str) -> Optional[str]:
        """Get the PyTorch index URL for the CUDA version."""
        return self.CUDA_COMPATIBILITY.get(cuda_version, {}).get("torch_index")


class DependencyResolver:
    """Resolves package dependencies and conflicts."""
    
    def __init__(self, python_exe: str):
        self.python_exe = python_exe
        self.package_cache = {}
        
    def analyze_requirements(self, requirements: List[str]) -> Dict[str, PackageInfo]:
        """Analyze requirements and build dependency graph."""
        packages = {}
        
        for req_line in requirements:
            if not req_line.strip() or req_line.strip().startswith('#'):
                continue
                
            try:
                # Parse requirement
                if '--index-url' in req_line:
                    # Handle custom index URLs
                    parts = req_line.split()
                    package_spec = parts[0]
                    index_url = None
                    for i, part in enumerate(parts):
                        if part == '--index-url' and i + 1 < len(parts):
                            index_url = parts[i + 1]
                            break
                else:
                    package_spec = req_line.strip()
                    index_url = None
                
                req = Requirement(package_spec)
                
                package_info = PackageInfo(
                    name=req.name,
                    version=str(req.specifier) if req.specifier else None,
                    source=index_url,
                    dependencies=[],
                    conflicts=[],
                    hardware_requirements=None
                )
                
                packages[req.name] = package_info
                
            except Exception as e:
                # Skip invalid requirements
                continue
        
        return packages
    
    def detect_conflicts(self, packages: Dict[str, PackageInfo]) -> List[ConflictResolution]:
        """Detect potential package conflicts."""
        conflicts = []
        
        # Check for known conflicting packages
        known_conflicts = {
            ("torch", "tensorflow"): "Both are deep learning frameworks - choose one",
            ("opencv-python", "opencv-contrib-python"): "Use opencv-contrib-python for full features",
            ("pillow", "PIL"): "PIL is deprecated, use Pillow instead"
        }
        
        package_names = set(packages.keys())
        
        for (pkg1, pkg2), reason in known_conflicts.items():
            if pkg1 in package_names and pkg2 in package_names:
                conflicts.append(ConflictResolution(
                    conflicting_packages=[pkg1, pkg2],
                    resolution_strategy="exclude",
                    recommended_versions={pkg1: packages[pkg1].version or "latest"},
                    reason=reason
                ))
        
        # Check version conflicts
        conflicts.extend(self._check_version_conflicts(packages))
        
        return conflicts
    
    def _check_version_conflicts(self, packages: Dict[str, PackageInfo]) -> List[ConflictResolution]:
        """Check for version conflicts between packages."""
        conflicts = []
        
        # Known version compatibility issues
        version_conflicts = {
            "torch": {
                "transformers": {
                    "2.0.0": ">=4.21.0",
                    "2.1.0": ">=4.30.0"
                }
            }
        }
        
        for base_pkg, dependent_pkgs in version_conflicts.items():
            if base_pkg not in packages:
                continue
                
            base_version = packages[base_pkg].version
            if not base_version:
                continue
                
            for dep_pkg, version_reqs in dependent_pkgs.items():
                if dep_pkg not in packages:
                    continue
                    
                for base_ver, required_dep_ver in version_reqs.items():
                    if base_version.startswith(base_ver):
                        conflicts.append(ConflictResolution(
                            conflicting_packages=[base_pkg, dep_pkg],
                            resolution_strategy="upgrade",
                            recommended_versions={dep_pkg: required_dep_ver},
                            reason=f"{base_pkg} {base_version} requires {dep_pkg} {required_dep_ver}"
                        ))
        
        return conflicts
    
    def resolve_conflicts(self, packages: Dict[str, PackageInfo], 
                         conflicts: List[ConflictResolution]) -> Dict[str, PackageInfo]:
        """Resolve detected conflicts."""
        resolved_packages = packages.copy()
        
        for conflict in conflicts:
            if conflict.resolution_strategy == "exclude":
                # Remove less preferred package
                for pkg in conflict.conflicting_packages[1:]:
                    if pkg in resolved_packages:
                        del resolved_packages[pkg]
                        
            elif conflict.resolution_strategy == "upgrade":
                # Upgrade packages to recommended versions
                for pkg, version in conflict.recommended_versions.items():
                    if pkg in resolved_packages:
                        resolved_packages[pkg].version = version
        
        return resolved_packages


class PackageInstallationOrchestrator(BaseInstallationComponent):
    """Orchestrates advanced package installation with conflict resolution."""
    
    def __init__(self, installation_path: str, python_exe: str):
        super().__init__(installation_path)
        self.python_exe = python_exe
        self.cuda_selector = None
        self.dependency_resolver = DependencyResolver(python_exe)
        
    def install_packages_with_resolution(self, requirements: List[str], 
                                       hardware_profile: Optional[HardwareProfile] = None,
                                       max_retries: int = 3) -> bool:
        """Install packages with advanced conflict resolution and hardware optimization."""
        
        # Initialize CUDA selector if GPU is available
        if hardware_profile and hardware_profile.gpu:
            self.cuda_selector = CUDAPackageSelector(hardware_profile.gpu)
        
        try:
            # Step 1: Analyze requirements
            self.logger.info("Analyzing package requirements...")
            packages = self.dependency_resolver.analyze_requirements(requirements)
            
            # Step 2: Apply hardware-specific package selection
            if self.cuda_selector:
                packages = self._apply_cuda_optimizations(packages)
            
            # Step 3: Detect conflicts
            self.logger.info("Detecting package conflicts...")
            conflicts = self.dependency_resolver.detect_conflicts(packages)
            
            if conflicts:
                self.logger.info(f"Found {len(conflicts)} potential conflicts")
                for conflict in conflicts:
                    self.logger.info(f"Conflict: {conflict.reason}")
            
            # Step 4: Resolve conflicts
            resolved_packages = self.dependency_resolver.resolve_conflicts(packages, conflicts)
            
            # Step 5: Install packages with retry mechanism
            return self._install_with_retry(resolved_packages, max_retries)
            
        except Exception as e:
            self.logger.error(f"Package installation orchestration failed: {e}")
            raise InstallationError(
                f"Advanced package installation failed: {str(e)}",
                ErrorCategory.SYSTEM,
                ["Try basic installation mode", "Check package compatibility", "Update pip"]
            )
    
    def _apply_cuda_optimizations(self, packages: Dict[str, PackageInfo]) -> Dict[str, PackageInfo]:
        """Apply CUDA-specific package optimizations."""
        cuda_version = self.cuda_selector.select_cuda_version()
        if not cuda_version:
            self.logger.info("No CUDA optimization applied - using CPU versions")
            return packages
        
        self.logger.info(f"Applying CUDA {cuda_version} optimizations")
        
        # Get CUDA-specific packages
        cuda_packages = self.cuda_selector.get_cuda_packages(cuda_version)
        torch_index = self.cuda_selector.get_torch_index_url(cuda_version)
        
        # Replace PyTorch packages with CUDA versions
        for pkg_name, cuda_spec in cuda_packages.items():
            if pkg_name in packages:
                # Parse CUDA package specification
                if '==' in cuda_spec:
                    name, version_spec = cuda_spec.split('==', 1)
                    packages[pkg_name].version = f"=={version_spec}"
                    packages[pkg_name].source = torch_index
                
        # Add xformers for high-end GPUs
        if (self.cuda_selector.gpu_info.vram_gb >= 8 and 
            "RTX" in self.cuda_selector.gpu_info.model):
            packages["xformers"] = PackageInfo(
                name="xformers",
                version=None,
                source=None,
                dependencies=[],
                conflicts=[],
                hardware_requirements={"min_vram_gb": "8"}
            )
        
        return packages
    
    def _install_with_retry(self, packages: Dict[str, PackageInfo], max_retries: int) -> bool:
        """Install packages with retry mechanism and fallback strategies."""
        
        # Group packages by installation strategy
        installation_groups = self._group_packages_for_installation(packages)
        
        for group_name, group_packages in installation_groups.items():
            self.logger.info(f"Installing {group_name} packages...")
            
            success = False
            for attempt in range(max_retries):
                try:
                    self._install_package_group(group_packages)
                    success = True
                    break
                    
                except Exception as e:
                    self.logger.warning(f"Installation attempt {attempt + 1} failed: {e}")
                    
                    if attempt < max_retries - 1:
                        # Apply fallback strategy
                        group_packages = self._apply_fallback_strategy(group_packages, e)
                        import time
                        time.sleep(2 ** attempt)  # Exponential backoff
            
            if not success:
                self.logger.error(f"Failed to install {group_name} packages after {max_retries} attempts")
                # Continue with other groups instead of failing completely
        
        return True
    
    def _group_packages_for_installation(self, packages: Dict[str, PackageInfo]) -> Dict[str, List[PackageInfo]]:
        """Group packages by installation strategy."""
        groups = {
            "core": [],
            "pytorch": [],
            "huggingface": [],
            "vision": [],
            "utilities": []
        }
        
        pytorch_packages = {"torch", "torchvision", "torchaudio", "xformers"}
        huggingface_packages = {"transformers", "diffusers", "accelerate", "huggingface-hub"}
        vision_packages = {"opencv-python", "pillow", "imageio"}
        core_packages = {"numpy", "scipy"}
        
        for package in packages.values():
            if package.name in core_packages:
                groups["core"].append(package)
            elif package.name in pytorch_packages:
                groups["pytorch"].append(package)
            elif package.name in huggingface_packages:
                groups["huggingface"].append(package)
            elif package.name in vision_packages:
                groups["vision"].append(package)
            else:
                groups["utilities"].append(package)
        
        return {k: v for k, v in groups.items() if v}  # Remove empty groups
    
    def _install_package_group(self, packages: List[PackageInfo]) -> None:
        """Install a group of packages."""
        if not packages:
            return
        
        # Build pip command
        cmd = [self.python_exe, "-m", "pip", "install"]
        
        # Add packages with their specifications
        for package in packages:
            if package.source:
                # Add index URL if specified
                cmd.extend(["--index-url", package.source])
            
            package_spec = package.name
            if package.version:
                package_spec += package.version
            
            cmd.append(package_spec)
        
        # Add common flags
        cmd.extend(["--no-cache-dir", "--timeout", "300", "--retries", "3"])
        
        # Execute installation
        self.logger.debug(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        self.logger.info(f"Successfully installed {len(packages)} packages")
    
    def _apply_fallback_strategy(self, packages: List[PackageInfo], error: Exception) -> List[PackageInfo]:
        """Apply fallback strategy when installation fails."""
        error_str = str(error).lower()
        
        # If CUDA installation fails, fallback to CPU versions
        if "cuda" in error_str or "torch" in error_str:
            self.logger.info("CUDA installation failed, falling back to CPU versions")
            for package in packages:
                if package.name in ["torch", "torchvision", "torchaudio"]:
                    package.source = None  # Use default PyPI
                    package.version = None  # Use latest compatible version
        
        # If specific version fails, try without version constraint
        elif "version" in error_str or "no matching distribution" in error_str:
            self.logger.info("Version constraint failed, trying without version specification")
            for package in packages:
                package.version = None
        
        return packages
    
    def validate_installation(self, packages: Dict[str, PackageInfo]) -> bool:
        """Validate that all packages were installed correctly."""
        self.logger.info("Validating package installation...")
        
        failed_packages = []
        
        for package_name in packages.keys():
            try:
                # Try to import the package
                result = subprocess.run([
                    self.python_exe, "-c", f"import {package_name}"
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    failed_packages.append(package_name)
                    
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Import test for {package_name} timed out")
            except Exception as e:
                self.logger.warning(f"Could not test import for {package_name}: {e}")
        
        if failed_packages:
            self.logger.error(f"Failed to validate packages: {failed_packages}")
            return False
        
        self.logger.info("All packages validated successfully")
        return True