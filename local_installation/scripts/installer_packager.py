"""
Installer Packaging System

This module handles the creation of distributable installation packages
with all necessary files, resources, and version management capabilities.
"""

import os
import json
import shutil
import zipfile
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

try:
    from interfaces import PackagingInterface
    from base_classes import BaseInstallationComponent
except ImportError:
    from interfaces import PackagingInterface
    from base_classes import BaseInstallationComponent


class InstallerPackager(BaseInstallationComponent, PackagingInterface):
    """
    Handles creation of distributable installation packages with offline capability
    and version management.
    """
    
    def __init__(self, source_dir: str = None, output_dir: str = None):
        self.source_dir = Path(source_dir) if source_dir else Path(__file__).parent.parent
        self.output_dir = Path(output_dir) if output_dir else self.source_dir / "dist"
        super().__init__(str(self.source_dir))
        self.temp_dir = None
        self.package_manifest = {}
        
    def create_package(self, version: str, package_name: str = "WAN22-Installer") -> str:
        """
        Create a complete installation package with all necessary files.
        
        Args:
            version: Version string (e.g., "1.0.0")
            package_name: Name of the package
            
        Returns:
            Path to created package
        """
        try:
            self.logger.info(f"Creating installation package {package_name} v{version}")
            
            # Create temporary directory for package assembly
            self.temp_dir = Path(tempfile.mkdtemp(prefix="wan22_package_"))
            package_dir = self.temp_dir / package_name
            package_dir.mkdir(parents=True, exist_ok=True)
            
            # Bundle directory structure
            self._bundle_directory_structure(package_dir)
            
            # Embed resources for offline installation
            self._embed_offline_resources(package_dir)
            
            # Create version manifest
            self._create_version_manifest(package_dir, version)
            
            # Create package archive
            package_path = self._create_package_archive(package_dir, package_name, version)
            
            # Generate integrity checksums
            self._generate_integrity_checksums(package_path)
            
            self.logger.info(f"Package created successfully: {package_path}")
            return str(package_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create package: {e}")
            raise
        finally:
            # Cleanup temporary directory
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _bundle_directory_structure(self, package_dir: Path) -> None:
        """Bundle all necessary files into the package directory."""
        self.logger.info("Bundling directory structure...")
        
        # Define core directories and files to include
        core_structure = {
            "scripts": ["*.py", "*.bat"],
            "application": ["*.py"],
            "resources": ["*.json", "*.txt", "*.yml", "*.yaml"],
            "examples": ["*.py", "*.json"],
            "logs": [".gitkeep"]  # Keep directory structure
        }
        
        # Copy core files
        for dir_name, patterns in core_structure.items():
            source_dir = self.source_dir / dir_name
            if source_dir.exists():
                target_dir = package_dir / dir_name
                target_dir.mkdir(parents=True, exist_ok=True)
                
                for pattern in patterns:
                    for file_path in source_dir.glob(pattern):
                        if file_path.is_file():
                            shutil.copy2(file_path, target_dir / file_path.name)
        
        # Copy root level files
        root_files = [
            "install.bat",
            "manage.bat", 
            "README.md",
            "LICENSE",
            "CHANGELOG.md"
        ]
        
        for filename in root_files:
            source_file = self.source_dir / filename
            if source_file.exists():
                shutil.copy2(source_file, package_dir / filename)
        
        # Create models, loras, and logs directories (empty but with .gitkeep)
        for dir_name in ["models", "loras", "logs"]:
            target_dir = package_dir / dir_name
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / ".gitkeep").touch()
        
        self.logger.info("Directory structure bundled successfully")
    
    def _embed_offline_resources(self, package_dir: Path) -> None:
        """Embed resources needed for offline installation."""
        self.logger.info("Embedding offline resources...")
        
        offline_dir = package_dir / "offline_resources"
        offline_dir.mkdir(parents=True, exist_ok=True)
        
        # Embed Python installer (if available)
        self._embed_python_installer(offline_dir)
        
        # Embed critical packages (if available)
        self._embed_critical_packages(offline_dir)
        
        # Create offline resource manifest
        self._create_offline_manifest(offline_dir)
        
        self.logger.info("Offline resources embedded successfully")
    
    def _embed_python_installer(self, offline_dir: Path) -> None:
        """Embed Python installer for offline installation."""
        python_dir = offline_dir / "python"
        python_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Python installer configuration
        python_config = {
            "version": "3.11.7",
            "architecture": ["x64", "x86"],
            "download_urls": {
                "x64": "https://www.python.org/ftp/python/3.11.7/python-3.11.7-embed-amd64.zip",
                "x86": "https://www.python.org/ftp/python/3.11.7/python-3.11.7-embed-win32.zip"
            },
            "checksums": {
                "x64": "sha256:placeholder_checksum_x64",
                "x86": "sha256:placeholder_checksum_x86"
            }
        }
        
        with open(python_dir / "python_config.json", "w") as f:
            json.dump(python_config, f, indent=2)
    
    def _embed_critical_packages(self, offline_dir: Path) -> None:
        """Embed critical Python packages for offline installation."""
        packages_dir = offline_dir / "packages"
        packages_dir.mkdir(parents=True, exist_ok=True)
        
        # Create critical packages list
        critical_packages = {
            "torch": {
                "version": "2.1.0",
                "variants": ["cpu", "cu118", "cu121"],
                "priority": 1
            },
            "transformers": {
                "version": "4.35.0",
                "variants": ["standard"],
                "priority": 2
            },
            "diffusers": {
                "version": "0.24.0",
                "variants": ["standard"],
                "priority": 3
            }
        }
        
        with open(packages_dir / "critical_packages.json", "w") as f:
            json.dump(critical_packages, f, indent=2)
    
    def _create_offline_manifest(self, offline_dir: Path) -> None:
        """Create manifest for offline resources."""
        manifest = {
            "created": datetime.now().isoformat(),
            "resources": {
                "python": {
                    "available": (offline_dir / "python").exists(),
                    "config_file": "python/python_config.json"
                },
                "packages": {
                    "available": (offline_dir / "packages").exists(),
                    "config_file": "packages/critical_packages.json"
                }
            },
            "installation_modes": {
                "online": "Download resources during installation",
                "offline": "Use embedded resources only",
                "hybrid": "Use embedded resources with online fallback"
            }
        }
        
        with open(offline_dir / "offline_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    
    def _create_version_manifest(self, package_dir: Path, version: str) -> None:
        """Create version manifest with update mechanisms."""
        self.logger.info(f"Creating version manifest for v{version}")
        
        manifest = {
            "version": version,
            "build_date": datetime.now().isoformat(),
            "build_info": {
                "python_version": "3.11+",
                "supported_os": ["Windows 10", "Windows 11"],
                "architecture": ["x64", "x86"]
            },
            "components": self._get_component_versions(),
            "update_info": {
                "update_server": "https://api.github.com/repos/wan22/installer/releases",
                "update_check_interval": 86400,  # 24 hours
                "auto_update": False
            },
            "compatibility": {
                "min_version": "1.0.0",
                "max_version": None,
                "breaking_changes": []
            }
        }
        
        with open(package_dir / "version_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        self.package_manifest = manifest
    
    def _get_component_versions(self) -> Dict[str, str]:
        """Get versions of all components."""
        return {
            "installer": "1.0.0",
            "system_detector": "1.0.0",
            "dependency_manager": "1.0.0",
            "model_downloader": "1.0.0",
            "config_generator": "1.0.0",
            "validator": "1.0.0"
        }
    
    def _create_package_archive(self, package_dir: Path, package_name: str, version: str) -> Path:
        """Create the final package archive."""
        self.logger.info("Creating package archive...")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create archive filename
        archive_name = f"{package_name}-v{version}.zip"
        archive_path = self.output_dir / archive_name
        
        # Create ZIP archive
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(package_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(package_dir)
                    zipf.write(file_path, arcname)
        
        self.logger.info(f"Package archive created: {archive_path}")
        return archive_path
    
    def _generate_integrity_checksums(self, package_path: Path) -> None:
        """Generate integrity checksums for the package."""
        self.logger.info("Generating integrity checksums...")
        
        # Calculate SHA256 checksum
        sha256_hash = hashlib.sha256()
        with open(package_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        checksum = sha256_hash.hexdigest()
        
        # Create checksum file
        checksum_file = package_path.with_suffix(package_path.suffix + ".sha256")
        with open(checksum_file, "w") as f:
            f.write(f"{checksum}  {package_path.name}\n")
        
        # Create integrity manifest
        integrity_manifest = {
            "package_file": package_path.name,
            "sha256": checksum,
            "size_bytes": package_path.stat().st_size,
            "created": datetime.now().isoformat(),
            "verification_instructions": [
                f"certutil -hashfile {package_path.name} SHA256",
                f"Compare output with: {checksum}"
            ]
        }
        
        integrity_file = package_path.with_suffix(".integrity.json")
        with open(integrity_file, "w") as f:
            json.dump(integrity_manifest, f, indent=2)
        
        self.logger.info(f"Integrity checksums generated: {checksum}")
    
    def verify_package_integrity(self, package_path: str) -> bool:
        """
        Verify the integrity of a package.
        
        Args:
            package_path: Path to the package file
            
        Returns:
            True if package is valid, False otherwise
        """
        try:
            package_path = Path(package_path)
            integrity_file = package_path.with_suffix(".integrity.json")
            
            if not integrity_file.exists():
                self.logger.error("Integrity file not found")
                return False
            
            # Load integrity manifest
            with open(integrity_file, "r") as f:
                integrity_data = json.load(f)
            
            # Calculate current checksum
            sha256_hash = hashlib.sha256()
            with open(package_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            current_checksum = sha256_hash.hexdigest()
            expected_checksum = integrity_data["sha256"]
            
            if current_checksum == expected_checksum:
                self.logger.info("Package integrity verified successfully")
                return True
            else:
                self.logger.error(f"Integrity check failed. Expected: {expected_checksum}, Got: {current_checksum}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to verify package integrity: {e}")
            return False
    
    def extract_package(self, package_path: str, extract_dir: str) -> bool:
        """
        Extract a package to the specified directory.
        
        Args:
            package_path: Path to the package file
            extract_dir: Directory to extract to
            
        Returns:
            True if extraction successful, False otherwise
        """
        try:
            package_path = Path(package_path)
            extract_dir = Path(extract_dir)
            
            # Verify package integrity first
            if not self.verify_package_integrity(str(package_path)):
                return False
            
            # Extract package
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(package_path, 'r') as zipf:
                zipf.extractall(extract_dir)
            
            self.logger.info(f"Package extracted successfully to: {extract_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to extract package: {e}")
            return False


class VersionManager(BaseInstallationComponent):
    """
    Handles version management and update mechanisms.
    """
    
    def __init__(self, installation_dir: str = None):
        self.installation_dir = Path(installation_dir) if installation_dir else Path.cwd()
        super().__init__(str(self.installation_dir))
        self.version_file = self.installation_dir / "version_manifest.json"
    
    def get_current_version(self) -> Optional[str]:
        """Get the current installed version."""
        try:
            if self.version_file.exists():
                with open(self.version_file, "r") as f:
                    manifest = json.load(f)
                return manifest.get("version")
            return None
        except Exception as e:
            self.logger.error(f"Failed to get current version: {e}")
            return None
    
    def check_for_updates(self) -> Dict:
        """
        Check for available updates.
        
        Returns:
            Dictionary with update information
        """
        try:
            current_version = self.get_current_version()
            if not current_version:
                return {"error": "Current version not found"}
            
            # In a real implementation, this would check GitHub releases API
            # For now, return mock data
            return {
                "current_version": current_version,
                "latest_version": "1.1.0",
                "update_available": True,
                "download_url": "https://github.com/wan22/installer/releases/download/v1.1.0/WAN22-Installer-v1.1.0.zip",
                "release_notes": "Bug fixes and performance improvements",
                "size_mb": 150.5
            }
            
        except Exception as e:
            self.logger.error(f"Failed to check for updates: {e}")
            return {"error": str(e)}
    
    def create_backup(self, backup_name: str = None) -> str:
        """
        Create a backup of the current installation.
        
        Args:
            backup_name: Optional backup name
            
        Returns:
            Path to backup directory
        """
        try:
            if not backup_name:
                backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            backup_dir = self.installation_dir / "backups" / backup_name
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy critical files
            critical_files = [
                "version_manifest.json",
                "config.json",
                "resources/default_config.json"
            ]
            
            for file_path in critical_files:
                source = self.installation_dir / file_path
                if source.exists():
                    target = backup_dir / file_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            
            self.logger.info(f"Backup created: {backup_dir}")
            return str(backup_dir)
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            raise
    
    def restore_backup(self, backup_name: str) -> bool:
        """
        Restore from a backup.
        
        Args:
            backup_name: Name of the backup to restore
            
        Returns:
            True if restore successful, False otherwise
        """
        try:
            backup_dir = self.installation_dir / "backups" / backup_name
            if not backup_dir.exists():
                self.logger.error(f"Backup not found: {backup_name}")
                return False
            
            # Restore files
            for backup_file in backup_dir.rglob("*"):
                if backup_file.is_file():
                    relative_path = backup_file.relative_to(backup_dir)
                    target_file = self.installation_dir / relative_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_file, target_file)
            
            self.logger.info(f"Backup restored: {backup_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {e}")
            return False