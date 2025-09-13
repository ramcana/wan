"""
Distribution Manager

Handles automated packaging scripts for release preparation,
integrity verification, and cross-system compatibility testing.
"""

import os
import json
import shutil
import subprocess
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import zipfile
import platform

try:
    from interfaces import InstallationError, ErrorCategory
    from base_classes import BaseInstallationComponent
    from installer_packager import InstallerPackager, VersionManager
except ImportError:
    from interfaces import InstallationError, ErrorCategory
    from base_classes import BaseInstallationComponent
    from installer_packager import InstallerPackager, VersionManager


class DistributionManager(BaseInstallationComponent):
    """
    Manages the complete distribution preparation process including
    automated packaging, integrity verification, and compatibility testing.
    """
    
    def __init__(self, source_dir: str = None, output_dir: str = None):
        self.source_dir = Path(source_dir) if source_dir else Path(__file__).parent.parent
        self.output_dir = Path(output_dir) if output_dir else self.source_dir / "dist"
        super().__init__(str(self.source_dir))
        
        self.packager = InstallerPackager(str(self.source_dir), str(self.output_dir))
        self.release_manifest = {}
        
    def prepare_release(self, version: str, release_notes: str = "", 
                       test_compatibility: bool = True) -> Dict[str, Any]:
        """
        Prepare a complete release with all distribution artifacts.
        
        Args:
            version: Release version (e.g., "1.0.0")
            release_notes: Release notes text
            test_compatibility: Whether to run compatibility tests
            
        Returns:
            Dictionary with release information and artifacts
        """
        try:
            self.logger.info(f"Preparing release v{version}")
            
            # Create release directory
            release_dir = self.output_dir / f"release-v{version}"
            release_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate release artifacts
            artifacts = self._generate_release_artifacts(version, release_dir)
            
            # Create release manifest
            manifest = self._create_release_manifest(version, release_notes, artifacts)
            
            # Run compatibility tests if requested
            if test_compatibility:
                compatibility_results = self._run_compatibility_tests(artifacts)
                manifest["compatibility_tests"] = compatibility_results
            
            # Generate distribution checksums
            self._generate_distribution_checksums(release_dir)
            
            # Create release package
            release_package = self._create_release_package(release_dir, version)
            
            # Save release manifest
            manifest_file = release_dir / "release_manifest.json"
            with open(manifest_file, "w") as f:
                json.dump(manifest, f, indent=2)
            
            self.release_manifest = manifest
            
            self.logger.info(f"Release v{version} prepared successfully")
            return {
                "version": version,
                "release_dir": str(release_dir),
                "release_package": str(release_package),
                "manifest": manifest,
                "artifacts": artifacts
            }
            
        except Exception as e:
            self.logger.error(f"Failed to prepare release: {e}")
            raise InstallationError(
                f"Release preparation failed: {str(e)}",
                ErrorCategory.SYSTEM,
                ["Check source files", "Verify build environment", "Review error logs"]
            )
    
    def _generate_release_artifacts(self, version: str, release_dir: Path) -> Dict[str, str]:
        """Generate all release artifacts."""
        self.logger.info("Generating release artifacts...")
        
        artifacts = {}
        
        # Create main installer package
        installer_package = self.packager.create_package(version, "WAN22-Installer")
        installer_name = f"WAN22-Installer-v{version}.zip"
        installer_target = release_dir / installer_name
        shutil.copy2(installer_package, installer_target)
        artifacts["installer"] = str(installer_target)
        
        # Copy integrity files
        integrity_source = Path(installer_package).with_suffix(".integrity.json")
        if integrity_source.exists():
            shutil.copy2(integrity_source, release_dir / f"{installer_name}.integrity.json")
            artifacts["integrity"] = str(release_dir / f"{installer_name}.integrity.json")
        
        checksum_source = Path(installer_package).with_suffix(".zip.sha256")
        if checksum_source.exists():
            shutil.copy2(checksum_source, release_dir / f"{installer_name}.sha256")
            artifacts["checksum"] = str(release_dir / f"{installer_name}.sha256")
        
        # Create source archive
        source_archive = self._create_source_archive(release_dir, version)
        artifacts["source"] = str(source_archive)
        
        # Generate documentation package
        docs_package = self._create_documentation_package(release_dir, version)
        if docs_package:
            artifacts["documentation"] = str(docs_package)
        
        # Create quick-start guide
        quickstart_guide = self._create_quickstart_guide(release_dir, version)
        artifacts["quickstart"] = str(quickstart_guide)
        
        self.logger.info(f"Generated {len(artifacts)} release artifacts")
        return artifacts
    
    def _create_source_archive(self, release_dir: Path, version: str) -> Path:
        """Create source code archive."""
        self.logger.info("Creating source archive...")
        
        source_archive = release_dir / f"WAN22-Source-v{version}.zip"
        
        # Files and directories to include in source archive
        include_patterns = [
            "*.py", "*.bat", "*.md", "*.txt", "*.json", "*.yml", "*.yaml"
        ]
        
        exclude_patterns = [
            "__pycache__", "*.pyc", "*.pyo", ".pytest_cache", 
            "dist", "build", "*.log", "temp", "tmp"
        ]
        
        with zipfile.ZipFile(source_archive, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.source_dir):
                # Filter out excluded directories
                dirs[:] = [d for d in dirs if not any(d.startswith(pattern.rstrip('*')) for pattern in exclude_patterns)]
                
                for file in files:
                    file_path = Path(root) / file
                    
                    # Check if file should be included
                    if any(file_path.match(pattern) for pattern in include_patterns):
                        # Check if file should be excluded
                        if not any(pattern in str(file_path) for pattern in exclude_patterns):
                            arcname = file_path.relative_to(self.source_dir)
                            zipf.write(file_path, arcname)
        
        self.logger.info(f"Source archive created: {source_archive}")
        return source_archive
    
    def _create_documentation_package(self, release_dir: Path, version: str) -> Optional[Path]:
        """Create documentation package."""
        self.logger.info("Creating documentation package...")
        
        docs_files = []
        
        # Collect documentation files
        for pattern in ["*.md", "*.txt"]:
            docs_files.extend(self.source_dir.glob(pattern))
        
        # Look for docs directory
        docs_dir = self.source_dir / "docs"
        if docs_dir.exists():
            for file_path in docs_dir.rglob("*"):
                if file_path.is_file():
                    docs_files.append(file_path)
        
        if not docs_files:
            self.logger.info("No documentation files found")
            return None
        
        docs_archive = release_dir / f"WAN22-Documentation-v{version}.zip"
        
        with zipfile.ZipFile(docs_archive, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for doc_file in docs_files:
                if doc_file.is_file():
                    arcname = doc_file.relative_to(self.source_dir)
                    zipf.write(doc_file, arcname)
        
        self.logger.info(f"Documentation package created: {docs_archive}")
        return docs_archive
    
    def _create_quickstart_guide(self, release_dir: Path, version: str) -> Path:
        """Create quick-start guide."""
        quickstart_content = f"""# WAN2.2 Local Installation - Quick Start Guide

Version: {version}
Release Date: {datetime.now().strftime('%Y-%m-%d')}

## System Requirements

- Windows 10/11 (64-bit)
- 8GB RAM minimum (16GB+ recommended)
- 50GB free disk space
- NVIDIA GPU with 6GB+ VRAM (recommended)

## Installation Steps

1. **Download the installer**
   - Download `WAN22-Installer-v{version}.zip`
   - Verify integrity using the provided `.sha256` file

2. **Extract and run**
   ```
   # Extract the ZIP file
   # Navigate to the extracted folder
   # Double-click install.bat
   ```

3. **Follow the installer**
   - The installer will automatically detect your hardware
   - It will download and configure all necessary components
   - Wait for the installation to complete

4. **Launch the application**
   - Use the desktop shortcut created by the installer
   - Or run the application from the installation directory

## Verification

To verify your download:
```cmd
certutil -hashfile WAN22-Installer-v{version}.zip SHA256
```

Compare the output with the checksum in `WAN22-Installer-v{version}.zip.sha256`

## Troubleshooting

### Common Issues

1. **Python not found**
   - The installer includes Python, no separate installation needed

2. **GPU not detected**
   - Ensure NVIDIA drivers are up to date
   - Check that your GPU has sufficient VRAM

3. **Download failures**
   - Check internet connection
   - Try running as administrator
   - Temporarily disable antivirus

### Getting Help

- Check the full documentation in the installation directory
- Review the troubleshooting guide
- Check system requirements

## What's New in v{version}

- Automated hardware detection and optimization
- Offline installation capability
- Improved error handling and recovery
- Enhanced performance for high-end systems

## Next Steps

After installation:
1. Review the configuration file for your system
2. Test the installation with a simple generation
3. Explore the examples and documentation

---

For detailed documentation, see the full user guide included with the installation.
"""
        
        quickstart_file = release_dir / "QUICKSTART.md"
        quickstart_file.write_text(quickstart_content, encoding='utf-8')
        
        self.logger.info(f"Quick-start guide created: {quickstart_file}")
        return quickstart_file
    
    def _create_release_manifest(self, version: str, release_notes: str, 
                               artifacts: Dict[str, str]) -> Dict[str, Any]:
        """Create comprehensive release manifest."""
        manifest = {
            "version": version,
            "release_date": datetime.now().isoformat(),
            "release_notes": release_notes,
            "build_info": {
                "build_system": platform.system(),
                "build_machine": platform.machine(),
                "python_version": platform.python_version(),
                "build_timestamp": datetime.now().isoformat()
            },
            "artifacts": {},
            "system_requirements": {
                "os": ["Windows 10", "Windows 11"],
                "architecture": ["x64"],
                "min_ram_gb": 8,
                "recommended_ram_gb": 16,
                "min_disk_space_gb": 50,
                "gpu_requirements": {
                    "min_vram_gb": 6,
                    "recommended_vram_gb": 12,
                    "supported_vendors": ["NVIDIA", "AMD"]
                }
            },
            "installation": {
                "installer_file": Path(artifacts.get("installer", "")).name,
                "installation_time_minutes": 15,
                "requires_admin": False,
                "offline_capable": True
            }
        }
        
        # Add artifact information with checksums
        for artifact_type, artifact_path in artifacts.items():
            if Path(artifact_path).exists():
                file_stats = Path(artifact_path).stat()
                checksum = self._calculate_file_checksum(artifact_path)
                
                manifest["artifacts"][artifact_type] = {
                    "filename": Path(artifact_path).name,
                    "size_bytes": file_stats.st_size,
                    "size_mb": round(file_stats.st_size / (1024 * 1024), 2),
                    "sha256": checksum,
                    "created": datetime.fromtimestamp(file_stats.st_ctime).isoformat()
                }
        
        return manifest
    
    def _run_compatibility_tests(self, artifacts: Dict[str, str]) -> Dict[str, Any]:
        """Run cross-system compatibility tests."""
        self.logger.info("Running compatibility tests...")
        
        results = {
            "test_timestamp": datetime.now().isoformat(),
            "test_system": {
                "os": platform.system(),
                "version": platform.version(),
                "architecture": platform.architecture(),
                "machine": platform.machine()
            },
            "tests": {}
        }
        
        # Test 1: Package integrity
        results["tests"]["package_integrity"] = self._test_package_integrity(artifacts)
        
        # Test 2: Archive extraction
        results["tests"]["archive_extraction"] = self._test_archive_extraction(artifacts)
        
        # Test 3: File permissions
        results["tests"]["file_permissions"] = self._test_file_permissions(artifacts)
        
        # Test 4: Path compatibility
        results["tests"]["path_compatibility"] = self._test_path_compatibility(artifacts)
        
        # Test 5: Size validation
        results["tests"]["size_validation"] = self._test_size_validation(artifacts)
        
        # Calculate overall compatibility score
        passed_tests = sum(1 for test in results["tests"].values() if test.get("passed", False))
        total_tests = len(results["tests"])
        results["compatibility_score"] = passed_tests / total_tests if total_tests > 0 else 0
        results["overall_status"] = "PASS" if results["compatibility_score"] >= 0.8 else "FAIL"
        
        self.logger.info(f"Compatibility tests completed: {results['overall_status']} ({passed_tests}/{total_tests})")
        return results
    
    def _test_package_integrity(self, artifacts: Dict[str, str]) -> Dict[str, Any]:
        """Test package integrity verification."""
        try:
            installer_path = artifacts.get("installer")
            if not installer_path or not Path(installer_path).exists():
                return {"passed": False, "error": "Installer package not found"}
            
            # Verify using the packager's integrity check
            is_valid = self.packager.verify_package_integrity(installer_path)
            
            return {
                "passed": is_valid,
                "message": "Package integrity verified" if is_valid else "Package integrity check failed"
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _test_archive_extraction(self, artifacts: Dict[str, str]) -> Dict[str, Any]:
        """Test archive extraction capability."""
        try:
            installer_path = artifacts.get("installer")
            if not installer_path or not Path(installer_path).exists():
                return {"passed": False, "error": "Installer package not found"}
            
            # Test extraction to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                success = self.packager.extract_package(installer_path, temp_dir)
                
                if success:
                    # Verify key files exist
                    extracted_path = Path(temp_dir)
                    required_files = ["install.bat", "version_manifest.json"]
                    missing_files = [f for f in required_files if not (extracted_path / f).exists()]
                    
                    if missing_files:
                        return {
                            "passed": False,
                            "error": f"Missing files after extraction: {missing_files}"
                        }
                    
                    return {"passed": True, "message": "Archive extraction successful"}
                else:
                    return {"passed": False, "error": "Archive extraction failed"}
                    
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _test_file_permissions(self, artifacts: Dict[str, str]) -> Dict[str, Any]:
        """Test file permissions and accessibility."""
        try:
            issues = []
            
            for artifact_type, artifact_path in artifacts.items():
                if Path(artifact_path).exists():
                    file_path = Path(artifact_path)
                    
                    # Test read access
                    if not os.access(file_path, os.R_OK):
                        issues.append(f"{artifact_type}: No read access")
                    
                    # Test if file is executable (for .bat files)
                    if file_path.suffix.lower() == '.bat':
                        if not os.access(file_path, os.X_OK):
                            issues.append(f"{artifact_type}: No execute access")
            
            return {
                "passed": len(issues) == 0,
                "message": "All files accessible" if len(issues) == 0 else f"Permission issues: {issues}",
                "issues": issues
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _test_path_compatibility(self, artifacts: Dict[str, str]) -> Dict[str, Any]:
        """Test path compatibility across systems."""
        try:
            issues = []
            
            for artifact_type, artifact_path in artifacts.items():
                file_path = Path(artifact_path)
                
                # Check for problematic characters
                problematic_chars = ['<', '>', ':', '"', '|', '?', '*']
                if any(char in str(file_path) for char in problematic_chars):
                    issues.append(f"{artifact_type}: Contains problematic characters")
                
                # Check path length (Windows limit is ~260 characters)
                if len(str(file_path)) > 250:
                    issues.append(f"{artifact_type}: Path too long")
                
                # Check for spaces at end of filename
                if file_path.name.endswith(' '):
                    issues.append(f"{artifact_type}: Filename ends with space")
            
            return {
                "passed": len(issues) == 0,
                "message": "All paths compatible" if len(issues) == 0 else f"Path issues: {issues}",
                "issues": issues
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _test_size_validation(self, artifacts: Dict[str, str]) -> Dict[str, Any]:
        """Test file sizes are reasonable."""
        try:
            size_limits = {
                "installer": 500 * 1024 * 1024,  # 500MB max
                "source": 100 * 1024 * 1024,     # 100MB max
                "documentation": 50 * 1024 * 1024  # 50MB max
            }
            
            issues = []
            
            for artifact_type, artifact_path in artifacts.items():
                if Path(artifact_path).exists():
                    file_size = Path(artifact_path).stat().st_size
                    limit = size_limits.get(artifact_type, 1024 * 1024 * 1024)  # 1GB default
                    
                    if file_size > limit:
                        size_mb = file_size / (1024 * 1024)
                        limit_mb = limit / (1024 * 1024)
                        issues.append(f"{artifact_type}: {size_mb:.1f}MB exceeds limit of {limit_mb:.1f}MB")
                    
                    # Check for suspiciously small files
                    if file_size < 1024:  # Less than 1KB
                        issues.append(f"{artifact_type}: File suspiciously small ({file_size} bytes)")
            
            return {
                "passed": len(issues) == 0,
                "message": "All file sizes reasonable" if len(issues) == 0 else f"Size issues: {issues}",
                "issues": issues
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _generate_distribution_checksums(self, release_dir: Path) -> None:
        """Generate checksums for all distribution files."""
        self.logger.info("Generating distribution checksums...")
        
        checksum_file = release_dir / "CHECKSUMS.txt"
        
        with open(checksum_file, "w") as f:
            f.write(f"# WAN2.2 Distribution Checksums\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Algorithm: SHA256\n\n")
            
            for file_path in sorted(release_dir.glob("*")):
                if file_path.is_file() and file_path.name != "CHECKSUMS.txt":
                    checksum = self._calculate_file_checksum(str(file_path))
                    f.write(f"{checksum}  {file_path.name}\n")
        
        self.logger.info(f"Distribution checksums saved to: {checksum_file}")
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _create_release_package(self, release_dir: Path, version: str) -> Path:
        """Create final release package containing all artifacts."""
        self.logger.info("Creating release package...")
        
        release_package = self.output_dir / f"WAN22-Release-v{version}.zip"
        
        with zipfile.ZipFile(release_package, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in release_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(release_dir)
                    zipf.write(file_path, arcname)
        
        self.logger.info(f"Release package created: {release_package}")
        return release_package
    
    def validate_release(self, release_dir: str) -> Dict[str, Any]:
        """
        Validate a prepared release for distribution readiness.
        
        Args:
            release_dir: Path to release directory
            
        Returns:
            Validation results
        """
        try:
            release_path = Path(release_dir)
            if not release_path.exists():
                return {"valid": False, "error": "Release directory not found"}
            
            # Load release manifest
            manifest_file = release_path / "release_manifest.json"
            if not manifest_file.exists():
                return {"valid": False, "error": "Release manifest not found"}
            
            with open(manifest_file, "r") as f:
                manifest = json.load(f)
            
            validation_results = {
                "valid": True,
                "version": manifest.get("version"),
                "validation_timestamp": datetime.now().isoformat(),
                "checks": {}
            }
            
            # Check required artifacts exist
            required_artifacts = ["installer", "quickstart"]
            for artifact in required_artifacts:
                if artifact in manifest.get("artifacts", {}):
                    artifact_file = release_path / manifest["artifacts"][artifact]["filename"]
                    validation_results["checks"][f"{artifact}_exists"] = {
                        "passed": artifact_file.exists(),
                        "message": f"{artifact} artifact present" if artifact_file.exists() else f"{artifact} artifact missing"
                    }
                else:
                    validation_results["checks"][f"{artifact}_exists"] = {
                        "passed": False,
                        "message": f"{artifact} not in manifest"
                    }
            
            # Verify checksums
            checksums_file = release_path / "CHECKSUMS.txt"
            validation_results["checks"]["checksums_file"] = {
                "passed": checksums_file.exists(),
                "message": "Checksums file present" if checksums_file.exists() else "Checksums file missing"
            }
            
            # Check compatibility test results
            if "compatibility_tests" in manifest:
                compat_score = manifest["compatibility_tests"].get("compatibility_score", 0)
                validation_results["checks"]["compatibility"] = {
                    "passed": compat_score >= 0.8,
                    "message": f"Compatibility score: {compat_score:.2f}",
                    "score": compat_score
                }
            
            # Calculate overall validation status
            failed_checks = [check for check in validation_results["checks"].values() if not check.get("passed", False)]
            validation_results["valid"] = len(failed_checks) == 0
            validation_results["failed_checks"] = len(failed_checks)
            validation_results["total_checks"] = len(validation_results["checks"])
            
            return validation_results
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
