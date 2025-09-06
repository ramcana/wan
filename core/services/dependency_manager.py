"""
Dependency Manager for Wan Model Compatibility System

This module handles remote code fetching, dependency management, and security validation
for custom pipeline code required by Wan models.
"""

import os
import json
import hashlib
import subprocess
import tempfile
import shutil
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


@dataclass
class RemoteCodeStatus:
    """Status of remote pipeline code availability"""
    is_available: bool
    source_url: Optional[str] = None
    version: Optional[str] = None
    security_hash: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class FetchResult:
    """Result of pipeline code fetching operation"""
    success: bool
    code_path: Optional[str] = None
    version: Optional[str] = None
    error_message: Optional[str] = None
    fallback_options: List[str] = None


@dataclass
class VersionCompatibility:
    """Compatibility status between local code and model version"""
    is_compatible: bool
    local_version: Optional[str] = None
    required_version: Optional[str] = None
    compatibility_score: float = 0.0
    warnings: List[str] = None
    recommendations: List[str] = None


@dataclass
class InstallationResult:
    """Result of dependency installation"""
    success: bool
    installed_packages: List[str] = None
    failed_packages: List[str] = None
    error_messages: List[str] = None
    installation_log: Optional[str] = None


@dataclass
class SecurityValidation:
    """Security validation result for remote code"""
    is_safe: bool
    risk_level: str  # "low", "medium", "high"
    detected_risks: List[str] = None
    security_warnings: List[str] = None
    mitigation_suggestions: List[str] = None


class DependencyManager:
    """
    Manages dependencies and remote code for custom Wan pipelines.
    
    Handles:
    - Remote pipeline code fetching from Hugging Face
    - Security validation and trust management
    - Version compatibility checking
    - Dependency installation and management
    - Fallback strategies for missing code
    """
    
    def __init__(self, cache_dir: str = None, trust_mode: str = "safe"):
        """
        Initialize DependencyManager.
        
        Args:
            cache_dir: Directory for caching downloaded code
            trust_mode: Security mode - "safe" or "trust"
        """
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.cache/wan_pipelines"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.trust_mode = trust_mode
        self.trusted_sources = {
            "huggingface.co",
            "hf.co", 
            "github.com/Wan-AI"
        }
        
        # Known pipeline repositories and their metadata
        self.known_pipelines = {
            "WanPipeline": {
                "repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                "file": "pipeline_wan.py",
                "class": "WanPipeline",
                "min_version": "0.1.0"
            }
        }
        
        logger.info(f"DependencyManager initialized with cache_dir={self.cache_dir}, trust_mode={self.trust_mode}")

    def check_remote_code_availability(self, model_path: str) -> RemoteCodeStatus:
        """
        Check if remote pipeline code is available for the model.
        
        Args:
            model_path: Path or identifier of the model
            
        Returns:
            RemoteCodeStatus with availability information
        """
        try:
            logger.info(f"Checking remote code availability for model: {model_path}")
            
            # Extract model identifier from path
            model_id = self._extract_model_id(model_path)
            
            # Check local cache first
            cached_info = self._check_cached_code(model_id)
            if cached_info:
                return RemoteCodeStatus(
                    is_available=True,
                    source_url=cached_info.get("source_url"),
                    version=cached_info.get("version"),
                    security_hash=cached_info.get("security_hash")
                )
            
            # Check if it's a known Hugging Face model
            if self._is_huggingface_model(model_id):
                return self._check_hf_remote_code(model_id)
            
            return RemoteCodeStatus(
                is_available=False,
                error_message=f"No remote code found for model: {model_id}"
            )
            
        except Exception as e:
            logger.error(f"Error checking remote code availability: {e}")
            return RemoteCodeStatus(
                is_available=False,
                error_message=f"Failed to check remote code: {str(e)}"
            )

    def fetch_pipeline_code(self, model_path: str, trust_remote_code: bool = True) -> FetchResult:
        """
        Attempt to fetch missing pipeline code from remote sources.
        
        Args:
            model_path: Path or identifier of the model
            trust_remote_code: Whether to allow remote code execution
            
        Returns:
            FetchResult with fetch operation details
        """
        try:
            logger.info(f"Fetching pipeline code for model: {model_path}")
            
            if not trust_remote_code:
                return FetchResult(
                    success=False,
                    error_message="Remote code fetching disabled (trust_remote_code=False)",
                    fallback_options=self._get_manual_installation_options(model_path)
                )
            
            model_id = self._extract_model_id(model_path)
            
            # Check security validation first
            security_result = self._validate_source_security(model_id)
            if not security_result.is_safe and self.trust_mode == "safe":
                return FetchResult(
                    success=False,
                    error_message=f"Security validation failed: {security_result.detected_risks}",
                    fallback_options=self._get_manual_installation_options(model_path)
                )
            
            # Try to fetch from Hugging Face
            if self._is_huggingface_model(model_id):
                return self._fetch_from_huggingface(model_id)
            
            # Try to fetch from known pipeline repositories
            return self._fetch_from_known_repos(model_id)
            
        except Exception as e:
            logger.error(f"Error fetching pipeline code: {e}")
            return FetchResult(
                success=False,
                error_message=f"Failed to fetch pipeline code: {str(e)}",
                fallback_options=self._get_manual_installation_options(model_path)
            )

    def validate_code_version(self, local_code: str, model_version: str) -> VersionCompatibility:
        """
        Validate compatibility between local code and model version.
        
        Args:
            local_code: Path to local pipeline code
            model_version: Version of the model
            
        Returns:
            VersionCompatibility with compatibility assessment
        """
        try:
            logger.info(f"Validating code version compatibility: {local_code} vs {model_version}")
            
            # Extract version from local code
            local_version = self._extract_code_version(local_code)
            
            # Perform compatibility check
            compatibility_score = self._calculate_compatibility_score(local_version, model_version)
            
            warnings = []
            recommendations = []
            
            if compatibility_score < 0.8:
                warnings.append(f"Version mismatch detected: local={local_version}, required={model_version}")
                recommendations.append("Consider updating pipeline code to match model version")
            
            if compatibility_score < 0.5:
                recommendations.append("Manual verification recommended before use")
            
            return VersionCompatibility(
                is_compatible=compatibility_score >= 0.7,
                local_version=local_version,
                required_version=model_version,
                compatibility_score=compatibility_score,
                warnings=warnings,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error validating code version: {e}")
            return VersionCompatibility(
                is_compatible=False,
                local_version=None,
                required_version=model_version,
                compatibility_score=0.0,
                warnings=[f"Version validation failed: {str(e)}"],
                recommendations=["Manual verification required"]
            )

    def install_dependencies(self, requirements: List[str]) -> InstallationResult:
        """
        Install missing dependencies for custom pipelines.
        
        Args:
            requirements: List of package requirements (e.g., ["torch>=2.0.0", "transformers"])
            
        Returns:
            InstallationResult with installation details
        """
        try:
            logger.info(f"Installing dependencies: {requirements}")
            
            installed_packages = []
            failed_packages = []
            error_messages = []
            installation_log = []
            
            for requirement in requirements:
                try:
                    # Check if package is already installed
                    if self._is_package_installed(requirement):
                        logger.info(f"Package already installed: {requirement}")
                        installed_packages.append(requirement)
                        continue
                    
                    # Install package using pip
                    result = self._install_package(requirement)
                    installation_log.append(result["log"])
                    
                    if result["success"]:
                        installed_packages.append(requirement)
                        logger.info(f"Successfully installed: {requirement}")
                    else:
                        failed_packages.append(requirement)
                        error_messages.append(result["error"])
                        logger.error(f"Failed to install {requirement}: {result['error']}")
                        
                except Exception as e:
                    failed_packages.append(requirement)
                    error_messages.append(str(e))
                    logger.error(f"Exception installing {requirement}: {e}")
            
            success = len(failed_packages) == 0
            
            return InstallationResult(
                success=success,
                installed_packages=installed_packages,
                failed_packages=failed_packages,
                error_messages=error_messages,
                installation_log="\n".join(installation_log)
            )
            
        except Exception as e:
            logger.error(f"Error installing dependencies: {e}")
            return InstallationResult(
                success=False,
                installed_packages=[],
                failed_packages=requirements,
                error_messages=[f"Installation failed: {str(e)}"],
                installation_log=None
            )

    def get_fallback_options(self, model_path: str) -> List[str]:
        """
        Get fallback options when remote code fetching fails.
        
        Args:
            model_path: Path or identifier of the model
            
        Returns:
            List of fallback option descriptions
        """
        model_id = self._extract_model_id(model_path)
        
        options = [
            f"Manual installation: Download pipeline code from {model_id} repository",
            "Use local pipeline: Place custom pipeline in local directory",
            "Alternative models: Try compatible models with standard pipelines",
            "Community support: Check forums for user-provided solutions"
        ]
        
        # Add specific instructions for known models
        if "Wan" in model_id:
            options.extend([
                "Install wan-pipeline package: pip install wan-pipeline",
                "Clone repository: git clone https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                "Download pipeline_wan.py manually from model repository"
            ])
        
        return options

    def _extract_model_id(self, model_path: str) -> str:
        """Extract model identifier from path or URL."""
        # Handle Hugging Face model IDs first (contains / but doesn't start with /)
        if "/" in model_path and not model_path.startswith("/"):
            return model_path
        
        # Handle local paths
        if os.path.exists(model_path) or model_path.startswith("/"):
            return os.path.basename(model_path)
        
        return model_path

    def _is_huggingface_model(self, model_id: str) -> bool:
        """Check if model ID represents a Hugging Face model."""
        if "/" not in model_id or model_id.startswith("/"):
            return False
        
        # Check if it looks like a domain/path (has dots in the first part)
        first_part = model_id.split("/")[0]
        if "." in first_part and not any(trusted in first_part for trusted in ["huggingface", "hf"]):
            # This looks like a domain/path, not a HF model ID
            return False
        
        return True

    def _check_hf_remote_code(self, model_id: str) -> RemoteCodeStatus:
        """Check remote code availability on Hugging Face."""
        try:
            # Check if model exists and has custom pipeline code
            api_url = f"https://huggingface.co/api/models/{model_id}"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                model_info = response.json()
                
                # Check for pipeline files
                files = model_info.get("siblings", [])
                pipeline_files = [f for f in files if f.get("rfilename", "").startswith("pipeline_")]
                
                if pipeline_files:
                    return RemoteCodeStatus(
                        is_available=True,
                        source_url=f"https://huggingface.co/{model_id}",
                        version=model_info.get("sha", "unknown")
                    )
            
            return RemoteCodeStatus(
                is_available=False,
                error_message=f"No pipeline code found in {model_id}"
            )
            
        except Exception as e:
            return RemoteCodeStatus(
                is_available=False,
                error_message=f"Failed to check Hugging Face: {str(e)}"
            )

    def _check_cached_code(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Check if pipeline code is cached locally."""
        cache_file = self.cache_dir / f"{model_id.replace('/', '_')}_info.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read cache file {cache_file}: {e}")
        
        return None

    def _validate_source_security(self, model_id: str) -> SecurityValidation:
        """Validate security of the model source."""
        try:
            # Extract domain from model ID or URL
            if self._is_huggingface_model(model_id):
                # For actual HF models, check if it's from a trusted HF namespace
                if "/" in model_id:
                    namespace = model_id.split("/")[0]
                    # Check if namespace contains suspicious domains
                    if "." in namespace and not any(trusted in namespace for trusted in ["huggingface", "hf"]):
                        domain = namespace  # Treat as potentially untrusted domain
                    else:
                        domain = "huggingface.co"  # Standard HF model
                else:
                    domain = "huggingface.co"
            else:
                # Try to parse as URL first
                parsed = urlparse(model_id)
                if parsed.netloc:
                    domain = parsed.netloc
                elif "/" in model_id and "." in model_id:
                    # Handle cases like "untrusted.com/model"
                    domain = model_id.split("/")[0]
                else:
                    domain = model_id
            
            # Check if source is trusted - exact match or trusted domain contains the domain
            is_trusted = False
            for trusted in self.trusted_sources:
                if domain == trusted or domain.endswith('.' + trusted) or trusted.startswith(domain + '/'):
                    is_trusted = True
                    break
            
            detected_risks = []
            security_warnings = []
            mitigation_suggestions = []
            
            if not is_trusted:
                detected_risks.append("Untrusted source domain")
                security_warnings.append(f"Source {domain} is not in trusted sources list")
                mitigation_suggestions.append("Verify source authenticity before proceeding")
            
            # Determine risk level
            if detected_risks:
                risk_level = "high"
            else:
                risk_level = "low"
            
            return SecurityValidation(
                is_safe=is_trusted,
                risk_level=risk_level,
                detected_risks=detected_risks,
                security_warnings=security_warnings,
                mitigation_suggestions=mitigation_suggestions
            )
            
        except Exception as e:
            return SecurityValidation(
                is_safe=False,
                risk_level="high",
                detected_risks=[f"Security validation failed: {str(e)}"],
                security_warnings=["Unable to validate source security"],
                mitigation_suggestions=["Manual security review required"]
            )

    def _fetch_from_huggingface(self, model_id: str) -> FetchResult:
        """Fetch pipeline code from Hugging Face."""
        try:
            # Use huggingface_hub to download pipeline files
            from huggingface_hub import hf_hub_download, list_repo_files
            
            # List files in the repository
            files = list_repo_files(model_id)
            pipeline_files = [f for f in files if f.startswith("pipeline_") and f.endswith(".py")]
            
            if not pipeline_files:
                return FetchResult(
                    success=False,
                    error_message=f"No pipeline files found in {model_id}",
                    fallback_options=self._get_manual_installation_options(model_id)
                )
            
            # Download the main pipeline file
            pipeline_file = pipeline_files[0]  # Use first pipeline file found
            
            downloaded_path = hf_hub_download(
                repo_id=model_id,
                filename=pipeline_file,
                cache_dir=str(self.cache_dir)
            )
            
            # Cache the download information
            self._cache_download_info(model_id, downloaded_path, "huggingface")
            
            return FetchResult(
                success=True,
                code_path=downloaded_path,
                version="latest"
            )
            
        except ImportError:
            return FetchResult(
                success=False,
                error_message="huggingface_hub not installed. Install with: pip install huggingface_hub",
                fallback_options=self._get_manual_installation_options(model_id)
            )
        except Exception as e:
            return FetchResult(
                success=False,
                error_message=f"Failed to fetch from Hugging Face: {str(e)}",
                fallback_options=self._get_manual_installation_options(model_id)
            )

    def _fetch_from_known_repos(self, model_id: str) -> FetchResult:
        """Fetch pipeline code from known repositories."""
        # Check if we have information about this pipeline
        for pipeline_name, info in self.known_pipelines.items():
            if pipeline_name.lower() in model_id.lower():
                return self._fetch_from_huggingface(info["repo"])
        
        return FetchResult(
            success=False,
            error_message=f"No known repository for model: {model_id}",
            fallback_options=self._get_manual_installation_options(model_id)
        )

    def _extract_code_version(self, code_path: str) -> Optional[str]:
        """Extract version information from pipeline code."""
        try:
            with open(code_path, 'r') as f:
                content = f.read()
            
            # Look for version patterns
            import re
            version_patterns = [
                r'__version__\s*=\s*["\']([^"\']+)["\']',
                r'VERSION\s*=\s*["\']([^"\']+)["\']',
                r'version\s*=\s*["\']([^"\']+)["\']'
            ]
            
            for pattern in version_patterns:
                match = re.search(pattern, content)
                if match:
                    return match.group(1)
            
            return "unknown"
            
        except Exception as e:
            logger.warning(f"Failed to extract version from {code_path}: {e}")
            return None

    def _calculate_compatibility_score(self, local_version: Optional[str], required_version: str) -> float:
        """Calculate compatibility score between versions."""
        if not local_version or local_version == "unknown":
            return 0.5  # Neutral score for unknown versions
        
        if local_version == required_version:
            return 1.0
        
        # Simple semantic version comparison
        try:
            local_parts = [int(x) for x in local_version.split('.')]
            required_parts = [int(x) for x in required_version.split('.')]
            
            # Pad shorter version with zeros
            max_len = max(len(local_parts), len(required_parts))
            local_parts.extend([0] * (max_len - len(local_parts)))
            required_parts.extend([0] * (max_len - len(required_parts)))
            
            # Calculate compatibility based on version differences
            score = 1.0
            for i, (local, required) in enumerate(zip(local_parts, required_parts)):
                weight = 1.0 / (2 ** i)  # Major version differences matter more
                if local != required:
                    score -= weight * 0.3
            
            return max(0.0, score)
            
        except ValueError:
            # Non-numeric versions, use string similarity
            if local_version in required_version or required_version in local_version:
                return 0.8
            return 0.3

    def _is_package_installed(self, requirement: str) -> bool:
        """Check if a package is already installed."""
        try:
            # Extract package name from requirement
            package_name = requirement.split('>=')[0].split('==')[0].split('<')[0].strip()
            
            result = subprocess.run(
                [subprocess.sys.executable, '-c', f'import {package_name}'],
                capture_output=True,
                text=True
            )
            
            return result.returncode == 0
            
        except Exception:
            return False

    def _install_package(self, requirement: str) -> Dict[str, Any]:
        """Install a single package using pip."""
        try:
            result = subprocess.run(
                [subprocess.sys.executable, '-m', 'pip', 'install', requirement],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            return {
                "success": result.returncode == 0,
                "log": result.stdout + result.stderr,
                "error": result.stderr if result.returncode != 0 else None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "log": f"Installation of {requirement} timed out",
                "error": "Installation timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "log": f"Failed to install {requirement}",
                "error": str(e)
            }

    def _get_manual_installation_options(self, model_path: str) -> List[str]:
        """Get manual installation options for a model."""
        return self.get_fallback_options(model_path)

    def _cache_download_info(self, model_id: str, code_path: str, source: str):
        """Cache information about downloaded code."""
        try:
            cache_file = self.cache_dir / f"{model_id.replace('/', '_')}_info.json"
            
            info = {
                "model_id": model_id,
                "code_path": code_path,
                "source": source,
                "download_time": str(Path(code_path).stat().st_mtime),
                "security_hash": self._calculate_file_hash(code_path)
            }
            
            with open(cache_file, 'w') as f:
                json.dump(info, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to cache download info: {e}")

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return "unknown"