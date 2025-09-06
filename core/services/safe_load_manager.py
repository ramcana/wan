"""
SafeLoadManager - Security and safe loading features for model handling

This module implements security validation, sandboxed environments, and safe loading
policies for handling trusted vs untrusted models and remote code execution.

Requirements addressed: 6.1, 6.2, 6.3, 6.4
"""

import os
import json
import hashlib
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
import requests
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class SafeLoadingOptions:
    """Configuration options for safe model loading"""
    allow_remote_code: bool
    use_sandbox: bool
    restricted_operations: List[str]
    timeout_seconds: int
    memory_limit_mb: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SafeLoadingOptions':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class SecurityValidation:
    """Result of security validation for remote code"""
    is_safe: bool
    risk_level: str  # "low", "medium", "high"
    detected_risks: List[str]
    mitigation_strategies: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    name: str
    description: str
    allow_remote_code: bool
    trusted_domains: List[str]
    blocked_domains: List[str]
    max_file_size_mb: int
    allowed_file_extensions: List[str]
    restricted_operations: List[str]
    sandbox_required: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityPolicy':
        """Create from dictionary"""
        return cls(**data)


class SandboxEnvironment:
    """Sandboxed environment for untrusted code execution"""
    
    def __init__(self, temp_dir: str, restrictions: List[str]):
        self.temp_dir = Path(temp_dir)
        self.restrictions = restrictions
        self.is_active = False
        self._original_env = {}
        
    def __enter__(self):
        """Enter sandbox context"""
        self.activate()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit sandbox context"""
        self.deactivate()
        
    def activate(self):
        """Activate sandbox restrictions"""
        if self.is_active:
            return
            
        # Store original environment
        self._original_env = dict(os.environ)
        
        # Apply restrictions
        if "network_access" in self.restrictions:
            os.environ["NO_PROXY"] = "*"
            os.environ["HTTP_PROXY"] = "127.0.0.1:0"  # Invalid proxy to block network
            
        if "file_system" in self.restrictions:
            # Limit to temp directory
            os.environ["TMPDIR"] = str(self.temp_dir)
            os.environ["TEMP"] = str(self.temp_dir)
            
        self.is_active = True
        logger.info(f"Sandbox activated with restrictions: {self.restrictions}")
        
    def deactivate(self):
        """Deactivate sandbox and restore environment"""
        if not self.is_active:
            return
            
        # Restore original environment
        os.environ.clear()
        os.environ.update(self._original_env)
        
        self.is_active = False
        logger.info("Sandbox deactivated")
        
    def execute_in_sandbox(self, func, *args, **kwargs):
        """Execute function within sandbox"""
        with self:
            return func(*args, **kwargs)


class SafeLoadManager:
    """Manage safe vs trust modes for model loading"""
    
    def __init__(self, default_mode: str = "safe", config_path: Optional[str] = None):
        self.mode = default_mode
        self.config_path = config_path or "security_config.json"
        self.trusted_sources: Set[str] = set()
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.current_policy: Optional[SecurityPolicy] = None
        
        # Load configuration
        self._load_security_policies()
        self._load_trusted_sources()
        
        # Set default policy
        if "default" in self.security_policies:
            self.current_policy = self.security_policies["default"]
        else:
            self.current_policy = self._create_default_policy()
            
    def _create_default_policy(self) -> SecurityPolicy:
        """Create default security policy"""
        return SecurityPolicy(
            name="default",
            description="Default security policy for safe loading",
            allow_remote_code=False,
            trusted_domains=["huggingface.co", "hf.co"],
            blocked_domains=[],
            max_file_size_mb=100,
            allowed_file_extensions=[".py", ".json", ".txt", ".md"],
            restricted_operations=["network_access", "file_system"],
            sandbox_required=True
        )
        
    def _load_security_policies(self):
        """Load security policies from configuration file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    
                policies_data = config.get("security_policies", {})
                for name, policy_data in policies_data.items():
                    self.security_policies[name] = SecurityPolicy.from_dict(policy_data)
                    
                logger.info(f"Loaded {len(self.security_policies)} security policies")
            else:
                logger.info("No security config found, using defaults")
                
        except Exception as e:
            logger.error(f"Failed to load security policies: {e}")
            
    def _load_trusted_sources(self):
        """Load trusted sources from configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    
                trusted_sources = config.get("trusted_sources", [])
                self.trusted_sources = set(trusted_sources)
                
                logger.info(f"Loaded {len(self.trusted_sources)} trusted sources")
                
        except Exception as e:
            logger.error(f"Failed to load trusted sources: {e}")
            
    def save_configuration(self):
        """Save current configuration to file"""
        try:
            config = {
                "security_policies": {
                    name: policy.to_dict() 
                    for name, policy in self.security_policies.items()
                },
                "trusted_sources": list(self.trusted_sources)
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Security configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save security configuration: {e}")
            
    def set_loading_mode(self, mode: str):
        """Set loading mode: 'safe' or 'trust'"""
        if mode not in ["safe", "trust"]:
            raise ValueError("Mode must be 'safe' or 'trust'")
        self.mode = mode
        logger.info(f"Loading mode set to: {mode}")
        
    def set_security_policy(self, policy_name: str):
        """Set active security policy"""
        if policy_name not in self.security_policies:
            raise ValueError(f"Security policy '{policy_name}' not found")
        self.current_policy = self.security_policies[policy_name]
        logger.info(f"Security policy set to: {policy_name}")
        
    def add_trusted_source(self, source: str):
        """Add a trusted source"""
        self.trusted_sources.add(source)
        logger.info(f"Added trusted source: {source}")
        
    def remove_trusted_source(self, source: str):
        """Remove a trusted source"""
        self.trusted_sources.discard(source)
        logger.info(f"Removed trusted source: {source}")
        
    def is_source_trusted(self, model_source: str) -> bool:
        """Check if model source is in trusted list"""
        # Check exact match
        if model_source in self.trusted_sources:
            return True
            
        # Check domain-based trust
        try:
            parsed_url = urlparse(model_source)
            domain = parsed_url.netloc.lower()
            
            # Check if domain is in trusted domains from current policy
            if self.current_policy:
                for trusted_domain in self.current_policy.trusted_domains:
                    if domain.endswith(trusted_domain.lower()):
                        return True
                        
            # Check if any trusted source is a prefix
            for trusted in self.trusted_sources:
                if model_source.startswith(trusted):
                    return True
                    
        except Exception as e:
            logger.warning(f"Failed to parse model source URL: {e}")
            
        return False
        
    def validate_remote_code_safety(self, code_source: str, code_content: Optional[str] = None) -> SecurityValidation:
        """Validate safety of remote code before execution"""
        risks = []
        mitigation_strategies = []
        risk_level = "low"
        
        # Check source domain
        try:
            parsed_url = urlparse(code_source)
            domain = parsed_url.netloc.lower()
            
            if self.current_policy and domain in [d.lower() for d in self.current_policy.blocked_domains]:
                risks.append(f"Code source from blocked domain: {domain}")
                risk_level = "high"
                
        except Exception:
            risks.append("Invalid or unparseable code source URL")
            risk_level = "medium"
            
        # Analyze code content if provided
        if code_content:
            dangerous_patterns = [
                ("import os", "Operating system access"),
                ("import subprocess", "Subprocess execution"),
                ("import sys", "System module access"),
                ("eval(", "Dynamic code evaluation"),
                ("exec(", "Dynamic code execution"),
                ("__import__", "Dynamic imports"),
                ("open(", "File system access"),
                ("requests.", "Network requests"),
                ("urllib", "Network access"),
                ("socket", "Network socket access")
            ]
            
            for pattern, description in dangerous_patterns:
                if pattern in code_content:
                    risks.append(f"Potentially dangerous operation: {description}")
                    if risk_level == "low":
                        risk_level = "medium"
                        
        # Determine overall safety
        is_safe = len(risks) == 0 or (risk_level == "low" and self.mode == "trust")
        
        # Generate mitigation strategies
        if risks:
            mitigation_strategies.extend([
                "Execute code in sandboxed environment",
                "Review code manually before execution",
                "Use restricted execution mode",
                "Monitor resource usage during execution"
            ])
            
        return SecurityValidation(
            is_safe=is_safe,
            risk_level=risk_level,
            detected_risks=risks,
            mitigation_strategies=mitigation_strategies
        )
        
    def create_sandboxed_environment(self, restrictions: Optional[List[str]] = None) -> SandboxEnvironment:
        """Create sandboxed environment for untrusted code execution"""
        if restrictions is None:
            restrictions = self.current_policy.restricted_operations if self.current_policy else []
            
        # Create temporary directory for sandbox
        temp_dir = tempfile.mkdtemp(prefix="wan_sandbox_")
        
        return SandboxEnvironment(temp_dir, restrictions)
        
    def get_safe_loading_options(self, model_path: str) -> SafeLoadingOptions:
        """Get safe loading options for specific model"""
        is_trusted = self.is_source_trusted(model_path)
        policy = self.current_policy or self._create_default_policy()
        
        if self.mode == "trust":
            # Trust mode - permissive settings regardless of source
            return SafeLoadingOptions(
                allow_remote_code=True,
                use_sandbox=False,
                restricted_operations=[],
                timeout_seconds=300,
                memory_limit_mb=0  # No limit
            )
        elif is_trusted:
            # Trusted source in safe mode - use policy but be more permissive
            return SafeLoadingOptions(
                allow_remote_code=True,
                use_sandbox=False,
                restricted_operations=policy.restricted_operations if policy.restricted_operations else [],
                timeout_seconds=300,
                memory_limit_mb=0  # No limit
            )
        else:
            # Untrusted source in safe mode - apply full restrictions
            return SafeLoadingOptions(
                allow_remote_code=policy.allow_remote_code,
                use_sandbox=policy.sandbox_required,
                restricted_operations=policy.restricted_operations,
                timeout_seconds=60,
                memory_limit_mb=1024  # 1GB limit
            )
            
    def validate_model_source(self, model_path: str) -> SecurityValidation:
        """Validate model source for security risks"""
        risks = []
        mitigation_strategies = []
        risk_level = "low"
        
        # Check if source is trusted
        if not self.is_source_trusted(model_path):
            risks.append("Model source is not in trusted list")
            risk_level = "medium"
            mitigation_strategies.append("Add source to trusted list if verified")
            
        # Check for suspicious patterns in path
        suspicious_patterns = [
            ("../", "Path traversal attempt"),
            ("~", "Home directory access"),
            ("file://", "Local file protocol"),
            ("ftp://", "FTP protocol"),
        ]
        
        for pattern, description in suspicious_patterns:
            if pattern in model_path:
                risks.append(f"Suspicious path pattern: {description}")
                if risk_level == "low":
                    risk_level = "medium"
                    
        # Additional checks for local paths
        if os.path.isabs(model_path) and not model_path.startswith(("/tmp", "/var/tmp")):
            if not self.is_source_trusted(model_path):
                risks.append("Absolute path to untrusted location")
                risk_level = "medium"
                
        is_safe = len(risks) == 0 or (risk_level == "low")
        
        if risks and not mitigation_strategies:
            mitigation_strategies.extend([
                "Use sandboxed loading",
                "Verify model integrity",
                "Monitor resource usage"
            ])
            
        return SecurityValidation(
            is_safe=is_safe,
            risk_level=risk_level,
            detected_risks=risks,
            mitigation_strategies=mitigation_strategies
        )
        
    @contextmanager
    def secure_model_loading(self, model_path: str):
        """Context manager for secure model loading"""
        options = self.get_safe_loading_options(model_path)
        sandbox = None
        
        try:
            if options.use_sandbox:
                sandbox = self.create_sandboxed_environment(options.restricted_operations)
                sandbox.activate()
                
            yield options
            
        finally:
            if sandbox:
                sandbox.deactivate()
                # Cleanup sandbox directory
                try:
                    shutil.rmtree(sandbox.temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to cleanup sandbox directory: {e}")