"""
Enhanced user guidance system for installation problems.

This module provides user-friendly error messages, troubleshooting guides,
interactive help, progress indicators, recovery strategy explanations,
and support ticket generation for installation issues.

Enhanced features for reliability system:
- Real-time progress indicators with estimated completion times
- Recovery strategy explanations with success likelihood
- Pre-filled error reports for support ticket creation
- Links to documentation and support resources
- Enhanced error message formatting with context
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import urllib.parse
from datetime import datetime, timedelta
import sys
import platform
import psutil
import traceback
from enum import Enum

# Try to import from local scripts directory
try:
    from interfaces import InstallationError, ErrorCategory
    from base_classes import BaseInstallationComponent
    from diagnostic_tool import InstallationDiagnosticTool, DiagnosticResult
except ImportError:
    # Fallback for testing - create minimal classes
    from enum import Enum
    
    class ErrorCategory(Enum):
        SYSTEM = "system"
        NETWORK = "network"
        PERMISSION = "permission"
        CONFIGURATION = "configuration"
    
    class InstallationError(Exception):
        def __init__(self, message: str, category: ErrorCategory, recovery_suggestions=None):
            super().__init__(message)
            self.message = message
            self.category = category
            self.recovery_suggestions = recovery_suggestions or []
    
    class BaseInstallationComponent:
        def __init__(self, installation_path: str, logger=None):
            self.installation_path = installation_path
            self.logger = logger or logging.getLogger(__name__)
    
    class DiagnosticResult:
        def __init__(self, success: bool, message: str):
            self.success = success
            self.message = message
    
    class InstallationDiagnosticTool:
        def __init__(self, installation_path: str, logger=None):
            self.installation_path = installation_path
            self.logger = logger
        
        def get_quick_health_check(self):
            return {
                "python_ok": True,
                "memory_ok": True,
                "disk_ok": True,
                "network_ok": True
            }
        
        def run_full_diagnostics(self):
            return {"status": "completed"}
        
        def generate_diagnostic_report(self, output_path=None):
            return "Diagnostic report generated"


class RecoveryStatus(Enum):
    """Status of a recovery operation."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SupportResourceType(Enum):
    """Types of support resources."""
    DOCUMENTATION = "documentation"
    COMMUNITY_FORUM = "community_forum"
    SUPPORT_TICKET = "support_ticket"
    VIDEO_TUTORIAL = "video_tutorial"
    FAQ = "faq"


@dataclass
class RecoveryStrategy:
    """A recovery strategy with success likelihood and progress tracking."""
    name: str
    description: str
    success_likelihood: float  # 0.0 to 1.0
    estimated_time_minutes: int
    steps: List[Dict[str, Any]]
    prerequisites: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    status: RecoveryStatus = RecoveryStatus.NOT_STARTED
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class TroubleshootingGuide:
    """A troubleshooting guide for a specific problem."""
    title: str
    problem_description: str
    symptoms: List[str]
    solutions: List[Dict[str, Any]]  # Each solution has 'step', 'description', 'command' (optional)
    related_errors: List[str]
    difficulty: str  # "easy", "medium", "hard"
    estimated_time: str
    recovery_strategies: List[RecoveryStrategy] = field(default_factory=list)


@dataclass
class ProgressIndicator:
    """Enhanced progress indicator for recovery operations."""
    operation_name: str
    current_step: int
    total_steps: int
    current_step_name: str
    estimated_completion_time: Optional[datetime] = None
    start_time: datetime = field(default_factory=datetime.now)
    progress_percentage: float = 0.0
    time_remaining_seconds: Optional[int] = None
    speed_info: Optional[str] = None  # e.g., "2.5 MB/s", "15 files/min"
    
    def update_progress(self, step: int, step_name: str, speed_info: Optional[str] = None):
        """Update progress information."""
        self.current_step = step
        self.current_step_name = step_name
        self.progress_percentage = (step / self.total_steps) * 100
        if speed_info:
            self.speed_info = speed_info
        
        # Calculate estimated completion time
        if step > 0:
            elapsed = datetime.now() - self.start_time
            avg_time_per_step = elapsed.total_seconds() / step
            remaining_steps = self.total_steps - step
            self.time_remaining_seconds = int(avg_time_per_step * remaining_steps)
            self.estimated_completion_time = datetime.now() + timedelta(seconds=self.time_remaining_seconds)
    
    def get_progress_bar(self, width: int = 40) -> str:
        """Generate a text-based progress bar."""
        filled = int(width * self.progress_percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {self.progress_percentage:.1f}%"
    
    def get_time_remaining_str(self) -> str:
        """Get formatted time remaining string."""
        if self.time_remaining_seconds is None:
            return "Calculating..."
        
        if self.time_remaining_seconds < 60:
            return f"{self.time_remaining_seconds}s remaining"
        elif self.time_remaining_seconds < 3600:
            minutes = self.time_remaining_seconds // 60
            seconds = self.time_remaining_seconds % 60
            return f"{minutes}m {seconds}s remaining"
        else:
            hours = self.time_remaining_seconds // 3600
            minutes = (self.time_remaining_seconds % 3600) // 60
            return f"{hours}h {minutes}m remaining"


@dataclass
class SupportResource:
    """A support resource with links and descriptions."""
    title: str
    description: str
    url: str
    resource_type: SupportResourceType
    relevance_score: float = 0.0  # 0.0 to 1.0


@dataclass
class SupportTicket:
    """Enhanced pre-filled support ticket information."""
    title: str
    description: str
    error_details: str
    system_info: Dict[str, Any]
    logs: List[str]
    steps_attempted: List[str]
    recovery_strategies_tried: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    severity: str = "medium"  # low, medium, high, critical
    category: str = "installation"
    environment_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_markdown(self) -> str:
        """Convert support ticket to markdown format."""
        lines = [
            f"# Support Ticket: {self.title}",
            "",
            f"**Timestamp:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Severity:** {self.severity.title()}",
            f"**Category:** {self.category.title()}",
            "",
            "## Problem Description",
            self.description,
            "",
            "## Error Details",
            "```",
            self.error_details,
            "```",
            "",
            "## System Information",
        ]
        
        for key, value in self.system_info.items():
            lines.append(f"- **{key}:** {value}")
        
        if self.environment_details:
            lines.extend([
                "",
                "## Environment Details",
            ])
            for key, value in self.environment_details.items():
                lines.append(f"- **{key}:** {value}")
        
        lines.extend([
            "",
            "## Steps Attempted",
        ])
        
        for i, step in enumerate(self.steps_attempted, 1):
            lines.append(f"{i}. {step}")
        
        if self.recovery_strategies_tried:
            lines.extend([
                "",
                "## Recovery Strategies Tried",
            ])
            for strategy in self.recovery_strategies_tried:
                lines.append(f"- {strategy}")
        
        if self.logs:
            lines.extend([
                "",
                "## Relevant Log Entries",
                "```",
            ])
            lines.extend(self.logs[-10:])  # Last 10 log entries
            lines.extend([
                "```",
            ])
        
        return "\n".join(lines)
    
    def to_url_encoded(self) -> str:
        """Convert to URL-encoded format for web forms."""
        content = self.to_markdown()
        return urllib.parse.quote(content)


class UserGuidanceSystem(BaseInstallationComponent):
    """Enhanced user guidance system for installation issues with reliability features."""
    
    def __init__(self, installation_path: str, logger: Optional[logging.Logger] = None):
        super().__init__(installation_path, logger)
        self.diagnostic_tool = InstallationDiagnosticTool(installation_path, logger)
        self.troubleshooting_guides = self._initialize_troubleshooting_guides()
        self.error_message_templates = self._initialize_error_templates()
        self.support_resources = self._initialize_support_resources()
        self.active_progress_indicators: Dict[str, ProgressIndicator] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        self._progress_lock = threading.Lock()
    
    def _initialize_troubleshooting_guides(self) -> Dict[str, TroubleshootingGuide]:
        """Initialize comprehensive troubleshooting guides."""
        guides = {}
        
        # Python Installation Issues
        guides["python_not_found"] = TroubleshootingGuide(
            title="Python Not Found or Incompatible Version",
            problem_description="The installer cannot find Python or the installed version is too old.",
            symptoms=[
                "Error message about Python not being found",
                "Python version is older than 3.9",
                "'python' command not recognized"
            ],
            solutions=[
                {
                    "step": 1,
                    "description": "Download and install Python 3.9 or later from python.org",
                    "command": "Visit https://www.python.org/downloads/ and download the latest version"
                },
                {
                    "step": 2,
                    "description": "During installation, make sure to check 'Add Python to PATH'",
                    "command": None
                },
                {
                    "step": 3,
                    "description": "Restart your command prompt and try again",
                    "command": "python --version"
                },
                {
                    "step": 4,
                    "description": "If still not working, manually add Python to your PATH environment variable",
                    "command": None
                }
            ],
            related_errors=["python not found", "python version", "command not recognized"],
            difficulty="easy",
            estimated_time="10-15 minutes"
        )
        
        # Network Connectivity Issues
        guides["network_issues"] = TroubleshootingGuide(
            title="Network Connectivity Problems",
            problem_description="Unable to download required packages or models due to network issues.",
            symptoms=[
                "Download timeouts or failures",
                "Connection refused errors",
                "SSL certificate errors",
                "Proxy-related errors"
            ],
            solutions=[
                {
                    "step": 1,
                    "description": "Check your internet connection",
                    "command": "ping google.com"
                },
                {
                    "step": 2,
                    "description": "Temporarily disable firewall and antivirus",
                    "command": None
                },
                {
                    "step": 3,
                    "description": "If behind a corporate firewall, configure proxy settings",
                    "command": "pip config set global.proxy http://your-proxy:port"
                },
                {
                    "step": 4,
                    "description": "Try using a VPN if in a restricted network",
                    "command": None
                },
                {
                    "step": 5,
                    "description": "Use alternative package index if PyPI is blocked",
                    "command": "pip install -i https://pypi.douban.com/simple/ package_name"
                }
            ],
            related_errors=["connection", "timeout", "ssl", "certificate", "proxy"],
            difficulty="medium",
            estimated_time="15-30 minutes"
        )
        
        # Permission Issues
        guides["permission_denied"] = TroubleshootingGuide(
            title="Permission Denied Errors",
            problem_description="Insufficient permissions to create files or install packages.",
            symptoms=[
                "Permission denied errors",
                "Access denied messages",
                "Unable to create directories",
                "Package installation failures"
            ],
            solutions=[
                {
                    "step": 1,
                    "description": "Run the installer as Administrator",
                    "command": "Right-click on install.bat and select 'Run as administrator'"
                },
                {
                    "step": 2,
                    "description": "Check if the installation directory is write-protected",
                    "command": None
                },
                {
                    "step": 3,
                    "description": "Choose a different installation directory (e.g., C:\\WAN22)",
                    "command": None
                },
                {
                    "step": 4,
                    "description": "Temporarily disable antivirus real-time protection",
                    "command": None
                },
                {
                    "step": 5,
                    "description": "Close any applications that might be using the files",
                    "command": None
                }
            ],
            related_errors=["permission", "access", "denied", "forbidden", "unauthorized"],
            difficulty="easy",
            estimated_time="5-10 minutes"
        )
        
        # Insufficient Resources
        guides["insufficient_resources"] = TroubleshootingGuide(
            title="Insufficient System Resources",
            problem_description="System doesn't meet minimum requirements for installation or operation.",
            symptoms=[
                "Out of memory errors",
                "Disk space warnings",
                "Slow installation progress",
                "System freezing during installation"
            ],
            solutions=[
                {
                    "step": 1,
                    "description": "Free up disk space (at least 50GB recommended)",
                    "command": "Use Disk Cleanup or delete unnecessary files"
                },
                {
                    "step": 2,
                    "description": "Close other applications to free up memory",
                    "command": "Check Task Manager and close unnecessary programs"
                },
                {
                    "step": 3,
                    "description": "Consider upgrading system memory if less than 8GB",
                    "command": None
                },
                {
                    "step": 4,
                    "description": "Install on a faster drive (SSD recommended)",
                    "command": None
                },
                {
                    "step": 5,
                    "description": "Use lightweight configuration for lower-end hardware",
                    "command": None
                }
            ],
            related_errors=["memory", "disk", "space", "insufficient", "out of"],
            difficulty="medium",
            estimated_time="20-60 minutes"
        )
        
        # GPU/CUDA Issues
        guides["gpu_cuda_issues"] = TroubleshootingGuide(
            title="GPU and CUDA Problems",
            problem_description="Issues with GPU detection, CUDA installation, or GPU acceleration.",
            symptoms=[
                "GPU not detected",
                "CUDA not available",
                "GPU memory errors",
                "Slow performance (CPU-only mode)"
            ],
            solutions=[
                {
                    "step": 1,
                    "description": "Update NVIDIA drivers to the latest version",
                    "command": "Visit https://www.nvidia.com/drivers/ or use GeForce Experience"
                },
                {
                    "step": 2,
                    "description": "Install CUDA Toolkit (version 11.8 or 12.1 recommended)",
                    "command": "Download from https://developer.nvidia.com/cuda-downloads"
                },
                {
                    "step": 3,
                    "description": "Verify GPU is properly seated and powered",
                    "command": "nvidia-smi"
                },
                {
                    "step": 4,
                    "description": "Check if GPU has sufficient VRAM (8GB+ recommended)",
                    "command": None
                },
                {
                    "step": 5,
                    "description": "Restart computer after driver installation",
                    "command": None
                }
            ],
            related_errors=["cuda", "gpu", "nvidia", "vram", "device"],
            difficulty="medium",
            estimated_time="30-45 minutes"
        )
        
        # Model Download Issues
        guides["model_download_issues"] = TroubleshootingGuide(
            title="Model Download Problems",
            problem_description="Issues downloading WAN2.2 models from Hugging Face or other sources.",
            symptoms=[
                "Model download timeouts",
                "Corrupted model files",
                "Authentication errors",
                "Slow download speeds"
            ],
            solutions=[
                {
                    "step": 1,
                    "description": "Check internet connection stability",
                    "command": None
                },
                {
                    "step": 2,
                    "description": "Clear browser cache and try downloading manually",
                    "command": "Visit https://huggingface.co/Wan2.2/ and download models manually"
                },
                {
                    "step": 3,
                    "description": "Use git-lfs for large file downloads",
                    "command": "git lfs install && git clone https://huggingface.co/Wan2.2/model-name"
                },
                {
                    "step": 4,
                    "description": "Try downloading during off-peak hours",
                    "command": None
                },
                {
                    "step": 5,
                    "description": "Use alternative download methods or mirrors",
                    "command": None
                }
            ],
            related_errors=["download", "model", "huggingface", "timeout", "corrupted"],
            difficulty="medium",
            estimated_time="30-120 minutes"
        )
        
        # Antivirus Interference
        guides["antivirus_interference"] = TroubleshootingGuide(
            title="Antivirus Software Interference",
            problem_description="Antivirus software is blocking installation or execution.",
            symptoms=[
                "Files being deleted after creation",
                "Installation suddenly stopping",
                "False positive virus warnings",
                "Quarantined files"
            ],
            solutions=[
                {
                    "step": 1,
                    "description": "Temporarily disable real-time protection",
                    "command": "Windows Security > Virus & threat protection > Manage settings"
                },
                {
                    "step": 2,
                    "description": "Add installation directory to antivirus exclusions",
                    "command": "Add the entire WAN22 folder to exclusions"
                },
                {
                    "step": 3,
                    "description": "Restore any quarantined files",
                    "command": "Check antivirus quarantine and restore files"
                },
                {
                    "step": 4,
                    "description": "Run installation as administrator",
                    "command": None
                },
                {
                    "step": 5,
                    "description": "Re-enable protection after successful installation",
                    "command": None
                }
            ],
            related_errors=["virus", "quarantine", "blocked", "deleted", "protection"],
            difficulty="easy",
            estimated_time="10-15 minutes"
        )
        
        return guides
    
    def _initialize_support_resources(self) -> Dict[str, List[SupportResource]]:
        """Initialize support resources for different error categories."""
        resources = {
            "general": [
                SupportResource(
                    title="WAN2.2 Installation Guide",
                    description="Complete installation guide with troubleshooting tips",
                    url="https://github.com/wan22/installation-guide",
                    resource_type=SupportResourceType.DOCUMENTATION,
                    relevance_score=0.9
                ),
                SupportResource(
                    title="Community Support Forum",
                    description="Get help from the community and developers",
                    url="https://github.com/wan22/discussions",
                    resource_type=SupportResourceType.COMMUNITY_FORUM,
                    relevance_score=0.8
                ),
                SupportResource(
                    title="Frequently Asked Questions",
                    description="Common questions and answers about installation",
                    url="https://github.com/wan22/wiki/FAQ",
                    resource_type=SupportResourceType.FAQ,
                    relevance_score=0.7
                ),
                SupportResource(
                    title="Video Tutorial: Installation Walkthrough",
                    description="Step-by-step video guide for installation",
                    url="https://youtube.com/watch?v=wan22-install",
                    resource_type=SupportResourceType.VIDEO_TUTORIAL,
                    relevance_score=0.6
                )
            ],
            "network": [
                SupportResource(
                    title="Network Troubleshooting Guide",
                    description="Resolve network connectivity and download issues",
                    url="https://github.com/wan22/wiki/network-troubleshooting",
                    resource_type=SupportResourceType.DOCUMENTATION,
                    relevance_score=0.9
                ),
                SupportResource(
                    title="Proxy Configuration Guide",
                    description="Configure proxy settings for corporate networks",
                    url="https://github.com/wan22/wiki/proxy-setup",
                    resource_type=SupportResourceType.DOCUMENTATION,
                    relevance_score=0.8
                )
            ],
            "permission": [
                SupportResource(
                    title="Permission Issues Guide",
                    description="Fix permission and administrator access problems",
                    url="https://github.com/wan22/wiki/permissions",
                    resource_type=SupportResourceType.DOCUMENTATION,
                    relevance_score=0.9
                )
            ],
            "system": [
                SupportResource(
                    title="System Requirements Guide",
                    description="Check and meet minimum system requirements",
                    url="https://github.com/wan22/wiki/system-requirements",
                    resource_type=SupportResourceType.DOCUMENTATION,
                    relevance_score=0.9
                ),
                SupportResource(
                    title="GPU Setup Guide",
                    description="Configure GPU and CUDA for optimal performance",
                    url="https://github.com/wan22/wiki/gpu-setup",
                    resource_type=SupportResourceType.DOCUMENTATION,
                    relevance_score=0.8
                )
            ]
        }
        return resources
    
    def _initialize_error_templates(self) -> Dict[ErrorCategory, Dict[str, str]]:
        """Initialize user-friendly error message templates."""
        return {
            ErrorCategory.NETWORK: {
                "title": "🌐 Network Connection Issue",
                "description": "There's a problem with your internet connection or network settings.",
                "general_advice": "Check your internet connection and try again. If you're behind a firewall or proxy, you may need to configure network settings."
            },
            ErrorCategory.PERMISSION: {
                "title": "🔒 Permission Problem",
                "description": "The installer doesn't have sufficient permissions to complete this operation.",
                "general_advice": "Try running the installer as Administrator, or choose a different installation directory where you have write permissions."
            },
            ErrorCategory.SYSTEM: {
                "title": "⚙️ System Issue",
                "description": "There's a problem with your system configuration or resources.",
                "general_advice": "Check that your system meets the minimum requirements and that you have sufficient disk space and memory available."
            },
            ErrorCategory.CONFIGURATION: {
                "title": "⚙️ Configuration Error",
                "description": "There's an issue with the configuration settings or file format.",
                "general_advice": "Check the configuration file syntax and ensure all required settings are present and valid."
            }
        }
    
    def format_user_friendly_error(self, error: InstallationError, 
                                 context: Optional[Dict[str, Any]] = None,
                                 recovery_strategies: Optional[List[RecoveryStrategy]] = None) -> str:
        """Format an enhanced error message with recovery strategies and support resources."""
        template = self.error_message_templates.get(error.category, {
            "title": "❌ Installation Error",
            "description": "An unexpected error occurred during installation.",
            "general_advice": "Please check the error details and try again."
        })
        
        lines = [
            "=" * 80,
            f"  {template['title']}",
            "=" * 80,
            "",
            "🔍 What happened:",
            f"   {error.message}",
            "",
            "💡 Why this might have occurred:",
            f"   {template['description']}",
            ""
        ]
        
        # Add system context if available
        if context:
            lines.extend([
                "🖥️  System Context:",
                f"   • Time: {context.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}",
                f"   • Phase: {context.get('phase', 'Unknown')}",
                f"   • Component: {context.get('component', 'Unknown')}",
                ""
            ])
        
        # Add recovery strategies with success likelihood
        if recovery_strategies:
            lines.extend([
                "🔧 Available Recovery Strategies:",
                ""
            ])
            
            for i, strategy in enumerate(recovery_strategies, 1):
                success_pct = int(strategy.success_likelihood * 100)
                time_str = f"{strategy.estimated_time_minutes} min"
                
                lines.extend([
                    f"   {i}. {strategy.name}",
                    f"      Success likelihood: {success_pct}% | Estimated time: {time_str}",
                    f"      {strategy.description}",
                    ""
                ])
                
                if strategy.prerequisites:
                    lines.append(f"      Prerequisites: {', '.join(strategy.prerequisites)}")
                if strategy.risks:
                    lines.append(f"      ⚠️  Risks: {', '.join(strategy.risks)}")
                lines.append("")
        
        # Add specific suggestions if available
        if error.recovery_suggestions:
            lines.extend([
                "📋 Immediate Actions:",
                ""
            ])
            for i, suggestion in enumerate(error.recovery_suggestions, 1):
                lines.append(f"   {i}. {suggestion}")
            lines.append("")
        
        # Add troubleshooting guide reference
        guide = self.find_relevant_troubleshooting_guide(error)
        if guide:
            lines.extend([
                f"📖 Detailed Troubleshooting Guide:",
                f"   Title: {guide.title}",
                f"   Difficulty: {guide.difficulty.title()} | Time: {guide.estimated_time}",
                f"   Run: python scripts/user_guidance.py --guide {guide.title.lower().replace(' ', '_')}",
                ""
            ])
        
        # Add support resources
        category_key = error.category.value if hasattr(error.category, 'value') else str(error.category).lower()
        resources = self.support_resources.get(category_key, self.support_resources.get("general", []))
        
        if resources:
            lines.extend([
                "🆘 Support Resources:",
                ""
            ])
            
            # Sort by relevance score
            sorted_resources = sorted(resources, key=lambda r: r.relevance_score, reverse=True)[:3]
            
            for resource in sorted_resources:
                icon = self._get_resource_icon(resource.resource_type)
                lines.extend([
                    f"   {icon} {resource.title}",
                    f"      {resource.description}",
                    f"      Link: {resource.url}",
                    ""
                ])
        
        lines.extend([
            "🔧 Quick Actions:",
            "   • Run diagnostics: python scripts/diagnostic_tool.py",
            "   • Interactive help: python scripts/user_guidance.py --interactive",
            "   • Generate support ticket: python scripts/user_guidance.py --support-ticket",
            "",
            "=" * 80
        ])
        
        return "\n".join(lines)
    
    def _get_resource_icon(self, resource_type: SupportResourceType) -> str:
        """Get icon for resource type."""
        icons = {
            SupportResourceType.DOCUMENTATION: "📚",
            SupportResourceType.COMMUNITY_FORUM: "💬",
            SupportResourceType.SUPPORT_TICKET: "🎫",
            SupportResourceType.VIDEO_TUTORIAL: "🎥",
            SupportResourceType.FAQ: "❓"
        }
        return icons.get(resource_type, "🔗")
    
    def find_relevant_troubleshooting_guide(self, error: InstallationError) -> Optional[TroubleshootingGuide]:
        """Find the most relevant troubleshooting guide for an error."""
        error_message_lower = error.message.lower()
        
        # Score each guide based on keyword matches
        best_guide = None
        best_score = 0
        
        for guide_key, guide in self.troubleshooting_guides.items():
            score = 0
            
            # Check related errors
            for related_error in guide.related_errors:
                if related_error.lower() in error_message_lower:
                    score += 2
            
            # Check symptoms
            for symptom in guide.symptoms:
                if any(word in error_message_lower for word in symptom.lower().split()):
                    score += 1
            
            # Also check error category
            if error.category == ErrorCategory.NETWORK and "network" in guide_key:
                score += 3
            elif error.category == ErrorCategory.PERMISSION and "permission" in guide_key:
                score += 3
            elif error.category == ErrorCategory.SYSTEM and ("system" in guide_key or "resource" in guide_key):
                score += 3
            
            if score > best_score:
                best_score = score
                best_guide = guide
        
        return best_guide if best_score > 0 else None
    
    def create_progress_indicator(self, operation_name: str, total_steps: int, 
                                initial_step_name: str = "Starting...") -> str:
        """Create a new progress indicator and return its ID."""
        progress_id = f"{operation_name}_{int(time.time())}"
        
        with self._progress_lock:
            self.active_progress_indicators[progress_id] = ProgressIndicator(
                operation_name=operation_name,
                current_step=0,
                total_steps=total_steps,
                current_step_name=initial_step_name
            )
        
        return progress_id
    
    def update_progress(self, progress_id: str, step: int, step_name: str, 
                       speed_info: Optional[str] = None) -> None:
        """Update progress for an operation."""
        with self._progress_lock:
            if progress_id in self.active_progress_indicators:
                indicator = self.active_progress_indicators[progress_id]
                indicator.update_progress(step, step_name, speed_info)
                
                # Display progress
                self._display_progress(indicator)
    
    def _display_progress(self, indicator: ProgressIndicator) -> None:
        """Display progress information to the user."""
        progress_bar = indicator.get_progress_bar(50)
        time_remaining = indicator.get_time_remaining_str()
        
        # Clear line and display progress
        print(f"\r{indicator.operation_name}: {progress_bar}", end="")
        print(f" | {indicator.current_step_name}", end="")
        if indicator.speed_info:
            print(f" | {indicator.speed_info}", end="")
        print(f" | {time_remaining}", end="", flush=True)
        
        # New line when complete
        if indicator.current_step >= indicator.total_steps:
            print()  # New line
    
    def complete_progress(self, progress_id: str) -> None:
        """Mark progress as complete and clean up."""
        with self._progress_lock:
            if progress_id in self.active_progress_indicators:
                indicator = self.active_progress_indicators[progress_id]
                indicator.current_step = indicator.total_steps
                indicator.progress_percentage = 100.0
                indicator.current_step_name = "Completed"
                self._display_progress(indicator)
                del self.active_progress_indicators[progress_id]
    
    def explain_recovery_strategy(self, strategy: RecoveryStrategy) -> str:
        """Generate detailed explanation of a recovery strategy."""
        lines = [
            f"🔧 Recovery Strategy: {strategy.name}",
            "=" * 60,
            "",
            f"Description: {strategy.description}",
            f"Success Likelihood: {int(strategy.success_likelihood * 100)}%",
            f"Estimated Time: {strategy.estimated_time_minutes} minutes",
            ""
        ]
        
        if strategy.prerequisites:
            lines.extend([
                "Prerequisites:",
                ""
            ])
            for prereq in strategy.prerequisites:
                lines.append(f"  ✓ {prereq}")
            lines.append("")
        
        if strategy.risks:
            lines.extend([
                "⚠️  Potential Risks:",
                ""
            ])
            for risk in strategy.risks:
                lines.append(f"  • {risk}")
            lines.append("")
        
        lines.extend([
            "Steps to Execute:",
            ""
        ])
        
        for i, step in enumerate(strategy.steps, 1):
            lines.append(f"  {i}. {step.get('description', 'Unknown step')}")
            if step.get('command'):
                lines.append(f"     Command: {step['command']}")
            if step.get('expected_result'):
                lines.append(f"     Expected: {step['expected_result']}")
            lines.append("")
        
        return "\n".join(lines)
    
    def execute_recovery_strategy(self, strategy: RecoveryStrategy, 
                                callback: Optional[Callable] = None) -> bool:
        """Execute a recovery strategy with progress tracking."""
        strategy.status = RecoveryStatus.IN_PROGRESS
        strategy.start_time = datetime.now()
        
        progress_id = self.create_progress_indicator(
            f"Recovery: {strategy.name}",
            len(strategy.steps),
            "Preparing..."
        )
        
        try:
            for i, step in enumerate(strategy.steps):
                step_name = step.get('description', f'Step {i+1}')
                self.update_progress(progress_id, i, step_name)
                
                # Execute step (this would be implemented by specific recovery handlers)
                if callback:
                    success = callback(step)
                    if not success:
                        strategy.status = RecoveryStatus.FAILED
                        strategy.error_message = f"Failed at step {i+1}: {step_name}"
                        return False
                
                # Simulate step execution time
                time.sleep(0.5)  # Remove in actual implementation
            
            self.complete_progress(progress_id)
            strategy.status = RecoveryStatus.COMPLETED
            strategy.completion_time = datetime.now()
            
            # Record successful recovery
            self.recovery_history.append({
                'strategy_name': strategy.name,
                'success': True,
                'duration_seconds': (strategy.completion_time - strategy.start_time).total_seconds(),
                'timestamp': strategy.completion_time
            })
            
            return True
            
        except Exception as e:
            strategy.status = RecoveryStatus.FAILED
            strategy.error_message = str(e)
            self.logger.error(f"Recovery strategy failed: {e}")
            return False
    
    def get_troubleshooting_guide(self, guide_name: str) -> Optional[TroubleshootingGuide]:
        """Get a specific troubleshooting guide."""
        return self.troubleshooting_guides.get(guide_name)
    
    def list_available_guides(self) -> List[Tuple[str, str, str]]:
        """List all available troubleshooting guides."""
        return [
            (key, guide.title, guide.difficulty)
            for key, guide in self.troubleshooting_guides.items()
        ]
    
    def format_troubleshooting_guide(self, guide: TroubleshootingGuide) -> str:
        """Format a troubleshooting guide for display."""
        lines = [
            "=" * 80,
            f"📖 {guide.title}",
            "=" * 80,
            "",
            "Problem Description:",
            f"  {guide.problem_description}",
            "",
            "Common Symptoms:",
        ]
        
        for symptom in guide.symptoms:
            lines.append(f"  • {symptom}")
        
        lines.extend([
            "",
            f"Difficulty: {guide.difficulty.title()} | Estimated Time: {guide.estimated_time}",
            "",
            "Step-by-Step Solution:",
            ""
        ])
        
        for solution in guide.solutions:
            lines.append(f"Step {solution['step']}: {solution['description']}")
            if solution.get('command'):
                lines.append(f"  Command: {solution['command']}")
            lines.append("")
        
        lines.extend([
            "=" * 80,
            ""
        ])
        
        return "\n".join(lines)
    
    def run_interactive_troubleshooter(self) -> None:
        """Run an interactive troubleshooting session."""
        print("\n🔧 WAN2.2 Interactive Troubleshooter")
        print("=" * 50)
        print("\nI'll help you diagnose and fix installation problems.")
        print("Let's start with a quick system check...\n")
        
        # Run quick health check
        health_check = self.diagnostic_tool.get_quick_health_check()
        
        issues_found = []
        if not health_check["python_ok"]:
            issues_found.append("python_not_found")
        if not health_check["memory_ok"]:
            issues_found.append("insufficient_resources")
        if not health_check["disk_ok"]:
            issues_found.append("insufficient_resources")
        if not health_check["network_ok"]:
            issues_found.append("network_issues")
        
        if not issues_found:
            print("✅ Quick health check passed! Your system looks good.")
            print("\nIf you're still experiencing issues, you can:")
            print("1. Run full diagnostics: python scripts/diagnostic_tool.py")
            print("2. Browse troubleshooting guides below")
        else:
            print("⚠️  Found some potential issues:")
            for issue in issues_found:
                guide = self.troubleshooting_guides.get(issue)
                if guide:
                    print(f"  • {guide.title}")
        
        print("\n📚 Available Troubleshooting Guides:")
        print("-" * 40)
        
        guides = self.list_available_guides()
        for i, (key, title, difficulty) in enumerate(guides, 1):
            print(f"{i:2d}. {title} ({difficulty})")
        
        print(f"{len(guides) + 1:2d}. Run full system diagnostics")
        print(f"{len(guides) + 2:2d}. Exit")
        
        while True:
            try:
                choice = input(f"\nSelect an option (1-{len(guides) + 2}): ").strip()
                
                if choice == str(len(guides) + 2):  # Exit
                    print("Goodbye! Feel free to run this troubleshooter again if needed.")
                    break
                elif choice == str(len(guides) + 1):  # Full diagnostics
                    print("\nRunning full system diagnostics...")
                    diagnostics = self.diagnostic_tool.run_full_diagnostics()
                    report = self.diagnostic_tool.generate_diagnostic_report()
                    print(report)
                    
                    # Save report
                    report_file = Path(self.installation_path) / "logs" / "diagnostic_report.txt"
                    self.diagnostic_tool.generate_diagnostic_report(str(report_file))
                    print(f"\nFull report saved to: {report_file}")
                    
                else:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(guides):
                        guide_key, _, _ = guides[choice_idx]
                        guide = self.troubleshooting_guides[guide_key]
                        print(self.format_troubleshooting_guide(guide))
                        
                        input("Press Enter to continue...")
                    else:
                        print("Invalid choice. Please try again.")
                        
            except (ValueError, KeyboardInterrupt):
                print("\nExiting troubleshooter...")
                break
    
    def generate_support_ticket(self, error: InstallationError, 
                              context: Optional[Dict[str, Any]] = None,
                              steps_attempted: Optional[List[str]] = None) -> SupportTicket:
        """Generate a pre-filled support ticket for an error."""
        # Collect system information
        system_info = self._collect_system_info()
        
        # Collect environment details
        environment_details = self._collect_environment_details()
        
        # Collect recent logs
        logs = self._collect_recent_logs()
        
        # Determine severity based on error category
        severity_map = {
            ErrorCategory.SYSTEM: "high",
            ErrorCategory.NETWORK: "medium",
            ErrorCategory.PERMISSION: "medium",
            ErrorCategory.CONFIGURATION: "low"
        }
        severity = severity_map.get(error.category, "medium")
        
        # Create ticket
        ticket = SupportTicket(
            title=f"Installation Error: {error.message[:50]}...",
            description=self._generate_ticket_description(error, context),
            error_details=self._format_error_details(error, context),
            system_info=system_info,
            logs=logs,
            steps_attempted=steps_attempted or [],
            recovery_strategies_tried=[h['strategy_name'] for h in self.recovery_history[-5:]],
            severity=severity,
            environment_details=environment_details
        )
        
        return ticket
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect comprehensive system information."""
        try:
            return {
                "OS": f"{platform.system()} {platform.release()}",
                "Python Version": platform.python_version(),
                "Architecture": platform.machine(),
                "Processor": platform.processor(),
                "Total RAM": f"{psutil.virtual_memory().total // (1024**3)} GB",
                "Available RAM": f"{psutil.virtual_memory().available // (1024**3)} GB",
                "Disk Space": f"{psutil.disk_usage('/').free // (1024**3)} GB free",
                "CPU Count": psutil.cpu_count(),
                "CPU Usage": f"{psutil.cpu_percent()}%"
            }
        except Exception as e:
            self.logger.warning(f"Failed to collect system info: {e}")
            return {"Error": "Failed to collect system information"}
    
    def _collect_environment_details(self) -> Dict[str, Any]:
        """Collect environment-specific details."""
        import os
        
        details = {}
        
        # Environment variables
        relevant_vars = ['PATH', 'PYTHONPATH', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES']
        for var in relevant_vars:
            if var in os.environ:
                details[f"ENV_{var}"] = os.environ[var][:200]  # Truncate long paths
        
        # Python packages (if available)
        try:
            import pkg_resources
            installed_packages = [f"{pkg.project_name}=={pkg.version}" 
                                for pkg in pkg_resources.working_set]
            details["Installed Packages"] = installed_packages[:20]  # First 20 packages
        except:
            details["Installed Packages"] = "Unable to collect"
        
        return details
    
    def _collect_recent_logs(self) -> List[str]:
        """Collect recent log entries."""
        logs = []
        log_dir = Path(self.installation_path) / "logs"
        
        if log_dir.exists():
            for log_file in log_dir.glob("*.log"):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # Get last 10 lines from each log file
                        recent_lines = lines[-10:] if len(lines) > 10 else lines
                        logs.extend([f"[{log_file.name}] {line.strip()}" for line in recent_lines])
                except Exception as e:
                    logs.append(f"[{log_file.name}] Error reading log: {e}")
        
        return logs[-50:]  # Return last 50 log entries total
    
    def _generate_ticket_description(self, error: InstallationError, 
                                   context: Optional[Dict[str, Any]]) -> str:
        """Generate a detailed description for the support ticket."""
        lines = [
            "I encountered an error during WAN2.2 installation that I cannot resolve.",
            "",
            f"Error Category: {error.category}",
            f"Error Message: {error.message}",
            ""
        ]
        
        if context:
            lines.extend([
                "Context:",
                f"- Installation Phase: {context.get('phase', 'Unknown')}",
                f"- Component: {context.get('component', 'Unknown')}",
                f"- Timestamp: {context.get('timestamp', 'Unknown')}",
                ""
            ])
        
        lines.extend([
            "I have attempted to resolve this issue using the troubleshooting guides",
            "and recovery strategies, but the problem persists. Please provide assistance.",
            "",
            "Additional details are provided in the system information and logs below."
        ])
        
        return "\n".join(lines)
    
    def _format_error_details(self, error: InstallationError, 
                            context: Optional[Dict[str, Any]]) -> str:
        """Format detailed error information."""
        lines = [
            f"Error: {error.message}",
            f"Category: {error.category}",
            f"Timestamp: {datetime.now().isoformat()}",
        ]
        
        if hasattr(error, 'stack_trace') and error.stack_trace:
            lines.extend([
                "",
                "Stack Trace:",
                error.stack_trace
            ])
        
        if context:
            lines.extend([
                "",
                "Context Information:",
                json.dumps(context, indent=2, default=str)
            ])
        
        return "\n".join(lines)
    
    def save_support_ticket(self, ticket: SupportTicket, 
                          output_path: Optional[str] = None) -> str:
        """Save support ticket to file and return the path."""
        if output_path is None:
            timestamp = ticket.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"support_ticket_{timestamp}.md"
            output_path = str(Path(self.installation_path) / "logs" / filename)
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save ticket
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ticket.to_markdown())
        
        self.logger.info(f"Support ticket saved to: {output_path}")
        return output_path
    
    def create_support_ticket_url(self, ticket: SupportTicket, 
                                platform: str = "github") -> str:
        """Create a URL for submitting the support ticket."""
        if platform == "github":
            base_url = "https://github.com/wan22/issues/new"
            params = {
                'title': ticket.title,
                'body': ticket.to_markdown()[:8000]  # GitHub has body length limits
            }
            return f"{base_url}?{urllib.parse.urlencode(params)}"
        
        # Add other platforms as needed
        return ""
    
    def generate_help_documentation(self, output_dir: Optional[str] = None) -> None:
        """Generate comprehensive help documentation."""
        if output_dir is None:
            output_dir = str(Path(self.installation_path) / "docs")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate main troubleshooting guide
        main_guide_lines = [
            "# WAN2.2 Installation Troubleshooting Guide",
            "",
            "This guide helps you resolve common installation issues.",
            "",
            "## Quick Start",
            "",
            "1. **Run the diagnostic tool**: `python scripts/diagnostic_tool.py`",
            "2. **Check system requirements**: Ensure you have Python 3.9+, 8GB+ RAM, 50GB+ disk space",
            "3. **Run as Administrator**: Right-click install.bat and select 'Run as administrator'",
            "",
            "## Common Issues and Solutions",
            ""
        ]
        
        for guide_key, guide in self.troubleshooting_guides.items():
            main_guide_lines.extend([
                f"### {guide.title}",
                "",
                f"**Problem**: {guide.problem_description}",
                "",
                "**Symptoms**:",
            ])
            
            for symptom in guide.symptoms:
                main_guide_lines.append(f"- {symptom}")
            
            main_guide_lines.extend([
                "",
                "**Solutions**:",
            ])
            
            for solution in guide.solutions:
                main_guide_lines.append(f"{solution['step']}. {solution['description']}")
                if solution.get('command'):
                    main_guide_lines.append(f"   ```")
                    main_guide_lines.append(f"   {solution['command']}")
                    main_guide_lines.append(f"   ```")
            
            main_guide_lines.extend([
                "",
                f"**Difficulty**: {guide.difficulty.title()} | **Time**: {guide.estimated_time}",
                "",
                "---",
                ""
            ])
        
        # Save main guide
        main_guide_file = output_path / "TROUBLESHOOTING.md"
        main_guide_file.write_text("\n".join(main_guide_lines), encoding='utf-8')
        
        # Generate FAQ
        faq_lines = [
            "# Frequently Asked Questions (FAQ)",
            "",
            "## Installation Questions",
            "",
            "**Q: What are the minimum system requirements?**",
            "A: Python 3.9+, 8GB RAM, 50GB disk space, Windows 10/11. GPU with 8GB+ VRAM recommended.",
            "",
            "**Q: How long does installation take?**",
            "A: Typically 30-60 minutes depending on internet speed and system performance.",
            "",
            "**Q: Can I install without internet?**",
            "A: No, internet connection is required to download Python packages and models.",
            "",
            "**Q: Do I need administrator privileges?**",
            "A: Yes, running as administrator is recommended to avoid permission issues.",
            "",
            "## Usage Questions",
            "",
            "**Q: How do I update the installation?**",
            "A: Run the installer again - it will detect existing installation and update components.",
            "",
            "**Q: Can I move the installation to another location?**",
            "A: Yes, but you'll need to run the installer again in the new location.",
            "",
            "**Q: How do I uninstall?**",
            "A: Simply delete the installation directory. No system-wide changes are made.",
            "",
            "## Troubleshooting Questions",
            "",
            "**Q: Installation fails with permission errors**",
            "A: Run as administrator and ensure antivirus isn't blocking the installation.",
            "",
            "**Q: Models fail to download**",
            "A: Check internet connection, firewall settings, and available disk space.",
            "",
            "**Q: GPU not detected**",
            "A: Update NVIDIA drivers and install CUDA toolkit. Restart after installation.",
            ""
        ]
        
        faq_file = output_path / "FAQ.md"
        faq_file.write_text("\n".join(faq_lines), encoding='utf-8')
        
        self.logger.info(f"Help documentation generated in: {output_path}")
        print(f"📚 Help documentation generated in: {output_path}")
        print(f"  • Troubleshooting Guide: {main_guide_file}")
        print(f"  • FAQ: {faq_file}")


def main():
    """Enhanced main function for running the user guidance system."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="WAN2.2 Enhanced User Guidance System")
    parser.add_argument("--path", default=".", help="Installation path")
    parser.add_argument("--interactive", action="store_true", help="Run interactive troubleshooter")
    parser.add_argument("--support-ticket", action="store_true", help="Generate support ticket")
    parser.add_argument("--guide", help="Show specific troubleshooting guide")
    parser.add_argument("--test-progress", action="store_true", help="Test progress indicators")
    parser.add_argument("--test-recovery", action="store_true", help="Test recovery strategy explanation")
    
    args = parser.parse_args()
    
    guidance = UserGuidanceSystem(args.path)
    
    if args.interactive:
        guidance.run_interactive_troubleshooter()
    elif args.support_ticket:
        # Demo support ticket generation
        from interfaces import InstallationError, ErrorCategory
        demo_error = InstallationError(
            "Demo installation error for testing",
            ErrorCategory.SYSTEM,
            recovery_suggestions=["Check system requirements", "Run as administrator"]
        )
        ticket = guidance.generate_support_ticket(
            demo_error,
            context={"phase": "demo", "component": "test"},
            steps_attempted=["Ran diagnostics", "Checked permissions"]
        )
        ticket_path = guidance.save_support_ticket(ticket)
        print(f"Demo support ticket generated: {ticket_path}")
        print(f"GitHub URL: {guidance.create_support_ticket_url(ticket)}")
    elif args.guide:
        guide = guidance.get_troubleshooting_guide(args.guide)
        if guide:
            print(guidance.format_troubleshooting_guide(guide))
        else:
            print(f"Guide '{args.guide}' not found. Available guides:")
            for key, title, difficulty in guidance.list_available_guides():
                print(f"  {key}: {title} ({difficulty})")
    elif args.test_progress:
        # Demo progress indicators
        print("Testing progress indicators...")
        progress_id = guidance.create_progress_indicator("Demo Operation", 5)
        for i in range(1, 6):
            guidance.update_progress(progress_id, i, f"Step {i}", f"{i*20}% complete")
            time.sleep(1)
        guidance.complete_progress(progress_id)
        print("\nProgress test completed!")
    elif args.test_recovery:
        # Demo recovery strategy explanation
        demo_strategy = RecoveryStrategy(
            name="Demo Recovery Strategy",
            description="This is a demonstration of recovery strategy explanation",
            success_likelihood=0.85,
            estimated_time_minutes=10,
            steps=[
                {"description": "Check system requirements", "command": "python --version"},
                {"description": "Verify network connectivity", "command": "ping google.com"},
                {"description": "Clear temporary files", "expected_result": "Disk space freed"}
            ],
            prerequisites=["Administrator access", "Internet connection"],
            risks=["May require system restart"]
        )
        print(guidance.explain_recovery_strategy(demo_strategy))
    else:
        guidance.run_interactive_troubleshooter()


if __name__ == "__main__":
    main()
