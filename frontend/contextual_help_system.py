"""
Contextual Help System for WAN22 System Optimization

This module provides contextual help for UI elements, settings, and optimization features.
It integrates with the documentation system to provide relevant help content.

Requirements addressed:
- 12.2: Create contextual help for optimization settings
- 12.2: Link relevant documentation sections in error messages
- 12.2: Add guided setup for first-time users with high-end hardware
"""

import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

from documentation_integration import DocumentationIntegration, DocumentationLink


class HelpContentType(Enum):
    """Types of help content"""
    TOOLTIP = "tooltip"
    MODAL = "modal"
    SIDEBAR = "sidebar"
    INLINE = "inline"
    GUIDED_TOUR = "guided_tour"


class HelpTrigger(Enum):
    """Help trigger events"""
    HOVER = "hover"
    CLICK = "click"
    FOCUS = "focus"
    ERROR = "error"
    FIRST_TIME = "first_time"


@dataclass
class HelpContent:
    """Help content for UI elements"""
    content_id: str
    title: str
    description: str
    content_type: HelpContentType
    trigger: HelpTrigger
    quick_tips: List[str]
    detailed_explanation: str
    documentation_links: List[DocumentationLink]
    related_settings: List[str]
    prerequisites: List[str]
    warnings: List[str]
    examples: List[Dict[str, Any]]
    video_url: Optional[str] = None
    priority: int = 0  # Higher priority content shown first


@dataclass
class GuidedTourStep:
    """Step in a guided tour"""
    step_id: str
    target_element: str
    title: str
    content: str
    position: str  # top, bottom, left, right
    action_required: bool
    validation_function: Optional[str]
    next_step_condition: Optional[str]


@dataclass
class ContextualHelpSession:
    """User's contextual help session"""
    session_id: str
    user_profile: str
    completed_tours: List[str]
    dismissed_tips: List[str]
    help_interactions: List[Dict[str, Any]]
    preferences: Dict[str, Any]


class ContextualHelpSystem:
    """
    Provides contextual help for UI elements and system features
    """
    
    def __init__(self, docs_base_path: str = "docs"):
        """
        Initialize contextual help system.
        
        Args:
            docs_base_path: Base path for documentation files
        """
        self.doc_integration = DocumentationIntegration(docs_base_path)
        self.logger = logging.getLogger(__name__)
        
        # Help content registry
        self._help_content: Dict[str, HelpContent] = {}
        self._guided_tours: Dict[str, List[GuidedTourStep]] = {}
        self._user_sessions: Dict[str, ContextualHelpSession] = {}
        
        # Initialize help content
        self._initialize_help_content()
        self._initialize_guided_tours()
    
    def get_help_content(self, 
                        element_id: str, 
                        trigger: HelpTrigger = HelpTrigger.HOVER,
                        user_context: Optional[Dict[str, Any]] = None) -> Optional[HelpContent]:
        """
        Get contextual help content for a UI element.
        
        Args:
            element_id: ID of the UI element
            trigger: How the help was triggered
            user_context: Additional user context
            
        Returns:
            Help content or None if not found
        """
        if element_id not in self._help_content:
            return None
        
        help_content = self._help_content[element_id]
        
        # Check if help should be shown based on trigger
        if help_content.trigger != trigger and trigger != HelpTrigger.CLICK:
            return None
        
        # Customize content based on user context
        if user_context:
            help_content = self._customize_help_content(help_content, user_context)
        
        # Log help interaction
        self._log_help_interaction(element_id, trigger, user_context)
        
        return help_content
    
    def get_guided_tour(self, tour_id: str, user_profile: str = "default") -> Optional[List[GuidedTourStep]]:
        """
        Get guided tour steps.
        
        Args:
            tour_id: ID of the guided tour
            user_profile: User profile for customization
            
        Returns:
            List of guided tour steps or None if not found
        """
        if tour_id not in self._guided_tours:
            return None
        
        tour_steps = self._guided_tours[tour_id].copy()
        
        # Customize tour based on user profile
        tour_steps = self._customize_guided_tour(tour_steps, user_profile)
        
        return tour_steps
    
    def start_user_session(self, user_id: str, user_profile: str = "default") -> str:
        """
        Start a contextual help session for a user.
        
        Args:
            user_id: User identifier
            user_profile: User profile type
            
        Returns:
            Session ID
        """
        session_id = f"help_session_{user_id}_{self._get_timestamp()}"
        
        session = ContextualHelpSession(
            session_id=session_id,
            user_profile=user_profile,
            completed_tours=[],
            dismissed_tips=[],
            help_interactions=[],
            preferences={}
        )
        
        self._user_sessions[session_id] = session
        
        self.logger.info(f"Started contextual help session: {session_id}")
        return session_id
    
    def should_show_first_time_help(self, 
                                   element_id: str, 
                                   session_id: str) -> bool:
        """
        Check if first-time help should be shown for an element.
        
        Args:
            element_id: ID of the UI element
            session_id: User session ID
            
        Returns:
            True if first-time help should be shown
        """
        if session_id not in self._user_sessions:
            return True
        
        session = self._user_sessions[session_id]
        
        # Check if user has dismissed this tip
        if element_id in session.dismissed_tips:
            return False
        
        # Check if user has interacted with this element before
        for interaction in session.help_interactions:
            if interaction.get("element_id") == element_id:
                return False
        
        return True
    
    def dismiss_help_tip(self, element_id: str, session_id: str):
        """
        Mark a help tip as dismissed by the user.
        
        Args:
            element_id: ID of the UI element
            session_id: User session ID
        """
        if session_id in self._user_sessions:
            session = self._user_sessions[session_id]
            if element_id not in session.dismissed_tips:
                session.dismissed_tips.append(element_id)
                self.logger.info(f"Help tip dismissed: {element_id} for session {session_id}")
    
    def complete_guided_tour(self, tour_id: str, session_id: str):
        """
        Mark a guided tour as completed.
        
        Args:
            tour_id: ID of the guided tour
            session_id: User session ID
        """
        if session_id in self._user_sessions:
            session = self._user_sessions[session_id]
            if tour_id not in session.completed_tours:
                session.completed_tours.append(tour_id)
                self.logger.info(f"Guided tour completed: {tour_id} for session {session_id}")
    
    def get_recommended_help_content(self, 
                                   user_context: Dict[str, Any],
                                   session_id: Optional[str] = None) -> List[HelpContent]:
        """
        Get recommended help content based on user context.
        
        Args:
            user_context: Current user context (current page, settings, etc.)
            session_id: Optional user session ID
            
        Returns:
            List of recommended help content
        """
        recommendations = []
        
        # Get context-based recommendations
        current_page = user_context.get("current_page", "")
        current_settings = user_context.get("current_settings", {})
        
        # Find relevant help content
        for content_id, help_content in self._help_content.items():
            relevance_score = self._calculate_help_relevance(
                help_content, current_page, current_settings, session_id
            )
            
            if relevance_score > 0.5:  # Threshold for recommendations
                recommendations.append((help_content, relevance_score))
        
        # Sort by relevance and priority
        recommendations.sort(key=lambda x: (x[1], x[0].priority), reverse=True)
        
        return [content for content, _ in recommendations[:5]]  # Top 5 recommendations
    
    def create_dynamic_help_content(self, 
                                  element_id: str,
                                  error_context: Dict[str, Any]) -> HelpContent:
        """
        Create dynamic help content based on error context.
        
        Args:
            element_id: ID of the UI element
            error_context: Context about the error or issue
            
        Returns:
            Dynamic help content
        """
        error_category = error_context.get("category", "unknown")
        error_message = error_context.get("message", "")
        
        # Get relevant documentation links
        doc_links = self.doc_integration.get_error_documentation_links(
            error_category, error_message, error_context
        )
        
        # Generate dynamic content
        title = f"Help for {element_id.replace('_', ' ').title()}"
        description = self._generate_dynamic_description(element_id, error_context)
        quick_tips = self._generate_dynamic_tips(error_category, error_context)
        
        return HelpContent(
            content_id=f"dynamic_{element_id}_{hash(str(error_context)) % 10000:04d}",
            title=title,
            description=description,
            content_type=HelpContentType.MODAL,
            trigger=HelpTrigger.ERROR,
            quick_tips=quick_tips,
            detailed_explanation=self._generate_detailed_explanation(error_category, error_context),
            documentation_links=doc_links,
            related_settings=error_context.get("related_settings", []),
            prerequisites=[],
            warnings=error_context.get("warnings", []),
            examples=[],
            priority=10  # High priority for error-based help
        )
    
    def _initialize_help_content(self):
        """Initialize help content for UI elements"""
        
        # VRAM Optimization Help
        self._help_content["vram_optimization_toggle"] = HelpContent(
            content_id="vram_optimization_toggle",
            title="VRAM Optimization",
            description="Automatically manages GPU memory usage to prevent out-of-memory errors and optimize performance.",
            content_type=HelpContentType.TOOLTIP,
            trigger=HelpTrigger.HOVER,
            quick_tips=[
                "Recommended for RTX 4080: Enable with 90% threshold",
                "Automatically applies CPU offloading when needed",
                "Prevents crashes during large model loading",
                "Can improve generation speed by reducing memory pressure"
            ],
            detailed_explanation="""
VRAM Optimization automatically monitors your GPU memory usage and applies optimization techniques when usage approaches the configured threshold.

**How it works:**
1. Continuously monitors VRAM usage
2. When usage exceeds threshold (default 90%), applies optimizations:
   - Moves text encoder to CPU memory
   - Moves VAE to CPU memory when not in use
   - Enables gradient checkpointing for large models
   - Clears unnecessary cached data

**Benefits:**
- Prevents out-of-memory crashes
- Allows loading larger models on limited VRAM
- Maintains performance while optimizing memory usage
- Automatic operation requires no manual intervention

**Recommended Settings:**
- RTX 4080 (16GB): 90% threshold
- RTX 3080 (10-12GB): 85% threshold
- RTX 3060 (12GB): 80% threshold
            """.strip(),
            documentation_links=[
                DocumentationLink(
                    title="VRAM Management Guide",
                    url="WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md#vram-management",
                    section="VRAM Management",
                    description="Complete guide to VRAM optimization",
                    relevance_score=1.0,
                    doc_type=self.doc_integration._documentation_urls[self.doc_integration.DocumentationType.USER_GUIDE]
                )
            ],
            related_settings=["vram_threshold", "cpu_offload_enabled", "gradient_checkpointing"],
            prerequisites=["NVIDIA GPU with CUDA support", "Sufficient system RAM for CPU offloading"],
            warnings=[
                "CPU offloading may slightly reduce performance",
                "Requires adequate system RAM (32GB+ recommended)"
            ],
            examples=[
                {
                    "scenario": "Loading TI2V-5B on RTX 4080",
                    "settings": {"threshold": 0.9, "cpu_offload": True},
                    "result": "Model loads successfully in 12GB instead of 18GB"
                }
            ]
        )
        
        # Quantization Strategy Help
        self._help_content["quantization_strategy"] = HelpContent(
            content_id="quantization_strategy",
            title="Quantization Strategy",
            description="Reduces model precision to save VRAM with minimal impact on generation quality.",
            content_type=HelpContentType.MODAL,
            trigger=HelpTrigger.CLICK,
            quick_tips=[
                "bf16: Best balance of quality and performance",
                "int8: Maximum VRAM savings, slight quality reduction",
                "FP8: Experimental, highest performance on RTX 4080",
                "none: Highest quality, maximum VRAM usage"
            ],
            detailed_explanation="""
Quantization reduces the precision of model weights and activations to save GPU memory while maintaining acceptable generation quality.

**Available Strategies:**

**bf16 (Brain Float 16) - Recommended:**
- Reduces VRAM usage by ~30%
- Minimal quality impact (<2%)
- Good performance on RTX 4080
- Best balance for most users

**int8 (8-bit Integer):**
- Reduces VRAM usage by ~50%
- Small quality impact (~5%)
- Slower than bf16 but uses less memory
- Good for VRAM-constrained systems

**FP8 (8-bit Float) - Experimental:**
- Maximum performance on supported hardware
- Requires RTX 4080 or newer
- Similar quality to bf16
- Fastest generation speed

**none (No Quantization):**
- Highest quality output
- Maximum VRAM usage
- Slowest generation
- Only for high-VRAM systems (24GB+)

**Automatic Selection:**
The system can automatically choose the best strategy based on your hardware and available VRAM.
            """.strip(),
            documentation_links=[
                DocumentationLink(
                    title="Quantization Settings Guide",
                    url="WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md#quantization-settings",
                    section="Quantization Settings",
                    description="Detailed quantization configuration",
                    relevance_score=1.0,
                    doc_type=self.doc_integration._documentation_urls[self.doc_integration.DocumentationType.USER_GUIDE]
                )
            ],
            related_settings=["quantization_timeout", "fallback_enabled", "quality_validation"],
            prerequisites=["Compatible GPU (RTX 3060 or newer)", "Sufficient VRAM for chosen strategy"],
            warnings=[
                "int8 quantization may cause slight quality reduction",
                "FP8 is experimental and may not work on all models",
                "Quantization process can take 5-10 minutes"
            ],
            examples=[
                {
                    "scenario": "RTX 4080 with TI2V-5B",
                    "recommended_strategy": "bf16",
                    "vram_usage": "12GB instead of 18GB",
                    "quality_impact": "<2%"
                },
                {
                    "scenario": "RTX 3060 12GB",
                    "recommended_strategy": "int8",
                    "vram_usage": "8GB instead of 16GB",
                    "quality_impact": "~5%"
                }
            ]
        )
        
        # Hardware Optimization Help
        self._help_content["hardware_optimization"] = HelpContent(
            content_id="hardware_optimization",
            title="Hardware Optimization",
            description="Applies optimizations specific to your detected hardware configuration for maximum performance.",
            content_type=HelpContentType.SIDEBAR,
            trigger=HelpTrigger.FOCUS,
            quick_tips=[
                "Automatically detects RTX 4080 and Threadripper PRO",
                "Enables tensor core acceleration",
                "Optimizes memory allocation patterns",
                "Can improve performance by 40-60%"
            ],
            detailed_explanation="""
Hardware Optimization automatically detects your system configuration and applies specific optimizations for your hardware.

**RTX 4080 Optimizations:**
- Tensor Core Utilization: Enables 4th-gen tensor cores for mixed precision
- Memory Optimization: Configures optimal memory allocation patterns
- Tile Size Optimization: Sets optimal tile sizes (256x256 for VAE)
- Bandwidth Optimization: Optimizes memory bandwidth usage

**Threadripper PRO Optimizations:**
- Multi-Core Utilization: Uses up to 32 cores for preprocessing
- NUMA Awareness: Allocates memory on local NUMA nodes
- Thread Affinity: Binds threads to optimal CPU cores
- Cache Optimization: Optimizes CPU cache usage patterns

**Memory Optimizations:**
- High-Speed RAM: Optimizes for DDR4-3200+ or DDR5
- Large Memory Pools: Configures large memory pools for 128GB+ systems
- Memory Interleaving: Enables memory channel interleaving

**Expected Performance Improvements:**
- Generation Speed: 40-60% faster
- Model Loading: 30-50% faster
- Memory Efficiency: 20-30% better utilization
- System Stability: Reduced thermal throttling
            """.strip(),
            documentation_links=[
                DocumentationLink(
                    title="Hardware Optimization Guide",
                    url="WAN22_PERFORMANCE_OPTIMIZATION_GUIDE.md#hardware-specific-optimizations",
                    section="Hardware-Specific Optimizations",
                    description="Detailed hardware optimization guide",
                    relevance_score=1.0,
                    doc_type=self.doc_integration._documentation_urls[self.doc_integration.DocumentationType.PERFORMANCE_OPTIMIZATION]
                )
            ],
            related_settings=["tensor_cores_enabled", "cpu_cores_utilized", "memory_pool_size"],
            prerequisites=["Compatible hardware (RTX 3060+ GPU)", "Adequate cooling for sustained performance"],
            warnings=[
                "May increase power consumption and heat generation",
                "Ensure adequate cooling for sustained performance",
                "Some optimizations require system restart"
            ],
            examples=[
                {
                    "hardware": "RTX 4080 + Threadripper PRO 5995WX",
                    "optimizations": ["tensor_cores", "multi_core", "numa_aware"],
                    "performance_gain": "60% faster generation"
                }
            ]
        )
        
        # System Health Monitoring Help
        self._help_content["health_monitoring"] = HelpContent(
            content_id="health_monitoring",
            title="System Health Monitoring",
            description="Continuously monitors system health and provides alerts to prevent hardware damage and maintain optimal performance.",
            content_type=HelpContentType.INLINE,
            trigger=HelpTrigger.FIRST_TIME,
            quick_tips=[
                "Monitors GPU temperature, VRAM, CPU, and memory usage",
                "Provides automatic alerts for safety thresholds",
                "Prevents hardware damage from overheating",
                "Maintains performance by preventing throttling"
            ],
            detailed_explanation="""
System Health Monitoring provides continuous oversight of your system's vital signs to ensure safe and optimal operation.

**Monitored Metrics:**
- GPU Temperature: Prevents overheating damage
- VRAM Usage: Prevents out-of-memory crashes
- CPU Usage: Monitors processing load
- System Memory: Tracks RAM usage
- Power Consumption: Monitors power draw
- Fan Speeds: Ensures adequate cooling

**Alert Thresholds:**
- GPU Temperature: Warning at 80°C, Critical at 85°C
- VRAM Usage: Warning at 90%, Critical at 95%
- CPU Usage: Warning at 80%, Critical at 90%
- Memory Usage: Warning at 85%, Critical at 95%

**Automatic Actions:**
- Performance Reduction: Reduces workload when thresholds exceeded
- Emergency Shutdown: Safe shutdown for critical conditions
- Cooling Optimization: Adjusts fan curves and power limits
- Workload Balancing: Distributes load across available resources

**Benefits:**
- Hardware Protection: Prevents damage from overheating
- Performance Optimization: Maintains optimal operating conditions
- Stability: Reduces crashes and system instability
- Longevity: Extends hardware lifespan through proper monitoring
            """.strip(),
            documentation_links=[
                DocumentationLink(
                    title="System Health Monitoring",
                    url="WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md#performance-monitoring",
                    section="Performance Monitoring",
                    description="System health monitoring setup",
                    relevance_score=1.0,
                    doc_type=self.doc_integration._documentation_urls[self.doc_integration.DocumentationType.USER_GUIDE]
                )
            ],
            related_settings=["monitoring_interval", "alert_thresholds", "automatic_actions"],
            prerequisites=["Hardware monitoring sensors", "Administrative privileges for system control"],
            warnings=[
                "Monitoring adds small system overhead (~1-2%)",
                "Automatic actions may interrupt generation processes",
                "Requires proper sensor drivers for accurate readings"
            ],
            examples=[
                {
                    "scenario": "GPU overheating during long generation",
                    "action": "Automatic performance reduction and increased cooling",
                    "result": "Temperature reduced, generation continues safely"
                }
            ]
        )
    
    def _initialize_guided_tours(self):
        """Initialize guided tours"""
        
        # First-time setup tour
        self._guided_tours["first_time_setup"] = [
            GuidedTourStep(
                step_id="welcome",
                target_element="main_interface",
                title="Welcome to WAN22 System Optimization",
                content="Let's optimize your system for the best performance. This tour will guide you through the essential settings.",
                position="center",
                action_required=False,
                validation_function=None,
                next_step_condition=None
            ),
            GuidedTourStep(
                step_id="hardware_detection",
                target_element="hardware_info_panel",
                title="Hardware Detection",
                content="We've detected your hardware configuration. Verify that your GPU and CPU are correctly identified.",
                position="right",
                action_required=True,
                validation_function="validate_hardware_detection",
                next_step_condition="hardware_confirmed"
            ),
            GuidedTourStep(
                step_id="vram_optimization",
                target_element="vram_optimization_toggle",
                title="Enable VRAM Optimization",
                content="Enable VRAM optimization to prevent out-of-memory errors and optimize performance for your GPU.",
                position="bottom",
                action_required=True,
                validation_function="validate_vram_optimization_enabled",
                next_step_condition="vram_optimization_enabled"
            ),
            GuidedTourStep(
                step_id="quantization_setup",
                target_element="quantization_strategy",
                title="Configure Quantization",
                content="Select the optimal quantization strategy for your hardware. We recommend bf16 for RTX 4080.",
                position="left",
                action_required=True,
                validation_function="validate_quantization_configured",
                next_step_condition="quantization_configured"
            ),
            GuidedTourStep(
                step_id="hardware_optimization",
                target_element="hardware_optimization",
                title="Apply Hardware Optimizations",
                content="Enable hardware-specific optimizations to maximize performance for your configuration.",
                position="top",
                action_required=True,
                validation_function="validate_hardware_optimizations_enabled",
                next_step_condition="hardware_optimizations_enabled"
            ),
            GuidedTourStep(
                step_id="completion",
                target_element="main_interface",
                title="Setup Complete!",
                content="Your system is now optimized for maximum performance. You can always adjust these settings later.",
                position="center",
                action_required=False,
                validation_function=None,
                next_step_condition=None
            )
        ]
        
        # Advanced optimization tour
        self._guided_tours["advanced_optimization"] = [
            GuidedTourStep(
                step_id="performance_monitoring",
                target_element="health_monitoring",
                title="System Health Monitoring",
                content="Enable continuous monitoring to protect your hardware and maintain optimal performance.",
                position="right",
                action_required=True,
                validation_function="validate_monitoring_enabled",
                next_step_condition="monitoring_enabled"
            ),
            GuidedTourStep(
                step_id="advanced_settings",
                target_element="advanced_settings_panel",
                title="Advanced Settings",
                content="Fine-tune advanced settings for your specific use case and hardware configuration.",
                position="left",
                action_required=False,
                validation_function=None,
                next_step_condition=None
            ),
            GuidedTourStep(
                step_id="performance_validation",
                target_element="benchmark_button",
                title="Validate Performance",
                content="Run performance benchmarks to verify that optimizations are working correctly.",
                position="bottom",
                action_required=True,
                validation_function="validate_benchmarks_run",
                next_step_condition="benchmarks_completed"
            )
        ]
    
    def _customize_help_content(self, 
                              help_content: HelpContent, 
                              user_context: Dict[str, Any]) -> HelpContent:
        """Customize help content based on user context"""
        # Create a copy to avoid modifying the original
        customized_content = HelpContent(**asdict(help_content))
        
        # Customize based on hardware profile
        hardware_profile = user_context.get("hardware_profile", "unknown")
        if hardware_profile == "high_end":
            # Add high-end specific tips
            customized_content.quick_tips.extend([
                "Your high-end hardware supports aggressive optimizations",
                "Consider enabling experimental features for maximum performance"
            ])
        elif hardware_profile == "budget":
            # Add budget-friendly tips
            customized_content.quick_tips.extend([
                "Focus on memory-saving optimizations",
                "Consider more aggressive quantization settings"
            ])
        
        return customized_content
    
    def _customize_guided_tour(self, 
                             tour_steps: List[GuidedTourStep], 
                             user_profile: str) -> List[GuidedTourStep]:
        """Customize guided tour based on user profile"""
        if user_profile == "advanced":
            # Skip basic steps for advanced users
            return [step for step in tour_steps if step.step_id not in ["welcome", "hardware_detection"]]
        elif user_profile == "beginner":
            # Add more detailed explanations for beginners
            for step in tour_steps:
                if step.step_id == "quantization_setup":
                    step.content += " Don't worry if this seems complex - the default settings work well for most users."
        
        return tour_steps
    
    def _calculate_help_relevance(self, 
                                help_content: HelpContent,
                                current_page: str,
                                current_settings: Dict[str, Any],
                                session_id: Optional[str]) -> float:
        """Calculate relevance score for help content"""
        relevance_score = 0.0
        
        # Page relevance
        if current_page in help_content.content_id:
            relevance_score += 0.5
        
        # Settings relevance
        for setting in help_content.related_settings:
            if setting in current_settings:
                relevance_score += 0.2
        
        # User session relevance
        if session_id and session_id in self._user_sessions:
            session = self._user_sessions[session_id]
            # Lower relevance for dismissed tips
            if help_content.content_id in session.dismissed_tips:
                relevance_score -= 0.3
        
        # Priority boost
        relevance_score += help_content.priority * 0.1
        
        return min(relevance_score, 1.0)  # Cap at 1.0
    
    def _generate_dynamic_description(self, 
                                    element_id: str, 
                                    error_context: Dict[str, Any]) -> str:
        """Generate dynamic description based on error context"""
        error_category = error_context.get("category", "unknown")
        
        descriptions = {
            "vram_memory": f"This setting helps manage GPU memory usage. The current error suggests your GPU is running out of memory.",
            "quantization": f"Quantization can help reduce memory usage. Consider enabling it to resolve the current issue.",
            "hardware_optimization": f"Hardware optimizations can improve performance and stability for your specific configuration.",
            "configuration": f"There may be an issue with your current configuration. This setting can help resolve it."
        }
        
        return descriptions.get(error_category, f"This setting is related to the current issue with {element_id}.")
    
    def _generate_dynamic_tips(self, 
                             error_category: str, 
                             error_context: Dict[str, Any]) -> List[str]:
        """Generate dynamic tips based on error context"""
        tips_map = {
            "vram_memory": [
                "Enable VRAM optimization to automatically manage memory",
                "Try reducing batch size or resolution",
                "Enable CPU offloading for non-critical components",
                "Consider using quantization to reduce memory usage"
            ],
            "quantization": [
                "Try bf16 quantization for best quality/performance balance",
                "Increase timeout if quantization is taking too long",
                "Enable fallback to non-quantized mode",
                "Check that your GPU supports the selected quantization method"
            ],
            "hardware_optimization": [
                "Run hardware detection to verify your configuration",
                "Enable optimizations specific to your GPU model",
                "Check that drivers are up to date",
                "Ensure adequate cooling for sustained performance"
            ]
        }
        
        return tips_map.get(error_category, ["Check the documentation for more information"])
    
    def _generate_detailed_explanation(self, 
                                     error_category: str, 
                                     error_context: Dict[str, Any]) -> str:
        """Generate detailed explanation for error context"""
        explanations = {
            "vram_memory": """
This error indicates that your GPU has run out of video memory (VRAM). This commonly happens when:
- Loading large models like TI2V-5B
- Using high resolutions or batch sizes
- Multiple applications are using GPU memory

Solutions:
1. Enable VRAM optimization to automatically manage memory
2. Use quantization to reduce model memory usage
3. Enable CPU offloading for text encoder and VAE
4. Reduce generation parameters (resolution, batch size)
            """.strip(),
            "quantization": """
Quantization reduces the precision of model weights to save memory. This error may occur when:
- Quantization process times out
- Selected quantization method is incompatible
- Insufficient resources for quantization process

Solutions:
1. Increase quantization timeout
2. Try a different quantization method (bf16 → int8)
3. Enable fallback to non-quantized mode
4. Ensure sufficient system resources
            """.strip()
        }
        
        return explanations.get(error_category, "Please refer to the documentation for more information about this issue.")
    
    def _log_help_interaction(self, 
                            element_id: str, 
                            trigger: HelpTrigger, 
                            user_context: Optional[Dict[str, Any]]):
        """Log help interaction for analytics"""
        interaction = {
            "element_id": element_id,
            "trigger": trigger.value,
            "timestamp": self._get_timestamp(),
            "user_context": user_context or {}
        }
        
        self.logger.info(f"Help interaction: {json.dumps(interaction)}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


# Example usage functions
def create_help_system_integration():
    """Create integration helpers for the help system"""
    help_system = ContextualHelpSystem()
    
    def get_help_for_element(element_id: str, trigger: str = "hover", user_context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Get help content for a UI element"""
        trigger_enum = HelpTrigger(trigger) if trigger in [t.value for t in HelpTrigger] else HelpTrigger.HOVER
        help_content = help_system.get_help_content(element_id, trigger_enum, user_context)
        
        if help_content:
            return {
                "title": help_content.title,
                "description": help_content.description,
                "content_type": help_content.content_type.value,
                "quick_tips": help_content.quick_tips,
                "detailed_explanation": help_content.detailed_explanation,
                "documentation_links": [
                    {
                        "title": link.title,
                        "url": link.url,
                        "description": link.description
                    }
                    for link in help_content.documentation_links
                ],
                "related_settings": help_content.related_settings,
                "warnings": help_content.warnings,
                "examples": help_content.examples
            }
        return None
    
    def start_guided_tour(tour_id: str, user_profile: str = "default") -> Optional[List[Dict[str, Any]]]:
        """Start a guided tour"""
        tour_steps = help_system.get_guided_tour(tour_id, user_profile)
        
        if tour_steps:
            return [
                {
                    "step_id": step.step_id,
                    "target_element": step.target_element,
                    "title": step.title,
                    "content": step.content,
                    "position": step.position,
                    "action_required": step.action_required
                }
                for step in tour_steps
            ]
        return None
    
    def get_contextual_recommendations(user_context: Dict[str, Any], session_id: str = None) -> List[Dict[str, Any]]:
        """Get contextual help recommendations"""
        recommendations = help_system.get_recommended_help_content(user_context, session_id)
        
        return [
            {
                "content_id": content.content_id,
                "title": content.title,
                "description": content.description,
                "quick_tips": content.quick_tips[:3],  # Top 3 tips
                "priority": content.priority
            }
            for content in recommendations
        ]
    
    return {
        "get_help": get_help_for_element,
        "start_tour": start_guided_tour,
        "get_recommendations": get_contextual_recommendations,
        "help_system": help_system
    }