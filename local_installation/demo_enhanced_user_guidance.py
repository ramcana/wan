"""
Demo script for the enhanced user guidance system.

This script demonstrates:
- Enhanced error message formatting with recovery strategies
- Progress indicators with estimated completion times
- Recovery strategy explanations and success likelihood display
- Support ticket generation with pre-filled error reports
- Links to documentation and support resources
"""

import sys
import os
import time
import tempfile
from pathlib import Path

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.user_guidance import (
    UserGuidanceSystem, RecoveryStrategy, ProgressIndicator, SupportTicket,
    RecoveryStatus, SupportResourceType, SupportResource
)
from interfaces import InstallationError, ErrorCategory


def demo_enhanced_error_formatting():
    """Demonstrate enhanced error message formatting."""
    print("\n" + "="*80)
    print("DEMO: Enhanced Error Message Formatting")
    print("="*80)
    
    # Create guidance system
    guidance = UserGuidanceSystem(".")
    
    # Create sample error
    error = InstallationError(
        "Failed to download model 'wan22-base': Connection timeout after 30 seconds",
        ErrorCategory.NETWORK,
        recovery_suggestions=[
            "Check your internet connection",
            "Try using a VPN if behind a firewall",
            "Configure proxy settings if needed"
        ]
    )
    
    # Create recovery strategies
    recovery_strategies = [
        RecoveryStrategy(
            name="Network Connectivity Check",
            description="Diagnose and fix network connectivity issues",
            success_likelihood=0.85,
            estimated_time_minutes=5,
            steps=[
                {"description": "Test internet connectivity", "command": "ping google.com"},
                {"description": "Check DNS resolution", "command": "nslookup huggingface.co"},
                {"description": "Test download speed", "command": "curl -o /dev/null -s -w '%{speed_download}' http://speedtest.com/mini.php"}
            ],
            prerequisites=["Command line access"],
            risks=["May reveal network configuration"]
        ),
        RecoveryStrategy(
            name="Alternative Download Sources",
            description="Try downloading from mirror servers or alternative sources",
            success_likelihood=0.75,
            estimated_time_minutes=10,
            steps=[
                {"description": "Switch to mirror server", "command": "export HF_ENDPOINT=https://hf-mirror.com"},
                {"description": "Retry model download", "command": "python download_models.py --retry"},
                {"description": "Verify model integrity", "command": "python verify_models.py"}
            ],
            prerequisites=["Alternative mirror access"],
            risks=["Mirror may have outdated models"]
        ),
        RecoveryStrategy(
            name="Manual Model Download",
            description="Download models manually using browser or git",
            success_likelihood=0.95,
            estimated_time_minutes=20,
            steps=[
                {"description": "Open browser to model page", "command": "start https://huggingface.co/wan22/model"},
                {"description": "Download model files manually", "expected_result": "All model files downloaded"},
                {"description": "Place files in correct directory", "command": "copy models/* ./models/"}
            ],
            prerequisites=["Web browser", "Manual file management"],
            risks=["Time-consuming", "Potential for file placement errors"]
        )
    ]
    
    # Add context information
    context = {
        "timestamp": "2024-01-15 14:30:22",
        "phase": "model_download",
        "component": "huggingface_downloader"
    }
    
    # Format and display the enhanced error message
    formatted_error = guidance.format_user_friendly_error(error, context, recovery_strategies)
    print(formatted_error)
    
    input("\nPress Enter to continue to the next demo...")


def demo_progress_indicators():
    """Demonstrate progress indicators with estimated completion times."""
    print("\n" + "="*80)
    print("DEMO: Progress Indicators with Estimated Completion Times")
    print("="*80)
    
    guidance = UserGuidanceSystem(".")
    
    # Demo 1: Model Download Progress
    print("\n📥 Simulating model download with progress tracking...")
    progress_id = guidance.create_progress_indicator("Model Download", 8, "Initializing download...")
    
    steps = [
        ("Connecting to server", "Establishing connection..."),
        ("Authenticating", "Verifying credentials..."),
        ("Downloading config.json", "2.1 KB/s"),
        ("Downloading tokenizer files", "156 KB/s"),
        ("Downloading model weights (1/3)", "2.3 MB/s"),
        ("Downloading model weights (2/3)", "2.1 MB/s"),
        ("Downloading model weights (3/3)", "1.8 MB/s"),
        ("Verifying integrity", "Checking checksums...")
    ]
    
    for i, (step_name, speed_info) in enumerate(steps, 1):
        guidance.update_progress(progress_id, i, step_name, speed_info)
        time.sleep(1.5)  # Simulate work
    
    guidance.complete_progress(progress_id)
    print("\n✅ Model download completed successfully!")
    
    # Demo 2: Dependency Installation Progress
    print("\n📦 Simulating dependency installation...")
    progress_id = guidance.create_progress_indicator("Dependency Installation", 6, "Preparing...")
    
    deps = [
        ("Installing torch", "15.2 MB/s"),
        ("Installing transformers", "8.7 MB/s"),
        ("Installing numpy", "12.1 MB/s"),
        ("Installing scipy", "6.3 MB/s"),
        ("Installing matplotlib", "4.8 MB/s"),
        ("Finalizing installation", "Cleaning up...")
    ]
    
    for i, (step_name, speed_info) in enumerate(deps, 1):
        guidance.update_progress(progress_id, i, step_name, speed_info)
        time.sleep(1.2)
    
    guidance.complete_progress(progress_id)
    print("\n✅ Dependencies installed successfully!")
    
    input("\nPress Enter to continue to the next demo...")


def demo_recovery_strategy_explanation():
    """Demonstrate recovery strategy explanations."""
    print("\n" + "="*80)
    print("DEMO: Recovery Strategy Explanations with Success Likelihood")
    print("="*80)
    
    guidance = UserGuidanceSystem(".")
    
    # Create detailed recovery strategy
    strategy = RecoveryStrategy(
        name="Model Validation and Recovery",
        description="Comprehensive model validation with automatic recovery for corrupted or missing files",
        success_likelihood=0.92,
        estimated_time_minutes=25,
        steps=[
            {
                "description": "Scan model directory for missing files",
                "command": "python scripts/validate_models.py --scan",
                "expected_result": "List of missing or corrupted files"
            },
            {
                "description": "Calculate checksums for existing files",
                "command": "python scripts/validate_models.py --checksum",
                "expected_result": "Checksum validation report"
            },
            {
                "description": "Download missing model files",
                "command": "python scripts/download_models.py --missing-only",
                "expected_result": "Missing files downloaded successfully"
            },
            {
                "description": "Re-download corrupted files",
                "command": "python scripts/download_models.py --corrupted-only",
                "expected_result": "Corrupted files replaced"
            },
            {
                "description": "Verify model loading functionality",
                "command": "python scripts/test_model_loading.py",
                "expected_result": "All models load successfully"
            },
            {
                "description": "Update model registry",
                "command": "python scripts/update_model_registry.py",
                "expected_result": "Model registry updated"
            }
        ],
        prerequisites=[
            "Internet connection for downloads",
            "At least 10GB free disk space",
            "Python environment properly configured"
        ],
        risks=[
            "Large download size (up to 8GB)",
            "May take significant time on slow connections",
            "Temporary disk space usage during download"
        ]
    )
    
    # Display strategy explanation
    explanation = guidance.explain_recovery_strategy(strategy)
    print(explanation)
    
    # Simulate strategy execution
    print("🚀 Simulating recovery strategy execution...")
    print("   (In real implementation, this would execute the actual recovery steps)")
    
    def mock_step_executor(step):
        """Mock step executor for demonstration."""
        print(f"   Executing: {step['description']}")
        time.sleep(0.8)  # Simulate execution time
        return True  # Simulate success
    
    success = guidance.execute_recovery_strategy(strategy, mock_step_executor)
    
    if success:
        print(f"\n✅ Recovery strategy '{strategy.name}' completed successfully!")
        print(f"   Status: {strategy.status.value}")
        print(f"   Duration: {(strategy.completion_time - strategy.start_time).total_seconds():.1f} seconds")
    else:
        print(f"\n❌ Recovery strategy failed: {strategy.error_message}")
    
    input("\nPress Enter to continue to the next demo...")


def demo_support_ticket_generation():
    """Demonstrate support ticket generation."""
    print("\n" + "="*80)
    print("DEMO: Support Ticket Generation with Pre-filled Error Reports")
    print("="*80)
    
    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        guidance = UserGuidanceSystem(temp_dir)
        
        # Create sample logs
        logs_dir = Path(temp_dir) / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        sample_log = logs_dir / "installation.log"
        sample_log.write_text("""2024-01-15 14:25:10 - INFO - Starting WAN2.2 installation
2024-01-15 14:25:15 - INFO - Checking system requirements
2024-01-15 14:25:20 - INFO - System requirements met
2024-01-15 14:25:25 - INFO - Creating virtual environment
2024-01-15 14:25:30 - INFO - Installing dependencies
2024-01-15 14:28:45 - INFO - Dependencies installed successfully
2024-01-15 14:28:50 - INFO - Starting model download
2024-01-15 14:30:15 - ERROR - Model download failed: Connection timeout
2024-01-15 14:30:20 - ERROR - Retry attempt 1 failed
2024-01-15 14:30:25 - ERROR - Retry attempt 2 failed
2024-01-15 14:30:30 - ERROR - All retry attempts exhausted
2024-01-15 14:30:35 - ERROR - Installation failed""")
        
        # Create complex error scenario
        error = InstallationError(
            "Model download failed after 3 retry attempts: Connection timeout to huggingface.co",
            ErrorCategory.NETWORK,
            recovery_suggestions=[
                "Check internet connection stability",
                "Configure proxy settings if behind corporate firewall",
                "Try downloading during off-peak hours",
                "Use alternative download methods"
            ]
        )
        
        # Add context
        context = {
            "phase": "model_download",
            "component": "huggingface_downloader",
            "timestamp": "2024-01-15 14:30:35",
            "retry_count": 3,
            "total_download_size": "7.2 GB",
            "downloaded_size": "2.1 GB"
        }
        
        # Steps attempted by user
        steps_attempted = [
            "Ran system diagnostics - all checks passed",
            "Verified internet connection - stable",
            "Tried running as administrator - same error",
            "Disabled antivirus temporarily - no change",
            "Attempted manual download via browser - also failed",
            "Checked firewall settings - no blocking rules found",
            "Tried different DNS servers (8.8.8.8, 1.1.1.1) - no improvement"
        ]
        
        # Add some recovery history
        guidance.recovery_history = [
            {
                'strategy_name': 'Network Connectivity Check',
                'success': False,
                'duration_seconds': 45.2,
                'timestamp': '2024-01-15 14:25:00'
            },
            {
                'strategy_name': 'Alternative Download Sources',
                'success': False,
                'duration_seconds': 120.8,
                'timestamp': '2024-01-15 14:27:00'
            }
        ]
        
        print("🎫 Generating comprehensive support ticket...")
        
        # Generate support ticket
        ticket = guidance.generate_support_ticket(error, context, steps_attempted)
        
        print(f"\n📋 Support Ticket Generated:")
        print(f"   Title: {ticket.title}")
        print(f"   Severity: {ticket.severity.upper()}")
        print(f"   Category: {ticket.category.title()}")
        print(f"   Steps Attempted: {len(ticket.steps_attempted)}")
        print(f"   Recovery Strategies Tried: {len(ticket.recovery_strategies_tried)}")
        print(f"   Log Entries: {len(ticket.logs)}")
        print(f"   System Info Fields: {len(ticket.system_info)}")
        
        # Save ticket
        ticket_path = guidance.save_support_ticket(ticket)
        print(f"\n💾 Ticket saved to: {ticket_path}")
        
        # Generate GitHub URL
        github_url = guidance.create_support_ticket_url(ticket, "github")
        print(f"\n🔗 GitHub Issue URL:")
        print(f"   {github_url[:100]}...")
        
        # Display ticket preview
        print(f"\n📄 Ticket Preview (first 500 characters):")
        print("-" * 60)
        markdown = ticket.to_markdown()
        print(markdown[:500] + "..." if len(markdown) > 500 else markdown)
        print("-" * 60)
        
        # Show the actual saved file
        print(f"\n📁 Viewing saved ticket file:")
        with open(ticket_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        for i, line in enumerate(lines[:20]):  # Show first 20 lines
            print(f"   {i+1:2d}: {line}")
        
        if len(lines) > 20:
            print(f"   ... ({len(lines) - 20} more lines)")
    
    input("\nPress Enter to continue to the next demo...")


def demo_support_resources_integration():
    """Demonstrate support resources integration."""
    print("\n" + "="*80)
    print("DEMO: Support Resources Integration")
    print("="*80)
    
    guidance = UserGuidanceSystem(".")
    
    # Test different error categories and their resources
    error_scenarios = [
        {
            "error": InstallationError("Connection refused by server", ErrorCategory.NETWORK),
            "category": "Network Issues"
        },
        {
            "error": InstallationError("Permission denied: cannot create directory", ErrorCategory.PERMISSION),
            "category": "Permission Issues"
        },
        {
            "error": InstallationError("Insufficient memory to load model", ErrorCategory.SYSTEM),
            "category": "System Issues"
        }
    ]
    
    for scenario in error_scenarios:
        print(f"\n🔍 {scenario['category']} - Support Resources:")
        print("-" * 50)
        
        formatted_error = guidance.format_user_friendly_error(scenario["error"])
        
        # Extract support resources section
        lines = formatted_error.split('\n')
        in_resources_section = False
        
        for line in lines:
            if "🆘 Support Resources:" in line:
                in_resources_section = True
                print(line)
            elif in_resources_section and line.strip():
                if line.startswith("🔧 Quick Actions:"):
                    break
                print(line)
            elif in_resources_section and not line.strip():
                print(line)
    
    print("\n📚 All Available Support Resources:")
    print("-" * 40)
    
    for category, resources in guidance.support_resources.items():
        print(f"\n{category.title()} Resources:")
        for resource in resources:
            icon = guidance._get_resource_icon(resource.resource_type)
            print(f"  {icon} {resource.title}")
            print(f"     {resource.description}")
            print(f"     Relevance: {resource.relevance_score:.1f}/1.0")
            print(f"     URL: {resource.url}")
    
    input("\nPress Enter to finish the demo...")


def main():
    """Run all demos."""
    print("🚀 WAN2.2 Enhanced User Guidance System Demo")
    print("=" * 80)
    print("This demo showcases the enhanced features of the user guidance system:")
    print("• Enhanced error message formatting with recovery strategies")
    print("• Progress indicators with estimated completion times")
    print("• Recovery strategy explanations with success likelihood")
    print("• Support ticket generation with pre-filled error reports")
    print("• Integration with documentation and support resources")
    print("=" * 80)
    
    input("Press Enter to start the demo...")
    
    try:
        # Run all demos
        demo_enhanced_error_formatting()
        demo_progress_indicators()
        demo_recovery_strategy_explanation()
        demo_support_ticket_generation()
        demo_support_resources_integration()
        
        print("\n" + "="*80)
        print("🎉 Demo completed successfully!")
        print("="*80)
        print("The enhanced user guidance system provides:")
        print("✅ Rich error messages with context and recovery options")
        print("✅ Real-time progress tracking with time estimates")
        print("✅ Detailed recovery strategy explanations")
        print("✅ Automated support ticket generation")
        print("✅ Integrated support resources and documentation links")
        print("\nThese features significantly improve the user experience")
        print("during installation issues and provide clear paths to resolution.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
