#!/usr/bin/env python3
"""
Example usage of the Unified CLI Tool

This script demonstrates various ways to use the unified CLI tool
for project cleanup and quality improvements.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

# Add the tools directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from unified_cli.cli import UnifiedCLI, WorkflowContext
from unified_cli.workflow_automation import WorkflowAutomation
from unified_cli.ide_integration import IDEIntegration


async def demonstrate_basic_usage():
    """Demonstrate basic CLI usage"""
    print("=" * 60)
    print("BASIC USAGE DEMONSTRATION")
    print("=" * 60)
    
    cli = UnifiedCLI()
    
    # List available tools
    print("\n1. Available Tools:")
    cli.list_tools()
    
    # List available workflows
    print("\n2. Available Workflows:")
    cli.list_workflows()
    
    # Run a single tool
    print("\n3. Running single tool (health check):")
    result = await cli.run_tool('health')
    cli.print_results([result])
    
    # Run a workflow
    print("\n4. Running pre-commit workflow:")
    results = await cli.run_workflow(WorkflowContext.PRE_COMMIT)
    cli.print_results(results)


async def demonstrate_team_collaboration():
    """Demonstrate team collaboration features"""
    print("\n" + "=" * 60)
    print("TEAM COLLABORATION DEMONSTRATION")
    print("=" * 60)
    
    cli = UnifiedCLI()
    
    # Set up team collaboration
    print("\n1. Setting up team collaboration:")
    team_standards = {
        'code_style': 'pep8',
        'max_complexity': 8,
        'min_test_coverage': 85,
        'documentation_required': True,
        'type_hints_required': True
    }
    
    cli.setup_team_collaboration("Demo Team", team_standards)
    
    # Check quality gates
    print("\n2. Checking quality gates:")
    success = cli.check_quality_gates('pre_commit')
    print(f"Quality gates passed: {success}")
    
    # Generate team report
    print("\n3. Generating team report:")
    report = cli.generate_team_report()
    print("Team Report Summary:")
    print(f"  Team: {report['team_name']}")
    print(f"  Timestamp: {report['timestamp']}")
    print(f"  Recommendations: {len(report['recommendations'])}")
    
    for i, rec in enumerate(report['recommendations'][:3], 1):
        print(f"    {i}. {rec}")
    
    # Share standards
    print("\n4. Sharing team standards:")
    cli.share_standards("demo_team_standards.json")
    
    # Send team notification
    print("\n5. Sending team notification:")
    cli.notify_team("Demo completed successfully!", "info")


async def demonstrate_workflow_automation():
    """Demonstrate workflow automation"""
    print("\n" + "=" * 60)
    print("WORKFLOW AUTOMATION DEMONSTRATION")
    print("=" * 60)
    
    automation = WorkflowAutomation()
    
    # Show current rules
    print("\n1. Current automation rules:")
    for rule in automation.rules:
        print(f"  {rule.name}:")
        print(f"    Patterns: {rule.trigger_patterns}")
        print(f"    Context: {rule.workflow_context.value}")
        print(f"    Debounce: {rule.debounce_seconds}s")
    
    # Simulate file change
    print("\n2. Simulating file change:")
    test_file = "example_test.py"
    await automation.handle_file_change(test_file, 'modified')
    
    # Show automation logs
    print("\n3. Automation logging enabled")
    print("   Logs will be saved to .kiro/automation-logs/")


async def demonstrate_ide_integration():
    """Demonstrate IDE integration features"""
    print("\n" + "=" * 60)
    print("IDE INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    integration = IDEIntegration()
    
    # Analyze a file
    print("\n1. Real-time file analysis:")
    
    # Create a sample Python file for analysis
    sample_file = Path("sample_analysis.py")
    sample_content = '''
def very_long_function_name_that_exceeds_line_length_limits_and_should_trigger_warnings():
    """This function has some quality issues for demonstration."""
    x = 1
    y = 2
    z = x + y
    if z > 0:
        if z > 1:
            if z > 2:
                print("Too much nesting")
    return z

class UndocumentedClass:
    def method_without_docs(self):
        pass
'''
    
    with open(sample_file, 'w') as f:
        f.write(sample_content)
    
    try:
        feedback = await integration.analyze_file_realtime(str(sample_file))
        
        print(f"  File: {feedback.file_path}")
        print(f"  Issues found: {len(feedback.issues)}")
        print(f"  Metrics: {feedback.metrics}")
        
        print("\n  Quality Issues:")
        for issue in feedback.issues[:5]:  # Show first 5 issues
            print(f"    Line {issue.line}: {issue.message} ({issue.severity})")
            if issue.fix_suggestion:
                print(f"      Fix: {issue.fix_suggestion}")
        
        # Format for IDE
        print("\n2. IDE-formatted feedback (LSP):")
        lsp_format = integration.format_feedback_for_ide(feedback, 'lsp')
        print(f"  Diagnostics: {len(lsp_format['diagnostics'])}")
        
    finally:
        # Clean up sample file
        if sample_file.exists():
            sample_file.unlink()


def demonstrate_cli_commands():
    """Demonstrate CLI commands using subprocess"""
    print("\n" + "=" * 60)
    print("CLI COMMANDS DEMONSTRATION")
    print("=" * 60)
    
    cli_path = Path(__file__).parent / "cli.py"
    
    commands = [
        ["python", str(cli_path), "list-tools"],
        ["python", str(cli_path), "list-workflows"],
        ["python", str(cli_path), "auto"],
    ]
    
    for i, cmd in enumerate(commands, 1):
        print(f"\n{i}. Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30,
                cwd=Path(__file__).parent.parent.parent
            )
            
            if result.returncode == 0:
                print("✓ Success")
                if result.stdout:
                    # Show first few lines of output
                    lines = result.stdout.split('\n')[:10]
                    for line in lines:
                        if line.strip():
                            print(f"  {line}")
            else:
                print("✗ Failed")
                if result.stderr:
                    print(f"  Error: {result.stderr}")
                    
        except subprocess.TimeoutExpired:
            print("✗ Timeout")
        except Exception as e:
            print(f"✗ Error: {e}")


async def demonstrate_custom_workflows():
    """Demonstrate custom workflow creation"""
    print("\n" + "=" * 60)
    print("CUSTOM WORKFLOWS DEMONSTRATION")
    print("=" * 60)
    
    cli = UnifiedCLI()
    
    # Custom tool combination
    print("\n1. Custom tool combination:")
    custom_tools = ['health', 'quality', 'test-audit']
    results = await cli.run_workflow(WorkflowContext.CUSTOM, custom_tools)
    cli.print_results(results)
    
    # Context-based workflow
    print("\n2. Context detection:")
    context = cli.get_context_from_git_status()
    print(f"Detected context: {context.value}")
    
    if context != WorkflowContext.CUSTOM:
        print("Running context-appropriate workflow...")
        results = await cli.run_workflow(context)
        success_count = sum(1 for r in results if r.success)
        print(f"Results: {success_count}/{len(results)} tools succeeded")


def create_sample_configurations():
    """Create sample configuration files"""
    print("\n" + "=" * 60)
    print("CREATING SAMPLE CONFIGURATIONS")
    print("=" * 60)
    
    kiro_dir = Path(".kiro")
    kiro_dir.mkdir(exist_ok=True)
    
    # Sample team configuration
    team_config = {
        "team_name": "Sample Development Team",
        "shared_standards": {
            "code_style": "pep8",
            "max_complexity": 10,
            "min_test_coverage": 80,
            "documentation_required": True,
            "type_hints_required": True
        },
        "notification_channels": ["console", "file"],
        "quality_gates": {
            "pre_commit": ["quality", "test-audit"],
            "pre_merge": ["test-coverage", "health", "review"],
            "release": ["test-audit", "test-coverage", "quality", "health", "docs"]
        },
        "review_requirements": {
            "min_reviewers": 1,
            "require_tests": True,
            "require_docs": True
        }
    }
    
    team_config_file = kiro_dir / "team-config.json"
    with open(team_config_file, 'w') as f:
        json.dump(team_config, f, indent=2)
    
    print(f"✓ Created: {team_config_file}")
    
    # Sample automation rules
    automation_rules = {
        "rules": [
            {
                "name": "python_quality_check",
                "trigger_patterns": ["*.py"],
                "workflow_context": "pre-commit",
                "delay_seconds": 2,
                "debounce_seconds": 10,
                "conditions": ["git_staged_changes"]
            },
            {
                "name": "test_validation",
                "trigger_patterns": ["test_*.py", "*_test.py"],
                "workflow_context": "post-commit",
                "delay_seconds": 1,
                "debounce_seconds": 5
            },
            {
                "name": "config_validation",
                "trigger_patterns": ["*.json", "*.yaml", "*.yml"],
                "workflow_context": "daily-maintenance",
                "delay_seconds": 5,
                "debounce_seconds": 15
            }
        ]
    }
    
    automation_rules_file = kiro_dir / "automation-rules.json"
    with open(automation_rules_file, 'w') as f:
        json.dump(automation_rules, f, indent=2)
    
    print(f"✓ Created: {automation_rules_file}")
    
    # Sample IDE configuration
    ide_config = {
        "thresholds": {
            "complexity": 10,
            "line_length": 88,
            "function_length": 50,
            "test_coverage": 80
        },
        "real_time_feedback": True,
        "auto_fix_suggestions": True,
        "notification_level": "info"
    }
    
    ide_config_file = kiro_dir / "ide-config.json"
    with open(ide_config_file, 'w') as f:
        json.dump(ide_config, f, indent=2)
    
    print(f"✓ Created: {ide_config_file}")


async def main():
    """Main demonstration function"""
    print("UNIFIED CLI TOOL - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Create sample configurations first
        create_sample_configurations()
        
        # Run demonstrations
        await demonstrate_basic_usage()
        await demonstrate_team_collaboration()
        await demonstrate_workflow_automation()
        await demonstrate_ide_integration()
        await demonstrate_custom_workflows()
        
        # CLI commands (non-async)
        demonstrate_cli_commands()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        print("\nNext Steps:")
        print("1. Review the created configuration files in .kiro/")
        print("2. Try running the CLI commands manually")
        print("3. Set up your team collaboration")
        print("4. Configure automation rules for your workflow")
        print("5. Integrate with your IDE for real-time feedback")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())