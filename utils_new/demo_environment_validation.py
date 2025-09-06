#!/usr/bin/env python3
"""
Demo script to show full environment validation capabilities
"""

import sys
from pathlib import Path

# Add local_testing_framework to path
sys.path.insert(0, str(Path(__file__).parent))

from local_testing_framework.environment_validator import EnvironmentValidator

def main():
    """Run full environment validation demo"""
    print("Local Testing Framework - Environment Validation Demo")
    print("=" * 55)
    
    # Create validator
    validator = EnvironmentValidator()
    
    # Run full validation
    print("Running comprehensive environment validation...\n")
    results = validator.validate_full_environment()
    
    # Generate and display report
    print("ENVIRONMENT VALIDATION REPORT")
    print("=" * 55)
    report = validator.generate_environment_report(results)
    print(report)
    
    print("\n" + "=" * 55)
    
    # Generate remediation instructions if needed
    if results.overall_status.value != "passed":
        print("\nREMEDIATION INSTRUCTIONS")
        print("=" * 55)
        instructions = validator.generate_remediation_instructions(results)
        print(instructions)
        
        # Show automated fix commands
        print("AUTOMATED FIX COMMANDS")
        print("=" * 55)
        commands = validator.get_automated_fix_commands(results)
        if commands:
            for i, cmd in enumerate(commands, 1):
                print(f"{i}. {cmd}")
        else:
            print("No automated fix commands available.")
    else:
        print("\n🎉 All validations passed! Your environment is ready.")

if __name__ == '__main__':
    main()