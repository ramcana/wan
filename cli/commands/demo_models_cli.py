#!/usr/bin/env python3
"""
Demo script showing Model Orchestrator CLI functionality.
This script demonstrates the key features without actually downloading models.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a CLI command and display the output."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        print(f"Exit code: {result.returncode}")
    except Exception as e:
        print(f"Error running command: {e}")

def main():
    """Demonstrate Model Orchestrator CLI commands."""
    print("Model Orchestrator CLI Demo")
    print("This demo shows the CLI interface without downloading actual models.")
    
    base_cmd = [sys.executable, "-m", "cli.main", "models"]
    
    # 1. Show help
    run_command(base_cmd + ["--help"])
    
    # 2. List available models
    run_command(base_cmd + ["list"])
    
    # 3. List models with detailed information
    run_command(base_cmd + ["list", "--detailed"])
    
    # 4. List models in JSON format
    run_command(base_cmd + ["list", "--json"])
    
    # 5. Check status of all models
    run_command(base_cmd + ["status"])
    
    # 6. Check status of specific model
    run_command(base_cmd + ["status", "--model", "t2v-A14B@2.2.0"])
    
    # 7. Check status in JSON format
    run_command(base_cmd + ["status", "--json"])
    
    # 8. Show help for ensure command
    run_command(base_cmd + ["ensure", "--help"])
    
    # 9. Show help for verify command
    run_command(base_cmd + ["verify", "--help"])
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)
    print("\nKey CLI Features Demonstrated:")
    print("✓ Human-readable and JSON output formats")
    print("✓ Comprehensive help system with examples")
    print("✓ Model listing with detailed information")
    print("✓ Status checking for individual and all models")
    print("✓ Proper exit codes for success/failure scenarios")
    print("✓ Rich formatting with tables and progress indicators")
    print("\nTo actually download models, use:")
    print("  python -m cli.main models ensure --only t2v-A14B@2.2.0")
    print("  python -m cli.main models ensure --all")

if __name__ == "__main__":
    main()