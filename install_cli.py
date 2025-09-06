#!/usr/bin/env python3
"""
Installation script for WAN CLI
Makes the CLI available globally and sets up IDE integration
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def main():
    """Install WAN CLI and set up integrations"""
    
    print("🚀 Installing WAN CLI...")
    
    project_root = Path(__file__).parent
    
    # 1. Install CLI in development mode
    print("📦 Installing CLI package...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], 
                      cwd=project_root, check=True)
        print("✅ CLI package installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install CLI package: {e}")
        return False
    
    # 2. Make wan-cli script executable (Unix-like systems)
    wan_cli_script = project_root / "wan-cli"
    if wan_cli_script.exists() and os.name != 'nt':
        try:
            os.chmod(wan_cli_script, 0o755)
            print("✅ Made wan-cli script executable")
        except Exception as e:
            print(f"⚠️ Could not make script executable: {e}")
    
    # 3. Set up VS Code integration (if .vscode exists)
    vscode_dir = project_root / ".vscode"
    if vscode_dir.exists():
        try:
            tasks_source = project_root / "cli" / "ide_integration" / "vscode_tasks.json"
            tasks_dest = vscode_dir / "tasks.json"
            
            if tasks_source.exists():
                shutil.copy2(tasks_source, tasks_dest)
                print("✅ VS Code tasks configured")
            else:
                print("⚠️ VS Code tasks template not found")
        except Exception as e:
            print(f"⚠️ Could not set up VS Code integration: {e}")
    else:
        print("ℹ️ VS Code directory not found - skipping IDE integration")
    
    # 4. Install pre-commit hooks
    print("🔗 Setting up pre-commit hooks...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "pre-commit"], 
                      cwd=project_root, check=True)
        subprocess.run(["pre-commit", "install"], 
                      cwd=project_root, check=True)
        print("✅ Pre-commit hooks installed")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Could not install pre-commit hooks: {e}")
    
    # 5. Test installation
    print("🧪 Testing installation...")
    try:
        result = subprocess.run([sys.executable, "-m", "cli.main", "--help"], 
                              cwd=project_root, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CLI installation test passed")
        else:
            print(f"❌ CLI installation test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Could not test CLI installation: {e}")
        return False
    
    # 6. Show next steps
    print("\n🎉 Installation complete!")
    print("\n📋 Next steps:")
    print("1. Run 'wan-cli init' to set up your project")
    print("2. Run 'wan-cli status' to check project health")
    print("3. Run 'wan-cli quick' for fast validation")
    print("4. Check 'wan-cli --help' for all available commands")
    
    if vscode_dir.exists():
        print("5. In VS Code: Ctrl+Shift+P → 'Tasks: Run Task' → 'WAN: Quick Validation'")
    
    print("\n📚 Documentation:")
    print("- Getting Started: docs/GETTING_STARTED.md")
    print("- CLI Reference: docs/CLI_REFERENCE.md")
    print("- CLI README: cli/README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)