#!/usr/bin/env python3
"""
Standalone script to run the WAN2.2 installation troubleshooter.

This script provides an interactive troubleshooting experience for users
experiencing installation problems.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from user_guidance import UserGuidanceSystem
    from diagnostic_tool import InstallationDiagnosticTool
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you're running this script from the correct directory.")
    sys.exit(1)


def main():
    """Main function to run the troubleshooter."""
    print("🔧 WAN2.2 Installation Troubleshooter")
    print("=" * 50)
    
    # Determine installation path
    if len(sys.argv) > 1:
        installation_path = sys.argv[1]
    else:
        # Use parent directory of scripts as installation path
        installation_path = str(current_dir.parent)
    
    print(f"Installation path: {installation_path}")
    print()
    
    try:
        # Create user guidance system
        guidance = UserGuidanceSystem(installation_path)
        
        # Run interactive troubleshooter
        guidance.run_interactive_troubleshooter()
        
    except KeyboardInterrupt:
        print("\n\nTroubleshooting session interrupted by user.")
    except Exception as e:
        print(f"\nError running troubleshooter: {e}")
        print("Please check the installation and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())