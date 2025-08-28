#!/usr/bin/env python3
"""
Standalone script to run comprehensive system diagnostics for WAN2.2 installation.

This script analyzes the system and generates a detailed diagnostic report
to help identify potential installation issues.
"""

import sys
import os
from pathlib import Path
import argparse

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from diagnostic_tool import InstallationDiagnosticTool
except ImportError as e:
    print(f"Error importing diagnostic tool: {e}")
    print("Please ensure you're running this script from the correct directory.")
    sys.exit(1)


def main():
    """Main function to run diagnostics."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive system diagnostics for WAN2.2 installation"
    )
    parser.add_argument(
        "--installation-path", "-p",
        default=str(current_dir.parent),
        help="Path to the installation directory"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for the diagnostic report"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick health check only"
    )
    parser.add_argument(
        "--generate-docs",
        action="store_true",
        help="Generate help documentation"
    )
    
    args = parser.parse_args()
    
    print("🔍 WAN2.2 System Diagnostics")
    print("=" * 50)
    print(f"Installation path: {args.installation_path}")
    print()
    
    try:
        # Create diagnostic tool
        diagnostic = InstallationDiagnosticTool(args.installation_path)
        
        if args.quick:
            # Run quick health check
            print("Running quick health check...")
            health_check = diagnostic.get_quick_health_check()
            
            print("\n📊 Quick Health Check Results:")
            print("-" * 30)
            print(f"Python OK: {'✅' if health_check['python_ok'] else '❌'}")
            print(f"Memory OK: {'✅' if health_check['memory_ok'] else '❌'}")
            print(f"Disk OK: {'✅' if health_check['disk_ok'] else '❌'}")
            print(f"Network OK: {'✅' if health_check['network_ok'] else '❌'}")
            print(f"Overall OK: {'✅' if health_check['overall_ok'] else '❌'}")
            
            if not health_check['overall_ok']:
                print("\n⚠️  Issues detected. Run full diagnostics for detailed analysis:")
                print(f"   python {__file__} --installation-path {args.installation_path}")
        
        elif args.generate_docs:
            # Generate help documentation
            print("Generating help documentation...")
            from user_guidance import UserGuidanceSystem
            guidance = UserGuidanceSystem(args.installation_path)
            guidance.generate_help_documentation()
            print("✅ Help documentation generated successfully!")
        
        else:
            # Run full diagnostics
            print("Running comprehensive system diagnostics...")
            print("This may take a few moments...\n")
            
            diagnostics = diagnostic.run_full_diagnostics()
            
            # Generate and display report
            output_file = args.output
            if not output_file:
                # Default output file
                output_file = str(Path(args.installation_path) / "logs" / "diagnostic_report.txt")
            
            report = diagnostic.generate_diagnostic_report(output_file)
            print(report)
            
            print(f"\n📄 Full diagnostic report saved to: {output_file}")
            
            # Show summary
            fail_count = sum(1 for result in diagnostics.diagnostic_results if result.status == "fail")
            warning_count = sum(1 for result in diagnostics.diagnostic_results if result.status == "warning")
            
            if fail_count > 0:
                print(f"\n❌ {fail_count} critical issues found that need attention.")
            elif warning_count > 0:
                print(f"\n⚠️  {warning_count} warnings found. Installation may still work.")
            else:
                print("\n✅ No major issues detected. System looks ready for installation.")
            
            print("\nFor interactive troubleshooting, run:")
            print(f"   python {current_dir / 'run_troubleshooter.py'} {args.installation_path}")
    
    except KeyboardInterrupt:
        print("\n\nDiagnostics interrupted by user.")
    except Exception as e:
        print(f"\nError running diagnostics: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())