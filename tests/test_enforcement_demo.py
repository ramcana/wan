"""
Simple test of the enforcement system.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tools.code_quality.enforcement.enforcement_cli import EnforcementCLI


def main():
    """Test the enforcement system."""
    print("🚀 Testing Quality Enforcement System")
    print("=" * 40)
    
    try:
        # Initialize enforcement CLI
        cli = EnforcementCLI()
        
        # Show status
        print("\n📊 Enforcement System Status:")
        cli.status()
        
        # Setup hooks
        print("\n🔧 Setting up pre-commit hooks...")
        hooks_result = cli.setup_hooks()
        print(f"Hooks setup: {'✅ Success' if hooks_result else '❌ Failed'}")
        
        # Setup CI
        print("\n🔄 Setting up GitHub Actions...")
        ci_result = cli.setup_ci('github')
        print(f"CI setup: {'✅ Success' if ci_result else '❌ Failed'}")
        
        # Create dashboard
        print("\n📈 Creating quality dashboard...")
        dashboard_result = cli.create_dashboard()
        print(f"Dashboard: {'✅ Success' if dashboard_result else '❌ Failed'}")
        
        # Show final status
        print("\n📊 Final Status:")
        cli.status()
        
        print("\n🎉 Enforcement system test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
traceback.print_exc()


if __name__ == "__main__":
    main()
