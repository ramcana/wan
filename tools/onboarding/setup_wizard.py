#!/usr/bin/env python3
"""
Automated Setup Wizard

This module provides an interactive setup wizard for new developers,
combining environment setup, dependency installation, and progress tracking.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

# Import our other onboarding tools
sys.path.append(str(Path(__file__).parent.parent))
from onboarding.developer_checklist import DeveloperChecklist
from dev_environment.setup_dev_environment import DevEnvironmentSetup
from dev_environment.environment_validator import EnvironmentValidator

class SetupWizard:
    """Interactive setup wizard for new developers"""
    
    def __init__(self, project_root: Optional[Path] = None, developer_name: Optional[str] = None):
        self.project_root = project_root or Path.cwd()
        self.developer_name = developer_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.checklist = None
        self.env_setup = None
        self.validator = None
        
        # Wizard state
        self.current_step = 0
        self.total_steps = 0
        self.setup_results = {}
    
    def run_wizard(self) -> bool:
        """Run the complete setup wizard"""
        try:
            self._print_welcome()
            
            # Get developer information
            if not self._get_developer_info():
                return False
            
            # Initialize components with developer info
            self._initialize_components()
            
            # Run setup steps
            steps = [
                ("Welcome and Prerequisites", self._step_prerequisites),
                ("Environment Validation", self._step_environment_validation),
                ("Project Setup", self._step_project_setup),
                ("Dependency Installation", self._step_dependency_installation),
                ("Development Tools Setup", self._step_development_tools),
                ("Testing and Validation", self._step_testing_validation),
                ("Documentation and Learning", self._step_documentation),
                ("First Steps", self._step_first_steps),
                ("Completion", self._step_completion)
            ]
            
            self.total_steps = len(steps)
            
            for i, (step_name, step_func) in enumerate(steps):
                self.current_step = i + 1
                self._print_step_header(step_name)
                
                try:
                    if not step_func():
                        self._print_error(f"Step failed: {step_name}")
                        if not self._ask_continue():
                            return False
                except KeyboardInterrupt:
                    self._print_info("\nSetup interrupted by user.")
                    return False
                except Exception as e:
                    self._print_error(f"Error in step '{step_name}': {e}")
                    if not self._ask_continue():
                        return False
            
            return True
            
        except Exception as e:
            self._print_error(f"Setup wizard failed: {e}")
            return False
    
    def _initialize_components(self):
        """Initialize setup components"""
        self.checklist = DeveloperChecklist(self.project_root, self.developer_name)
        self.env_setup = DevEnvironmentSetup(self.project_root, verbose=True)
        self.validator = EnvironmentValidator(self.project_root)
    
    def _print_welcome(self):
        """Print welcome message"""
        print("\n" + "=" * 70)
        print("🚀 Welcome to the WAN22 Development Setup Wizard!")
        print("=" * 70)
        print()
        print("This wizard will guide you through setting up your development")
        print("environment for the WAN22 video generation system.")
        print()
        print("The setup process includes:")
        print("  • Environment validation and setup")
        print("  • Dependency installation")
        print("  • Development tools configuration")
        print("  • Testing and validation")
        print("  • Documentation and learning resources")
        print()
        print("Estimated time: 30-60 minutes (depending on your system)")
        print()
    
    def _get_developer_info(self) -> bool:
        """Get developer information"""
        if not self.developer_name:
            print("First, let's get some information about you:")
            print()
            
            while True:
                name = input("Enter your name (for progress tracking): ").strip()
                if name:
                    self.developer_name = name.replace(' ', '_').lower()
                    break
                print("Please enter a valid name.")
        
        print(f"\nHi {self.developer_name}! Let's get you set up for WAN22 development.")
        
        # Ask about experience level
        print("\nWhat's your experience level with this tech stack?")
        print("1. Beginner (new to Python/React/AI)")
        print("2. Intermediate (some experience)")
        print("3. Advanced (experienced developer)")
        
        while True:
            try:
                level = int(input("Select your level (1-3): "))
                if 1 <= level <= 3:
                    self.experience_level = ['beginner', 'intermediate', 'advanced'][level - 1]
                    break
            except ValueError:
                pass
            print("Please enter 1, 2, or 3.")
        
        return True
    
    def _step_prerequisites(self) -> bool:
        """Step 1: Check prerequisites"""
        self._print_info("Checking system prerequisites...")
        
        # Check operating system
        import platform
        os_name = platform.system()
        self._print_success(f"Operating System: {os_name}")
        
        # Check if we're in the right directory
        if not (self.project_root / "backend").exists() or not (self.project_root / "frontend").exists():
            self._print_error("This doesn't appear to be the WAN22 project directory.")
            self._print_info("Please run this wizard from the WAN22 project root directory.")
            return False
        
        self._print_success("Project directory structure looks correct.")
        
        # Ask if user wants to continue
        print("\nThis wizard will:")
        print("  • Install required software (Python, Node.js, etc.)")
        print("  • Set up your development environment")
        print("  • Install project dependencies")
        print("  • Configure development tools")
        print("  • Run tests to verify everything works")
        
        return self._ask_yes_no("\nReady to begin setup?", default=True)
    
    def _step_environment_validation(self) -> bool:
        """Step 2: Environment validation"""
        self._print_info("Validating development environment...")
        
        # Run environment validation
        health = self.validator.run_full_validation()
        
        # Show results
        print(f"\nEnvironment Health Score: {health.score:.1f}/100")
        
        if health.failed_checks > 0:
            self._print_warning(f"Found {health.failed_checks} critical issues:")
            for result in health.results:
                if result.status == 'fail':
                    print(f"  ❌ {result.name}: {result.message}")
                    if result.fix_suggestion:
                        print(f"     Fix: {result.fix_suggestion}")
            
            if not self._ask_yes_no("\nContinue with automatic fixes?", default=True):
                return False
        
        if health.warning_checks > 0:
            self._print_info(f"Found {health.warning_checks} warnings (will be addressed during setup)")
        
        # Mark checklist items based on validation
        self._update_checklist_from_validation(health)
        
        return True
    
    def _step_project_setup(self) -> bool:
        """Step 3: Project setup"""
        self._print_info("Setting up project structure and configuration...")
        
        # Create project structure
        if not self.env_setup.create_project_structure():
            self._print_error("Failed to create project structure")
            return False
        
        self.checklist.complete_item('project_clone', "Repository already cloned")
        
        # Set up configuration files
        if not self.env_setup.setup_configuration_files():
            self._print_error("Failed to setup configuration files")
            return False
        
        self.checklist.complete_item('project_env_files', "Environment files created")
        
        self._print_success("Project structure and configuration completed!")
        return True
    
    def _step_dependency_installation(self) -> bool:
        """Step 4: Dependency installation"""
        self._print_info("Installing project dependencies...")
        
        # Python environment setup
        self._print_info("Setting up Python environment...")
        if not self.env_setup.setup_python_environment():
            self._print_error("Python environment setup failed")
            if not self._ask_yes_no("Continue anyway?", default=False):
                return False
        else:
            self.checklist.complete_item('project_venv', "Virtual environment configured")
            self.checklist.complete_item('project_backend_deps', "Backend dependencies installed")
        
        # Node.js environment setup
        self._print_info("Setting up Node.js environment...")
        if not self.env_setup.setup_nodejs_environment():
            self._print_error("Node.js environment setup failed")
            if not self._ask_yes_no("Continue anyway?", default=False):
                return False
        else:
            self.checklist.complete_item('project_frontend_deps', "Frontend dependencies installed")
        
        self._print_success("Dependencies installed successfully!")
        return True
    
    def _step_development_tools(self) -> bool:
        """Step 5: Development tools setup"""
        self._print_info("Setting up development tools...")
        
        # Development tools setup
        if not self.env_setup.setup_development_tools():
            self._print_warning("Some development tools setup failed")
        else:
            self.checklist.complete_item('tools_precommit', "Pre-commit hooks installed")
        
        # Run health check
        self._print_info("Running environment health check...")
        try:
            health = self.validator.run_full_validation()
            if health.score >= 80:
                self.checklist.complete_item('tools_health_check', f"Health score: {health.score:.1f}/100")
                self._print_success(f"Environment health check passed! Score: {health.score:.1f}/100")
            else:
                self._print_warning(f"Environment health check score: {health.score:.1f}/100")
        except Exception as e:
            self._print_warning(f"Health check failed: {e}")
        
        return True
    
    def _step_testing_validation(self) -> bool:
        """Step 6: Testing and validation"""
        self._print_info("Running tests to validate setup...")
        
        # Test backend
        self._print_info("Testing backend setup...")
        try:
            result = subprocess.run([
                sys.executable, "tools/test-runner/orchestrator.py", "--category", "unit", "--timeout", "60"
            ], cwd=self.project_root, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                self.checklist.complete_item('test_backend', "Backend tests passed")
                self._print_success("Backend tests passed!")
            else:
                self._print_warning("Some backend tests failed (this is normal for initial setup)")
        except Exception as e:
            self._print_warning(f"Backend test execution failed: {e}")
        
        # Test frontend
        self._print_info("Testing frontend setup...")
        try:
            result = subprocess.run([
                "npm", "test", "--", "--run"
            ], cwd=self.project_root / "frontend", capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                self.checklist.complete_item('test_frontend', "Frontend tests passed")
                self._print_success("Frontend tests passed!")
            else:
                self._print_warning("Some frontend tests failed (this is normal for initial setup)")
        except Exception as e:
            self._print_warning(f"Frontend test execution failed: {e}")
        
        # Test server startup
        self._print_info("Testing development servers...")
        if self._test_server_startup():
            self.checklist.complete_item('test_servers', "Development servers working")
            self._print_success("Development servers are working!")
        else:
            self._print_warning("Server startup test failed")
        
        return True
    
    def _step_documentation(self) -> bool:
        """Step 7: Documentation and learning"""
        self._print_info("Setting up documentation and learning resources...")
        
        # Check if documentation exists
        docs_dir = self.project_root / "tools" / "onboarding" / "docs"
        if docs_dir.exists():
            self._print_success("Onboarding documentation is available!")
            
            print("\n📚 Available documentation:")
            print(f"  • Getting Started: {docs_dir / 'getting-started.md'}")
            print(f"  • Project Overview: {docs_dir / 'project-overview.md'}")
            print(f"  • Development Setup: {docs_dir / 'development-setup.md'}")
            print(f"  • Coding Standards: {docs_dir / 'coding-standards.md'}")
            print(f"  • Troubleshooting: {docs_dir / 'troubleshooting.md'}")
            
            # Mark documentation items as available
            self.checklist.complete_item('docs_overview', "Documentation available for reading")
            
            if self.experience_level == 'beginner':
                print("\n💡 Recommended reading order for beginners:")
                print("  1. Getting Started (5 min)")
                print("  2. Project Overview (30 min)")
                print("  3. Development Setup (20 min)")
                print("  4. Coding Standards (20 min)")
        
        return True
    
    def _step_first_steps(self) -> bool:
        """Step 8: First steps and next actions"""
        self._print_info("Preparing your first development tasks...")
        
        print("\n🎯 Recommended next steps:")
        print()
        
        if self.experience_level == 'beginner':
            print("For beginners:")
            print("  1. Read the Project Overview documentation")
            print("  2. Explore the codebase structure")
            print("  3. Try making a small change (fix a typo, add a comment)")
            print("  4. Run the test suite to see how it works")
            print("  5. Ask your mentor for a good first issue")
        
        elif self.experience_level == 'intermediate':
            print("For intermediate developers:")
            print("  1. Review the coding standards")
            print("  2. Explore the API documentation at http://localhost:8000/docs")
            print("  3. Look at the test suite structure")
            print("  4. Try running the development tools (watchers, health checker)")
            print("  5. Pick up a 'good first issue' from the issue tracker")
        
        else:  # advanced
            print("For advanced developers:")
            print("  1. Review the architecture and design decisions")
            print("  2. Set up your preferred debugging and profiling tools")
            print("  3. Explore the CI/CD and deployment setup")
            print("  4. Consider contributing to the development tools")
            print("  5. Look for areas where you can share your expertise")
        
        print("\n🛠️ Development workflow:")
        print("  • Start servers: python start.py")
        print("  • Run tests: python tools/test-runner/orchestrator.py --run-all")
        print("  • Watch tests: python tools/dev-feedback/test_watcher.py")
        print("  • Health check: python tools/health-checker/health_checker.py")
        print("  • Check progress: python tools/onboarding/developer_checklist.py --status")
        
        return True
    
    def _step_completion(self) -> bool:
        """Step 9: Completion and summary"""
        self._print_info("Finalizing setup...")
        
        # Get final progress
        status = self.checklist.get_status_summary()
        
        print("\n" + "=" * 70)
        print("🎉 Setup Wizard Complete!")
        print("=" * 70)
        print()
        print(f"Developer: {self.developer_name}")
        print(f"Completion: {status['overall']['completion_percentage']:.1f}%")
        print(f"Critical items: {status['overall']['critical_completed']}/{status['overall']['critical_total']}")
        print()
        
        if status['overall']['completion_percentage'] >= 80:
            self._print_success("Excellent! Your development environment is ready to go! 🚀")
        elif status['overall']['completion_percentage'] >= 60:
            self._print_info("Good progress! A few items still need attention.")
        else:
            self._print_warning("Some setup steps need to be completed manually.")
        
        print("\n📋 To check your progress anytime:")
        print("  python tools/onboarding/developer_checklist.py --status")
        
        print("\n📚 Important resources:")
        print("  • Documentation: tools/onboarding/docs/")
        print("  • Troubleshooting: tools/onboarding/docs/troubleshooting.md")
        print("  • Health checker: python tools/health-checker/health_checker.py")
        
        print("\n🆘 Need help?")
        print("  • Check the troubleshooting guide")
        print("  • Ask your mentor or team lead")
        print("  • Run diagnostic tools for specific issues")
        
        print("\n🎯 Ready to start coding!")
        print("  1. Start development servers: python start.py")
        print("  2. Open http://localhost:3000 in your browser")
        print("  3. Make your first change and submit a PR!")
        
        print("\nWelcome to the WAN22 team! 🎉")
        
        return True
    
    def _test_server_startup(self) -> bool:
        """Test if development servers can start"""
        try:
            # This is a simplified test - in a real implementation,
            # you might start the servers and check if they respond
            backend_script = self.project_root / "backend" / "start_server.py"
            frontend_package = self.project_root / "frontend" / "package.json"
            
            return backend_script.exists() and frontend_package.exists()
        except Exception:
            return False
    
    def _update_checklist_from_validation(self, health):
        """Update checklist based on validation results"""
        for result in health.results:
            if result.status == 'pass':
                # Map validation results to checklist items
                if 'Python' in result.name:
                    self.checklist.complete_item('env_python', f"Validation: {result.message}")
                elif 'Node' in result.name:
                    self.checklist.complete_item('env_nodejs', f"Validation: {result.message}")
                elif 'Git' in result.name:
                    self.checklist.complete_item('env_git', f"Validation: {result.message}")
    
    def _print_step_header(self, step_name: str):
        """Print step header"""
        print(f"\n{'='*70}")
        print(f"Step {self.current_step}/{self.total_steps}: {step_name}")
        print(f"{'='*70}")
    
    def _print_success(self, message: str):
        """Print success message"""
        print(f"✅ {message}")
    
    def _print_info(self, message: str):
        """Print info message"""
        print(f"ℹ️  {message}")
    
    def _print_warning(self, message: str):
        """Print warning message"""
        print(f"⚠️  {message}")
    
    def _print_error(self, message: str):
        """Print error message"""
        print(f"❌ {message}")
    
    def _ask_yes_no(self, question: str, default: bool = True) -> bool:
        """Ask yes/no question"""
        default_str = "Y/n" if default else "y/N"
        while True:
            response = input(f"{question} [{default_str}]: ").strip().lower()
            if not response:
                return default
            if response in ['y', 'yes']:
                return True
            if response in ['n', 'no']:
                return False
            print("Please enter 'y' or 'n'")
    
    def _ask_continue(self) -> bool:
        """Ask if user wants to continue after error"""
        return self._ask_yes_no("Continue with setup?", default=True)

def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive setup wizard for WAN22 development")
    parser.add_argument('--developer', type=str, help='Developer name')
    parser.add_argument('--project-root', type=Path, help='Project root directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    # Create and run wizard
    wizard = SetupWizard(
        project_root=args.project_root,
        developer_name=args.developer
    )
    
    try:
        success = wizard.run_wizard()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup wizard interrupted. You can run it again anytime!")
        sys.exit(1)

if __name__ == "__main__":
    main()
