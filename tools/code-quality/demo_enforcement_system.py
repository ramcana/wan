"""
Demonstration of the automated quality enforcement system.
"""

import sys
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from enforcement.enforcement_cli import EnforcementCLI


def create_demo_project() -> Path:
    """Create a demo project for testing enforcement."""
    demo_dir = Path(tempfile.mkdtemp(prefix="quality_enforcement_demo_"))
    
    # Create project structure
    (demo_dir / "src").mkdir()
    (demo_dir / "tests").mkdir()
    (demo_dir / ".git" / "hooks").mkdir(parents=True)
    
    # Create sample Python files with various quality issues
    
    # Good quality file
    good_file = demo_dir / "src" / "good_module.py"
    good_file.write_text('''"""
A well-written module demonstrating good code quality.
"""

from typing import List, Optional


class DataProcessor:
    """Processes data with proper error handling and documentation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the data processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.processed_count = 0
    
    def process_items(self, items: List[str]) -> List[str]:
        """Process a list of items.
        
        Args:
            items: List of items to process
            
        Returns:
            List of processed items
            
        Raises:
            ValueError: If items list is empty
        """
        if not items:
            raise ValueError("Items list cannot be empty")
        
        processed = []
        for item in items:
            processed_item = self._process_single_item(item)
            processed.append(processed_item)
        
        self.processed_count += len(processed)
        return processed
    
    def _process_single_item(self, item: str) -> str:
        """Process a single item.
        
        Args:
            item: Item to process
            
        Returns:
            Processed item
        """
        return item.strip().upper()
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            "processed_count": self.processed_count,
            "config_items": len(self.config)
        }
''')
    
    # File with quality issues
    bad_file = demo_dir / "src" / "bad_module.py"
    bad_file.write_text('''# Bad quality file with multiple issues

import os,sys,json
import requests

def process_data(data):
    # No docstring, no type hints
    result=[]
    for i in data:
        if i:
            x=i.strip()
            if len(x)>0:
                result.append(x.upper())
    return result

class processor:  # Bad naming convention
    def __init__(self,config):
        self.config=config
        self.count=0
    
    def do_stuff(self,items):  # Vague function name
        # Complex nested logic
        if items:
            if len(items)>0:
                for item in items:
                    if item:
                        if isinstance(item,str):
                            if len(item.strip())>0:
                                self.count+=1
                                return item.upper()
        return None

# Unused imports and variables
unused_var = "this is not used"
another_unused = 42

def long_function_with_many_parameters(param1,param2,param3,param4,param5,param6,param7,param8):
    # Too many parameters, no documentation
    return param1+param2+param3+param4+param5+param6+param7+param8
''')
    
    # Test file
    test_file = demo_dir / "tests" / "test_modules.py"
    test_file.write_text('''"""
Test file for the demo modules.
"""

import pytest
from src.good_module import DataProcessor


class TestDataProcessor:
    """Test cases for DataProcessor."""
    
    def test_process_items_success(self):
        """Test successful item processing."""
        processor = DataProcessor()
        items = ["  hello  ", "  world  "]
        result = processor.process_items(items)
        
        assert result == ["HELLO", "WORLD"]
        assert processor.processed_count == 2
    
    def test_process_items_empty_list(self):
        """Test processing empty list raises error."""
        processor = DataProcessor()
        
        with pytest.raises(ValueError, match="Items list cannot be empty"):
            processor.process_items([])
    
    def test_get_stats(self):
        """Test getting processing statistics."""
        config = {"setting1": "value1", "setting2": "value2"}
        processor = DataProcessor(config)
        
        stats = processor.get_stats()
        
        assert stats["processed_count"] == 0
        assert stats["config_items"] == 2
''')
    
    # Requirements file
    requirements_file = demo_dir / "requirements.txt"
    requirements_file.write_text('''pytest>=7.0.0
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.0.0
''')
    
    return demo_dir


def demonstrate_enforcement_system():
    """Demonstrate the complete enforcement system."""
    print("🚀 Quality Enforcement System Demonstration")
    print("=" * 50)
    
    # Create demo project
    print("\n📁 Creating demo project...")
    demo_dir = create_demo_project()
    print(f"Demo project created at: {demo_dir}")
    
    try:
        # Initialize enforcement CLI
        cli = EnforcementCLI(demo_dir)
        
        # Show initial status
        print("\n📊 Initial enforcement status:")
        cli.status()
        
        # Setup pre-commit hooks
        print("\n🔧 Setting up pre-commit hooks...")
        hooks_success = cli.setup_hooks()
        if hooks_success:
            print("✅ Pre-commit hooks setup completed")
        else:
            print("❌ Pre-commit hooks setup failed")
        
        # Setup CI/CD integration
        print("\n🔄 Setting up CI/CD integration...")
        ci_success = cli.setup_ci('github')
        if ci_success:
            print("✅ GitHub Actions workflow created")
        else:
            print("❌ GitHub Actions setup failed")
        
        # Create quality metrics dashboard
        print("\n📈 Creating quality metrics dashboard...")
        dashboard_success = cli.create_dashboard()
        if dashboard_success:
            print("✅ Quality metrics dashboard configured")
        else:
            print("❌ Dashboard creation failed")
        
        # Show updated status
        print("\n📊 Updated enforcement status:")
        cli.status()
        
        # Run quality checks
        print("\n🔍 Running quality checks on demo files...")
        
        # Check the good file
        good_file = demo_dir / "src" / "good_module.py"
        print(f"\n📄 Checking good quality file: {good_file.name}")
        good_result = cli.run_checks([str(good_file)])
        
        # Check the bad file
        bad_file = demo_dir / "src" / "bad_module.py"
        print(f"\n📄 Checking poor quality file: {bad_file.name}")
        bad_result = cli.run_checks([str(bad_file)])
        
        # Generate comprehensive report
        print("\n📋 Generating quality reports...")
        
        # Console report
        print("\n--- Console Report ---")
        cli.generate_report('console')
        
        # HTML report
        print("\n--- HTML Report ---")
        cli.generate_report('html')
        html_report = demo_dir / "quality-report.html"
        if html_report.exists():
            print(f"✅ HTML report generated: {html_report}")
        
        # JSON report
        print("\n--- JSON Report ---")
        cli.generate_report('json')
        json_report = demo_dir / "quality-report.json"
        if json_report.exists():
            print(f"✅ JSON report generated: {json_report}")
        
        # Show created files
        print("\n📁 Files created by enforcement system:")
        enforcement_files = [
            ".github/workflows/code-quality.yml",
            ".pre-commit-config.yaml",
            "quality-config.yaml",
            "quality-report.html",
            "quality-report.json"
        ]
        
        for file_path in enforcement_files:
            full_path = demo_dir / file_path
            if full_path.exists():
                print(f"✅ {file_path}")
                
                # Show snippet of important files
                if file_path.endswith('.yml') or file_path.endswith('.yaml'):
                    print(f"   📄 Content preview:")
                    content = full_path.read_text()
                    lines = content.split('\n')[:10]
                    for line in lines:
                        print(f"   {line}")
                    if len(content.split('\n')) > 10:
                        print("   ...")
                    print()
            else:
                print(f"❌ {file_path}")
        
        # Demonstrate hook functionality
        print("\n🪝 Demonstrating pre-commit hook functionality...")
        
        # Check if manual hooks were installed
        hook_file = demo_dir / ".git" / "hooks" / "pre-commit"
        if hook_file.exists():
            print("✅ Pre-commit hook installed")
            print("📄 Hook content preview:")
            content = hook_file.read_text()
            lines = content.split('\n')[:15]
            for line in lines:
                print(f"   {line}")
            print("   ...")
        
        print("\n🎯 Enforcement System Summary:")
        print("=" * 40)
        print("✅ Pre-commit hooks: Prevent bad code from being committed")
        print("✅ CI/CD integration: Automated quality checks on every push/PR")
        print("✅ Quality metrics: Track code quality trends over time")
        print("✅ Multiple report formats: Console, HTML, and JSON reports")
        print("✅ Configurable thresholds: Customize quality standards")
        print("✅ Multiple CI platforms: GitHub Actions, GitLab CI, Jenkins")
        
        print(f"\n📍 Demo project location: {demo_dir}")
        print("🔍 Explore the generated files to see the enforcement system in action!")
        
        return demo_dir
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        return None


def show_enforcement_features():
    """Show key features of the enforcement system."""
    print("\n🌟 Key Features of the Quality Enforcement System:")
    print("=" * 55)
    
    features = [
        {
            "name": "Pre-commit Hooks",
            "description": "Automatically check code quality before commits",
            "benefits": [
                "Prevents bad code from entering the repository",
                "Provides immediate feedback to developers",
                "Supports both pre-commit framework and manual hooks",
                "Configurable quality standards"
            ]
        },
        {
            "name": "CI/CD Integration",
            "description": "Automated quality checks in continuous integration",
            "benefits": [
                "Supports GitHub Actions, GitLab CI, and Jenkins",
                "Generates detailed quality reports",
                "Fails builds on quality violations",
                "Tracks quality metrics over time"
            ]
        },
        {
            "name": "Quality Metrics Dashboard",
            "description": "Track and monitor code quality trends",
            "benefits": [
                "Historical quality tracking",
                "Configurable quality thresholds",
                "Alert system for quality regressions",
                "Multiple report formats (HTML, JSON, console)"
            ]
        },
        {
            "name": "Automated Enforcement",
            "description": "Enforce quality standards automatically",
            "benefits": [
                "Consistent quality across the team",
                "Reduced manual code review overhead",
                "Early detection of quality issues",
                "Integration with existing development workflows"
            ]
        }
    ]
    
    for feature in features:
        print(f"\n🔧 {feature['name']}")
        print(f"   {feature['description']}")
        print("   Benefits:")
        for benefit in feature['benefits']:
            print(f"   • {benefit}")


if __name__ == "__main__":
    try:
        # Show features
        show_enforcement_features()
        
        # Run demonstration
        demo_dir = demonstrate_enforcement_system()
        
        if demo_dir:
            print(f"\n🎉 Demonstration completed successfully!")
            print(f"📁 Demo files are available at: {demo_dir}")
            print("\n💡 To clean up the demo:")
            print(f"   rm -rf {demo_dir}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()