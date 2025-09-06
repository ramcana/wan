import pytest
#!/usr/bin/env python3
"""
Codebase Organization Validator
Validates that the codebase organization is working correctly.
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import List, Dict, Any

class OrganizationValidator:
    def __init__(self):
        self.root_dir = Path(".")
        self.issues = []
        self.successes = []
    
    def log_success(self, message: str):
        """Log a successful validation."""
        self.successes.append(f"✅ {message}")
        print(f"✅ {message}")
    
    def log_issue(self, message: str):
        """Log a validation issue."""
        self.issues.append(f"❌ {message}")
        print(f"❌ {message}")
    
    def validate_directory_structure(self):
        """Validate that key directories exist and are organized."""
        print("\n🔍 Validating Directory Structure...")
        
        required_dirs = [
            "reports", "temp", "archive", "config", "tools", 
            "backend", "frontend", "docs", "scripts", "tests"
        ]
        
        for dir_name in required_dirs:
            dir_path = self.root_dir / dir_name
            if dir_path.exists() and dir_path.is_dir():
                self.log_success(f"Directory exists: {dir_name}/")
            else:
                self.log_issue(f"Missing directory: {dir_name}/")
    
    def validate_configuration_system(self):
        """Validate the unified configuration system."""
        print("\n🔍 Validating Configuration System...")
        
        # Check unified config exists
        unified_config = self.root_dir / "config" / "unified-config.yaml"
        if unified_config.exists():
            self.log_success("Unified configuration file exists")
            
            try:
                with open(unified_config, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check key sections
                required_sections = [
                    "system", "api", "models", "hardware", "generation",
                    "logging", "security", "performance"
                ]
                
                for section in required_sections:
                    if section in config:
                        self.log_success(f"Config section present: {section}")
                    else:
                        self.log_issue(f"Missing config section: {section}")
                        
            except Exception as e:
                self.log_issue(f"Failed to load unified config: {e}")
        else:
            self.log_issue("Unified configuration file missing")
        
        # Check for old config files (should be backed up)
        old_configs = ["config.json", "startup_config.json", "backend/config.json"]
        for config_file in old_configs:
            if (self.root_dir / config_file).exists():
                self.log_issue(f"Old config file still present: {config_file}")
    
    def validate_tool_system(self):
        """Validate the tool system organization."""
        print("\n🔍 Validating Tool System...")
        
        tools_dir = self.root_dir / "tools"
        if not tools_dir.exists():
            self.log_issue("Tools directory missing")
            return
        
        # Check tool registry
        registry_file = tools_dir / "registry.py"
        if registry_file.exists():
            self.log_success("Tool registry exists")
            
            try:
                # Import and test registry
                sys.path.insert(0, str(self.root_dir))
                from tools.registry import list_tools, get_tool_module
                
                tools = list_tools()
                self.log_success(f"Tool registry loaded: {len(tools)} tools")
                
                # Test a few tool imports
                test_tools = ["unified-cli", "health-checker", "quality-monitor"]
                for tool in test_tools:
                    if tool in tools:
                        try:
                            get_tool_module(tool)
                            self.log_success(f"Tool imports correctly: {tool}")
                        except Exception as e:
                            self.log_issue(f"Tool import failed: {tool} - {e}")
                
            except Exception as e:
                self.log_issue(f"Tool registry import failed: {e}")
        else:
            self.log_issue("Tool registry missing")
        
        # Check for __init__.py files in tool directories
        for tool_dir in tools_dir.iterdir():
            if tool_dir.is_dir() and not tool_dir.name.startswith('.'):
                init_file = tool_dir / "__init__.py"
                if init_file.exists():
                    self.log_success(f"Tool has __init__.py: {tool_dir.name}")
                else:
                    self.log_issue(f"Tool missing __init__.py: {tool_dir.name}")
    
    def validate_file_organization(self):
        """Validate that files have been properly organized."""
        print("\n🔍 Validating File Organization...")
        
        # Check reports directory
        reports_dir = self.root_dir / "reports"
        if reports_dir.exists():
            report_files = list(reports_dir.glob("*.md")) + list(reports_dir.glob("*.json"))
            if report_files:
                self.log_success(f"Reports organized: {len(report_files)} files")
            else:
                self.log_issue("Reports directory empty")
        
        # Check temp directory
        temp_dir = self.root_dir / "temp"
        if temp_dir.exists():
            temp_files = list(temp_dir.glob("temp*"))
            if temp_files:
                self.log_success(f"Temp files organized: {len(temp_files)} files")
        
        # Check archive directory
        archive_dir = self.root_dir / "archive"
        if archive_dir.exists():
            archive_files = list(archive_dir.iterdir())
            if archive_files:
                self.log_success(f"Archive files organized: {len(archive_files)} files")
        
        # Check root directory cleanliness
        root_files = [f for f in self.root_dir.iterdir() if f.is_file()]
        essential_files = [
            ".dockerignore", ".env", ".gitignore", ".gitkeep_tests", 
            ".pre-commit-config.yaml", "README.md", "main.py", "start.py",
            "pytest.ini", "nginx.conf", "wan22_tasks.db"
        ]
        
        non_essential_count = 0
        for file in root_files:
            if file.name not in essential_files and not file.name.startswith('.'):
                non_essential_count += 1
        
        if non_essential_count < 10:  # Allow some flexibility
            self.log_success(f"Root directory clean: {len(root_files)} total files")
        else:
            self.log_issue(f"Root directory cluttered: {non_essential_count} non-essential files")
    
    def validate_backup_system(self):
        """Validate that backups were created properly."""
        print("\n🔍 Validating Backup System...")
        
        backup_dir = self.root_dir / "config_backups"
        if backup_dir.exists():
            migration_dirs = list(backup_dir.glob("migration_*"))
            if migration_dirs:
                latest_backup = max(migration_dirs, key=lambda x: x.name)
                self.log_success(f"Configuration backups exist: {latest_backup.name}")
                
                # Check backup contents
                backup_files = list(latest_backup.glob("*.json")) + list(latest_backup.glob("*.yaml"))
                if backup_files:
                    self.log_success(f"Backup contains {len(backup_files)} config files")
                else:
                    self.log_issue("Backup directory empty")
            else:
                self.log_issue("No migration backups found")
        else:
            self.log_issue("Backup directory missing")
    
    def generate_report(self):
        """Generate a validation report."""
        print("\n" + "="*60)
        print("📊 VALIDATION REPORT")
        print("="*60)
        
        print(f"\n✅ SUCCESSES ({len(self.successes)}):")
        for success in self.successes:
            print(f"  {success}")
        
        if self.issues:
            print(f"\n❌ ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  {issue}")
        else:
            print(f"\n🎉 NO ISSUES FOUND!")
        
        print(f"\n📈 SUMMARY:")
        print(f"  Total Checks: {len(self.successes) + len(self.issues)}")
        print(f"  Passed: {len(self.successes)}")
        print(f"  Failed: {len(self.issues)}")
        print(f"  Success Rate: {len(self.successes)/(len(self.successes) + len(self.issues))*100:.1f}%")
        
        if len(self.issues) == 0:
            print(f"\n🎯 RESULT: CODEBASE ORGANIZATION SUCCESSFUL!")
            return True
        else:
            print(f"\n⚠️  RESULT: SOME ISSUES NEED ATTENTION")
            return False
    
    def validate_all(self):
        """Run all validation checks."""
        print("🚀 Starting Codebase Organization Validation...")
        
        self.validate_directory_structure()
        self.validate_configuration_system()
        self.validate_tool_system()
        self.validate_file_organization()
        self.validate_backup_system()
        
        return self.generate_report()

if __name__ == "__main__":
    validator = OrganizationValidator()
    success = validator.validate_all()
    sys.exit(0 if success else 1)