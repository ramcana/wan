#!/usr/bin/env python3
"""
Tool Import Fixer
Fixes import issues in the tools directory by adding missing __init__.py files
and updating import statements.
"""

import os
import sys
from pathlib import Path

def add_init_files():
    """Add __init__.py files to all tool directories."""
    tools_dir = Path("tools")
    
    for tool_dir in tools_dir.iterdir():
        if tool_dir.is_dir() and not tool_dir.name.startswith('.'):
            init_file = tool_dir / "__init__.py"
            if not init_file.exists():
                with open(init_file, 'w') as f:
                    f.write(f'"""Tool: {tool_dir.name}"""\n')
                print(f"Created: {init_file}")

def fix_relative_imports():
    """Fix relative import issues in tool files."""
    tools_dir = Path("tools")
    
    for tool_dir in tools_dir.iterdir():
        if tool_dir.is_dir() and not tool_dir.name.startswith('.'):
            for py_file in tool_dir.glob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                    
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                    
                    # Fix relative imports
                    original_content = content
                    content = content.replace("from .", f"from tools.{tool_dir.name}.")
                    content = content.replace("from ..", f"from tools.")
                    
                    if content != original_content:
                        with open(py_file, 'w') as f:
                            f.write(content)
                        print(f"Fixed imports in: {py_file}")
                        
                except Exception as e:
                    print(f"Error processing {py_file}: {e}")

def create_tool_registry():
    """Create a registry of all available tools."""
    tools_dir = Path("tools")
    registry_file = tools_dir / "registry.py"
    
    tools = []
    for tool_dir in tools_dir.iterdir():
        if tool_dir.is_dir() and not tool_dir.name.startswith('.'):
            cli_file = tool_dir / "cli.py"
            if cli_file.exists():
                tools.append(tool_dir.name)
    
    registry_content = f'''"""
Tool Registry
Auto-generated registry of available tools.
"""

AVAILABLE_TOOLS = {tools}

def get_tool_module(tool_name: str):
    """Get the module for a specific tool."""
    if tool_name not in AVAILABLE_TOOLS:
        raise ImportError(f"Tool '{{tool_name}}' not found")
    
    try:
        module = __import__(f"tools.{{tool_name}}.cli", fromlist=[""])
        return module
    except ImportError as e:
        raise ImportError(f"Failed to import tool '{{tool_name}}': {{e}}")

def list_tools():
    """List all available tools."""
    return AVAILABLE_TOOLS.copy()
'''
    
    with open(registry_file, 'w') as f:
        f.write(registry_content)
    
    print(f"Created tool registry: {registry_file}")

def main():
    """Run all import fixes."""
    print("Fixing tool imports...")
    
    # Add __init__.py files
    add_init_files()
    
    # Fix relative imports
    fix_relative_imports()
    
    # Create tool registry
    create_tool_registry()
    
    print("Tool import fixes completed!")

if __name__ == "__main__":
    main()