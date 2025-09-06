"""
Tool Registry
Auto-generated registry of available tools.
"""

AVAILABLE_TOOLS = ['code-quality', 'code-review', 'codebase-cleanup', 'config-analyzer', 'doc-generator', 'health-checker', 'maintenance-reporter', 'maintenance-scheduler', 'project-structure-analyzer', 'quality-monitor', 'test-auditor', 'training-system', 'unified-cli']

def get_tool_module(tool_name: str):
    """Get the module for a specific tool."""
    if tool_name not in AVAILABLE_TOOLS:
        raise ImportError(f"Tool '{tool_name}' not found")
    
    try:
        module = __import__(f"tools.{tool_name}.cli", fromlist=[""])
        return module
    except ImportError as e:
        raise ImportError(f"Failed to import tool '{tool_name}': {e}")

def list_tools():
    """List all available tools."""
    return AVAILABLE_TOOLS.copy()
