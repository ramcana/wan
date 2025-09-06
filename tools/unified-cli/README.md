# Unified CLI Tool

The Unified CLI Tool provides a single entry point for all project cleanup and quality improvement tools, with workflow automation, team collaboration features, and IDE integration.

## Features

### 🔧 Tool Integration

- **Unified Interface**: Single command to access all quality tools
- **Context-Aware Execution**: Automatically selects appropriate tools based on development context
- **Parallel Execution**: Run multiple tools simultaneously for faster feedback
- **Smart Workflows**: Predefined workflows for common development scenarios

### 🤝 Team Collaboration

- **Shared Standards**: Define and share team coding standards
- **Quality Gates**: Enforce quality requirements at different stages
- **Team Notifications**: Coordinate quality improvements across team members
- **Standards Synchronization**: Import/export team standards for consistency

### 🔄 Workflow Automation

- **File Watching**: Automatically run tools when files change
- **Context Detection**: Smart detection of development context (pre-commit, post-commit, etc.)
- **Debouncing**: Prevent excessive tool runs with intelligent timing
- **Custom Rules**: Define custom automation rules for your workflow

### 🎯 IDE Integration

- **Real-time Feedback**: Live quality feedback as you code
- **WebSocket Server**: Connect IDEs for instant notifications
- **Language Server Protocol**: Standard LSP integration for popular editors
- **Performance Monitoring**: Track tool performance impact

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Make CLI executable
chmod +x tools/unified-cli/cli.py

# Add to PATH (optional)
export PATH=$PATH:$(pwd)/tools/unified-cli
```

## Quick Start

### Basic Usage

```bash
# List available tools
python tools/unified-cli/cli.py list-tools

# List available workflows
python tools/unified-cli/cli.py list-workflows

# Run a specific tool
python tools/unified-cli/cli.py tool test-audit

# Run a predefined workflow
python tools/unified-cli/cli.py workflow pre-commit

# Auto-detect context and run appropriate workflow
python tools/unified-cli/cli.py auto
```

### Team Collaboration

```bash
# Set up team collaboration
python tools/unified-cli/cli.py team setup "My Team"

# Check quality gates before merge
python tools/unified-cli/cli.py team check-gates pre_merge

# Generate team report
python tools/unified-cli/cli.py team report

# Share team standards
python tools/unified-cli/cli.py team share-standards --output team_standards.json

# Import team standards
python tools/unified-cli/cli.py team import-standards team_standards.json

# Send team notification
python tools/unified-cli/cli.py team notify "Code review completed" --level info
```

### Workflow Automation

```bash
# Start automation daemon
python tools/unified-cli/workflow_automation.py daemon

# This will:
# - Watch for file changes
# - Auto-run appropriate tools
# - Apply debouncing logic
# - Log all activities
```

### IDE Integration

```bash
# Start IDE integration server
python tools/unified-cli/ide_integration.py server 8765

# Connect your IDE to ws://localhost:8765 for real-time feedback
```

## Available Tools

| Tool              | Description                      | Use Case                 |
| ----------------- | -------------------------------- | ------------------------ |
| `test-audit`      | Audit and fix test suite issues  | Test quality improvement |
| `test-coverage`   | Analyze test coverage            | Coverage reporting       |
| `test-runner`     | Execute tests with orchestration | Test execution           |
| `config`          | Manage unified configuration     | Configuration management |
| `config-analyzer` | Analyze configuration landscape  | Config consolidation     |
| `structure`       | Analyze project structure        | Documentation generation |
| `cleanup`         | Clean up duplicate/dead code     | Codebase maintenance     |
| `quality`         | Check code quality standards     | Quality enforcement      |
| `review`          | Automated code review            | Code review assistance   |
| `health`          | Check project health             | Health monitoring        |
| `monitor`         | Monitor quality metrics          | Quality tracking         |
| `maintenance`     | Schedule maintenance tasks       | Automated maintenance    |
| `report`          | Generate maintenance reports     | Reporting                |
| `docs`            | Generate documentation           | Documentation            |
| `dev-env`         | Manage development environment   | Environment setup        |
| `feedback`        | Provide development feedback     | Developer assistance     |
| `onboarding`      | Help with team onboarding        | New developer support    |

## Workflow Contexts

### Pre-defined Workflows

| Context             | Tools                                            | When to Use               |
| ------------------- | ------------------------------------------------ | ------------------------- |
| `pre-commit`        | quality, test-audit, health                      | Before committing changes |
| `post-commit`       | test-coverage, monitor, docs                     | After committing changes  |
| `daily-maintenance` | health, cleanup, maintenance                     | Daily maintenance tasks   |
| `weekly-cleanup`    | cleanup, structure, config, report               | Weekly deep cleanup       |
| `release-prep`      | test-audit, test-coverage, quality, health, docs | Before releases           |
| `onboarding`        | structure, docs, health                          | New team member setup     |
| `debugging`         | health, test-audit, monitor                      | When debugging issues     |

### Custom Workflows

```bash
# Run custom combination of tools
python tools/unified-cli/cli.py custom quality test-audit docs --parallel

# Create custom automation rules in .kiro/automation-rules.json
{
  "rules": [
    {
      "name": "my_custom_rule",
      "trigger_patterns": ["src/**/*.py"],
      "workflow_context": "pre-commit",
      "delay_seconds": 1,
      "debounce_seconds": 5,
      "conditions": ["tests_passing"]
    }
  ]
}
```

## Team Collaboration Features

### Quality Gates

Quality gates ensure code meets team standards before key milestones:

- **pre_commit**: Run before committing code
- **pre_merge**: Run before merging to main branch
- **release**: Run before creating releases

### Shared Standards

Define team-wide standards in `.kiro/team-config.json`:

```json
{
  "team_name": "My Development Team",
  "shared_standards": {
    "code_style": "pep8",
    "max_complexity": 10,
    "min_test_coverage": 80,
    "documentation_required": true
  },
  "quality_gates": {
    "pre_commit": ["quality", "test-audit"],
    "pre_merge": ["test-coverage", "health", "review"],
    "release": ["test-audit", "test-coverage", "quality", "health", "docs"]
  },
  "review_requirements": {
    "min_reviewers": 1,
    "require_tests": true,
    "require_docs": true
  }
}
```

### Notifications

Configure team notifications:

```json
{
  "notification_channels": ["console", "file", "slack", "email"]
}
```

## Configuration

### Project Configuration

Create `.kiro/unified-cli-config.json`:

```json
{
  "default_workflow": "auto",
  "parallel_execution": true,
  "timeout_seconds": 300,
  "notification_level": "info",
  "auto_fix": false,
  "quality_thresholds": {
    "test_coverage": 80,
    "code_quality": 8,
    "documentation": 70
  }
}
```

### Automation Rules

Create `.kiro/automation-rules.json`:

```json
{
  "rules": [
    {
      "name": "python_quality_check",
      "trigger_patterns": ["*.py"],
      "workflow_context": "pre-commit",
      "delay_seconds": 2,
      "debounce_seconds": 10,
      "conditions": ["git_staged_changes"]
    }
  ]
}
```

### IDE Integration

Create `.kiro/ide-config.json`:

```json
{
  "thresholds": {
    "complexity": 10,
    "line_length": 88,
    "function_length": 50,
    "test_coverage": 80
  },
  "real_time_feedback": true,
  "auto_fix_suggestions": true
}
```

## Advanced Usage

### Custom Tool Integration

Add your own tools to the unified CLI:

```python
# In tools/unified-cli/cli.py
from my_custom_tool.cli import MyCustomToolCLI

class UnifiedCLI:
    def __init__(self):
        self.tools = {
            # ... existing tools
            'my-tool': MyCustomToolCLI()
        }
```

### Workflow Hooks

Create hooks that run at specific workflow stages:

```python
# In .kiro/hooks/pre_workflow.py
def pre_workflow_hook(context, tools):
    """Run before any workflow"""
    print(f"Starting {context} workflow with tools: {tools}")

# In .kiro/hooks/post_workflow.py
def post_workflow_hook(context, results):
    """Run after workflow completion"""
    success_count = sum(1 for r in results if r.success)
    print(f"Workflow completed: {success_count}/{len(results)} tools succeeded")
```

### Performance Monitoring

Monitor tool performance:

```bash
# Enable performance monitoring
export UNIFIED_CLI_MONITOR=true

# View performance reports
python tools/unified-cli/cli.py tool monitor --performance-report
```

## Troubleshooting

### Common Issues

1. **Tool not found**: Ensure all tool dependencies are installed
2. **Permission denied**: Check file permissions and PATH configuration
3. **Timeout errors**: Increase timeout in configuration
4. **Git integration issues**: Ensure you're in a git repository

### Debug Mode

```bash
# Enable debug logging
export UNIFIED_CLI_DEBUG=true

# Run with verbose output
python tools/unified-cli/cli.py --verbose workflow pre-commit
```

### Log Files

- Automation logs: `.kiro/automation-logs/`
- Team notifications: `.kiro/team-notifications.log`
- Performance logs: `.kiro/performance-logs/`

## Contributing

1. Add new tools to the `tools/` directory
2. Update the tool registry in `cli.py`
3. Add appropriate workflow configurations
4. Update documentation
5. Add tests for new functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.
