# WAN Unified CLI System

A comprehensive, context-aware command-line interface that unifies all project tools into a single point of entry.

## 🎯 Goals Achieved

✅ **Single Point of Entry**: One CLI (`wan-cli`) for all project operations  
✅ **Context Awareness**: Smart defaults based on project state and environment  
✅ **IDE Integration**: VS Code tasks and keyboard shortcuts  
✅ **Pre-commit Hooks**: Automatic quality checks on every commit  
✅ **Comprehensive Testing**: E2E testing framework for meta-tools  
✅ **Documentation System**: Auto-generated docs and training materials

## 🚀 Quick Start

```bash
# Install the CLI
pip install -e .

# First-time setup
wan-cli init

# Daily workflow
wan-cli quick           # Quick validation (30 seconds)
wan-cli status          # Project health overview
wan-cli test run --fast # Fast test suite
```

## 📁 Architecture

```
cli/
├── main.py                 # Main CLI entry point
├── commands/               # Command modules
│   ├── test.py            # Testing commands
│   ├── clean.py           # Cleanup commands
│   ├── config.py          # Configuration commands
│   ├── docs.py            # Documentation commands
│   ├── quality.py         # Quality commands
│   ├── health.py          # Health monitoring commands
│   └── deploy.py          # Deployment commands
├── workflows/              # Workflow automation
│   └── quick_validation.py # Fast feedback loops
├── testing/                # E2E testing framework
│   └── e2e_test_framework.py
└── ide_integration/        # IDE integration files
    └── vscode_tasks.json   # VS Code tasks
```

## 🛠 Command Groups

### Core Commands

- `wan-cli init` - First-time project setup
- `wan-cli status` - Overall project health
- `wan-cli quick` - Fast validation suite (< 30s)

### Testing (`wan-cli test`)

- `test run` - Run test suite with smart defaults
- `test flaky` - Detect and fix flaky tests
- `test audit` - Comprehensive test analysis
- `test validate` - Test suite integrity check

### Cleanup (`wan-cli clean`)

- `clean duplicates` - Find and remove duplicate files
- `clean dead-code` - Remove unused code
- `clean imports` - Fix and organize imports
- `clean naming` - Standardize naming conventions
- `clean all` - Run all cleanup operations

### Quality (`wan-cli quality`)

- `quality check` - Code quality analysis
- `quality format` - Code formatting
- `quality lint` - Linting with auto-fix
- `quality types` - Type hint validation
- `quality review` - Code review automation

### Configuration (`wan-cli config`)

- `config validate` - Validate all configurations
- `config unify` - Merge scattered configs
- `config migrate` - Version migration
- `config show` - Display current config

### Documentation (`wan-cli docs`)

- `docs generate` - Generate documentation
- `docs validate` - Check doc integrity
- `docs serve` - Local documentation server
- `docs search` - Search documentation

### Health (`wan-cli health`)

- `health check` - System health monitoring
- `health monitor` - Real-time monitoring
- `health dashboard` - Web dashboard
- `health baseline` - Performance baselines

### Deployment (`wan-cli deploy`)

- `deploy validate` - Pre-deployment checks
- `deploy deploy` - Deploy to environment
- `deploy rollback` - Rollback deployment
- `deploy monitor` - Deployment monitoring

## 🔧 Context Awareness

The CLI automatically adapts based on:

- **Project State**: Detects issues and suggests relevant commands
- **Environment**: Different behavior for dev/staging/production
- **File Changes**: Focuses on changed files when possible
- **Previous Runs**: Learns from past executions
- **Configuration**: Respects project-specific settings

## 🎮 IDE Integration

### VS Code

1. Copy tasks file:

   ```bash
   cp cli/ide_integration/vscode_tasks.json .vscode/tasks.json
   ```

2. Available tasks:
   - **Ctrl+Shift+P** → "Tasks: Run Task" → "WAN: Quick Validation"
   - **F1** → "WAN: Code Quality Check"
   - **Ctrl+Shift+T** → "WAN: Full Test Suite"

### Pre-commit Hooks

Automatically runs on every commit:

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: wan-quick-validation
      name: WAN Quick Validation
      entry: python -m cli.main quick
      language: system
      pass_filenames: false
      always_run: true
```

## 🧪 Testing Framework

### E2E Testing

The CLI includes a comprehensive E2E testing framework:

```python
# Example E2E test
def test_import_fixer():
    with E2ETestFramework() as framework:
        # Arrange: Setup test project with broken imports
        project = framework.setup_test_project("project_with_broken_imports")

        # Act: Run the import fixer
        result = framework.run_tool(["clean", "imports", "--fix"], project)

        # Assert: Verify imports were fixed
        assert result['success']
        framework.assert_file_contains(project / "main.py", "from .utils.helpers")
```

### Test Fixtures

Purpose-built sample projects for testing:

- `test_fixtures/project_with_broken_imports/`
- `test_fixtures/project_with_duplicate_files/`
- `test_fixtures/project_with_flaky_tests/`
- `test_fixtures/project_with_quality_issues/`

### Performance Testing

Ensures tools perform within acceptable limits:

```python
def test_quick_validation_is_fast():
    # Quick validation must complete in under 30 seconds
    duration = time_command(["quick"])
    assert duration < 30
```

## 📚 Documentation System

### Auto-generated Documentation

- **CLI Reference**: Complete command documentation
- **Getting Started**: Step-by-step onboarding guide
- **Best Practices**: Recommended workflows
- **Troubleshooting**: Common issues and solutions

### Training Materials

- **Interactive Tutorials**: Hands-on exercises
- **Video Guides**: Screen recordings of common workflows
- **Team Onboarding**: New developer checklist

## 🔄 Workflow Examples

### Daily Development

```bash
# Morning routine (2 minutes)
wan-cli status          # Check overall health
wan-cli quick           # Quick validation
wan-cli test run --fast # Fast tests

# Before committing (1 minute)
wan-cli quick                    # Pre-commit validation
wan-cli clean imports --fix      # Fix imports
wan-cli quality format           # Format code

# Weekly maintenance (5 minutes)
wan-cli clean all --execute      # Deep cleanup
wan-cli health baseline --compare # Performance trends
wan-cli docs generate            # Update docs
```

### Problem Solving

```bash
# "My tests are flaky"
wan-cli test flaky --fix

# "My code quality is poor"
wan-cli quality check --fix

# "I have import issues"
wan-cli clean imports --fix

# "My configs are scattered"
wan-cli config unify
```

### CI/CD Pipeline

```bash
# Fast feedback (< 2 minutes)
wan-cli quick

# Full validation (< 10 minutes)
wan-cli test run --coverage
wan-cli quality check --strict
wan-cli deploy validate $ENVIRONMENT
```

## ⚡ Performance Characteristics

- **Quick validation**: < 30 seconds
- **Fast test suite**: < 2 minutes
- **Full test suite**: < 10 minutes
- **Code quality check**: < 1 minute
- **Documentation generation**: < 30 seconds

## 🔧 Configuration

### Smart Defaults

The CLI works out of the box with intelligent defaults:

- Parallel test execution
- Auto-fix where safe
- Coverage reporting
- Smart file filtering

### Customization

Override defaults via:

1. **Configuration files**: `config/unified-config.yaml`
2. **Environment variables**: `WAN_*` prefixed
3. **Command-line options**: Per-command overrides

### Example Configuration

```yaml
# config/unified-config.yaml
quality:
  strict_mode: false
  auto_fix: true

testing:
  parallel: true
  coverage_threshold: 80

cleanup:
  aggressive_mode: false
  backup_before_delete: true

health:
  monitoring_enabled: true
  alert_threshold: 0.7
```

## 🚨 Error Handling

### Graceful Degradation

- Commands continue working even if some tools are unavailable
- Clear error messages with suggested fixes
- Automatic fallbacks for missing dependencies

### Recovery Mechanisms

- Automatic backup before destructive operations
- Rollback capabilities for failed operations
- Safe mode for critical operations

## 📊 Monitoring and Analytics

### Health Monitoring

- Real-time system health tracking
- Performance baseline comparisons
- Automated alerting for regressions

### Usage Analytics

- Command usage patterns
- Performance metrics
- Error rate tracking

## 🔮 Future Enhancements

### Planned Features

- **AI-powered suggestions**: Smart recommendations based on project analysis
- **Team collaboration**: Shared configurations and best practices
- **Plugin system**: Extensible architecture for custom tools
- **Cloud integration**: Remote execution and monitoring
- **Advanced analytics**: Predictive maintenance and optimization

### Extensibility

The CLI is designed for easy extension:

```python
# Add new command group
from cli.main import app
from typer import Typer

custom_app = Typer()

@custom_app.command()
def my_command():
    """Custom command"""
    pass

app.add_typer(custom_app, name="custom")
```

## 🤝 Contributing

### Adding New Commands

1. Create command module in `cli/commands/`
2. Add to main CLI in `cli/main.py`
3. Write E2E tests in `cli/testing/`
4. Update documentation

### Testing Your Changes

```bash
# Run CLI tests
python -m pytest tests/test_unified_cli.py -v

# Run E2E tests
python -m pytest cli/testing/e2e_test_framework.py -v

# Test CLI directly
python -m cli.main --help
```

## 📝 License

This CLI system is part of the WAN project and follows the same licensing terms.

---

**The WAN CLI: One tool to rule them all** 🚀
