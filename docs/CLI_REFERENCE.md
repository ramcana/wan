# WAN CLI Reference

The WAN CLI (`wan-cli`) is your unified toolkit for project quality and maintenance.

## Installation

```bash
# Make the CLI available globally
pip install -e .

# Or run directly
python -m cli.main --help
```

## Quick Start

```bash
# First time setup
wan-cli init

# Quick health check
wan-cli quick

# Full project status
wan-cli status
```

## Command Groups

### Testing (`wan-cli test`)

Run and manage your test suite with intelligent defaults.

```bash
# Run all tests with coverage
wan-cli test run --coverage

# Run only fast tests
wan-cli test run --fast

# Detect flaky tests
wan-cli test flaky

# Fix flaky tests automatically
wan-cli test flaky --fix

# Run comprehensive test audit
wan-cli test audit

# Validate test suite integrity
wan-cli test validate
```

### Cleanup (`wan-cli clean`)

Clean up and maintain your codebase automatically.

```bash
# Find duplicates (dry run)
wan-cli clean duplicates

# Remove duplicates
wan-cli clean duplicates --remove

# Find dead code
wan-cli clean dead-code

# Remove dead code
wan-cli clean dead-code --remove

# Fix imports
wan-cli clean imports --fix

# Standardize naming
wan-cli clean naming --fix

# Run all cleanup operations
wan-cli clean all --execute
```

### Configuration (`wan-cli config`)

Manage and validate configuration files.

```bash
# Validate all configs
wan-cli config validate

# Fix config issues
wan-cli config validate --fix

# Unify scattered configs
wan-cli config unify

# Show current config
wan-cli config show

# Set config value
wan-cli config set database.host localhost

# Create backup
wan-cli config backup
```

### Documentation (`wan-cli docs`)

Generate and maintain project documentation.

```bash
# Generate docs
wan-cli docs generate

# Generate and serve docs
wan-cli docs generate --serve

# Validate documentation
wan-cli docs validate

# Fix doc issues
wan-cli docs validate --fix

# Serve docs locally
wan-cli docs serve

# Search documentation
wan-cli docs search "authentication"

# Show doc structure
wan-cli docs structure
```

### Quality (`wan-cli quality`)

Analyze and improve code quality.

```bash
# Run quality checks
wan-cli quality check

# Auto-fix quality issues
wan-cli quality check --fix

# Format code
wan-cli quality format

# Check formatting only
wan-cli quality format --check

# Analyze complexity
wan-cli quality complexity

# Run linting
wan-cli quality lint --fix

# Check type hints
wan-cli quality types --add-hints

# Check documentation
wan-cli quality docs-check --add-missing

# Review code changes
wan-cli quality review
```

### Health (`wan-cli health`)

Monitor and maintain system health.

```bash
# Quick health check
wan-cli health check --quick

# Detailed health analysis
wan-cli health check --detailed

# Real-time monitoring
wan-cli health monitor

# Launch health dashboard
wan-cli health dashboard

# Generate health report
wan-cli health report

# Establish performance baseline
wan-cli health baseline

# Compare with baseline
wan-cli health baseline --compare

# Setup alerts
wan-cli health alerts --setup

# Optimize performance
wan-cli health optimize
```

### Deployment (`wan-cli deploy`)

Manage deployments and production operations.

```bash
# Validate deployment readiness
wan-cli deploy validate production

# Create backup
wan-cli deploy backup

# Deploy to environment
wan-cli deploy deploy production

# Simulate deployment
wan-cli deploy deploy production --dry-run

# Check deployment status
wan-cli deploy status production

# Rollback deployment
wan-cli deploy rollback production

# Monitor deployment
wan-cli deploy monitor production

# View deployment logs
wan-cli deploy logs production --follow
```

## Global Options

All commands support these global options:

- `--verbose, -v`: Enable verbose output
- `--config`: Specify custom config file
- `--help`: Show help for any command

## Examples

### Daily Development Workflow

```bash
# Morning routine
wan-cli status                    # Check overall health
wan-cli test run --fast          # Run fast tests
wan-cli quality check --fix      # Fix any quality issues

# Before committing
wan-cli quick                    # Quick validation
wan-cli clean imports --fix      # Organize imports
wan-cli test run --pattern="test_new_feature"  # Test your changes

# Weekly maintenance
wan-cli clean all --execute      # Deep cleanup
wan-cli health baseline --compare # Check performance trends
wan-cli docs generate            # Update documentation
```

### Fixing Common Issues

```bash
# "My tests are flaky"
wan-cli test flaky --fix

# "My imports are messy"
wan-cli clean imports --fix

# "My code quality is poor"
wan-cli quality check --fix

# "I have duplicate files"
wan-cli clean duplicates --remove

# "My configs are scattered"
wan-cli config unify

# "My documentation is outdated"
wan-cli docs generate --serve
```

### CI/CD Integration

```bash
# In your CI pipeline
wan-cli quick                    # Fast validation
wan-cli test run --coverage      # Full test suite
wan-cli quality check --strict   # Strict quality checks
wan-cli deploy validate $ENV     # Validate deployment
```

## Configuration

The CLI uses smart defaults but can be configured via:

1. `config/unified-config.yaml` - Main configuration
2. Environment variables - `WAN_*` prefixed
3. Command-line options - Override everything

## IDE Integration

### VS Code

Copy `cli/ide_integration/vscode_tasks.json` to `.vscode/tasks.json` to get:

- **Ctrl+Shift+P** → "Tasks: Run Task" → "WAN: Quick Validation"
- **F1** → "WAN: Code Quality Check"
- And many more...

### Pre-commit Hooks

The CLI integrates with pre-commit hooks automatically:

```bash
# Install hooks
pre-commit install

# Test hooks
pre-commit run --all-files
```

## Troubleshooting

### Common Issues

1. **"Command not found"**

   ```bash
   # Make sure you're in the project directory
   python -m cli.main --help
   ```

2. **"Import errors"**

   ```bash
   # Fix imports first
   wan-cli clean imports --fix
   ```

3. **"Tool runs slowly"**

   ```bash
   # Use quick mode
   wan-cli quick
   wan-cli test run --fast
   ```

4. **"Configuration errors"**
   ```bash
   # Validate and fix configs
   wan-cli config validate --fix
   ```

### Getting Help

- `wan-cli --help` - General help
- `wan-cli COMMAND --help` - Command-specific help
- `wan-cli COMMAND SUBCOMMAND --help` - Subcommand help

## Advanced Usage

### Custom Workflows

Create custom workflows by chaining commands:

```bash
# Custom quality workflow
wan-cli clean imports --fix && \
wan-cli quality format && \
wan-cli quality check --fix && \
wan-cli test run --fast
```

### Scripting

Use the CLI in scripts:

```bash
#!/bin/bash
set -e

echo "Running pre-deployment checks..."
wan-cli quick
wan-cli test run --coverage
wan-cli deploy validate production

echo "All checks passed! Ready to deploy."
```

### Performance Monitoring

Set up continuous monitoring:

```bash
# In a cron job or CI
wan-cli health baseline --compare
wan-cli health report --format=json > health_report.json
```
