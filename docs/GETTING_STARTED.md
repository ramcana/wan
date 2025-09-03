# Getting Started with WAN CLI

Welcome to the WAN Project Quality & Maintenance Toolkit! This guide will get you up and running in minutes.

## What is WAN CLI?

WAN CLI is your unified development companion that brings together all project tools into a single, intelligent command-line interface. Instead of remembering dozens of different commands, you have one tool that does it all.

## Quick Setup

### 1. First Command

```bash
# I just cloned the project. What's the first command I run?
python -m cli.main init
```

This will:

- Validate your environment
- Set up necessary configurations
- Run initial health checks
- Guide you through any required setup

### 2. Check Project Health

```bash
# How do I see the project structure and health?
python -m cli.main status
```

This shows you:

- Overall project health score
- Critical issues that need attention
- Quick recommendations

### 3. Quick Validation

```bash
# Is everything working correctly?
python -m cli.main quick
```

This runs fast checks for:

- Import issues
- Syntax errors
- Configuration problems
- Basic test smoke tests

## Common Scenarios

### "I have a flaky test, what do I do?"

```bash
# Find flaky tests
wan-cli test flaky

# Get help on fixing them
wan-cli test flaky --help

# Attempt automatic fixes
wan-cli test flaky --fix
```

### "My code quality is poor"

```bash
# Check what's wrong
wan-cli quality check

# Fix automatically where possible
wan-cli quality check --fix

# Format code properly
wan-cli quality format
```

### "My imports are messy"

```bash
# See what's wrong with imports
wan-cli clean imports

# Fix them automatically
wan-cli clean imports --fix
```

### "I want to clean up the codebase"

```bash
# See what can be cleaned (dry run)
wan-cli clean all

# Actually perform the cleanup
wan-cli clean all --execute
```

### "I need to run tests"

```bash
# Run fast tests for quick feedback
wan-cli test run --fast

# Run full test suite with coverage
wan-cli test run --coverage

# Run tests matching a pattern
wan-cli test run --pattern="test_auth"
```

## Daily Workflow

### Morning Routine (2 minutes)

```bash
wan-cli status          # Check overall health
wan-cli quick           # Quick validation
wan-cli test run --fast # Fast test run
```

### Before Committing (1 minute)

```bash
wan-cli quick                    # Quick validation
wan-cli clean imports --fix      # Fix imports
wan-cli quality format           # Format code
```

### Weekly Maintenance (5 minutes)

```bash
wan-cli clean all --execute      # Deep cleanup
wan-cli health check --detailed  # Detailed health check
wan-cli docs generate            # Update docs
wan-cli health baseline --compare # Check performance trends
```

## IDE Integration

### VS Code Users

1. Copy the tasks file:

   ```bash
   cp cli/ide_integration/vscode_tasks.json .vscode/tasks.json
   ```

2. Now you can use:
   - **Ctrl+Shift+P** → "Tasks: Run Task" → "WAN: Quick Validation"
   - **F1** → "WAN: Code Quality Check"
   - And many more tasks available in the command palette

### Any Editor

You can run any command from your editor's terminal or task runner:

```bash
# Quick validation
python -m cli.main quick

# Fix code quality
python -m cli.main quality check --fix

# Run tests
python -m cli.main test run --fast
```

## Understanding the Output

### Health Scores

- **90-100%**: Excellent health, keep it up!
- **70-89%**: Good health, minor issues to address
- **50-69%**: Moderate issues, needs attention
- **Below 50%**: Serious issues, immediate action required

### Exit Codes

- **0**: Success, all good
- **1**: Issues found or command failed
- **2**: Critical errors, immediate attention needed

### Verbosity Levels

```bash
# Normal output
wan-cli status

# Verbose output (more details)
wan-cli status --verbose

# Quiet mode (minimal output)
wan-cli status --quiet
```

## Configuration

The CLI uses smart defaults, but you can customize:

### Global Configuration

Edit `config/unified-config.yaml`:

```yaml
quality:
  strict_mode: false
  auto_fix: true

testing:
  parallel: true
  coverage_threshold: 80

cleanup:
  aggressive_mode: false
  backup_before_delete: true
```

### Environment Variables

```bash
export WAN_VERBOSE=true
export WAN_AUTO_FIX=true
export WAN_PARALLEL_TESTS=true
```

### Command-line Options

```bash
# Override config for this run
wan-cli quality check --strict --fix
```

## Getting Help

### Built-in Help

```bash
wan-cli --help                    # General help
wan-cli test --help              # Command group help
wan-cli test run --help          # Specific command help
```

### Common Commands Reference

| Task                | Command                       |
| ------------------- | ----------------------------- |
| Quick health check  | `wan-cli quick`               |
| Run fast tests      | `wan-cli test run --fast`     |
| Fix code quality    | `wan-cli quality check --fix` |
| Clean up codebase   | `wan-cli clean all --execute` |
| Fix imports         | `wan-cli clean imports --fix` |
| Generate docs       | `wan-cli docs generate`       |
| Check system health | `wan-cli health check`        |
| Project status      | `wan-cli status`              |

## Troubleshooting

### "Command not found"

Make sure you're in the project directory:

```bash
cd /path/to/your/project
python -m cli.main --help
```

### "Import errors when running commands"

Fix imports first:

```bash
python -m cli.main clean imports --fix
```

### "Commands run slowly"

Use fast modes:

```bash
wan-cli quick                    # Instead of full validation
wan-cli test run --fast         # Instead of full test suite
wan-cli health check --quick    # Instead of detailed health check
```

### "Configuration errors"

Validate and fix configs:

```bash
wan-cli config validate --fix
```

### "Tool seems broken"

Reset to defaults:

```bash
wan-cli init                     # Re-run initialization
wan-cli config validate --fix    # Fix any config issues
wan-cli status                   # Check overall health
```

## Next Steps

Once you're comfortable with the basics:

1. **Read the full [CLI Reference](CLI_REFERENCE.md)** for all available commands
2. **Set up IDE integration** for seamless development
3. **Configure pre-commit hooks** for automatic quality checks
4. **Explore advanced features** like health monitoring and deployment tools
5. **Customize the configuration** to match your team's preferences

## Pro Tips

1. **Use tab completion**: Most shells support tab completion for commands
2. **Chain commands**: `wan-cli quick && wan-cli test run --fast`
3. **Use aliases**: Add `alias wq="wan-cli quick"` to your shell profile
4. **Monitor trends**: Run `wan-cli health baseline --compare` regularly
5. **Automate with CI**: Add `wan-cli quick` to your CI pipeline

## Need More Help?

- Check the [CLI Reference](CLI_REFERENCE.md) for detailed command documentation
- Look at the [Troubleshooting Guide](TROUBLESHOOTING.md) for common issues
- Review the [Best Practices](BEST_PRACTICES.md) for optimal usage patterns

Happy coding! 🚀
