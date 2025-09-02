# Fast Feedback Development Tools

This directory contains tools for fast feedback during development, including watch modes, selective test execution, and debugging utilities.

## Components

- `test_watcher.py` - Watch mode for tests with selective execution
- `config_watcher.py` - Development server with hot-reloading for configuration changes
- `debug_tools.py` - Debugging tools with comprehensive logging and error reporting
- `feedback_cli.py` - Command-line interface for fast feedback tools

## Usage

### Test Watcher

```bash
# Watch all tests
python tools/dev-feedback/test_watcher.py

# Watch specific test categories
python tools/dev-feedback/test_watcher.py --category unit
python tools/dev-feedback/test_watcher.py --category integration

# Watch specific files or patterns
python tools/dev-feedback/test_watcher.py --pattern "test_*.py"
python tools/dev-feedback/test_watcher.py --files backend/tests/test_api.py

# Fast mode (skip slow tests)
python tools/dev-feedback/test_watcher.py --fast
```

### Configuration Watcher

```bash
# Watch configuration files for changes
python tools/dev-feedback/config_watcher.py

# Watch specific config files
python tools/dev-feedback/config_watcher.py --files config/unified-config.yaml

# Auto-reload services on config changes
python tools/dev-feedback/config_watcher.py --reload-services
```

### Debug Tools

```bash
# Enable debug logging
python tools/dev-feedback/debug_tools.py --enable

# Analyze logs
python tools/dev-feedback/debug_tools.py --analyze

# Generate debug report
python tools/dev-feedback/debug_tools.py --report debug_report.json
```

### CLI Interface

```bash
# Interactive feedback tools
python tools/dev-feedback/feedback_cli.py

# Start test watcher
python tools/dev-feedback/feedback_cli.py watch-tests

# Start config watcher
python tools/dev-feedback/feedback_cli.py watch-config

# Debug session
python tools/dev-feedback/feedback_cli.py debug
```

## Features

- Real-time test execution on file changes
- Selective test execution based on changed files
- Configuration hot-reloading
- Comprehensive debug logging
- Performance monitoring
- Error analysis and reporting
- Integration with existing test infrastructure
