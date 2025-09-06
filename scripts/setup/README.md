# Pre-commit Hooks Setup

This directory contains setup scripts for installing and configuring pre-commit hooks for the WAN22 Video Generation System.

## Quick Setup

### Windows

```bash
scripts\setup\install_pre_commit_hooks.bat
```

### Linux/macOS

```bash
python scripts/setup/install_pre_commit_hooks.py
```

## What Gets Installed

The pre-commit hooks system includes:

### Code Quality Checks

- **Black**: Python code formatting
- **isort**: Import statement sorting
- **Flake8**: Python linting
- **Bandit**: Security vulnerability scanning

### Project-Specific Checks

- **Test Health Check**: Validates test organization and functionality
- **Configuration Validation**: Checks config files for syntax and security
- **Documentation Link Check**: Validates internal links in markdown files
- **Import Validation**: Ensures import paths are correct

### Frontend Checks (if applicable)

- **Prettier**: Code formatting for JS/TS/CSS
- **ESLint**: JavaScript/TypeScript linting

## Hook Behavior

### On Commit

The following hooks run automatically on `git commit`:

- Code formatting (auto-fixes)
- Linting checks
- Test health validation
- Configuration validation
- Documentation link checking
- Import path validation

### On Push

Additional hooks run on `git push`:

- Performance regression checks
- Comprehensive security scans

## Manual Usage

### Run All Hooks

```bash
pre-commit run --all-files
```

### Run Specific Hook

```bash
pre-commit run black
pre-commit run test-health-check
```

### Skip Hooks (Emergency)

```bash
git commit --no-verify
```

### Update Hooks

```bash
pre-commit autoupdate
```

## Configuration

The pre-commit configuration is defined in `.pre-commit-config.yaml` at the project root.

### Custom Hook Scripts

- `tools/health-checker/pre_commit_test_health.py`
- `tools/config-manager/pre_commit_config_validation.py`
- `tools/doc-generator/pre_commit_link_check.py`
- `tools/health-checker/pre_commit_import_validation.py`

## Troubleshooting

### Hook Fails to Run

1. Check Python version (3.8+ required)
2. Ensure all dependencies are installed
3. Verify file permissions on Unix systems

### Performance Issues

- Hooks are designed to run quickly
- Use `--files` flag to run on specific files only
- Consider adjusting hook configuration for large repositories

### Bypassing Hooks

Only bypass hooks in emergencies:

```bash
git commit --no-verify -m "Emergency fix"
```

## Integration with CI/CD

The same checks run in GitHub Actions workflows:

- `.github/workflows/test-suite.yml`
- `.github/workflows/config-validation.yml`
- `.github/workflows/docs-build.yml`

This ensures consistency between local development and CI/CD environments.
