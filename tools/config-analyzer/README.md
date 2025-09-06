# Configuration Landscape Analyzer

A comprehensive tool for analyzing configuration files across a project, identifying conflicts, dependencies, and providing consolidation recommendations.

## Features

- **Comprehensive Scanning**: Automatically discovers all configuration files (JSON, YAML, INI, ENV, TOML)
- **Conflict Detection**: Identifies conflicting settings across different configuration files
- **Dependency Analysis**: Maps relationships between configuration files
- **Consolidation Recommendations**: Provides actionable recommendations for configuration cleanup
- **Migration Planning**: Generates detailed migration plans for configuration consolidation

## Installation

No additional dependencies required beyond Python standard library and PyYAML:

```bash
pip install pyyaml
```

## Usage

### Basic Analysis

```bash
# Analyze current project
python tools/config-analyzer/cli.py analyze

# Analyze specific directory
python tools/config-analyzer/cli.py analyze --project-root /path/to/project
```

### Detailed Analysis

```bash
# Show configuration conflicts
python tools/config-analyzer/cli.py analyze --show-conflicts

# Show detailed file information
python tools/config-analyzer/cli.py analyze --show-files

# Show migration plan
python tools/config-analyzer/cli.py analyze --show-migration-plan

# Show everything
python tools/config-analyzer/cli.py analyze --show-conflicts --show-files --show-migration-plan
```

### Export Reports

```bash
# Export to JSON
python tools/config-analyzer/cli.py analyze --export config_analysis.json

# Export to YAML
python tools/config-analyzer/cli.py analyze --export config_analysis.yaml --format yaml
```

### View Existing Reports

```bash
# Show summary from existing report
python tools/config-analyzer/cli.py summary config_analysis.json
```

## Supported Configuration Formats

- **JSON**: `.json` files
- **YAML**: `.yaml`, `.yml` files
- **INI**: `.ini`, `.cfg`, `.conf` files
- **Environment**: `.env` files
- **TOML**: `.toml` files (requires `tomllib` for Python 3.11+)

## Special File Detection

The analyzer also recognizes configuration files by common naming patterns:

- `config.*`
- `*config.*`
- `settings.*`
- `docker-compose.yml`
- `pytest.ini`
- `setup.cfg`
- `pyproject.toml`

## Output Sections

### Overview

- Total configuration files found
- Breakdown by file type
- Summary of issues detected

### Conflicts

- Settings that have different values across files
- Severity classification (high/medium/low)
- Affected files and values

### Dependencies

- References between configuration files
- Import relationships
- Override hierarchies

### Recommendations

- Consolidation suggestions
- Standardization recommendations
- Organization improvements

### Migration Plan

- Phased approach to configuration consolidation
- File mapping to unified structure
- Priority and complexity assessment

## Example Output

```
============================================================
CONFIGURATION LANDSCAPE ANALYSIS SUMMARY
============================================================

📊 OVERVIEW:
  Total configuration files found: 23
  File types:
    JSON: 12 files
    YAML: 8 files
    ENV: 3 files

⚠️  ISSUES DETECTED:
  Configuration conflicts: 5
  Duplicate settings: 12
  Dependencies: 3

🔥 HIGH PRIORITY CONFLICTS:
    database.port: 2 files with different values
    api.host: 3 files with different values

💡 RECOMMENDATIONS:
  1. Consider consolidating 23 configuration files into a unified configuration system
  2. Standardize on a single configuration format (YAML recommended)
  3. Implement environment-specific configuration overrides
  4. Configuration files are scattered across 8 directories. Consider centralizing them

📋 MIGRATION PLAN:
  Phase 1: Backup and Analysis
  Phase 2: Unified Schema Design
  Phase 3: Implementation
  Phase 4: Migration
```

## Integration with Development Workflow

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: config-analysis
        name: Configuration Analysis
        entry: python tools/config-analyzer/cli.py analyze --export config_analysis.json
        language: system
        pass_filenames: false
        always_run: true
```

### CI/CD Integration

```yaml
# .github/workflows/config-validation.yml
name: Configuration Validation
on: [push, pull_request]

jobs:
  config-analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.9"
      - name: Install dependencies
        run: pip install pyyaml
      - name: Run configuration analysis
        run: |
          python tools/config-analyzer/cli.py analyze --export config_analysis.json
          python tools/config-analyzer/cli.py analyze --show-conflicts
      - name: Upload analysis results
        uses: actions/upload-artifact@v3
        with:
          name: config-analysis
          path: config_analysis.json
```

## API Usage

For programmatic usage:

```python
from tools.config_analyzer.config_landscape_analyzer import ConfigLandscapeAnalyzer

# Create analyzer
analyzer = ConfigLandscapeAnalyzer('.')

# Generate report
report = analyzer.generate_report()

# Access results
print(f"Found {report.total_files} configuration files")
print(f"Detected {len(report.conflicts)} conflicts")

# Get recommendations
for recommendation in report.recommendations:
    print(f"- {recommendation}")
```

## Configuration File Structure Analysis

The analyzer provides detailed insights into:

- **File Size**: Identifies oversized configuration files
- **Complexity**: Detects deeply nested configurations
- **Settings Count**: Tracks number of configuration options
- **Dependencies**: Maps file relationships
- **Conflicts**: Identifies contradictory settings

## Migration Planning

The tool generates a comprehensive migration plan including:

1. **Backup Strategy**: Safe backup of existing configurations
2. **Schema Design**: Unified configuration structure
3. **File Mapping**: How existing files map to new structure
4. **Priority Assessment**: Which files to migrate first
5. **Validation Steps**: How to verify migration success

## Best Practices

1. **Run Regularly**: Include in CI/CD pipeline for continuous monitoring
2. **Review Conflicts**: Address high-priority conflicts first
3. **Gradual Migration**: Follow the phased migration plan
4. **Backup First**: Always backup configurations before changes
5. **Validate Changes**: Test thoroughly after configuration changes

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure read access to all configuration files
2. **Encoding Issues**: Files should be UTF-8 encoded
3. **Large Files**: Very large config files may need manual review
4. **Complex Structures**: Deeply nested configs may require custom handling

### Debug Mode

For debugging issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

analyzer = ConfigLandscapeAnalyzer('.')
report = analyzer.generate_report()
```

## Contributing

To extend the analyzer:

1. Add new file type support in `CONFIG_EXTENSIONS`
2. Add new naming patterns in `CONFIG_PATTERNS`
3. Extend conflict detection logic in `detect_conflicts()`
4. Add new recommendation rules in `generate_recommendations()`
