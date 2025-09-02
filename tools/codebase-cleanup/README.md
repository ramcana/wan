# Codebase Cleanup and Organization Tools

Comprehensive tools for cleaning up and organizing codebases, focusing on duplicate detection, dead code removal, and naming standardization.

## Features

### 1. Duplicate Detection and Removal (`duplicate_detector.py`)

- **Exact Duplicate Detection**: Identifies files with identical content using MD5 hashing
- **Near-Duplicate Detection**: Finds similar files using content similarity analysis
- **Code Similarity Analysis**: Compares code structure using AST parsing for Python files
- **Safe Removal**: Creates backups before removing duplicates with rollback capability
- **Smart Filtering**: Excludes common build/cache directories and very small files

### 2. Dead Code Analysis and Removal (`dead_code_analyzer.py`)

- **Unused Function Detection**: Identifies functions that are never called
- **Dead Class Analysis**: Finds classes that are never instantiated
- **Unused Import Detection**: Locates imports that are never used
- **Dead File Identification**: Finds files that are never imported or referenced
- **Safe Removal**: Comprehensive testing before removal with backup capabilities

### 3. Naming Standardization (`naming_standardizer.py`)

- **Convention Analysis**: Identifies inconsistent naming patterns
- **Multi-Convention Support**: Supports snake_case, camelCase, PascalCase, kebab-case
- **Safe Refactoring**: Updates references when renaming with backup and rollback
- **File Organization**: Groups related functionality logically
- **Automated Enforcement**: Tools for CI/CD integration

## Installation

```bash
# Install from project root
pip install -e .

# Or run directly
python -m tools.codebase-cleanup --help
```

## Usage

### Command Line Interface

```bash
# Scan for duplicates
python -m tools.codebase-cleanup duplicates

# Remove exact duplicates automatically
python -m tools.codebase-cleanup duplicates --remove

# Analyze dead code
python -m tools.codebase-cleanup dead-code

# Standardize naming conventions
python -m tools.codebase-cleanup naming --convention snake_case

# Rollback duplicate removal
python -m tools.codebase-cleanup duplicates --rollback backups/duplicates/duplicate_removal_20240101_120000
```

### Programmatic Usage

```python
from tools.codebase_cleanup.duplicate_detector import DuplicateDetector

# Initialize detector
detector = DuplicateDetector(".", backup_dir="backups/duplicates")

# Scan for duplicates
report = detector.scan_for_duplicates()

# Print results
print(f"Found {len(report.duplicate_files)} duplicate files")
print(f"Potential savings: {report.potential_savings / 1024:.1f} KB")

# Remove duplicates safely
if report.duplicate_groups:
    results = detector.safe_remove_duplicates(
        report.duplicate_groups,
        auto_remove_exact=True
    )
    print(results)
```

## Configuration

### Exclusion Patterns

The tools automatically exclude common directories and files:

- `__pycache__`, `.git`, `.pytest_cache`
- `node_modules`, `dist`, `build`
- `.venv`, `venv`, `.env`
- Files smaller than 10 bytes

### Similarity Thresholds

- **Exact Duplicates**: 100% identical content (MD5 hash match)
- **Near Duplicates**: 80% content similarity (configurable)
- **Code Similarity**: AST-based comparison for code files

## Reports

All tools generate detailed JSON reports with:

- **Duplicate Report**: File paths, sizes, similarity scores, recommendations
- **Dead Code Report**: Unused functions, classes, imports, files
- **Naming Report**: Convention violations, suggested fixes, impact analysis

## Safety Features

### Backup and Rollback

- Automatic backup creation before any destructive operations
- Complete rollback capability using backup manifests
- Preserves directory structure in backups

### Validation

- Comprehensive testing before code removal
- Import dependency analysis
- Reference checking before renaming

### Dry Run Mode

- Preview changes before applying them
- Detailed impact analysis
- Risk assessment for each operation

## Integration

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: duplicate-check
        name: Check for duplicates
        entry: python -m tools.codebase-cleanup duplicates
        language: system
        pass_filenames: false
```

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Check code quality
  run: |
    python -m tools.codebase-cleanup duplicates --output duplicates.json
    python -m tools.codebase-cleanup dead-code --output dead-code.json
    python -m tools.codebase-cleanup naming --output naming.json
```

## Examples

### Duplicate Detection Example

```bash
$ python -m tools.codebase-cleanup duplicates
Scanning for duplicates in: .
Scanning 1247 files...
Scan complete. Found 23 duplicate files.

Duplicate Detection Results:
Files scanned: 1247
Duplicate files found: 23
Potential savings: 156.3 KB
Duplicate groups: 8

Recommendations:
- Found 5 groups of exact duplicates - safe to remove
- Found 3 groups of similar files - review before removal
- Configuration file duplicates found - consolidate to avoid conflicts
```

### Dead Code Analysis Example

```bash
$ python -m tools.codebase-cleanup dead-code
Analyzing dead code in: .

Dead Code Analysis Results:
Files analyzed: 342
Dead functions: 15
Dead classes: 3
Unused imports: 47
Dead files: 2

Recommendations:
- Remove 15 unused functions to reduce complexity
- Clean up 47 unused imports to improve load times
- Consider removing 2 dead files after verification
```

## Best Practices

1. **Always Review Before Removal**: Check reports before applying automatic fixes
2. **Test After Changes**: Run full test suite after cleanup operations
3. **Incremental Cleanup**: Process one type of issue at a time
4. **Backup Verification**: Verify backups before proceeding with removals
5. **Team Coordination**: Coordinate cleanup activities with team members

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure write permissions for backup directories
2. **Large File Processing**: May be slow for very large codebases
3. **Binary File Handling**: Some binary files may not be processed correctly

### Performance Optimization

- Use `--exclude-pattern` to skip unnecessary directories
- Process smaller subdirectories separately for large projects
- Run during off-peak hours for large cleanup operations

## Contributing

1. Add new detection algorithms to respective analyzer classes
2. Extend CLI with new commands and options
3. Add tests for new functionality
4. Update documentation with examples

## License

Part of the WAN22 project cleanup and quality improvement initiative.
