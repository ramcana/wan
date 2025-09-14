# Model Orchestrator Migration Guide

This guide covers the migration and compatibility tools for transitioning from legacy model configurations to the new WAN Model Orchestrator system.

## Overview

The migration tools provide:

- **Configuration Migration**: Convert legacy `config.json` to `models.toml` manifest
- **Path Migration**: Move model files from legacy locations to orchestrator structure
- **Backward Compatibility**: Adapters for legacy path resolution
- **Feature Flags**: Gradual rollout control with environment-based configuration
- **Validation Tools**: Comprehensive manifest and configuration validation
- **Rollback System**: Safe rollback to previous configurations

## Quick Start

### 1. Check Current Configuration

```bash
# Show current feature flags
python -m backend.core.model_orchestrator.migration_cli show-flags

# Validate existing manifest (if any)
python -m backend.core.model_orchestrator.migration_cli validate-manifest config/models.toml
```

### 2. Create Rollback Point

```bash
# Create backup before migration
python -m backend.core.model_orchestrator.migration_cli create-rollback config.json config/models.toml
```

### 3. Migrate Configuration

```bash
# Migrate legacy config.json to models.toml
python -m backend.core.model_orchestrator.migration_cli migrate-config \
    config.json \
    config/models.toml \
    --legacy-models-dir models \
    --scan-files
```

### 4. Validate Migration

```bash
# Validate migrated manifest
python -m backend.core.model_orchestrator.migration_cli validate-manifest \
    config/models.toml \
    --legacy-config config.json
```

### 5. Migrate Model Files (Optional)

```bash
# Migrate model files to new structure
python -m backend.core.model_orchestrator.migration_cli migrate-paths \
    models \
    orchestrator_models \
    t2v-A14B i2v-A14B ti2v-5B \
    --dry-run
```

## Feature Flags

Control the rollout of orchestrator features using environment variables:

### Core Features

```bash
export WAN_ENABLE_ORCHESTRATOR=true          # Enable model orchestrator
export WAN_ENABLE_MANIFEST_VALIDATION=true   # Enable manifest validation
export WAN_ENABLE_LEGACY_FALLBACK=true       # Enable legacy path fallback
export WAN_ENABLE_PATH_MIGRATION=false       # Enable automatic path migration
export WAN_ENABLE_AUTO_DOWNLOAD=false        # Enable automatic downloads
```

### Safety Features

```bash
export WAN_STRICT_VALIDATION=false           # Enable strict validation mode
export WAN_ENABLE_INTEGRITY_CHECKS=true      # Enable file integrity checks
export WAN_ENABLE_DISK_SPACE_CHECKS=true     # Enable disk space checks
export WAN_ENABLE_CONCURRENT_DOWNLOADS=false # Enable concurrent downloads
```

### Storage Backends

```bash
export WAN_ENABLE_S3_BACKEND=false           # Enable S3/MinIO backend
export WAN_ENABLE_HF_BACKEND=true            # Enable HuggingFace backend
export WAN_ENABLE_LOCAL_BACKEND=true         # Enable local storage backend
```

### Monitoring

```bash
export WAN_ENABLE_METRICS=false              # Enable Prometheus metrics
export WAN_ENABLE_STRUCTURED_LOGGING=true    # Enable structured logging
export WAN_ENABLE_HEALTH_ENDPOINTS=true      # Enable health endpoints
```

### Development

```bash
export WAN_DEBUG_MODE=false                  # Enable debug mode
export WAN_DRY_RUN_MODE=false                # Enable dry-run mode
export WAN_VERBOSE_LOGGING=false             # Enable verbose logging
```

## Rollout Stages

The system supports gradual rollout through predefined stages:

### Stage 1: Development

- Basic orchestrator functionality
- Manifest validation
- Legacy fallback support
- Local and HuggingFace backends

### Stage 2: Staging

- All development features
- Integrity checks
- Disk space management
- Health endpoints
- Structured logging

### Stage 3: Production

- All staging features
- Concurrent downloads
- Metrics collection
- Performance tracking
- S3 backend support
- Component deduplication

## Migration Workflow

### 1. Pre-Migration Assessment

```python
from backend.core.model_orchestrator.migration_manager import ConfigurationMigrator
from backend.core.model_orchestrator.validation_tools import ComprehensiveValidator

# Load and analyze legacy configuration
migrator = ConfigurationMigrator()
legacy_config = migrator.load_legacy_config("config.json")

# Check what models are currently configured
print("Legacy models:", legacy_config.models)
```

### 2. Configuration Migration

```python
# Perform migration with file scanning
result = migrator.migrate_configuration(
    legacy_config_path="config.json",
    output_manifest_path="config/models.toml",
    legacy_models_dir="models",
    backup=True,
    scan_files=True
)

if result.success:
    print("Migration successful!")
    print(f"Manifest: {result.manifest_path}")
    if result.backup_path:
        print(f"Backup: {result.backup_path}")
else:
    print("Migration failed:")
    for error in result.errors:
        print(f"  - {error}")
```

### 3. Validation

```python
# Comprehensive validation
validator = ComprehensiveValidator()
report = validator.validate_manifest("config/models.toml")

print(f"Validation: {'PASSED' if report.valid else 'FAILED'}")
print(f"Issues: {report.summary['total']} total, {report.summary['errors']} errors")

# Show issues by category
for category in ['schema', 'security', 'performance', 'compatibility']:
    issues = report.get_issues_by_category(category)
    if issues:
        print(f"\n{category.title()} Issues:")
        for issue in issues:
            print(f"  [{issue.severity.upper()}] {issue.message}")
```

### 4. Path Migration

```python
from backend.core.model_orchestrator.migration_manager import LegacyPathAdapter

# Set up path adapter
adapter = LegacyPathAdapter(
    legacy_models_dir="models",
    orchestrator_models_root="orchestrator_models"
)

# Migrate model files
for model_name in ["t2v-A14B", "i2v-A14B", "ti2v-5B"]:
    if adapter.path_exists_in_legacy(model_name):
        # Dry run first
        success = adapter.migrate_model_files(model_name, dry_run=True)
        print(f"Dry run {model_name}: {'✓' if success else '✗'}")

        # Actual migration
        if success:
            adapter.migrate_model_files(model_name, dry_run=False)
            print(f"Migrated {model_name}")
```

## Backward Compatibility

### Legacy Path Resolution

The system provides backward compatibility through path adapters:

```python
from backend.core.model_orchestrator.migration_manager import LegacyPathAdapter

adapter = LegacyPathAdapter("models", "orchestrator_models")

# Map legacy paths to new paths
legacy_path = adapter.get_legacy_path("t2v-A14B")
new_path = adapter.map_legacy_path("t2v-A14B")

print(f"Legacy: {legacy_path}")
print(f"New: {new_path}")
```

### Feature Flag Compatibility

Use feature flags to maintain compatibility during transition:

```python
from backend.core.model_orchestrator.feature_flags import is_feature_enabled

# Check if orchestrator is enabled
if is_feature_enabled('enable_orchestrator'):
    # Use new orchestrator system
    from backend.core.model_orchestrator.model_ensurer import ModelEnsurer
    ensurer = ModelEnsurer()
    model_path = ensurer.ensure("t2v-A14B@2.2.0")
else:
    # Fall back to legacy system
    model_path = "models/t2v-A14B"
```

## Rollback Procedures

### Creating Rollback Points

```bash
# Create rollback point before major changes
python -m backend.core.model_orchestrator.migration_cli create-rollback \
    config.json config/models.toml \
    --rollback-dir .rollbacks
```

### Executing Rollbacks

```bash
# List available rollback points
python -m backend.core.model_orchestrator.migration_cli list-rollbacks

# Execute rollback
python -m backend.core.model_orchestrator.migration_cli execute-rollback rollback_1234567890
```

### Programmatic Rollback

```python
from backend.core.model_orchestrator.migration_manager import RollbackManager

manager = RollbackManager()

# Create rollback point
rollback_id = manager.create_rollback_point(
    config_paths=["config.json", "config/models.toml"],
    rollback_dir=".rollbacks"
)

# Later, execute rollback if needed
success = manager.execute_rollback(rollback_id, ".rollbacks")
```

## Validation Tools

### Schema Validation

```python
from backend.core.model_orchestrator.validation_tools import ManifestSchemaValidator

validator = ManifestSchemaValidator()
report = validator.validate_manifest_schema("config/models.toml")

if not report.valid:
    for issue in report.get_issues_by_severity("error"):
        print(f"ERROR: {issue.message}")
```

### Security Validation

```python
from backend.core.model_orchestrator.validation_tools import SecurityValidator

validator = SecurityValidator()
report = validator.validate_security("config/models.toml")

security_issues = report.get_issues_by_category("security")
for issue in security_issues:
    print(f"[{issue.severity.upper()}] {issue.message}")
    if issue.suggestion:
        print(f"  Suggestion: {issue.suggestion}")
```

### Performance Validation

```python
from backend.core.model_orchestrator.validation_tools import PerformanceValidator

validator = PerformanceValidator()
report = validator.validate_performance("config/models.toml")

for issue in report.get_issues_by_category("performance"):
    print(f"Performance: {issue.message}")
```

### File Integrity Validation

```python
from backend.core.model_orchestrator.validation_tools import ComprehensiveValidator

validator = ComprehensiveValidator()

# Validate individual file
report = validator.validate_file_integrity(
    "models/t2v-A14B/model.safetensors",
    expected_sha256="abc123...",
    expected_size=1073741824
)

if not report.valid:
    for issue in report.issues:
        print(f"Integrity issue: {issue.message}")
```

## Troubleshooting

### Common Migration Issues

1. **Missing tomli_w package**

   ```bash
   pip install tomli_w
   ```

2. **Permission errors during file migration**

   ```bash
   # Run with appropriate permissions
   sudo python -m backend.core.model_orchestrator.migration_cli migrate-paths ...
   ```

3. **Path length issues on Windows**
   ```bash
   # Enable long path support or use shorter paths
   git config --system core.longpaths true
   ```

### Validation Failures

1. **Schema validation errors**

   - Check manifest syntax with TOML validator
   - Ensure all required fields are present
   - Verify model ID format (model@version)

2. **Security validation warnings**

   - Review executable files in manifest
   - Check for path traversal attempts
   - Verify source URL security

3. **Performance validation warnings**
   - Consider model compression for large files
   - Bundle small files for better performance
   - Review VRAM requirements

### Feature Flag Issues

1. **Features not enabling**

   ```bash
   # Check environment variables
   env | grep WAN_

   # Verify feature flag loading
   python -c "from backend.core.model_orchestrator.feature_flags import get_feature_flags; print(get_feature_flags().to_json())"
   ```

2. **Configuration conflicts**

   ```python
   from backend.core.model_orchestrator.feature_flags import get_feature_flags

   flags = get_feature_flags()
   issues = flags.validate_configuration()

   for issue_key, issue_msg in issues.items():
       print(f"{issue_key}: {issue_msg}")
   ```

## Best Practices

### Migration Planning

1. **Test in development first**

   - Use dry-run mode for all operations
   - Validate on a copy of production data
   - Test rollback procedures

2. **Gradual rollout**

   - Start with development stage features
   - Monitor system behavior
   - Gradually enable production features

3. **Backup everything**
   - Create rollback points before changes
   - Keep multiple backup generations
   - Test restore procedures

### Validation Strategy

1. **Multi-layer validation**

   - Schema validation for structure
   - Security validation for safety
   - Performance validation for efficiency
   - Compatibility validation for cross-platform support

2. **Continuous validation**
   - Validate after each migration step
   - Set up automated validation in CI/CD
   - Monitor validation metrics

### Feature Flag Management

1. **Environment-based configuration**

   - Use environment variables for deployment-specific settings
   - Document all feature flags
   - Use configuration files for complex setups

2. **Staged rollouts**
   - Follow the predefined rollout stages
   - Monitor system health at each stage
   - Have rollback plans ready

## API Reference

### Migration Manager Classes

- `ConfigurationMigrator`: Migrate legacy configurations
- `LegacyPathAdapter`: Handle legacy path compatibility
- `ManifestValidator`: Validate manifest files
- `RollbackManager`: Manage configuration rollbacks

### Validation Tools

- `ManifestSchemaValidator`: Schema and structure validation
- `SecurityValidator`: Security issue detection
- `PerformanceValidator`: Performance analysis
- `CompatibilityValidator`: Cross-platform compatibility
- `ComprehensiveValidator`: All-in-one validation

### Feature Flags

- `OrchestratorFeatureFlags`: Feature flag configuration
- `FeatureFlagManager`: Feature flag management
- `get_feature_flags()`: Global feature flag access
- `is_feature_enabled()`: Check individual features

## Examples

See `migration_demo.py` for a complete demonstration of all migration and validation features.

```bash
# Run the demo
python -m backend.core.model_orchestrator.migration_demo
```

This will demonstrate:

- Feature flag configuration
- Complete migration workflow
- Validation tools usage
- Path migration
- Rollback procedures
