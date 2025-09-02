# Configuration Management System

A comprehensive configuration management system for the WAN22 project that provides unified configuration schema, validation, migration, and API access with hot-reloading capabilities.

## Features

- **Unified Configuration Schema**: Single, comprehensive configuration structure
- **Configuration Migration**: Automatic discovery and migration of scattered config files
- **Validation System**: Comprehensive validation with detailed error reporting
- **Configuration API**: Programmatic access with hot-reloading and change notifications
- **CLI Tools**: Command-line interface for configuration management
- **Environment Overrides**: Environment-specific configuration support

## Components

### 1. UnifiedConfig (`unified_config.py`)

The core configuration schema that consolidates all system settings:

```python
from tools.config_manager import UnifiedConfig

# Create default configuration
config = UnifiedConfig()

# Load from file
config = UnifiedConfig.from_file('config/unified-config.yaml')

# Save to file
config.save_to_file('config/my-config.yaml')

# Convert to dict/JSON/YAML
config_dict = config.to_dict()
json_str = config.to_json()
yaml_str = config.to_yaml()
```

### 2. ConfigurationValidator (`config_validator.py`)

Comprehensive validation system with detailed error reporting:

```python
from tools.config_manager import ConfigurationValidator

validator = ConfigurationValidator()
result = validator.validate_config(config)

print(f"Valid: {result.is_valid}")
print(f"Issues: {len(result.issues)}")

# Generate human-readable report
report = validator.generate_validation_report(result)
print(report)
```

### 3. ConfigurationUnifier (`config_unifier.py`)

Migration system for consolidating scattered configuration files:

```python
from tools.config_manager import ConfigurationUnifier

unifier = ConfigurationUnifier()

# Discover existing config files
sources = unifier.discover_config_files()

# Preview migration
preview = unifier.generate_migration_preview()

# Perform migration
report = unifier.migrate_to_unified_config(
    output_path="config/unified-config.yaml",
    create_backup=True
)
```

### 4. ConfigurationAPI (`config_api.py`)

Programmatic API with hot-reloading and change notifications:

```python
from tools.config_manager import ConfigurationAPI

# Create API instance
with ConfigurationAPI(auto_reload=True) as api:
    # Get configuration values
    port = api.get_config('api.port')

    # Set configuration values
    api.set_config('api.port', 9000)

    # Register change callback
    def on_change(event):
        print(f"Config changed: {event.field_path}")

    api.register_change_callback(on_change)

    # Validate configuration
    result = api.validate_current_config()

    # Apply environment overrides
    api.apply_environment_overrides('production')
```

## CLI Usage

The configuration management system includes a comprehensive CLI tool:

### Basic Operations

```bash
# Get configuration value
python -m tools.config_manager.config_cli get api.port

# Set configuration value
python -m tools.config_manager.config_cli set api.port 9000 --save

# Get entire configuration
python -m tools.config_manager.config_cli get --format yaml

# Show configuration information
python -m tools.config_manager.config_cli info
```

### Validation

```bash
# Validate current configuration
python -m tools.config_manager.config_cli validate

# Validate specific file
python -m tools.config_manager.config_cli validate --config-file config/test.yaml

# Get validation results as JSON
python -m tools.config_manager.config_cli validate --format json
```

### Import/Export

```bash
# Export configuration
python -m tools.config_manager.config_cli export --format yaml --output backup.yaml

# Import configuration
python -m tools.config_manager.config_cli import --input new-config.yaml --save

# Import from stdin
cat config.yaml | python -m tools.config_manager.config_cli import --save
```

### Environment Management

```bash
# Apply environment overrides
python -m tools.config_manager.config_cli environment production --save

# Available environments: development, staging, production, testing
```

### Configuration Watching

```bash
# Watch for configuration file changes
python -m tools.config_manager.config_cli watch
```

## Migration CLI

Separate CLI tool for migrating existing configurations:

```bash
# Discover existing configuration files
python -m tools.config_manager.migration_cli discover

# Preview migration
python -m tools.config_manager.migration_cli preview

# Perform migration
python -m tools.config_manager.migration_cli migrate --output config/unified-config.yaml

# Rollback migration
python -m tools.config_manager.migration_cli rollback /path/to/backup

# Validate migrated configuration
python -m tools.config_manager.migration_cli validate config/unified-config.yaml
```

## Configuration Schema

The unified configuration includes the following sections:

- **system**: Core system settings (name, version, directories)
- **api**: API server configuration (host, port, CORS)
- **database**: Database connection settings
- **models**: Model management configuration
- **hardware**: Hardware optimization settings
- **generation**: Video generation parameters
- **ui**: User interface settings
- **frontend**: Frontend application configuration
- **websocket**: WebSocket communication settings
- **logging**: Logging configuration
- **security**: Security and authentication settings
- **performance**: Performance monitoring configuration
- **recovery**: Error recovery settings
- **environment_validation**: Environment validation rules
- **prompt_enhancement**: Prompt enhancement settings
- **features**: Feature flags

## Environment Overrides

The system supports environment-specific configuration overrides:

```yaml
# config/environments/production.yaml
system:
  debug: false
  log_level: "WARNING"

api:
  workers: 4
  host: "0.0.0.0"

security:
  authentication_enabled: true
```

## Validation Rules

The validation system includes comprehensive checks for:

- **Schema Validation**: JSON Schema validation against defined structure
- **Port Ranges**: Valid port numbers and conflict detection
- **File Paths**: Path validation and portability checks
- **Memory Limits**: VRAM and memory limit consistency
- **Timeout Values**: Reasonable timeout ranges
- **Dependency Consistency**: Cross-setting validation
- **Environment Consistency**: Environment-specific validation
- **Security Settings**: Security best practices
- **Performance Settings**: Performance optimization validation

## Hot-Reloading

The configuration API supports automatic reloading when configuration files change:

```python
# Enable auto-reload
api = ConfigurationAPI(auto_reload=True)

# Register callback for changes
def on_config_change(event):
    print(f"Configuration changed: {event.field_path}")
    # Restart services, update settings, etc.

api.register_change_callback(on_config_change)
```

## Error Handling

The system provides detailed error reporting with:

- **Severity Levels**: INFO, WARNING, ERROR, CRITICAL
- **Fix Suggestions**: Specific recommendations for resolving issues
- **Field Paths**: Exact location of configuration problems
- **Validation Context**: Understanding why validation failed

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI
from tools.config_manager import ConfigurationAPI

app = FastAPI()
config_api = ConfigurationAPI()

@app.on_event("startup")
async def startup():
    # Load configuration
    config = config_api.get_config()

    # Configure application
    app.state.config = config

    # Register for config changes
    config_api.register_change_callback(on_config_change)

def on_config_change(event):
    # Handle configuration changes
    if event.field_path.startswith('api.'):
        # Restart server or update settings
        pass
```

### Service Integration

```python
class VideoGenerationService:
    def __init__(self):
        self.config_api = ConfigurationAPI()
        self.config_api.register_change_callback(self._on_config_change)

    def _on_config_change(self, event):
        if event.field_path.startswith('generation.'):
            self._update_generation_settings()

    def _update_generation_settings(self):
        config = self.config_api.get_config()
        # Update service settings based on new configuration
```

## Best Practices

1. **Use Environment Overrides**: Define environment-specific settings in separate files
2. **Validate Changes**: Always validate configuration changes before applying
3. **Monitor Changes**: Use change callbacks to update services when configuration changes
4. **Backup Before Migration**: Always create backups when migrating configurations
5. **Use Descriptive Paths**: Use clear, hierarchical configuration paths
6. **Handle Errors Gracefully**: Check validation results and handle errors appropriately

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed (`pip install watchdog pyyaml jsonschema`)
2. **File Permissions**: Ensure configuration files are readable/writable
3. **Schema Validation**: Check that configuration values match expected types
4. **Port Conflicts**: Ensure API and frontend use different ports
5. **Environment Overrides**: Verify environment-specific files exist and are valid

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use CLI with verbose flag
python -m tools.config_manager.config_cli --verbose get api.port
```
