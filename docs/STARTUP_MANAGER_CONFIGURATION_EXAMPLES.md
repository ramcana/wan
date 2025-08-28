# WAN22 Startup Manager Configuration Examples

## Table of Contents

1. [Basic Configurations](#basic-configurations)
2. [Development Environments](#development-environments)
3. [Team Configurations](#team-configurations)
4. [Production-like Environments](#production-like-environments)
5. [CI/CD Configurations](#cicd-configurations)
6. [Troubleshooting Configurations](#troubleshooting-configurations)
7. [Performance Optimization](#performance-optimization)
8. [Platform-Specific Configurations](#platform-specific-configurations)

## Basic Configurations

### Default Configuration

```json
{
  "backend": {
    "host": "localhost",
    "port": 8000,
    "auto_port": true,
    "timeout": 30,
    "reload": true,
    "log_level": "info",
    "workers": 1
  },
  "frontend": {
    "host": "localhost",
    "port": 3000,
    "auto_port": true,
    "timeout": 30,
    "open_browser": true,
    "hot_reload": true
  },
  "retry_attempts": 3,
  "retry_delay": 2.0,
  "verbose_logging": false,
  "auto_fix_issues": true,
  "performance_monitoring": {
    "enabled": true,
    "collect_metrics": true,
    "analytics": true
  }
}
```

### Minimal Configuration

```json
{
  "backend": {
    "port": 8000
  },
  "frontend": {
    "port": 3000
  }
}
```

### Verbose Configuration

```json
{
  "backend": {
    "port": 8000,
    "auto_port": true,
    "timeout": 60
  },
  "frontend": {
    "port": 3000,
    "auto_port": true,
    "timeout": 60
  },
  "verbose_logging": true,
  "auto_fix_issues": true,
  "performance_monitoring": {
    "enabled": true,
    "collect_metrics": true,
    "analytics": true,
    "detailed_logging": true
  }
}
```

## Development Environments

### Local Development

```json
{
  "backend": {
    "host": "localhost",
    "port": 8000,
    "auto_port": true,
    "timeout": 30,
    "reload": true,
    "log_level": "debug",
    "workers": 1
  },
  "frontend": {
    "host": "localhost",
    "port": 3000,
    "auto_port": true,
    "timeout": 30,
    "open_browser": true,
    "hot_reload": true
  },
  "retry_attempts": 5,
  "retry_delay": 1.0,
  "verbose_logging": true,
  "auto_fix_issues": true,
  "windows_optimizations": {
    "firewall_exceptions": true,
    "process_priority": "normal"
  },
  "performance_monitoring": {
    "enabled": true,
    "collect_metrics": true,
    "analytics": true
  }
}
```

### Docker Development

```json
{
  "backend": {
    "host": "0.0.0.0",
    "port": 8000,
    "auto_port": false,
    "timeout": 60,
    "reload": true,
    "log_level": "info",
    "workers": 1
  },
  "frontend": {
    "host": "0.0.0.0",
    "port": 3000,
    "auto_port": false,
    "timeout": 60,
    "open_browser": false,
    "hot_reload": true
  },
  "retry_attempts": 3,
  "retry_delay": 5.0,
  "verbose_logging": false,
  "auto_fix_issues": false,
  "windows_optimizations": {
    "enabled": false
  },
  "performance_monitoring": {
    "enabled": false
  }
}
```

### Remote Development (WSL/SSH)

```json
{
  "backend": {
    "host": "0.0.0.0",
    "port": 8000,
    "auto_port": true,
    "timeout": 45,
    "reload": true,
    "log_level": "info",
    "workers": 1
  },
  "frontend": {
    "host": "0.0.0.0",
    "port": 3000,
    "auto_port": true,
    "timeout": 45,
    "open_browser": false,
    "hot_reload": true
  },
  "retry_attempts": 5,
  "retry_delay": 3.0,
  "verbose_logging": true,
  "auto_fix_issues": true,
  "network_configuration": {
    "bind_all_interfaces": true,
    "allow_external_connections": true
  }
}
```

## Team Configurations

### Small Team (2-5 developers)

```json
{
  "backend": {
    "port": 8000,
    "auto_port": true,
    "timeout": 30,
    "reload": true,
    "log_level": "info"
  },
  "frontend": {
    "port": 3000,
    "auto_port": true,
    "timeout": 30,
    "hot_reload": true
  },
  "retry_attempts": 3,
  "verbose_logging": false,
  "auto_fix_issues": true,
  "team_settings": {
    "shared_port_ranges": {
      "backend": [8000, 8010],
      "frontend": [3000, 3010]
    },
    "conflict_resolution": "automatic"
  },
  "performance_monitoring": {
    "enabled": true,
    "team_analytics": true
  }
}
```

### Large Team (10+ developers)

```json
{
  "backend": {
    "port": 8000,
    "auto_port": true,
    "timeout": 45,
    "reload": true,
    "log_level": "warning"
  },
  "frontend": {
    "port": 3000,
    "auto_port": true,
    "timeout": 45,
    "hot_reload": true
  },
  "retry_attempts": 5,
  "retry_delay": 2.0,
  "verbose_logging": false,
  "auto_fix_issues": true,
  "team_settings": {
    "shared_port_ranges": {
      "backend": [8000, 8050],
      "frontend": [3000, 3050]
    },
    "developer_isolation": true,
    "port_assignment_strategy": "hash_based"
  },
  "performance_monitoring": {
    "enabled": true,
    "team_analytics": true,
    "aggregated_metrics": true
  }
}
```

### Distributed Team (Multiple Time Zones)

```json
{
  "backend": {
    "port": 8000,
    "auto_port": true,
    "timeout": 60,
    "reload": true,
    "log_level": "info"
  },
  "frontend": {
    "port": 3000,
    "auto_port": true,
    "timeout": 60,
    "hot_reload": true
  },
  "retry_attempts": 5,
  "retry_delay": 3.0,
  "verbose_logging": true,
  "auto_fix_issues": true,
  "team_settings": {
    "timezone_aware": true,
    "shared_resources": false,
    "independent_environments": true
  },
  "logging": {
    "include_timezone": true,
    "detailed_environment_info": true
  }
}
```

## Production-like Environments

### Staging Environment

```json
{
  "backend": {
    "host": "0.0.0.0",
    "port": 8000,
    "auto_port": false,
    "timeout": 60,
    "reload": false,
    "log_level": "warning",
    "workers": 2
  },
  "frontend": {
    "host": "0.0.0.0",
    "port": 3000,
    "auto_port": false,
    "timeout": 60,
    "open_browser": false,
    "hot_reload": false
  },
  "retry_attempts": 3,
  "retry_delay": 5.0,
  "verbose_logging": false,
  "auto_fix_issues": false,
  "production_mode": {
    "enabled": true,
    "security_checks": true,
    "performance_optimization": true
  },
  "monitoring": {
    "health_checks": true,
    "metrics_collection": true,
    "alerting": true
  }
}
```

### Performance Testing

```json
{
  "backend": {
    "host": "0.0.0.0",
    "port": 8000,
    "auto_port": false,
    "timeout": 120,
    "reload": false,
    "log_level": "error",
    "workers": 4
  },
  "frontend": {
    "host": "0.0.0.0",
    "port": 3000,
    "auto_port": false,
    "timeout": 120,
    "open_browser": false,
    "hot_reload": false
  },
  "retry_attempts": 1,
  "retry_delay": 10.0,
  "verbose_logging": false,
  "auto_fix_issues": false,
  "performance_testing": {
    "enabled": true,
    "resource_monitoring": true,
    "benchmark_mode": true,
    "load_testing_ready": true
  }
}
```

### Load Testing Environment

```json
{
  "backend": {
    "host": "0.0.0.0",
    "port": 8000,
    "auto_port": false,
    "timeout": 180,
    "reload": false,
    "log_level": "critical",
    "workers": 8
  },
  "frontend": {
    "host": "0.0.0.0",
    "port": 3000,
    "auto_port": false,
    "timeout": 180,
    "open_browser": false,
    "hot_reload": false
  },
  "retry_attempts": 1,
  "retry_delay": 30.0,
  "verbose_logging": false,
  "auto_fix_issues": false,
  "load_testing": {
    "enabled": true,
    "max_connections": 1000,
    "resource_limits": {
      "cpu_percent": 90,
      "memory_percent": 85
    }
  }
}
```

## CI/CD Configurations

### GitHub Actions

```json
{
  "backend": {
    "host": "localhost",
    "port": 8000,
    "auto_port": true,
    "timeout": 120,
    "reload": false,
    "log_level": "info",
    "workers": 1
  },
  "frontend": {
    "host": "localhost",
    "port": 3000,
    "auto_port": true,
    "timeout": 120,
    "open_browser": false,
    "hot_reload": false
  },
  "retry_attempts": 3,
  "retry_delay": 10.0,
  "verbose_logging": true,
  "auto_fix_issues": true,
  "ci_mode": {
    "enabled": true,
    "headless": true,
    "no_interactive": true,
    "exit_on_failure": true
  },
  "testing": {
    "wait_for_ready": true,
    "health_check_retries": 10,
    "test_mode": true
  }
}
```

### Jenkins

```json
{
  "backend": {
    "host": "0.0.0.0",
    "port": 8000,
    "auto_port": true,
    "timeout": 180,
    "reload": false,
    "log_level": "info",
    "workers": 2
  },
  "frontend": {
    "host": "0.0.0.0",
    "port": 3000,
    "auto_port": true,
    "timeout": 180,
    "open_browser": false,
    "hot_reload": false
  },
  "retry_attempts": 5,
  "retry_delay": 15.0,
  "verbose_logging": true,
  "auto_fix_issues": true,
  "jenkins": {
    "workspace_isolation": true,
    "parallel_builds": true,
    "artifact_collection": true
  }
}
```

### Docker CI

```json
{
  "backend": {
    "host": "0.0.0.0",
    "port": 8000,
    "auto_port": false,
    "timeout": 300,
    "reload": false,
    "log_level": "info",
    "workers": 1
  },
  "frontend": {
    "host": "0.0.0.0",
    "port": 3000,
    "auto_port": false,
    "timeout": 300,
    "open_browser": false,
    "hot_reload": false
  },
  "retry_attempts": 3,
  "retry_delay": 20.0,
  "verbose_logging": true,
  "auto_fix_issues": false,
  "docker_ci": {
    "container_mode": true,
    "network_mode": "bridge",
    "resource_constraints": true
  }
}
```

## Troubleshooting Configurations

### Debug Mode

```json
{
  "backend": {
    "port": 8000,
    "auto_port": true,
    "timeout": 300,
    "reload": true,
    "log_level": "debug",
    "workers": 1
  },
  "frontend": {
    "port": 3000,
    "auto_port": true,
    "timeout": 300,
    "hot_reload": true
  },
  "retry_attempts": 10,
  "retry_delay": 1.0,
  "verbose_logging": true,
  "auto_fix_issues": true,
  "debug_mode": {
    "enabled": true,
    "trace_calls": true,
    "detailed_errors": true,
    "step_by_step": true
  },
  "diagnostics": {
    "auto_collect": true,
    "save_reports": true,
    "include_system_info": true
  }
}
```

### Network Issues

```json
{
  "backend": {
    "host": "127.0.0.1",
    "port": 8080,
    "auto_port": true,
    "timeout": 60,
    "reload": true,
    "log_level": "debug"
  },
  "frontend": {
    "host": "127.0.0.1",
    "port": 3001,
    "auto_port": true,
    "timeout": 60,
    "hot_reload": true
  },
  "retry_attempts": 5,
  "retry_delay": 5.0,
  "verbose_logging": true,
  "auto_fix_issues": true,
  "network_troubleshooting": {
    "alternative_hosts": ["localhost", "0.0.0.0"],
    "port_ranges": {
      "backend": [8080, 8090],
      "frontend": [3001, 3010]
    },
    "firewall_detection": true
  }
}
```

### Permission Issues

```json
{
  "backend": {
    "port": 8080,
    "auto_port": true,
    "timeout": 45,
    "reload": true,
    "log_level": "info"
  },
  "frontend": {
    "port": 3001,
    "auto_port": true,
    "timeout": 45,
    "hot_reload": true
  },
  "retry_attempts": 3,
  "retry_delay": 3.0,
  "verbose_logging": true,
  "auto_fix_issues": true,
  "permission_handling": {
    "safe_port_ranges": [
      [8080, 8090],
      [3001, 3010],
      [9000, 9010]
    ],
    "avoid_privileged_ports": true,
    "user_space_only": true
  },
  "windows_optimizations": {
    "firewall_exceptions": false,
    "admin_elevation": false
  }
}
```

## Performance Optimization

### Fast Startup

```json
{
  "backend": {
    "port": 8000,
    "auto_port": true,
    "timeout": 15,
    "reload": true,
    "log_level": "warning",
    "workers": 1
  },
  "frontend": {
    "port": 3000,
    "auto_port": true,
    "timeout": 15,
    "hot_reload": true
  },
  "retry_attempts": 2,
  "retry_delay": 1.0,
  "verbose_logging": false,
  "auto_fix_issues": true,
  "performance_optimization": {
    "fast_mode": true,
    "skip_non_critical_checks": true,
    "parallel_startup": true,
    "cache_validations": true
  },
  "validation": {
    "minimal_checks": true,
    "cache_results": true
  }
}
```

### Resource Constrained

```json
{
  "backend": {
    "port": 8000,
    "auto_port": true,
    "timeout": 60,
    "reload": true,
    "log_level": "error",
    "workers": 1
  },
  "frontend": {
    "port": 3000,
    "auto_port": true,
    "timeout": 60,
    "hot_reload": false
  },
  "retry_attempts": 2,
  "retry_delay": 2.0,
  "verbose_logging": false,
  "auto_fix_issues": false,
  "resource_constraints": {
    "low_memory_mode": true,
    "minimal_logging": true,
    "reduced_monitoring": true
  },
  "performance_monitoring": {
    "enabled": false
  }
}
```

### High Performance

```json
{
  "backend": {
    "port": 8000,
    "auto_port": true,
    "timeout": 30,
    "reload": false,
    "log_level": "warning",
    "workers": 4
  },
  "frontend": {
    "port": 3000,
    "auto_port": true,
    "timeout": 30,
    "hot_reload": false
  },
  "retry_attempts": 3,
  "retry_delay": 1.0,
  "verbose_logging": false,
  "auto_fix_issues": true,
  "high_performance": {
    "process_priority": "high",
    "cpu_affinity": true,
    "memory_optimization": true,
    "io_optimization": true
  },
  "windows_optimizations": {
    "process_priority": "above_normal",
    "ssd_optimization": true
  }
}
```

## Platform-Specific Configurations

### Windows Development

```json
{
  "backend": {
    "port": 8000,
    "auto_port": true,
    "timeout": 30,
    "reload": true,
    "log_level": "info"
  },
  "frontend": {
    "port": 3000,
    "auto_port": true,
    "timeout": 30,
    "hot_reload": true
  },
  "retry_attempts": 3,
  "retry_delay": 2.0,
  "verbose_logging": false,
  "auto_fix_issues": true,
  "windows_optimizations": {
    "firewall_exceptions": true,
    "process_priority": "normal",
    "service_integration": false,
    "defender_exclusions": true,
    "path_length_handling": true
  },
  "windows_specific": {
    "handle_long_paths": true,
    "use_short_names": false,
    "admin_detection": true
  }
}
```

### WSL (Windows Subsystem for Linux)

```json
{
  "backend": {
    "host": "0.0.0.0",
    "port": 8000,
    "auto_port": true,
    "timeout": 45,
    "reload": true,
    "log_level": "info"
  },
  "frontend": {
    "host": "0.0.0.0",
    "port": 3000,
    "auto_port": true,
    "timeout": 45,
    "hot_reload": true
  },
  "retry_attempts": 5,
  "retry_delay": 3.0,
  "verbose_logging": true,
  "auto_fix_issues": true,
  "wsl_configuration": {
    "cross_platform_paths": true,
    "windows_integration": true,
    "network_forwarding": true
  },
  "windows_optimizations": {
    "enabled": false
  }
}
```

### macOS Development

```json
{
  "backend": {
    "port": 8000,
    "auto_port": true,
    "timeout": 30,
    "reload": true,
    "log_level": "info"
  },
  "frontend": {
    "port": 3000,
    "auto_port": true,
    "timeout": 30,
    "hot_reload": true
  },
  "retry_attempts": 3,
  "retry_delay": 2.0,
  "verbose_logging": false,
  "auto_fix_issues": true,
  "macos_optimizations": {
    "use_fsevents": true,
    "security_permissions": true,
    "spotlight_exclusions": true
  },
  "windows_optimizations": {
    "enabled": false
  }
}
```

## Environment Variable Examples

### Development Environment Variables

```bash
# Basic configuration
set WAN22_BACKEND_PORT=8000
set WAN22_FRONTEND_PORT=3000
set WAN22_VERBOSE_LOGGING=true

# Advanced configuration
set WAN22_AUTO_FIX_ISSUES=true
set WAN22_PERFORMANCE_MONITORING=true
set WAN22_RETRY_ATTEMPTS=5

# Debug configuration
set WAN22_LOG_LEVEL=debug
set WAN22_DEBUG_MODE=true
set WAN22_TRACE_CALLS=true
```

### CI/CD Environment Variables

```bash
# CI mode
set WAN22_CI_MODE=true
set WAN22_HEADLESS=true
set WAN22_NO_INTERACTIVE=true
set WAN22_EXIT_ON_FAILURE=true

# Testing configuration
set WAN22_TEST_MODE=true
set WAN22_WAIT_FOR_READY=true
set WAN22_HEALTH_CHECK_RETRIES=10
```

### Production Environment Variables

```bash
# Production mode
set WAN22_PRODUCTION_MODE=true
set WAN22_SECURITY_CHECKS=true
set WAN22_PERFORMANCE_OPTIMIZATION=true

# Monitoring
set WAN22_HEALTH_CHECKS=true
set WAN22_METRICS_COLLECTION=true
set WAN22_ALERTING=true
```

## Usage Examples

### Using Configuration Files

```bash
# Use specific configuration file
start_both_servers.bat --config custom_config.json

# Use environment-specific configuration
start_both_servers.bat --config configs/development.json
start_both_servers.bat --config configs/staging.json
start_both_servers.bat --config configs/production.json
```

### Combining Configuration Methods

```bash
# Configuration file + environment variables
set WAN22_VERBOSE_LOGGING=true
start_both_servers.bat --config team_config.json

# Configuration file + command line arguments
start_both_servers.bat --config base_config.json --backend-port 8080 --verbose
```

### Dynamic Configuration

```bash
# Generate configuration based on environment
python scripts/generate_config.py --environment development > startup_config.json
start_both_servers.bat

# Use configuration templates
python scripts/apply_config_template.py --template team --output startup_config.json
start_both_servers.bat
```

These configuration examples provide a comprehensive starting point for different development scenarios. Choose the configuration that best matches your environment and customize as needed.
