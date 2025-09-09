# WAN Project Structure

This document describes the organized structure of the WAN (Video Generation) project after the comprehensive cleanup performed on 2025-01-06.

## Root Directory Structure

```
wan/
├── README.md                    # Main project documentation
├── main.py                      # Main application entry point
├── start.py                     # Application startup script
├── setup.py                     # Python package setup
├── config.json                  # Main configuration file
├── performance-config.json      # Performance configuration
├── pytest.ini                   # Pytest configuration
├── requirements*.txt            # Python dependencies
├── nginx.conf                   # Nginx configuration
├── wan-cli                      # CLI executable
├── wan22_tasks.db              # Task database
├── *.log                       # Log files
├── *.pid                       # Process ID files
├── start_*.py|.sh|.bat         # Various startup scripts
└── stop_*.sh                   # Stop scripts
```

## Directory Structure

### `/backend/`

Backend API and services

- FastAPI application
- API endpoints
- Business logic
- Database models

### `/frontend/`

Frontend user interface

- Web interface components
- Static assets
- UI configuration

### `/cli/`

Command-line interface

- CLI commands and utilities
- Command parsing and execution

### `/core/`

Core application logic

- Core algorithms
- Model management
- Pipeline processing

### `/models/`

Model definitions and configurations

- Model architectures
- Configuration files
- Model utilities

### `/config/`

Configuration files and templates

- Environment configurations
- Application settings
- Template configurations

### `/docs/`

Comprehensive documentation

- User guides
- Developer documentation
- API documentation
- Deployment guides
- Training materials

#### `/docs/archive/`

Historical documentation (moved from root)

- Task completion summaries
- Legacy implementation reports
- Historical progress tracking

### `/tests/`

Test suite organization

- Unit tests
- Integration tests
- End-to-end tests
- Performance tests
- Test utilities and fixtures

#### `/tests/archive/`

Legacy test files (moved from root)

- Historical test implementations
- Validation scripts
- Legacy test utilities

### `/scripts/`

Utility and automation scripts

- CI/CD scripts
- Deployment scripts
- Maintenance utilities

#### `/scripts/utils/`

Code maintenance utilities (moved from root)

- Syntax fixing scripts
- Import fixing utilities
- Code quality tools

### `/reports/`

Generated reports and metrics

- Health monitoring reports
- Validation results
- Test results and coverage
- Quality metrics

#### `/reports/health/`

Health monitoring and system status

- Health check results
- System dashboards
- Performance metrics

#### `/reports/validation/`

Validation and verification reports

- Model validation results
- System validation metrics

#### `/reports/tests/`

Test execution results

- Test reports
- Test execution logs

#### `/reports/coverage/`

Code coverage reports

### `/demo_examples/`

Example scripts and demonstrations (moved from root)

- Deployment examples
- Quick test scripts

### `/infrastructure/`

Infrastructure as code

- Docker configurations
- Deployment manifests
- Infrastructure scripts

### `/data/`

Data files and datasets

- Training data
- Test datasets
- Configuration data

### `/logs/`

Application logs

- Runtime logs
- Error logs
- Debug information

### `/temp/`

Temporary files

- Build artifacts
- Temporary processing files

## Key Files

### Configuration

- `config.json` - Main application configuration
- `performance-config.json` - Performance tuning settings
- `pytest.ini` - Test configuration
- `requirements*.txt` - Python dependencies

### Entry Points

- `main.py` - Main application entry point
- `start.py` - Application startup with configuration
- `wan-cli` - Command-line interface

### Documentation

- `README.md` - Project overview and quick start
- `docs/` - Comprehensive documentation
- `docs/PROJECT_STRUCTURE.md` - This file

## Migration Notes

This structure was established during a comprehensive cleanup on 2025-01-06:

1. **Documentation Consolidation**: All scattered `.md` files from root moved to `docs/archive/`
2. **Test Organization**: All `test_*.py` files from root moved to `tests/archive/`
3. **Utility Scripts**: All `fix_*.py` and utility scripts moved to `scripts/utils/`
4. **Reports Organization**: All report files organized into `reports/` with subdirectories
5. **Demo Code**: Demo and example scripts moved to `demo_examples/`

All original file locations are preserved in git history for reference.

## Best Practices

1. **New Documentation**: Add to appropriate `docs/` subdirectory
2. **New Tests**: Add to appropriate `tests/` subdirectory
3. **New Scripts**: Add to `scripts/` with appropriate subdirectory
4. **Reports**: Generated reports should go to `reports/` subdirectories
5. **Configuration**: Use `config/` for new configuration files

This structure supports scalable development and maintains clear separation of concerns.
