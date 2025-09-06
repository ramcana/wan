# Local Testing Framework - Examples

This directory contains comprehensive examples and templates for using the Local Testing Framework with the Wan2.2 UI Variant. The examples are organized by category and provide practical implementations for various use cases.

## Directory Structure

```
examples/
├── configurations/          # Configuration file examples
├── environment_templates/   # Environment variable templates
├── workflows/              # Automated workflow scripts
├── deployment/             # Deployment configurations
├── automation/             # CI/CD pipeline examples
└── README.md              # This file
```

## Configuration Examples

### Basic Configuration (`configurations/basic_config.json`)

A minimal configuration suitable for development and testing:

- Standard performance targets (720p < 9min, 1080p < 17min)
- Basic optimizations enabled
- Local development settings

**Usage:**

```bash
cp local_testing_framework/examples/configurations/basic_config.json config.json
python -m local_testing_framework validate-env
```

### High Performance Configuration (`configurations/high_performance_config.json`)

Optimized for high-end hardware with multiple GPUs:

- Aggressive performance targets (720p < 7min, 1080p < 14min)
- Advanced optimizations (xformers, torch compile)
- Multi-GPU support

**Usage:**

```bash
cp local_testing_framework/examples/configurations/high_performance_config.json config.json
python -m local_testing_framework test-performance --benchmark
```

### Low Memory Configuration (`configurations/low_memory_config.json`)

Optimized for systems with limited VRAM and memory:

- Conservative performance targets
- Maximum memory optimizations enabled
- CPU offload and attention slicing

**Usage:**

```bash
cp local_testing_framework/examples/configurations/low_memory_config.json config.json
python -m local_testing_framework test-performance --vram-test
```

### Production Configuration (`configurations/production_config.json`)

Production-ready configuration with security and monitoring:

- Production-grade performance targets
- Security features enabled
- Comprehensive monitoring and alerting

**Usage:**

```bash
cp local_testing_framework/examples/configurations/production_config.json config.json
python -m local_testing_framework run-all --report-format json
```

## Environment Templates

### Basic Environment (`environment_templates/basic.env`)

Essential environment variables for local development:

```bash
cp local_testing_framework/examples/environment_templates/basic.env .env
# Edit .env with your actual values
```

### Development Environment (`environment_templates/development.env`)

Extended environment for development workflows:

- Debug logging enabled
- Development-specific optimizations
- Testing framework integration

### Production Environment (`environment_templates/production.env`)

Production environment with security and monitoring:

- Production logging configuration
- Security settings
- Monitoring and alerting endpoints

## Workflow Examples

### Daily Development Workflow (`workflows/daily_development_workflow.py`)

Automated script for daily development tasks:

```bash
# Quick development check
python local_testing_framework/examples/workflows/daily_development_workflow.py --quick

# Full development validation with reports
python local_testing_framework/examples/workflows/daily_development_workflow.py --report --fix
```

**Features:**

- Environment validation with auto-fix
- Performance testing
- Integration testing
- Sample data generation
- System diagnostics

### Pre-Deployment Validation (`workflows/pre_deployment_validation.py`)

Comprehensive validation before deployment:

```bash
# Staging validation
python local_testing_framework/examples/workflows/pre_deployment_validation.py --environment staging

# Production validation (strict mode)
python local_testing_framework/examples/workflows/pre_deployment_validation.py --environment production --strict
```

**Features:**

- Multi-phase validation
- Environment-specific configuration
- Security validation
- Production readiness checks
- Deployment reports

### Continuous Monitoring (`workflows/continuous_monitoring_workflow.py`)

Long-running monitoring for development and production:

```bash
# Development monitoring (1 hour)
python local_testing_framework/examples/workflows/continuous_monitoring_workflow.py --duration 3600 --profile dev

# Production monitoring with alerts (24 hours)
python local_testing_framework/examples/workflows/continuous_monitoring_workflow.py --duration 86400 --alerts --profile prod
```

**Features:**

- Real-time resource monitoring
- Configurable alert thresholds
- Automatic cleanup procedures
- Periodic reporting
- Performance trend analysis

## Deployment Examples

### Docker Deployment

Complete Docker setup with multi-service architecture:

```bash
# Build and deploy with Docker Compose
cd local_testing_framework/examples/deployment/docker
docker-compose up -d

# Or use the deployment script
bash local_testing_framework/examples/deployment/scripts/deploy.sh docker staging
```

**Includes:**

- Multi-stage Dockerfile
- Docker Compose with Redis, Prometheus, Grafana
- GPU support configuration
- Health checks and monitoring

### Kubernetes Deployment

Production-ready Kubernetes manifests:

```bash
# Deploy to Kubernetes
kubectl apply -f local_testing_framework/examples/deployment/kubernetes/deployment.yaml

# Or use the deployment script
bash local_testing_framework/examples/deployment/scripts/deploy.sh kubernetes production
```

**Features:**

- Horizontal pod autoscaling
- GPU resource allocation
- Persistent volume claims
- Ingress configuration
- Health and readiness probes

### Deployment Script (`deployment/scripts/deploy.sh`)

Universal deployment script supporting multiple platforms:

```bash
# Docker deployment
./local_testing_framework/examples/deployment/scripts/deploy.sh docker staging

# Kubernetes deployment
./local_testing_framework/examples/deployment/scripts/deploy.sh kubernetes production

# Systemd service deployment
./local_testing_framework/examples/deployment/scripts/deploy.sh systemd production
```

**Features:**

- Multi-platform support (Docker, Kubernetes, systemd)
- Pre and post-deployment validation
- Environment-specific configuration
- Comprehensive error handling

## Automation Examples

### CI/CD Pipeline (`automation/ci_cd_pipeline.yml`)

Complete GitHub Actions workflow:

```yaml
# Copy to .github/workflows/ci-cd.yml
cp local_testing_framework/examples/automation/ci_cd_pipeline.yml .github/workflows/
```

**Pipeline Stages:**

1. Environment validation
2. Unit and integration testing
3. Security and code quality checks
4. Docker image building
5. Staging deployment
6. Production deployment
7. Continuous monitoring setup
8. Nightly comprehensive tests

## Usage Patterns

### Development Workflow

1. **Setup Environment:**

   ```bash
   cp local_testing_framework/examples/configurations/basic_config.json config.json
   cp local_testing_framework/examples/environment_templates/development.env .env
   # Edit .env with your values
   ```

2. **Daily Development:**

   ```bash
   python local_testing_framework/examples/workflows/daily_development_workflow.py --quick
   ```

3. **Pre-commit Testing:**
   ```bash
   python -m local_testing_framework test-performance --resolution 720p
   python -m local_testing_framework test-integration --ui
   ```

### Staging Deployment

1. **Pre-deployment Validation:**

   ```bash
   python local_testing_framework/examples/workflows/pre_deployment_validation.py --environment staging
   ```

2. **Deploy:**

   ```bash
   bash local_testing_framework/examples/deployment/scripts/deploy.sh docker staging
   ```

3. **Monitor:**
   ```bash
   python local_testing_framework/examples/workflows/continuous_monitoring_workflow.py --duration 3600 --alerts
   ```

### Production Deployment

1. **Comprehensive Validation:**

   ```bash
   python local_testing_framework/examples/workflows/pre_deployment_validation.py --environment production --strict
   ```

2. **Deploy:**

   ```bash
   bash local_testing_framework/examples/deployment/scripts/deploy.sh kubernetes production
   ```

3. **Continuous Monitoring:**
   ```bash
   python local_testing_framework/examples/workflows/continuous_monitoring_workflow.py --duration 86400 --alerts --profile prod
   ```

## Customization

### Creating Custom Configurations

1. **Copy Base Configuration:**

   ```bash
   cp local_testing_framework/examples/configurations/basic_config.json my_config.json
   ```

2. **Modify Settings:**

   - Adjust performance targets
   - Enable/disable optimizations
   - Configure directories and paths

3. **Validate Configuration:**
   ```bash
   python -m local_testing_framework validate-env --config my_config.json
   ```

### Creating Custom Workflows

1. **Copy Base Workflow:**

   ```bash
   cp local_testing_framework/examples/workflows/daily_development_workflow.py my_workflow.py
   ```

2. **Customize Steps:**

   - Add custom validation steps
   - Modify test parameters
   - Add custom reporting

3. **Test Workflow:**
   ```bash
   python my_workflow.py --quick
   ```

### Creating Custom Deployment

1. **Copy Deployment Files:**

   ```bash
   cp -r local_testing_framework/examples/deployment/docker my_deployment/
   ```

2. **Modify Configuration:**

   - Update Dockerfile
   - Modify docker-compose.yml
   - Adjust environment variables

3. **Test Deployment:**
   ```bash
   cd my_deployment
   docker-compose up -d
   ```

## Best Practices

### Configuration Management

- Use environment-specific configurations
- Keep sensitive data in environment variables
- Validate configurations before deployment
- Version control configuration templates

### Testing Workflows

- Run quick tests frequently during development
- Use comprehensive tests before deployment
- Monitor performance trends over time
- Automate testing in CI/CD pipelines

### Deployment Practices

- Always run pre-deployment validation
- Use staged deployments (dev → staging → production)
- Monitor applications after deployment
- Have rollback procedures ready

### Monitoring and Alerting

- Set appropriate alert thresholds
- Monitor both system and application metrics
- Use automated recovery procedures
- Generate regular reports

## Troubleshooting

### Common Issues

1. **Configuration Errors:**

   ```bash
   python -m local_testing_framework validate-env --fix
   ```

2. **Performance Issues:**

   ```bash
   python -m local_testing_framework diagnose --system --cuda --memory
   ```

3. **Deployment Failures:**
   ```bash
   python local_testing_framework/examples/workflows/pre_deployment_validation.py --environment staging
   ```

### Getting Help

- Check the troubleshooting guide: `docs/TROUBLESHOOTING.md`
- Run diagnostics: `python -m local_testing_framework diagnose`
- Review logs: `tail -f wan22_errors.log`
- Generate reports: `python -m local_testing_framework run-all --report-format html`

## Contributing

To contribute new examples:

1. Follow the existing directory structure
2. Include comprehensive documentation
3. Add usage examples and test cases
4. Update this README with new examples

For more information, see the developer guide: `docs/DEVELOPER_GUIDE.md`
