#!/bin/bash

# Wan2.2 UI Variant Deployment Script
# This script automates the deployment process with comprehensive validation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DEPLOYMENT_TYPE="${1:-docker}"
ENVIRONMENT="${2:-staging}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking deployment prerequisites..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check if local testing framework is available
    if ! python3 -m local_testing_framework --help &> /dev/null; then
        log_error "Local Testing Framework is not installed"
        exit 1
    fi
    
    # Check deployment type specific prerequisites
    case $DEPLOYMENT_TYPE in
        docker)
            if ! command -v docker &> /dev/null; then
                log_error "Docker is required but not installed"
                exit 1
            fi
            if ! command -v docker-compose &> /dev/null; then
                log_error "Docker Compose is required but not installed"
                exit 1
            fi
            ;;
        kubernetes)
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl is required but not installed"
                exit 1
            fi
            ;;
        systemd)
            if ! command -v systemctl &> /dev/null; then
                log_error "systemd is required but not available"
                exit 1
            fi
            ;;
    esac
    
    log_success "Prerequisites check passed"
}

# Pre-deployment validation
run_pre_deployment_validation() {
    log_info "Running pre-deployment validation..."
    
    cd "$PROJECT_ROOT"
    
    # Run comprehensive validation
    if python3 local_testing_framework/examples/workflows/pre_deployment_validation.py \
        --environment "$ENVIRONMENT" --strict; then
        log_success "Pre-deployment validation passed"
    else
        log_error "Pre-deployment validation failed"
        log_info "Check the validation report for details"
        exit 1
    fi
}

# Docker deployment
deploy_docker() {
    log_info "Deploying with Docker..."
    
    cd "$PROJECT_ROOT"
    
    # Copy configuration
    local config_file="local_testing_framework/examples/configurations/${ENVIRONMENT}_config.json"
    if [[ -f "$config_file" ]]; then
        cp "$config_file" config.json
        log_success "Configuration copied for $ENVIRONMENT"
    else
        log_warning "No specific configuration for $ENVIRONMENT, using default"
    fi
    
    # Copy environment template
    local env_file="local_testing_framework/examples/environment_templates/${ENVIRONMENT}.env"
    if [[ -f "$env_file" ]]; then
        cp "$env_file" .env.template
        log_info "Please update .env.template with your actual values and rename to .env"
    fi
    
    # Build and start containers
    log_info "Building Docker image..."
    docker-compose -f local_testing_framework/examples/deployment/docker/docker-compose.yml build
    
    log_info "Starting containers..."
    docker-compose -f local_testing_framework/examples/deployment/docker/docker-compose.yml up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Verify deployment
    if docker-compose -f local_testing_framework/examples/deployment/docker/docker-compose.yml ps | grep -q "Up"; then
        log_success "Docker deployment completed successfully"
        log_info "Application should be available at http://localhost:8080"
    else
        log_error "Docker deployment failed"
        exit 1
    fi
}

# Kubernetes deployment
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace
    kubectl create namespace wan22 --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply configurations
    kubectl apply -f local_testing_framework/examples/deployment/kubernetes/deployment.yaml
    
    # Wait for deployment
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/wan22-ui-variant -n wan22
    
    # Get service information
    kubectl get services -n wan22
    
    log_success "Kubernetes deployment completed successfully"
}

# Systemd deployment
deploy_systemd() {
    log_info "Deploying with systemd..."
    
    # Create systemd service file
    cat > /tmp/wan22.service << EOF
[Unit]
Description=Wan2.2 UI Variant
After=network.target

[Service]
Type=simple
User=wan22
WorkingDirectory=$PROJECT_ROOT
Environment=PYTHONPATH=$PROJECT_ROOT
ExecStart=/usr/bin/python3 main.py --config config.json
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Install service
    sudo cp /tmp/wan22.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable wan22.service
    sudo systemctl start wan22.service
    
    # Check status
    if sudo systemctl is-active --quiet wan22.service; then
        log_success "Systemd deployment completed successfully"
    else
        log_error "Systemd deployment failed"
        sudo systemctl status wan22.service
        exit 1
    fi
}

# Post-deployment validation
run_post_deployment_validation() {
    log_info "Running post-deployment validation..."
    
    # Wait for application to be fully ready
    sleep 60
    
    # Run validation tests
    cd "$PROJECT_ROOT"
    
    # Test environment
    if python3 -m local_testing_framework validate-env; then
        log_success "Environment validation passed"
    else
        log_warning "Environment validation failed"
    fi
    
    # Test basic functionality
    if python3 -m local_testing_framework test-integration --quick; then
        log_success "Basic functionality test passed"
    else
        log_warning "Basic functionality test failed"
    fi
    
    # Generate deployment report
    python3 -m local_testing_framework run-all --report-format json --output reports/deployment
    
    log_success "Post-deployment validation completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f /tmp/wan22.service
}

# Main deployment function
main() {
    log_info "Starting Wan2.2 UI Variant deployment"
    log_info "Deployment type: $DEPLOYMENT_TYPE"
    log_info "Environment: $ENVIRONMENT"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    run_pre_deployment_validation
    
    case $DEPLOYMENT_TYPE in
        docker)
            deploy_docker
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
        systemd)
            deploy_systemd
            ;;
        *)
            log_error "Unsupported deployment type: $DEPLOYMENT_TYPE"
            log_info "Supported types: docker, kubernetes, systemd"
            exit 1
            ;;
    esac
    
    run_post_deployment_validation
    
    log_success "Deployment completed successfully!"
    log_info "Check the application logs and monitoring for ongoing status"
}

# Show usage
show_usage() {
    echo "Usage: $0 [deployment_type] [environment]"
    echo ""
    echo "Deployment types:"
    echo "  docker      - Deploy using Docker Compose (default)"
    echo "  kubernetes  - Deploy to Kubernetes cluster"
    echo "  systemd     - Deploy as systemd service"
    echo ""
    echo "Environments:"
    echo "  staging     - Staging environment (default)"
    echo "  production  - Production environment"
    echo ""
    echo "Examples:"
    echo "  $0 docker staging"
    echo "  $0 kubernetes production"
    echo "  $0 systemd production"
}

# Handle command line arguments
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    show_usage
    exit 0
fi

# Run main function
main "$@"