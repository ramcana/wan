#!/bin/bash

# Rollback script for React Frontend + FastAPI Backend deployment
# This script safely rolls back to the previous Gradio system

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_BASE_DIR="$SCRIPT_DIR/migration_backup"
LOG_FILE="$SCRIPT_DIR/rollback.log"
GRADIO_PORT=7860
NEW_SYSTEM_PORT=8000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

# Function to check if a service is running
check_service() {
    local port=$1
    local service_name=$2
    
    if curl -f "http://localhost:$port" > /dev/null 2>&1; then
        return 0  # Service is running
    else
        return 1  # Service is not running
    fi
}

# Function to find the latest backup
find_latest_backup() {
    if [ ! -d "$BACKUP_BASE_DIR" ]; then
        error "Backup directory $BACKUP_BASE_DIR not found"
        exit 1
    fi
    
    local latest_backup=$(ls -t "$BACKUP_BASE_DIR" 2>/dev/null | head -1)
    if [ -z "$latest_backup" ]; then
        error "No backups found in $BACKUP_BASE_DIR"
        exit 1
    fi
    
    echo "$BACKUP_BASE_DIR/$latest_backup"
}

# Function to stop new system
stop_new_system() {
    log "Stopping new React + FastAPI system..."
    
    # Stop Docker containers
    if command -v docker-compose &> /dev/null; then
        if [ -f "docker-compose.yml" ]; then
            docker-compose down 2>/dev/null || true
            log "Docker containers stopped"
        fi
    fi
    
    # Stop uvicorn processes
    pkill -f "uvicorn backend.main:app" 2>/dev/null || true
    
    # Stop any processes on the new system port
    if lsof -ti:$NEW_SYSTEM_PORT > /dev/null 2>&1; then
        lsof -ti:$NEW_SYSTEM_PORT | xargs kill -9 2>/dev/null || true
        log "Processes on port $NEW_SYSTEM_PORT terminated"
    fi
    
    # Wait for processes to stop
    sleep 3
    
    if check_service $NEW_SYSTEM_PORT "new system"; then
        warning "New system may still be running on port $NEW_SYSTEM_PORT"
    else
        success "New system stopped successfully"
    fi
}

# Function to restore configuration
restore_configuration() {
    local backup_dir=$1
    log "Restoring configuration from $backup_dir..."
    
    if [ -f "$backup_dir/config.json" ]; then
        cp "$backup_dir/config.json" "$SCRIPT_DIR/"
        success "Configuration restored"
    else
        warning "No config.json found in backup, using existing configuration"
    fi
}

# Function to restore outputs
restore_outputs() {
    local backup_dir=$1
    log "Restoring outputs from $backup_dir..."
    
    if [ -d "$backup_dir/outputs_backup" ]; then
        # Remove current outputs
        if [ -d "$SCRIPT_DIR/outputs" ]; then
            rm -rf "$SCRIPT_DIR/outputs"
        fi
        
        # Restore from backup
        cp -r "$backup_dir/outputs_backup" "$SCRIPT_DIR/outputs"
        success "Outputs restored"
    else
        warning "No outputs backup found, keeping current outputs"
    fi
}

# Function to restore models
restore_models() {
    local backup_dir=$1
    log "Checking for models backup..."
    
    if [ -d "$backup_dir/models_backup" ]; then
        log "Restoring models from backup..."
        
        # Remove current models
        if [ -d "$SCRIPT_DIR/models" ]; then
            rm -rf "$SCRIPT_DIR/models"
        fi
        
        # Restore from backup
        cp -r "$backup_dir/models_backup" "$SCRIPT_DIR/models"
        success "Models restored"
    else
        log "No models backup found, keeping current models"
    fi
}

# Function to restore LoRAs
restore_loras() {
    local backup_dir=$1
    log "Checking for LoRAs backup..."
    
    if [ -d "$backup_dir/loras_backup" ]; then
        log "Restoring LoRAs from backup..."
        
        # Remove current LoRAs
        if [ -d "$SCRIPT_DIR/loras" ]; then
            rm -rf "$SCRIPT_DIR/loras"
        fi
        
        # Restore from backup
        cp -r "$backup_dir/loras_backup" "$SCRIPT_DIR/loras"
        success "LoRAs restored"
    else
        log "No LoRAs backup found, keeping current LoRAs"
    fi
}

# Function to start Gradio system
start_gradio_system() {
    log "Starting Gradio system..."
    
    # Check if ui.py exists
    if [ ! -f "$SCRIPT_DIR/ui.py" ]; then
        error "ui.py not found. Cannot start Gradio system."
        exit 1
    fi
    
    # Check if port is available
    if check_service $GRADIO_PORT "existing service"; then
        warning "Port $GRADIO_PORT is already in use. Attempting to stop existing service..."
        lsof -ti:$GRADIO_PORT | xargs kill -9 2>/dev/null || true
        sleep 3
    fi
    
    # Start Gradio in background
    cd "$SCRIPT_DIR"
    nohup python3 ui.py > gradio.log 2>&1 &
    GRADIO_PID=$!
    
    log "Gradio system started with PID: $GRADIO_PID"
    
    # Wait for startup
    log "Waiting for Gradio system to start..."
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if check_service $GRADIO_PORT "Gradio"; then
            success "Gradio system is running on port $GRADIO_PORT"
            return 0
        fi
        
        sleep 2
        attempt=$((attempt + 1))
        log "Waiting for Gradio startup... ($attempt/$max_attempts)"
    done
    
    error "Gradio system failed to start within expected time"
    return 1
}

# Function to verify rollback
verify_rollback() {
    log "Verifying rollback..."
    
    # Check if Gradio is responding
    if check_service $GRADIO_PORT "Gradio"; then
        success "Gradio system is responding on port $GRADIO_PORT"
    else
        error "Gradio system is not responding"
        return 1
    fi
    
    # Check if new system is stopped
    if check_service $NEW_SYSTEM_PORT "new system"; then
        warning "New system is still running on port $NEW_SYSTEM_PORT"
    else
        success "New system is properly stopped"
    fi
    
    # Test basic functionality
    log "Testing basic functionality..."
    
    # Check if we can access the Gradio interface
    if curl -f "http://localhost:$GRADIO_PORT" > /dev/null 2>&1; then
        success "Gradio interface is accessible"
    else
        warning "Gradio interface may not be fully ready"
    fi
    
    # Check if models are accessible
    if [ -d "$SCRIPT_DIR/models" ] && [ "$(ls -A $SCRIPT_DIR/models 2>/dev/null)" ]; then
        success "Models directory is accessible"
    else
        warning "Models directory may be empty or inaccessible"
    fi
    
    return 0
}

# Function to create rollback report
create_rollback_report() {
    local backup_dir=$1
    local report_file="$SCRIPT_DIR/rollback_report_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$report_file" << EOF
{
  "rollback_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "backup_used": "$backup_dir",
  "gradio_status": "$(check_service $GRADIO_PORT "Gradio" && echo "running" || echo "stopped")",
  "new_system_status": "$(check_service $NEW_SYSTEM_PORT "new system" && echo "running" || echo "stopped")",
  "files_restored": {
    "configuration": $([ -f "$backup_dir/config.json" ] && echo "true" || echo "false"),
    "outputs": $([ -d "$backup_dir/outputs_backup" ] && echo "true" || echo "false"),
    "models": $([ -d "$backup_dir/models_backup" ] && echo "true" || echo "false"),
    "loras": $([ -d "$backup_dir/loras_backup" ] && echo "true" || echo "false")
  },
  "verification": {
    "gradio_accessible": $(curl -f "http://localhost:$GRADIO_PORT" > /dev/null 2>&1 && echo "true" || echo "false"),
    "models_available": $([ -d "$SCRIPT_DIR/models" ] && [ "$(ls -A $SCRIPT_DIR/models 2>/dev/null)" ] && echo "true" || echo "false")
  }
}
EOF
    
    log "Rollback report created: $report_file"
}

# Function to show rollback summary
show_summary() {
    local backup_dir=$1
    
    echo
    echo "=================================="
    echo "       ROLLBACK SUMMARY"
    echo "=================================="
    echo
    echo "Backup used: $backup_dir"
    echo "Gradio URL: http://localhost:$GRADIO_PORT"
    echo "Log file: $LOG_FILE"
    echo
    
    if check_service $GRADIO_PORT "Gradio"; then
        success "✓ Rollback completed successfully"
        echo "✓ Gradio system is running"
    else
        error "✗ Rollback may have issues"
        echo "✗ Gradio system is not responding"
    fi
    
    echo
    echo "Next steps:"
    echo "1. Test video generation functionality"
    echo "2. Verify all models are working"
    echo "3. Check outputs directory for previous videos"
    echo "4. Review rollback report and logs"
    echo
}

# Main rollback function
main() {
    log "Starting rollback process..."
    
    # Check if running as root (not recommended)
    if [ "$EUID" -eq 0 ]; then
        warning "Running as root. This is not recommended."
    fi
    
    # Find latest backup
    local backup_dir=$(find_latest_backup)
    log "Using backup: $backup_dir"
    
    # Confirm rollback
    if [ "${1:-}" != "--force" ]; then
        echo -e "${YELLOW}This will rollback to the Gradio system using backup:${NC}"
        echo "$backup_dir"
        echo
        read -p "Are you sure you want to proceed? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Rollback cancelled by user"
            exit 0
        fi
    fi
    
    # Execute rollback steps
    stop_new_system
    restore_configuration "$backup_dir"
    restore_outputs "$backup_dir"
    restore_models "$backup_dir"
    restore_loras "$backup_dir"
    
    if start_gradio_system; then
        if verify_rollback; then
            create_rollback_report "$backup_dir"
            show_summary "$backup_dir"
            success "Rollback completed successfully!"
            exit 0
        else
            error "Rollback verification failed"
            exit 1
        fi
    else
        error "Failed to start Gradio system"
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [--force] [--help]"
        echo
        echo "Options:"
        echo "  --force    Skip confirmation prompt"
        echo "  --help     Show this help message"
        echo
        echo "This script rolls back from the new React + FastAPI system"
        echo "to the previous Gradio system using the latest backup."
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac