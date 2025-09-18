# Model Orchestrator Deployment Guide

## Overview

This guide covers deploying the Model Orchestrator in various environments, from development setups to production clusters. It includes configuration, security, monitoring, and operational considerations.

## Deployment Architectures

### Single Node Deployment

Simplest deployment for development or small-scale production:

```
┌─────────────────────────────────────┐
│           Single Node               │
│  ┌─────────────┐  ┌─────────────┐   │
│  │ WAN2.2 App  │  │ Model Orch. │   │
│  └─────────────┘  └─────────────┘   │
│  ┌─────────────────────────────────┐ │
│  │      Local Storage              │ │
│  └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

### Multi-Node with Shared Storage

Scalable deployment with shared model storage:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Node 1    │    │   Node 2    │    │   Node 3    │
│ WAN2.2 App  │    │ WAN2.2 App  │    │ WAN2.2 App  │
│ Model Orch. │    │ Model Orch. │    │ Model Orch. │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
              ┌─────────────────────────┐
              │    Shared Storage       │
              │  (NFS/S3/Distributed)   │
              └─────────────────────────┘
```

### Cloud-Native Deployment

Kubernetes-based deployment with cloud storage:

```
┌─────────────────────────────────────────────────────┐
│                 Kubernetes Cluster                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │    Pod 1    │  │    Pod 2    │  │    Pod 3    │  │
│  │ WAN2.2 App  │  │ WAN2.2 App  │  │ WAN2.2 App  │  │
│  │ Model Orch. │  │ Model Orch. │  │ Model Orch. │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
                           │
              ┌─────────────────────────┐
              │    Cloud Storage        │
              │   (S3/GCS/Azure)        │
              └─────────────────────────┘
```

## Environment Setup

### Development Environment

#### Prerequisites

- Python 3.8+
- Git
- Docker (optional, for testing with MinIO)

#### Installation

```bash
# Clone repository
git clone <repository-url>
cd wan22

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

#### Configuration

Create `.env` file:

```bash
# Basic configuration
MODELS_ROOT=/data/models
WAN_MODELS_MANIFEST=config/models.toml

# Development settings
LOG_LEVEL=DEBUG
MAX_CONCURRENT_DOWNLOADS=2
ENABLE_METRICS=true

# Optional: HuggingFace token
HF_TOKEN=your_token_here

# Optional: Local MinIO for testing
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
AWS_ENDPOINT_URL=http://localhost:8000
```

#### Running

```bash
# Start the application
python -m backend.main

# Or with development server
python -m backend.main --reload --debug
```

### Production Environment

#### System Requirements

**Minimum Requirements:**

- CPU: 4 cores
- RAM: 8GB
- Storage: 500GB SSD (for model cache)
- Network: 1Gbps

**Recommended Requirements:**

- CPU: 8+ cores
- RAM: 32GB+
- Storage: 2TB+ NVMe SSD
- Network: 10Gbps

#### Operating System Setup

**Ubuntu/Debian:**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.8 python3.8-venv python3.8-dev
sudo apt install -y build-essential curl wget git

# Create application user
sudo useradd -m -s /bin/bash wan22
sudo usermod -aG sudo wan22

# Create directories
sudo mkdir -p /opt/wan22 /data/models
sudo chown -R wan22:wan22 /opt/wan22 /data/models
```

**CentOS/RHEL:**

```bash
# Update system
sudo yum update -y

# Install dependencies
sudo yum install -y python38 python38-devel python38-pip
sudo yum groupinstall -y "Development Tools"

# Create application user
sudo useradd -m wan22
sudo usermod -aG wheel wan22

# Create directories
sudo mkdir -p /opt/wan22 /data/models
sudo chown -R wan22:wan22 /opt/wan22 /data/models
```

#### Application Deployment

```bash
# Switch to application user
sudo su - wan22

# Deploy application
cd /opt/wan22
git clone <repository-url> .
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create configuration
cp config/production.env.example .env
# Edit .env with production settings

# Create systemd service
sudo tee /etc/systemd/system/wan22.service > /dev/null <<EOF
[Unit]
Description=WAN2.2 Model Orchestrator
After=network.target

[Service]
Type=simple
User=wan22
Group=wan22
WorkingDirectory=/opt/wan22
Environment=PATH=/opt/wan22/venv/bin
ExecStart=/opt/wan22/venv/bin/python -m backend.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable wan22
sudo systemctl start wan22
```

## Configuration Management

### Environment Variables

#### Required Variables

```bash
# Core configuration
MODELS_ROOT=/data/models                    # Model storage directory
WAN_MODELS_MANIFEST=/opt/wan22/config/models.toml  # Manifest file path

# Application settings
HOST=0.0.0.0                               # Bind address
PORT=8000                                  # Port number
WORKERS=4                                  # Number of worker processes
```

#### Optional Variables

```bash
# Performance tuning
MAX_CONCURRENT_DOWNLOADS=8                 # Concurrent download limit
DOWNLOAD_TIMEOUT=3600                      # Download timeout (seconds)
RETRY_ATTEMPTS=3                           # Number of retry attempts
RETRY_BACKOFF_FACTOR=2.0                   # Exponential backoff factor

# Storage configuration
MAX_TOTAL_SIZE=1099511627776               # 1TB storage limit
MAX_MODEL_AGE=2592000                      # 30 days in seconds
ENABLE_GARBAGE_COLLECTION=true             # Auto cleanup

# Logging and monitoring
LOG_LEVEL=INFO                             # Logging level
LOG_FORMAT=json                            # Log format (json/text)
ENABLE_METRICS=true                        # Prometheus metrics
METRICS_PORT=9090                          # Metrics port

# Security
ENABLE_AUTH=true                           # Enable authentication
JWT_SECRET=your-secret-key                 # JWT signing key
CORS_ORIGINS=https://your-domain.com       # CORS allowed origins

# External services
HF_TOKEN=your_hf_token                     # HuggingFace token
AWS_ACCESS_KEY_ID=your_access_key          # S3 access key
AWS_SECRET_ACCESS_KEY=your_secret_key      # S3 secret key
AWS_ENDPOINT_URL=https://s3.amazonaws.com  # S3 endpoint
AWS_REGION=us-east-1                       # S3 region
```

### Configuration Files

#### Production Configuration Template

```bash
# /opt/wan22/.env
# WAN2.2 Model Orchestrator Production Configuration

# Core Settings
MODELS_ROOT=/data/models
WAN_MODELS_MANIFEST=/opt/wan22/config/models.toml
HOST=0.0.0.0
PORT=8000
WORKERS=8

# Performance
MAX_CONCURRENT_DOWNLOADS=16
DOWNLOAD_TIMEOUT=7200
RETRY_ATTEMPTS=5
RETRY_BACKOFF_FACTOR=1.5

# Storage Management
MAX_TOTAL_SIZE=2199023255552  # 2TB
MAX_MODEL_AGE=7776000         # 90 days
ENABLE_GARBAGE_COLLECTION=true
GC_CHECK_INTERVAL=3600        # 1 hour

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/wan22/orchestrator.log

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=60

# Security
ENABLE_AUTH=true
JWT_SECRET=your-production-secret-key
CORS_ORIGINS=https://your-production-domain.com
ENABLE_RATE_LIMITING=true
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600

# External Services
HF_TOKEN=your_production_hf_token
AWS_ACCESS_KEY_ID=your_production_access_key
AWS_SECRET_ACCESS_KEY=your_production_secret_key
AWS_ENDPOINT_URL=https://s3.amazonaws.com
AWS_REGION=us-east-1
HF_HUB_ENABLE_HF_TRANSFER=1
```

## Storage Configuration

### Local Storage

#### Directory Structure

```bash
/data/models/
├── .tmp/                    # Temporary downloads
├── .locks/                  # Lock files
├── .state/                  # State information
├── .cache/                  # Metadata cache
├── components/              # Shared components
│   ├── tokenizer@v1.0/
│   └── vae@v2.1/
└── wan22/                   # Model directories
    ├── t2v-A14B@2.2.0/
    ├── i2v-A14B@2.2.0/
    └── ti2v-5b@2.2.0/
```

#### Permissions

```bash
# Set appropriate permissions
sudo chown -R wan22:wan22 /data/models
sudo chmod -R 755 /data/models
sudo chmod -R 700 /data/models/.locks
sudo chmod -R 700 /data/models/.state
```

#### Backup Strategy

```bash
# Daily backup script
#!/bin/bash
BACKUP_DIR="/backup/models"
SOURCE_DIR="/data/models"
DATE=$(date +%Y%m%d)

# Create backup
rsync -av --exclude='.tmp' --exclude='.locks' \
    "$SOURCE_DIR/" "$BACKUP_DIR/models-$DATE/"

# Keep only last 7 days
find "$BACKUP_DIR" -name "models-*" -mtime +7 -exec rm -rf {} \;
```

### S3/MinIO Storage

#### S3 Configuration

```bash
# AWS S3
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_REGION=us-east-1
AWS_ENDPOINT_URL=https://s3.amazonaws.com

# Bucket policy for model storage
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::ACCOUNT:user/wan22-service"
            },
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::wan22-models",
                "arn:aws:s3:::wan22-models/*"
            ]
        }
    ]
}
```

#### MinIO Setup

```bash
# Install MinIO server
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
sudo mv minio /usr/local/bin/

# Create MinIO user and directories
sudo useradd -r -s /sbin/nologin minio
sudo mkdir -p /data/minio
sudo chown minio:minio /data/minio

# Create systemd service
sudo tee /etc/systemd/system/minio.service > /dev/null <<EOF
[Unit]
Description=MinIO Object Storage
After=network.target

[Service]
Type=simple
User=minio
Group=minio
ExecStart=/usr/local/bin/minio server /data/minio --console-address ":9001"
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Start MinIO
sudo systemctl daemon-reload
sudo systemctl enable minio
sudo systemctl start minio

# Configure client
mc alias set local http://localhost:8000 minioadmin minioadmin
mc mb local/wan22-models
```

### Network File Systems

#### NFS Setup

**Server Configuration:**

```bash
# Install NFS server
sudo apt install -y nfs-kernel-server

# Create export directory
sudo mkdir -p /export/models
sudo chown nobody:nogroup /export/models
sudo chmod 755 /export/models

# Configure exports
echo "/export/models *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports

# Start NFS server
sudo systemctl restart nfs-kernel-server
sudo exportfs -a
```

**Client Configuration:**

```bash
# Install NFS client
sudo apt install -y nfs-common

# Mount NFS share
sudo mkdir -p /data/models
sudo mount -t nfs server-ip:/export/models /data/models

# Add to fstab for persistent mount
echo "server-ip:/export/models /data/models nfs defaults 0 0" | sudo tee -a /etc/fstab
```

## Security Configuration

### Authentication and Authorization

#### JWT Configuration

```bash
# Generate secure JWT secret
JWT_SECRET=$(openssl rand -base64 32)
echo "JWT_SECRET=$JWT_SECRET" >> .env

# Configure token expiration
JWT_EXPIRATION=3600  # 1 hour
JWT_REFRESH_EXPIRATION=86400  # 24 hours
```

#### API Key Management

```python
# Create API keys for service accounts
from backend.core.model_orchestrator.auth import create_api_key

# Create service account key
service_key = create_api_key(
    name="model-service",
    permissions=["models:read", "models:download"],
    expires_in=timedelta(days=365)
)
```

### Network Security

#### Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 8000/tcp    # Application
sudo ufw allow 9090/tcp    # Metrics (internal only)
sudo ufw --force enable

# iptables
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 9090 -s 10.0.0.0/8 -j ACCEPT
sudo iptables -A INPUT -j DROP
```

#### TLS/SSL Configuration

```bash
# Generate self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Use Let's Encrypt (production)
sudo apt install -y certbot
sudo certbot certonly --standalone -d your-domain.com

# Configure application
SSL_CERT_PATH=/etc/letsencrypt/live/your-domain.com/fullchain.pem
SSL_KEY_PATH=/etc/letsencrypt/live/your-domain.com/privkey.pem
```

### Credential Management

#### Environment-based Secrets

```bash
# Use systemd environment files
sudo mkdir -p /etc/wan22
sudo tee /etc/wan22/secrets.env > /dev/null <<EOF
HF_TOKEN=your_secret_token
AWS_SECRET_ACCESS_KEY=your_secret_key
JWT_SECRET=your_jwt_secret
EOF

sudo chmod 600 /etc/wan22/secrets.env
sudo chown root:root /etc/wan22/secrets.env

# Update systemd service
[Service]
EnvironmentFile=/etc/wan22/secrets.env
```

#### HashiCorp Vault Integration

```python
# Vault configuration
VAULT_URL=https://vault.example.com
VAULT_TOKEN=your_vault_token
VAULT_MOUNT_POINT=wan22

# Retrieve secrets from Vault
from backend.core.model_orchestrator.vault_client import VaultClient

vault = VaultClient(url=VAULT_URL, token=VAULT_TOKEN)
secrets = vault.get_secrets("wan22/model-orchestrator")
```

## Monitoring and Observability

### Prometheus Metrics

#### Metrics Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "wan22-orchestrator"
    static_configs:
      - targets: ["localhost:9090"]
    scrape_interval: 5s
    metrics_path: /metrics
```

#### Key Metrics to Monitor

```promql
# Download success rate
rate(model_downloads_total{status="success"}[5m]) / rate(model_downloads_total[5m])

# Average download duration
rate(model_download_duration_seconds_sum[5m]) / rate(model_download_duration_seconds_count[5m])

# Storage usage
model_storage_bytes_used / model_storage_bytes_total

# Error rate
rate(model_errors_total[5m])

# Lock contention
rate(lock_timeouts_total[5m])
```

### Grafana Dashboards

#### Dashboard Configuration

```json
{
  "dashboard": {
    "title": "WAN2.2 Model Orchestrator",
    "panels": [
      {
        "title": "Download Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(model_downloads_total{status=\"success\"}[5m]) / rate(model_downloads_total[5m])"
          }
        ]
      },
      {
        "title": "Storage Usage",
        "type": "gauge",
        "targets": [
          {
            "expr": "model_storage_bytes_used / model_storage_bytes_total * 100"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration

#### Structured Logging

```python
# logging_config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        if hasattr(record, 'correlation_id'):
            log_entry['correlation_id'] = record.correlation_id

        if hasattr(record, 'model_id'):
            log_entry['model_id'] = record.model_id

        return json.dumps(log_entry)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/wan22/orchestrator.log')
    ]
)

for handler in logging.root.handlers:
    handler.setFormatter(JSONFormatter())
```

#### Log Aggregation

**ELK Stack Configuration:**

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/wan22/*.log
  json.keys_under_root: true
  json.add_error_key: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "wan22-orchestrator-%{+yyyy.MM.dd}"

# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "wan22-orchestrator" {
    json {
      source => "message"
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "wan22-orchestrator-%{+YYYY.MM.dd}"
  }
}
```

## High Availability and Scaling

### Load Balancing

#### Nginx Configuration

```nginx
# /etc/nginx/sites-available/wan22
upstream wan22_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://wan22_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Increase timeouts for large downloads
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    location /health {
        proxy_pass http://wan22_backend;
        access_log off;
    }

    location /metrics {
        proxy_pass http://wan22_backend;
        allow 10.0.0.0/8;
        deny all;
    }
}
```

#### HAProxy Configuration

```
# /etc/haproxy/haproxy.cfg
global
    daemon
    maxconn 4096

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend wan22_frontend
    bind *:80
    default_backend wan22_servers

backend wan22_servers
    balance roundrobin
    option httpchk GET /health
    server web1 10.0.1.10:8000 check
    server web2 10.0.1.11:8000 check
    server web3 10.0.1.12:8000 check
```

### Database Clustering

#### PostgreSQL High Availability

```bash
# Primary server setup
sudo apt install -y postgresql-12 postgresql-contrib-12

# Configure primary
sudo -u postgres psql -c "CREATE USER replicator WITH REPLICATION ENCRYPTED PASSWORD 'password';"

# postgresql.conf
wal_level = replica
max_wal_senders = 3
wal_keep_segments = 64
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/12/main/archive/%f'

# pg_hba.conf
host replication replicator 10.0.1.0/24 md5

# Standby server setup
sudo -u postgres pg_basebackup -h primary-ip -D /var/lib/postgresql/12/main -U replicator -P -v -R -W

# recovery.conf
standby_mode = 'on'
primary_conninfo = 'host=primary-ip port=5432 user=replicator password=password'
```

### Container Orchestration

#### Docker Compose

```yaml
# docker-compose.yml
version: "3.8"

services:
  wan22-orchestrator:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODELS_ROOT=/data/models
      - DATABASE_URL=postgresql://user:pass@db:5432/wan22
    volumes:
      - models_data:/data/models
      - ./config:/app/config
    depends_on:
      - db
      - redis
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: "2"
          memory: 4G
        reservations:
          cpus: "1"
          memory: 2G

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=wan22
      - POSTGRES_USER=wan22
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - wan22-orchestrator

volumes:
  models_data:
  postgres_data:
  redis_data:
```

#### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wan22-orchestrator
  labels:
    app: wan22-orchestrator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: wan22-orchestrator
  template:
    metadata:
      labels:
        app: wan22-orchestrator
    spec:
      containers:
        - name: orchestrator
          image: wan22/orchestrator:latest
          ports:
            - containerPort: 8000
          env:
            - name: MODELS_ROOT
              value: "/data/models"
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: wan22-secrets
                  key: database-url
          volumeMounts:
            - name: models-storage
              mountPath: /data/models
          resources:
            requests:
              memory: "2Gi"
              cpu: "1"
            limits:
              memory: "4Gi"
              cpu: "2"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
      volumes:
        - name: models-storage
          persistentVolumeClaim:
            claimName: models-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: wan22-orchestrator-service
spec:
  selector:
    app: wan22-orchestrator
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 2Ti
  storageClassName: fast-ssd
```

## Backup and Disaster Recovery

### Backup Strategy

#### Model Data Backup

```bash
#!/bin/bash
# backup-models.sh

BACKUP_ROOT="/backup/wan22"
MODELS_ROOT="/data/models"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_ROOT/models_$DATE"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup model files (exclude temporary and lock files)
rsync -av \
    --exclude='.tmp' \
    --exclude='.locks' \
    --exclude='*.partial' \
    "$MODELS_ROOT/" "$BACKUP_DIR/"

# Create manifest
echo "Backup created: $DATE" > "$BACKUP_DIR/backup_info.txt"
echo "Source: $MODELS_ROOT" >> "$BACKUP_DIR/backup_info.txt"
echo "Size: $(du -sh $BACKUP_DIR | cut -f1)" >> "$BACKUP_DIR/backup_info.txt"

# Compress backup
tar -czf "$BACKUP_ROOT/models_$DATE.tar.gz" -C "$BACKUP_ROOT" "models_$DATE"
rm -rf "$BACKUP_DIR"

# Cleanup old backups (keep last 7 days)
find "$BACKUP_ROOT" -name "models_*.tar.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_ROOT/models_$DATE.tar.gz"
```

#### Database Backup

```bash
#!/bin/bash
# backup-database.sh

DB_NAME="wan22"
DB_USER="wan22"
BACKUP_DIR="/backup/wan22/database"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# Create database dump
pg_dump -h localhost -U "$DB_USER" -d "$DB_NAME" \
    --no-password --clean --create \
    > "$BACKUP_DIR/wan22_$DATE.sql"

# Compress backup
gzip "$BACKUP_DIR/wan22_$DATE.sql"

# Cleanup old backups
find "$BACKUP_DIR" -name "wan22_*.sql.gz" -mtime +30 -delete

echo "Database backup completed: $BACKUP_DIR/wan22_$DATE.sql.gz"
```

### Disaster Recovery

#### Recovery Procedures

**Complete System Recovery:**

```bash
#!/bin/bash
# disaster-recovery.sh

BACKUP_ROOT="/backup/wan22"
MODELS_ROOT="/data/models"

# 1. Stop services
sudo systemctl stop wan22
sudo systemctl stop postgresql

# 2. Restore model data
LATEST_BACKUP=$(ls -t $BACKUP_ROOT/models_*.tar.gz | head -1)
echo "Restoring from: $LATEST_BACKUP"

# Clear existing data
sudo rm -rf "$MODELS_ROOT"/*

# Extract backup
tar -xzf "$LATEST_BACKUP" -C "$BACKUP_ROOT"
BACKUP_DIR=$(basename "$LATEST_BACKUP" .tar.gz)
sudo cp -r "$BACKUP_ROOT/$BACKUP_DIR"/* "$MODELS_ROOT/"
sudo chown -R wan22:wan22 "$MODELS_ROOT"

# 3. Restore database
LATEST_DB_BACKUP=$(ls -t $BACKUP_ROOT/database/wan22_*.sql.gz | head -1)
echo "Restoring database from: $LATEST_DB_BACKUP"

sudo -u postgres dropdb wan22
sudo -u postgres createdb wan22
gunzip -c "$LATEST_DB_BACKUP" | sudo -u postgres psql wan22

# 4. Start services
sudo systemctl start postgresql
sudo systemctl start wan22

# 5. Verify recovery
sleep 10
curl -f http://localhost:8000/health || echo "Health check failed"

echo "Disaster recovery completed"
```

#### Automated Recovery Testing

```bash
#!/bin/bash
# test-recovery.sh

# Create test environment
docker-compose -f docker-compose.test.yml up -d

# Wait for services to start
sleep 30

# Run recovery test
./disaster-recovery.sh

# Verify functionality
python -m pytest tests/test_disaster_recovery.py

# Cleanup
docker-compose -f docker-compose.test.yml down -v
```

## Performance Tuning

### System Optimization

#### Kernel Parameters

```bash
# /etc/sysctl.conf
# Network optimization
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 65536 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 5000

# File system optimization
fs.file-max = 2097152
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# Apply changes
sudo sysctl -p
```

#### File System Tuning

```bash
# Mount options for model storage
/dev/sdb1 /data/models ext4 defaults,noatime,nodiratime 0 2

# For high-performance storage
/dev/nvme0n1 /data/models xfs defaults,noatime,largeio,inode64 0 2
```

### Application Optimization

#### Connection Pooling

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

#### Caching Configuration

```python
# cache.py
import redis
from functools import wraps

redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True,
    max_connections=100
)

def cache_result(expiration=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)

            # Execute function and cache result
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator
```

## Troubleshooting

### Common Issues

#### High Memory Usage

**Symptoms:**

- OOM kills
- Slow performance
- Swap usage

**Solutions:**

```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Tune garbage collection
export GC_CHECK_INTERVAL=1800  # 30 minutes
export MAX_TOTAL_SIZE=$((500 * 1024**3))  # 500GB

# Optimize download concurrency
export MAX_CONCURRENT_DOWNLOADS=4
```

#### Slow Downloads

**Symptoms:**

- Long download times
- Timeouts
- Network errors

**Solutions:**

```bash
# Enable HF transfer acceleration
export HF_HUB_ENABLE_HF_TRANSFER=1

# Tune network settings
echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf
sysctl -p

# Check bandwidth
iperf3 -c speedtest.example.com
```

#### Lock Contention

**Symptoms:**

- Lock timeout errors
- Slow model access
- High CPU usage

**Solutions:**

```bash
# Increase lock timeout
export LOCK_TIMEOUT=600  # 10 minutes

# Check for stale locks
wan models cleanup-locks

# Monitor lock usage
curl http://localhost:9090/metrics | grep lock
```

### Diagnostic Tools

#### Health Check Script

```bash
#!/bin/bash
# health-check.sh

echo "=== WAN2.2 Model Orchestrator Health Check ==="

# Check service status
echo "Service Status:"
systemctl is-active wan22 || echo "Service not running"

# Check disk space
echo "Disk Usage:"
df -h /data/models

# Check memory usage
echo "Memory Usage:"
free -h

# Check network connectivity
echo "Network Connectivity:"
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health

# Check model status
echo "Model Status:"
wan models status --json | jq '.models[] | select(.state != "COMPLETE") | {id: .model_id, state: .state}'

# Check recent errors
echo "Recent Errors:"
journalctl -u wan22 --since "1 hour ago" --grep ERROR

echo "=== Health Check Complete ==="
```

#### Performance Monitoring

```bash
#!/bin/bash
# monitor-performance.sh

# Monitor system resources
echo "=== System Resources ==="
top -bn1 | head -20

# Monitor network usage
echo "=== Network Usage ==="
iftop -t -s 10

# Monitor disk I/O
echo "=== Disk I/O ==="
iostat -x 1 5

# Monitor application metrics
echo "=== Application Metrics ==="
curl -s http://localhost:9090/metrics | grep -E "(download|storage|error)_"
```

This deployment guide provides comprehensive coverage of production deployment scenarios, from single-node setups to cloud-native Kubernetes deployments. It includes security best practices, monitoring configuration, and troubleshooting procedures to ensure reliable operation in production environments.
