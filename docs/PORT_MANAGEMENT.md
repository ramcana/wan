# Port Management in WAN2.2

This document describes the port management strategy for the WAN2.2 Video Generation System to ensure consistent port usage across all components.

## Port Configuration Overview

All ports in the WAN2.2 system are centrally managed to avoid conflicts and ensure consistency. The following table lists the standard ports used by different components:

| Component | Port | Environment Variable | Description |
|-----------|------|---------------------|-------------|
| Backend API | 8000 | `BACKEND_PORT` | FastAPI server for API endpoints |
| Frontend Dev | 3000 | `FRONTEND_PORT` | Vite development server |
| Gradio UI | 7860 | `GRADIO_PORT` | Legacy Gradio interface |
| Database | 5432 | `DATABASE_PORT` | PostgreSQL database (optional) |
| Redis | 6379 | `REDIS_PORT` | Redis cache (optional) |

## Configuration Management

### Environment Variables

Ports are configured using environment variables to allow easy customization without code changes:

```bash
# Backend server port
BACKEND_PORT=8000

# Frontend development server port
FRONTEND_PORT=3000

# Gradio UI port
GRADIO_PORT=7860

# Database port (if using external database)
DATABASE_PORT=5432

# Redis port (for caching)
REDIS_PORT=6379
```

### Configuration Files

The infrastructure includes a centralized port configuration module at `infrastructure/config/ports.py` that defines all port-related constants used throughout the application.

## Best Practices

### 1. Use Environment Variables

Always use environment variables for port configuration rather than hardcoding values:

**Good:**
```python
import os
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
```

**Avoid:**
```python
BACKEND_PORT = 8000  # Hardcoded value
```

### 2. Reference Centralized Configuration

Use the centralized port configuration module when possible:

```python
from infrastructure.config.ports import BACKEND_PORT, get_backend_url
```

### 3. Consistent API Prefix

All API endpoints should use the standard prefix `/api/v1` to maintain consistency.

### 4. Proxy-Aware Configuration

When configuring for reverse proxy setups, use relative paths or environment-based URLs:

```typescript
// Use environment variable or relative path for proxy compatibility
const baseURL = import.meta.env.VITE_API_URL || '';
```

## Validation

### Automated Port Validation

The system includes a port validation script that checks for consistent port usage across the codebase:

```bash
# Run port validation
cd scripts
python validate-ports.py
```

The script checks:
- Hardcoded port numbers that don't match standard configuration
- Environment variable consistency
- Configuration file integrity

### Manual Validation

To manually verify port configurations:

1. Check environment files (`.env`, `.env.*`)
2. Review docker-compose files for port mappings
3. Verify Nginx/Traefik configuration files
4. Check startup scripts for port references

## Docker Configuration

When running in Docker, ensure ports are properly mapped in `docker-compose.yml`:

```yaml
services:
  wan22-app:
    ports:
      - "${BACKEND_PORT:-8000}:${BACKEND_PORT:-8000}"
```

## Reverse Proxy Configuration

When using reverse proxies (Nginx/Traefik), configure them to route requests to the appropriate internal ports without exposing them externally.

### Nginx Example
```nginx
location /api/ {
    proxy_pass http://backend:8000;
}
```

### Traefik Example
```yaml
labels:
  - "traefik.http.services.wan22-backend.loadbalancer.server.port=8000"
```

## Troubleshooting

### Port Conflicts

If you encounter port conflicts:

1. Check which processes are using the port:
   ```bash
   netstat -ano | findstr :8000
   ```

2. Change the port in environment configuration:
   ```bash
   echo "BACKEND_PORT=8001" >> .env.local
   ```

3. Update docker-compose port mappings if needed

### Connection Issues

If components can't connect:

1. Verify port configurations match between client and server
2. Check firewall settings
3. Ensure containers are on the same network (in Docker)
4. Validate reverse proxy configuration

## Adding New Ports

When adding new services that require ports:

1. Add the port to the standard ports table
2. Create an environment variable with a descriptive name
3. Add the configuration to `infrastructure/config/ports.py`
4. Update the port validation script if needed
5. Document the new port in this file