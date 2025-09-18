# WAN2.2 Traefik Setup

This document describes how to set up Traefik as a reverse proxy for the WAN2.2 Video Generation System, allowing both frontend and backend to be served from the same domain.

## Architecture

With the Traefik setup:
- Both frontend and backend are accessible via `http://app.localhost`
- API requests to `/api/*` are automatically routed to the backend service
- Static frontend files are served by the backend service
- Services can run on any internal ports without affecting the external interface

## Setup Instructions

### 1. Build the Frontend

First, build the frontend application:

```bash
cd frontend
npm run build
```

### 2. Start Services with Traefik

Use the provided docker-compose override to start all services with Traefik:

```bash
docker-compose -f backend/docker-compose.yml -f docker-compose.traefik.yml up -d
```

### 3. Access the Application

Once the services are running, access the application at:
- http://app.localhost

### 4. Access Traefik Dashboard

Traefik dashboard is available at:
- http://localhost:8080

## Configuration Details

### Traefik Configuration

The Traefik configuration consists of:

1. **Static Configuration** (`traefik/traefik.yml`):
   - Defines entry points (web on port 80)
   - Configures Docker provider
   - Enables API dashboard
   - Sets logging level

2. **Dynamic Configuration** (`traefik/dynamic_conf.yml`):
   - Defines routers for frontend and API
   - Sets up services and load balancing
   - Configures routing rules

### Docker Labels

The backend service is configured with Docker labels that tell Traefik how to route requests:
- `traefik.enable=true` - Enables Traefik for this service
- `traefik.http.routers.wan22-app.rule=Host(\`app.localhost\`)` - Routing rule
- `traefik.http.services.wan22-app.loadbalancer.server.port=8000` - Backend port

## Development vs Production

### Development Mode

For development with hot reloading:
```bash
# Start services in development mode
docker-compose -f backend/docker-compose.yml up -d wan22-dev traefik
```

In development mode:
- Frontend runs on port 3000 with hot reloading
- Backend runs on port 8000
- Traefik routes requests to the appropriate services

### Production Mode

For production deployment:
```bash
# Build frontend and start production services
docker-compose -f backend/docker-compose.yml -f docker-compose.traefik.yml up -d
```

In production mode:
- Frontend is built and served by the backend service
- Backend runs in production mode
- Traefik routes requests to the backend service

## Customization

### Domain Name

To use a custom domain name:
1. Update the routing rules in `traefik/dynamic_conf.yml`
2. Update the Docker labels on the backend service
3. Update your system's hosts file or DNS configuration

### SSL/HTTPS

To enable HTTPS:
1. Modify `traefik/traefik.yml` to add a websecure entry point
2. Add TLS configuration to the routers
3. Provide SSL certificates or configure Let's Encrypt

### Middleware

Traefik supports middleware for additional functionality:
- Add authentication
- Implement rate limiting
- Add compression
- Configure headers

## Troubleshooting

### Service Not Accessible

1. Check that all containers are running:
   ```bash
   docker-compose -f backend/docker-compose.yml -f docker-compose.traefik.yml ps
   ```

2. Check Traefik logs:
   ```bash
   docker logs wan22-traefik
   ```

3. Check backend logs:
   ```bash
   docker logs wan22-production
   ```

### API Requests Failing

1. Verify the backend is accessible directly:
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

2. Check Traefik routing configuration in `traefik/dynamic_conf.yml`

### Traefik Dashboard Not Accessible

1. Verify Traefik is running:
   ```bash
   docker ps | grep traefik
   ```

2. Check Traefik configuration for API settings

## Benefits of This Setup

1. **Automatic Service Discovery**: Traefik automatically discovers services via Docker labels
2. **Dynamic Configuration**: Configuration updates without restarting Traefik
3. **Single Endpoint**: Users access both frontend and backend through one URL
4. **Port Abstraction**: Internal services can run on any ports
5. **Security**: Traefik provides middleware for security features
6. **Performance**: Efficient routing with load balancing capabilities
7. **Scalability**: Easy to add new services and routing rules
8. **Flexibility**: Simple to modify routing and add middleware