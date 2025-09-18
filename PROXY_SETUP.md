# WAN2.2 Reverse Proxy Setup

This document describes how to set up Nginx as a reverse proxy for the WAN2.2 Video Generation System, allowing both frontend and backend to be served from the same domain.

## Architecture

With the reverse proxy setup:
- Both frontend and backend are accessible via `http://app.localhost`
- API requests to `/api/*` are automatically proxied to the backend service
- Static frontend files are served directly by Nginx
- Services can run on any internal ports without affecting the external interface

## Setup Instructions

### 1. Build the Frontend

First, build the frontend application:

```bash
cd frontend
npm run build
```

### 2. Start Services with Proxy

Use the provided docker-compose override to start all services with the Nginx reverse proxy:

```bash
# On Linux/Mac
./scripts/build-and-proxy.sh

# On Windows
scripts\build-and-proxy.bat
```

Alternatively, you can start the services manually:

```bash
docker-compose -f backend/docker-compose.yml -f docker-compose.proxy.yml up -d
```

### 3. Access the Application

Once the services are running, access the application at:
- http://app.localhost

## Configuration Details

### Nginx Configuration

The proxy configuration (`nginx/app.conf`) sets up:

1. **Frontend Serving**: Static files from the built frontend application
2. **API Proxying**: Requests to `/api/*` are forwarded to the backend service
3. **WebSocket Support**: WebSocket connections are properly handled for real-time updates
4. **Security Headers**: Appropriate security headers are added
5. **Rate Limiting**: API requests are rate-limited to prevent abuse
6. **Caching**: Static assets are cached for better performance

### Environment Configuration

The frontend is configured to work with the proxy setup:
- `VITE_API_URL` is left empty in the `.env` file
- When `VITE_API_URL` is empty, the API client uses relative paths
- Relative paths work seamlessly with reverse proxy setups

## Development vs Production

### Development Mode

For development with hot reloading:
```bash
# Start services in development mode
docker-compose -f backend/docker-compose.yml up -d wan22-dev nginx
```

In development mode:
- Frontend runs on port 3000 with hot reloading
- Backend runs on port 8000
- Nginx proxies requests to the appropriate services

### Production Mode

For production deployment:
```bash
# Build frontend and start production services
./scripts/build-and-proxy.sh
```

In production mode:
- Frontend is built and served as static files
- Backend runs in production mode
- Nginx serves both frontend and proxies API requests

## Customization

### Domain Name

To use a custom domain name:
1. Update the `server_name` directive in `nginx/app.conf`
2. Update your system's hosts file or DNS configuration

### SSL/HTTPS

To enable HTTPS:
1. Uncomment the HTTPS server block in `nginx/app.conf`
2. Provide SSL certificates in the `ssl/` directory
3. Update paths to certificate files as needed

### Rate Limiting

Adjust rate limiting parameters in `nginx/app.conf`:
- `limit_req_zone` directives control rate limits
- `burst` parameters in location blocks allow for request bursts

## Troubleshooting

### Service Not Accessible

1. Check that all containers are running:
   ```bash
   docker-compose -f backend/docker-compose.yml -f docker-compose.proxy.yml ps
   ```

2. Check Nginx logs:
   ```bash
   docker logs wan22-nginx-proxy
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

2. Check that API paths are correctly proxied in `nginx/app.conf`

### Frontend Not Loading

1. Verify the frontend build was successful
2. Check that static files are in the `frontend/dist/` directory
3. Verify Nginx configuration for static file serving

## Benefits of This Setup

1. **Single Endpoint**: Users access both frontend and backend through one URL
2. **Port Abstraction**: Internal services can run on any ports
3. **Security**: Nginx provides an additional security layer
4. **Performance**: Nginx efficiently serves static files and can cache responses
5. **Scalability**: Easy to add load balancing and additional services
6. **Flexibility**: Simple to modify routing rules and add new endpoints