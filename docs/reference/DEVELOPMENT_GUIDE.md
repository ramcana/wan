---
category: reference
last_updated: '2025-09-15T16:23:25.779474'
original_path: frontend\DEVELOPMENT_GUIDE.md
tags:
- configuration
- api
- troubleshooting
- installation
title: WAN22 Frontend Development Guide
---

# WAN22 Frontend Development Guide

## Quick Start

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Start Development Server
```bash
# Option A: Frontend only (requires backend running separately)
npm run dev

# Option B: Start both frontend and backend together
npm run start:full

# Option C: Frontend with mock API (no backend needed)
npm run dev:mock
```

### 3. Check Backend Connection
```bash
npm run health:check
```

## Environment Configuration

### Development (.env.development)
- API URL: `http://localhost:8000`
- WebSocket: `ws://localhost:8000`
- Mock API: `false`
- Debug mode: `true`

### Production (.env.production)
- API URL: `/api` (relative)
- WebSocket: `wss://your-domain.com/ws`
- Mock API: `false`
- Debug mode: `false`

## Backend Integration

### Starting Backend
```powershell
# From project root
$env:WAN_MODELS_ROOT="D:\AI\models"
python -m backend --host 0.0.0.0 --port 8000
```

### API Endpoints
- Health: `GET /api/v1/system/health`
- System Stats: `GET /api/v1/system/stats`
- Generation: `POST /api/v1/generation/submit`
- Queue Status: `GET /api/v1/queue/status`
- Models: `GET /api/v1/models`

## Development Features

### 1. Connection Status Monitoring
- Real-time backend health checks
- Automatic port detection (8000, 8000, 3001, 5000)
- Visual connection indicators
- Retry mechanisms

### 2. Error Handling
- Enhanced error boundaries with context
- API error classification
- Development debugging info
- User-friendly error messages

### 3. Mock API Support
```bash
# Enable mock mode for offline development
VITE_MOCK_API=true npm run dev
```

### 4. Hot Reloading
- Environment variable changes trigger reload
- Cache clearing on service worker updates
- Configuration synchronization

## Project Structure

```
src/
├── lib/                    # Core utilities and API layer
│   ├── api-client.ts      # HTTP client with error handling
│   ├── api-schemas.ts     # Type definitions and validation
│   ├── utils.ts           # Common utilities
│   ├── mock-data.ts       # Mock API responses
│   └── startup-validator.ts # Backend connection validation
├── hooks/                 # Custom React hooks
│   ├── api/              # API-specific hooks
│   └── use-backend-status.ts # Connection monitoring
├── components/
│   ├── ui/               # Reusable UI components
│   │   └── connection-status.tsx # Backend status indicator
│   └── error/            # Error handling components
└── pages/                # Route components
```

## Safety Measures

### 1. Backend Protection
- Rate limiting on API calls
- Request timeout configuration
- Graceful degradation when backend unavailable
- No direct backend modifications from frontend

### 2. Error Recovery
- Automatic retry mechanisms
- Fallback to cached data
- User guidance for common issues
- Debug information in development mode

### 3. Development Workflow
1. Always start backend first
2. Verify connection with health check
3. Use mock mode for UI-only development
4. Monitor browser console for errors
5. Check connection status indicator

## Troubleshooting

### Backend Not Detected
1. Ensure backend is running on port 8000
2. Check Windows firewall settings
3. Verify CORS configuration
4. Use `npm run health:check`

### API Errors
1. Check backend logs for errors
2. Verify API endpoint URLs
3. Confirm request/response formats
4. Check network connectivity

### Build Issues
1. Clear node_modules: `rm -rf node_modules && npm install`
2. Clear Vite cache: `npx vite --force`
3. Check TypeScript errors: `npm run build`

### Environment Issues
1. Copy `.env.example` to `.env.development`
2. Verify environment variable names (must start with `VITE_`)
3. Restart dev server after env changes

## Scripts Reference

- `npm run dev` - Start development server
- `npm run dev:mock` - Start with mock API
- `npm run build` - Build for production
- `npm run build:prod` - Build with production config
- `npm run start:backend` - Start backend server
- `npm run start:full` - Start both frontend and backend
- `npm run health:check` - Check backend connectivity
- `npm run test` - Run tests
- `npm run lint` - Run linter

## API Integration Examples

### Making API Calls
```typescript
import { get, post } from '@/lib/api-client';

// Get system stats
const stats = await get('/system/stats');

// Submit generation request
const result = await post('/generation/submit', formData);
```

### Using React Query Hooks
```typescript
import { useSystemStats } from '@/hooks/api/use-system';

function SystemMonitor() {
  const { data, isLoading, error } = useSystemStats();
  // Component logic
}
```

### Error Handling
```typescript
import { ApiError } from '@/lib/api-client';

try {
  const data = await get('/some/endpoint');
} catch (error) {
  if (error instanceof ApiError) {
    console.log('API Error:', error.status, error.message);
  }
}
```

## Best Practices

1. **Always use the API layer** - Don't make direct fetch calls
2. **Handle errors gracefully** - Provide user feedback and retry options
3. **Monitor connection status** - Use the connection status component
4. **Test offline scenarios** - Use mock mode for development
5. **Follow TypeScript patterns** - Use proper types and schemas
6. **Keep backend safe** - Never modify backend state unexpectedly

## Production Deployment

1. Build with production config: `npm run build:prod`
2. Configure reverse proxy for API routes
3. Set up WebSocket proxy for real-time features
4. Enable HTTPS for production
5. Configure proper CORS headers
6. Set up monitoring and logging
