# Requirements Document

## Introduction

This document outlines the requirements for fixing critical network connectivity issues between the React frontend and FastAPI backend in the WAN22 video generation application. The system currently experiences port mismatches, service worker conflicts, and request stream handling errors that prevent proper API communication.

## Requirements

### Requirement 1

**User Story:** As a developer, I want the frontend to correctly connect to the backend API endpoints, so that API calls succeed without port mismatch or caching errors.

#### Acceptance Criteria

1. WHEN in development mode THEN the system SHALL use Vite's server.proxy to forward /api calls to http://localhost:9000
2. WHEN environment variables are updated THEN the system SHALL unregister service workers via navigator.serviceWorker.getRegistrations() and clear browser cache
3. WHEN API endpoints are called THEN the system SHALL log the resolved URL on each API call for debugging
4. WHEN cached configurations conflict with current settings THEN the system SHALL force refresh the service worker using skipWaiting() and clients.claim()
5. WHEN both servers start THEN the system SHALL query a backend health endpoint to confirm port connectivity and display the correct URLs being used

### Requirement 2

**User Story:** As a developer, I want service worker request handling to work correctly, so that API requests don't fail due to "body stream already read" errors.

#### Acceptance Criteria

1. WHEN service workers intercept mutation requests (POST/PUT/DELETE) THEN the system SHALL use request.clone() before await clone.text() to preserve the original stream
2. WHEN storing offline requests THEN the system SHALL read the body from the cloned request using const clonedRequest = request.clone(); const body = await clonedRequest.text()
3. WHEN a POST request is intercepted THEN cloning SHALL succeed without "body stream already read" errors in 100% of cases
4. WHEN request body reading fails THEN the system SHALL use ReadableStream APIs for recreation if streams are disturbed
5. WHEN service worker errors occur THEN the system SHALL bypass the service worker and allow direct network requests to proceed

### Requirement 3

**User Story:** As a developer, I want proper error handling for network failures, so that I can diagnose and fix connectivity issues quickly.

#### Acceptance Criteria

1. WHEN CORS errors occur THEN the system SHALL verify allow_origins includes frontend URL (http://localhost:3000) and suggest app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"]) fixes
2. WHEN timeout errors happen THEN the system SHALL implement retry logic with exponential backoff with max 3 attempts
3. WHEN connection is refused THEN the system SHALL check if the backend server is running on the expected port using fetch tests
4. WHEN network requests fail THEN the system SHALL provide detailed error information including URL, method, status, and CORS headers
5. WHEN errors are logged THEN the system SHALL include FastAPI CORS settings and Vite proxy rules in diagnostic output

### Requirement 4

**User Story:** As a developer, I want automatic configuration synchronization between frontend and backend, so that port and URL changes don't break connectivity.

#### Acceptance Criteria

1. WHEN backend starts on a different port THEN the frontend proxy configuration SHALL update automatically
2. WHEN environment variables change THEN both frontend and backend SHALL use consistent values
3. WHEN development vs production modes are used THEN the system SHALL apply appropriate URL configurations
4. WHEN configuration files are updated THEN the system SHALL validate connectivity before proceeding
5. WHEN servers restart THEN the system SHALL re-establish connections with updated configurations

### Requirement 6

**User Story:** As a developer, I want real-time connectivity monitoring, so that I can detect and respond to network issues immediately.

#### Acceptance Criteria

1. WHEN servers are running THEN the system SHALL continuously monitor connectivity via heartbeat endpoint every 30 seconds (configurable)
2. WHEN connectivity is lost THEN the system SHALL alert the developer and attempt automatic recovery with exponential backoff
3. WHEN network latency increases above 5 seconds THEN the system SHALL adjust timeout values accordingly and log performance warnings
4. WHEN connection quality degrades THEN the system SHALL provide performance recommendations such as reducing request frequency
5. WHEN monitoring detects issues THEN the system SHALL log detailed diagnostic information including response times and error patterns

### Requirement 5

**User Story:** As a developer, I want comprehensive connectivity diagnostics, so that I can quickly identify and resolve network issues.

#### Acceptance Criteria

1. WHEN connectivity issues occur THEN the system SHALL run automated diagnostics using ping or fetch tests for port/CORS verification
2. WHEN diagnostics run THEN the system SHALL include checks for FastAPI CORS settings and Vite proxy rules
3. WHEN problems are found THEN the system SHALL provide step-by-step resolution instructions with specific configuration examples
4. WHEN diagnostics complete THEN the system SHALL generate a connectivity report with all findings and suggested fixes
5. WHEN fixes are applied THEN the system SHALL re-run diagnostics to verify resolution and log success/failure status

### Requirement 7

**User Story:** As a developer, I want seamless development workflow integration, so that connectivity fixes don't disrupt my development process.

#### Acceptance Criteria

1. WHEN using Vite THEN changes SHALL trigger HMR without service worker reregistration causing connectivity issues
2. WHEN connectivity fixes are applied THEN the system SHALL preserve existing development server sessions and maintain hot reload functionality
3. WHEN configuration changes are made THEN the system SHALL hot-reload without requiring full server restarts using Vite's config reload
4. WHEN errors are fixed THEN the system SHALL automatically retry failed requests and clear error states in the UI
5. WHEN workflow is interrupted THEN the system SHALL provide quick recovery options with minimal downtime and clear status indicators

### Requirement 4

**User Story:** As a developer, I want automatic configuration synchronization between frontend and backend, so that port and URL changes don't break connectivity.

#### Acceptance Criteria

1. WHEN backend starts on a different port THEN the frontend SHALL query a backend health endpoint to confirm port on startup and update proxy configuration
2. WHEN environment variables change THEN both frontend and backend SHALL use consistent values loaded via dotenv for reliable env handling
3. WHEN development vs production modes are used THEN the system SHALL apply appropriate URL configurations with environment-specific settings
4. WHEN configuration files are updated THEN the system SHALL validate connectivity using fetch tests before proceeding with startup
5. WHEN servers restart THEN the system SHALL re-establish connections with updated configurations and log the resolved connection details
