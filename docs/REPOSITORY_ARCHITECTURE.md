# WAN2.2 Repository Architecture Guide

This document provides a comprehensive overview of the WAN2.2 repository structure and explains how all components relate to each other.

## 🏗️ High-Level Architecture

WAN2.2 is a **modern video generation system** built with a **microservices-inspired architecture** that separates concerns into distinct layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    WAN2.2 System                           │
├─────────────────────────────────────────────────────────────┤
│  Frontend Layer    │  Backend Layer    │  Core Layer       │
│  ┌─────────────┐   │  ┌─────────────┐  │  ┌─────────────┐  │
│  │   React     │◄──┤  │   FastAPI   │◄─┤  │   Domain    │  │
│  │   TypeScript│   │  │   REST API  │  │  │   Models    │  │
│  │   Tailwind  │   │  │   WebSocket │  │  │   Services  │  │
│  └─────────────┘   │  └─────────────┘  │  └─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                Infrastructure Layer                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │
│  │   Config    │   │   Storage   │   │  Hardware   │      │
│  │ Management  │   │ Management  │   │ Monitoring  │      │
│  └─────────────┘   └─────────────┘   └─────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                   Support Systems                          │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │
│  │   Testing   │   │    Tools    │   │    Docs     │      │
│  │ Framework   │   │   Scripts   │   │  Guides     │      │
│  └─────────────┘   └─────────────┘   └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Directory Structure Overview

### Core Application Components

#### `/backend/` - FastAPI Backend Service

**Purpose**: REST API server handling video generation requests, model management, and business logic

```
backend/
├── api/                    # API endpoints and routing
│   ├── v1/endpoints/      # Versioned API endpoints
│   ├── model_management.py # Model lifecycle management
│   └── performance.py     # Performance monitoring endpoints
├── core/                  # Core business logic
│   ├── model_*.py         # Model management services
│   ├── performance_*.py   # Performance monitoring
│   └── enhanced_*.py      # Enhanced features
├── services/              # Business services
│   └── generation_service.py # Video generation pipeline
├── websocket/             # Real-time communication
│   ├── manager.py         # WebSocket connection management
│   └── progress_integration.py # Progress updates
├── tests/                 # Backend-specific tests
└── app.py                 # FastAPI application entry point
```

**Key Features**:

- Async FastAPI with automatic OpenAPI documentation
- WebSocket support for real-time progress updates
- Model management and health monitoring
- Performance analytics and optimization
- CORS configuration for frontend integration

#### `/frontend/` - React Frontend Application

**Purpose**: Modern web interface for video generation with real-time updates

```
frontend/
├── src/
│   ├── components/        # React components
│   ├── lib/              # Utility libraries
│   │   ├── api-client.ts # Backend API integration
│   │   ├── stream-manager.ts # Real-time updates
│   │   └── cache-manager.ts # Client-side caching
│   └── tests/            # Frontend tests
├── public/               # Static assets
└── package.json          # Node.js dependencies
```

**Key Features**:

- TypeScript for type safety
- Tailwind CSS for responsive design
- Real-time progress tracking via WebSocket
- Offline support and caching
- Accessibility compliance

#### `/core/` - Domain Logic Layer

**Purpose**: Pure business logic independent of frameworks

```
core/
├── models/               # Domain models and entities
├── services/            # Core business services
└── interfaces/          # Abstract interfaces for dependency injection
```

**Key Features**:

- Framework-agnostic business logic
- Domain-driven design principles
- Dependency injection interfaces

#### `/infrastructure/` - Infrastructure Layer

**Purpose**: External concerns like configuration, storage, and hardware management

```
infrastructure/
├── config/              # Configuration management
├── storage/            # File and model storage
└── hardware/           # Hardware monitoring and optimization
```

### Support Systems

#### `/local_testing_framework/` - Comprehensive Testing System

**Purpose**: Automated testing, validation, and quality assurance system

```
local_testing_framework/
├── cli/                 # Command-line interface
├── tests/              # Test suites
├── models/             # Test result models
├── examples/           # Usage examples
├── environment_validator.py # System validation
├── performance_tester.py    # Performance benchmarks
├── integration_tester.py    # Integration tests
└── production_validator.py  # Production readiness checks
```

**Relationship to Main App**:

- **Independent testing system** that validates the main application
- Can be run standalone: `python -m local_testing_framework`
- Provides comprehensive validation of system functionality
- Generates detailed reports on system health and performance
- **Not part of the runtime application** - it's a development/QA tool

#### `/tools/` - Development and Maintenance Tools

**Purpose**: Automated tools for code quality, maintenance, and development workflow

```
tools/
├── code-quality/        # Code analysis and formatting
├── health-checker/      # System health monitoring
├── test-runner/         # Test execution and reporting
├── doc-generator/       # Documentation generation
├── config-analyzer/     # Configuration analysis
└── unified-cli/         # Unified command interface
```

**Key Tools**:

- **Code Quality**: Automated code analysis, formatting, and style enforcement
- **Health Checker**: Continuous monitoring of system health and performance
- **Test Runner**: Orchestrated test execution across the entire system
- **Documentation Generator**: Automated documentation generation and validation

#### `/scripts/` - Automation and Deployment Scripts

**Purpose**: Startup management, deployment automation, and system administration

```
scripts/
├── startup_manager/     # Intelligent startup system
│   ├── startup_manager.py # Main startup orchestrator
│   ├── port_manager.py    # Port conflict resolution
│   ├── environment_validator.py # Environment checks
│   └── recovery_engine.py # Error recovery
├── setup/              # Installation and setup scripts
└── deployment/         # Deployment automation
```

**Key Features**:

- **Intelligent Startup Manager**: Automatic port resolution, environment validation
- **Error Recovery**: Automatic recovery from common startup issues
- **Cross-platform Support**: Windows, macOS, and Linux compatibility

#### `/tests/` - Global Test Suite

**Purpose**: System-wide integration tests and test utilities

```
tests/
├── integration/         # Cross-component integration tests
├── performance/         # System performance tests
├── acceptance/          # User acceptance tests
├── utils/              # Test utilities and fixtures
└── conftest.py         # Pytest configuration
```

#### `/docs/` - Documentation System

**Purpose**: Comprehensive documentation for users, developers, and operators

```
docs/
├── user-guides/         # End-user documentation
├── developer-guides/    # Development documentation
├── operations/          # Deployment and operations guides
├── api/                # API documentation
└── troubleshooting/     # Problem resolution guides
```

## 🔄 Component Relationships

### Data Flow Architecture

```
┌─────────────┐    HTTP/WS     ┌─────────────┐    Function     ┌─────────────┐
│   Frontend  │◄──────────────►│   Backend   │◄───────────────►│    Core     │
│   (React)   │                │  (FastAPI)  │     Calls       │  (Domain)   │
└─────────────┘                └─────────────┘                 └─────────────┘
       │                              │                               │
       │                              │                               │
       ▼                              ▼                               ▼
┌─────────────┐                ┌─────────────┐                 ┌─────────────┐
│   Browser   │                │  Database   │                 │   Models    │
│   Storage   │                │   Files     │                 │  Storage    │
└─────────────┘                └─────────────┘                 └─────────────┘
```

### Startup Flow

```
1. User runs: python start.py
   ├── System validation (requirements, ports, dependencies)
   ├── Backend startup (FastAPI on port 8000)
   ├── Frontend startup (React dev server on port 3000)
   └── Browser launch (http://localhost:3000)

2. Alternative: start_both_servers.bat
   ├── Intelligent startup manager
   ├── Port conflict resolution
   ├── Environment validation
   └── Error recovery
```

### Testing Relationship

```
Main Application                    Local Testing Framework
┌─────────────────┐                ┌─────────────────────────┐
│   WAN2.2 App    │                │   Testing System        │
│                 │                │                         │
│ ┌─────────────┐ │   validates    │ ┌─────────────────────┐ │
│ │  Frontend   │ │◄───────────────┤ │  Integration Tests  │ │
│ │  Backend    │ │                │ │  Performance Tests  │ │
│ │  Core       │ │                │ │  Environment Tests  │ │
│ └─────────────┘ │                │ └─────────────────────┘ │
└─────────────────┘                └─────────────────────────┘
```

## 🚀 Entry Points and Usage

### For End Users

```bash
# Simplest startup - just works
python start.py

# Windows batch file
start_both_servers.bat

# Advanced startup with options
python main.py --mode full
```

### For Developers

```bash
# Run tests
python -m local_testing_framework

# Code quality checks
python -m tools.code-quality

# Health monitoring
python -m tools.health-checker

# Documentation generation
python -m tools.doc-generator
```

### For System Administrators

```bash
# Production deployment
python scripts/deploy_production.py

# System diagnostics
python start.py --diagnostics

# Configuration management
python -m tools.config_manager
```

## 🔧 Configuration Management

### Configuration Hierarchy

```
1. Environment Variables (.env)
2. Configuration Files (config/*.yaml)
3. Runtime Configuration (UI settings)
4. Default Values (code defaults)
```

### Key Configuration Files

- `.env` - Environment variables and secrets
- `config/unified-config.yaml` - Main application configuration
- `backend/config.json` - Backend-specific settings
- `frontend/.env.local` - Frontend environment variables

## 📊 Monitoring and Observability

### Health Monitoring

- **System Health**: CPU, memory, disk usage
- **Application Health**: API response times, error rates
- **Model Health**: Model loading status, inference performance
- **Infrastructure Health**: Port availability, dependency status

### Performance Monitoring

- **Startup Performance**: Boot time analysis and optimization
- **Runtime Performance**: Request/response metrics
- **Resource Usage**: Memory and GPU utilization
- **User Experience**: Frontend performance metrics

## 🔄 Development Workflow

### Local Development

1. **Setup**: `python start.py` (handles all dependencies)
2. **Development**: Edit code with hot reload enabled
3. **Testing**: `python -m local_testing_framework`
4. **Quality**: `python -m tools.code-quality`
5. **Documentation**: Auto-generated from code

### Production Deployment

1. **Validation**: System requirements and configuration
2. **Build**: Frontend build and backend packaging
3. **Deploy**: Automated deployment with health checks
4. **Monitor**: Continuous health and performance monitoring

## 🎯 Key Design Principles

### Separation of Concerns

- **Frontend**: User interface and experience
- **Backend**: API and business logic coordination
- **Core**: Pure business logic
- **Infrastructure**: External system concerns
- **Tools**: Development and maintenance automation

### Modularity

- Each component can be developed, tested, and deployed independently
- Clear interfaces between components
- Minimal coupling, maximum cohesion

### Developer Experience

- **Single Command Startup**: `python start.py` handles everything
- **Intelligent Error Recovery**: Automatic problem resolution
- **Comprehensive Documentation**: Generated and maintained automatically
- **Quality Automation**: Code quality, testing, and health monitoring

### Production Readiness

- **Health Monitoring**: Continuous system health validation
- **Performance Optimization**: Automated performance tuning
- **Error Recovery**: Graceful handling of failures
- **Scalability**: Designed for growth and expansion

## 🤝 Contributing

When working with this repository:

1. **Understand the Layer**: Know which layer your changes affect
2. **Follow the Architecture**: Respect the separation of concerns
3. **Use the Tools**: Leverage the automated quality and testing tools
4. **Update Documentation**: Documentation is generated from code
5. **Test Thoroughly**: Use the comprehensive testing framework

## 📚 Additional Resources

- **User Guide**: `docs/user-guides/` - How to use the application
- **Developer Guide**: `docs/developer-guides/` - How to develop and extend
- **API Documentation**: `http://localhost:8000/docs` - Interactive API docs
- **Troubleshooting**: `docs/troubleshooting/` - Common issues and solutions
- **Architecture Decisions**: `.kiro/specs/` - Design decisions and specifications

---

This architecture enables **rapid development**, **reliable operation**, and **easy maintenance** while providing a **great user experience** and **comprehensive developer tools**.
