# Developer Guide

This section contains technical documentation for developers working on the WAN22 Video Generation System.

## Contents

- [Architecture](architecture.md) - System architecture and design
- [API Reference](../api/index.md) - Complete API documentation
- [Contributing](contributing.md) - How to contribute to the project
- [Development Setup](development-setup.md) - Setting up development environment
- [Testing](testing.md) - Testing guidelines and procedures
- [Performance](performance.md) - Performance optimization guidelines

## Architecture Overview

The WAN22 system follows a modern microservices architecture with clear separation of concerns:

- **Backend Services**: FastAPI-based REST API with WebSocket support
- **Frontend Application**: React/Vite SPA with TypeScript
- **Infrastructure Layer**: Configuration management, startup orchestration, and deployment tools
- **AI Pipeline**: Model management and video generation pipeline

## Development Workflow

1. Set up your development environment using the [Development Setup Guide](development-setup.md)
2. Review the [Architecture Documentation](architecture.md) to understand system design
3. Follow the [Contributing Guidelines](contributing.md) for code standards
4. Use the [Testing Guide](testing.md) to ensure code quality
5. Refer to the [API Reference](../api/index.md) for integration details

## Key Technologies

- **Backend**: Python, FastAPI, PyTorch, Transformers
- **Frontend**: TypeScript, React, Vite, Tailwind CSS
- **Infrastructure**: Docker, GitHub Actions, Batch Scripts
- **AI/ML**: Hugging Face Transformers, Diffusers, CUDA

## Code Organization

The codebase is organized into clear modules:

- `/backend` - Backend services and API
- `/frontend` - Frontend application
- `/scripts` - Utility and deployment scripts
- `/tests` - Comprehensive test suite
- `/docs` - Documentation
- `/config` - Configuration files
