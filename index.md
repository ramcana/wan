# WAN22 Video Generation System

Welcome to the WAN22 Video Generation System documentation.

## Overview

The WAN22 Video Generation System is a comprehensive AI-powered video generation platform that supports multiple model types and provides a robust API for creating high-quality videos from text and image inputs.

## Key Features

- **Multi-Model Support**: T2V-A14B, I2V-A14B, and TI2V-5B models
- **Model Orchestrator**: Automated model management and deployment
- **Hardware Optimization**: RTX 4080 + Threadripper PRO optimizations
- **REST API**: Comprehensive FastAPI-based interface
- **Real-time Monitoring**: Performance metrics and health checks

## Quick Links

- [System Capabilities Report](WAN22_System_Capabilities_Report.md) - Complete system overview
- [Model Orchestrator Requirements](.kiro/specs/model-orchestrator/requirements.md) - Technical requirements
- [Configuration Guide](config/models.toml) - Model configuration

## Getting Started

1. Review the [System Capabilities Report](WAN22_System_Capabilities_Report.md)
2. Check the [Model Orchestrator documentation](.kiro/specs/model-orchestrator/requirements.md)
3. Configure your models using the [configuration files](config/models.toml)

## API Documentation

The system provides a comprehensive REST API. When running locally, you can access:

- **API Documentation**: http://127.0.0.1:8000/docs
- **System Health**: http://127.0.0.1:8000/health
- **Dashboard**: http://127.0.0.1:8000/api/v1/dashboard/html

## Support

For technical support and documentation issues, please refer to the project repository.
