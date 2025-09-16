---
category: developer
last_updated: '2025-09-15T22:50:00.840869'
original_path: tools\project-structure-analyzer\example-analysis-output\documentation\component_backend_api.md
tags:
- configuration
- api
- performance
title: backend.api
---

# backend.api

**Type:** package
**Path:** `backend\api`

**Purpose:** API endpoints and handlers



## Overview

This component contains 8 files.
It depends on 5 other components.
It is used by 5 other components.

## Files

- `deployment_health.py`
- `enhanced_model_configuration.py` - Configuration File
- `enhanced_model_management.py`
- `fallback_recovery.py`
- `model_management.py`
- `performance.py`
- `performance_dashboard.py`
- `__init__.py`

## Dependencies

**Config:**
- config
  - References config
- config
  - References config.json
- config
  - References config

**Import:**
- core
  - Imports EnhancedModelDownloader
- backend
  - Imports get_fallback_recovery_system


## Used By

**Api Call:**
- backend.api.routes
  - Calls API endpoint /performance
- backend.api.routes
  - Calls API endpoint /dashboard
- backend.api.routes
  - Calls API endpoint /health
- backend.api.routes
  - Calls API endpoint /export
- backend.api.v1.endpoints
  - Calls API endpoint /status

