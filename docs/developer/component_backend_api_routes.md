---
category: developer
last_updated: '2025-09-15T22:50:00.840869'
original_path: tools\project-structure-analyzer\example-analysis-output\documentation\component_backend_api_routes.md
tags:
- api
- performance
title: backend.api.routes
---

# backend.api.routes

**Type:** package
**Path:** `backend\api\routes`



## Overview

This component contains 2 files.
It depends on 5 other components.

## Files

- `analytics.py`
- `__init__.py`

## Dependencies

**Api Call:**
- backend.api
  - Calls API endpoint /performance
- backend.api
  - Calls API endpoint /dashboard
- backend.api
  - Calls API endpoint /health
- backend.api
  - Calls API endpoint /export

**Import:**
- backend
  - Imports AnalyticsCollector

