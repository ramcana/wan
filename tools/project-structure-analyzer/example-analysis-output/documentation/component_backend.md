# backend

**Type:** module
**Path:** `backend`

**Purpose:** Backend application



## Overview

This component contains 15 files.
It depends on 10 other components.
It is used by 59 other components.

## Files

- `app.py`
- `demo_cors_validation.py`
- `diagnose_system.py`
- `final_system_check.py`
- `fix_imports.py`
- `fix_vram_validation.py`
- `init_db.py`
- `main.py`
- `optimize_for_rtx4080.py`
- `start_full_stack.py`
- `start_server.py`
- `test_cuda_detection.py`
- `test_enhanced_endpoints_simple.py`
- `test_json_request.py`
- `test_real_ai_ready.py`

## Dependencies

**Config:**
- config
  - References config
- config
  - References config
- config
  - References config.json
- config
  - References config.json
- config
  - References config.json
- ... and 1 more

**File Reference:**
- frontend
  - References frontend_vram_config.json
- frontend
  - References frontend_vram_config.json

**Import:**
- core
  - Imports get_fallback_recovery_system
- infrastructure
  - Imports GenerationErrorHandler


## Used By

**File Reference:**
- backend.scripts.deployment
  - References backend/config.json
- backend.scripts.deployment
  - References backend/core/enhanced_model_config.py
- backend.scripts.deployment
  - References backend/api/enhanced_model_management.py
- backend.scripts.deployment
  - References backend/services/generation_service.py
- backend.scripts.deployment
  - References backend/websocket/model_notifications.py
- ... and 44 more

**Import:**
- backend.api
  - Imports get_fallback_recovery_system
- backend.api.routes
  - Imports AnalyticsCollector
- backend.api.v1.endpoints
  - Imports GenerationRequest
- backend.core
  - Imports FallbackRecoverySystem
- backend.examples
  - Imports IntegratedErrorHandler
- ... and 5 more

