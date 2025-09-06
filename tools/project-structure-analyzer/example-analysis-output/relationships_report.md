# Component Relationship Analysis Report

## Overview

- **Total Components:** 73
- **Total Dependencies:** 883
- **Circular Dependencies:** 3
- **Critical Components:** 10
- **Isolated Components:** 14
- **Entry Points:** 58

## Critical Components

These components are heavily depended upon by others:

- **backend** (59 dependencies)
  - Purpose: Backend application
- **scripts** (32 dependencies)
  - Purpose: Automation scripts
- **core** (12 dependencies)
  - Purpose: Core business logic
- **tests** (43 dependencies)
  - Purpose: Test suite
- **infrastructure** (7 dependencies)
- **local_testing_framework** (3 dependencies)
- **tools** (16 dependencies)
  - Purpose: Development tools
- **frontend** (15 dependencies)
  - Purpose: Frontend application
- **backend.api** (5 dependencies)
  - Purpose: API endpoints and handlers

## Entry Points

These components serve as application entry points:

- **tools.test-auditor**
  - Files: 7
- **local_installation.scripts**
  - Purpose: Automation scripts
  - Files: 50
- **tests.examples**
  - Purpose: Example code
  - Files: 2
- **tools.dev-feedback**
  - Files: 4
- **tools.health-checker**
  - Files: 28
- **backend.api.routes**
  - Files: 2
- **tools.test-runner**
  - Files: 7
- **backend.services**
  - Purpose: Business services
  - Files: 4
- **backend.tests**
  - Purpose: Test suite
  - Files: 68
- **backend**
  - Purpose: Backend application
  - Files: 15
- **backend.api.v1.endpoints**
  - Files: 10
- **infrastructure.config**
  - Purpose: Configuration management
  - Files: 14
- **local_installation.WAN22-Installation-Package.application**
  - Files: 5
- **local_installation.backups.reliability_deployment_20250805_183919.scripts**
  - Purpose: Automation scripts
  - Files: 3
- **tests.config**
  - Purpose: Configuration management
  - Files: 4
- **local_testing_framework.examples.workflows**
  - Files: 3
- **backend.migration**
  - Files: 2
- **tools.config-analyzer**
  - Files: 3
- **backend.scripts**
  - Purpose: Automation scripts
  - Files: 6
- **local_installation**
  - Files: 68
- **local_installation.application**
  - Files: 5
- **backend.monitoring**
  - Purpose: System monitoring
  - Files: 3
- **local_installation.backups.reliability_deployment_20250805_184057.scripts**
  - Purpose: Automation scripts
  - Files: 3
- **local_installation.WAN22-Installation-Package.scripts**
  - Purpose: Automation scripts
  - Files: 32
- **tests.unit**
  - Files: 9
- **tests.utils**
  - Purpose: Utility functions
  - Files: 11
- **local_installation.tests**
  - Purpose: Test suite
  - Files: 6
- **local_installation.backups.reliability_deployment_20250805_183952.scripts**
  - Purpose: Automation scripts
  - Files: 3
- **local_testing_framework**
  - Files: 12
- **tools.dev-environment**
  - Files: 4
- **backend.core**
  - Purpose: Core business logic
  - Files: 19
- **backend.config**
  - Purpose: Configuration management
  - Files: 3
- **local_installation.WAN22-Installation-Package.resources**
  - Files: 1
- **infrastructure.hardware**
  - Files: 41
- **tools.onboarding**
  - Files: 3
- **local_installation.backups.reliability_deployment_20250805_183848.scripts**
  - Purpose: Automation scripts
  - Files: 3
- **tools.test-quality**
  - Files: 8
- **backend.websocket**
  - Purpose: WebSocket handlers
  - Files: 4
- **tests.fixtures**
  - Files: 2
- **tools.project-structure-analyzer**
  - Files: 9
- **local_installation.backups.reliability_deployment_20250805_184024.scripts**
  - Purpose: Automation scripts
  - Files: 3
- **tests.e2e**
  - Files: 2
- **core.services**
  - Purpose: Business services
  - Files: 48
- **scripts.setup**
  - Files: 2
- **local_testing_framework.models**
  - Purpose: Data models and schemas
  - Files: 3
- **tools.config_manager**
  - Files: 8
- **tools.health-checker.checkers**
  - Files: 5
- **backend.scripts.deployment**
  - Files: 6
- **scripts.startup_manager**
  - Files: 17
- **tests.integration**
  - Files: 12
- **local_installation.resources**
  - Files: 1
- **local_testing_framework.tests**
  - Purpose: Test suite
  - Files: 16
- **tests.performance**
  - Files: 15
- **infrastructure.storage**
  - Files: 2
- **local_testing_framework.cli**
  - Files: 2
- **tools.doc-generator**
  - Files: 11
- **backend.repositories**
  - Files: 2
- **backend.examples**
  - Purpose: Example code
  - Files: 11

## Circular Dependencies

⚠️ These circular dependencies should be resolved:

**Cycle 1:** backend → frontend → utils_new → backend
**Cycle 2:** frontend → utils_new → scripts → frontend
**Cycle 3:** backend → frontend → utils_new → scripts → backend

## Component Details

### utils_new

**Type:** package
**Files:** 305
**Complexity Score:** 1211
**Dependencies:** 199 out, 2 in

**Depends on:**
- local_testing_framework (import)
- core (import)
- backend (import)
- config (config)
- config (config)

### local_installation

**Type:** module
**Files:** 68
**Complexity Score:** 346
**Dependencies:** 70 out, 0 in

**Depends on:**
- scripts (import)
- tests (import)
- config (config)
- config (config)
- config (config)

### local_installation.scripts

**Purpose:** Automation scripts
**Type:** package
**Files:** 50
**Complexity Score:** 235
**Dependencies:** 45 out, 0 in

**Depends on:**
- config (config)
- config (config)
- config (config)
- config (config)
- config (config)

### core.services

**Purpose:** Business services
**Type:** package
**Files:** 48
**Complexity Score:** 231
**Dependencies:** 40 out, 0 in

**Depends on:**
- core (import)
- infrastructure (import)
- config (config)
- config (config)
- config (config)

### backend.tests

**Purpose:** Test suite
**Type:** module
**Files:** 68
**Complexity Score:** 226
**Dependencies:** 30 out, 0 in

**Depends on:**
- tests (import)
- backend (import)
- core (import)
- scripts (import)
- infrastructure (import)

### infrastructure.hardware

**Type:** package
**Files:** 41
**Complexity Score:** 181
**Dependencies:** 33 out, 0 in

**Depends on:**
- core (import)
- infrastructure (import)
- utils_new (import)
- config (config)
- config (config)

### tools.health-checker

**Type:** package
**Files:** 28
**Complexity Score:** 179
**Dependencies:** 41 out, 0 in

**Depends on:**
- config (config)
- config (config)
- config (config)
- config (config)
- config (config)

### backend

**Purpose:** Backend application
**Type:** module
**Files:** 15
**Complexity Score:** 178
**Dependencies:** 10 out, 59 in

**Depends on:**
- core (import)
- infrastructure (import)
- config (config)
- config (config)
- config (config)

### local_installation.WAN22-Installation-Package.scripts

**Purpose:** Automation scripts
**Type:** package
**Files:** 32
**Complexity Score:** 139
**Dependencies:** 25 out, 0 in

**Depends on:**
- config (config)
- config (config)
- config (config)
- config (config)
- config (config)

### frontend

**Purpose:** Frontend application
**Type:** module
**Files:** 19
**Complexity Score:** 128
**Dependencies:** 20 out, 15 in

**Depends on:**
- infrastructure (import)
- utils_new (import)
- core (import)
- config (config)
- config (config)

### scripts

**Purpose:** Automation scripts
**Type:** package
**Files:** 9
**Complexity Score:** 127
**Dependencies:** 15 out, 32 in

**Depends on:**
- config (config)
- config (config)
- config (config)
- config (config)
- config (config)

### scripts.startup_manager

**Type:** package
**Files:** 17
**Complexity Score:** 118
**Dependencies:** 28 out, 0 in

**Depends on:**
- config (config)
- config (config)
- config (config)
- config (config)
- config (config)

### backend.core

**Purpose:** Core business logic
**Type:** package
**Files:** 19
**Complexity Score:** 107
**Dependencies:** 18 out, 0 in

**Depends on:**
- backend (import)
- infrastructure (import)
- core (import)
- scripts (import)
- config (config)

### tests.performance

**Type:** package
**Files:** 15
**Complexity Score:** 96
**Dependencies:** 22 out, 0 in

**Depends on:**
- scripts (import)
- config (config)
- config (config)
- config (config)
- config (config)

### tests

**Purpose:** Test suite
**Type:** package
**Files:** 2
**Complexity Score:** 93
**Dependencies:** 1 out, 43 in

**Depends on:**
- config (config)
