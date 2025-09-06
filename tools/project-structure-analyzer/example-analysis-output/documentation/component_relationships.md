# Component Relationships

This document explains how different components in the project interact with each other.

## Architecture Overview

The project consists of 73 main components with 883 dependencies between them.

### Entry Points
These components serve as application entry points:

- **tools.test-auditor**
- **local_installation.scripts**: Automation scripts
- **tests.examples**: Example code
- **tools.dev-feedback**
- **tools.health-checker**
- **backend.api.routes**
- **tools.test-runner**
- **backend.services**: Business services
- **backend.tests**: Test suite
- **backend**: Backend application

### Critical Components
These components are heavily used by others:

- **backend**: Used by 59 components
  - Backend application
- **scripts**: Used by 32 components
  - Automation scripts
- **core**: Used by 12 components
  - Core business logic
- **tests**: Used by 43 components
  - Test suite
- **infrastructure**: Used by 7 components
- **local_testing_framework**: Used by 3 components
- **tools**: Used by 16 components
  - Development tools
- **frontend**: Used by 15 components
  - Frontend application
- **backend.api**: Used by 5 components
  - API endpoints and handlers


## Dependency Types

The project uses several types of dependencies:

- **Config**: 689 dependencies
  - Configuration file references
- **File Reference**: 135 dependencies
  - Direct file path references in code
- **Import**: 54 dependencies
  - Python import statements between modules
- **Api Call**: 5 dependencies
  - HTTP API calls between services


## ⚠️ Circular Dependencies

The following circular dependencies were detected and should be resolved:

### Cycle 1
```
backend → frontend → utils_new → backend
```

**Suggested resolution:**
- Extract common functionality into a shared module
- Use dependency injection or event-driven patterns
- Refactor to create a clear hierarchy

### Cycle 2
```
frontend → utils_new → scripts → frontend
```

**Suggested resolution:**
- Extract common functionality into a shared module
- Use dependency injection or event-driven patterns
- Refactor to create a clear hierarchy

### Cycle 3
```
backend → frontend → utils_new → scripts → backend
```

**Suggested resolution:**
- Extract common functionality into a shared module
- Use dependency injection or event-driven patterns
- Refactor to create a clear hierarchy


## Component Details

Detailed information about each component and its relationships:

### utils_new

**Type:** package
**Files:** 305
**Dependencies:** 199 out, 2 in

**Depends on:**
- local_testing_framework (import)
  - Imports EnvironmentValidator
- core (import)
  - Imports ModelManager
- backend (import)
  - Imports EnvironmentConfig
- config (config)
  - References config.json
- config (config)
  - References config

**Used by:**
- frontend (import)
- infrastructure.hardware (import)

### local_installation

**Type:** module
**Files:** 68
**Dependencies:** 70 out, 0 in

**Depends on:**
- scripts (import)
  - Imports DiagnosticMonitor
- tests (import)
  - Imports AutomatedTestFramework
- config (config)
  - References config.json
- config (config)
  - References config
- config (config)
  - References config

### backend

**Purpose:** Backend application
**Type:** module
**Files:** 15
**Dependencies:** 10 out, 59 in

**Depends on:**
- core (import)
  - Imports get_fallback_recovery_system
- infrastructure (import)
  - Imports GenerationErrorHandler
- config (config)
  - References config
- config (config)
  - References config
- config (config)
  - References config.json

**Used by:**
- backend.api (import)
- backend.api.routes (import)
- backend.api.v1.endpoints (import)
- backend.core (import)
- backend.examples (import)

### scripts

**Purpose:** Automation scripts
**Type:** package
**Files:** 9
**Dependencies:** 15 out, 32 in

**Depends on:**
- config (config)
  - References config
- config (config)
  - References config
- config (config)
  - References ALERTING_SETUP_SUMMARY.md
- config (config)
  - References config
- config (config)
  - References package.json

**Used by:**
- backend.core (import)
- backend.tests (import)
- local_installation (import)
- local_installation.backups.reliability_deployment_20250805_183848.scripts (import)
- local_installation.backups.reliability_deployment_20250805_183919.scripts (import)

### local_installation.scripts

**Purpose:** Automation scripts
**Type:** package
**Files:** 50
**Dependencies:** 45 out, 0 in

**Depends on:**
- config (config)
  - References config
- config (config)
  - References config.json
- config (config)
  - References config
- config (config)
  - References config
- config (config)
  - References config

### tests

**Purpose:** Test suite
**Type:** package
**Files:** 2
**Dependencies:** 1 out, 43 in

**Depends on:**
- config (config)
  - References config

**Used by:**
- backend.tests (import)
- local_installation (import)
- tests.config (import)
- tests.examples (import)
- tests.fixtures (import)

### tools.health-checker

**Type:** package
**Files:** 28
**Dependencies:** 41 out, 0 in

**Depends on:**
- config (config)
  - References config
- config (config)
  - References config
- config (config)
  - References config.json
- config (config)
  - References config
- config (config)
  - References config.json

### core.services

**Purpose:** Business services
**Type:** package
**Files:** 48
**Dependencies:** 40 out, 0 in

**Depends on:**
- core (import)
  - Imports OptimizationManager
- infrastructure (import)
  - Imports ArchitectureDetector
- config (config)
  - References config
- config (config)
  - References config
- config (config)
  - References config

### frontend

**Purpose:** Frontend application
**Type:** module
**Files:** 19
**Dependencies:** 20 out, 15 in

**Depends on:**
- infrastructure (import)
  - Imports SafeEventHandler
- utils_new (import)
  - Imports ComponentValidator
- core (import)
  - Imports LoRAManager
- config (config)
  - References config
- config (config)
  - References config

**Used by:**
- backend (file_reference)
- backend (file_reference)
- scripts (file_reference)
- scripts.startup_manager (file_reference)
- scripts.startup_manager (file_reference)

### infrastructure.hardware

**Type:** package
**Files:** 41
**Dependencies:** 33 out, 0 in

**Depends on:**
- core (import)
  - Imports OptimizationManager
- infrastructure (import)
  - Imports UserFriendlyError
- utils_new (import)
  - Imports ComponentValidator
- config (config)
  - References config.json
- config (config)
  - References config.json

### backend.tests

**Purpose:** Test suite
**Type:** module
**Files:** 68
**Dependencies:** 30 out, 0 in

**Depends on:**
- tests (import)
  - Imports TestEnhancedModelAvailabilityIntegration
- backend (import)
  - Imports EnhancedModelDownloader
- core (import)
  - Imports CORSValidator
- scripts (import)
  - Imports EnhancedModelAvailabilityValidator
- infrastructure (import)
  - Imports ErrorCategory

### scripts.startup_manager

**Type:** package
**Files:** 17
**Dependencies:** 28 out, 0 in

**Depends on:**
- config (config)
  - References config
- config (config)
  - References config
- config (config)
  - References config.json
- config (config)
  - References config
- config (config)
  - References config.json

### local_installation.WAN22-Installation-Package.scripts

**Purpose:** Automation scripts
**Type:** package
**Files:** 32
**Dependencies:** 25 out, 0 in

**Depends on:**
- config (config)
  - References config
- config (config)
  - References config.json
- config (config)
  - References config
- config (config)
  - References config
- config (config)
  - References config

### tests.unit

**Type:** package
**Files:** 9
**Dependencies:** 24 out, 0 in

**Depends on:**
- tests (import)
  - Imports TestConfig
- scripts (import)
  - Imports SystemDetector
- tools (import)
  - Imports UnifiedConfig
- config (config)
  - References config
- config (config)
  - References config

### tests.performance

**Type:** package
**Files:** 15
**Dependencies:** 22 out, 0 in

**Depends on:**
- scripts (import)
  - Imports InteractiveCLI
- config (config)
  - References config
- config (config)
  - References config
- config (config)
  - References config.json
- config (config)
  - References config.json

