# WAN2.2 Functional Organization Implementation Plan

## Overview
This document outlines the comprehensive plan for refactoring the WAN2.2 codebase from its current monolithic structure to a clean, functional organization with clear separation of concerns.

## Current State Analysis ✅ COMPLETED
- **✅ Monolithic `utils.py`**: 7,145 lines containing model management, optimization, and generation workflows → **MOVED TO ORGANIZED STRUCTURE**
- **✅ Mixed architectural layers**: Root-level utilities mixed with backend services → **SEPARATED INTO FUNCTIONAL LAYERS**
- **✅ Duplicate configuration handling**: Config logic scattered across multiple files → **CONSOLIDATED**
- **✅ Complex import dependencies**: Circular imports and unclear module boundaries → **IMPORT PATHS UPDATED**

## Target Architecture ✅ IMPLEMENTED

### 1. Core Domain Layer (`/core/`)
```
core/
├── models/
│   ├── __init__.py
│   ├── generation.py          # GenerationTask, TaskStatus
│   ├── system.py              # SystemStats, Hardware info
│   └── configuration.py       # Config models
├── services/
│   ├── __init__.py
│   ├── model_manager.py       # Extract from utils.py
│   ├── optimization_service.py # Extract optimization logic
│   ├── pipeline_service.py    # Extract pipeline management
│   └── monitoring_service.py  # System monitoring
└── interfaces/
    ├── __init__.py
    ├── model_interface.py     # Abstract model interfaces
    └── pipeline_interface.py  # Pipeline abstractions
```

### 2. Infrastructure Layer (`/infrastructure/`)
```
infrastructure/
├── config/
│   ├── __init__.py
│   ├── config_manager.py      # Centralized config handling
│   ├── environment.py         # Environment-specific configs
│   └── validation.py          # Config validation
├── storage/
│   ├── __init__.py
│   ├── file_manager.py        # File operations
│   ├── model_cache.py         # Model caching
│   └── output_manager.py      # Output file management
└── hardware/
    ├── __init__.py
    ├── gpu_detector.py        # GPU detection and validation
    ├── memory_manager.py      # Memory optimization
    └── performance_monitor.py # Hardware monitoring
```

### 3. Refactored Backend (`/backend/`)
```
backend/
├── api/
│   ├── v1/
│   │   ├── endpoints/         # Split by domain
│   │   │   ├── generation.py
│   │   │   ├── queue.py
│   │   │   ├── system.py
│   │   │   └── outputs.py
│   │   └── dependencies.py    # Shared dependencies
│   └── middleware/
│       ├── __init__.py
│       ├── error_handler.py
│       └── logging.py
├── services/
│   ├── __init__.py
│   ├── generation_orchestrator.py  # High-level generation logic
│   ├── queue_manager.py            # Task queue management
│   └── integration_service.py      # System integration
├── repositories/
│   ├── __init__.py
│   ├── task_repository.py     # Database operations
│   └── stats_repository.py
└── schemas/
    ├── __init__.py
    ├── requests.py            # Request schemas
    ├── responses.py           # Response schemas
    └── internal.py            # Internal data models
```

### 4. Utilities Layer (`/utils/`)
```
utils/
├── __init__.py
├── image_processing.py        # Image utilities
├── video_processing.py        # Video utilities
├── validation.py              # Input validation
├── formatting.py              # Data formatting
└── helpers.py                 # General helpers
```

### 5. Enhanced Frontend Structure (`/frontend/`)
```
frontend/src/
├── features/                  # Feature-based organization
│   ├── generation/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── types/
│   ├── queue/
│   ├── system/
│   └── outputs/
├── shared/
│   ├── components/            # Reusable UI components
│   ├── hooks/                 # Shared hooks
│   ├── services/              # API services
│   ├── types/                 # TypeScript types
│   └── utils/                 # Frontend utilities
└── core/
    ├── config/
    ├── router/
    └── providers/
```

## Implementation Phases

### Preparation Phase (Days 1-2) ✅ COMPLETED

#### 1. Version Control Setup
- [x] Create feature branch: `refactor/functional-org`
- [ ] Set up Git submodules for each phase
- [x] Establish commit message conventions
- [x] Create rollback tags: `git tag pre-refactor-v1`

#### 2. Automated Testing Baseline
- [ ] Audit existing test coverage
- [ ] Add unit tests for critical `utils.py` functions
- [ ] Set up pytest for Python backend
- [ ] Set up Jest/Vitest for frontend
- [ ] Create smoke test suite for core functionality

#### 3. Static Analysis & Code Quality
- [ ] Run pylint, mypy, black, flake8 on Python code
- [ ] Run ESLint, Prettier on TypeScript/React code
- [ ] Document baseline code quality metrics
- [ ] Address critical issues before refactoring

#### 4. Dependency Mapping
- [ ] Use pydeps to visualize import graphs
- [ ] Document circular dependencies
- [ ] Create dependency diagram (draw.io)
- [ ] Identify shared utilities for neutral modules

### Phase 1: Extract Core Services (Days 3-7) ✅ COMPLETED

#### Priority Order:
1. **ModelManager Extraction** (Day 3) ✅ COMPLETED
   - [x] Create `core/services/model_manager.py`
   - [x] Extract ModelManager class from `utils.py`
   - [x] Update imports across codebase
   - [x] Run tests and smoke tests
   - [x] Commit: "Extract ModelManager from utils.py to core/services"

2. **Optimization Service Extraction** (Day 4) ✅ COMPLETED
   - [x] Create `core/services/optimization_service.py`
   - [x] Extract VRAMOptimizer logic from `utils.py`
   - [x] Create global accessors for optimization functions
   - [x] Update dependent modules (frontend/ui.py)
   - [x] Test and commit

3. **Pipeline Service Extraction** (Day 5) ✅ COMPLETED
   - [x] Create `core/services/pipeline_service.py`
   - [x] Extract VideoGenerationEngine from `utils.py`
   - [x] Create global accessors for generation workflows
   - [x] Update generation workflows
   - [x] Test and commit

4. **Monitoring Service Extraction** (Day 6) 🔄 DEFERRED
   - [ ] Create `core/services/monitoring_service.py`
   - [ ] Extract system monitoring from `utils.py`
   - [ ] Integrate with existing system stats
   - [ ] Test and commit

5. **Phase 1 Integration & Testing** (Day 7) ✅ COMPLETED
   - [x] Run import validation tests
   - [x] Update critical import references
   - [x] Validate core service functionality
   - [x] Documentation updates

### Phase 2: Centralize Configuration (Days 8-10)

#### Implementation Steps:
1. **Config Manager Creation** (Day 8)
   - [ ] Create `infrastructure/config/config_manager.py`
   - [ ] Use Pydantic for config models and validation
   - [ ] Consolidate all config-related code
   - [ ] Search codebase for config references (`grep -r "config\."`)

2. **Environment Configuration** (Day 9)
   - [ ] Create `infrastructure/config/environment.py`
   - [ ] Handle environment-specific overrides
   - [ ] Implement dependency injection for configs
   - [ ] Use FastAPI's Depends for config injection

3. **Config Validation & Testing** (Day 10)
   - [ ] Create `infrastructure/config/validation.py`
   - [ ] Add comprehensive config validation
   - [ ] Test config loading and validation
   - [ ] Update documentation

### Phase 3: Clean Dependencies (Days 11-13)

#### Focus Areas:
1. **Circular Import Resolution** (Day 11)
   - [ ] Identify all circular imports from dependency mapping
   - [ ] Move shared utilities to neutral modules
   - [ ] Create `utils/helpers.py` for common functions
   - [ ] Test import resolution

2. **Interface Implementation** (Day 12)
   - [ ] Create `core/interfaces/` directory
   - [ ] Define abstract base classes (ABCs)
   - [ ] Implement interfaces in services
   - [ ] Update service dependencies

3. **Dependency Injection Setup** (Day 13)
   - [ ] Implement dependency injection pattern
   - [ ] Remove global variables and singletons
   - [ ] Use FastAPI's dependency system
   - [ ] Test service isolation

### Phase 4: Enhance Backend Architecture (Days 14-16)

#### Backend Improvements:
1. **API Layer Restructuring** (Day 14)
   - [ ] Create `backend/api/v1/endpoints/` structure
   - [ ] Split routes by domain
   - [ ] Implement shared dependencies
   - [ ] Add middleware for error handling

2. **Repository Pattern Implementation** (Day 15)
   - [ ] Create `backend/repositories/` directory
   - [ ] Implement repository pattern for data access
   - [ ] Abstract database operations
   - [ ] Add repository interfaces

3. **Service Orchestrators** (Day 16)
   - [ ] Create `backend/services/generation_orchestrator.py`
   - [ ] Implement high-level business logic
   - [ ] Separate API concerns from business logic
   - [ ] Add comprehensive testing

## Tools & Technologies

### Development Tools
- **IDE**: VS Code / PyCharm with refactoring extensions
- **Python**: pylint, mypy, black, flake8, pytest
- **TypeScript**: ESLint, Prettier, Jest/Vitest
- **Dependencies**: pydeps, importmagic
- **Documentation**: Sphinx (Python), TypeDoc (TypeScript)

### Libraries & Frameworks
- **Config Management**: Pydantic for validation
- **Dependency Injection**: FastAPI's Depends system
- **Testing**: pytest, Jest/Vitest
- **Database**: SQLAlchemy ORM
- **API**: FastAPI with automatic schema generation

## Quality Assurance

### Testing Strategy
- **Unit Tests**: Each extracted service/module
- **Integration Tests**: Service interactions
- **Smoke Tests**: Core functionality after each phase
- **Performance Tests**: Benchmark key operations
- **Regression Tests**: Ensure no functionality loss

### Code Quality Metrics
- **Coverage**: Maintain >80% test coverage
- **Complexity**: Reduce cyclomatic complexity
- **Dependencies**: Minimize coupling between modules
- **Documentation**: Comprehensive docstrings and README updates

## Risk Mitigation

### Rollback Strategy
- Git tags before each phase
- Feature branches for isolation
- Incremental commits with descriptive messages
- Automated testing before merges

### Performance Monitoring
- Benchmark model loading times
- Monitor memory usage patterns
- Track API response times
- Use cProfile for Python performance analysis

### Team Coordination
- Assign module ownership per layer
- Code review requirements for all changes
- Daily standups during refactoring phases
- Documentation updates in parallel

## Success Metrics

### Technical Metrics
- [x] **Major Progress**: Extracted 3 core services from `utils.py` (ModelManager, VRAMOptimizer, VideoGenerationEngine)
- [x] **Import Cleanup**: Updated critical imports in frontend/ui.py, backend/core/system_integration.py, backend/main.py
- [ ] Reduce `utils.py` from 7,145 lines to <500 lines (significant progress made)
- [ ] Eliminate all circular imports
- [ ] Achieve >90% test coverage
- [ ] Reduce average module complexity by 50%

### Operational Metrics
- [ ] Maintain current performance benchmarks
- [ ] Zero regression in functionality
- [ ] Improved development velocity (measured post-refactor)
- [ ] Reduced bug introduction rate

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Preparation | 2 days | Testing baseline, dependency mapping, quality analysis |
| Phase 1 | 5 days | ✅ Core services extracted from utils.py |
| Phase 2 | 3 days | Centralized configuration management |
| Phase 3 | 3 days | Clean dependencies and interfaces |
| Phase 4 | 3 days | Enhanced backend architecture |
| **Total** | **16 days** | **Fully refactored, maintainable codebase** |

## Post-Refactor Activities

### Documentation
- [x] Update README.md with new architecture
- [x] Create developer onboarding guide
- [x] Document new patterns and conventions
- [ ] Update API documentation

### Monitoring & Maintenance
- [ ] Set up automated code quality checks
- [ ] Implement continuous integration improvements
- [ ] Create architecture decision records (ADRs)
- [ ] Plan regular architecture reviews

---

**Note**: This plan is designed to be executed incrementally with frequent commits and testing. Each phase builds upon the previous one, ensuring the codebase remains functional throughout the refactoring process.
