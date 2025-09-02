# WAN2.2 Codebase Organization Plan

## Current State Analysis

Based on the file structure and open files, I've identified several organizational issues that need addressing:

### 1. Configuration Fragmentation

- Multiple config files: `config.json`, `startup_config.json`, `backend/config.json`
- Duplicate settings across different config files
- No unified configuration schema

### 2. Root Directory Clutter

- Many summary/report files in root: `*_SUMMARY.md`, `*_FIX_SUMMARY.md`
- Temporary files: `temp.pt`, `temp_*.txt`
- Version files: `0.30.0`, `0.33.0`, `4.44.0`
- Test files mixed with main code: `test_*.py`

### 3. Documentation Scattered

- Multiple README files in different locations
- Documentation mixed with code
- No clear documentation hierarchy

### 4. Tool Import Issues

- Many tools have import errors
- Inconsistent module structure
- Missing `__init__.py` files

## Proposed Organization Structure

```
wan2.2/
├── src/                           # Main source code
│   ├── backend/                   # FastAPI backend (existing)
│   ├── frontend/                  # React frontend (existing)
│   ├── core/                      # Core domain logic (existing)
│   └── infrastructure/            # Infrastructure layer (existing)
├── tools/                         # Development tools (existing, needs fixing)
├── tests/                         # All test files (existing, needs organization)
├── docs/                          # All documentation (existing, needs organization)
├── config/                        # Unified configuration
├── scripts/                       # Build and deployment scripts (existing)
├── data/                          # Data files and models (existing)
├── reports/                       # Generated reports and summaries
├── temp/                          # Temporary files
└── archive/                       # Archived/deprecated files
```

## Implementation Steps

### Phase 1: Immediate Cleanup (High Priority)

1. Move all summary/report files to `reports/` directory
2. Move temporary files to `temp/` directory
3. Archive version files to `archive/`
4. Move root-level test files to appropriate test directories

### Phase 2: Configuration Consolidation

1. Create unified configuration schema
2. Merge duplicate configuration settings
3. Implement configuration validation
4. Update all components to use unified config

### Phase 3: Tool Infrastructure Repair

1. Fix import issues in tools
2. Add missing `__init__.py` files
3. Standardize tool interfaces
4. Create tool integration tests

### Phase 4: Documentation Organization

1. Consolidate all documentation in `docs/`
2. Create clear documentation hierarchy
3. Generate navigation and search indices
4. Validate all documentation links

### Phase 5: Test Organization

1. Organize tests by component
2. Fix broken test imports
3. Implement test isolation
4. Create comprehensive test suite

## Benefits of This Organization

1. **Cleaner Root Directory**: Only essential files at root level
2. **Logical Grouping**: Related files grouped together
3. **Better Navigation**: Clear hierarchy for developers
4. **Improved Maintainability**: Easier to find and update files
5. **Enhanced Development Experience**: Better IDE support and tooling

## Risk Mitigation

1. **Backup Strategy**: Create full backup before any moves
2. **Incremental Changes**: Implement changes in phases
3. **Import Updates**: Automatically update all import statements
4. **Testing**: Validate functionality after each phase
5. **Rollback Plan**: Ability to revert changes if issues arise
