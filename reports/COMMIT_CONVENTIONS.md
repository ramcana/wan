# Git Commit Message Conventions for WAN2.2

## Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

## Types
- **feat**: New feature
- **fix**: Bug fix
- **refactor**: Code refactoring without functionality change
- **move**: File/directory reorganization
- **config**: Configuration changes
- **test**: Adding or updating tests
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **perf**: Performance improvements
- **chore**: Maintenance tasks

## Scopes
- **core**: Core services and models
- **backend**: FastAPI backend
- **frontend**: React frontend
- **infra**: Infrastructure layer
- **utils**: Utility functions
- **config**: Configuration management
- **tests**: Test-related changes

## Examples
```
feat(core): extract ModelManager from utils.py

- Create core/services/model_manager.py
- Move ModelManager class with 500+ lines
- Update imports across 15 files
- Add interface for dependency injection

Closes #123
```

```
move(backend): reorganize API routes to v1/endpoints

- Move api/routes/*.py to api/v1/endpoints/
- Update import statements
- Maintain backward compatibility

Part of functional organization phase 1
```

```
refactor(infra): centralize configuration management

- Create infrastructure/config/config_manager.py
- Consolidate scattered config logic
- Add Pydantic validation
- Remove duplicate config handling

Breaking change: config imports updated
```

## Rules
1. Use present tense ("add" not "added")
2. Keep subject line under 50 characters
3. Capitalize subject line
4. No period at end of subject
5. Use body to explain what and why, not how
6. Reference issues/PRs in footer
