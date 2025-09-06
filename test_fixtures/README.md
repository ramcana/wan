# Test Fixtures

This directory contains purpose-built sample projects for testing our meta-tools.

## Structure

- `project_with_broken_imports/` - Sample project with import issues
- `project_with_duplicate_files/` - Sample project with duplicate code
- `project_with_flaky_tests/` - Sample project with unreliable tests
- `project_with_config_issues/` - Sample project with configuration problems
- `project_with_quality_issues/` - Sample project with code quality problems

## Usage

These fixtures are used by our E2E testing framework to ensure our tools work correctly without risking the main codebase.

Each fixture should:

1. Be minimal but representative
2. Have a clear problem that our tools should fix
3. Include expected outcomes after tool application
4. Be completely isolated from the main project
