---
title: tools.onboarding.setup_wizard
category: api
tags: [api, tools]
---

# tools.onboarding.setup_wizard

Automated Setup Wizard

This module provides an interactive setup wizard for new developers,
combining environment setup, dependency installation, and progress tracking.

## Classes

### SetupWizard

Interactive setup wizard for new developers

#### Methods

##### __init__(self: Any, project_root: <ast.Subscript object at 0x000001942CAF99C0>, developer_name: <ast.Subscript object at 0x000001942CAF9900>)



##### run_wizard(self: Any) -> bool

Run the complete setup wizard

##### _initialize_components(self: Any)

Initialize setup components

##### _print_welcome(self: Any)

Print welcome message

##### _get_developer_info(self: Any) -> bool

Get developer information

##### _step_prerequisites(self: Any) -> bool

Step 1: Check prerequisites

##### _step_environment_validation(self: Any) -> bool

Step 2: Environment validation

##### _step_project_setup(self: Any) -> bool

Step 3: Project setup

##### _step_dependency_installation(self: Any) -> bool

Step 4: Dependency installation

##### _step_development_tools(self: Any) -> bool

Step 5: Development tools setup

##### _step_testing_validation(self: Any) -> bool

Step 6: Testing and validation

##### _step_documentation(self: Any) -> bool

Step 7: Documentation and learning

##### _step_first_steps(self: Any) -> bool

Step 8: First steps and next actions

##### _step_completion(self: Any) -> bool

Step 9: Completion and summary

##### _test_server_startup(self: Any) -> bool

Test if development servers can start

##### _update_checklist_from_validation(self: Any, health: Any)

Update checklist based on validation results

##### _print_step_header(self: Any, step_name: str)

Print step header

##### _print_success(self: Any, message: str)

Print success message

##### _print_info(self: Any, message: str)

Print info message

##### _print_warning(self: Any, message: str)

Print warning message

##### _print_error(self: Any, message: str)

Print error message

##### _ask_yes_no(self: Any, question: str, default: bool) -> bool

Ask yes/no question

##### _ask_continue(self: Any) -> bool

Ask if user wants to continue after error

