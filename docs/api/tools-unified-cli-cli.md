---
title: tools.unified-cli.cli
category: api
tags: [api, tools]
---

# tools.unified-cli.cli



## Classes

### WorkflowContext

Development workflow contexts that determine which tools to run

### ToolResult

Result from running a tool

### WorkflowConfig

Configuration for workflow automation

### TeamCollaborationConfig

Configuration for team collaboration features

### MockTool

Mock tool for tools that aren't implemented yet

#### Methods

##### __init__(self: Any, name: str)



##### run(self: Any, args: <ast.Subscript object at 0x00000194289B32B0>) -> <ast.Subscript object at 0x00000194289B02E0>

Mock run method

### UnifiedCLI

Unified CLI for all project cleanup and quality tools

#### Methods

##### __init__(self: Any)



##### load_team_config(self: Any) -> <ast.Subscript object at 0x000001942A2E66E0>

Load team collaboration configuration

##### save_team_config(self: Any, config: TeamCollaborationConfig)

Save team collaboration configuration

##### setup_team_collaboration(self: Any, team_name: str, standards: <ast.Subscript object at 0x000001942A2E6890>)

Set up team collaboration features

##### check_quality_gates(self: Any, gate_type: str) -> bool

Check if quality gates are met for a specific context

##### notify_team(self: Any, message: str, level: str)

Send notification to team channels

##### generate_team_report(self: Any) -> <ast.Subscript object at 0x0000019428471A50>

Generate a comprehensive team report

##### generate_team_recommendations(self: Any, report: <ast.Subscript object at 0x00000194284735B0>) -> <ast.Subscript object at 0x000001942CE20910>

Generate recommendations for the team based on project state

##### share_standards(self: Any, standards_file: str)

Share team standards with other team members

##### import_standards(self: Any, standards_file: str)

Import team standards from a shared file

##### get_context_from_git_status(self: Any) -> WorkflowContext

Determine workflow context based on git status and environment

##### print_results(self: Any, results: <ast.Subscript object at 0x000001942CE2A0E0>)

Print workflow results in a formatted way

##### list_tools(self: Any)

List all available tools

##### list_workflows(self: Any)

List all available workflow contexts

## Constants

### PRE_COMMIT

Type: `str`

Value: `pre-commit`

### POST_COMMIT

Type: `str`

Value: `post-commit`

### DAILY_MAINTENANCE

Type: `str`

Value: `daily-maintenance`

### WEEKLY_CLEANUP

Type: `str`

Value: `weekly-cleanup`

### RELEASE_PREP

Type: `str`

Value: `release-prep`

### ONBOARDING

Type: `str`

Value: `onboarding`

### DEBUGGING

Type: `str`

Value: `debugging`

### CUSTOM

Type: `str`

Value: `custom`

