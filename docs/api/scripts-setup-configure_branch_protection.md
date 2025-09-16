---
title: scripts.setup.configure_branch_protection
category: api
tags: [api, scripts]
---

# scripts.setup.configure_branch_protection

Configure branch protection rules with health monitoring integration.

This script sets up branch protection rules that require health checks
to pass before allowing merges to protected branches.

## Classes

### BranchProtectionManager

Manages GitHub branch protection rules with health monitoring integration.

#### Methods

##### __init__(self: Any, repo_owner: str, repo_name: str, github_token: str)



##### get_current_protection(self: Any, branch: str) -> <ast.Subscript object at 0x0000019430306980>

Get current branch protection settings.

##### configure_health_monitoring_protection(self: Any, branch: str) -> Dict

Configure branch protection with health monitoring requirements.

##### apply_protection(self: Any, branch: str, config: Dict) -> bool

Apply branch protection configuration.

##### setup_health_monitoring_protection(self: Any, branches: <ast.Subscript object at 0x000001943035ED10>) -> bool

Set up health monitoring branch protection for specified branches.

##### create_health_check_ruleset(self: Any) -> bool

Create a repository ruleset for health monitoring.

