---
title: scripts.performance-validation
category: api
tags: [api, scripts]
---

# scripts.performance-validation



## Classes

### PerformanceValidator

Main performance validation orchestrator

#### Methods

##### __init__(self: Any, config_path: str)



##### _load_config(self: Any, config_path: str) -> <ast.Subscript object at 0x000001942750A500>

Load performance validation configuration

##### _analyze_bundle_size(self: Any) -> <ast.Subscript object at 0x00000194275CD750>

Analyze frontend bundle size

##### validate_deployment_readiness(self: Any) -> bool

Validate if system is ready for deployment based on performance results

##### generate_deployment_report(self: Any) -> str

Generate comprehensive deployment readiness report

