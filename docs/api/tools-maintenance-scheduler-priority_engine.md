---
title: tools.maintenance-scheduler.priority_engine
category: api
tags: [api, tools]
---

# tools.maintenance-scheduler.priority_engine

Priority engine for maintenance tasks based on impact and effort analysis.

## Classes

### ImpactAnalysis

Analysis of task impact on project health.

### EffortAnalysis

Analysis of effort required to complete a task.

#### Methods

##### __post_init__(self: Any)



### TaskPriorityEngine

Engine for calculating task priorities based on impact and effort analysis.

Uses a sophisticated scoring system that considers:
- Business impact (security, performance, quality)
- Technical debt reduction
- Effort required
- Historical success rates
- Dependencies and blocking relationships

#### Methods

##### __init__(self: Any, config: <ast.Subscript object at 0x000001942C6634F0>)



##### get_priority_score(self: Any, task: MaintenanceTask, history: <ast.Subscript object at 0x000001942C6633A0>) -> float

Calculate comprehensive priority score for a task.

Returns a score from 0-100 where higher scores indicate higher priority.

##### analyze_impact(self: Any, task: MaintenanceTask) -> ImpactAnalysis

Analyze the potential impact of completing a task.

##### analyze_effort(self: Any, task: MaintenanceTask, history: <ast.Subscript object at 0x000001942C6D2380>) -> EffortAnalysis

Analyze the effort required to complete a task.

##### calculate_urgency(self: Any, task: MaintenanceTask) -> float

Calculate urgency score based on timing and dependencies.

##### get_recommended_execution_order(self: Any, tasks: <ast.Subscript object at 0x000001942CCB60E0>, history_map: <ast.Subscript object at 0x000001942CCB6920>) -> <ast.Subscript object at 0x000001942CCB40A0>

Get tasks in recommended execution order based on priority scores.

##### _calculate_effort_score(self: Any, effort_analysis: EffortAnalysis) -> float

Convert effort analysis to a score (higher score = less effort).

##### _get_priority_adjustment(self: Any, priority: TaskPriority) -> float

Get priority level adjustment multiplier.

##### _analyze_config_impact(self: Any, config: Dict) -> <ast.Subscript object at 0x000001942CCF7490>

Analyze task configuration to determine impact adjustments.

