---
title: tools.code-review.technical_debt_tracker
category: api
tags: [api, tools]
---

# tools.code-review.technical_debt_tracker

Technical Debt Tracking and Prioritization System

This module provides comprehensive technical debt tracking, analysis, and prioritization
to help teams manage and reduce technical debt systematically.

## Classes

### DebtCategory

Categories of technical debt

### DebtSeverity

Severity levels for technical debt

### DebtStatus

Status of technical debt items

### TechnicalDebtItem

Represents a technical debt item

#### Methods

##### __post_init__(self: Any)



### DebtMetrics

Technical debt metrics

### DebtRecommendation

Recommendation for addressing technical debt

### TechnicalDebtTracker

Main technical debt tracking system

#### Methods

##### __init__(self: Any, project_root: str, db_path: str)



##### _init_database(self: Any)

Initialize SQLite database for debt tracking

##### _load_debt_items(self: Any)

Load debt items from database

##### add_debt_item(self: Any, item: TechnicalDebtItem) -> str

Add a new technical debt item

##### update_debt_item(self: Any, item_id: str, updates: <ast.Subscript object at 0x000001942C67E740>) -> bool

Update an existing debt item

##### resolve_debt_item(self: Any, item_id: str, resolution_notes: str) -> bool

Mark a debt item as resolved

##### get_debt_item(self: Any, item_id: str) -> <ast.Subscript object at 0x000001942C6DA4D0>

Get a debt item by ID

##### get_debt_items_by_file(self: Any, file_path: str) -> <ast.Subscript object at 0x000001942C6DA0E0>

Get all debt items for a specific file

##### get_debt_items_by_category(self: Any, category: DebtCategory) -> <ast.Subscript object at 0x000001942C6D9CF0>

Get all debt items in a specific category

##### get_debt_items_by_severity(self: Any, severity: DebtSeverity) -> <ast.Subscript object at 0x000001942C6D9900>

Get all debt items with specific severity

##### get_prioritized_debt_items(self: Any, limit: int) -> <ast.Subscript object at 0x000001942C6D92D0>

Get debt items sorted by priority score

##### calculate_debt_metrics(self: Any) -> DebtMetrics

Calculate comprehensive debt metrics

##### generate_recommendations(self: Any) -> <ast.Subscript object at 0x0000019428DDF130>

Generate recommendations for addressing technical debt

##### _generate_debt_id(self: Any, item: TechnicalDebtItem) -> str

Generate unique ID for debt item

##### _calculate_priority_score(self: Any, item: TechnicalDebtItem) -> float

Calculate priority score for debt item

##### _save_debt_item(self: Any, item: TechnicalDebtItem)

Save debt item to database

##### _log_debt_action(self: Any, debt_item_id: str, action: str, old_status: str, new_status: str, notes: str)

Log debt item action to history

##### _analyze_debt_trend(self: Any) -> str

Analyze debt trend over time

##### export_debt_report(self: Any, output_path: str) -> <ast.Subscript object at 0x00000194285C57E0>

Export comprehensive debt report

## Constants

### CODE_QUALITY

Type: `str`

Value: `code_quality`

### ARCHITECTURE

Type: `str`

Value: `architecture`

### DOCUMENTATION

Type: `str`

Value: `documentation`

### TESTING

Type: `str`

Value: `testing`

### PERFORMANCE

Type: `str`

Value: `performance`

### SECURITY

Type: `str`

Value: `security`

### MAINTAINABILITY

Type: `str`

Value: `maintainability`

### SCALABILITY

Type: `str`

Value: `scalability`

### CRITICAL

Type: `str`

Value: `critical`

### HIGH

Type: `str`

Value: `high`

### MEDIUM

Type: `str`

Value: `medium`

### LOW

Type: `str`

Value: `low`

### IDENTIFIED

Type: `str`

Value: `identified`

### ACKNOWLEDGED

Type: `str`

Value: `acknowledged`

### PLANNED

Type: `str`

Value: `planned`

### IN_PROGRESS

Type: `str`

Value: `in_progress`

### RESOLVED

Type: `str`

Value: `resolved`

### DEFERRED

Type: `str`

Value: `deferred`

### WONT_FIX

Type: `str`

Value: `wont_fix`

