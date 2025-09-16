---
title: tools.quality-monitor.models
category: api
tags: [api, tools]
---

# tools.quality-monitor.models

Quality monitoring data models and types.

## Classes

### AlertSeverity

Alert severity levels.

### MetricType

Types of quality metrics.

### TrendDirection

Trend direction for metrics.

### QualityMetric

Individual quality metric data point.

#### Methods

##### to_dict(self: Any) -> <ast.Subscript object at 0x000001942F3BE740>

Convert to dictionary for serialization.

### QualityTrend

Quality trend analysis for a metric.

#### Methods

##### to_dict(self: Any) -> <ast.Subscript object at 0x000001942F3BCD60>

Convert to dictionary for serialization.

### QualityAlert

Quality alert for regressions or maintenance needs.

#### Methods

##### to_dict(self: Any) -> <ast.Subscript object at 0x000001942F3BCEB0>

Convert to dictionary for serialization.

### QualityThreshold

Quality threshold configuration.

#### Methods

##### to_dict(self: Any) -> <ast.Subscript object at 0x000001942F3BCD00>

Convert to dictionary for serialization.

### QualityRecommendation

Automated quality improvement recommendation.

#### Methods

##### to_dict(self: Any) -> <ast.Subscript object at 0x00000194318D6D10>

Convert to dictionary for serialization.

### QualityDashboard

Quality monitoring dashboard data.

#### Methods

##### to_dict(self: Any) -> <ast.Subscript object at 0x00000194318DE5F0>

Convert to dictionary for serialization.

##### to_json(self: Any) -> str

Convert to JSON string.

## Constants

### LOW

Type: `str`

Value: `low`

### MEDIUM

Type: `str`

Value: `medium`

### HIGH

Type: `str`

Value: `high`

### CRITICAL

Type: `str`

Value: `critical`

### TEST_COVERAGE

Type: `str`

Value: `test_coverage`

### CODE_COMPLEXITY

Type: `str`

Value: `code_complexity`

### DOCUMENTATION_COVERAGE

Type: `str`

Value: `documentation_coverage`

### DUPLICATE_CODE

Type: `str`

Value: `duplicate_code`

### DEAD_CODE

Type: `str`

Value: `dead_code`

### STYLE_VIOLATIONS

Type: `str`

Value: `style_violations`

### TYPE_HINT_COVERAGE

Type: `str`

Value: `type_hint_coverage`

### PERFORMANCE

Type: `str`

Value: `performance`

### IMPROVING

Type: `str`

Value: `improving`

### STABLE

Type: `str`

Value: `stable`

### DEGRADING

Type: `str`

Value: `degrading`

### UNKNOWN

Type: `str`

Value: `unknown`

