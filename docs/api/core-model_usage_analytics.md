---
title: core.model_usage_analytics
category: api
tags: [api, core]
---

# core.model_usage_analytics

Model Usage Analytics System - Minimal Implementation
Tracks model usage patterns and provides basic recommendations.

## Classes

### UsageEventType

Types of usage events to track

### UsageData

Individual usage data point

### UsageStatistics

Usage statistics for a model

### CleanupRecommendation

Recommendation for model cleanup

### PreloadRecommendation

Recommendation for model preloading

### CleanupAction

Individual cleanup action

### CleanupRecommendations

Comprehensive cleanup recommendations

### ModelUsageEventDB

Database model for usage events

### ModelUsageAnalytics

Model Usage Analytics System

#### Methods

##### __init__(self: Any, models_dir: <ast.Subscript object at 0x000001942FD64700>)



## Constants

### GENERATION_REQUEST

Type: `str`

Value: `generation_request`

### GENERATION_COMPLETE

Type: `str`

Value: `generation_complete`

### GENERATION_FAILED

Type: `str`

Value: `generation_failed`

### MODEL_LOAD

Type: `str`

Value: `model_load`

