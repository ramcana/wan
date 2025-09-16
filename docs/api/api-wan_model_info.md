---
title: api.wan_model_info
category: api
tags: [api, api]
---

# api.wan_model_info

WAN Model Information and Capabilities API
Provides comprehensive WAN model information, capabilities, health monitoring,
performance metrics, comparison system, and dashboard integration.

## Classes

### WANModelCapabilities

WAN model capabilities information

### WANModelHealthMetrics

WAN model health monitoring metrics

### WANModelPerformanceMetrics

WAN model performance metrics

### WANModelComparison

WAN model comparison data

### WANModelRecommendation

WAN model recommendation

### WANModelInfoAPI

WAN Model Information and Capabilities API implementation

#### Methods

##### __init__(self: Any)



##### _get_fallback_model_config(self: Any, model_type: str) -> <ast.Subscript object at 0x000001942F79B580>

Get fallback model configuration

##### _get_fallback_model_info(self: Any, model_type: str) -> <ast.Subscript object at 0x000001942F79AF20>

Get fallback model information

##### _determine_model_capabilities(self: Any, model_type: str, config: <ast.Subscript object at 0x000001942F79ABC0>, info: <ast.Subscript object at 0x000001942F79AD70>) -> <ast.Subscript object at 0x000001942F7746A0>

Determine model capabilities based on type and configuration

##### _generate_comparison_recommendation(self: Any, model_a: str, model_b: str, perf_a: WANModelPerformanceMetrics, perf_b: WANModelPerformanceMetrics) -> str

Generate comparison recommendation

##### _define_model_use_cases(self: Any, model_a: str, model_b: str, perf_a: WANModelPerformanceMetrics, perf_b: WANModelPerformanceMetrics) -> <ast.Subscript object at 0x000001942FD02FB0>

Define use cases for each model

##### _identify_trade_offs(self: Any, model_a: str, model_b: str, perf_a: WANModelPerformanceMetrics, perf_b: WANModelPerformanceMetrics) -> <ast.Subscript object at 0x00000194344D18A0>

Identify trade-offs between models

##### _calculate_use_case_bonus(self: Any, model_type: str, use_case: str) -> float

Calculate bonus score based on use case matching

## Constants

### WAN_MODELS_AVAILABLE

Type: `bool`

Value: `True`

### INFRASTRUCTURE_AVAILABLE

Type: `bool`

Value: `True`

### WAN_MODELS_AVAILABLE

Type: `bool`

Value: `False`

### INFRASTRUCTURE_AVAILABLE

Type: `bool`

Value: `False`

