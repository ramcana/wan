---
title: tests.test_hardware_optimization_integration
category: api
tags: [api, tests]
---

# tests.test_hardware_optimization_integration

Test hardware optimization integration with generation service

## Classes

### TestHardwareOptimizationIntegration

Test hardware optimization integration with generation service

#### Methods

##### mock_hardware_profile(self: Any)

Mock hardware profile for testing

##### mock_system_optimizer(self: Any, mock_hardware_profile: Any)

Mock WAN22SystemOptimizer for testing

##### generation_service(self: Any)

Create generation service for testing

##### test_vram_monitor_creation(self: Any, mock_hardware_profile: Any)

Test VRAM monitor creation and functionality

##### test_vram_availability_check(self: Any)

Test VRAM availability checking

##### test_vram_optimization_suggestions(self: Any)

Test VRAM optimization suggestions

##### test_vram_requirements_estimation(self: Any, generation_service: Any)

Test VRAM requirements estimation

##### test_queue_status_with_hardware_optimization(self: Any, generation_service: Any, mock_system_optimizer: Any)

Test queue status includes hardware optimization information

##### test_generation_stats_with_hardware_optimization(self: Any, generation_service: Any, mock_system_optimizer: Any)

Test generation stats include hardware optimization information

