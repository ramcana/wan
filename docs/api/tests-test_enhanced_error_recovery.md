---
title: tests.test_enhanced_error_recovery
category: api
tags: [api, tests]
---

# tests.test_enhanced_error_recovery

Tests for Enhanced Error Recovery System

This module tests the sophisticated error categorization, multi-strategy recovery,
intelligent fallback integration, automatic repair triggers, and user-friendly
error messages with actionable recovery steps.

## Classes

### TestEnhancedErrorRecovery

Test suite for EnhancedErrorRecovery class

#### Methods

##### mock_base_recovery(self: Any)

Create mock base recovery system

##### mock_availability_manager(self: Any)

Create mock model availability manager

##### mock_fallback_manager(self: Any)

Create mock intelligent fallback manager

##### mock_health_monitor(self: Any)

Create mock model health monitor

##### mock_enhanced_downloader(self: Any)

Create mock enhanced model downloader

##### mock_websocket_manager(self: Any)

Create mock WebSocket manager

##### enhanced_recovery(self: Any, mock_base_recovery: Any, mock_availability_manager: Any, mock_fallback_manager: Any, mock_health_monitor: Any, mock_enhanced_downloader: Any, mock_websocket_manager: Any)

Create enhanced error recovery system with mocked dependencies

##### test_initialization(self: Any, enhanced_recovery: Any)

Test enhanced error recovery system initialization

##### test_strategy_mapping_completeness(self: Any, enhanced_recovery: Any)

Test that all enhanced failure types have recovery strategies

##### test_error_messages_completeness(self: Any, enhanced_recovery: Any)

Test that error messages are properly configured

##### test_convert_to_base_failure_type(self: Any, enhanced_recovery: Any)

Test conversion from enhanced to base failure types

##### test_update_strategy_success_rate(self: Any, enhanced_recovery: Any)

Test strategy success rate tracking

##### test_reset_metrics(self: Any, enhanced_recovery: Any)

Test metrics reset functionality

### TestErrorContextCreation

Test error context creation and categorization

#### Methods

##### enhanced_recovery(self: Any)

Create minimal enhanced recovery for testing

### TestConvenienceFunction

Test the convenience function for creating enhanced error recovery

#### Methods

##### test_create_enhanced_error_recovery_minimal(self: Any)

Test creating enhanced error recovery with minimal parameters

##### test_create_enhanced_error_recovery_with_generation_service(self: Any)

Test creating enhanced error recovery with generation service

##### test_create_enhanced_error_recovery_with_all_components(self: Any)

Test creating enhanced error recovery with all components

### TestIntegrationScenarios

Test integration scenarios with various failure types

#### Methods

##### full_recovery_system(self: Any)

Create a full recovery system with all mocked components

