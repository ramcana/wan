---
title: API Reference
category: reference
tags: [api, reference]
---

# API Reference

Auto-generated API documentation for the WAN22 Video Generation System.

## Modules


### app

WAN22 FastAPI Backend
Main FastAPI application entry point

#### Classes

- **PromptRequest**: ...
- **GenerationRequest**: ...
- **ConnectionManager**: ...

### demo_cors_validation

Demo script to test CORS validation functionality


### diagnose_system


### final_system_check

Final comprehensive system check for real AI generation


### fix_imports

Quick fix script to ensure all import paths are correct for running from backend directory


### fix_vram_validation

VRAM Validation Fix for RTX 4080
Ensures proper VRAM detection and validation


### init_db

Initialize the database with required tables


### main

Wan2.2 UI Variant - Main Application Entry Point
Handles configuration loading, command-line arguments, and application lifecycle

#### Classes

- **ApplicationConfig**: Manages application configuration loading and validation...
- **ApplicationManager**: Manages the application lifecycle and cleanup...
- **GenerationErrorHandler**: ...
- **WAN22SystemOptimizer**: ...

### optimize_for_rtx4080

RTX 4080 Optimization Script
Optimizes Wan2.2 for RTX 4080 16GB VRAM with Threadripper PRO 5995WX


### start_full_stack

Full Stack Startup Script for Wan2.2 React + FastAPI
Starts both backend and frontend services for development testing

#### Classes

- **FullStackRunner**: ...

### start_server

Simple script to start the FastAPI backend server


### test_cuda_detection

Test CUDA detection to debug the issue


### test_enhanced_endpoints_simple

Simple test to verify enhanced model management endpoints are working


### test_json_request

Test JSON request handling for generation endpoint


### test_real_ai_ready

Test if the system is ready for real AI generation


### __init__


### __main__

Backend module entry point for running as: python -m backend


### api.deployment_health

Deployment Health Check API Endpoints

This module provides health check endpoints for validating the deployment
of the enhanced model availability system.

#### Classes

- **HealthStatus**: Health status levels...
- **ComponentStatus**: Status of a system component...
- **SystemHealthResponse**: System health check response...
- **DeploymentValidationResponse**: Deployment validation response...
- **PerformanceMetrics**: Performance metrics response...

### api.enhanced_generation

Enhanced Generation API for Phase 1 MVP
Provides seamless T2V, I2V, and TI2V generation with auto-detection and model switching

#### Classes

- **GenerationRequest**: Enhanced generation request with auto-detection capabilities...
- **GenerationResponse**: Enhanced generation response...
- **ModelDetectionService**: Service for auto-detecting optimal model type based on inputs...
- **PromptEnhancementService**: Service for enhancing prompts for better generation results...
- **GenerationParams**: ...
- **ModelType**: ...

### api.enhanced_model_configuration

Enhanced Model Configuration API Endpoints

Provides REST API endpoints for managing enhanced model availability configuration,
including user preferences, admin policies, and feature flags.


### api.enhanced_model_management

Enhanced Model Management API Endpoints
Provides comprehensive model status, download management, health monitoring,
analytics, cleanup, and fallback suggestion endpoints.

#### Classes

- **DownloadControlRequest**: Request model for download control operations...
- **CleanupRequest**: Request model for storage cleanup operations...
- **FallbackSuggestionRequest**: Request model for fallback suggestions...
- **EnhancedModelManagementAPI**: Enhanced model management API implementation...

### api.fallback_recovery


### api.model_management

Enhanced Model Management API
Provides model status and management endpoints using SystemIntegration and ConfigurationBridge

#### Classes

- **ModelManagementAPI**: Enhanced model management API using existing infrastructure...

### api.performance

Performance monitoring API endpoints.
Provides access to performance metrics, analysis, and optimization recommendations.


### api.performance_dashboard

Performance Dashboard API Endpoints

Provides REST API endpoints for accessing performance monitoring data
and dashboard visualization support.


### api.wan_model_dashboard

WAN Model Dashboard Integration
Provides dashboard-specific endpoints and real-time data for WAN model monitoring

#### Classes

- **DashboardMetrics**: Dashboard metrics summary...
- **ModelStatusSummary**: Model status summary for dashboard...
- **SystemAlert**: System alert for dashboard...
- **WANModelDashboard**: WAN Model Dashboard Integration...

### api.wan_model_info

WAN Model Information and Capabilities API
Provides comprehensive WAN model information, capabilities, health monitoring,
performance metrics, comparison system, and dashboard integration.

#### Classes

- **WANModelCapabilities**: WAN model capabilities information...
- **WANModelHealthMetrics**: WAN model health monitoring metrics...
- **WANModelPerformanceMetrics**: WAN model performance metrics...
- **WANModelComparison**: WAN model comparison data...
- **WANModelRecommendation**: WAN model recommendation...
- **WANModelInfoAPI**: WAN Model Information and Capabilities API implementation...

### api.__init__


### api.middleware.__init__

API middleware


### api.routes.analytics

#### Classes

- **PerformanceMetric**: ...
- **ErrorReport**: ...
- **UserJourneyEvent**: ...
- **AnalyticsQuery**: ...

### api.routes.__init__


### api.v1.__init__

API v1 endpoints


### api.v1.endpoints.generation

Generation API endpoints


### api.v1.endpoints.lora

LoRA management API endpoints


### api.v1.endpoints.models

Model management endpoints
Provides model status, download progress, and integrity verification

#### Classes

- **ModelStatusResponse**: Response model for model status...
- **ModelDownloadRequest**: Request model for model download...
- **ModelDownloadProgressResponse**: Response model for download progress...
- **ModelIntegrityResponse**: Response model for model integrity check...
- **ModelHealthInfo**: Enhanced health information for a single model...
- **OrchestratorHealthResponse**: Enhanced health response for the model orchestrator...

### api.v1.endpoints.optimization

Advanced optimization presets and recommendations API


### api.v1.endpoints.outputs

Outputs management API endpoints


### api.v1.endpoints.prompt

Prompt enhancement API endpoints
Updated version - git operations working correctly

#### Classes

- **PromptEnhanceRequest**: Request model for prompt enhancement...
- **PromptEnhanceResponse**: Response model for prompt enhancement...
- **PromptPreviewRequest**: Request model for prompt enhancement preview...
- **PromptPreviewResponse**: Response model for prompt enhancement preview...
- **PromptValidationRequest**: Request model for prompt validation...
- **PromptValidationResponse**: Response model for prompt validation...

### api.v1.endpoints.queue

Queue management API endpoints


### api.v1.endpoints.system

System monitoring and optimization API endpoints
Fixed version - git operations working correctly


### api.v1.endpoints.test_models_health

Tests for Model Health API endpoints

#### Classes

- **TestModelsHealthEndpoints**: Test cases for model health API endpoints....

### api.v1.endpoints.websocket

WebSocket endpoints for real-time updates


### api.v1.endpoints.__init__

API v1 endpoints


### api.v2.router


### api.v2.__init__


### config.config_validator

Configuration validation for ensuring existing config.json works with new system.

#### Classes

- **ConfigValidationResult**: Result of configuration validation....
- **ConfigValidator**: Validates and migrates existing config.json for new system....

### config.environment

Environment-specific configuration management.

#### Classes

- **Environment**: Application environments....
- **EnvironmentConfig**: Manages environment-specific configuration....

### config.__init__

Configuration validation and migration utilities.


### core.configuration_bridge

Configuration Bridge for FastAPI backend integration

#### Classes

- **ConfigurationBridge**: Configuration adapter for existing config.json structure with FastAPI...

### core.config_validation

Configuration Validation System for Enhanced Model Management

Provides comprehensive validation for configuration settings, including
business rule validation, constraint checking, and migration support.

#### Classes

- **ValidationError**: Represents a configuration validation error...
- **ValidationResult**: Result of configuration validation...
- **ConfigurationValidator**: Validates enhanced model configuration settings...

### core.cors_validator

CORS Configuration Validator
Validates and manages CORS settings for the WAN22 FastAPI backend

#### Classes

- **CORSValidator**: CORS configuration validator and manager...

### core.enhanced_error_recovery

#### Classes

- **EnhancedFailureType**: Enhanced failure types with more granular categorization...
- **RecoveryStrategy**: Enhanced recovery strategies...
- **ErrorSeverity**: Error severity levels...
- **ErrorContext**: Enhanced error context with detailed information...
- **RecoveryResult**: Enhanced recovery result with detailed information...
- **RecoveryMetrics**: Metrics for recovery success tracking and optimization...
- **EnhancedErrorRecovery**: Enhanced Error Recovery System that extends FallbackRecoverySystem
with sophisticated error categori...

### core.enhanced_model_config

Enhanced Model Configuration Management System

This module provides comprehensive configuration management for enhanced model availability features,
including user preferences, admin controls, feature flags, and runtime configuration updates.

#### Classes

- **AutomationLevel**: Automation levels for model management features...
- **FeatureFlag**: Feature flags for gradual rollout...
- **DownloadConfig**: Configuration for enhanced download features...
- **HealthMonitoringConfig**: Configuration for model health monitoring...
- **FallbackConfig**: Configuration for intelligent fallback system...
- **AnalyticsConfig**: Configuration for usage analytics...
- **UpdateConfig**: Configuration for model update management...
- **NotificationConfig**: Configuration for real-time notifications...
- **StorageConfig**: Configuration for storage management...
- **UserPreferences**: User-specific preferences for model management...
- **AdminPolicies**: System-wide administrative policies...
- **FeatureFlagConfig**: Feature flag configuration for gradual rollout...
- **EnhancedModelConfiguration**: Complete configuration for enhanced model availability system...
- **ConfigurationManager**: Manages enhanced model configuration with validation and runtime updates...

### core.enhanced_model_downloader

Enhanced Model Downloader with Retry Logic
Provides intelligent retry mechanisms, exponential backoff, partial download recovery,
and advanced download management features for WAN2.2 models.

#### Classes

- **DownloadStatus**: Download status enumeration...
- **DownloadError**: Custom exception for download errors...
- **DownloadProgress**: Enhanced download progress tracking...
- **DownloadResult**: Result of a download operation...
- **RetryConfig**: Configuration for retry logic...
- **BandwidthConfig**: Configuration for bandwidth management...
- **EnhancedModelDownloader**: Enhanced model downloader with intelligent retry mechanisms,
exponential backoff, partial download r...

### core.fallback_recovery_system

Fallback and Recovery System for Real AI Model Integration

This module implements comprehensive fallback and recovery mechanisms that automatically
handle failures in model loading, generation pipeline, and system optimization.

#### Classes

- **RecoveryAction**: Types of recovery actions that can be performed...
- **FailureType**: Types of failures that can trigger recovery...
- **RecoveryAttempt**: Information about a recovery attempt...
- **SystemHealthStatus**: Current system health status...
- **FallbackRecoverySystem**: Comprehensive fallback and recovery system that handles various failure scenarios
and automatically ...

### core.integrated_error_handler

Integrated Error Handler for Real AI Model Integration

This module provides enhanced error handling that bridges the FastAPI backend
with the existing GenerationErrorHandler from the Wan2.2 infrastructure.

#### Classes

- **IntegratedErrorHandler**: Enhanced error handler that integrates FastAPI backend with existing
GenerationErrorHandler infrastr...
- **ErrorCategory**: ...
- **ErrorSeverity**: ...
- **RecoveryAction**: ...
- **UserFriendlyError**: ...

### core.intelligent_fallback_manager

Intelligent Fallback Manager for Enhanced Model Availability

This module provides smart alternatives when preferred models are unavailable,
implements model compatibility scoring algorithms, and manages fallback strategies
with request queuing and wait time estimation.

#### Classes

- **FallbackType**: Types of fallback strategies available...
- **ModelCapability**: Model capabilities for compatibility scoring...
- **GenerationRequirements**: Requirements for a generation request...
- **ModelSuggestion**: Suggestion for an alternative model...
- **FallbackStrategy**: Strategy for handling unavailable models...
- **EstimatedWaitTime**: Estimated wait time for model availability...
- **QueuedRequest**: Queued generation request waiting for model availability...
- **QueueResult**: Result of queuing a request...
- **IntelligentFallbackManager**: Intelligent fallback manager that provides smart alternatives when preferred models
are unavailable,...

### core.model_availability_manager

Model Availability Manager
Central coordination system for model availability, lifecycle management, and download prioritization.
Integrates with existing ModelManager, EnhancedModelDownloader, and ModelHealthMonitor.

#### Classes

- **ModelAvailabilityStatus**: Enhanced model availability status...
- **ModelPriority**: Model download priority levels...
- **DetailedModelStatus**: Comprehensive model status information...
- **ModelRequestResult**: Result of a model availability request...
- **CleanupRecommendation**: Model cleanup recommendation...
- **CleanupResult**: Result of cleanup operation...
- **RetentionPolicy**: Policy for model retention and cleanup...
- **ModelAvailabilityManager**: Central coordination system for model availability, lifecycle management,
and download prioritizatio...

### core.model_health_monitor

Model Health Monitor
Provides integrity checking, performance monitoring, corruption detection,
and automated health checks for WAN2.2 models.

#### Classes

- **HealthStatus**: Model health status enumeration...
- **CorruptionType**: Types of corruption that can be detected...
- **IntegrityResult**: Result of model integrity check...
- **PerformanceMetrics**: Performance metrics for model operations...
- **PerformanceHealth**: Health assessment based on performance metrics...
- **CorruptionReport**: Detailed corruption analysis report...
- **SystemHealthReport**: Overall system health report...
- **HealthCheckConfig**: Configuration for health monitoring...
- **ModelHealthMonitor**: Comprehensive model health monitoring system with integrity checking,
performance monitoring, corrup...

### core.model_integration_bridge

#### Classes

- **ModelStatus**: Model availability status...
- **ModelType**: Supported model types...
- **HardwareProfile**: Hardware profile information...
- **ModelIntegrationStatus**: Status of model integration...
- **GenerationParams**: Parameters for video generation...
- **GenerationResult**: Result of video generation...
- **ModelIntegrationBridge**: Bridges existing ModelManager with FastAPI backend
Provides adapter methods to convert between exist...
- **WANPipelineFactory**: ...
- **WANModelStatus**: ...
- **WANModelType**: ...

### core.model_update_manager

#### Classes

- **UpdateStatus**: Update status enumeration...
- **UpdatePriority**: Update priority levels...
- **UpdateType**: Types of updates...
- **ModelVersion**: Model version information...
- **UpdateInfo**: Information about an available update...
- **UpdateProgress**: Update progress tracking...
- **UpdateResult**: Result of an update operation...
- **UpdateSchedule**: Update scheduling configuration...
- **RollbackInfo**: Information about a rollback operation...
- **ModelUpdateManager**: Comprehensive model update management system with version checking,
update detection, safe update pr...

### core.model_usage_analytics

Model Usage Analytics System - Minimal Implementation
Tracks model usage patterns and provides basic recommendations.

#### Classes

- **UsageEventType**: Types of usage events to track...
- **UsageData**: Individual usage data point...
- **UsageStatistics**: Usage statistics for a model...
- **CleanupRecommendation**: Recommendation for model cleanup...
- **PreloadRecommendation**: Recommendation for model preloading...
- **CleanupAction**: Individual cleanup action...
- **CleanupRecommendations**: Comprehensive cleanup recommendations...
- **ModelUsageEventDB**: Database model for usage events...
- **ModelUsageAnalytics**: Model Usage Analytics System...

### core.performance_monitor

Performance monitoring system for real AI model integration.
Tracks generation performance, resource usage, and provides optimization recommendations.

#### Classes

- **PerformanceMetrics**: Performance metrics for a generation task....
- **SystemPerformanceSnapshot**: System performance snapshot at a point in time....
- **PerformanceAnalysis**: Analysis of performance trends and recommendations....
- **PerformanceMonitor**: Monitors and analyzes system and generation performance....

### core.performance_monitoring_system

Performance Monitoring and Optimization System for Enhanced Model Availability

This module provides comprehensive performance tracking for download operations,
health checks, fallback strategies, and system resource usage monitoring.

#### Classes

- **PerformanceMetricType**: Types of performance metrics tracked...
- **PerformanceMetric**: Individual performance metric data...
- **SystemResourceSnapshot**: System resource usage snapshot...
- **PerformanceReport**: Comprehensive performance report...
- **PerformanceTracker**: Tracks individual performance metrics...
- **SystemResourceMonitor**: Monitors system resource usage continuously...
- **PerformanceAnalyzer**: Analyzes performance data and provides optimization recommendations...
- **PerformanceMonitoringSystem**: Main performance monitoring and optimization system...

### core.runtime_config_updater

Runtime Configuration Update System

Provides hot-reload capabilities for configuration changes without requiring
application restart, with proper validation and rollback mechanisms.

#### Classes

- **RuntimeConfigurationUpdater**: Manages runtime configuration updates without application restart...
- **ConfigurationChangeHandler**: File system event handler for configuration file changes...

### core.system_integration

#### Classes

- **SystemIntegration**: Manages integration with existing Wan2.2 system components...
- **MockModelManager**: ...
- **MockModelDownloader**: ...
- **SimplifiedWanPipelineLoader**: ...
- **SimpleConfigurationBridge**: ...
- **ModelBridge**: ...
- **PipelineWrapper**: ...
- **GenerationResult**: ...

### core.__init__

Core backend modules for system integration


### core.models.__init__


### core.models.wan_models.wan_base_model

#### Classes

- **WANModelType**: ...
- **WANModelStatus**: ...
- **HardwareProfile**: ...
- **WanBasePipeline**: Minimal base so imports work; swap for the real base later....

### core.models.wan_models.wan_hardware_optimizer

#### Classes

- **WANHardwareOptimizer**: ...

### core.models.wan_models.wan_model_config

#### Classes

- **WANModelConfig**: ...
- **WANModelInfo**: ...

### core.models.wan_models.wan_model_downloader

#### Classes

- **WANModelDownloader**: ...

### core.models.wan_models.wan_pipeline_factory

#### Classes

- **WANPipelineConfig**: ...
- **WanT2VPipeline**: ...
- **WanI2VPipeline**: ...
- **WanTI2VPipeline**: ...
- **WANPipelineFactory**: ...

### core.models.wan_models.wan_progress_tracker

#### Classes

- **ProgressUpdate**: ...
- **WANProgressTracker**: Minimal, in-memory progress tracker.
Replace with a NVML/WS-backed implementation later if needed....

### core.models.wan_models.wan_vram_monitor

#### Classes

- **VRAMStats**: ...
- **WANVRAMMonitor**: ...

### core.models.wan_models.__init__


### core.model_orchestrator.component_deduplicator

Component Deduplication System - Content-addressed storage for shared model components.

This module implements a deduplication system that identifies common files across
models and creates hardlinks/symlinks to save disk space while maintaining
reference tracking to prevent premature deletion.

#### Classes

- **ComponentInfo**: Information about a shared component....
- **DeduplicationResult**: Result of a deduplication operation....
- **ComponentDeduplicator**: Manages component deduplication with content-addressed storage.

Features:
- Content-addressed stora...

### core.model_orchestrator.credential_cli

Command-line interface for secure credential management.

This module provides CLI commands for managing credentials securely,
including storing, retrieving, and rotating credentials.

#### Classes

- **CredentialCLI**: Command-line interface for credential management....

### core.model_orchestrator.credential_manager

Secure credential management system for the Model Orchestrator.

This module provides secure storage and retrieval of credentials using system keyring,
environment variables, and secure configuration files. It also handles credential masking
in logs and supports presigned URLs and temporary access tokens.

#### Classes

- **CredentialConfig**: Configuration for credential storage and retrieval....
- **CredentialMasker**: Utility class for masking sensitive information in logs and outputs....
- **CredentialStore**: Secure credential storage using system keyring with fallbacks....
- **PresignedURLManager**: Manager for handling presigned URLs and temporary access tokens....
- **SecureCredentialManager**: Main credential manager that combines all security features.

This class provides a high-level inter...

### core.model_orchestrator.download_manager

Advanced download manager with parallel downloads, bandwidth limiting, and queue management.

#### Classes

- **DownloadPriority**: Priority levels for download queue....
- **DownloadTask**: Individual download task....
- **DownloadProgress**: Progress information for a download....
- **BandwidthLimiter**: Token bucket bandwidth limiter....
- **ConnectionPool**: Enhanced HTTP connection pool with adaptive connection management....
- **ModelDownloadQueue**: Queue for managing downloads of a specific model....
- **DownloadMetrics**: Metrics for download performance analysis....
- **ParallelDownloadManager**: Advanced download manager with parallel downloads, bandwidth limiting,
queue management, and compreh...

### core.model_orchestrator.encryption_manager

At-rest encryption support for sensitive models in the Model Orchestrator.

This module provides encryption and decryption capabilities for model files
that contain sensitive information or require additional security measures.

#### Classes

- **EncryptionConfig**: Configuration for model encryption....
- **EncryptionMetadata**: Metadata for encrypted files....
- **KeyManager**: Manages encryption keys with rotation support....
- **FileEncryptor**: Handles encryption and decryption of individual files....
- **ModelEncryptionManager**: Main encryption manager for model files.

This class provides high-level encryption and decryption o...

### core.model_orchestrator.error_recovery

Error recovery and retry logic for the Model Orchestrator system.

#### Classes

- **RetryStrategy**: Retry strategies for different types of failures....
- **FailureCategory**: Categories of failures for recovery strategy selection....
- **RetryConfig**: Configuration for retry behavior....
- **RecoveryContext**: Context information for error recovery....
- **ErrorRecoveryManager**: Manages error recovery strategies and retry logic....

### core.model_orchestrator.exceptions

Exception classes for the Model Orchestrator system.

#### Classes

- **ErrorCode**: Structured error codes for consistent error handling....
- **ModelOrchestratorError**: Base exception for all Model Orchestrator errors....
- **ModelNotFoundError**: Raised when a requested model is not found in the manifest....
- **VariantNotFoundError**: Raised when a requested variant is not available for a model....
- **InvalidModelIdError**: Raised when a model ID format is invalid....
- **ManifestValidationError**: Raised when manifest validation fails....
- **SchemaVersionError**: Raised when manifest schema version is incompatible....
- **LockTimeoutError**: Raised when a lock cannot be acquired within the specified timeout....
- **LockError**: Raised when a lock operation fails....
- **NoSpaceError**: Raised when insufficient disk space is available....
- **ChecksumError**: Raised when file checksum verification fails....
- **SizeMismatchError**: Raised when file size doesn't match expected size....
- **IncompleteDownloadError**: Raised when a download is incomplete....
- **IntegrityVerificationError**: Raised when comprehensive integrity verification fails....
- **ManifestSignatureError**: Raised when manifest signature verification fails....
- **ModelValidationError**: Raised when model validation fails....
- **InvalidInputError**: Raised when input validation fails for a model type....
- **MigrationError**: Raised when configuration migration fails....
- **ValidationError**: Raised when validation fails....

### core.model_orchestrator.feature_flags

Feature Flags System - Gradual rollout and configuration management.

This module provides a centralized feature flag system for controlling
the rollout of model orchestrator features with environment-based configuration.

#### Classes

- **OrchestratorFeatureFlags**: Feature flags for model orchestrator functionality....
- **FeatureFlagManager**: Manager for feature flag operations and rollout control....

### core.model_orchestrator.garbage_collector

Garbage Collector - Disk space management with LRU-based cleanup.

#### Classes

- **GCTrigger**: Reasons for triggering garbage collection....
- **ModelInfo**: Information about a model for garbage collection....
- **GCConfig**: Configuration for garbage collection....
- **GCResult**: Result of garbage collection operation....
- **DiskUsage**: Disk usage information....
- **GarbageCollector**: Manages disk space through configurable retention policies....

### core.model_orchestrator.gpu_health

GPU-based health checks for WAN2.2 models.

This module provides smoke tests for t2v/i2v/ti2v models using minimal
GPU operations to validate model functionality without full inference.

#### Classes

- **HealthStatus**: Health check status values....
- **HealthCheckResult**: Result of a health check operation....
- **GPUHealthChecker**: Performs lightweight GPU health checks for WAN2.2 models.

Uses minimal denoise steps at low resolut...

### core.model_orchestrator.integrity_verifier

Comprehensive integrity verification system for model files.

This module provides enhanced integrity verification including SHA256 checksums,
HuggingFace ETag verification, manifest signature verification, and retry logic
for handling various integrity failure scenarios.

#### Classes

- **VerificationMethod**: Available verification methods in order of preference....
- **FileVerificationResult**: Result of verifying a single file....
- **IntegrityVerificationResult**: Result of comprehensive integrity verification....
- **HFFileMetadata**: HuggingFace file metadata for ETag verification....
- **IntegrityVerifier**: Comprehensive integrity verification system.

Provides multiple verification methods with fallback s...

### core.model_orchestrator.lock_manager

Cross-process locking system for model orchestrator.

This module provides OS-appropriate file locking to prevent concurrent download conflicts
and ensure atomic model operations across multiple processes.

#### Classes

- **LockManager**: Cross-process file locking manager with timeout and cleanup capabilities....

### core.model_orchestrator.logging_config

Structured logging configuration for the Model Orchestrator system.

#### Classes

- **StructuredFormatter**: Custom formatter that outputs structured JSON logs....
- **CorrelationIdFilter**: Filter to add correlation ID to log records....
- **ModelOrchestratorLogger**: Centralized logger configuration for the Model Orchestrator....
- **LogContext**: Context manager for setting correlation ID and additional context....
- **PerformanceTimer**: Context manager for measuring and logging operation performance....

### core.model_orchestrator.memory_optimizer

Memory optimization utilities for large model downloads and processing.

#### Classes

- **MemoryStats**: Memory usage statistics....
- **MemoryMonitor**: Monitor and track memory usage during downloads....
- **StreamingFileHandler**: Memory-efficient file handler for large downloads....
- **StreamingWriter**: Memory-efficient streaming writer....
- **StreamingReader**: Memory-efficient streaming reader....
- **MemoryOptimizer**: Main memory optimization coordinator for model downloads....

### core.model_orchestrator.metrics

Prometheus metrics for Model Orchestrator observability.

This module provides comprehensive metrics collection for downloads, errors,
storage usage, and performance tracking with limited cardinality.

#### Classes

- **MetricType**: Types of metrics collected by the orchestrator....
- **MetricEvent**: Represents a metric event with labels and value....
- **MetricsCollector**: Collects and manages Prometheus metrics for the Model Orchestrator.

Provides both Prometheus integr...
- **CollectorRegistry**: ...

### core.model_orchestrator.migration_cli

Migration CLI - Command-line interface for migration and compatibility tools.

This module provides CLI commands for migrating configurations, validating
manifests, and managing rollbacks.


### core.model_orchestrator.migration_demo

Migration Demo - Demonstration script for migration and compatibility tools.

This script demonstrates the complete migration workflow from legacy
configuration to the new model orchestrator system.


### core.model_orchestrator.migration_manager

Migration Manager - Configuration migration and backward compatibility tools.

This module provides tools for migrating from legacy configuration formats
to the new model orchestrator system, with backward compatibility adapters
and gradual rollout support.

#### Classes

- **LegacyConfig**: Legacy configuration structure from config.json....
- **MigrationResult**: Result of a configuration migration....
- **FeatureFlags**: Feature flags for gradual rollout of model orchestrator....
- **LegacyPathAdapter**: Adapter for resolving legacy model paths to orchestrator paths....
- **ConfigurationMigrator**: Tool for migrating legacy configurations to model orchestrator format....
- **ManifestValidator**: Validator for model manifest files and configurations....
- **RollbackManager**: Manager for rolling back migrations and configurations....

### core.model_orchestrator.model_ensurer

Model Ensurer - Atomic download orchestration with preflight checks.

#### Classes

- **ModelStatus**: Status of a model in the local storage....
- **VerificationResult**: Result of model integrity verification....
- **ModelStatusInfo**: Detailed status information for a model....
- **FailedSource**: Information about a failed source....
- **ModelEnsurer**: Orchestrates atomic model downloads with preflight checks....

### core.model_orchestrator.model_registry

Model Registry - Manifest parsing and model specification management.

This module handles loading and validating the models.toml manifest file,
providing typed access to model specifications with proper validation.

#### Classes

- **FileSpec**: Specification for a single file within a model....
- **VramEstimation**: VRAM estimation parameters for a model....
- **ModelDefaults**: Default parameters for model inference....
- **ModelSpec**: Complete specification for a model including all variants and metadata....
- **NormalizedModelId**: Normalized model identifier components....
- **ModelRegistry**: Registry for managing model specifications from a TOML manifest.

Provides validation, normalization...

### core.model_orchestrator.model_resolver

Model path resolution with cross-platform support and atomic operations.

This module provides deterministic path resolution for WAN2.2 models with support
for variants, temporary directories, and cross-platform compatibility including
Windows long paths and WSL scenarios.

#### Classes

- **PathIssue**: Represents a path validation issue....
- **ModelResolver**: Provides deterministic path resolution for WAN2.2 models with cross-platform support.

Handles:
- De...

### core.model_orchestrator.performance_benchmarks

Performance benchmarking and optimization testing for the model orchestrator.

#### Classes

- **BenchmarkResult**: Results from a performance benchmark....
- **MockFileSpec**: Mock file specification for testing....
- **MockHttpServer**: Mock HTTP server for testing downloads without external dependencies....
- **PerformanceBenchmark**: Comprehensive performance benchmarking suite for the model orchestrator....

### core.model_orchestrator.test_advanced_concurrency

Tests for advanced concurrency and performance features.

#### Classes

- **MockFileSpec**: Mock file specification for testing....
- **TestBandwidthLimiter**: Test bandwidth limiting functionality....
- **TestConnectionPool**: Test enhanced connection pool functionality....
- **TestModelDownloadQueue**: Test model download queue functionality....
- **TestDownloadMetrics**: Test download metrics tracking....
- **TestParallelDownloadManager**: Test parallel download manager functionality....
- **TestMemoryOptimizer**: Test memory optimization functionality....
- **TestStreamingFileHandler**: Test streaming file handler functionality....
- **TestIntegration**: Integration tests for advanced concurrency features....

### core.model_orchestrator.test_chaos_recovery

Chaos tests for network failures and interruptions in the Model Orchestrator.

#### Classes

- **NetworkFailureSimulator**: Simulates various network failure scenarios....
- **DiskFailureSimulator**: Simulates disk-related failures....
- **ProcessKillSimulator**: Simulates process interruptions and kills....
- **TestNetworkFailureRecovery**: Test recovery from various network failures....
- **TestDiskFailureRecovery**: Test recovery from disk-related failures....
- **TestProcessInterruptionRecovery**: Test recovery from process interruptions....
- **TestConcurrencyFailureRecovery**: Test recovery from concurrency-related failures....
- **TestIntegratedChaosScenarios**: Test complex scenarios with multiple failure types....

### core.model_orchestrator.test_component_deduplicator

Tests for Component Deduplication System.

Tests component sharing across t2v/i2v/ti2v models with various scenarios
including hardlink/symlink creation, reference tracking, and cleanup.

#### Classes

- **TestComponentDeduplicator**: Test suite for ComponentDeduplicator....

### core.model_orchestrator.test_concurrent_ensure

Test for concurrent ensure() calls - validates that two concurrent ensure() calls
for the same model result in one downloading while the other waits, and both succeed.

#### Classes

- **MockStorageBackend**: Mock storage backend that simulates slow downloads....
- **TestConcurrentEnsure**: Test concurrent ensure() calls for the same model....

### core.model_orchestrator.test_enhanced_integrity

Tests for enhanced integrity verification in ModelEnsurer.

#### Classes

- **MockStorageBackend**: Mock storage backend for testing....
- **TestEnhancedIntegrityVerification**: Test enhanced integrity verification in ModelEnsurer....
- **TestIntegrityVerificationEdgeCases**: Test edge cases for integrity verification....
- **CorrectContentBackend**: ...
- **RetryableBackend**: ...
- **HFMetadataBackend**: ...
- **WrongSizeBackend**: ...
- **FailingVerificationBackend**: ...

### core.model_orchestrator.test_error_recovery

Tests for the error recovery and retry logic system.

#### Classes

- **TestRetryConfiguration**: Test retry configuration and strategy selection....
- **TestFailureCategories**: Test failure category classification....
- **TestDelayCalculation**: Test retry delay calculation strategies....
- **TestRetryLogic**: Test the core retry logic....
- **TestRecoveryStrategies**: Test specific recovery strategies for different error types....
- **TestConvenienceFunctions**: Test convenience functions and decorators....
- **TestRecoveryContext**: Test recovery context management....
- **TestErrorRecoveryIntegration**: Test integration of error recovery with other components....

### core.model_orchestrator.test_garbage_collector

Tests for the Garbage Collector.

#### Classes

- **TestGarbageCollector**: Test cases for the GarbageCollector class....
- **TestGCConfig**: Test cases for GCConfig....
- **TestModelInfo**: Test cases for ModelInfo dataclass....
- **TestDiskUsage**: Test cases for DiskUsage dataclass....
- **TestGCResult**: Test cases for GCResult dataclass....

### core.model_orchestrator.test_integrity_verifier

Tests for comprehensive integrity verification system.

#### Classes

- **TestIntegrityVerifier**: Test suite for IntegrityVerifier....
- **TestIntegrityVerifierEdgeCases**: Test edge cases and error conditions for IntegrityVerifier....

### core.model_orchestrator.test_lock_manager

Unit tests for the LockManager class.

#### Classes

- **TestLockManager**: Test cases for LockManager functionality....
- **TestLockManagerIntegration**: Integration tests for LockManager with real file operations....

### core.model_orchestrator.test_migration_tools

Tests for migration and compatibility tools.

This module contains comprehensive tests for configuration migration,
validation, rollback functionality, and backward compatibility.

#### Classes

- **TestLegacyConfig**: Test LegacyConfig data class....
- **TestFeatureFlags**: Test FeatureFlags functionality....
- **TestLegacyPathAdapter**: Test LegacyPathAdapter functionality....
- **TestConfigurationMigrator**: Test ConfigurationMigrator functionality....
- **TestManifestValidator**: Test ManifestValidator functionality....
- **TestRollbackManager**: Test RollbackManager functionality....
- **TestMigrationIntegration**: Integration tests for migration tools....

### core.model_orchestrator.test_mock_backend

Simple test to verify the mock backend works correctly.


### core.model_orchestrator.test_models_toml_validator

Test suite for the models.toml validator.

Tests various validation scenarios including:
- Schema version issues
- Duplicate detection
- Path traversal vulnerabilities
- Windows case sensitivity issues
- Malformed TOML structures

#### Classes

- **TestModelsTomlValidator**: Test cases for the ModelsTomlValidator....

### core.model_orchestrator.test_model_ensurer

Integration tests for ModelEnsurer - atomic download orchestration with preflight checks.

#### Classes

- **MockStorageBackend**: Mock storage backend for testing....
- **TestModelEnsurer**: Test cases for ModelEnsurer....

### core.model_orchestrator.test_model_registry

Unit tests for the Model Registry system.

Tests manifest parsing, validation, model ID normalization, and error handling.

#### Classes

- **TestModelIdNormalization**: Test model ID normalization functionality....
- **TestManifestParsing**: Test manifest parsing and validation....
- **TestPathSafetyValidation**: Test path safety validation....
- **TestModelSpecRetrieval**: Test model specification retrieval....

### core.model_orchestrator.test_model_resolver

Unit tests for ModelResolver class.

Tests cross-platform path handling, atomic operations, and Windows long path scenarios.

#### Classes

- **TestModelResolver**: Test cases for ModelResolver class....
- **TestPathIssue**: Test cases for PathIssue class....

### core.model_orchestrator.test_observability_integration

Integration tests for Model Orchestrator observability features.

Tests the complete observability stack including metrics collection,
structured logging, GPU health checks, and performance monitoring.

#### Classes

- **TestMetricsIntegration**: Test metrics collection and reporting....
- **TestStructuredLogging**: Test structured logging with correlation IDs....
- **TestGPUHealthChecks**: Test GPU-based health checking functionality....
- **TestHealthServiceIntegration**: Test integration of health service with observability features....
- **TestEndToEndObservability**: End-to-end tests for complete observability workflow....

### core.model_orchestrator.test_performance_runner

Simple performance benchmark runner for testing.


### core.model_orchestrator.test_security_credentials

Comprehensive security tests for credential management in the Model Orchestrator.

These tests verify secure credential storage, retrieval, masking, and handling
of presigned URLs and temporary access tokens.

#### Classes

- **TestCredentialMasker**: Test credential masking functionality....
- **TestCredentialStore**: Test secure credential storage....
- **TestPresignedURLManager**: Test presigned URL management....
- **TestSecureCredentialManager**: Test the main credential manager....
- **TestSecureCredentialContext**: Test secure credential context manager....
- **TestModelEncryption**: Test model encryption functionality....
- **TestIntegrationSecurity**: Integration tests for security features....

### core.model_orchestrator.test_security_integration

Integration tests for security features in the Model Orchestrator.

These tests verify that all security components work together correctly,
including credential management, encryption, logging masking, and storage backend integration.

#### Classes

- **TestSecurityIntegration**: Integration tests for security features....
- **TestSecurityCompliance**: Test security compliance and best practices....

### core.model_orchestrator.test_validation_tools

Tests for validation tools.

This module contains comprehensive tests for manifest validation,
security checks, performance analysis, and compatibility validation.

#### Classes

- **TestValidationIssue**: Test ValidationIssue functionality....
- **TestValidationReport**: Test ValidationReport functionality....
- **TestManifestSchemaValidator**: Test ManifestSchemaValidator functionality....
- **TestSecurityValidator**: Test SecurityValidator functionality....
- **TestPerformanceValidator**: Test PerformanceValidator functionality....
- **TestCompatibilityValidator**: Test CompatibilityValidator functionality....
- **TestComprehensiveValidator**: Test ComprehensiveValidator functionality....

### core.model_orchestrator.test_wan22_handler

Tests for WAN2.2-specific model handling.

Tests cover:
- Sharded model support and selective downloading
- Model-specific input validation
- Text embedding caching
- VAE configuration and memory optimization
- Development/production variant switching
- Component validation and file management

#### Classes

- **TestTextEmbeddingCache**: Test text embedding cache functionality....
- **TestWAN22ModelHandler**: Test WAN2.2 model handler functionality....
- **TestWAN22Integration**: Integration tests for WAN2.2 model handling....

### core.model_orchestrator.test_wan22_integration

Integration tests for WAN2.2 model handling with the orchestrator.

Tests the complete integration between WAN2.2 handler and model orchestrator,
including selective downloading, variant switching, and component validation.

#### Classes

- **MockStorageBackend**: Mock storage backend for testing....
- **TestWAN22Integration**: Test WAN2.2 integration with model orchestrator....

### core.model_orchestrator.validate_current_models

Validation script for the current models.toml file.

This script validates the current models.toml file and provides a summary
of the validation results, demonstrating that the validator works correctly.


### core.model_orchestrator.validate_models_toml

Models.toml Validator

Validates the models.toml manifest file for:
- Schema version compatibility
- No duplicate model IDs or file paths
- No path traversal vulnerabilities
- Windows case sensitivity compatibility
- Proper TOML structure and required fields

#### Classes

- **ModelsTomlValidator**: Comprehensive validator for models.toml manifest files....

### core.model_orchestrator.validate_task_21

Task 21 Validation Script

This script validates that all components of Task 21 (comprehensive testing and documentation)
have been properly implemented according to the requirements.

#### Classes

- **Task21Validator**: Validates Task 21 implementation completeness....

### core.model_orchestrator.validation_tools

Validation Tools - Comprehensive validation for manifests and configurations.

This module provides advanced validation tools for model manifests,
configuration files, and system compatibility checks.

#### Classes

- **ValidationIssue**: Represents a validation issue with severity and context....
- **ValidationReport**: Comprehensive validation report....
- **ManifestSchemaValidator**: Validator for manifest schema and structure....
- **SecurityValidator**: Validator for security-related issues in manifests....
- **PerformanceValidator**: Validator for performance-related issues in manifests....
- **CompatibilityValidator**: Validator for compatibility issues across platforms and configurations....
- **ComprehensiveValidator**: Comprehensive validator that runs all validation checks....

### core.model_orchestrator.wan22_example

Example script demonstrating WAN2.2-specific model handling features.

This script shows how to:
1. Use selective downloading for development variants
2. Validate inputs for different model types
3. Configure VAE settings for memory optimization
4. Cache text embeddings for performance
5. Estimate memory usage for different configurations


### core.model_orchestrator.wan22_handler

WAN2.2-specific model handling and processing.

This module provides specialized functionality for WAN2.2 models including:
- Sharded model support with selective downloading
- Model-specific conditioning and preprocessing
- Text embedding caching for multi-clip sequences
- VAE optimization and memory management
- Development/production variant switching
- Input validation for different model types

#### Classes

- **ShardInfo**: Information about a model shard....
- **ComponentInfo**: Information about a model component....
- **TextEmbeddingCache**: Cache for text embeddings to avoid recomputation....
- **WAN22ModelHandler**: Handler for WAN2.2-specific model operations....

### core.model_orchestrator.__init__


### core.model_orchestrator.storage_backends.base_store

Base storage backend interface.

#### Classes

- **DownloadResult**: Result of a download operation....
- **StorageBackend**: Abstract base class for storage backends....

### core.model_orchestrator.storage_backends.hf_store

HuggingFace Hub storage backend.

#### Classes

- **HFFileMetadata**: HuggingFace file metadata for integrity verification....
- **HFStore**: HuggingFace Hub storage backend using huggingface_hub....

### core.model_orchestrator.storage_backends.s3_store

S3/MinIO storage backend with parallel downloads and resume capability.

#### Classes

- **S3Config**: Configuration for S3/MinIO backend....
- **S3Store**: S3/MinIO storage backend with parallel downloads and resume capability....

### core.model_orchestrator.storage_backends.test_hf_store

Tests for HuggingFace storage backend.

#### Classes

- **TestHFStore**: Test cases for HFStore....

### core.model_orchestrator.storage_backends.test_s3_store

Tests for S3/MinIO storage backend.

#### Classes

- **MockFileSpec**: Mock FileSpec for testing....
- **TestS3Store**: Test suite for S3Store....
- **TestS3Config**: Test suite for S3Config....

### core.model_orchestrator.storage_backends.__init__

Storage backends for model orchestrator.


### core.model_orchestrator.tests.test_cross_platform_compatibility

Cross-platform compatibility tests for Model Orchestrator.

Tests Windows, WSL, and Unix-specific behaviors including:
- Path handling and long path support
- File system operations and atomic moves
- Lock mechanisms and process synchronization
- Case sensitivity and reserved names

#### Classes

- **TestWindowsCompatibility**: Test Windows-specific behaviors and limitations....
- **TestWSLCompatibility**: Test WSL (Windows Subsystem for Linux) specific behaviors....
- **TestUnixCompatibility**: Test Unix/Linux specific behaviors....
- **TestCrossPlatformPathHandling**: Test path handling that works across all platforms....
- **TestPlatformSpecificErrorHandling**: Test platform-specific error handling and messages....

### core.model_orchestrator.tests.test_end_to_end_workflows

End-to-end integration tests for complete Model Orchestrator workflows.

Tests complete user journeys from model request to ready-to-use model paths,
including all storage backends, error recovery, and concurrent access scenarios.

#### Classes

- **TestEndToEndWorkflows**: Test complete workflows from user request to model availability....
- **TestCrossWorkflowIntegration**: Test integration between different workflow components....

### core.model_orchestrator.tests.test_performance_load

Performance and load testing suites for Model Orchestrator.

Tests system behavior under various load conditions:
- Concurrent downloads and access patterns
- Large model handling and memory usage
- Network bandwidth and timeout scenarios
- Storage backend performance characteristics

#### Classes

- **PerformanceTestBase**: Base class for performance tests with common utilities....
- **TestConcurrentPerformance**: Test performance under concurrent load....
- **TestMemoryPerformance**: Test memory usage and performance characteristics....
- **TestNetworkPerformance**: Test network-related performance characteristics....
- **TestStoragePerformance**: Test storage backend performance characteristics....
- **TestScalabilityLimits**: Test system behavior at scale limits....

### core.model_orchestrator.tests.test_requirements_validation

Requirements validation tests for Model Orchestrator.

This module validates that all requirements from the specification
are properly implemented and tested.

#### Classes

- **TestRequirement1_UnifiedModelManifest**: Test Requirement 1: Unified Model Manifest System with Versioning....
- **TestRequirement3_DeterministicPathResolution**: Test Requirement 3: Deterministic Path Resolution....
- **TestRequirement4_AtomicDownloads**: Test Requirement 4: Atomic Downloads with Concurrency Safety....
- **TestRequirement5_IntegrityVerification**: Test Requirement 5: Comprehensive Integrity and Trust Chain....
- **TestRequirement10_DiskSpaceManagement**: Test Requirement 10: Disk Space Management and Garbage Collection....
- **TestRequirement12_Observability**: Test Requirement 12: Comprehensive Observability and Error Classification....
- **TestRequirement13_ProductionAPI**: Test Requirement 13: Production API Surface and CLI Tools....

### core.model_orchestrator.tests.test_runner

Comprehensive test runner for Model Orchestrator.

This script runs all test suites and generates comprehensive reports
for end-to-end validation of the Model Orchestrator system.

#### Classes

- **TestRunner**: Comprehensive test runner for Model Orchestrator....

### core.model_orchestrator.tests.__init__


### core.services.model_manager

#### Classes

- **ModelManager**: Minimal stub to satisfy imports; extend if your code actually uses it....

### core.services.utils


### core.services.__init__


### examples.enhanced_downloader_demo

Enhanced Model Downloader Demo
Demonstrates the enhanced model downloader functionality with retry logic,
progress tracking, and download management features.


### examples.enhanced_error_recovery_demo

Enhanced Error Recovery System Demo

This demo shows how to use the Enhanced Error Recovery System with various
failure scenarios and recovery strategies.

#### Classes

- **EnhancedErrorRecoveryDemo**: Demo class for Enhanced Error Recovery System...

### examples.enhanced_model_configuration_demo

Enhanced Model Configuration System Demo

Demonstrates the configuration management system for enhanced model availability features.


### examples.hardware_optimization_integration_example


### examples.integrated_error_handler_example

#### Classes

- **MockGenerationService**: ...

### examples.intelligent_fallback_manager_demo

#### Classes

- **DemoAvailabilityManager**: Demo availability manager that simulates different model states...
- **MockStatus**: ...
- **MockModelStatus**: ...

### examples.model_health_monitor_demo

Model Health Monitor Demo
Demonstrates the functionality of the Model Health Monitor system.


### examples.model_update_manager_demo

Model Update Manager Demo
Demonstrates the functionality of the Model Update Management System
including version checking, update detection, safe updates, and rollback capabilities.

#### Classes

- **ModelUpdateManagerDemo**: Demo class for Model Update Manager functionality...

### examples.model_usage_analytics_demo

Model Usage Analytics System Demo
Demonstrates the functionality of the model usage analytics system including
tracking, analysis, and recommendation generation.


### examples.real_generation_example

Example usage of the Real Generation Pipeline
Demonstrates T2V, I2V, and TI2V generation with progress tracking


### examples.websocket_model_notifications_demo

#### Classes

- **MockWebSocketConnection**: Mock WebSocket connection for demonstration...

### migration.data_migrator

Data migration utilities for migrating existing Gradio outputs to new SQLite system.

#### Classes

- **DataMigrator**: Handles migration of existing Gradio outputs to new SQLite system....

### migration.__init__

Migration utilities for data migration.


### monitoring.logger

Production logging system with structured logging and monitoring.

#### Classes

- **StructuredFormatter**: Custom formatter for structured JSON logging....
- **PerformanceLogger**: Logger for performance metrics and monitoring....
- **ErrorLogger**: Logger for error tracking and monitoring....

### monitoring.metrics

Application metrics collection and monitoring.

#### Classes

- **SystemMetrics**: System resource metrics....
- **ApplicationMetrics**: Application-specific metrics....
- **PerformanceMetrics**: Performance metrics....
- **MetricsCollector**: Collects and stores application metrics....
- **MetricsMonitor**: Background metrics monitoring service....

### monitoring.__init__

Monitoring and logging utilities.


### repositories.database

Database configuration and models using SQLAlchemy

#### Classes

- **TaskStatusEnum**: Task status enumeration for database...
- **ModelTypeEnum**: Model type enumeration for database...
- **GenerationTaskDB**: Database model for generation tasks...
- **SystemStatsDB**: Database model for system statistics (for historical data)...

### repositories.__init__

Data repositories


### schemas.schemas

Pydantic models for API request/response schemas

#### Classes

- **ModelType**: Supported model types...
- **TaskStatus**: Task status enumeration...
- **QuantizationLevel**: Quantization levels for optimization...
- **GenerationRequest**: Request model for video generation...
- **GenerationResponse**: Response model for generation requests...
- **TaskInfo**: Task information model...
- **QueueStatus**: Queue status information...
- **SystemStats**: System resource statistics...
- **OptimizationSettings**: Optimization settings model...
- **VideoMetadata**: Generated video metadata...
- **OutputsResponse**: Response model for outputs listing...
- **ErrorResponse**: Error response model...
- **HealthResponse**: Health check response...
- **PromptEnhanceRequest**: Request model for prompt enhancement...
- **PromptEnhanceResponse**: Response model for prompt enhancement...
- **PromptPreviewResponse**: Response model for prompt enhancement preview...
- **LoRAInfo**: LoRA file information...
- **LoRAListResponse**: Response model for LoRA listing...
- **LoRAUploadResponse**: Response model for LoRA upload...
- **LoRAStatusResponse**: Response model for LoRA status...
- **LoRAPreviewResponse**: Response model for LoRA preview...
- **LoRAMemoryImpactResponse**: Response model for LoRA memory impact estimation...

### schemas.__init__

API schemas


### scripts.config_migration

Configuration migration script to migrate from existing systems to FastAPI integration.
Handles merging configurations from different sources and ensuring compatibility.

#### Classes

- **ConfigurationMigrator**: Handles configuration migration from existing systems....

### scripts.config_migration_tool

Configuration Migration Tool for Enhanced Model Management

This tool helps migrate existing configurations to the new enhanced model
management configuration format, with validation and backup capabilities.

#### Classes

- **ConfigurationMigrationTool**: Tool for migrating and managing enhanced model configurations...

### scripts.deployment_validator

Deployment validation script to verify all components are working correctly
after real AI model integration deployment.

#### Classes

- **DeploymentValidator**: Validates deployment of real AI model integration....

### scripts.final_validation

Final validation script for real AI model integration.
Comprehensive validation of all components and system readiness.

#### Classes

- **FinalValidator**: Comprehensive final validation for the real AI model integration system....

### scripts.migrate_to_real_generation

Migration script to transition from mock to real AI generation mode.
This script handles configuration updates, database migrations, and validation.

#### Classes

- **RealGenerationMigrator**: Handles migration from mock to real AI generation mode....

### scripts.performance_dashboard

Performance dashboard for monitoring real AI model integration performance.
Provides real-time monitoring and analysis of generation performance.

#### Classes

- **PerformanceDashboard**: Real-time performance dashboard for AI model integration....

### scripts.deployment.config_backup_restore

Configuration Backup and Restore Tools for Enhanced Model Availability System

This script provides comprehensive backup and restore capabilities for all
configuration files and settings related to the enhanced model availability system.

#### Classes

- **BackupType**: Types of configuration backups...
- **BackupStatus**: Status of backup operations...
- **ConfigFile**: Represents a configuration file...
- **BackupManifest**: Manifest of a configuration backup...
- **RestoreResult**: Result of a restore operation...
- **ConfigurationBackupManager**: Manages configuration backups and restores...

### scripts.deployment.deploy

Deployment Automation Script for Enhanced Model Availability System

This script automates the deployment of the enhanced model availability system,
including validation, migration, monitoring setup, and rollback capabilities.

#### Classes

- **DeploymentPhase**: Deployment phases...
- **DeploymentStatus**: Deployment status...
- **DeploymentResult**: Result of deployment operation...
- **EnhancedModelAvailabilityDeployer**: Main deployment orchestrator...

### scripts.deployment.deployment_validator

Deployment Validator for Enhanced Model Availability System

This script validates that the enhanced model availability system is properly
deployed and all components are functioning correctly.

#### Classes

- **ValidationLevel**: Validation severity levels...
- **ValidationResult**: Result of a validation check...
- **DeploymentValidationReport**: Complete deployment validation report...
- **EnhancedModelAvailabilityValidator**: Validates enhanced model availability system deployment...

### scripts.deployment.model_migration

Model Migration Script for Enhanced Model Availability System

This script migrates existing model installations to work with the enhanced
model availability system, ensuring compatibility and data preservation.

#### Classes

- **MigrationResult**: Result of a migration operation...
- **ModelMigrationStatus**: Status of model migration...
- **ModelMigrationManager**: Manages migration of existing model installations...

### scripts.deployment.monitoring_setup

Monitoring and Alerting Setup for Enhanced Model Availability System

This script sets up monitoring and alerting for production deployment of the
enhanced model availability system, ensuring operational visibility and proactive issue detection.

#### Classes

- **AlertLevel**: Alert severity levels...
- **MetricType**: Types of metrics to monitor...
- **MonitoringMetric**: Represents a monitoring metric...
- **AlertRule**: Represents an alert rule...
- **Alert**: Represents an active alert...
- **MetricsCollector**: Collects metrics from the enhanced model availability system...
- **AlertManager**: Manages alerts and notifications...
- **EnhancedModelAvailabilityMonitor**: Main monitoring system for enhanced model availability...

### scripts.deployment.rollback_manager

Rollback Manager for Enhanced Model Availability System

This script provides rollback capabilities for failed deployments of the
enhanced model availability system, ensuring system stability and data integrity.

#### Classes

- **RollbackType**: Types of rollback operations...
- **RollbackStatus**: Status of rollback operations...
- **RollbackPoint**: Represents a rollback point in the system...
- **RollbackResult**: Result of a rollback operation...
- **RollbackManager**: Manages rollback operations for enhanced model availability system...

### services.demo_health_service

Demo script showing how to use the Model Health Service.

This demonstrates the basic usage of the health monitoring endpoint
for the Model Orchestrator.

#### Classes

- **MockModelRegistry**: Mock model registry for demo purposes....
- **MockModelResolver**: Mock model resolver for demo purposes....
- **MockModelEnsurer**: Mock model ensurer for demo purposes....

### services.generation_service

Enhanced Generation service with real AI model integration
Integrates with existing Wan2.2 system using ModelIntegrationBridge and RealGenerationPipeline

#### Classes

- **TaskSubmissionResult**: Structured response for generation task submissions....
- **ModelType**: Centralized model type definitions...
- **VRAMMonitor**: Enhanced VRAM monitoring and management for generation tasks...
- **GenerationService**: Enhanced service for managing video generation tasks with real AI integration...
- **FallbackErrorHandler**: ...

### services.generation_service_analytics_integration

Integration module for adding usage analytics to the existing generation service.
This module provides hooks and utilities to track model usage without modifying the core generation service.

#### Classes

- **GenerationServiceAnalyticsIntegration**: Integration class to add analytics tracking to the generation service.
This class provides methods t...

### services.model_health_service

Model Health Service - Enhanced health monitoring with observability features.

Provides comprehensive model health checking including GPU validation,
performance metrics, and detailed diagnostics with structured logging.

#### Classes

- **ModelHealthInfo**: Enhanced health information for a single model....
- **OrchestratorHealthResponse**: Enhanced orchestrator health response with observability metrics....
- **ModelHealthService**: Enhanced service for providing comprehensive model health monitoring....

### services.real_generation_pipeline

Real Generation Pipeline
Integrates with existing WanPipelineLoader infrastructure for actual video generation

#### Classes

- **GenerationStage**: Stages of video generation process...
- **ProgressUpdate**: Progress update information...
- **RealGenerationPipeline**: Real generation pipeline using existing WanPipelineLoader infrastructure
Handles T2V, I2V, and TI2V ...
- **SimplePipelineWrapper**: ...
- **FallbackGenerationConfig**: ...
- **SimpleGenerationResult**: ...
- **SimpleGenerationResult**: ...

### services.test_health_integration

Integration tests for Model Health Service with Model Orchestrator components.

#### Classes

- **TestHealthServiceIntegration**: Integration tests for health service with orchestrator components....

### services.test_model_health_service

Tests for Model Health Service

#### Classes

- **TestModelHealthService**: Test cases for ModelHealthService....
- **TestModelHealthServiceIntegration**: Integration tests for ModelHealthService....

### services.test_wan_pipeline_integration

Integration tests for WAN Pipeline Integration with Model Orchestrator.

Tests the integration between the Model Orchestrator and WAN pipeline loading,
including component validation, VRAM estimation, and model-specific handling.

#### Classes

- **TestWanPipelineIntegration**: Test WAN Pipeline Integration functionality....
- **TestGlobalFunctions**: Test global functions for WAN pipeline integration....
- **TestWanModelSpecs**: Test WAN model specifications....
- **TestWanPipelineLoaderIntegration**: Integration tests for WAN pipeline loader with Model Orchestrator....

### services.wan_orchestrator_setup

WAN Pipeline Integration Setup.

This module provides setup functions to initialize the WAN pipeline integration
with the Model Orchestrator when both systems are available.


### services.wan_pipeline_integration

WAN Pipeline Integration with Model Orchestrator.

This module provides the integration layer between the Model Orchestrator
and WAN pipeline loading, including model-specific handling and component validation.

#### Classes

- **WanModelType**: WAN model types with their characteristics....
- **WanModelSpec**: Specification for a WAN model type....
- **ComponentValidationResult**: Result of component validation....
- **WanPipelineIntegration**: Integration layer between Model Orchestrator and WAN pipeline loading....

### services.__init__

Backend services


### tests.debug_comprehensive

Debug script to check comprehensive integration suite


### tests.run_comprehensive_tests

#### Classes

- **ComprehensiveTestRunner**: Runner for comprehensive integration tests...

### tests.run_comprehensive_test_validation

Comprehensive Test Validation Script

This script runs the complete Enhanced Model Availability testing suite,
including integration tests, stress tests, chaos engineering tests,
performance benchmarks, and user acceptance tests.

It generates a comprehensive validation report with pass/fail status,
performance metrics, and recommendations.

Usage:
    python run_comprehensive_test_validation.py [options]

Options:
    --quick: Run quick validation (reduced test iterations)
    --full: Run full validation suite (default)
    --performance-only: Run only performance tests
    --stress-only: Run only stress tests
    --chaos-only: Run only chaos engineering tests
    --user-acceptance-only: Run only user acceptance tests
    --report-format: json|html|text (default: text)
    --output-file: Output file for report (default: stdout)

#### Classes

- **ComprehensiveTestValidator**: Main test validation orchestrator....

### tests.run_enhanced_model_availability_tests

Enhanced Model Availability Test Runner

Simple script to run all enhanced model availability tests with proper
async handling and comprehensive reporting.

Usage:
    python run_enhanced_model_availability_tests.py


### tests.test_advanced_system_features

Test suite for Task 12: Advanced system features
Tests WebSocket support, Chart.js integration, time range selection, and optimization presets

#### Classes

- **TestWebSocketSupport**: Test WebSocket support for sub-second updates...
- **TestAdvancedCharts**: Test Chart.js integration and historical data...
- **TestTimeRangeSelection**: Test interactive time range selection...
- **TestOptimizationPresets**: Test advanced optimization presets and recommendations...
- **TestIntegrationRequirements**: Test that all requirements are met...

### tests.test_analytics_simple

Simple test to verify analytics system basic functionality

#### Classes

- **UsageEventType**: ...
- **UsageData**: ...

### tests.test_backwards_compatibility

Test backwards compatibility with existing Gradio system.
Ensures model files, LoRA weights, and generation results are identical.

#### Classes

- **TestBackwardsCompatibility**: Test suite for backwards compatibility with existing Gradio system....
- **TestFullBackwardsCompatibility**: Full integration test for backwards compatibility....

### tests.test_chaos_engineering

Chaos Engineering Tests for Enhanced Model Availability System

This module implements chaos engineering principles to validate system resilience
under various failure conditions, including component failures, network partitions,
and resource exhaustion scenarios.

Requirements covered: 1.4, 2.4, 3.4, 4.4, 6.4, 7.4

#### Classes

- **ChaosExperiment**: Configuration for a chaos engineering experiment....
- **ChaosResult**: Result of a chaos engineering experiment....
- **ChaosEngineeringTestSuite**: Comprehensive chaos engineering test suite....
- **TestChaosEngineeringTestSuite**: Pytest wrapper for chaos engineering test suite....

### tests.test_comprehensive_integration_suite

Comprehensive Testing Suite for Real AI Model Integration
Tests all aspects of the integration including ModelIntegrationBridge, 
RealGenerationPipeline, end-to-end workflows, and performance benchmarks.

#### Classes

- **ComprehensiveTestMetrics**: Comprehensive metrics collection for integration testing...
- **TestModelIntegrationBridgeComprehensive**: Comprehensive tests for ModelIntegrationBridge functionality...
- **TestRealGenerationPipelineComprehensive**: Comprehensive tests for RealGenerationPipeline with all model types...
- **TestEndToEndIntegration**: End-to-end integration tests from FastAPI to real model generation...
- **TestPerformanceBenchmarks**: Performance benchmarking tests for generation speed and resource usage...

### tests.test_configuration_bridge

Test suite for Configuration Bridge
Tests configuration loading, validation, and runtime updates

#### Classes

- **TestConfigurationBridge**: Test cases for ConfigurationBridge functionality...

### tests.test_cors_validation

Test CORS validation functionality

#### Classes

- **TestCORSValidator**: Test CORS validator functionality...
- **TestCORSIntegration**: Test CORS integration with FastAPI app...

### tests.test_deployment_system

Test Suite for Enhanced Model Availability Deployment System

This module contains comprehensive tests for the deployment automation,
validation, rollback, and monitoring systems.

#### Classes

- **TestDeploymentValidator**: Test deployment validation functionality...
- **TestRollbackManager**: Test rollback functionality...
- **TestConfigurationBackupManager**: Test configuration backup and restore...
- **TestModelMigrationManager**: Test model migration functionality...
- **TestEnhancedModelAvailabilityMonitor**: Test monitoring system...
- **TestEnhancedModelAvailabilityDeployer**: Test deployment orchestration...
- **TestDeploymentIntegration**: Integration tests for deployment system...
- **TestDeploymentPerformance**: Performance tests for deployment system...

### tests.test_download_stress_testing

Stress Testing Module for Download Management and Retry Logic

This module contains comprehensive stress tests for the enhanced model downloader,
focusing on retry mechanisms, concurrent downloads, and failure recovery under
high load conditions.

Requirements covered: 1.4, 5.4

#### Classes

- **StressTestMetrics**: Metrics collected during stress testing....
- **DownloadStressTestSuite**: Comprehensive stress testing suite for download operations....
- **TestDownloadStressTestSuite**: Pytest wrapper for stress test suite....

### tests.test_end_to_end_comprehensive

Comprehensive End-to-End Integration Tests
Tests complete workflows from FastAPI endpoints to real model generation

#### Classes

- **TestEndToEndWorkflows**: Test complete end-to-end workflows...
- **TestAPICompatibility**: Test API compatibility and contract maintenance...
- **TestWebSocketIntegration**: Test WebSocket integration during generation...
- **TaskStatus**: ...

### tests.test_enhanced_error_recovery

Tests for Enhanced Error Recovery System

This module tests the sophisticated error categorization, multi-strategy recovery,
intelligent fallback integration, automatic repair triggers, and user-friendly
error messages with actionable recovery steps.

#### Classes

- **TestEnhancedErrorRecovery**: Test suite for EnhancedErrorRecovery class...
- **TestErrorContextCreation**: Test error context creation and categorization...
- **TestConvenienceFunction**: Test the convenience function for creating enhanced error recovery...
- **TestIntegrationScenarios**: Test integration scenarios with various failure types...

### tests.test_enhanced_generation_service_integration

Integration Tests for Enhanced Generation Service
Tests the integration of ModelAvailabilityManager, enhanced download retry logic,
intelligent fallback, usage analytics tracking, health monitoring, and error recovery.

#### Classes

- **TestEnhancedGenerationServiceIntegration**: Test suite for enhanced generation service integration...
- **TestGenerationServiceErrorRecovery**: Test suite for generation service error recovery...

### tests.test_enhanced_model_availability_comprehensive

Comprehensive Testing Suite for Enhanced Model Availability System
Tests all enhanced components working together including integration tests,
end-to-end workflows, performance benchmarks, and stress testing.

#### Classes

- **TestMetrics**: Metrics collection for comprehensive testing...
- **ComprehensiveTestSuite**: Comprehensive test suite for enhanced model availability...
- **TestEnhancedModelDownloaderIntegration**: Integration tests for Enhanced Model Downloader...

### tests.test_enhanced_model_availability_integration

Comprehensive Integration Tests for Enhanced Model Availability System

This module contains integration tests that verify all enhanced model availability
components work together correctly, covering end-to-end workflows from model
requests to fallback scenarios.

Requirements covered: 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4

#### Classes

- **TestEnhancedModelAvailabilityIntegration**: Integration tests for the complete enhanced model availability system....
- **TestEndToEndScenarios**: End-to-end scenario tests covering complete user workflows....
- **TestPerformanceBenchmarks**: Performance benchmark tests for enhanced features....
- **TestStressTests**: Stress tests for download management and retry logic....
- **TestChaosEngineering**: Chaos engineering tests for failure scenario validation....
- **TestUserAcceptanceScenarios**: User acceptance tests for enhanced model management workflows....

### tests.test_enhanced_model_configuration

Tests for Enhanced Model Configuration Management System

Tests configuration management, validation, feature flags, and API endpoints.

#### Classes

- **TestEnhancedModelConfiguration**: Test configuration data classes and basic functionality...
- **TestConfigurationManager**: Test configuration manager functionality...
- **TestConfigurationValidation**: Test configuration validation functionality...
- **TestConfigurationMigration**: Test configuration migration functionality...
- **TestGlobalConfigurationManager**: Test global configuration manager functionality...
- **TestConfigurationAPI**: Test configuration API endpoints...

### tests.test_enhanced_model_downloader

Comprehensive unit tests for Enhanced Model Downloader
Tests retry logic, download management, bandwidth limiting, and error handling.

#### Classes

- **TestEnhancedModelDownloader**: Test suite for Enhanced Model Downloader...
- **TestRetryConfig**: Test retry configuration...
- **TestBandwidthConfig**: Test bandwidth configuration...
- **TestDownloadProgress**: Test download progress tracking...
- **TestDownloadResult**: Test download result object...

### tests.test_enhanced_model_management_api

Integration Tests for Enhanced Model Management API Endpoints
Tests all new enhanced model management endpoints with various scenarios.

#### Classes

- **TestEnhancedModelManagementAPI**: Test suite for Enhanced Model Management API...
- **TestDetailedModelStatusEndpoint**: Test /api/v1/models/status/detailed endpoint...
- **TestDownloadManagementEndpoint**: Test /api/v1/models/download/manage endpoint...
- **TestHealthMonitoringEndpoint**: Test /api/v1/models/health endpoint...
- **TestUsageAnalyticsEndpoint**: Test /api/v1/models/analytics endpoint...
- **TestStorageCleanupEndpoint**: Test /api/v1/models/cleanup endpoint...
- **TestFallbackSuggestionEndpoint**: Test /api/v1/models/fallback/suggest endpoint...
- **TestAPIIntegration**: Test API integration and error handling...

### tests.test_error_handling

Test error handling and validation for the generation API


### tests.test_fallback_recovery_system

Tests for the Fallback and Recovery System

#### Classes

- **TestFallbackRecoverySystem**: Test cases for the FallbackRecoverySystem...

### tests.test_final_integration_validation

Final integration and validation tests for real AI model integration.
Comprehensive end-to-end testing to ensure all components work together correctly.

#### Classes

- **TestFinalIntegrationValidation**: Comprehensive integration validation tests....
- **TestSystemIntegrationValidation**: Test system integration components....
- **TestGenerationServiceValidation**: Test generation service integration....
- **TestPerformanceMonitoringValidation**: Test performance monitoring integration....
- **TestAPIIntegrationValidation**: Test API integration and compatibility....
- **TestErrorHandlingValidation**: Test error handling and recovery systems....
- **TestConfigurationValidation**: Test configuration and deployment validation....
- **TestPerformanceBenchmarkValidation**: Test performance benchmarks and targets....
- **TestEndToEndValidation**: End-to-end integration validation....

### tests.test_hardware_optimization_integration

Test hardware optimization integration with generation service

#### Classes

- **TestHardwareOptimizationIntegration**: Test hardware optimization integration with generation service...

### tests.test_health_endpoint

Test the health endpoint directly


### tests.test_health_endpoint_integration

Integration tests for the enhanced health endpoint

#### Classes

- **TestHealthEndpointIntegration**: Test the enhanced system health endpoint...
- **TestHealthEndpointConnectivity**: Test health endpoint connectivity validation features...

### tests.test_i2v_ti2v_api

Test I2V/TI2V API endpoints and image upload functionality


### tests.test_integrated_error_handler

Tests for Integrated Error Handler

Tests the enhanced error handling system that bridges FastAPI backend
with existing GenerationErrorHandler infrastructure.

#### Classes

- **TestIntegratedErrorHandler**: Test the IntegratedErrorHandler class...
- **TestConvenienceFunctions**: Test convenience functions for common error scenarios...
- **TestGlobalErrorHandler**: Test global error handler instance management...
- **TestErrorHandlerIntegration**: Test integration with existing error handling infrastructure...
- **TestErrorRecovery**: Test automatic error recovery functionality...
- **TestErrorContextEnhancement**: Test error context enhancement for FastAPI integration...

### tests.test_intelligent_fallback_manager

Unit tests for Intelligent Fallback Manager

Tests model compatibility scoring algorithms, alternative model suggestions,
fallback strategy decision engine, request queuing, and wait time calculations.

#### Classes

- **MockAvailabilityStatus**: Mock availability status for testing...
- **MockModelStatus**: Mock model status for testing...
- **MockAvailabilityManager**: Mock availability manager for testing...
- **TestIntelligentFallbackManager**: Test suite for IntelligentFallbackManager...
- **TestGlobalInstanceManagement**: Test global instance management functions...
- **TestErrorHandling**: Test error handling in various scenarios...
- **TestEdgeCases**: Test edge cases and boundary conditions...

### tests.test_lora_integration

Test LoRA integration in the real generation pipeline

#### Classes

- **TestLoRAIntegration**: Test LoRA integration functionality...

### tests.test_lora_integration_simple

Simple LoRA integration test without external dependencies

#### Classes

- **MockLoRATracker**: ...

### tests.test_lora_validation

Test LoRA parameter validation logic

#### Classes

- **MockGenerationParams**: Mock generation parameters for testing...
- **MockLoRAValidator**: Mock LoRA validator for testing validation logic...
- **TestLoRAValidation**: Test LoRA validation logic without external dependencies...

### tests.test_model_availability_manager

Integration Tests for Model Availability Manager
Tests the central coordination system with existing ModelManager and ModelDownloader

#### Classes

- **TestModelAvailabilityManager**: Test suite for ModelAvailabilityManager...
- **TestModelAvailabilityManagerIntegration**: Integration tests with real components...

### tests.test_model_availability_manager_basic

Basic Tests for Model Availability Manager
Tests core functionality without complex async fixtures


### tests.test_model_availability_manager_simple

Simple Tests for Model Availability Manager
Tests core functionality without complex dependencies

#### Classes

- **TestModelAvailabilityManagerSimple**: Simple test suite for ModelAvailabilityManager core functionality...

### tests.test_model_download_integration

Test script for model download integration
Tests the integration between ModelIntegrationBridge and existing download infrastructure


### tests.test_model_health_monitor

Unit tests for Model Health Monitor
Tests integrity checking, performance monitoring, corruption detection,
and automated health checks.

#### Classes

- **TestModelHealthMonitor**: Test suite for ModelHealthMonitor...
- **TestIntegrityChecking**: Test integrity checking functionality...
- **TestPerformanceMonitoring**: Test performance monitoring functionality...
- **TestCorruptionDetection**: Test corruption detection functionality...
- **TestHealthChecks**: Test scheduled health checks functionality...
- **TestSystemHealthReport**: Test system health reporting functionality...
- **TestUtilityFunctions**: Test utility and convenience functions...
- **TestErrorHandling**: Test error handling and edge cases...

### tests.test_model_integration_bridge

Tests for Model Integration Bridge

#### Classes

- **TestModelIntegrationBridge**: Test cases for ModelIntegrationBridge...

### tests.test_model_integration_comprehensive

Comprehensive Model Integration Tests
Focused testing for ModelIntegrationBridge functionality with all model types

#### Classes

- **TestModelIntegrationBridgeDetailed**: Detailed tests for ModelIntegrationBridge functionality...
- **ModelIntegrationBridge**: ...
- **ModelStatus**: ...
- **ModelType**: ...
- **ModelIntegrationStatus**: ...
- **GenerationParams**: ...
- **GenerationResult**: ...

### tests.test_model_management_endpoints

Test Model Management API Endpoints
Tests for the new model status and management endpoints

#### Classes

- **TestModelManagementEndpoints**: Test cases for model management endpoints...

### tests.test_model_update_manager

Unit tests for Model Update Manager
Tests model version checking, update detection, safe update processes,
rollback capability, and update scheduling functionality.

#### Classes

- **TestModelUpdateManager**: Test suite for ModelUpdateManager...
- **TestModelUpdateManagerIntegration**: Integration tests for ModelUpdateManager with other components...

### tests.test_model_update_manager_simple

Simple tests for Model Update Manager
Basic functionality tests without complex async fixtures.


### tests.test_model_usage_analytics

Unit tests for Model Usage Analytics System
Tests analytics collection, recommendation algorithms, and reporting functionality.

#### Classes

- **TestModelUsageAnalytics**: Test cases for ModelUsageAnalytics class...
- **TestGenerationServiceIntegration**: Test cases for generation service analytics integration...
- **TestAnalyticsReporting**: Test cases for analytics reporting functionality...
- **TestAnalyticsAlgorithms**: Test cases for analytics algorithms and calculations...

### tests.test_performance_benchmarks

Performance Benchmarking Tests
Comprehensive performance testing for generation speed and resource usage

#### Classes

- **PerformanceBenchmarkSuite**: Performance benchmark test suite...
- **TestGenerationPerformanceBenchmarks**: Generation performance benchmark tests...
- **TestAPIPerformanceBenchmarks**: API performance benchmark tests...
- **TestResourceUsageBenchmarks**: Resource usage benchmark tests...
- **TaskStatus**: ...

### tests.test_performance_benchmarks_enhanced

Performance Benchmarking Tests for Enhanced Model Availability Features

This module contains comprehensive performance benchmarks for all enhanced
model availability components, measuring response times, throughput, and
resource utilization under various load conditions.

Requirements covered: 1.4, 2.4, 5.4, 6.4, 8.4

#### Classes

- **PerformanceBenchmark**: Performance benchmark result....
- **LoadTestResult**: Load test result with detailed metrics....
- **PerformanceBenchmarkSuite**: Comprehensive performance benchmarking suite....
- **TestPerformanceBenchmarkSuite**: Pytest wrapper for performance benchmark suite....

### tests.test_performance_integration

Integration tests for Performance Monitoring System with Enhanced Model Components

Tests the integration of performance monitoring with enhanced model downloader,
health monitor, fallback manager, and other components.

#### Classes

- **TestPerformanceIntegrationWithDownloader**: Test performance monitoring integration with enhanced model downloader...
- **TestPerformanceIntegrationWithHealthMonitor**: Test performance monitoring integration with model health monitor...
- **TestPerformanceIntegrationWithFallbackManager**: Test performance monitoring integration with intelligent fallback manager...
- **TestPerformanceSystemIntegration**: Test overall performance monitoring system integration...

### tests.test_performance_monitoring_system

Comprehensive tests for the Performance Monitoring System

Tests performance tracking, resource monitoring, analysis, and optimization
validation for the enhanced model availability system.

#### Classes

- **TestPerformanceTracker**: Test performance tracking functionality...
- **TestSystemResourceMonitor**: Test system resource monitoring...
- **TestPerformanceAnalyzer**: Test performance analysis and reporting...
- **TestPerformanceMonitoringSystem**: Test the main performance monitoring system...
- **TestPerformanceBenchmarks**: Performance benchmarking tests...

### tests.test_queue_api_integration

Integration test for queue API endpoints


### tests.test_queue_simple

Simple test for queue persistence functionality


### tests.test_real_generation_pipeline

Tests for Real Generation Pipeline

#### Classes

- **TestRealGenerationPipeline**: Test suite for RealGenerationPipeline...

### tests.test_simple_validation

Simple validation test to check if basic test structure works

#### Classes

- **TestSimpleValidation**: Simple test class for validation...

### tests.test_t2v_generation

Test T2V generation API endpoint


### tests.test_task_2_3_validation

#### Classes

- **MockUploadFile**: Mock UploadFile for testing...

### tests.test_user_acceptance_workflows

User Acceptance Tests for Enhanced Model Management Workflows

This module contains comprehensive user acceptance tests that validate the
enhanced model management system from the user's perspective, covering
complete workflows and user experience scenarios.

Requirements covered: 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4

#### Classes

- **UserAction**: Represents a user action in a workflow....
- **WorkflowResult**: Result of a user workflow test....
- **UserAcceptanceTestSuite**: Comprehensive user acceptance test suite....
- **TestUserAcceptanceTestSuite**: Pytest wrapper for user acceptance test suite....

### tests.test_websocket_model_notifications

WebSocket Model Notifications Integration Tests
Tests for real-time model status updates, download progress, health monitoring,
and fallback strategy notifications via WebSocket.

#### Classes

- **MockWebSocket**: Mock WebSocket for testing...
- **TestModelDownloadNotifier**: Test model download WebSocket notifications...
- **TestModelHealthNotifier**: Test model health monitoring WebSocket notifications...
- **TestModelAvailabilityNotifier**: Test model availability WebSocket notifications...
- **TestFallbackNotifier**: Test fallback strategy WebSocket notifications...
- **TestAnalyticsNotifier**: Test analytics WebSocket notifications...
- **TestModelNotificationIntegrator**: Test the main model notification integrator...
- **TestWebSocketManagerEnhancements**: Test the enhanced WebSocket manager methods...

### tests.test_websocket_progress_integration

Test WebSocket Progress Integration
Tests the enhanced WebSocket progress tracking system for real AI model generation

#### Classes

- **TestProgressIntegration**: Test the progress integration system...
- **TestWebSocketManagerEnhancements**: Test the enhanced WebSocket manager functionality...
- **TestVRAMMonitoring**: Test VRAM monitoring functionality...

### tests.validate_test_suite

#### Classes

- **TestSuiteValidator**: Validator for comprehensive test suite...

### utils.thumbnail_generator

Thumbnail generation utilities for video files

#### Classes

- **ThumbnailGenerator**: Generate thumbnails for video files using ffmpeg...

### utils.__init__

Utility modules for the backend


### websocket.manager

WebSocket manager for real-time system monitoring and updates
Provides sub-second updates for system stats and generation progress

#### Classes

- **ConnectionManager**: Manages WebSocket connections for real-time updates...

### websocket.model_notifications

Model Notifications WebSocket Integration
Integrates enhanced model availability components with WebSocket real-time notifications.

#### Classes

- **ModelDownloadNotifier**: Handles WebSocket notifications for model download events...
- **ModelHealthNotifier**: Handles WebSocket notifications for model health monitoring events...
- **ModelAvailabilityNotifier**: Handles WebSocket notifications for model availability changes...
- **FallbackNotifier**: Handles WebSocket notifications for fallback strategy events...
- **AnalyticsNotifier**: Handles WebSocket notifications for analytics updates...
- **ModelNotificationIntegrator**: Main integration class that coordinates all model-related WebSocket notifications...

### websocket.progress_integration

WebSocket Progress Integration for Real AI Model Generation
Provides detailed progress tracking and real-time updates for generation pipeline

#### Classes

- **GenerationStage**: Stages of video generation process...
- **ProgressIntegration**: Integration class for sending detailed progress updates via WebSocket
Connects the real generation p...

### websocket.__init__

WebSocket module for real-time updates


### tools.conftest


### tools.registry

Tool Registry
Auto-generated registry of available tools.


### tools.__init__


### tools.branch-analyzer.__init__

Tool: branch-analyzer


### tools.code-review.code_reviewer

Code Review and Refactoring Assistance System

This module provides automated code review suggestions, refactoring recommendations,
and technical debt tracking to improve code quality and maintainability.

#### Classes

- **ReviewSeverity**: Severity levels for code review issues...
- **IssueCategory**: Categories of code review issues...
- **CodeIssue**: Represents a code review issue...
- **RefactoringRecommendation**: Represents a refactoring recommendation...
- **TechnicalDebtItem**: Represents a technical debt item...
- **CodeReviewer**: Main code review and refactoring assistance system...
- **ComplexityAnalyzer**: Analyzes code complexity...
- **MaintainabilityAnalyzer**: Analyzes code maintainability...
- **PerformanceAnalyzer**: Analyzes potential performance issues...
- **SecurityAnalyzer**: Analyzes potential security issues...

### tools.code-review.refactoring_engine

Refactoring Recommendation Engine

This module provides intelligent refactoring recommendations based on code analysis
and quality metrics to help improve code maintainability and performance.

#### Classes

- **RefactoringType**: Types of refactoring recommendations...
- **RefactoringPattern**: Represents a refactoring pattern...
- **RefactoringSuggestion**: Represents a specific refactoring suggestion...
- **RefactoringEngine**: Main refactoring recommendation engine...

### tools.code-review.simple_test

Simple test to verify the code review system works


### tools.code-review.technical_debt_tracker

Technical Debt Tracking and Prioritization System

This module provides comprehensive technical debt tracking, analysis, and prioritization
to help teams manage and reduce technical debt systematically.

#### Classes

- **DebtCategory**: Categories of technical debt...
- **DebtSeverity**: Severity levels for technical debt...
- **DebtStatus**: Status of technical debt items...
- **TechnicalDebtItem**: Represents a technical debt item...
- **DebtMetrics**: Technical debt metrics...
- **DebtRecommendation**: Recommendation for addressing technical debt...
- **TechnicalDebtTracker**: Main technical debt tracking system...

### tools.code-review.test_code_review_system

Test suite for the Code Review and Refactoring Assistance Tools

This test file verifies that the core functionality of the code review system works correctly.


### tools.codebase-cleanup.dead_code_analyzer

#### Classes

- **DeadFunction**: Represents a dead/unused function...
- **DeadClass**: Represents a dead/unused class...
- **UnusedImport**: Represents an unused import...
- **DeadFile**: Represents a dead/unused file...
- **DeadCodeReport**: Report of dead code analysis...
- **DeadCodeAnalyzer**: Comprehensive dead code analysis system that identifies:
- Unused functions and methods
- Dead class...

### tools.codebase-cleanup.duplicate_detector

#### Classes

- **DuplicateFile**: Represents a duplicate file with metadata...
- **DuplicateReport**: Report of duplicate detection analysis...
- **DuplicateDetector**: Comprehensive duplicate detection system that identifies:
- Exact file duplicates (by hash)
- Near-d...

### tools.codebase-cleanup.example_usage

Example usage of the Codebase Cleanup Tools

This script demonstrates how to use the duplicate detection system
to clean up a codebase.


### tools.codebase-cleanup.naming_standardizer

#### Classes

- **NamingViolation**: Represents a naming convention violation...
- **InconsistentPattern**: Represents an inconsistent naming pattern across the codebase...
- **OrganizationSuggestion**: Represents a file organization suggestion...
- **NamingReport**: Report of naming convention analysis...
- **NamingStandardizer**: Comprehensive naming standardization system that:
- Analyzes naming conventions across the codebase
...

### tools.codebase-cleanup.test_dead_code_analyzer

Test suite for dead code analysis functionality


### tools.codebase-cleanup.test_duplicate_detector

Test suite for duplicate detection functionality


### tools.codebase-cleanup.test_integration

Integration test for all codebase cleanup tools


### tools.codebase-cleanup.test_naming_standardizer

Test suite for naming standardization functionality


### tools.codebase-cleanup.__init__

Codebase Cleanup and Organization Tools

This package provides comprehensive tools for cleaning up and organizing codebases:
- Duplicate detection and removal
- Dead code analysis and removal  
- Naming standardization and file organization


### tools.codebase-cleanup.backups.dead_code.dead_code_removal_20250902_202530.backup_sample


### tools.codebase-cleanup.backups.dead_code.dead_code_removal_20250902_202531.backup_sample


### tools.codebase-cleanup.backups.dead_code.dead_code_removal_20250902_224228.backup_sample


### tools.codebase-cleanup.backups.dead_code.dead_code_removal_20250902_234648.backup_sample


### tools.codebase-cleanup.backups.dead_code.dead_code_removal_20250902_234649.backup_sample


### tools.codebase-cleanup.backups.dead_code.dead_code_removal_20250903_102326.backup_sample


### tools.codebase-cleanup.backups.dead_code.dead_code_removal_20250903_102327.backup_sample


### tools.codebase-cleanup.backups.dead_code.dead_code_removal_20250903_111014.backup_sample


### tools.codebase-cleanup.backups.dead_code.dead_code_removal_20250903_111015.backup_sample


### tools.codebase-cleanup.backups.dead_code.dead_code_removal_20250903_120256.backup_sample


### tools.codebase-cleanup.backups.dead_code.dead_code_removal_20250903_120257.backup_sample


### tools.codebase-cleanup.backups.duplicates.duplicate_removal_20250902_202531.file2


### tools.codebase-cleanup.backups.duplicates.duplicate_removal_20250902_202532.file2


### tools.codebase-cleanup.backups.duplicates.duplicate_removal_20250902_224229.file2


### tools.codebase-cleanup.backups.duplicates.duplicate_removal_20250902_234650.file2


### tools.codebase-cleanup.backups.duplicates.duplicate_removal_20250903_102327.file2


### tools.codebase-cleanup.backups.duplicates.duplicate_removal_20250903_102328.file2


### tools.codebase-cleanup.backups.duplicates.duplicate_removal_20250903_111015.file2


### tools.codebase-cleanup.backups.duplicates.duplicate_removal_20250903_111016.file2


### tools.codebase-cleanup.backups.duplicates.duplicate_removal_20250903_120257.file2


### tools.codebase-cleanup.backups.duplicates.duplicate_removal_20250903_120258.file2


### tools.code_quality.ci_quality_check

Simplified code quality checker for CI environments.
This version has minimal dependencies and runs basic quality checks.


### tools.code_quality.cli

Command-line interface for code quality checking system.


### tools.code_quality.demo

Demonstration of the code quality checking system.

#### Classes

- **QualityIssueType**: ...
- **QualitySeverity**: ...
- **QualityIssue**: ...
- **QualityMetrics**: ...
- **QualityReport**: ...
- **SimpleQualityChecker**: Simplified quality checker for demonstration....

### tools.code_quality.demo_enforcement_system

Demonstration of the automated quality enforcement system.


### tools.code_quality.example_usage

Example usage of the code quality checking system.


### tools.code_quality.integration_test

Integration test for the code quality checking system.


### tools.code_quality.models

Data models for code quality checking system.

#### Classes

- **QualityIssueType**: Types of quality issues that can be detected.

Attributes:
    FORMATTING: Code formatting issues (s...
- **QualitySeverity**: Severity levels for quality issues.

Attributes:
    ERROR: Critical issues that must be fixed
    W...
- **QualityIssue**: Represents a single code quality issue.

Attributes:
    file_path: Path to the file containing the ...
- **QualityMetrics**: Code quality metrics for a file or project.

Attributes:
    total_lines: Total number of lines in t...
- **QualityReport**: Comprehensive quality report for code analysis.

Attributes:
    timestamp: When the report was gene...
- **QualityConfig**: Configuration for quality checking.

Attributes:
    line_length: Maximum line length for formatting...

### tools.code_quality.quality_checker

Main code quality checking engine.

#### Classes

- **QualityChecker**: Main code quality checking engine....

### tools.code_quality.test_basic_functionality

Basic functionality test for code quality system.


### tools.code_quality.test_enforcement_system

Test suite for the automated quality enforcement system.

#### Classes

- **TestPreCommitHookManager**: Test pre-commit hook management....
- **TestCIIntegration**: Test CI/CD integration....
- **TestEnforcementCLI**: Test enforcement CLI....
- **TestIntegration**: Integration tests for the enforcement system....

### tools.code_quality.__init__

Code Quality Checking System

A comprehensive system for enforcing code quality standards including
formatting, style, documentation, type hints, and complexity analysis.


### tools.code_quality.analyzers.complexity_analyzer

Code complexity analysis and recommendations.

#### Classes

- **ComplexityAnalyzer**: Analyzes code complexity and provides refactoring recommendations....

### tools.code_quality.analyzers.__init__

Code analysis modules.


### tools.code_quality.enforcement.ci_integration

CI/CD integration for automated quality checking.

#### Classes

- **CIIntegration**: Manages CI/CD integration for code quality enforcement....

### tools.code_quality.enforcement.enforcement_cli

CLI for automated quality enforcement system.

#### Classes

- **EnforcementCLI**: Command-line interface for quality enforcement....

### tools.code_quality.enforcement.pre_commit_hooks

Pre-commit hook management for automated quality enforcement.

#### Classes

- **PreCommitHookManager**: Manages pre-commit hooks for code quality enforcement....

### tools.code_quality.enforcement.__init__

Code quality enforcement module.

This module provides automated quality enforcement through:
- Pre-commit hooks for local development
- CI/CD integration for automated checking
- Quality metrics tracking and reporting


### tools.code_quality.formatters.code_formatter

Code formatting checker and fixer.

#### Classes

- **CodeFormatter**: Handles code formatting checking and fixing....

### tools.code_quality.formatters.__init__

Code formatting modules.


### tools.code_quality.validators.documentation_validator

Documentation completeness validator.

#### Classes

- **DocumentationValidator**: Validates documentation completeness for functions, classes, and modules....

### tools.code_quality.validators.style_validator

Style validation using flake8 and custom rules.

#### Classes

- **StyleValidator**: Validates code style using flake8 and custom rules....

### tools.code_quality.validators.type_hint_validator

Type hint validation and enforcement.

#### Classes

- **TypeHintValidator**: Validates type hints for functions and methods....

### tools.code_quality.validators.__init__

Code validation modules.


### tools.config-analyzer.cli

Configuration Analysis CLI

Command-line interface for configuration landscape analysis.


### tools.config-analyzer.config_dependency_mapper

#### Classes

- **ConfigMapping**: Maps configuration files to their purpose and relationships....
- **ConfigDependencyMapper**: Maps configuration dependencies for consolidation planning....

### tools.config-analyzer.config_landscape_analyzer

#### Classes

- **ConfigFile**: Represents a configuration file in the project....
- **ConfigConflict**: Represents a conflict between configuration settings....
- **ConfigDependency**: Represents a dependency relationship between configs....
- **ConfigAnalysisReport**: Complete analysis report of the configuration landscape....
- **ConfigLandscapeAnalyzer**: Analyzes the configuration landscape of a project....

### tools.config-analyzer.__init__

Tool: config-analyzer


### tools.config_manager.config_api

Configuration API and Management System

This module provides a comprehensive API for configuration management,
including get/set operations, hot-reloading, and change notifications.

#### Classes

- **ConfigChangeEvent**: Represents a configuration change event...
- **ConfigurationChangeHandler**: Handles configuration change notifications...
- **ConfigFileWatcher**: Watches configuration files for changes...
- **ConfigurationAPI**: Comprehensive configuration API with hot-reloading and change notifications...

### tools.config_manager.config_cli

Configuration Management CLI

Command-line interface for managing unified configuration,
including get/set operations, validation, and monitoring.


### tools.config_manager.config_unifier

#### Classes

- **ConfigSource**: Represents a discovered configuration source...
- **MigrationReport**: Report of configuration migration process...
- **ConfigurationUnifier**: Handles migration of scattered configuration files to unified system...

### tools.config_manager.config_validator

Configuration Validation System

This module provides comprehensive validation for the unified configuration,
including schema validation, dependency checking, and consistency validation.

#### Classes

- **ValidationSeverity**: Severity levels for validation issues...
- **ValidationIssue**: Represents a configuration validation issue...
- **ValidationResult**: Result of configuration validation...
- **ConfigurationValidator**: Comprehensive configuration validation system...

### tools.config_manager.migration_cli

Configuration Migration CLI Tool

Command-line interface for migrating scattered configuration files
to the unified configuration system.


### tools.config_manager.pre_commit_config_validation

Pre-commit hook for configuration validation.
Validates configuration files against schemas and checks for consistency.


### tools.config_manager.unified_config

Unified Configuration Schema

This module defines the comprehensive configuration schema for the WAN22 project,
including all system, service, and environment settings with validation rules.

#### Classes

- **LogLevel**: Supported logging levels...
- **QuantizationLevel**: Supported model quantization levels...
- **Environment**: Supported deployment environments...
- **SystemConfig**: Core system configuration...
- **APIConfig**: API server configuration...
- **DatabaseConfig**: Database configuration...
- **ModelConfig**: Model management configuration...
- **HardwareConfig**: Hardware optimization configuration...
- **GenerationConfig**: Video generation configuration...
- **UIConfig**: User interface configuration...
- **FrontendConfig**: Frontend application configuration...
- **WebSocketConfig**: WebSocket configuration...
- **LoggingConfig**: Logging configuration...
- **SecurityConfig**: Security configuration...
- **PerformanceConfig**: Performance monitoring configuration...
- **RecoveryConfig**: Error recovery configuration...
- **EnvironmentValidationConfig**: Environment validation configuration...
- **PromptEnhancementConfig**: Prompt enhancement configuration...
- **FeatureFlags**: Feature flags configuration...
- **EnvironmentOverrides**: Environment-specific configuration overrides...
- **UnifiedConfig**: Unified configuration schema for the WAN22 project.

This class provides a comprehensive configurati...

### tools.config_manager.__init__

Configuration Management System

This module provides unified configuration management for the WAN22 project,
including schema definition, validation, migration, and API access.


### tools.deployment-gates.deployment_gate_status

Deployment Gate Status Checker
This script provides a comprehensive status check for deployment gates


### tools.deployment-gates.simple_test_runner

Simple test runner for deployment gates
This provides a minimal working test runner for CI/CD when pytest has issues


### tools.dev-environment.dependency_detector

Dependency Detection and Installation Guidance

This module provides automated dependency detection and installation guidance
for the WAN22 development environment.

#### Classes

- **DependencyInfo**: Information about a dependency...
- **SystemInfo**: System information...
- **DependencyDetector**: Detects and validates development dependencies...

### tools.dev-environment.dev_environment_cli

Development Environment CLI

Interactive command-line interface for development environment management.


### tools.dev-environment.environment_validator

Development Environment Validator

This module provides comprehensive validation and health checking
for the WAN22 development environment.

#### Classes

- **ValidationResult**: Result of a validation check...
- **EnvironmentHealth**: Overall environment health status...
- **EnvironmentValidator**: Validates development environment setup and health...

### tools.dev-environment.setup_dev_environment

Automated Development Environment Setup

This module provides automated setup for the WAN22 development environment,
including dependency installation, configuration, and validation.

#### Classes

- **DevEnvironmentSetup**: Automated development environment setup...

### tools.dev-environment.__init__

Tool: dev-environment


### tools.dev-feedback.config_watcher

Configuration Watcher with Hot-Reloading

This module provides hot-reloading for configuration changes during development.

#### Classes

- **ConfigChange**: Configuration change event...
- **ServiceConfig**: Service configuration for reloading...
- **ConfigFileHandler**: File system event handler for configuration watching...
- **ConfigWatcher**: Watch configuration files and provide hot-reloading...

### tools.dev-feedback.debug_tools

Debug Tools with Comprehensive Logging

This module provides debugging tools with comprehensive logging and error reporting.

#### Classes

- **LogEntry**: Log entry structure...
- **ErrorPattern**: Error pattern for analysis...
- **DebugSession**: Debug session information...
- **DebugLogHandler**: Custom log handler for debug tools...
- **DebugTools**: Comprehensive debugging tools...

### tools.dev-feedback.feedback_cli

Fast Feedback Development Tools CLI

Interactive command-line interface for fast feedback development tools.


### tools.dev-feedback.test_watcher

#### Classes

- **TestResult**: Test execution result...
- **WatchConfig**: Configuration for test watcher...
- **TestFileHandler**: File system event handler for test watching...
- **TestWatcher**: Watch files and run tests with selective execution...

### tools.dev-feedback.__init__

Tool: dev-feedback


### tools.doc_generator.cli

Documentation Generator CLI

Unified command-line interface for all WAN22 documentation tools.
Provides commands for generation, validation, serving, and management.


### tools.doc_generator.documentation_generator

#### Classes

- **DocumentationPage**: Represents a documentation page with metadata...
- **APIDocumentation**: Represents API documentation extracted from code...
- **MigrationReport**: Report of documentation migration process...
- **DocumentationGenerator**: Main class for consolidating existing documentation and generating API docs...

### tools.doc_generator.generate_docs

Documentation Generation CLI

Command-line interface for generating consolidated documentation
and API documentation from code annotations.


### tools.doc_generator.metadata_manager

Documentation Metadata Manager

Manages documentation metadata, cross-references, and relationships
between documentation pages.

#### Classes

- **DocumentMetadata**: Metadata for a documentation page...
- **CrossReference**: Cross-reference between documentation pages...
- **DocumentationIndex**: Complete documentation index with metadata and relationships...
- **MetadataManager**: Manages documentation metadata and cross-references...

### tools.doc_generator.migration_tool

Documentation Migration Tool

Specialized tool for migrating scattered documentation files
to a unified structure with proper categorization and metadata.

#### Classes

- **MigrationRule**: Rule for migrating documentation files...
- **DocumentationMigrator**: Tool for migrating scattered documentation to unified structure...

### tools.doc_generator.navigation_generator

Navigation Generator

Automatically generates navigation menus and site structure
for WAN22 documentation based on file organization and metadata.

#### Classes

- **NavigationItem**: Navigation menu item...
- **NavigationConfig**: Configuration for navigation generation...
- **NavigationGenerator**: Generates navigation structure for documentation...

### tools.doc_generator.pre_commit_link_check

Pre-commit hook for documentation link checking.
Validates internal links in markdown files.


### tools.doc_generator.search_indexer

Documentation Search Indexer

Advanced search indexing for WAN22 documentation with full-text search,
tag-based filtering, and content categorization.

#### Classes

- **SearchDocument**: Document for search indexing...
- **SearchResult**: Search result with relevance scoring...
- **SearchIndexer**: Advanced search indexer for documentation...

### tools.doc_generator.server

Documentation Server

Static site generator and server for WAN22 documentation with search functionality.
Uses MkDocs for static site generation and provides development server capabilities.

#### Classes

- **ServerConfig**: Configuration for documentation server...
- **DocumentationServer**: Documentation server using MkDocs for static site generation...

### tools.doc_generator.validator

Documentation Validator

Validates documentation for broken links, content quality,
freshness, and compliance with style guidelines.

#### Classes

- **ValidationIssue**: Represents a validation issue...
- **ValidationReport**: Complete validation report...
- **LinkCheckResult**: Result of link checking...
- **DocumentationValidator**: Comprehensive documentation validator...

### tools.doc_generator.__init__

Documentation Generator Package

This package provides tools for consolidating scattered documentation,
generating API documentation from code annotations, and managing
documentation migration.


### tools.health-checker.analyze_trends

Analyze health trends from current and historical health reports


### tools.health-checker.automated_alerting

Automated Alerting and Notification System

This module handles automated alerting for critical health issues,
escalation policies, and integration with project management tools.

#### Classes

- **AlertLevel**: Alert severity levels...
- **AlertRule**: Configuration for an alert rule...
- **EscalationPolicy**: Escalation policy configuration...
- **AlertHistory**: Track alert history for rate limiting and escalation...
- **AutomatedAlertingSystem**: Automated alerting system with escalation and rate limiting...

### tools.health-checker.automated_monitoring

Automated health monitoring and alerting system.

This module provides automated health monitoring with trend tracking,
alerting, and continuous improvement recommendations.

#### Classes

- **AutomatedHealthMonitor**: Automated health monitoring system with continuous improvement tracking....

### tools.health-checker.badge_generator

Health status badge generator for project visibility.


### tools.health-checker.baseline_and_improvement

Comprehensive baseline establishment and continuous improvement implementation.

This script implements task 9.3: Establish baseline metrics and continuous improvement
- Run comprehensive health analysis to establish current baseline
- Create health improvement roadmap based on current issues
- Implement automated health trend tracking and alerting

#### Classes

- **BaselineAndImprovementManager**: Manages baseline establishment and continuous improvement for project health....
- **ProjectHealthChecker**: ...
- **BaselineEstablisher**: ...
- **ContinuousImprovementTracker**: ...
- **AutomatedHealthMonitor**: ...
- **RecommendationEngine**: ...
- **Severity**: ...

### tools.health-checker.bulletproof_health_check

Bulletproof Health Check for CI
This is a completely self-contained health check that avoids all potential CI issues


### tools.health-checker.ci_health_check

Ultra-simple health check specifically for CI environments
This is designed to be fast, reliable, and never hang


### tools.health-checker.ci_integration

CI/CD integration utilities for health monitoring.

This module provides utilities for integrating health monitoring
with CI/CD pipelines and deployment gates.

#### Classes

- **CIHealthIntegration**: Integrates health monitoring with CI/CD pipelines....

### tools.health-checker.cli

Command-line interface for the health monitoring system

#### Classes

- **HealthMonitorCLI**: Command-line interface for health monitoring...

### tools.health-checker.dashboard_server

Simple health monitoring dashboard server

#### Classes

- **HealthDashboard**: Real-time health monitoring dashboard...

### tools.health-checker.establish_baseline

Baseline metrics establishment for health monitoring system.

This script establishes baseline health metrics for the project and sets up
continuous improvement tracking and alerting.

#### Classes

- **BaselineEstablisher**: Establishes and manages baseline health metrics....
- **ContinuousImprovementTracker**: Tracks continuous improvement metrics and trends....

### tools.health-checker.example_usage

Example usage of the health monitoring system


### tools.health-checker.generate_dashboard

Generate HTML dashboard from health report


### tools.health-checker.health_analytics

Health analytics and trend analysis system

#### Classes

- **HealthAnalytics**: Advanced analytics for health monitoring data...

### tools.health-checker.health_checker

Main project health checker implementation

#### Classes

- **ProjectHealthChecker**: Main health checker that orchestrates all health checks and generates reports...
- **Severity**: ...
- **HealthCategory**: ...
- **HealthConfig**: ...
- **HealthIssue**: ...
- **ComponentHealth**: ...
- **HealthTrends**: ...
- **HealthReport**: ...
- **TestHealthChecker**: ...
- **DocumentationHealthChecker**: ...
- **ConfigurationHealthChecker**: ...
- **CodeQualityChecker**: ...
- **HealthCheckCache**: ...
- **PerformanceProfiler**: ...
- **LightweightHealthChecker**: ...
- **HealthCheckTask**: ...
- **ParallelHealthExecutor**: ...

### tools.health-checker.health_models

Health monitoring data models and enums

#### Classes

- **Severity**: Issue severity levels...
- **HealthCategory**: Health check categories...
- **HealthIssue**: Represents a project health issue...
- **Recommendation**: Actionable recommendation for improvement...
- **ComponentHealth**: Health status for a specific component...
- **HealthTrends**: Historical health trend data...
- **HealthReport**: Comprehensive project health report...
- **HealthConfig**: Configuration for health monitoring...

### tools.health-checker.health_notifier

Health notification and alerting system

#### Classes

- **NotificationChannel**: Base class for notification channels...
- **ConsoleNotificationChannel**: Console/stdout notification channel...
- **EmailNotificationChannel**: Email notification channel...
- **SlackNotificationChannel**: Slack notification channel...
- **WebhookNotificationChannel**: Generic webhook notification channel...
- **FileNotificationChannel**: File-based notification channel...
- **HealthNotifier**: Main health notification system that manages multiple channels and alert rules...
- **CIPipelineIntegration**: Integration with CI/CD pipelines...

### tools.health-checker.health_reporter

Health reporting and analytics system

#### Classes

- **HealthReporter**: Generates comprehensive health reports with analytics and visualizations...

### tools.health-checker.parallel_executor

Parallel execution system for health checks.

This module provides parallel and asynchronous execution capabilities
for health checks to improve performance in CI/CD environments.

#### Classes

- **HealthCheckTask**: Represents a health check task for parallel execution....
- **TaskResult**: Represents the result of a health check task....
- **ResourceMonitor**: Monitors system resources during parallel execution....
- **DependencyResolver**: Resolves task dependencies for parallel execution....
- **ParallelHealthExecutor**: Executes health checks in parallel with resource management....
- **AsyncHealthExecutor**: Asynchronous executor for I/O-bound health checks....

### tools.health-checker.performance_optimizer

Performance optimizer for health monitoring system.

This module provides performance profiling, caching, and optimization
capabilities for health checks to ensure fast execution in CI/CD environments.

#### Classes

- **HealthCheckCache**: Caching system for health check results....
- **PerformanceProfiler**: Performance profiler for health checks....
- **IncrementalAnalyzer**: Incremental analysis system for large codebases....
- **LightweightHealthChecker**: Lightweight health checker for frequent execution....

### tools.health-checker.pre_commit_import_validation

Pre-commit hook for import path validation.
Validates that import statements are correct and follow project structure.

#### Classes

- **ImportVisitor**: AST visitor to collect import statements....

### tools.health-checker.pre_commit_test_health

Pre-commit hook for test health checking.
Validates that tests are properly organized and functional.


### tools.health-checker.production_deployment

Production Health Monitoring Deployment System

This module handles the deployment and configuration of health monitoring
for production environments with appropriate thresholds and reporting.

#### Classes

- **ProductionHealthConfig**: Production-specific health monitoring configuration...
- **ProductionHealthMonitor**: Production health monitoring system with automated reporting and alerting...

### tools.health-checker.production_deployment_simple

Simplified Production Health Monitoring Deployment System

This module handles the deployment and configuration of health monitoring
for production environments with standard library dependencies only.

#### Classes

- **ProductionHealthConfig**: Production-specific health monitoring configuration...
- **ProductionHealthMonitor**: Production health monitoring system with automated reporting and alerting...

### tools.health-checker.production_health_checks

Production-Specific Health Checks

This module implements health checks specifically designed for production environments,
including performance monitoring, security validation, and system resource checks.

#### Classes

- **ProductionHealthResult**: Result of production-specific health check...
- **ProductionHealthChecker**: Production-specific health checks for system validation...

### tools.health-checker.recommendation_engine

Actionable recommendations engine for project health improvements

#### Classes

- **RecommendationRule**: Base class for recommendation rules...
- **TestSuiteRecommendationRule**: Recommendations for test suite improvements...
- **DocumentationRecommendationRule**: Recommendations for documentation improvements...
- **ConfigurationRecommendationRule**: Recommendations for configuration improvements...
- **CodeQualityRecommendationRule**: Recommendations for code quality improvements...
- **CriticalIssueRecommendationRule**: Recommendations for addressing critical issues...
- **TrendBasedRecommendationRule**: Recommendations based on health trends...
- **RecommendationEngine**: Main recommendation engine that generates actionable improvement suggestions...

### tools.health-checker.run_baseline_implementation

Simplified baseline implementation runner for Task 9.3


### tools.health-checker.run_health_check

Wrapper script for running health checks from CI/CD workflows
This script provides the interface expected by the GitHub Actions workflow

#### Classes

- **TimeoutError**: ...

### tools.health-checker.run_performance_tests

Performance testing script for health monitoring system.

This script runs various performance tests to validate and optimize
the health monitoring system performance.

#### Classes

- **HealthPerformanceTester**: Performance tester for health monitoring system....

### tools.health-checker.simple_health_check

Simple health check script for deployment gates
This provides a minimal working health check for CI/CD


### tools.health-checker.start_health_monitoring

Simple health monitoring service.


### tools.health-checker.task_9_3_implementation

Task 9.3 Implementation: Establish baseline metrics and continuous improvement

This script implements the complete baseline establishment and continuous improvement
system as specified in task 9.3 of the project health improvements spec.

Requirements addressed:
- 4.4: Health trend analysis and historical tracking
- 4.6: Health reporting and analytics
- 4.8: Automated health notifications and alerting

#### Classes

- **Task93Implementation**: Complete implementation of Task 9.3: Establish baseline metrics and continuous improvement...

### tools.health-checker.test_monitoring

Test the monitoring service with a single run.


### tools.health-checker.checkers.code_quality_checker

#### Classes

- **CodeQualityChecker**: Checks code quality metrics and issues...

### tools.health-checker.checkers.configuration_health_checker

#### Classes

- **ConfigurationHealthChecker**: Checks the health of project configuration...

### tools.health-checker.checkers.documentation_health_checker

Documentation health checker

#### Classes

- **DocumentationHealthChecker**: Checks the health of project documentation...

### tools.health-checker.checkers.__init__

Individual health checker implementations


### tools.maintenance-reporter.example_usage

Example usage of the comprehensive maintenance reporting system.
Demonstrates all major features and workflows.


### tools.maintenance-reporter.models

Maintenance reporting data models and types.

#### Classes

- **MaintenanceOperationType**: Types of maintenance operations....
- **MaintenanceStatus**: Status of maintenance operations....
- **ImpactLevel**: Impact level of maintenance operations....
- **MaintenanceOperation**: Individual maintenance operation record....
- **MaintenanceImpactAnalysis**: Analysis of maintenance operation impact....
- **MaintenanceRecommendation**: Maintenance recommendation based on project analysis....
- **MaintenanceScheduleOptimization**: Maintenance scheduling optimization analysis....
- **MaintenanceReport**: Comprehensive maintenance report....
- **MaintenanceAuditTrail**: Audit trail for maintenance operations....

### tools.maintenance-reporter.test_integration

Integration tests for the comprehensive maintenance reporting system.

#### Classes

- **TestMaintenanceReportingIntegration**: Integration tests for the maintenance reporting system....

### tools.maintenance-reporter.__init__

Tool: maintenance-reporter


### tools.maintenance-scheduler.cli

Command-line interface for the automated maintenance scheduling system.


### tools.maintenance-scheduler.example_usage

Example usage of the Automated Maintenance Scheduling System.

This script demonstrates how to:
1. Create and configure maintenance tasks
2. Set up scheduling
3. Execute tasks with rollback capabilities
4. Monitor execution and metrics


### tools.maintenance-scheduler.history_tracker

History tracking system for maintenance operations.

#### Classes

- **MaintenanceHistoryTracker**: Tracks and manages maintenance operation history and metrics....

### tools.maintenance-scheduler.models

Data models for the automated maintenance scheduling system.

#### Classes

- **TaskPriority**: Priority levels for maintenance tasks....
- **TaskStatus**: Status of maintenance tasks....
- **TaskCategory**: Categories of maintenance tasks....
- **MaintenanceTask**: Represents a maintenance task to be scheduled and executed....
- **MaintenanceResult**: Result of a maintenance task execution....
- **TaskSchedule**: Schedule configuration for maintenance tasks....
- **MaintenanceHistory**: Historical record of maintenance operations....
- **MaintenanceMetrics**: Metrics for maintenance system performance....

### tools.maintenance-scheduler.priority_engine

Priority engine for maintenance tasks based on impact and effort analysis.

#### Classes

- **ImpactAnalysis**: Analysis of task impact on project health....
- **EffortAnalysis**: Analysis of effort required to complete a task....
- **TaskPriorityEngine**: Engine for calculating task priorities based on impact and effort analysis.

Uses a sophisticated sc...

### tools.maintenance-scheduler.rollback_manager

#### Classes

- **RollbackPoint**: Represents a rollback point for maintenance operations....
- **RollbackManager**: Manages rollback points and operations for safe automated maintenance.

Provides capabilities to:
- ...

### tools.maintenance-scheduler.scheduler

Main maintenance scheduler that orchestrates automated maintenance tasks.

#### Classes

- **MaintenanceScheduler**: Main scheduler for automated maintenance tasks.

Handles task scheduling, execution, prioritization,...

### tools.maintenance-scheduler.task_manager

Task management system for maintenance tasks.

#### Classes

- **MaintenanceTaskManager**: Manages maintenance tasks including storage, retrieval, and lifecycle management....

### tools.maintenance-scheduler.test_basic


### tools.maintenance-scheduler.test_integration

Integration tests for the automated maintenance scheduling system.

#### Classes

- **TestMaintenanceSchedulerIntegration**: Integration tests for the complete maintenance scheduling system....
- **TestMaintenanceSchedulerCLI**: Test CLI functionality....

### tools.onboarding.developer_checklist

Developer Checklist and Progress Tracking

This module provides a comprehensive checklist for new developers
and tracks their onboarding progress.

#### Classes

- **ChecklistItem**: Individual checklist item...
- **ChecklistProgress**: Overall checklist progress...
- **DeveloperChecklist**: Manages developer onboarding checklist and progress tracking...

### tools.onboarding.onboarding_cli

Onboarding CLI

Interactive command-line interface for comprehensive developer onboarding.


### tools.onboarding.setup_wizard

Automated Setup Wizard

This module provides an interactive setup wizard for new developers,
combining environment setup, dependency installation, and progress tracking.

#### Classes

- **SetupWizard**: Interactive setup wizard for new developers...

### tools.onboarding.__init__

Tool: onboarding


### tools.project-structure-analyzer.complexity_analyzer

#### Classes

- **FileComplexity**: Complexity metrics for a single file....
- **ComponentComplexity**: Complexity metrics for a component/directory....
- **ProjectComplexityReport**: Complete project complexity analysis....
- **ProjectComplexityAnalyzer**: Analyzes project complexity and identifies documentation needs....

### tools.project-structure-analyzer.component_analyzer

#### Classes

- **ImportInfo**: Information about an import statement....
- **ComponentDependency**: Represents a dependency between components....
- **ComponentInfo**: Information about a project component....
- **ComponentRelationshipMap**: Complete map of component relationships....
- **ComponentRelationshipAnalyzer**: Analyzes relationships and dependencies between project components....

### tools.project-structure-analyzer.documentation_generator

#### Classes

- **DocumentationSection**: Represents a section of documentation....
- **DocumentationTemplate**: Template for generating documentation....
- **DocumentationGenerator**: Generates comprehensive project documentation....

### tools.project-structure-analyzer.documentation_validator

Documentation Validation and Maintenance System

Implements documentation link checking, freshness validation,
completeness analysis, and accessibility checking.

#### Classes

- **LinkValidationResult**: Result of link validation....
- **DocumentationIssue**: Represents a documentation issue....
- **DocumentationMetrics**: Metrics about documentation quality....
- **DocumentationValidationReport**: Complete documentation validation report....
- **DocumentationValidator**: Validates and maintains project documentation....

### tools.project-structure-analyzer.example_usage

Example usage of the Project Structure Analysis Engine

This script demonstrates how to use the various components
of the project structure analyzer programmatically.


### tools.project-structure-analyzer.structure_analyzer

#### Classes

- **FileInfo**: Information about a file in the project....
- **DirectoryInfo**: Information about a directory in the project....
- **ProjectStructure**: Complete project structure analysis....
- **ProjectStructureAnalyzer**: Analyzes project directory structure and identifies components....

### tools.project-structure-analyzer.visualization_generator

Mermaid Visualization Generator

Generates Mermaid diagrams for project structure visualization
including component relationships and dependency graphs.

#### Classes

- **MermaidDiagram**: Represents a Mermaid diagram....
- **MermaidVisualizationGenerator**: Generates Mermaid diagrams for project visualization....

### tools.quality-monitor.example_usage

Example usage of the quality monitoring and alerting system.


### tools.quality-monitor.models

Quality monitoring data models and types.

#### Classes

- **AlertSeverity**: Alert severity levels....
- **MetricType**: Types of quality metrics....
- **TrendDirection**: Trend direction for metrics....
- **QualityMetric**: Individual quality metric data point....
- **QualityTrend**: Quality trend analysis for a metric....
- **QualityAlert**: Quality alert for regressions or maintenance needs....
- **QualityThreshold**: Quality threshold configuration....
- **QualityRecommendation**: Automated quality improvement recommendation....
- **QualityDashboard**: Quality monitoring dashboard data....

### tools.test-auditor.cli

Test Auditor CLI

Command-line interface for the comprehensive test suite auditor.
Provides various commands for analyzing and reporting on test suite health.

#### Classes

- **TestAuditorCLI**: Command-line interface for test auditor...

### tools.test-auditor.coverage_analyzer

#### Classes

- **FileCoverage**: Coverage information for a single file...
- **CoverageGap**: Represents a gap in test coverage...
- **CoverageReport**: Complete coverage analysis report...
- **CoverageThresholdManager**: Manages coverage thresholds and violations...
- **CoverageDataCollector**: Collects coverage data from various sources...
- **CoverageGapAnalyzer**: Analyzes coverage gaps and generates recommendations...
- **CoverageAnalyzer**: Main coverage analyzer that orchestrates all analysis...

### tools.test-auditor.example_usage

Test Auditor Example Usage

This script demonstrates how to use the comprehensive test auditor system
to analyze test suite health and generate actionable improvement plans.


### tools.test-auditor.orchestrator

Test Suite Audit Orchestrator

Main orchestrator that coordinates all test auditing components to provide
a comprehensive analysis of the test suite health and quality.

#### Classes

- **ComprehensiveTestAnalysis**: Complete test suite analysis combining all audit components...
- **TestSuiteHealthScorer**: Calculates overall health score for test suite...
- **ActionPlanGenerator**: Generates actionable plans to improve test suite health...
- **TestSuiteOrchestrator**: Main orchestrator for comprehensive test suite analysis...

### tools.test-auditor.test_auditor

#### Classes

- **TestIssue**: Represents a specific issue found in a test...
- **TestFileAnalysis**: Analysis results for a single test file...
- **TestSuiteAuditReport**: Complete audit report for the entire test suite...
- **TestDiscoveryEngine**: Discovers and categorizes test files across the project...
- **TestDependencyAnalyzer**: Analyzes test dependencies, imports, and fixtures...
- **TestPerformanceProfiler**: Profiles test execution performance to identify slow tests...
- **TestAuditor**: Main test auditor that orchestrates all analysis components...

### tools.test-auditor.test_runner

#### Classes

- **TestExecutionResult**: Result of executing a single test file...
- **TestSuiteExecutionReport**: Complete execution report for test suite...
- **TestIsolationManager**: Manages test isolation and cleanup...
- **TestTimeoutManager**: Manages test timeouts with configurable limits...
- **TestRetryManager**: Manages test retry logic for flaky tests...
- **TestCoverageCollector**: Collects test coverage information...
- **TestExecutor**: Executes individual test files with full isolation and monitoring...
- **ParallelTestRunner**: Runs tests in parallel with resource management...

### tools.test-quality.coverage_cli

Coverage Analysis CLI

Command-line interface for the comprehensive test coverage analysis system.
Provides easy access to coverage analysis, threshold enforcement, and trend tracking.


### tools.test-quality.coverage_system

Comprehensive Test Coverage Analysis System

Enhanced coverage reporting system that identifies untested code paths,
enforces coverage thresholds for new code, generates detailed reports
with actionable recommendations, and tracks coverage trends over time.

#### Classes

- **CoverageTrend**: Coverage trend data point...
- **NewCodeCoverage**: Coverage analysis for new/changed code...
- **CoverageThresholdResult**: Result of coverage threshold enforcement...
- **CoverageTrendTracker**: Tracks coverage trends over time...
- **NewCodeCoverageAnalyzer**: Analyzes coverage for new/changed code...
- **CoverageThresholdEnforcer**: Enforces coverage thresholds for new code...
- **DetailedCoverageReporter**: Generates detailed coverage reports with actionable recommendations...
- **ComprehensiveCoverageSystem**: Main system that orchestrates all coverage analysis components...

### tools.test-quality.example_usage

Example Usage of Comprehensive Coverage System

This script demonstrates how to use the coverage analysis system
programmatically and shows various analysis scenarios.


### tools.test-quality.flaky_test_detector

#### Classes

- **TestExecution**: Single test execution record...
- **FlakyTestPattern**: Pattern analysis for a flaky test...
- **FlakyTestRecommendation**: Recommendation for fixing a flaky test...
- **QuarantineDecision**: Decision about quarantining a flaky test...
- **FlakyTestStatisticalAnalyzer**: Performs statistical analysis to identify flaky tests...
- **FlakyTestTracker**: Tracks flaky test executions and maintains historical data...
- **FlakyTestRecommendationEngine**: Generates recommendations for fixing flaky tests...
- **FlakyTestQuarantineManager**: Manages quarantine decisions for flaky tests...
- **FlakyTestDetectionSystem**: Main system that orchestrates flaky test detection and management...

### tools.test-quality.integration_example

Test Quality Integration Example

Demonstrates how to integrate all test quality tools into a comprehensive
development workflow, including CI/CD integration and automated monitoring.

#### Classes

- **TestQualityIntegration**: Integrated test quality management system...

### tools.test-quality.performance_optimizer

#### Classes

- **TestPerformanceMetric**: Performance metrics for a single test...
- **TestPerformanceProfile**: Complete performance profile for test execution...
- **PerformanceRegression**: Detected performance regression...
- **OptimizationRecommendation**: Performance optimization recommendation...
- **TestPerformanceProfiler**: Profiles test performance and identifies bottlenecks...
- **TestCacheManager**: Manages test caching and memoization for expensive operations...
- **PerformanceRegressionDetector**: Detects performance regressions in test execution...
- **TestOptimizationRecommendationEngine**: Generates optimization recommendations based on performance analysis...
- **TestPerformanceOptimizer**: Main system that orchestrates all performance optimization components...

### tools.test-quality.test_quality_cli

Test Quality Improvement CLI

Unified command-line interface for all test quality improvement tools:
- Coverage analysis and threshold enforcement
- Performance optimization and regression detection
- Flaky test detection and management


### tools.test-runner.coverage_analyzer

#### Classes

- **FileCoverage**: Coverage information for a single file...
- **ModuleCoverage**: Coverage information for a module (directory)...
- **CoverageReport**: Complete coverage report...
- **CoverageTrend**: Coverage trend analysis over time...
- **CoverageHistory**: Historical coverage data...
- **CoverageThresholdValidator**: Validates coverage against thresholds and policies...
- **CoverageAnalyzer**: Main coverage analyzer with measurement, reporting, and trend analysis...

### tools.test-runner.example_usage

Example usage of the complete Test Suite Infrastructure and Orchestration system


### tools.test-runner.orchestrator

#### Classes

- **TestCategory**: Test categories for organization and execution...
- **TestStatus**: Test execution status...
- **TestDetail**: Individual test result details...
- **CategoryResults**: Results for a specific test category...
- **TestSummary**: Overall test suite summary...
- **TestResults**: Complete test execution results...
- **TestConfig**: Test configuration loaded from YAML...
- **ResourceManager**: Manages system resources during test execution...
- **TestSuiteOrchestrator**: Main orchestrator for test suite execution with category management and parallel execution...

### tools.test-runner.runner_engine

Test Runner Engine - Core test execution with timeout handling and discovery

#### Classes

- **TestDiscoveryMethod**: Methods for discovering tests...
- **TestExecutionContext**: Context for test execution...
- **ExecutionProgress**: Progress tracking for test execution...
- **ProgressMonitor**: Monitors and reports test execution progress...
- **TestDiscovery**: Discovers and categorizes tests based on patterns and file structure...
- **TimeoutManager**: Manages test execution timeouts with graceful handling...
- **TestRunnerEngine**: Core test execution engine with timeout handling and progress monitoring...

### tools.test-runner.run_test_audit

Test Audit Runner - Script to audit and fix existing tests


### tools.training-system.models

Training System Models

Data models for the training and documentation system.

#### Classes

- **DifficultyLevel**: Training difficulty levels....
- **ModuleType**: Training module types....
- **QuestionType**: Assessment question types....
- **TrainingModule**: Training module definition....
- **LearningPath**: Personalized learning path....
- **ExerciseStep**: Step in a practice exercise....
- **PracticeExercise**: Hands-on practice exercise....
- **AssessmentQuestion**: Assessment question....
- **AssessmentResult**: Assessment result....
- **Assessment**: Knowledge assessment....
- **TroubleshootingStep**: Step in troubleshooting wizard....
- **TroubleshootingWizard**: Interactive troubleshooting wizard....
- **TrainingResource**: Training resource (video, document, etc.)....
- **Achievement**: Training achievement/badge....
- **ModuleProgress**: Progress for a specific module....
- **UserProgress**: User's overall training progress....
- **Certificate**: Training completion certificate....
- **FeedbackItem**: User feedback item....
- **TrainingMetrics**: Training system metrics....

### tools.unified-cli.cli

#### Classes

- **WorkflowContext**: Development workflow contexts that determine which tools to run...
- **ToolResult**: Result from running a tool...
- **WorkflowConfig**: Configuration for workflow automation...
- **TeamCollaborationConfig**: Configuration for team collaboration features...
- **MockTool**: Mock tool for tools that aren't implemented yet...
- **UnifiedCLI**: Unified CLI for all project cleanup and quality tools...

### tools.unified-cli.example_usage

Example usage of the Unified CLI Tool

This script demonstrates various ways to use the unified CLI tool
for project cleanup and quality improvements.


### tools.unified-cli.team_collaboration


### tools.unified-cli.test_integration

Integration tests for the Unified CLI Tool

Tests all major functionality including tool integration,
workflow automation, team collaboration, and IDE integration.

#### Classes

- **TestUnifiedCLI**: Test the main UnifiedCLI functionality...
- **TestWorkflowAutomation**: Test workflow automation functionality...
- **TestIDEIntegration**: Test IDE integration functionality...
- **TestIntegrationWorkflows**: Test end-to-end integration workflows...

### tools.unified-cli.__init__

Tool: unified-cli


### scripts.comprehensive_model_fix

Comprehensive Model Fix
Fixes all model-related issues including file structure, integrity checks, and validation


### scripts.debug_vram


### scripts.demo_health_check_integration

Demo script to test the health check integration
Shows the backend health endpoint working with port detection


### scripts.deploy_phase1_mvp

Phase 1 MVP Deployment Script
Validates and deploys the WAN models MVP with T2V, I2V, and TI2V support

#### Classes

- **Phase1Validator**: Validates Phase 1 MVP requirements...

### scripts.deploy_production_health

Production Health Monitoring Deployment Script

This script handles the deployment of health monitoring to production environments,
including configuration validation, service setup, and initial health checks.

#### Classes

- **ProductionDeploymentManager**: Manages deployment of health monitoring to production...

### scripts.deploy_wan_models

WAN Model Deployment Script

Command-line interface for deploying WAN models from placeholder to production
with comprehensive validation, rollback, and monitoring capabilities.


### scripts.download_models

WAN Model Download Script
Downloads and sets up WAN model files for the video generation system.

#### Classes

- **ModelDownloadManager**: Manages downloading and setup of WAN models...

### scripts.fix_health_checker_imports

Fix relative imports in health-checker module


### scripts.fix_tool_imports

Tool Import Fixer
Fixes import issues in the tools directory by adding missing __init__.py files
and updating import statements.


### scripts.install_cli

Installation script for WAN CLI
Makes the CLI available globally and sets up IDE integration


### scripts.manage_alerts

Alert Management CLI

Command-line interface for managing health monitoring alerts.


### scripts.migrate_configs

Configuration Migration Script
Consolidates scattered configuration files into the unified config system.

#### Classes

- **ConfigMigrator**: ...

### scripts.model_validation_recovery

WAN Model Validation and Recovery Script
Validates model integrity and provides recovery mechanisms for corrupted models.

#### Classes

- **ModelValidationRecovery**: Handles model validation and recovery operations...

### scripts.monitor_wan_models

WAN Model Monitoring Script

Command-line interface for monitoring deployed WAN models with real-time
health checking, alerting, and performance metrics.

#### Classes

- **MonitoringCLI**: CLI wrapper for monitoring service...

### scripts.optimize_model_loading_rtx4080

RTX 4080 Model Loading Optimization
Optimize model loading speed for RTX 4080 with 16GB VRAM


### scripts.performance-validation

#### Classes

- **PerformanceValidator**: Main performance validation orchestrator...

### scripts.quick_test_fix

Quick test of the pipeline loader fix


### scripts.run-comprehensive-tests

#### Classes

- **ComprehensiveTestRunner**: ...

### scripts.setup_alerting

#### Classes

- **AlertingSetupManager**: Manages setup and configuration of the alerting system...

### scripts.simple_enforcement_test


### scripts.startup_manager

WAN22 Server Startup Manager

Main entry point for the intelligent server startup system.
Provides CLI interface and orchestrates all startup components.

#### Classes

- **StartupManager**: Main startup manager orchestrator...

### scripts.validate-task-14-completion

Task 14 Completion Validator
Validates that all requirements for Task 14: Advanced testing and monitoring are met

#### Classes

- **Task14Validator**: ...

### scripts.validate_organization

#### Classes

- **OrganizationValidator**: ...

### scripts.validate_wan_deployment

WAN Model Deployment Validation Script

Standalone validation utility for WAN model deployments with comprehensive
pre and post-deployment checks.


### scripts.verify_code_review_implementation

Verify that the code review implementation is complete and functional


### scripts.__init__


### scripts.ci-cd.validate_deployment_gates

Validate Deployment Gates
Quick validation script to ensure all deployment gate components are working


### scripts.setup.configure_branch_protection

Configure branch protection rules with health monitoring integration.

This script sets up branch protection rules that require health checks
to pass before allowing merges to protected branches.

#### Classes

- **BranchProtectionManager**: Manages GitHub branch protection rules with health monitoring integration....

### scripts.setup.install_pre_commit_hooks

Setup script for installing pre-commit hooks.
This script installs and configures pre-commit hooks for the project.


### scripts.startup_manager.analytics

Usage analytics and optimization system for startup manager.

This module provides anonymous usage analytics to identify common failure patterns,
optimization suggestions based on system configuration and usage patterns,
and performance benchmarking against baseline startup times.

#### Classes

- **OptimizationCategory**: Categories of optimization suggestions....
- **OptimizationPriority**: Priority levels for optimization suggestions....
- **SystemProfile**: Anonymous system profile for analytics....
- **FailurePattern**: Identified failure pattern from analytics....
- **OptimizationSuggestion**: Performance optimization suggestion....
- **BenchmarkResult**: Performance benchmark result....
- **UsageAnalytics**: Anonymous usage analytics data....
- **AnalyticsEngine**: Usage analytics and optimization engine.

Features:
- Anonymous usage analytics collection
- Failure...

### scripts.startup_manager.cli

Interactive CLI interface for the WAN22 Server Startup Manager.

This module provides a rich, user-friendly command-line interface with:
- Progress bars and spinners
- Interactive prompts with clear options
- Verbose/quiet modes
- Colored output and structured display

#### Classes

- **VerbosityLevel**: Verbosity levels for CLI output...
- **CLIOptions**: CLI configuration options...
- **InteractiveCLI**: Rich-based interactive CLI for startup management...
- **_DummyProgress**: Dummy progress context for quiet mode...

### scripts.startup_manager.config

Configuration models for the startup manager using Pydantic.
Handles loading and validation of startup_config.json with environment variable overrides.

#### Classes

- **ServerConfig**: Base configuration for server instances....
- **BackendConfig**: Configuration for FastAPI backend server....
- **FrontendConfig**: Configuration for React frontend server....
- **LoggingConfig**: Configuration for logging system....
- **RecoveryConfig**: Configuration for recovery and error handling....
- **EnvironmentConfig**: Configuration for environment validation....
- **SecurityConfig**: Configuration for security settings....
- **StartupConfig**: Main startup configuration containing all server and system settings....
- **ConfigLoader**: Handles loading and validation of startup configuration with environment overrides....

### scripts.startup_manager.diagnostics

System diagnostics and troubleshooting features for the startup manager.

This module provides comprehensive system information collection, diagnostic mode,
and log analysis tools to help identify and resolve common startup issues.

#### Classes

- **SystemInfo**: System information data structure....
- **DiagnosticResult**: Diagnostic check result....
- **LogAnalysisResult**: Log analysis result....
- **SystemDiagnostics**: System diagnostics and information collection.

Provides comprehensive system information gathering ...
- **LogAnalyzer**: Log analysis tools for identifying common issues and patterns.

Analyzes startup logs to identify co...
- **DiagnosticMode**: Diagnostic mode that captures detailed startup process information.

Provides comprehensive diagnost...

### scripts.startup_manager.environment_validator

Environment Validator for WAN22 Startup Manager.
Validates system requirements, dependencies, and configurations before server startup.

#### Classes

- **ValidationStatus**: Status of validation checks....
- **ValidationIssue**: Represents a validation issue found during environment checking....
- **ValidationResult**: Result of environment validation....
- **DependencyValidator**: Validates Python and Node.js dependencies and environments....
- **ConfigurationValidator**: Validates and repairs configuration files....
- **EnvironmentValidator**: Main environment validator that orchestrates all validation checks....

### scripts.startup_manager.error_handler

Error handling and user guidance system for the startup manager.

This module provides:
- Structured error display with clear messages
- Interactive error resolution
- Context-sensitive help and troubleshooting
- Error classification and recovery suggestions

#### Classes

- **ErrorSeverity**: Error severity levels...
- **ErrorCategory**: Categories of startup errors...
- **RecoveryAction**: Represents a recovery action that can be taken...
- **StartupError**: Structured representation of a startup error...
- **ErrorClassifier**: Classifies errors and creates structured error objects...
- **ErrorDisplayManager**: Manages the display of errors and user interaction...
- **HelpSystem**: Provides context-sensitive help and troubleshooting guidance...

### scripts.startup_manager.logger

Comprehensive logging system for the startup manager.

This module provides structured logging with multiple outputs, rotation policies,
and different verbosity levels for debugging and troubleshooting.

#### Classes

- **LogLevel**: Log levels with color mapping....
- **LogEntry**: Structured log entry for JSON logging....
- **ColoredFormatter**: Custom formatter that adds colors to console output....
- **JSONFormatter**: Custom formatter for JSON output....
- **StartupLogger**: Comprehensive logging system for startup manager.

Features:
- Multiple output formats (console, fil...

### scripts.startup_manager.performance_monitor

Performance monitoring system for startup manager.

This module provides comprehensive timing metrics collection, success/failure rate tracking,
resource usage monitoring, and trend analysis for the startup process.

#### Classes

- **MetricType**: Types of metrics collected....
- **StartupPhase**: Startup phases for timing measurement....
- **TimingMetric**: Individual timing measurement....
- **ResourceSnapshot**: System resource usage snapshot....
- **StartupSession**: Complete startup session data....
- **PerformanceStats**: Aggregated performance statistics....
- **PerformanceMonitor**: Comprehensive performance monitoring system.

Features:
- Timing metrics collection for each startup...
- **TimingContext**: Context manager for timing operations....

### scripts.startup_manager.port_manager

Port Manager Component for WAN22 Server Startup Management System

This module handles port availability checking, conflict detection, and resolution
with Windows-specific handling for firewall and permission issues.

#### Classes

- **PortStatus**: Port availability status...
- **ConflictResolution**: Port conflict resolution strategies...
- **ProcessInfo**: Information about a process using a port...
- **PortConflict**: Information about a port conflict...
- **PortAllocation**: Result of port allocation...
- **PortCheckResult**: Result of port availability check...
- **PortManager**: Manages port allocation, conflict detection, and resolution for server startup.

Handles Windows-spe...

### scripts.startup_manager.preferences

User preference management system for the startup manager.
Handles persistent user preferences, configuration migration, and backup/restore.

#### Classes

- **UserPreferences**: User preferences for startup manager behavior....
- **ConfigurationVersion**: Configuration version information for migration tracking....
- **PreferenceManager**: Manages user preferences, configuration migration, and backup/restore....

### scripts.startup_manager.preference_cli

Command-line interface for managing user preferences.


### scripts.startup_manager.process_manager

Process Manager for WAN22 server startup management.
Handles server process lifecycle, health monitoring, and cleanup.

#### Classes

- **ProcessStatus**: Process status enumeration....
- **ProcessInfo**: Information about a managed process....
- **ProcessResult**: Result of a process operation....
- **HealthMonitor**: Health monitoring for server processes....
- **ProcessManager**: Manages server process lifecycle and health monitoring....

### scripts.startup_manager.recovery_engine

Recovery Engine for Server Startup Management System

This module provides error classification, recovery strategies, and intelligent
failure handling for the WAN22 server startup process.

#### Classes

- **ErrorType**: Classification of different error types that can occur during startup...
- **RecoveryAction**: Represents a specific recovery action that can be taken...
- **StartupError**: Represents an error that occurred during startup...
- **RecoveryResult**: Result of a recovery attempt...
- **ErrorPatternMatcher**: Matches error messages to specific error types using patterns...
- **RetryStrategy**: Implements exponential backoff retry logic...
- **RecoveryEngine**: Main recovery engine that handles error classification and recovery strategies...
- **FailurePattern**: Represents a detected failure pattern...
- **FallbackConfiguration**: Manages fallback configurations when primary recovery methods fail...
- **IntelligentFailureHandler**: Handles intelligent failure detection and learning...

### scripts.startup_manager.service_wrapper

Windows Service wrapper for WAN22 Server Manager

#### Classes

- **WAN22Service**: ...

### scripts.startup_manager.utils

Core utility functions for system detection and path management.

#### Classes

- **SystemDetector**: Utility class for detecting system information and capabilities....
- **PathManager**: Utility class for managing project paths and file operations....

### scripts.startup_manager.windows_utils

Windows-specific utilities for the startup manager.
Handles UAC, Windows services, firewall exceptions, and other Windows-specific features.

#### Classes

- **WindowsPermissionManager**: Manages Windows permissions and UAC elevation...
- **WindowsFirewallManager**: Manages Windows Firewall exceptions...
- **WindowsServiceManager**: Manages Windows services for background server management...
- **WindowsRegistryManager**: Manages Windows Registry operations for startup configuration...
- **WindowsSystemInfo**: Provides Windows system information...
- **WindowsOptimizer**: Windows-specific optimizations for the startup manager...

### scripts.startup_manager.__init__


### scripts.utils.comprehensive_syntax_fixer

Comprehensive syntax error detection and fixing script.
Scans all Python files and fixes common syntax issues.


### scripts.utils.emergency_fix_health_gate

Emergency Fix for Health Gate
This script creates a simple, bulletproof health check that always passes
when the basic health conditions are met.


### scripts.utils.fix_critical_test_issues

Fix critical test issues found by the audit.
Focus on the most common and safe fixes.


### scripts.utils.fix_deployment_gates

Fix common deployment gates workflow issues
This script addresses typical problems that cause workflow failures


### scripts.utils.fix_deployment_gates_ci

Fix Deployment Gates CI Issues
This script ensures deployment gates work correctly in CI environments


### scripts.utils.fix_indentation

Fix indentation issues introduced by the import fixer.


### scripts.utils.fix_project_syntax_errors

Fix syntax errors specifically in project files (not venv/site-packages).


### scripts.utils.fix_remaining_syntax_errors

Fix remaining syntax errors by targeting files with known issues.


### scripts.utils.fix_syntax_errors

Fix syntax errors introduced by our import fixer.
Focus on indentation issues with print statements.


### scripts.utils.fix_test_assertions

Comprehensive test assertion fixer.
Finds test functions lacking assertions and adds appropriate ones.


### scripts.utils.fix_test_imports

Quick fix for common test import issues found by the audit.


### scripts.utils.simple_assertion_fixer

Simple test assertion fixer.


### scripts.utils.simple_health_check

Simple health check script for CI/CD workflows

