# Requirements Document

## Introduction

The Model Orchestrator is a comprehensive model management system that eliminates the "where are the models / how do we download them" problem in WAN2.2. It provides a unified approach to model discovery, downloading, and path resolution through a manifest-driven architecture with multiple storage backends (local, S3/MinIO, HuggingFace Hub) and deterministic path policies. The system integrates directly into the WAN2.2 pipeline loading process and provides both programmatic APIs and CLI tools for model management.

## Requirements

### Requirement 1: Unified Model Manifest System with Versioning

**User Story:** As a developer, I want a single source of truth for all WAN2.2 model definitions with proper versioning and variant support, so that I can consistently reference specific model versions and precisions across environments.

#### Acceptance Criteria

1. WHEN the system initializes THEN it SHALL load model definitions from a single `models.toml` manifest file with schema versioning
2. WHEN a model is referenced by ID THEN the system SHALL resolve it using canonical model_id format (e.g., "t2v-A14B@2.2.0")
3. WHEN the manifest defines model variants (fp16/bf16/int8) THEN the system SHALL support variant-specific resolution
4. WHEN the manifest includes per-file metadata (size, sha256) THEN the system SHALL validate all required fields are present
5. WHEN multiple environments access the same manifest THEN they SHALL resolve identical model IDs to the same file structure
6. WHEN the manifest schema version is incompatible THEN the system SHALL fail gracefully with migration guidance
7. WHEN sharded models use index.json files THEN the system SHALL download only referenced shards
8. IF a model ID or variant is not found THEN the system SHALL raise a clear error with available options

### Requirement 2: Multi-Source Storage Backend System

**User Story:** As a system administrator, I want models to be automatically sourced from the fastest available location (local → S3 → HuggingFace), so that development is fast and production is reliable.

#### Acceptance Criteria

1. WHEN a model is requested THEN the system SHALL attempt to source it from storage backends in priority order
2. WHEN local storage contains all required files THEN the system SHALL use the local copy without network access
3. WHEN local storage is incomplete THEN the system SHALL attempt the next configured source in priority order
4. WHEN S3/MinIO backend is configured THEN the system SHALL support resumable downloads with size verification
5. WHEN HuggingFace Hub is used THEN the system SHALL leverage allow_patterns to download only required file types
6. WHEN all sources fail THEN the system SHALL raise a comprehensive error indicating which sources were attempted

### Requirement 3: Deterministic Path Resolution

**User Story:** As a developer, I want all model paths to be resolved deterministically from a single MODELS_ROOT, so that there's no path drift between different parts of the system.

#### Acceptance Criteria

1. WHEN the system resolves a model path THEN it SHALL use only the configured MODELS_ROOT and model ID
2. WHEN multiple services request the same model THEN they SHALL receive identical absolute paths
3. WHEN the MODELS_ROOT changes THEN all model paths SHALL resolve relative to the new root
4. WHEN a model directory is created THEN it SHALL follow the pattern `{MODELS_ROOT}/wan22/{model_id}`
5. IF MODELS_ROOT is not configured THEN the system SHALL raise a configuration error

### Requirement 4: Atomic Downloads with Concurrency Safety

**User Story:** As a user, I want models to be downloaded atomically and safely handle concurrent access, so that partial downloads never appear complete and multiple processes don't conflict.

#### Acceptance Criteria

1. WHEN a model is requested for the first time THEN the system SHALL download only the required files to a temporary directory
2. WHEN a download completes successfully THEN the system SHALL atomically rename from temporary to final location
3. WHEN multiple processes request the same model THEN only one SHALL download while others wait for completion
4. WHEN a download is interrupted THEN the system SHALL resume from the last completed chunk without corruption
5. WHEN partial files exist locally THEN the system SHALL verify integrity before considering them complete
6. WHEN cross-process file locks are required THEN the system SHALL use OS-appropriate locking mechanisms
7. WHEN downloads are concurrent THEN the system SHALL respect max_concurrent_downloads configuration
8. WHEN retry attempts are made THEN the system SHALL use exponential backoff with jitter per source

### Requirement 5: Comprehensive Integrity and Trust Chain

**User Story:** As a system administrator, I want robust integrity verification with fallback mechanisms, so that corrupted models are detected and handled appropriately.

#### Acceptance Criteria

1. WHEN per-file checksums (sha256) and sizes are provided in manifest THEN the system SHALL verify both after download
2. WHEN checksum verification fails THEN the system SHALL re-download the affected files up to retry limit
3. WHEN HuggingFace ETag/size data is available and checksums are missing THEN the system SHALL use HF metadata for verification
4. WHEN repeated checksum failures occur THEN the system SHALL fail with detailed error listing exact files and sources attempted
5. WHEN manifest signatures are present THEN the system SHALL verify manifest authenticity before processing
6. WHEN all required files are present and verified THEN the system SHALL mark the model as complete
7. IF verification data is completely missing THEN the system SHALL perform basic completeness checks using file existence and sizes

### Requirement 6: Pipeline Integration

**User Story:** As a developer, I want the Model Orchestrator to integrate seamlessly with existing WAN2.2 pipeline loading, so that I can replace hardcoded paths with orchestrated model resolution.

#### Acceptance Criteria

1. WHEN the pipeline loader requests a model THEN it SHALL receive an absolute path to the complete model directory
2. WHEN the model is not locally available THEN the system SHALL automatically ensure it's downloaded before returning the path
3. WHEN the pipeline loader calls `get_wan_paths(model_id)` THEN it SHALL return a ready-to-use model directory path
4. WHEN model loading fails THEN the system SHALL provide clear error messages indicating the failure reason
5. IF the model ID is invalid THEN the system SHALL raise an error before attempting any downloads

### Requirement 7: CLI Management Interface

**User Story:** As a system administrator, I want command-line tools to pre-download, verify, and manage models, so that I can prepare systems and troubleshoot model issues.

#### Acceptance Criteria

1. WHEN I run `wan models ensure --all` THEN the system SHALL download all models defined in the manifest
2. WHEN I run `wan models ensure --only {model_id}` THEN the system SHALL download only the specified model
3. WHEN I run `wan models check` THEN the system SHALL report the status of all configured models
4. WHEN downloads complete successfully THEN the CLI SHALL display the local path for each model
5. WHEN downloads fail THEN the CLI SHALL display clear error messages with troubleshooting guidance

### Requirement 8: Secure Configuration and Credential Management

**User Story:** As a developer, I want secure credential handling and comprehensive configuration options, so that I can deploy safely across different environments with proper secret management.

#### Acceptance Criteria

1. WHEN MODELS_ROOT is set THEN the system SHALL use it as the base directory for all model storage
2. WHEN WAN_MODELS_MANIFEST is set THEN the system SHALL load the manifest from the specified path
3. WHEN HF_TOKEN is configured THEN the system SHALL read it from keyring/secret store if available, environment otherwise
4. WHEN S3 credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL) are set THEN the system SHALL use them for S3/MinIO access
5. WHEN credentials or URLs are logged THEN the system SHALL mask sensitive values in all log output
6. WHEN presigned URLs are provided THEN the system SHALL support them for temporary access
7. WHEN at-rest encryption is configured THEN the system SHALL support encrypted model storage
8. IF required environment variables are missing THEN the system SHALL provide clear configuration guidance with examples

### Requirement 9: Performance Optimization

**User Story:** As a user, I want model downloads to be as fast as possible, so that I can start working with models quickly.

#### Acceptance Criteria

1. WHEN HF_HUB_ENABLE_HF_TRANSFER is enabled THEN the system SHALL use multi-connection downloads from HuggingFace
2. WHEN downloading from S3/MinIO THEN the system SHALL use parallel transfers for multiple files
3. WHEN allow_patterns are configured THEN the system SHALL skip unnecessary files to reduce download size
4. WHEN using shared HF cache THEN the system SHALL avoid duplicate downloads across different virtual environments
5. WHEN local or S3 sources are available THEN the system SHALL prefer them over HuggingFace for speed

### Requirement 10: Disk Space Management and Garbage Collection

**User Story:** As a system administrator, I want automatic disk space management with configurable quotas and garbage collection, so that the system doesn't consume unlimited storage.

#### Acceptance Criteria

1. WHEN a download is requested THEN the system SHALL perform preflight free-space checks before starting
2. WHEN disk quota limits are configured THEN the system SHALL enforce global or per-family storage limits
3. WHEN storage quota is exceeded THEN the system SHALL trigger LRU/TTL-based garbage collection
4. WHEN models are marked as "pinned" THEN garbage collection SHALL preserve them regardless of age
5. WHEN garbage collection runs THEN it SHALL report reclaimed bytes and preserved models
6. WHEN insufficient disk space exists THEN the system SHALL provide actionable error messages before starting downloads
7. IF disk space cannot be reclaimed THEN the system SHALL fail with clear guidance on manual cleanup options

### Requirement 11: Cross-Platform File System Compatibility

**User Story:** As a developer, I want the system to handle Windows, WSL, and Unix file system quirks correctly, so that it works reliably across all deployment platforms.

#### Acceptance Criteria

1. WHEN running on Windows THEN the system SHALL support long paths (>260 characters) with appropriate OS configuration guidance
2. WHEN path length limits are exceeded THEN the system SHALL provide clear error messages with workaround suggestions
3. WHEN running on WSL THEN path mapping between Windows and Linux SHALL be stable and reversible
4. WHEN case-insensitive file systems are used THEN the system SHALL handle filename collisions appropriately
5. WHEN Windows reserved names are encountered THEN the system SHALL handle them without corruption
6. WHEN UNC paths are used THEN the system SHALL support them for network storage scenarios
7. IF file system limitations prevent operation THEN the system SHALL provide platform-specific guidance

### Requirement 12: Comprehensive Observability and Error Classification

**User Story:** As a system administrator, I want detailed metrics, structured logging, and categorized errors, so that I can monitor system health and troubleshoot issues effectively.

#### Acceptance Criteria

1. WHEN download operations occur THEN the system SHALL emit metrics for started/completed/failed downloads with duration and bytes
2. WHEN errors occur THEN the system SHALL classify them with specific error codes (NO_SPACE, AUTH_FAIL, CHECKSUM_FAIL, LOCK_TIMEOUT, RATE_LIMIT)
3. WHEN rate limiting is encountered THEN the system SHALL implement exponential backoff and reflect retry attempts in logs
4. WHEN health endpoints are queried THEN they SHALL return per-model status without triggering side effects
5. WHEN metrics endpoints are available THEN they SHALL provide Prometheus-compatible format
6. WHEN structured logging is enabled THEN all log entries SHALL include correlation IDs and contextual metadata
7. IF monitoring systems are configured THEN the system SHALL integrate with standard observability tools

### Requirement 13: Production API Surface and CLI Tools

**User Story:** As a developer and administrator, I want comprehensive Python APIs and CLI tools with both human and machine-readable output, so that I can integrate and automate model management.

#### Acceptance Criteria

1. WHEN using Python API THEN `ensure(model_id, variant=None)` SHALL return Path to ready model directory
2. WHEN querying model status THEN `status(model_id)` SHALL return structured data with state, missing files, and bytes needed
3. WHEN using CLI tools THEN they SHALL support both human-readable and `--json` output for automation
4. WHEN CLI operations fail THEN they SHALL return non-zero exit codes with clear error messages
5. WHEN garbage collection is invoked THEN `gc()` API and `wan models gc` CLI SHALL support dry-run and size limits
6. WHEN JSON output is requested THEN the schema SHALL be stable and documented
7. IF backward compatibility is required THEN the system SHALL provide migration adapters for existing configurations

### Requirement 14: WAN2.2 Pipeline Integration Specifics

**User Story:** As a WAN2.2 developer, I want the orchestrator to handle WAN2.2-specific model requirements like sharded weights, optional components, and development variants, so that it integrates seamlessly with the existing pipeline.

#### Acceptance Criteria

1. WHEN sharded models use index.json files THEN the system SHALL download only referenced weight shards
2. WHEN optional components (text encoder, VAE) are configured THEN the system SHALL support selective downloading
3. WHEN LoRA overlays are specified THEN the system SHALL handle overlay model management
4. WHEN development mode is enabled THEN the system SHALL support lightweight model variants for faster iteration
5. WHEN precision downcasting is configured THEN the system SHALL support dev-mode precision variants
6. WHEN production deployment occurs THEN the system SHALL default to full-precision variants
7. IF model components are missing THEN the system SHALL provide clear guidance on which components are required for specific operations
