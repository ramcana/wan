# Implementation Plan

## Sprint 0: Project Wiring (½ day)

- [ ] 0.1 Set up package structure and development environment
  - Create backend/core/model_orchestrator/ package with **init**.py
  - Add empty class stubs for ModelRegistry, ModelResolver, LockManager, ModelEnsurer
  - Set up CI configuration (ruff + mypy + pytest) and pre-commit hooks
  - Configure basic logging with structured output and correlation IDs
  - Add environment variable handling for MODELS_ROOT, WAN_MODELS_MANIFEST
  - Write basic smoke test to validate package imports and environment setup

## MVP Sprint 1: Core Orchestrator Foundation

- [x] 1. Implement manifest system with validation

  - Implement TOML manifest parsing with schema_version validation
  - Define core data models (ModelSpec, FileSpec) with WAN2.2 model support
  - Add model ID normalization function returning canonical {model_id, version, variant}
  - Create manifest validator with path safety checks (no .., Windows case collisions)
  - Add schema version compatibility checking with migration guidance
  - Write unit tests for manifest parsing, validation, and ID normalization
  - _Requirements: 1.1, 1.2, 1.3, 1.6_

- [x] 2. Implement deterministic path resolver with atomic operations

  - Create ModelResolver class with cross-platform path handling
  - Implement deterministic path generation from MODELS_ROOT
  - Add support for model variants in path resolution
  - Create temp directory strategy: {MODELS_ROOT}/.tmp/{model}@{variant}.{uuid}.partial
  - Ensure temp and final paths are on same filesystem for atomic rename
  - Handle Windows long path scenarios with appropriate error messages
  - Write unit tests for path resolution and atomic operations across platforms
  - _Requirements: 3.1, 3.2, 3.3, 4.2_

- [x] 3. Build cross-process locking system

  - Implement LockManager with OS-appropriate file locking (fcntl/msvcrt)
  - Create context manager for safe lock acquisition and release
  - Add timeout-based lock acquisition with clear error handling
  - Implement stale lock cleanup for crashed processes
  - Write unit tests for concurrent lock scenarios
  - _Requirements: 4.3, 4.6_

- [x] 4. Create HuggingFace storage backend

  - Implement HFStore class using huggingface_hub snapshot_download
  - Add support for allow_patterns to download only required file types
  - Integrate with shared HF cache (HF_HOME, HF_HUB_CACHE)
  - Enable hf_transfer for multi-connection downloads when available
  - Write unit tests with mocked HuggingFace Hub responses
  - _Requirements: 2.5, 4.1, 4.4, 9.1, 9.4_

- [x] 5. Implement atomic download orchestration with preflight checks

  - Create ModelEnsurer class with atomic download workflow
  - Add preflight disk space check: bytes_needed ≤ free_space - safety_margin
  - Implement temporary directory downloads with fsync() before atomic rename
  - Add basic file completeness checking (existence and size)
  - Integrate cross-process locking with download operations
  - Add optional checksum verification when manifest provides hashes
  - Create .verified.json file after successful verification for fast status checks
  - Write integration tests for complete download workflows and disk space scenarios
  - _Requirements: 4.1, 4.2, 4.5, 5.1, 5.7, 10.1_

- [x] 6. Build core API surface with status states

  - Implement ensure(model_id, variant) -> Path method with structured error codes
  - Create status(model_id, variant) -> ModelStatus with states: NOT_PRESENT | PARTIAL | VERIFYING | COMPLETE | CORRUPT
  - Add verify_integrity(model_id) method with checksum validation
  - Implement HuggingFace fallback logic (log once if hf_transfer unavailable, continue without)
  - Add optional negative cache for recently failed sources (in-memory TTL)
  - Write unit tests for all API methods with various scenarios and error conditions
  - _Requirements: 6.1, 6.3, 6.4, 13.1, 13.2_

- [x] 7. Create CLI interface

  - Implement `wan models status` command with human-readable output
  - Add `wan models ensure --only {model_id}` for selective downloads
  - Support `--json` flag for machine-readable output
  - Implement proper exit codes for success/failure scenarios
  - Add progress indicators for download operations
  - Write CLI integration tests with various command combinations
  - _Requirements: 7.1, 7.2, 7.4, 13.3, 13.4_

- [x] 8. Add basic health monitoring endpoint (no GPU)

  - Create /health/models endpoint that reports orchestrator status only
  - Implement dry-run model status checking: present/partial/missing + bytes_needed
  - Return structured JSON showing missing components per model without downloads
  - Add ?dry_run=true parameter to prevent any side effects
  - Ensure response time <100ms for 3 models by using .verified.json cache
  - Add basic error handling and timeout protection for health checks
  - Write tests for health endpoint with various model states
  - _Requirements: 10.1, 10.2, 12.4_

- [x] 9. Integrate with WAN pipeline loader with model-specific handling

  - Create pipeline class mapping (t2v-A14B → WanT2VPipeline, i2v-A14B → WanI2VPipeline, ti2v-5b → WanTI2VPipeline)
  - Modify wan_pipeline_loader.py to validate required components per model type
  - Implement component validation before GPU initialization (prevent loading image_encoder for t2v)
  - Add model-specific VRAM estimation using manifest parameters
  - Replace hardcoded paths with get_wan_paths(model_id) and validate component completeness
  - Write integration tests for each model type with correct component loading
  - _Requirements: 6.1, 6.2, 6.3, 14.1, 14.4_

- [x] 10. Create WAN2.2-specific models.toml manifest

  - Define manifest schema with model-specific required components (text_encoder, image_encoder, unet, vae)
  - Add t2v-A14B manifest with text_encoder + unet + vae (no image_encoder)
  - Add i2v-A14B manifest with image_encoder + unet + vae (optional text_encoder)
  - Add ti2v-5b manifest with text_encoder + image_encoder + unet + vae (dual conditioning)
  - Include per-model defaults (fps, frames, scheduler, precision, guidance_scale)
  - Add VRAM estimation parameters (params_billion, family_size) for each model
  - Support shard-aware downloads using index.json files for large models
  - _Requirements: 1.1, 1.3, 1.7, 14.1, 14.2_

### MVP Definition of Done

- `ensure(model_id, variant)` returns ready absolute path or raises with structured error code
- Atomic download + cross-process lock + checksum (if present) + basic size checks
- CLI: `wan models status|ensure --only t2v-A14B --json` with proper exit codes
- `/health/models?dry_run=true` lists models and missing bytes without downloading
- Two concurrent `ensure()` calls: one downloads, one waits, both succeed
- Kill during download → next `ensure()` resumes and completes; no stray .partial files
- Disk nearly full: `ensure()` fails with NO_SPACE and suggests `wan models gc`

## Sprint 2: Production Hardening

- [x] 11. Add comprehensive error handling and recovery

  - Implement structured error codes (NO_SPACE, AUTH_FAIL, CHECKSUM_FAIL, etc.)
  - Add retry logic with exponential backoff for transient failures
  - Create error recovery strategies for common failure modes
  - Implement proper logging with correlation IDs and structured metadata
  - Write chaos tests for network failures and interruptions
  - _Requirements: 4.8, 5.3, 5.4, 12.1, 12.2_

- [x] 12. Implement disk space management

  - Add preflight disk space checks before downloads
  - Create basic garbage collection with LRU-based cleanup
  - Implement configurable disk quotas and size limits
  - Add model pinning to protect critical models from GC
  - Create `wan models gc` CLI command with dry-run support
  - _Requirements: 10.1, 10.3, 10.4, 10.5_

- [x] 13. Add S3/MinIO storage backend

  - Implement S3Store class with boto3 integration
  - Support custom endpoints for MinIO compatibility
  - Add parallel downloads with configurable concurrency
  - Implement resume capability using HTTP Range requests
  - Write tests with mocked S3 responses and error scenarios
  - _Requirements: 2.4, 8.4, 9.2_

- [x] 14. Enhance observability and monitoring

  - Add Prometheus metrics for downloads, errors, and storage usage (limit cardinality: model_id, variant, source only)
  - Implement structured logging with configurable levels and correlation IDs
  - Add GPU-based health checks: t2v/i2v/ti2v smoke tests with 1-2 denoise steps at 128x128
  - Create detailed health endpoints with per-model diagnostics and component validation
  - Add performance metrics and download duration tracking
  - Write monitoring integration tests
  - _Requirements: 12.1, 12.3, 12.5, 12.6, 14.6_

- [x] 15. Add comprehensive integrity verification

  - Implement per-file SHA256 checksum verification
  - Add fallback to HuggingFace ETag verification when checksums unavailable
  - Create manifest signature verification for enhanced security
  - Implement retry logic for checksum failures
  - Write tests for various integrity failure scenarios
  - _Requirements: 5.1, 5.2, 5.3, 5.5_

- [x] 16. Implement component deduplication system

  - Create shared component store at {MODELS_ROOT}/components/<name>@<version>
  - Implement content-addressed storage for shared files (tokenizers, common encoders)
  - Add hardlink/symlink/junction creation during atomic move step
  - Detect duplicate files across models and create links to save disk space
  - Add component reference tracking to prevent premature deletion
  - Write tests for component sharing across t2v/i2v/ti2v models
  - _Requirements: 10.6, 11.7_

## Sprint 3: Advanced Features

- [x] 17. Enhance WAN2.2-specific model handling

  - Add support for sharded models with index.json files (selective shard downloading)
  - Implement model-specific conditioning and preprocessing (image normalization for i2v/ti2v)
  - Add text embedding caching for multi-clip sequences in t2v/ti2v
  - Configure VAE tile size and decode strategy per model to prevent OOM
  - Create development/production variant switching with different precision defaults
  - Add model-specific input validation (prevent image input to t2v, require image for i2v/ti2v)
  - Write tests for WAN2.2 model structure variations and input validation
  - _Requirements: 14.1, 14.2, 14.3, 14.5, 14.6, 14.7_

- [x] 18. Add advanced concurrency and performance features

  - Implement parallel file downloads within single models
  - Add bandwidth limiting and connection pooling
  - Create download queue management for multiple concurrent models
  - Optimize memory usage for large model downloads
  - Write performance benchmarks and optimization tests
  - _Requirements: 9.2, 9.3_

- [x] 19. Enhance security and credential management

  - Implement secure credential storage using system keyring
  - Add support for presigned URLs and temporary access
  - Create credential masking in all log output
  - Add at-rest encryption support for sensitive models
  - Write security tests for credential handling
  - _Requirements: 8.3, 8.5, 8.6, 8.7_

- [x] 20. Create migration and compatibility tools

  - Build configuration migration tool from existing setups
  - Create backward compatibility adapters for legacy paths
  - Implement gradual rollout with feature flags
  - Add validation tools for manifest and configuration
  - Write migration tests and rollback procedures
  - _Requirements: 13.7_

- [x] 21. Add comprehensive testing and documentation

  - Create end-to-end integration tests for complete workflows
  - Add cross-platform compatibility tests (Windows, WSL, Unix)
  - Implement performance and load testing suites
  - Create user documentation and deployment guides
  - Add troubleshooting guides and operational runbooks
  - _Requirements: All requirements validation_

## Implementation Guidelines

### Code Quality Standards

- Keep temp/locks/state in `{MODELS_ROOT}/.tmp`, `/.locks`, `/.state`
- Use same-volume temp dirs to guarantee atomic rename
- Always `fsync()` before atomic rename on large files (especially Windows)
- Never log tokens or full presigned URLs; redact sensitive information
- On Windows, document enabling long paths and prefer junctions over symlinks

### Sanity Checklist (validate while building)

- [x] `models.toml` passes validator (schema_version, no dupes, no traversal, Windows case OK)

- [ ] `status()` returns PARTIAL with exact missing files and bytes_needed

- [x] Two concurrent `ensure()` calls for same model: one downloads, one waits, both succeed

- [ ] Kill during download → next `ensure()` resumes and completes; no stray .partial remains
- [ ] Disk nearly full: `ensure()` fails with NO_SPACE and suggests `wan models gc`
- [ ] Health endpoint returns read-only status in <100ms for 3 models
- [ ] CLI `--json` output is stable and parseable (add JSON schema test)
