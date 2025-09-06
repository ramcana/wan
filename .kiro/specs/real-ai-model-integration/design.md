# Design Document

## Overview

This design integrates the real AI models from the existing Wan2.2 system into the FastAPI backend by leveraging the substantial existing infrastructure found in the local installation system. The integration will bridge the React frontend with the proven model management, download, and optimization systems while maintaining API compatibility.

## Architecture

### High-Level Architecture

```
React Frontend
    ↓ (HTTP/WebSocket)
FastAPI Backend (backend/app.py)
    ↓ (Integration Layer)
System Integration (backend/core/system_integration.py)
    ↓ (Existing Infrastructure)
┌─────────────────────────────────────────────────────────┐
│ Existing Wan2.2 Infrastructure                         │
├─────────────────────────────────────────────────────────┤
│ • ModelManager (core/services/model_manager.py)        │
│ • ModelDownloader (local_installation/scripts/)        │
│ • ModelConfigurationManager                            │
│ • WAN22SystemOptimizer (backend/main.py)               │
│ • WanPipelineLoader (core/services/)                   │
│ • OptimizationManager                                  │
└─────────────────────────────────────────────────────────┘
    ↓ (Model Loading & Generation)
Real AI Models (T2V-A14B, I2V-A14B, TI2V-5B)
```

### Integration Strategy

The design follows a **bridge pattern** where the FastAPI backend acts as a bridge between the React frontend and the existing Wan2.2 infrastructure, minimizing changes to both systems while enabling real AI model integration.

## Components and Interfaces

### 1. Enhanced Generation Service

**Location**: `backend/services/generation_service.py`

**Responsibilities**:

- Bridge between FastAPI and existing model infrastructure
- Manage generation queue with real AI processing
- Coordinate with existing ModelManager and optimization systems
- Provide progress updates via WebSocket

**Key Methods**:

```python
class GenerationService:
    async def initialize_with_existing_infrastructure(self)
    async def load_model_via_model_manager(self, model_type: str)
    async def generate_video_with_real_pipeline(self, task: GenerationTaskDB)
    async def handle_model_download_if_missing(self, model_type: str)
    def integrate_with_wan22_optimizer(self, optimizer: WAN22SystemOptimizer)
```

### 2. Model Integration Bridge

**Location**: `backend/core/model_integration_bridge.py` (new)

**Responsibilities**:

- Adapt existing ModelManager interface for FastAPI use
- Handle model loading with existing optimization
- Coordinate with ModelDownloader for missing models
- Manage model lifecycle and memory

**Key Methods**:

```python
class ModelIntegrationBridge:
    def __init__(self, model_manager: ModelManager, downloader: ModelDownloader)
    async def ensure_model_available(self, model_type: str) -> bool
    async def load_model_with_optimization(self, model_type: str, hardware_profile: HardwareProfile)
    async def generate_with_existing_pipeline(self, params: GenerationParams) -> GenerationResult
    def get_model_status_from_existing_system(self) -> Dict[str, ModelStatus]
```

### 3. Enhanced System Integration

**Location**: `backend/core/system_integration.py` (enhanced)

**Responsibilities**:

- Initialize existing Wan2.2 components for FastAPI use
- Coordinate between different existing systems
- Provide unified interface for FastAPI backend
- Handle system optimization and hardware detection

**Enhanced Methods**:

```python
class SystemIntegration:
    async def initialize_wan22_infrastructure(self) -> bool
    async def setup_model_management_bridge(self) -> ModelIntegrationBridge
    async def initialize_system_optimizer(self) -> WAN22SystemOptimizer
    async def setup_model_downloader(self) -> ModelDownloader
    def get_integrated_system_status(self) -> SystemStatus
```

### 4. Real Generation Pipeline

**Location**: `backend/services/real_generation_pipeline.py` (new)

**Responsibilities**:

- Execute actual video generation using existing WAN pipeline
- Handle different model types (T2V, I2V, TI2V)
- Manage generation parameters and optimization
- Provide detailed progress tracking

**Key Methods**:

```python
class RealGenerationPipeline:
    def __init__(self, wan_pipeline_loader: WanPipelineLoader)
    async def generate_t2v(self, prompt: str, params: T2VParams) -> GenerationResult
    async def generate_i2v(self, image_path: str, prompt: str, params: I2VParams) -> GenerationResult
    async def generate_ti2v(self, image_path: str, prompt: str, params: TI2VParams) -> GenerationResult
    def setup_progress_callbacks(self, websocket_manager: ConnectionManager)
```

## Data Models

### Enhanced Generation Parameters

```python
@dataclass
class EnhancedGenerationParams:
    # Basic parameters (existing)
    prompt: str
    model_type: str
    resolution: str
    steps: int

    # Enhanced parameters for real generation
    image_path: Optional[str] = None
    end_image_path: Optional[str] = None
    lora_path: Optional[str] = None
    lora_strength: float = 1.0

    # Optimization parameters (from existing system)
    quantization_level: Optional[str] = None
    enable_offload: bool = True
    vae_tile_size: int = 256
    max_vram_usage_gb: Optional[float] = None

    # Generation-specific parameters
    guidance_scale: float = 7.5
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    fps: float = 8.0
    num_frames: int = 16
```

### Model Status Integration

```python
@dataclass
class IntegratedModelStatus:
    # From existing ModelManager
    model_id: str
    is_cached: bool
    is_loaded: bool
    is_valid: bool
    size_mb: float

    # From existing ModelConfigurationManager
    status: ModelStatus  # MISSING, AVAILABLE, CORRUPTED, etc.
    version: str
    model_type: ModelType

    # Integration-specific
    download_progress: Optional[float] = None
    optimization_applied: bool = False
    hardware_compatible: bool = True
    estimated_vram_usage_mb: float = 0.0
```

### Generation Result Enhancement

```python
@dataclass
class EnhancedGenerationResult:
    # Basic result info
    success: bool
    task_id: str
    output_path: Optional[str] = None

    # Generation metadata
    generation_time_seconds: float = 0.0
    model_used: str = ""
    parameters_used: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics (from existing system)
    peak_vram_usage_mb: float = 0.0
    average_vram_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    # Optimization info
    optimizations_applied: List[str] = field(default_factory=list)
    quantization_used: Optional[str] = None
    offload_used: bool = False

    # Error handling
    error_message: Optional[str] = None
    error_category: Optional[str] = None
    recovery_suggestions: List[str] = field(default_factory=list)
```

## Error Handling

### Integration with Existing Error System

The design leverages the existing error handling infrastructure from the local installation system:

```python
class IntegratedErrorHandler:
    def __init__(self, existing_error_handler: GenerationErrorHandler):
        self.existing_handler = existing_error_handler

    async def handle_model_loading_error(self, error: Exception, model_type: str) -> ErrorResponse:
        # Use existing error categorization and recovery suggestions
        recovery_info = self.existing_handler.handle_error(error, {"model_type": model_type})

        # Adapt for FastAPI response
        return ErrorResponse(
            error_type="model_loading_error",
            message=recovery_info.message,
            suggestions=recovery_info.recovery_suggestions,
            can_retry=True
        )

    async def handle_generation_error(self, error: Exception, task_id: str) -> ErrorResponse:
        # Categorize error using existing system
        if "CUDA out of memory" in str(error):
            return self._handle_vram_error(error, task_id)
        elif "Model not found" in str(error):
            return self._handle_missing_model_error(error, task_id)
        else:
            return self._handle_generic_error(error, task_id)
```

### Error Categories and Recovery

1. **Model Loading Errors**

   - Missing model → Trigger automatic download
   - Corrupted model → Re-download and validate
   - VRAM insufficient → Apply optimization settings

2. **Generation Errors**

   - CUDA OOM → Reduce batch size, enable offloading
   - Invalid parameters → Validate and suggest corrections
   - Pipeline errors → Fallback to alternative settings

3. **System Errors**
   - Hardware issues → Use existing hardware detection
   - Configuration errors → Apply default configurations
   - Network errors → Retry with exponential backoff

## Testing Strategy

### Integration Testing Approach

1. **Model Manager Integration Tests**

   ```python
   async def test_model_manager_integration():
       # Test that FastAPI can use existing ModelManager
       bridge = ModelIntegrationBridge(model_manager, downloader)
       assert await bridge.ensure_model_available("t2v-A14B")
       assert bridge.get_model_status("t2v-A14B").is_loaded
   ```

2. **Generation Pipeline Tests**

   ```python
   async def test_real_generation_pipeline():
       # Test actual generation with existing infrastructure
       pipeline = RealGenerationPipeline(wan_pipeline_loader)
       result = await pipeline.generate_t2v("test prompt", test_params)
       assert result.success
       assert result.output_path.exists()
   ```

3. **System Integration Tests**
   ```python
   async def test_full_system_integration():
       # Test end-to-end integration
       response = await client.post("/api/v1/generation/submit", data=test_request)
       assert response.status_code == 200
       # Verify real model was used, not mock
       assert "mock" not in response.json()["message"]
   ```

### Testing Phases

1. **Phase 1: Component Integration**

   - Test ModelManager bridge functionality
   - Verify ModelDownloader integration
   - Validate system optimization integration

2. **Phase 2: Generation Pipeline**

   - Test real model loading
   - Verify generation with each model type
   - Validate progress tracking and WebSocket updates

3. **Phase 3: End-to-End Integration**

   - Test complete workflow from React frontend
   - Verify API compatibility maintained
   - Test error handling and recovery

4. **Phase 4: Performance and Optimization**
   - Benchmark generation performance
   - Test hardware optimization integration
   - Validate memory management

## Implementation Phases

### Phase 1: Infrastructure Bridge (Week 1)

- Create ModelIntegrationBridge
- Enhance SystemIntegration class
- Set up basic model loading with existing ModelManager
- Test model availability checking

### Phase 2: Generation Pipeline (Week 2)

- Implement RealGenerationPipeline
- Integrate with existing WanPipelineLoader
- Add progress tracking and WebSocket updates
- Test basic generation functionality

### Phase 3: Advanced Features (Week 3)

- Integrate ModelDownloader for missing models
- Add hardware optimization integration
- Implement comprehensive error handling
- Add LoRA support integration

### Phase 4: Testing and Optimization (Week 4)

- Comprehensive integration testing
- Performance optimization and benchmarking
- Documentation and deployment preparation
- Final validation and bug fixes

## Security Considerations

1. **Model File Security**

   - Use existing integrity verification from ModelDownloader
   - Validate model files before loading
   - Secure model storage with existing directory structure

2. **API Security**

   - Maintain existing FastAPI security measures
   - Validate generation parameters
   - Rate limiting for generation requests

3. **Resource Security**
   - Use existing VRAM management and limits
   - Prevent resource exhaustion attacks
   - Monitor system resource usage

## Performance Considerations

1. **Model Loading Optimization**

   - Leverage existing model caching from ModelManager
   - Use existing quantization and offloading strategies
   - Implement model preloading for frequently used models

2. **Generation Performance**

   - Use existing hardware optimization from WAN22SystemOptimizer
   - Apply existing chunked processing for large generations
   - Implement generation queue management

3. **Memory Management**
   - Use existing VRAM monitoring and management
   - Implement model unloading when not in use
   - Apply existing memory optimization strategies

## Deployment Strategy

1. **Gradual Rollout**

   - Deploy with feature flag to enable/disable real generation
   - Maintain mock generation as fallback
   - Monitor system performance and stability

2. **Configuration Management**

   - Use existing configuration system from local installation
   - Allow runtime switching between mock and real generation
   - Provide configuration validation and migration

3. **Monitoring and Logging**
   - Integrate with existing logging system
   - Monitor generation performance and success rates
   - Track model usage and optimization effectiveness
