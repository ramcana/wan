# Design Document

## Overview

This design implements the actual WAN video generation models (T2V-A14B, I2V-A14B, TI2V-5B) to replace placeholder model references with functional implementations. The design leverages the existing comprehensive infrastructure including ModelIntegrationBridge, RealGenerationPipeline, hardware optimization systems, and WebSocket progress tracking. The implementation creates the WAN model architecture as the MVP foundation while maintaining compatibility with all existing APIs and frontend interfaces.

## Architecture

### High-Level Architecture

```
React Frontend (unchanged)
    ↓ (HTTP/WebSocket)
FastAPI Backend (existing endpoints)
    ↓ (existing integration)
Enhanced Generation Service (existing)
    ↓ (existing bridge)
Model Integration Bridge (existing)
    ↓ (NEW: Real WAN Models)
┌─────────────────────────────────────────────────────────┐
│ WAN Model Implementations (NEW)                         │
├─────────────────────────────────────────────────────────┤
│ • WAN T2V-A14B Pipeline                                 │
│ • WAN I2V-A14B Pipeline                                 │
│ • WAN TI2V-5B Pipeline                                  │
│ • WAN Model Architecture Components                     │
│ • WAN-specific Optimization Strategies                  │
└─────────────────────────────────────────────────────────┘
    ↓ (PyTorch/Diffusers)
Hardware (RTX 4080 + Threadripper PRO)
```

### Integration Strategy

The design follows a **model implementation pattern** where actual WAN model code replaces placeholder references while maintaining full compatibility with the existing infrastructure. All existing components (ModelIntegrationBridge, RealGenerationPipeline, hardware optimization) remain unchanged and work seamlessly with the new WAN implementations.

## Components and Interfaces

### 1. WAN Model Architecture Implementation

**Location**: `core/models/wan_models/` (new directory)

**Components**:

- `wan_t2v_a14b.py` - Text-to-Video A14B model implementation
- `wan_i2v_a14b.py` - Image-to-Video A14B model implementation
- `wan_ti2v_5b.py` - Text+Image-to-Video 5B model implementation
- `wan_base_model.py` - Shared WAN model architecture components
- `wan_pipeline_factory.py` - Factory for creating WAN pipeline instances

**Key Classes**:

```python
class WANBaseModel(torch.nn.Module):
    """Base class for all WAN model implementations"""
    def __init__(self, model_config: Dict[str, Any])
    def load_pretrained_weights(self, checkpoint_path: str)
    def optimize_for_hardware(self, hardware_profile: HardwareProfile)
    def estimate_vram_usage(self) -> float
    def get_model_info(self) -> Dict[str, Any]

class WANT2VA14B(WANBaseModel):
    """WAN Text-to-Video A14B model implementation"""
    def __init__(self, model_config: Dict[str, Any])
    def generate_video(self, prompt: str, **kwargs) -> torch.Tensor
    def setup_diffusion_pipeline(self)
    def apply_temporal_attention(self, features: torch.Tensor) -> torch.Tensor

class WANI2VA14B(WANBaseModel):
    """WAN Image-to-Video A14B model implementation"""
    def __init__(self, model_config: Dict[str, Any])
    def generate_video(self, image: torch.Tensor, prompt: str, **kwargs) -> torch.Tensor
    def encode_input_image(self, image: torch.Tensor) -> torch.Tensor
    def apply_image_conditioning(self, features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor

class WANTI2V5B(WANBaseModel):
    """WAN Text+Image-to-Video 5B model implementation"""
    def __init__(self, model_config: Dict[str, Any])
    def generate_video(self, image: torch.Tensor, prompt: str, end_image: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor
    def interpolate_between_images(self, start_image: torch.Tensor, end_image: torch.Tensor, num_frames: int) -> torch.Tensor
```

### 2. WAN Pipeline Integration

**Location**: `core/services/wan_pipeline_loader.py` (enhanced existing)

**Responsibilities**:

- Load actual WAN model implementations instead of placeholder references
- Integrate WAN models with existing pipeline infrastructure
- Handle WAN-specific optimization and configuration
- Provide unified interface for existing RealGenerationPipeline

**Enhanced Methods**:

```python
class WanPipelineLoader:
    async def load_wan_t2v_pipeline(self, model_config: Dict[str, Any]) -> WANT2VPipeline:
        """Load actual WAN T2V model implementation"""
        # Create WAN T2V model instance
        model = WANT2VA14B(model_config)

        # Load pretrained weights
        await model.load_pretrained_weights(model_config["checkpoint_path"])

        # Apply hardware optimizations
        if self.hardware_profile:
            model.optimize_for_hardware(self.hardware_profile)

        # Create pipeline wrapper
        return WANT2VPipeline(model, model_config)

    async def load_wan_i2v_pipeline(self, model_config: Dict[str, Any]) -> WANI2VPipeline:
        """Load actual WAN I2V model implementation"""
        # Similar implementation for I2V

    async def load_wan_ti2v_pipeline(self, model_config: Dict[str, Any]) -> WANTI2VPipeline:
        """Load actual WAN TI2V model implementation"""
        # Similar implementation for TI2V
```

### 3. WAN Model Configuration System

**Location**: `core/config/wan_model_configs.py` (new)

**Responsibilities**:

- Define WAN model architectures and parameters
- Handle model-specific configuration and optimization settings
- Provide hardware-specific configuration profiles
- Manage model checkpoint and weight file locations

**Configuration Structure**:

```python
WAN_MODEL_CONFIGS = {
    "t2v-A14B": {
        "model_class": "WANT2VA14B",
        "architecture": {
            "num_layers": 14,
            "hidden_dim": 1024,
            "attention_heads": 16,
            "temporal_layers": 8,
            "max_frames": 16,
            "resolution": [1280, 720]
        },
        "training": {
            "checkpoint_url": "https://huggingface.co/wan-ai/wan-t2v-a14b/resolve/main/pytorch_model.bin",
            "config_url": "https://huggingface.co/wan-ai/wan-t2v-a14b/resolve/main/config.json"
        },
        "optimization": {
            "fp16_enabled": True,
            "gradient_checkpointing": True,
            "cpu_offload_enabled": True,
            "vram_estimate_gb": 8.5
        },
        "hardware_profiles": {
            "rtx_4080": {
                "batch_size": 1,
                "enable_xformers": True,
                "vae_tile_size": 256,
                "cpu_offload": False
            },
            "low_vram": {
                "batch_size": 1,
                "enable_xformers": True,
                "vae_tile_size": 128,
                "cpu_offload": True
            }
        }
    },
    "i2v-A14B": {
        # Similar structure for I2V model
    },
    "ti2v-5B": {
        # Similar structure for TI2V model (smaller, 5B parameters)
    }
}
```

### 4. Enhanced Model Integration Bridge

**Location**: `backend/core/model_integration_bridge.py` (enhanced existing)

**Enhanced Methods**:

```python
class ModelIntegrationBridge:
    async def load_wan_model_implementation(self, model_type: str) -> WANBaseModel:
        """Load actual WAN model implementation instead of placeholder"""

        # Get WAN model configuration
        model_config = WAN_MODEL_CONFIGS.get(model_type)
        if not model_config:
            raise ValueError(f"Unknown WAN model type: {model_type}")

        # Download model weights if not cached
        checkpoint_path = await self._ensure_wan_model_weights(model_type, model_config)

        # Create model instance
        model_class = globals()[model_config["model_class"]]
        model = model_class(model_config)

        # Load weights
        await model.load_pretrained_weights(checkpoint_path)

        # Apply hardware optimizations
        if self.hardware_profile:
            model.optimize_for_hardware(self.hardware_profile)

        return model

    async def _ensure_wan_model_weights(self, model_type: str, model_config: Dict[str, Any]) -> str:
        """Ensure WAN model weights are downloaded and cached"""

        # Check if weights are already cached
        cache_path = Path("models") / f"wan-{model_type.lower()}" / "pytorch_model.bin"

        if cache_path.exists():
            # Verify integrity
            if await self._verify_model_integrity(cache_path, model_config):
                return str(cache_path)

        # Download weights using existing downloader infrastructure
        if self.model_downloader:
            download_result = await self.model_downloader.download_wan_model(
                model_type, model_config["training"]["checkpoint_url"]
            )

            if download_result.success:
                return download_result.local_path
            else:
                raise RuntimeError(f"Failed to download WAN model {model_type}: {download_result.error}")

        raise RuntimeError(f"Model downloader not available for WAN model {model_type}")
```

## Data Models

### WAN Model Configuration

```python
@dataclass
class WANModelConfig:
    """Configuration for WAN model instances"""
    model_type: str  # t2v-A14B, i2v-A14B, ti2v-5B
    architecture: Dict[str, Any]
    checkpoint_path: str
    optimization_settings: Dict[str, Any]
    hardware_profile: Optional[str] = None

    # Generation parameters
    max_frames: int = 16
    default_resolution: Tuple[int, int] = (1280, 720)
    default_fps: float = 8.0

    # Performance settings
    estimated_vram_gb: float = 8.0
    supports_cpu_offload: bool = True
    supports_quantization: bool = True

@dataclass
class WANGenerationResult:
    """Result from WAN model generation"""
    success: bool
    frames: Optional[torch.Tensor] = None
    generation_time: float = 0.0
    peak_memory_mb: float = 0.0
    memory_used_mb: float = 0.0
    applied_optimizations: List[str] = field(default_factory=list)
    model_info: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
```

### Enhanced Model Status

```python
@dataclass
class WANModelStatus:
    """Status information for WAN models"""
    model_type: str
    is_implemented: bool  # True when actual WAN model is loaded
    is_weights_available: bool
    is_loaded: bool
    implementation_version: str

    # Model-specific information
    architecture_info: Dict[str, Any]
    parameter_count: int
    estimated_vram_gb: float

    # Performance metrics
    average_generation_time: Optional[float] = None
    success_rate: Optional[float] = None
    hardware_compatibility: Dict[str, bool] = field(default_factory=dict)
```

## Error Handling

### WAN Model-Specific Error Handling

```python
class WANModelErrorHandler:
    """Handle errors specific to WAN model implementations"""

    def handle_wan_model_error(self, error: Exception, model_type: str, context: Dict[str, Any]) -> ErrorResponse:
        """Handle WAN model-specific errors with targeted recovery suggestions"""

        error_message = str(error)

        # WAN model loading errors
        if "checkpoint" in error_message.lower() or "weights" in error_message.lower():
            return ErrorResponse(
                error_type="wan_model_loading_error",
                message=f"Failed to load WAN {model_type} model weights",
                suggestions=[
                    f"Verify WAN {model_type} model weights are downloaded",
                    "Check internet connection for model download",
                    "Verify sufficient disk space for model files",
                    f"Try re-downloading WAN {model_type} model"
                ],
                recovery_actions=["download_wan_model", "verify_model_integrity"]
            )

        # WAN architecture errors
        elif "architecture" in error_message.lower() or "config" in error_message.lower():
            return ErrorResponse(
                error_type="wan_architecture_error",
                message=f"WAN {model_type} architecture configuration error",
                suggestions=[
                    f"Verify WAN {model_type} model configuration",
                    "Check model architecture parameters",
                    "Ensure model version compatibility"
                ],
                recovery_actions=["reset_model_config", "download_model_config"]
            )

        # WAN generation errors
        elif "generation" in error_message.lower() or "inference" in error_message.lower():
            return ErrorResponse(
                error_type="wan_generation_error",
                message=f"WAN {model_type} generation failed",
                suggestions=[
                    "Check input parameters for WAN model requirements",
                    "Verify sufficient VRAM for WAN model inference",
                    "Try reducing generation parameters (frames, resolution)",
                    "Enable CPU offloading for WAN model"
                ],
                recovery_actions=["optimize_generation_params", "enable_cpu_offload"]
            )

        # Default WAN error handling
        else:
            return ErrorResponse(
                error_type="wan_model_error",
                message=f"WAN {model_type} model error: {error_message}",
                suggestions=[
                    f"Check WAN {model_type} model status",
                    "Verify hardware compatibility",
                    "Try reloading the WAN model"
                ],
                recovery_actions=["reload_wan_model", "check_hardware_compatibility"]
            )
```

## Testing Strategy

### WAN Model Implementation Testing

1. **Model Architecture Tests**

   ```python
   async def test_wan_t2v_model_creation():
       """Test WAN T2V model can be created and initialized"""
       config = WAN_MODEL_CONFIGS["t2v-A14B"]
       model = WANT2VA14B(config)
       assert model is not None
       assert model.get_parameter_count() > 0

   async def test_wan_model_weight_loading():
       """Test WAN model weights can be loaded"""
       model = WANT2VA14B(WAN_MODEL_CONFIGS["t2v-A14B"])
       # Mock weight loading for testing
       await model.load_pretrained_weights("test_weights.bin")
       assert model.is_weights_loaded()
   ```

2. **Integration Tests**

   ```python
   async def test_wan_model_integration_bridge():
       """Test WAN models integrate with existing bridge"""
       bridge = ModelIntegrationBridge()
       await bridge.initialize()

       # Test WAN model loading
       model = await bridge.load_wan_model_implementation("t2v-A14B")
       assert isinstance(model, WANT2VA14B)
       assert model.is_ready_for_inference()

   async def test_wan_generation_pipeline():
       """Test WAN models work with existing generation pipeline"""
       pipeline = RealGenerationPipeline()
       await pipeline.initialize()

       # Test T2V generation with WAN model
       result = await pipeline.generate_t2v("test prompt", test_params)
       assert result.success
       assert result.frames is not None
   ```

3. **Hardware Optimization Tests**
   ```python
   async def test_wan_rtx4080_optimization():
       """Test WAN models work with RTX 4080 optimizations"""
       hardware_profile = HardwareProfile(
           gpu_name="RTX 4080",
           total_vram_gb=16.0,
           available_vram_gb=14.0
       )

       model = WANT2VA14B(WAN_MODEL_CONFIGS["t2v-A14B"])
       model.optimize_for_hardware(hardware_profile)

       # Verify optimizations applied
       assert model.is_optimized_for_hardware()
       assert model.get_vram_usage() <= hardware_profile.available_vram_gb
   ```

## Implementation Phases

### Phase 1: WAN Model Architecture (Week 1)

**Tasks:**

- Create WAN model base classes and architecture
- Implement WAN T2V-A14B model structure
- Create model configuration system
- Set up model weight download infrastructure

**Deliverables:**

- `core/models/wan_models/` directory with base implementations
- WAN model configuration files
- Basic model loading and initialization

### Phase 2: WAN Model Integration (Week 2)

**Tasks:**

- Integrate WAN models with existing ModelIntegrationBridge
- Update WanPipelineLoader to use actual WAN implementations
- Implement WAN I2V-A14B and TI2V-5B models
- Add WAN-specific error handling

**Deliverables:**

- Working WAN T2V generation
- WAN I2V and TI2V model implementations
- Enhanced error handling for WAN models

### Phase 3: Hardware Optimization Integration (Week 3)

**Tasks:**

- Integrate WAN models with existing hardware optimization
- Implement RTX 4080-specific optimizations for WAN models
- Add VRAM monitoring for actual WAN model usage
- Optimize WAN models for Threadripper PRO CPU

**Deliverables:**

- Hardware-optimized WAN model performance
- Real VRAM usage monitoring
- CPU optimization for preprocessing

### Phase 4: Testing and Validation (Week 4)

**Tasks:**

- Comprehensive testing of WAN model implementations
- Validate integration with existing infrastructure
- Performance benchmarking and optimization
- Documentation and deployment preparation

**Deliverables:**

- Complete test suite for WAN models
- Performance benchmarks
- Production-ready WAN model implementation

## Security Considerations

1. **Model Weight Security**

   - Verify integrity of downloaded WAN model weights
   - Use secure download channels for model checkpoints
   - Implement checksum validation for model files

2. **Model Execution Security**
   - Validate input parameters to prevent injection attacks
   - Limit resource usage to prevent DoS attacks
   - Secure model inference environment

## Performance Considerations

1. **WAN Model Optimization**

   - Implement model-specific optimizations for RTX 4080
   - Use efficient memory management for 14B/5B parameter models
   - Optimize temporal attention mechanisms for video generation

2. **Hardware Utilization**

   - Leverage existing RTX 4080 tensor core optimizations
   - Implement efficient CPU-GPU memory transfers
   - Use existing VRAM monitoring for optimal resource usage

3. **Generation Performance**
   - Optimize WAN model inference for real-time feedback
   - Implement efficient video encoding and output
   - Use existing progress tracking for accurate time estimates

## Deployment Strategy

1. **Gradual Model Rollout**

   - Deploy WAN T2V model first as primary MVP
   - Add WAN I2V and TI2V models incrementally
   - Maintain fallback to mock generation during transition

2. **Model Weight Management**

   - Implement automatic WAN model weight downloading
   - Use existing model caching and validation systems
   - Provide manual model management options

3. **Performance Monitoring**
   - Monitor WAN model performance and success rates
   - Track resource usage and optimization effectiveness
   - Collect metrics for future model improvements
