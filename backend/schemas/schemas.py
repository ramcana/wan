"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class ModelType(str, Enum):
    """Supported model types"""
    T2V_A14B = "T2V-A14B"
    I2V_A14B = "I2V-A14B"
    TI2V_5B = "TI2V-5B"

class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class QuantizationLevel(str, Enum):
    """Quantization levels for optimization"""
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"

class GenerationRequest(BaseModel):
    """Request model for video generation"""
    model_type: ModelType
    prompt: str = Field(..., min_length=1, max_length=500)
    resolution: str = Field(default="1280x720", pattern=r"^\d+x\d+$")
    steps: int = Field(default=50, ge=1, le=100)
    lora_path: Optional[str] = None
    lora_strength: float = Field(default=1.0, ge=0.0, le=2.0)
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v.strip()

class GenerationResponse(BaseModel):
    """Response model for generation requests"""
    task_id: str
    status: TaskStatus
    message: str
    estimated_time_minutes: Optional[int] = None

class TaskInfo(BaseModel):
    """Task information model"""
    id: str
    model_type: ModelType
    prompt: str
    image_path: Optional[str] = None
    resolution: str
    steps: int
    lora_path: Optional[str] = None
    lora_strength: float
    status: TaskStatus
    progress: int = Field(ge=0, le=100)
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    estimated_time_minutes: Optional[int] = None

class QueueStatus(BaseModel):
    """Queue status information"""
    total_tasks: int
    pending_tasks: int
    processing_tasks: int
    completed_tasks: int
    failed_tasks: int
    cancelled_tasks: int = 0
    tasks: List[TaskInfo]

class SystemStats(BaseModel):
    """System resource statistics"""
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    gpu_percent: float
    vram_used_mb: float
    vram_total_mb: float
    vram_percent: float
    timestamp: datetime

class OptimizationSettings(BaseModel):
    """Optimization settings model"""
    quantization: QuantizationLevel = QuantizationLevel.BF16
    enable_offload: bool = True
    vae_tile_size: int = Field(default=256, ge=128, le=512)
    max_vram_usage_gb: float = Field(default=12.0, ge=4.0, le=24.0)

class VideoMetadata(BaseModel):
    """Generated video metadata"""
    id: str
    filename: str
    file_path: str
    thumbnail_path: Optional[str] = None
    prompt: str
    model_type: ModelType
    resolution: str
    duration_seconds: Optional[float] = None
    file_size_mb: float
    created_at: datetime
    generation_time_minutes: Optional[float] = None

class OutputsResponse(BaseModel):
    """Response model for outputs listing"""
    videos: List[VideoMetadata]
    total_count: int
    total_size_mb: float

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    timestamp: datetime
    system_info: Optional[Dict[str, Any]] = None

class PromptEnhanceRequest(BaseModel):
    """Request model for prompt enhancement"""
    prompt: str = Field(..., min_length=1, max_length=500)
    apply_vace: Optional[bool] = None
    apply_cinematic: bool = True
    apply_style: bool = True

class PromptEnhanceResponse(BaseModel):
    """Response model for prompt enhancement"""
    original_prompt: str
    enhanced_prompt: str
    enhancements_applied: List[str]
    character_count: Dict[str, int]
    detected_style: Optional[str] = None
    vace_detected: bool = False

class PromptPreviewResponse(BaseModel):
    """Response model for prompt enhancement preview"""
    original_prompt: str
    preview_enhanced: str
    suggested_enhancements: List[str]
    detected_style: Optional[str] = None
    vace_detected: bool = False
    character_count: Dict[str, int]
    quality_score: Optional[float] = None

# LoRA-related schemas
class LoRAInfo(BaseModel):
    """LoRA file information"""
    name: str
    filename: str
    path: str
    size_mb: float
    modified_time: str
    is_loaded: bool = False
    is_applied: bool = False
    current_strength: float = 0.0

class LoRAListResponse(BaseModel):
    """Response model for LoRA listing"""
    loras: List[LoRAInfo]
    total_count: int
    total_size_mb: float

class LoRAUploadResponse(BaseModel):
    """Response model for LoRA upload"""
    success: bool
    message: str
    lora_name: str
    file_path: str
    size_mb: float

class LoRAStatusResponse(BaseModel):
    """Response model for LoRA status"""
    name: str
    exists: bool
    path: Optional[str] = None
    size_mb: Optional[float] = None
    is_loaded: bool = False
    is_applied: bool = False
    current_strength: float = 0.0
    modified_time: Optional[str] = None

class LoRAPreviewResponse(BaseModel):
    """Response model for LoRA preview"""
    lora_name: str
    base_prompt: str
    enhanced_prompt: str
    style_indicators: List[str]
    preview_note: str

class LoRAMemoryImpactResponse(BaseModel):
    """Response model for LoRA memory impact estimation"""
    lora_name: str
    file_size_mb: float
    estimated_memory_mb: float
    vram_available_mb: float
    can_load: bool
    memory_impact: str  # "low", "medium", "high"
    recommendation: str
