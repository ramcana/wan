from pydantic import BaseModel, Field, validator
from typing import Optional
import re
from enum import Enum


class VideoGenerationMode(str, Enum):
    TEXT_TO_VIDEO = "t2v"
    IMAGE_TO_VIDEO = "i2v"
    TEXT_IMAGE_TO_VIDEO = "ti2v"


class VideoRequest(BaseModel):
    mode: VideoGenerationMode
    prompt: str = Field(..., min_length=5, max_length=500)
    negative_prompt: Optional[str] = Field(None, max_length=300)
    width: int = Field(1024, ge=256, le=1920)
    height: int = Field(576, ge=256, le=1080)
    duration: float = Field(5.0, ge=1.0, le=10.0)
    fps: int = Field(24, ge=12, le=30)
    seed: Optional[int] = Field(None, ge=0, le=2**32 - 1)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    num_inference_steps: int = Field(50, ge=10, le=100)

    @validator("prompt")
    def validate_prompt(cls, v):
        # Remove potentially harmful content
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")

        # Check for prohibited content patterns
        prohibited_patterns = [
            r"\b(nude|naked|nsfw|porn|sexual)\b",
            r"\b(violence|blood|gore|death|kill)\b",
            r"\b(hate|racist|discrimination)\b",
        ]

        for pattern in prohibited_patterns:
            if re.search(pattern, v.lower()):
                raise ValueError("Prompt contains prohibited content")

        return v.strip()

    @validator("negative_prompt")
    def validate_negative_prompt(cls, v):
        if v:
            return v.strip()
        return v

    @validator("width", "height")
    def validate_dimensions(cls, v):
        # Ensure dimensions are multiples of 8 (common requirement for
        # video models)
        if v % 8 != 0:
            raise ValueError("Dimensions must be multiples of 8")
        return v


class ImageUpload(BaseModel):
    filename: str = Field(..., max_length=255)
    content_type: str
    size: int = Field(..., le=10 * 1024 * 1024)  # 10MB max

    @validator("filename")
    def validate_filename(cls, v):
        # Sanitize filename
        if not v:
            raise ValueError("Filename cannot be empty")

        # Remove dangerous characters
        safe_filename = re.sub(r"[^a-zA-Z0-9._-]", "", v)
        if not safe_filename:
            raise ValueError("Invalid filename")

        return safe_filename

    @validator("content_type")
    def validate_content_type(cls, v):
        allowed_types = ["image/jpeg", "image/png", "image/webp"]
        if v not in allowed_types:
            raise ValueError(f"Content type must be one of: {allowed_types}")
        return v


class LoRAConfig(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    strength: float = Field(1.0, ge=0.1, le=2.0)

    @validator("name")
    def validate_lora_name(cls, v):
        # Only allow alphanumeric and safe characters
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "LoRA name can only contain alphanumeric characters, "
                "underscores, and hyphens"
            )
        return v
