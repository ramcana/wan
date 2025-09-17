# Security Implementation Plan - WAN Video Generation System

## Overview
This plan implements a comprehensive security layer for the WAN video generation system, adding authentication, rate limiting, and input validation while maintaining the existing architecture.

## Phase 1: Foundation & Authentication (Weeks 1-2)

### 1.1 Database Schema Extensions

**File: `backend/models/auth.py`**
```python
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to rate limiting
    rate_limits = relationship("UserRateLimit", back_populates="user")
    api_keys = relationship("APIKey", back_populates="user")

class APIKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    key_hash = Column(String(255), unique=True, nullable=False)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    last_used = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    
    user = relationship("User", back_populates="api_keys")

class UserRateLimit(Base):
    __tablename__ = "user_rate_limits"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    endpoint = Column(String(100), nullable=False)
    requests_count = Column(Integer, default=0)
    window_start = Column(DateTime, default=datetime.utcnow)
    daily_limit = Column(Integer, default=100)
    hourly_limit = Column(Integer, default=10)
    
    user = relationship("User", back_populates="rate_limits")
```

### 1.2 Authentication Service

**File: `backend/services/auth_service.py`**
```python
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import secrets
import hashlib

class AuthService:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        return self.pwd_context.hash(password)

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "type": "access"})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(self, data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[dict]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            return None

    def generate_api_key(self) -> tuple[str, str]:
        """Generate API key and return (key, hash)"""
        api_key = f"wan_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        return api_key, key_hash

    def verify_api_key(self, api_key: str, stored_hash: str) -> bool:
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        return secrets.compare_digest(key_hash, stored_hash)
```

### 1.3 Authentication Middleware

**File: `backend/middleware/auth_middleware.py`**
```python
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from backend.services.auth_service import AuthService
from backend.models.auth import User, APIKey
from backend.database import get_db
import re

security = HTTPBearer(auto_error=False)

class AuthMiddleware:
    def __init__(self, auth_service: AuthService):
        self.auth_service = auth_service

    async def get_current_user(
        self, 
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: Session = Depends(get_db)
    ) -> User:
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header required"
            )

        token = credentials.credentials
        
        # Check if it's an API key
        if token.startswith("wan_"):
            return await self._verify_api_key(token, db)
        
        # Otherwise, treat as JWT token
        return await self._verify_jwt_token(token, db)

    async def _verify_api_key(self, api_key: str, db: Session) -> User:
        # Find API key in database
        api_key_record = db.query(APIKey).filter(
            APIKey.is_active == True
        ).all()
        
        for record in api_key_record:
            if self.auth_service.verify_api_key(api_key, record.key_hash):
                if record.expires_at and record.expires_at < datetime.utcnow():
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="API key expired"
                    )
                
                # Update last used
                record.last_used = datetime.utcnow()
                db.commit()
                
                user = db.query(User).filter(User.id == record.user_id).first()
                if not user or not user.is_active:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User account inactive"
                    )
                return user
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    async def _verify_jwt_token(self, token: str, db: Session) -> User:
        payload = self.auth_service.verify_token(token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )

        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )

        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        return user

    def require_admin(self, current_user: User = Depends(get_current_user)) -> User:
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        return current_user
```

## Phase 2: Rate Limiting (Week 3)

### 2.1 Rate Limiting Service

**File: `backend/services/rate_limit_service.py`**
```python
from typing import Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from backend.models.auth import User, UserRateLimit
import asyncio
from collections import defaultdict
import time

class RateLimitService:
    def __init__(self):
        # In-memory cache for performance
        self._cache: Dict[str, Dict] = defaultdict(dict)
        self._default_limits = {
            "video_generation": {"hourly": 5, "daily": 20},
            "api_general": {"hourly": 100, "daily": 1000},
            "upload": {"hourly": 50, "daily": 200}
        }

    async def check_rate_limit(
        self, 
        user_id: str, 
        endpoint: str, 
        db: Session,
        custom_limits: Optional[Dict] = None
    ) -> tuple[bool, Dict]:
        """
        Check if user has exceeded rate limits
        Returns: (is_allowed, limit_info)
        """
        now = datetime.utcnow()
        cache_key = f"{user_id}:{endpoint}"
        
        # Get limits (custom or default)
        limits = custom_limits or self._default_limits.get(endpoint, self._default_limits["api_general"])
        
        # Check cache first for performance
        if cache_key in self._cache:
            cache_data = self._cache[cache_key]
            if self._is_cache_valid(cache_data, now):
                return self._evaluate_limits(cache_data, limits, now)
        
        # Fallback to database
        rate_record = db.query(UserRateLimit).filter(
            UserRateLimit.user_id == user_id,
            UserRateLimit.endpoint == endpoint
        ).first()
        
        if not rate_record:
            rate_record = UserRateLimit(
                user_id=user_id,
                endpoint=endpoint,
                requests_count=0,
                window_start=now,
                daily_limit=limits["daily"],
                hourly_limit=limits["hourly"]
            )
            db.add(rate_record)
        
        # Reset counters if windows have expired
        self._reset_expired_windows(rate_record, now)
        
        # Check limits
        hourly_count = self._get_hourly_count(cache_key, now)
        daily_count = rate_record.requests_count
        
        is_allowed = (
            hourly_count < limits["hourly"] and 
            daily_count < limits["daily"]
        )
        
        if is_allowed:
            # Increment counters
            rate_record.requests_count += 1
            self._increment_cache(cache_key, now)
            db.commit()
        
        limit_info = {
            "hourly_limit": limits["hourly"],
            "hourly_remaining": max(0, limits["hourly"] - hourly_count - (1 if is_allowed else 0)),
            "daily_limit": limits["daily"],
            "daily_remaining": max(0, limits["daily"] - daily_count - (1 if is_allowed else 0)),
            "reset_time": self._get_next_reset_time(now)
        }
        
        return is_allowed, limit_info

    def _is_cache_valid(self, cache_data: Dict, now: datetime) -> bool:
        return (now - cache_data.get("last_updated", datetime.min)).seconds < 60

    def _evaluate_limits(self, cache_data: Dict, limits: Dict, now: datetime) -> tuple[bool, Dict]:
        hourly_count = len([
            req for req in cache_data.get("requests", [])
            if (now - req).seconds < 3600
        ])
        
        is_allowed = hourly_count < limits["hourly"]
        
        limit_info = {
            "hourly_limit": limits["hourly"],
            "hourly_remaining": max(0, limits["hourly"] - hourly_count),
            "daily_limit": limits["daily"],
            "daily_remaining": limits["daily"],  # Cache doesn't track daily
            "reset_time": self._get_next_reset_time(now)
        }
        
        return is_allowed, limit_info

    def _reset_expired_windows(self, rate_record: UserRateLimit, now: datetime):
        # Reset daily counter if day has passed
        if rate_record.window_start.date() < now.date():
            rate_record.requests_count = 0
            rate_record.window_start = now

    def _get_hourly_count(self, cache_key: str, now: datetime) -> int:
        if cache_key not in self._cache:
            return 0
        
        requests = self._cache[cache_key].get("requests", [])
        hour_ago = now - timedelta(hours=1)
        return len([req for req in requests if req > hour_ago])

    def _increment_cache(self, cache_key: str, now: datetime):
        if cache_key not in self._cache:
            self._cache[cache_key] = {"requests": [], "last_updated": now}
        
        self._cache[cache_key]["requests"].append(now)
        self._cache[cache_key]["last_updated"] = now
        
        # Cleanup old requests (keep only last 24 hours)
        day_ago = now - timedelta(days=1)
        self._cache[cache_key]["requests"] = [
            req for req in self._cache[cache_key]["requests"] 
            if req > day_ago
        ]

    def _get_next_reset_time(self, now: datetime) -> datetime:
        # Next hour reset
        return (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
```

### 2.2 Rate Limiting Middleware

**File: `backend/middleware/rate_limit_middleware.py`**
```python
from fastapi import HTTPException, Depends, Request, status
from backend.services.rate_limit_service import RateLimitService
from backend.models.auth import User
from backend.middleware.auth_middleware import AuthMiddleware
from backend.database import get_db
from sqlalchemy.orm import Session

class RateLimitMiddleware:
    def __init__(self, rate_limit_service: RateLimitService):
        self.rate_limit_service = rate_limit_service
        
        # Endpoint-specific limits
        self.endpoint_limits = {
            "/api/v1/video/generate": {"hourly": 5, "daily": 20},
            "/api/v1/video/upload": {"hourly": 20, "daily": 100},
            "/api/v1/models/": {"hourly": 50, "daily": 500},
        }

    async def check_rate_limit(
        self,
        request: Request,
        current_user: User = Depends(AuthMiddleware.get_current_user),
        db: Session = Depends(get_db)
    ):
        """Rate limiting dependency"""
        endpoint_pattern = self._get_endpoint_pattern(request.url.path)
        custom_limits = self.endpoint_limits.get(endpoint_pattern)
        
        is_allowed, limit_info = await self.rate_limit_service.check_rate_limit(
            user_id=current_user.id,
            endpoint=endpoint_pattern,
            db=db,
            custom_limits=custom_limits
        )
        
        if not is_allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(limit_info["hourly_limit"]),
                    "X-RateLimit-Remaining": str(limit_info["hourly_remaining"]),
                    "X-RateLimit-Reset": str(int(limit_info["reset_time"].timestamp())),
                    "Retry-After": str((limit_info["reset_time"] - datetime.utcnow()).seconds)
                }
            )
        
        # Add rate limit headers to successful requests
        request.state.rate_limit_headers = {
            "X-RateLimit-Limit": str(limit_info["hourly_limit"]),
            "X-RateLimit-Remaining": str(limit_info["hourly_remaining"]),
            "X-RateLimit-Reset": str(int(limit_info["reset_time"].timestamp())),
        }

    def _get_endpoint_pattern(self, path: str) -> str:
        """Map request path to endpoint pattern for rate limiting"""
        if path.startswith("/api/v1/video/generate"):
            return "/api/v1/video/generate"
        elif path.startswith("/api/v1/video/upload"):
            return "/api/v1/video/upload"
        elif path.startswith("/api/v1/models"):
            return "/api/v1/models/"
        else:
            return "api_general"
```

## Phase 3: Input Validation (Week 4)

### 3.1 Input Validation Schemas

**File: `backend/schemas/validation.py`**
```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Union
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
    seed: Optional[int] = Field(None, ge=0, le=2**32-1)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    num_inference_steps: int = Field(50, ge=10, le=100)
    
    @validator('prompt')
    def validate_prompt(cls, v):
        # Remove potentially harmful content
        if not v or not v.strip():
            raise ValueError('Prompt cannot be empty')
        
        # Check for prohibited content patterns
        prohibited_patterns = [
            r'\b(nude|naked|nsfw|porn|sexual)\b',
            r'\b(violence|blood|gore|death|kill)\b',
            r'\b(hate|racist|discrimination)\b'
        ]
        
        for pattern in prohibited_patterns:
            if re.search(pattern, v.lower()):
                raise ValueError('Prompt contains prohibited content')
        
        return v.strip()
    
    @validator('negative_prompt')
    def validate_negative_prompt(cls, v):
        if v:
            return v.strip()
        return v

    @validator('width', 'height')
    def validate_dimensions(cls, v):
        # Ensure dimensions are multiples of 8 (common requirement for video models)
        if v % 8 != 0:
            raise ValueError('Dimensions must be multiples of 8')
        return v

class ImageUpload(BaseModel):
    filename: str = Field(..., max_length=255)
    content_type: str
    size: int = Field(..., le=10*1024*1024)  # 10MB max
    
    @validator('filename')
    def validate_filename(cls, v):
        # Sanitize filename
        if not v:
            raise ValueError('Filename cannot be empty')
        
        # Remove dangerous characters
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '', v)
        if not safe_filename:
            raise ValueError('Invalid filename')
        
        return safe_filename
    
    @validator('content_type')
    def validate_content_type(cls, v):
        allowed_types = [
            'image/jpeg', 'image/png', 'image/webp'
        ]
        if v not in allowed_types:
            raise ValueError(f'Content type must be one of: {allowed_types}')
        return v

class LoRAConfig(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    strength: float = Field(1.0, ge=0.1, le=2.0)
    
    @validator('name')
    def validate_lora_name(cls, v):
        # Only allow alphanumeric and safe characters
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('LoRA name can only contain alphanumeric characters, underscores, and hyphens')
        return v
```

### 3.2 Input Sanitization Service

**File: `backend/services/input_validation_service.py`**
```python
import re
import html
import bleach
from typing import Any, Dict, List
import magic
from pathlib import Path

class InputValidationService:
    def __init__(self):
        # Allowed HTML tags for rich text inputs (if any)
        self.allowed_tags = []
        self.allowed_attributes = {}
        
        # File type validation
        self.allowed_image_types = {
            'image/jpeg': [b'\xff\xd8\xff'],
            'image/png': [b'\x89\x50\x4e\x47'],
            'image/webp': [b'\x52\x49\x46\x46']
        }
        
        # Content filters
        self.prohibited_content_patterns = [
            r'\b(?:nude|naked|nsfw|porn|sexual|erotic)\b',
            r'\b(?:violence|blood|gore|death|kill|murder)\b',
            r'\b(?:hate|racist|discrimination|nazi|terrorist)\b',
            r'\b(?:child|minor|kid|teen).*(?:nude|sexual|porn)\b',
            r'(?:javascript|script|iframe|object|embed|form)',
        ]

    def sanitize_text(self, text: str, max_length: int = 1000) -> str:
        """Sanitize and clean text input"""
        if not text:
            return ""
        
        # Limit length
        text = text[:max_length]
        
        # HTML escape
        text = html.escape(text)
        
        # Remove potentially dangerous characters
        text = re.sub(r'[<>"\']', '', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def validate_content_policy(self, text: str) -> tuple[bool, List[str]]:
        """Check if text violates content policy"""
        violations = []
        
        for pattern in self.prohibited_content_patterns:
            if re.search(pattern, text.lower(), re.IGNORECASE):
                violations.append(f"Content matches prohibited pattern: {pattern}")
        
        return len(violations) == 0, violations

    def validate_file_upload(self, file_content: bytes, filename: str, content_type: str) -> tuple[bool, List[str]]:
        """Validate uploaded file"""
        errors = []
        
        # Check file size (10MB max)
        if len(file_content) > 10 * 1024 * 1024:
            errors.append("File size exceeds 10MB limit")
        
        # Validate file extension
        file_ext = Path(filename).suffix.lower()
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        if file_ext not in allowed_extensions:
            errors.append(f"File extension {file_ext} not allowed")
        
        # Validate MIME type
        if content_type not in self.allowed_image_types:
            errors.append(f"Content type {content_type} not allowed")
        
        # Validate file signature (magic bytes)
        if not self._validate_file_signature(file_content, content_type):
            errors.append("File signature doesn't match content type")
        
        # Scan for embedded scripts or malicious content
        if self._contains_malicious_content(file_content):
            errors.append("File contains potentially malicious content")
        
        return len(errors) == 0, errors

    def _validate_file_signature(self, content: bytes, content_type: str) -> bool:
        """Check if file signature matches declared content type"""
        if content_type not in self.allowed_image_types:
            return False
        
        signatures = self.allowed_image_types[content_type]
        for signature in signatures:
            if content.startswith(signature):
                return True
        
        return False

    def _contains_malicious_content(self, content: bytes) -> bool:
        """Basic check for malicious content in files"""
        try:
            # Convert to string for pattern matching
            content_str = content.decode('utf-8', errors='ignore').lower()
            
            malicious_patterns = [
                r'<script',
                r'javascript:',
                r'vbscript:',
                r'onload=',
                r'onerror=',
                r'eval\(',
                r'document\.cookie'
            ]
            
            for pattern in malicious_patterns:
                if re.search(pattern, content_str):
                    return True
            
        except Exception:
            # If we can't decode, consider it suspicious
            return True
        
        return False

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage"""
        if not filename:
            return "unknown_file"
        
        # Remove path components
        filename = Path(filename).name
        
        # Remove dangerous characters
        filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        
        # Limit length
        name_part = Path(filename).stem[:50]
        ext_part = Path(filename).suffix[:10]
        
        return f"{name_part}{ext_part}"

    def validate_json_payload(self, payload: Dict[str, Any], max_depth: int = 5, max_size: int = 1024*1024) -> tuple[bool, List[str]]:
        """Validate JSON payload for depth and size"""
        errors = []
        
        # Check payload size (serialized)
        try:
            import json
            payload_str = json.dumps(payload)
            if len(payload_str) > max_size:
                errors.append(f"Payload size {len(payload_str)} exceeds limit {max_size}")
        except Exception as e:
            errors.append(f"Payload serialization error: {str(e)}")
        
        # Check depth
        def check_depth(obj, current_depth=0):
            if current_depth > max_depth:
                return False
            
            if isinstance(obj, dict):
                return all(check_depth(v, current_depth + 1) for v in obj.values())
            elif isinstance(obj, list):
                return all(check_depth(item, current_depth + 1) for item in obj)
            
            return True
        
        if not check_depth(payload):
            errors.append(f"Payload depth exceeds limit {max_depth}")
        
        return len(errors) == 0, errors
```

## Phase 4: Integration & API Endpoints (Week 5)

### 4.1 Updated API Endpoints with Security

**File: `backend/api/v1/endpoints/video.py`**
```python
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from backend.schemas.validation import VideoRequest, ImageUpload
from backend.services.input_validation_service import InputValidationService
from backend.middleware.auth_middleware import AuthMiddleware
from backend.middleware.rate_limit_middleware import RateLimitMiddleware
from backend.services.video_service import VideoService
from backend.database import get_db
from backend.models.auth import User
import uuid

router = APIRouter()

# Initialize services
input_validator = InputValidationService()
auth_middleware = AuthMiddleware()
rate_limit_middleware = RateLimitMiddleware()

@router.post("/generate")
async def generate_video(
    request: Request,
    video_request: VideoRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(auth_middleware.get_current_user),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(rate_limit_middleware.check_rate_limit)
):
    """Generate video with security validation"""
    
    # Validate content policy
    is_valid, violations = input_validator.validate_content_policy(video_request.prompt)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Content policy violation",
                "violations": violations
            }
        )
    
    # Sanitize inputs
    sanitized_prompt = input_validator.sanitize_text(video_request.prompt, 500)
    sanitized_negative_prompt = input_validator.sanitize_text(
        video_request.negative_prompt or "", 300
    )
    
    # Create video generation task
    task_id = str(uuid.uuid4())
    
    # Queue the generation task
    background_tasks.add_task(
        VideoService.generate_video_async,
        task_id=task_id,
        user_id=current_user.id,
        prompt=sanitized_prompt,
        negative_prompt=sanitized_negative_prompt,
        **video_request.dict(exclude={'prompt', 'negative_prompt'})
    )
    
    # Add rate limit headers to response
    response_headers = getattr(request.state, 'rate_limit_headers', {})
    
    return JSONResponse(
        content={
            "task_id": task_id,
            "status": "queued",
            "message": "Video generation started"
        },
        headers=response_headers
    )

@router.post("/upload-image")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(auth_middleware.get_current_user),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(rate_limit_middleware.check_rate_limit)
):
    """Upload image with security validation"""
    
    # Read file content
    file_content = await file.read()
    
    # Validate file upload
    is_valid, errors = input_validator.validate_file_upload(
        file_content, file.filename, file.content_type
    )
    
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "File validation failed",
                "errors": errors
            }
        )
    
    # Sanitize filename
    safe_filename = input_validator.sanitize_filename(file.filename)
    
    # Store file securely
    file_id = str(uuid.uuid4())
    storage_path = f"uploads/{current_user.id}/{file_id}_{safe_filename}"
    
    # Save file and create database record
    try:
        # Store file using your storage service
        await VideoService.store_uploaded_file(
            file_content, storage_path, current_user.id
        )
        
        response_headers = getattr(request.state, 'rate_limit_headers', {})
        
        return JSONResponse(
            content={
                "file_id": file_id,
                "filename": safe_filename,
                "size": len(file_content),
                "status": "uploaded"
            },
            headers=response_headers
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(e)}"
        )

@router.get("/task/{task_id}")
async def get_task_status(
    task_id: str,
    current_user: User = Depends(auth_middleware.get_current_user),
    db: Session = Depends(get_db)
):
    """Get video generation task status"""
    
    # Validate task_id format
    try:
        uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format")
    
    # Get task status (ensure user can only see their own tasks)
    task_status = await VideoService.get_task_status(task_id, current_user.id)
    
    if not task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_status
```

### 4.2 Authentication Endpoints

**File: `backend/api/v1/endpoints/auth.py`**
```python
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from backend.schemas.auth import UserCreate, UserLogin, Token, APIKeyCreate, APIKeyResponse
from backend.services.auth_service import AuthService
from backend.services.input_validation_service import InputValidationService
from backend.models.auth import User, APIKey
from backend.database import get_db
from backend.middleware.auth_middleware import AuthMiddleware
from datetime import datetime, timedelta
import re

router = APIRouter()
auth_service = AuthService()
input_validator = InputValidationService()
auth_middleware = AuthMiddleware()

@router.post("/register", response_model=dict)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """Register new user with input validation"""
    
    # Validate email format
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
    if not re.match(email_pattern, user_data.email):
        raise HTTPException(
            status_code=400,
            detail="Invalid email format"
        )
    
    # Validate password strength
    if len(user_data.password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters long"
        )
    
    if not re.search(r'[A-Z]', user_data.password):
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least one uppercase letter"
        )
    
    if not re.search(r'[0-9]', user_data.password):
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least one number"
        )
    
    # Sanitize inputs
    username = input_validator.sanitize_text(user_data.username, 50)
    email = user_data.email.lower().strip()
    
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.username == username) | (User.email == email)
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Username or email already registered"
        )
    
    # Create user
    hashed_password = auth_service.get_password_hash(user_data.password)
    new_user = User(
        username=username,
        email=email,
        hashed_password=hashed_password
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {
        "message": "User registered successfully",
        "user_id": new_user.id
    }

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login user and return JWT tokens"""
    
    # Find user by username or email
    user = db.query(User).filter(
        (User.username == form_data.username) | 
        (User.email == form_data.username)
    ).first()
    
    if not user or not auth_service.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is inactive"
        )
    
    # Create tokens
    access_token = auth_service.create_access_token(data={"sub": user.id})
    refresh_token = auth_service.create_refresh_token(data={"sub": user.id})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@router.post("/refresh", response_model=dict)
async def refresh_token(
    refresh_token: str,
    db: Session = Depends(get_db)
):
    """Refresh access token"""
    
    payload = auth_service.verify_token(refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new access token
    new_access_token = auth_service.create_access_token(data={"sub": user.id})
    
    return {
        "access_token": new_access_token,
        "token_type": "bearer"
    }

@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    key_data: APIKeyCreate,
    current_user: User = Depends(auth_middleware.get_current_user),
    db: Session = Depends(get_db)
):
    """Create new API key for user"""
    
    # Validate key name
    key_name = input_validator.sanitize_text(key_data.name, 100)
    if not key_name:
        raise HTTPException(
            status_code=400,
            detail="API key name is required"
        )
    
    # Check if user already has a key with this name
    existing_key = db.query(APIKey).filter(
        APIKey.user_id == current_user.id,
        APIKey.name == key_name,
        APIKey.is_active == True
    ).first()
    
    if existing_key:
        raise HTTPException(
            status_code=400,
            detail="API key with this name already exists"
        )
    
    # Generate API key
    api_key, key_hash = auth_service.generate_api_key()
    
    # Set expiration if provided
    expires_at = None
    if key_data.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=key_data.expires_in_days)
    
    # Create API key record
    new_api_key = APIKey(
        key_hash=key_hash,
        user_id=current_user.id,
        name=key_name,
        expires_at=expires_at
    )
    
    db.add(new_api_key)
    db.commit()
    
    return {
        "api_key": api_key,  # Only returned once!
        "key_id": new_api_key.id,
        "name": key_name,
        "expires_at": expires_at,
        "created_at": new_api_key.created_at
    }

@router.get("/api-keys")
async def list_api_keys(
    current_user: User = Depends(auth_middleware.get_current_user),
    db: Session = Depends(get_db)
):
    """List user's API keys (without the actual keys)"""
    
    keys = db.query(APIKey).filter(
        APIKey.user_id == current_user.id,
        APIKey.is_active == True
    ).all()
    
    return [
        {
            "key_id": key.id,
            "name": key.name,
            "created_at": key.created_at,
            "last_used": key.last_used,
            "expires_at": key.expires_at
        }
        for key in keys
    ]

@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: User = Depends(auth_middleware.get_current_user),
    db: Session = Depends(get_db)
):
    """Revoke API key"""
    
    api_key = db.query(APIKey).filter(
        APIKey.id == key_id,
        APIKey.user_id == current_user.id,
        APIKey.is_active == True
    ).first()
    
    if not api_key:
        raise HTTPException(
            status_code=404,
            detail="API key not found"
        )
    
    api_key.is_active = False
    db.commit()
    
    return {"message": "API key revoked successfully"}
```

### 4.3 Security Configuration

**File: `backend/core/security_config.py`**
```python
from pydantic import BaseSettings
from typing import List
import os

class SecuritySettings(BaseSettings):
    # JWT Settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-super-secret-key-change-this")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Rate Limiting
    DEFAULT_RATE_LIMIT_PER_HOUR: int = 100
    VIDEO_GENERATION_LIMIT_PER_HOUR: int = 5
    VIDEO_GENERATION_LIMIT_PER_DAY: int = 20
    UPLOAD_LIMIT_PER_HOUR: int = 20
    
    # Content Validation
    MAX_PROMPT_LENGTH: int = 500
    MAX_NEGATIVE_PROMPT_LENGTH: int = 300
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]
    
    # Security Headers
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE"]
    CORS_HEADERS: List[str] = ["*"]
    
    # Content Security
    ENABLE_CONTENT_FILTER: bool = True
    STRICT_CONTENT_VALIDATION: bool = True
    
    class Config:
        env_file = ".env"

security_settings = SecuritySettings()
```

### 4.4 Main Application Integration

**File: `backend/main.py` (Security Integration)**
```python
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time
import uuid

# Import security components
from backend.core.security_config import security_settings
from backend.api.v1.endpoints import auth, video
from backend.middleware.auth_middleware import AuthMiddleware
from backend.services.auth_service import AuthService

app = FastAPI(
    title="WAN Video Generation API",
    description="Secure AI Video Generation System",
    version="2.2.0"
)

# Initialize security services
auth_service = AuthService(
    secret_key=security_settings.SECRET_KEY,
    algorithm=security_settings.ALGORITHM
)

# Security Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=security_settings.CORS_ORIGINS,
    allow_credentials=security_settings.CORS_CREDENTIALS,
    allow_methods=security_settings.CORS_METHODS,
    allow_headers=security_settings.CORS_HEADERS,
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.yourdomain.com"]
)

@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses"""
    
    # Add request ID for tracking
    request.state.request_id = str(uuid.uuid4())
    
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Request-ID"] = request.state.request_id
    
    # Add rate limit headers if present
    if hasattr(request.state, 'rate_limit_headers'):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value
    
    return response

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log requests for security monitoring"""
    start_time = time.time()
    
    # Log request
    print(f"Request: {request.method} {request.url} - {request.client.host}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    print(f"Response: {response.status_code} - {process_time:.4f}s")
    
    return response

# Exception handlers
@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc):
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "message": "Too many requests. Please try again later.",
            "request_id": getattr(request.state, 'request_id', None)
        }
    )

@app.exception_handler(401)
async def auth_exception_handler(request: Request, exc):
    return JSONResponse(
        status_code=401,
        content={
            "error": "Authentication failed",
            "message": "Invalid or expired credentials",
            "request_id": getattr(request.state, 'request_id', None)
        }
    )

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(video.router, prefix="/api/v1/video", tags=["video"])

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.2.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        ssl_keyfile=None,  # Add SSL cert paths for production
        ssl_certfile=None
    )
```

## Phase 5: Frontend Security Integration (Week 6)

### 5.1 Authentication Context

**File: `frontend/src/contexts/AuthContext.tsx`**
```typescript
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface User {
  id: string;
  username: string;
  email: string;
  isAdmin: boolean;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
  isAuthenticated: boolean;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check for stored authentication
    const storedToken = localStorage.getItem('access_token');
    if (storedToken) {
      validateToken(storedToken);
    } else {
      setIsLoading(false);
    }
  }, []);

  const validateToken = async (token: string) => {
    try {
      const response = await fetch('/api/v1/auth/me', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const userData = await response.json();
        setUser(userData);
        setToken(token);
      } else {
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
      }
    } catch (error) {
      console.error('Token validation failed:', error);
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
    } finally {
      setIsLoading(false);
    }
  };

  const login = async (username: string, password: string): Promise<boolean> => {
    try {
      const formData = new FormData();
      formData.append('username', username);
      formData.append('password', password);

      const response = await fetch('/api/v1/auth/login', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('refresh_token', data.refresh_token);
        
        setToken(data.access_token);
        await validateToken(data.access_token);
        
        return true;
      }
      return false;
    } catch (error) {
      console.error('Login failed:', error);
      return false;
    }
  };

  const logout = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    setUser(null);
    setToken(null);
  };

  const value = {
    user,
    token,
    login,
    logout,
    isAuthenticated: !!user,
    isLoading
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
```

### 5.2 Secure API Client

**File: `frontend/src/services/apiClient.ts`**
```typescript
import { useAuth } from '../contexts/AuthContext';

class ApiClient {
  private baseURL: string;
  private getToken: () => string | null;

  constructor(baseURL: string, getToken: () => string | null) {
    this.baseURL = baseURL;
    this.getToken = getToken;
  }

  private async request<T>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    const token = this.getToken();

    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...(token && { 'Authorization': `Bearer ${token}` }),
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      // Handle rate limiting
      if (response.status === 429) {
        const retryAfter = response.headers.get('Retry-After');
        throw new Error(`Rate limited. Retry after ${retryAfter} seconds`);
      }

      // Handle authentication errors
      if (response.status === 401) {
        // Try to refresh token
        const refreshed = await this.refreshToken();
        if (refreshed) {
          // Retry the original request
          return this.request(endpoint, options);
        } else {
          throw new Error('Authentication failed');
        }
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  private async refreshToken(): Promise<boolean> {
    try {
      const refreshToken = localStorage.getItem('refresh_token');
      if (!refreshToken) return false;

      const response = await fetch(`${this.baseURL}/api/v1/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });

      if (response.ok) {
        const data = await response.json();
        localStorage.setItem('access_token', data.access_token);
        return true;
      }
      
      return false;
    } catch {
      return false;
    }
  }

  // Video generation methods
  async generateVideo(request: VideoGenerationRequest): Promise<VideoGenerationResponse> {
    return this.request<VideoGenerationResponse>('/api/v1/video/generate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async uploadImage(file: File): Promise<ImageUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    return this.request<ImageUploadResponse>('/api/v1/video/upload-image', {
      method: 'POST',
      body: formData,
      headers: {}, // Let browser set Content-Type for FormData
    });
  }

  async getTaskStatus(taskId: string): Promise<TaskStatusResponse> {
    return this.request<TaskStatusResponse>(`/api/v1/video/task/${taskId}`);
  }

  // API Key management
  async createApiKey(name: string, expiresInDays?: number): Promise<ApiKeyResponse> {
    return this.request<ApiKeyResponse>('/api/v1/auth/api-keys', {
      method: 'POST',
      body: JSON.stringify({ name, expires_in_days: expiresInDays }),
    });
  }

  async listApiKeys(): Promise<ApiKeyListResponse> {
    return this.request<ApiKeyListResponse>('/api/v1/auth/api-keys');
  }

  async revokeApiKey(keyId: string): Promise<void> {
    return this.request<void>(`/api/v1/auth/api-keys/${keyId}`, {
      method: 'DELETE',
    });
  }
}

// Hook for using the API client
export const useApiClient = () => {
  const { token } = useAuth();
  
  return new ApiClient(
    process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000',
    () => token
  );
};

// Types
interface VideoGenerationRequest {
  mode: 't2v' | 'i2v' | 'ti2v';
  prompt: string;
  negative_prompt?: string;
  width: number;
  height: number;
  duration: number;
  fps: number;
  seed?: number;
  guidance_scale: number;
  num_inference_steps: number;
}

interface VideoGenerationResponse {
  task_id: string;
  status: string;
  message: string;
}

interface ImageUploadResponse {
  file_id: string;
  filename: string;
  size: number;
  status: string;
}

interface TaskStatusResponse {
  task_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress?: number;
  result_url?: string;
  error_message?: string;
}

interface ApiKeyResponse {
  api_key: string;
  key_id: string;
  name: string;
  expires_at?: string;
  created_at: string;
}

interface ApiKeyListResponse {
  keys: Array<{
    key_id: string;
    name: string;
    created_at: string;
    last_used?: string;
    expires_at?: string;
  }>;
}
```

## Phase 6: Testing & Deployment (Week 7-8)

### 6.1 Security Tests

**File: `backend/tests/test_security.py`**
```python
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.auth_service import AuthService
import json

client = TestClient(app)
auth_service = AuthService("test-secret-key")

class TestAuthentication:
    def test_register_user(self):
        response = client.post("/api/v1/auth/register", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "TestPass123"
        })
        assert response.status_code == 200
        assert "user_id" in response.json()

    def test_register_weak_password(self):
        response = client.post("/api/v1/auth/register", json={
            "username": "testuser2",
            "email": "test2@example.com",
            "password": "weak"
        })
        assert response.status_code == 400
        assert "Password must be at least 8 characters" in response.json()["detail"]

    def test_login_success(self):
        # First register a user
        client.post("/api/v1/auth/register", json={
            "username": "logintest",
            "email": "login@example.com",
            "password": "LoginTest123"
        })
        
        # Then login
        response = client.post("/api/v1/auth/login", data={
            "username": "logintest",
            "password": "LoginTest123"
        })
        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_login_invalid_credentials(self):
        response = client.post("/api/v1/auth/login", data={
            "username": "nonexistent",
            "password": "wrongpassword"
        })
        assert response.status_code == 401

class TestRateLimiting:
    def test_rate_limit_video_generation(self):
        # Login and get token
        client.post("/api/v1/auth/register", json={
            "username": "ratetest",
            "email": "rate@example.com",
            "password": "RateTest123"
        })
        
        login_response = client.post("/api/v1/auth/login", data={
            "username": "ratetest",
            "password": "RateTest123"
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Make multiple requests to trigger rate limit
        request_data = {
            "mode": "t2v",
            "prompt": "A simple test video",
            "width": 1024,
            "height": 576,
            "duration": 5.0,
            "fps": 24
        }
        
        # Should work for first few requests
        for _ in range(3):
            response = client.post(
                "/api/v1/video/generate", 
                json=request_data,
                headers=headers
            )
            assert response.status_code in [200, 202]
        
        # Eventually should hit rate limit (depending on test configuration)

class TestInputValidation:
    def test_malicious_prompt_blocked(self):
        # Register and login
        client.post("/api/v1/auth/register", json={
            "username": "validationtest",
            "email": "validation@example.com",
            "password": "ValidationTest123"
        })
        
        login_response = client.post("/api/v1/auth/login", data={
            "username": "validationtest",
            "password": "ValidationTest123"
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test blocked content
        response = client.post("/api/v1/video/generate", json={
            "mode": "t2v",
            "prompt": "nude person doing inappropriate things",
            "width": 1024,
            "height": 576,
            "duration": 5.0,
            "fps": 24
        }, headers=headers)
        
        assert response.status_code == 400
        assert "Content policy violation" in response.json()["detail"]["message"]

    def test_invalid_file_upload_blocked(self):
        # Register and login
        client.post("/api/v1/auth/register", json={
            "username": "filetest",
            "email": "file@example.com", 
            "password": "FileTest123"
        })
        
        login_response = client.post("/api/v1/auth/login", data={
            "username": "filetest",
            "password": "FileTest123"
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test malicious file upload
        malicious_content = b"<script>alert('xss')</script>"
        
        response = client.post(
            "/api/v1/video/upload-image",
            files={"file": ("test.jpg", malicious_content, "image/jpeg")},
            headers=headers
        )
        
        assert response.status_code == 400
        assert "File validation failed" in response.json()["detail"]["message"]

class TestApiKeys:
    def test_create_api_key(self):
        # Register and login
        client.post("/api/v1/auth/register", json={
            "username": "apitest",
            "email": "api@example.com",
            "password": "ApiTest123"
        })
        
        login_response = client.post("/api/v1/auth/login", data={
            "username": "apitest", 
            "password": "ApiTest123"
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        response = client.post("/api/v1/auth/api-keys", json={
            "name": "test-key",
            "expires_in_days": 30
        }, headers=headers)
        
        assert response.status_code == 200
        assert "api_key" in response.json()
        assert response.json()["api_key"].startswith("wan_")

    def test_api_key_authentication(self):
        # Create API key first
        client.post("/api/v1/auth/register", json={
            "username": "apikeytest",
            "email": "apikey@example.com",
            "password": "ApiKeyTest123"
        })
        
        login_response = client.post("/api/v1/auth/login", data={
            "username": "apikeytest",
            "password": "ApiKeyTest123"
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        api_key_response = client.post("/api/v1/auth/api-keys", json={
            "name": "auth-test-key"
        }, headers=headers)
        
        api_key = api_key_response.json()["api_key"]
        
        # Use API key for authentication
        api_headers = {"Authorization": f"Bearer {api_key}"}
        
        response = client.post("/api/v1/video/generate", json={
            "mode": "t2v",
            "prompt": "A beautiful landscape",
            "width": 1024,
            "height": 576,
            "duration": 5.0,
            "fps": 24
        }, headers=api_headers)
        
        assert response.status_code in [200, 202]  # Should work with API key
```

### 6.2 Environment Configuration

**File: `.env.example`**
```bash
# Security Configuration
SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/wan_db

# Rate Limiting
DEFAULT_RATE_LIMIT_PER_HOUR=100
VIDEO_GENERATION_LIMIT_PER_HOUR=5
VIDEO_GENERATION_LIMIT_PER_DAY=20
UPLOAD_LIMIT_PER_HOUR=20

# File Upload Settings
MAX_FILE_SIZE_MB=10
UPLOAD_DIRECTORY=./uploads

# Content Security
ENABLE_CONTENT_FILTER=true
STRICT_CONTENT_VALIDATION=true

# CORS Settings
CORS_ORIGINS=["http://localhost:3000","https://yourdomain.com"]
ALLOWED_HOSTS=["localhost","127.0.0.1","yourdomain.com"]

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log

# Redis (for rate limiting cache)
REDIS_URL=redis://localhost:6379/0

# Model Configuration
MODEL_PATH=./models
CUDA_VISIBLE_DEVICES=0

# Production Settings
DEBUG=false
ENVIRONMENT=production
```

### 6.3 Docker Configuration with Security

**File: `docker-compose.yml`**
```yaml
version: '3.8'

services:
  backend:
    build: 
      context: .
      dockerfile: backend/Dockerfile
    environment:
      - SECRET_KEY=${SECRET_KEY}
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=redis://redis:6379/0
      - ENVIRONMENT=production
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    volumes:
      - ./models:/app/models:ro
      - ./uploads:/app/uploads
      - ./logs:/app/logs
    restart: unless-stopped
    security_opt:
      - no-new-privileges:true
    user: "1000:1000"
    
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_BASE_URL=http://localhost:8000
    restart: unless-stopped
    security_opt:
      - no-new-privileges:true

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=wan_db
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    security_opt:
      - no-new-privileges:true

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped
    security_opt:
      - no-new-privileges:true

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl/certs:ro
    depends_on:
      - backend
      - frontend
    restart: unless-stopped
    security_opt:
      - no-new-privileges:true

volumes:
  postgres_data:
  redis_data:
```

**File: `backend/Dockerfile`**
```dockerfile
FROM python:3.9-slim

# Security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY core/ ./core/
COPY infrastructure/ ./infrastructure/
COPY utils_new/ ./utils_new/

# Set ownership
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.4 Nginx Security Configuration

**File: `nginx.conf`**
```nginx
events {
    worker_connections 1024;
}

http {
    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    
    # Hide Nginx version
    server_tokens off;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/m;
    limit_req_zone $binary_remote_addr zone=video:10m rate=2r/m;
    
    upstream backend {
        server backend:8000;
    }
    
    upstream frontend {
        server frontend:3000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name localhost;
        
        # SSL Configuration
        ssl_certificate /etc/ssl/certs/server.crt;
        ssl_certificate_key /etc/ssl/certs/server.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        
        # Security settings
        client_max_body_size 10M;
        client_body_timeout 60s;
        client_header_timeout 60s;
        
        # API endpoints
        location /api/ {
            limit_req zone=api burst=5 nodelay;
            
            # Video generation endpoints have stricter limits
            location /api/v1/video/generate {
                limit_req zone=video burst=1 nodelay;
                proxy_pass http://backend;
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header X-Forwarded-Proto $scheme;
                
                # Longer timeout for video generation
                proxy_read_timeout 300s;
                proxy_send_timeout 300s;
            }
            
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Frontend
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Block common attack vectors
        location ~* \.(php|asp|aspx|jsp)$ {
            deny all;
        }
        
        location ~* /\. {
            deny all;
        }
    }
}
```

### 6.5 Monitoring & Logging

**File: `backend/services/security_monitor.py`**
```python
import logging
from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import asyncio

class SecurityMonitor:
    def __init__(self):
        self.failed_attempts = defaultdict(deque)
        self.suspicious_patterns = defaultdict(int)
        self.blocked_ips = set()
        self.logger = logging.getLogger("security")
        
        # Configure security logger
        handler = logging.FileHandler("logs/security.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.WARNING)

    def log_failed_auth(self, ip_address: str, username: str, reason: str):
        """Log failed authentication attempt"""
        now = datetime.utcnow()
        
        # Track failed attempts per IP
        self.failed_attempts[ip_address].append(now)
        
        # Remove old attempts (older than 1 hour)
        hour_ago = now - timedelta(hours=1)
        while (self.failed_attempts[ip_address] and 
               self.failed_attempts[ip_address][0] < hour_ago):
            self.failed_attempts[ip_address].popleft()
        
        # Check for brute force
        if len(self.failed_attempts[ip_address]) >= 5:
            self._handle_brute_force(ip_address)
        
        # Log the attempt
        self.logger.warning(
            f"Failed authentication - IP: {ip_address}, User: {username}, "
            f"Reason: {reason}, Total attempts: {len(self.failed_attempts[ip_address])}"
        )

    def log_suspicious_request(self, ip_address: str, endpoint: str, reason: str, payload: dict = None):
        """Log suspicious request"""
        self.suspicious_patterns[ip_address] += 1
        
        self.logger.warning(
            f"Suspicious request - IP: {ip_address}, Endpoint: {endpoint}, "
            f"Reason: {reason}, Payload: {json.dumps(payload) if payload else 'None'}"
        )
        
        # Block IP if too many suspicious requests
        if self.suspicious_patterns[ip_address] >= 10:
            self._block_ip(ip_address)

    def log_content_policy_violation(self, user_id: str, prompt: str, violations: List[str]):
        """Log content policy violation"""
        self.logger.error(
            f"Content policy violation - User: {user_id}, "
            f"Prompt: {prompt[:100]}..., Violations: {violations}"
        )

    def log_rate_limit_exceeded(self, user_id: str, endpoint: str, ip_address: str):
        """Log rate limit exceeded"""
        self.logger.info(
            f"Rate limit exceeded - User: {user_id}, Endpoint: {endpoint}, IP: {ip_address}"
        )

    def _handle_brute_force(self, ip_address: str):
        """Handle potential brute force attack"""
        self._block_ip(ip_address)
        
        self.logger.critical(
            f"BRUTE FORCE DETECTED - Blocking IP: {ip_address}"
        )
        
        # Could integrate with fail2ban or similar here

    def _block_ip(self, ip_address: str):
        """Block IP address"""
        self.blocked_ips.add(ip_address)
        
        self.logger.critical(
            f"IP BLOCKED - {ip_address} due to suspicious activity"
        )

    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked"""
        return ip_address in self.blocked_ips

    def get_security_stats(self) -> Dict:
        """Get security statistics"""
        return {
            "blocked_ips_count": len(self.blocked_ips),
            "suspicious_ips": len(self.suspicious_patterns),
            "total_failed_attempts": sum(
                len(attempts) for attempts in self.failed_attempts.values()
            )
        }

# Global security monitor instance
security_monitor = SecurityMonitor()
```

## Implementation Timeline & Next Steps

### Week 1-2: Foundation
- [ ] Set up database schema for authentication
- [ ] Implement JWT authentication service
- [ ] Create user registration/login endpoints
- [ ] Add basic middleware for auth

### Week 3: Rate Limiting
- [ ] Implement rate limiting service with Redis
- [ ] Add rate limiting middleware
- [ ] Configure endpoint-specific limits
- [ ] Add rate limit headers

### Week 4: Input Validation
- [ ] Create validation schemas with Pydantic
- [ ] Implement content policy filters
- [ ] Add file upload validation
- [ ] Test malicious input blocking

### Week 5: API Integration
- [ ] Update existing endpoints with security
- [ ] Add API key management
- [ ] Implement proper error handling
- [ ] Add security headers

### Week 6: Frontend Security
- [ ] Create authentication context
- [ ] Implement secure API client
- [ ] Add token refresh logic
- [ ] Handle rate limiting on frontend

### Week 7-8: Testing & Deployment
- [ ] Write comprehensive security tests
- [ ] Set up monitoring and logging
- [ ] Configure production deployment
- [ ] Document security features

### Production Checklist

**Before Deployment:**
- [ ] Change all default passwords and keys
- [ ] Configure SSL certificates
- [ ] Set up database backups
- [ ] Configure log rotation
- [ ] Test disaster recovery
- [ ] Security audit and penetration testing
- [ ] Set up monitoring alerts
- [ ] Document incident response procedures

**Post-Deployment:**
- [ ] Monitor security logs daily
- [ ] Regular security updates
- [ ] Periodic access reviews
- [ ] Performance monitoring
- [ ] User feedback collection

This implementation provides enterprise-grade security while maintaining the existing functionality and user experience of the WAN video generation system.