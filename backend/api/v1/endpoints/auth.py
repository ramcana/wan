from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from backend.schemas.auth import (
    UserCreate, Token, APIKeyCreate, APIKeyResponse
)
from backend.services.auth_service import AuthService
from backend.services.input_validation_service import InputValidationService
from backend.models.auth import User, APIKey
from backend.database import get_db
from backend.middleware.auth_middleware import AuthMiddleware
from backend.core.security_config import security_settings
from datetime import datetime, timedelta
import re


router = APIRouter()
auth_service = AuthService(secret_key=security_settings.SECRET_KEY)
input_validator = InputValidationService()
auth_middleware = AuthMiddleware(auth_service=auth_service)


@router.post("/register", response_model=dict)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """Register new user with input validation"""
    
    # Validate email format
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
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
    
    user_is_active = False
    if user is not None:
        user_is_active = bool(user.is_active)
    
    password_valid = False
    if user is not None:
        password_valid = auth_service.verify_password(
            form_data.password, str(user.hashed_password)
        )
    
    if not user or not password_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    if not user_is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is inactive"
        )
    
    # Create tokens
    access_token = auth_service.create_access_token(
        data={"sub": str(user.id)}
    )
    refresh_token = auth_service.create_refresh_token(
        data={"sub": str(user.id)}
    )
    
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
    
    user_is_active = False
    if user is not None:
        user_is_active = bool(user.is_active)
    
    if not user or not user_is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new access token
    new_access_token = auth_service.create_access_token(
        data={"sub": str(user.id)}
    )
    
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
        APIKey.is_active.is_(True)
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
        expires_at = datetime.utcnow() + timedelta(
            days=key_data.expires_in_days
        )
    
    # Create API key record
    new_api_key = APIKey(
        key_hash=key_hash,
        user_id=str(current_user.id),
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
        APIKey.is_active.is_(True)
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
        APIKey.is_active.is_(True)
    ).first()
    
    if not api_key:
        raise HTTPException(
            status_code=404,
            detail="API key not found"
        )
    
    # Instead of directly assigning, we'll use SQLAlchemy's update method
    db.query(APIKey).filter(APIKey.id == key_id).update(
        {"is_active": False}
    )
    db.commit()
    
    return {"message": "API key revoked successfully"}