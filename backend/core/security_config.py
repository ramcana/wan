from typing import List
import os


class SecuritySettings:
    """Security configuration settings."""

    def __init__(self):
        # JWT Settings
        self.SECRET_KEY: str = os.getenv(
            "SECRET_KEY", "your-super-secret-key-change-this"
        )
        self.ALGORITHM: str = "HS256"
        self.ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
        self.REFRESH_TOKEN_EXPIRE_DAYS: int = 7

        # Rate Limiting
        self.DEFAULT_RATE_LIMIT_PER_HOUR: int = 100
        self.VIDEO_GENERATION_LIMIT_PER_HOUR: int = 5
        self.VIDEO_GENERATION_LIMIT_PER_DAY: int = 20
        self.UPLOAD_LIMIT_PER_HOUR: int = 20

        # Content Validation
        self.MAX_PROMPT_LENGTH: int = 500
        self.MAX_NEGATIVE_PROMPT_LENGTH: int = 300
        self.MAX_FILE_SIZE_MB: int = 10
        self.ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]

        # Security Headers
        self.CORS_ORIGINS: List[str] = [
            "http://localhost:3000",
            "http://127.0.0.1:8080",
        ]
        self.CORS_CREDENTIALS: bool = True
        self.CORS_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE"]
        self.CORS_HEADERS: List[str] = ["*"]

        # Content Security
        self.ENABLE_CONTENT_FILTER: bool = True
        self.STRICT_CONTENT_VALIDATION: bool = True


security_settings = SecuritySettings()
