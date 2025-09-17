from fastapi import HTTPException, Depends, Request, status
from backend.services.rate_limit_service import RateLimitService
from backend.models.auth import User
from backend.middleware.auth_middleware import AuthMiddleware
from backend.repositories.database import get_db
from sqlalchemy.orm import Session
from datetime import datetime


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
        db: Session = Depends(get_db),
    ):
        """Rate limiting dependency"""
        endpoint_pattern = self._get_endpoint_pattern(request.url.path)
        custom_limits = self.endpoint_limits.get(endpoint_pattern)

        result = await self.rate_limit_service.check_rate_limit(
            user_id=str(current_user.id),
            endpoint=endpoint_pattern,
            db=db,
            custom_limits=custom_limits,
        )
        is_allowed, limit_info = result

        if not is_allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(limit_info["hourly_limit"]),
                    "X-RateLimit-Remaining": str(limit_info["hourly_remaining"]),
                    "X-RateLimit-Reset": str(int(limit_info["reset_time"].timestamp())),
                    "Retry-After": str(
                        (limit_info["reset_time"] - datetime.utcnow()).seconds
                    ),
                },
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
