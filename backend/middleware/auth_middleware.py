from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from backend.services.auth_service import AuthService
from backend.models.auth import User, APIKey
from backend.repositories.database import get_db
from datetime import datetime
from typing import Optional


security = HTTPBearer(auto_error=False)


class AuthMiddleware:
    def __init__(self, auth_service: AuthService):
        self.auth_service = auth_service

    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: Session = Depends(get_db),
    ) -> User:
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header required",
            )

        token = credentials.credentials

        # Check if it's an API key
        if token.startswith("wan_"):
            return await self._verify_api_key(token, db)

        # Otherwise, treat as JWT token
        return await self._verify_jwt_token(token, db)

    async def _verify_api_key(self, api_key: str, db: Session) -> User:
        # Find API key in database
        api_key_records = db.query(APIKey).filter(APIKey.is_active.is_(True)).all()

        for record in api_key_records:
            if self.auth_service.verify_api_key(api_key, str(record.key_hash)):
                # Check if API key has expired
                has_expired = False
                if record.expires_at is not None:
                    if record.expires_at < datetime.utcnow():
                        has_expired = True

                if has_expired:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="API key expired",
                    )

                # Update last used
                record.last_used = datetime.utcnow()
                db.commit()

                user = db.query(User).filter(User.id == record.user_id).first()
                user_is_active = False
                if user is not None:
                    user_is_active = bool(user.is_active)

                if not user or not user_is_active:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User account inactive",
                    )
                return user

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )

    async def _verify_jwt_token(self, token: str, db: Session) -> User:
        payload = self.auth_service.verify_token(token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            )

        user_id: Optional[str] = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload"
            )

        user = db.query(User).filter(User.id == user_id).first()
        user_is_active = False
        if user is not None:
            user_is_active = bool(user.is_active)

        if not user or not user_is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive",
            )
        return user

    def require_admin(self, current_user: User = Depends(get_current_user)) -> User:
        is_admin = False
        if current_user is not None:
            is_admin = bool(current_user.is_admin)

        if not is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
            )
        return current_user
