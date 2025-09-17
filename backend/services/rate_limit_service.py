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
            "upload": {"hourly": 50, "daily": 200},
        }

    async def check_rate_limit(
        self,
        user_id: str,
        endpoint: str,
        db: Session,
        custom_limits: Optional[Dict] = None,
    ) -> tuple[bool, Dict]:
        """
        Check if user has exceeded rate limits
        Returns: (is_allowed, limit_info)
        """
        now = datetime.utcnow()
        cache_key = f"{user_id}:{endpoint}"

        # Get limits (custom or default)
        limits = custom_limits or self._default_limits.get(
            endpoint, self._default_limits["api_general"]
        )

        # Check cache first for performance
        if cache_key in self._cache:
            cache_data = self._cache[cache_key]
            if self._is_cache_valid(cache_data, now):
                return self._evaluate_limits(cache_data, limits, now)

        # Fallback to database
        rate_record = (
            db.query(UserRateLimit)
            .filter(
                UserRateLimit.user_id == user_id, UserRateLimit.endpoint == endpoint
            )
            .first()
        )

        if not rate_record:
            rate_record = UserRateLimit(
                user_id=user_id,
                endpoint=endpoint,
                requests_count=0,
                window_start=now,
                daily_limit=limits["daily"],
                hourly_limit=limits["hourly"],
            )
            db.add(rate_record)

        # Reset counters if windows have expired
        self._reset_expired_windows(rate_record, now)

        # Check limits
        hourly_count = self._get_hourly_count(cache_key, now)
        daily_count = rate_record.requests_count

        is_allowed = hourly_count < limits["hourly"] and daily_count < limits["daily"]

        if is_allowed:
            # Increment counters
            rate_record.requests_count += 1
            self._increment_cache(cache_key, now)
            db.commit()

        limit_info = {
            "hourly_limit": limits["hourly"],
            "hourly_remaining": max(
                0, limits["hourly"] - hourly_count - (1 if is_allowed else 0)
            ),
            "daily_limit": limits["daily"],
            "daily_remaining": max(
                0, limits["daily"] - daily_count - (1 if is_allowed else 0)
            ),
            "reset_time": self._get_next_reset_time(now),
        }

        return is_allowed, limit_info

    def _is_cache_valid(self, cache_data: Dict, now: datetime) -> bool:
        return (now - cache_data.get("last_updated", datetime.min)).seconds < 60

    def _evaluate_limits(
        self, cache_data: Dict, limits: Dict, now: datetime
    ) -> tuple[bool, Dict]:
        hourly_count = len(
            [
                req
                for req in cache_data.get("requests", [])
                if (now - req).seconds < 3600
            ]
        )

        is_allowed = hourly_count < limits["hourly"]

        limit_info = {
            "hourly_limit": limits["hourly"],
            "hourly_remaining": max(0, limits["hourly"] - hourly_count),
            "daily_limit": limits["daily"],
            "daily_remaining": limits["daily"],  # Cache doesn't track daily
            "reset_time": self._get_next_reset_time(now),
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
            req for req in self._cache[cache_key]["requests"] if req > day_ago
        ]

    def _get_next_reset_time(self, now: datetime) -> datetime:
        # Next hour reset
        return (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
