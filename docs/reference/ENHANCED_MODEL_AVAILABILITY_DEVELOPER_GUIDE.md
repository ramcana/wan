---
category: reference
last_updated: '2025-09-15T22:49:59.921342'
original_path: docs\ENHANCED_MODEL_AVAILABILITY_DEVELOPER_GUIDE.md
tags:
- configuration
- api
- troubleshooting
- installation
- performance
title: Enhanced Model Availability - Developer Guide
---

# Enhanced Model Availability - Developer Guide

## Overview

This guide provides comprehensive information for developers working with or extending the Enhanced Model Availability system. It covers architecture, APIs, extension points, development setup, and best practices.

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────┐
│ Enhanced Model Availability System                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │ API Layer       │  │ WebSocket       │              │
│  │ - FastAPI       │  │ - Real-time     │              │
│  │ - REST Endpoints│  │ - Notifications │              │
│  └─────────────────┘  └─────────────────┘              │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Core Services Layer                                 │ │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │ │
│  │ │Availability │ │ Enhanced    │ │ Health      │   │ │
│  │ │ Manager     │ │ Downloader  │ │ Monitor     │   │ │
│  │ └─────────────┘ └─────────────┘ └─────────────┘   │ │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │ │
│  │ │ Fallback    │ │ Usage       │ │ Error       │   │ │
│  │ │ Manager     │ │ Analytics   │ │ Recovery    │   │ │
│  │ └─────────────┘ └─────────────┘ └─────────────┘   │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Integration Layer                                   │ │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │ │
│  │ │ Model       │ │ Model       │ │ Fallback    │   │ │
│  │ │ Manager     │ │ Downloader  │ │ Recovery    │   │ │
│  │ └─────────────┘ └─────────────┘ └─────────────┘   │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Storage & Data Layer                                │ │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │ │
│  │ │ File System │ │ Database    │ │ Cache       │   │ │
│  │ │ Storage     │ │ (Optional)  │ │ Layer       │   │ │
│  │ └─────────────┘ └─────────────┘ └─────────────┘   │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Core Interfaces

#### Base Interfaces

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class ModelAvailabilityStatus(Enum):
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    MISSING = "missing"
    CORRUPTED = "corrupted"
    UPDATING = "updating"

@dataclass
class ModelStatus:
    model_id: str
    is_available: bool
    availability_status: ModelAvailabilityStatus
    size_mb: float
    integrity_score: float
    performance_score: float
    last_health_check: datetime
    # ... additional fields

class IModelAvailabilityProvider(ABC):
    """Interface for model availability providers"""

    @abstractmethod
    async def get_model_status(self, model_id: str) -> ModelStatus:
        """Get status for a specific model"""
        pass

    @abstractmethod
    async def get_all_models_status(self) -> Dict[str, ModelStatus]:
        """Get status for all models"""
        pass

    @abstractmethod
    async def ensure_model_available(self, model_id: str) -> bool:
        """Ensure a model is available for use"""
        pass

class IModelDownloader(ABC):
    """Interface for model downloaders"""

    @abstractmethod
    async def download_model(self, model_id: str, **kwargs) -> DownloadResult:
        """Download a model"""
        pass

    @abstractmethod
    async def get_download_progress(self, model_id: str) -> DownloadProgress:
        """Get download progress"""
        pass

    @abstractmethod
    async def cancel_download(self, model_id: str) -> bool:
        """Cancel an active download"""
        pass

class IModelHealthMonitor(ABC):
    """Interface for model health monitoring"""

    @abstractmethod
    async def check_model_health(self, model_id: str) -> HealthResult:
        """Check health of a specific model"""
        pass

    @abstractmethod
    async def get_system_health(self) -> SystemHealthResult:
        """Get overall system health"""
        pass

    @abstractmethod
    async def repair_model(self, model_id: str) -> RepairResult:
        """Attempt to repair a model"""
        pass
```

## Development Setup

### Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/enhanced-model-availability.git
cd enhanced-model-availability

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .

# Set up pre-commit hooks
pre-commit install

# Run initial tests
pytest tests/
```

### Development Dependencies

```txt
# requirements-dev.txt
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
black>=22.0.0
isort>=5.10.0
flake8>=4.0.0
mypy>=0.950
pre-commit>=2.17.0
httpx>=0.23.0
pytest-mock>=3.7.0
factory-boy>=3.2.0
faker>=15.0.0
```

### Configuration for Development

```json
{
  "development": {
    "debug": true,
    "log_level": "DEBUG",
    "auto_reload": true,
    "test_mode": true
  },
  "storage": {
    "models_directory": "./dev_models",
    "cache_directory": "./dev_cache",
    "max_storage_gb": 10
  },
  "downloads": {
    "max_concurrent_downloads": 2,
    "use_mock_downloads": true,
    "mock_download_delay_seconds": 5
  },
  "features": {
    "enhanced_downloads": true,
    "health_monitoring": true,
    "intelligent_fallback": true,
    "usage_analytics": false
  }
}
```

## Core Components Deep Dive

### ModelAvailabilityManager

The central coordinator for all model availability operations.

```python
class ModelAvailabilityManager:
    """Central coordinator for model availability"""

    def __init__(
        self,
        model_manager: IModelManager,
        downloader: IModelDownloader,
        health_monitor: IModelHealthMonitor,
        analytics: IModelUsageAnalytics,
        config: Config
    ):
        self.model_manager = model_manager
        self.downloader = downloader
        self.health_monitor = health_monitor
        self.analytics = analytics
        self.config = config
        self._status_cache = {}
        self._cache_ttl = 300  # 5 minutes

    async def get_comprehensive_status(self) -> Dict[str, DetailedModelStatus]:
        """Get comprehensive status for all models"""
        # Check cache first
        if self._is_cache_valid():
            return self._status_cache

        # Gather status from all sources
        basic_status = await self.model_manager.get_all_models_status()
        download_status = await self.downloader.get_all_download_status()
        health_status = await self.health_monitor.get_all_health_status()
        usage_stats = await self.analytics.get_all_usage_stats()

        # Combine into comprehensive status
        comprehensive_status = {}
        for model_id in basic_status.keys():
            comprehensive_status[model_id] = DetailedModelStatus(
                **basic_status[model_id].__dict__,
                download_progress=download_status.get(model_id),
                health_info=health_status.get(model_id),
                usage_stats=usage_stats.get(model_id)
            )

        # Update cache
        self._status_cache = comprehensive_status
        self._cache_timestamp = time.time()

        return comprehensive_status

    async def handle_model_request(self, model_id: str) -> ModelRequestResult:
        """Handle a request for a specific model"""
        status = await self.get_model_status(model_id)

        if status.is_available:
            return ModelRequestResult(
                success=True,
                model_id=model_id,
                action_taken="model_ready"
            )

        # Model not available, determine best action
        if status.availability_status == ModelAvailabilityStatus.DOWNLOADING:
            # Already downloading, provide wait estimate
            progress = await self.downloader.get_download_progress(model_id)
            return ModelRequestResult(
                success=False,
                model_id=model_id,
                action_taken="download_in_progress",
                estimated_wait_time=progress.eta_seconds
            )

        elif status.availability_status == ModelAvailabilityStatus.MISSING:
            # Start download
            download_result = await self.downloader.download_model(model_id)
            return ModelRequestResult(
                success=download_result.success,
                model_id=model_id,
                action_taken="download_started",
                download_id=download_result.download_id
            )

        elif status.availability_status == ModelAvailabilityStatus.CORRUPTED:
            # Attempt repair
            repair_result = await self.health_monitor.repair_model(model_id)
            return ModelRequestResult(
                success=repair_result.success,
                model_id=model_id,
                action_taken="repair_attempted",
                repair_id=repair_result.repair_id
            )

        else:
            # Unknown status, return error
            return ModelRequestResult(
                success=False,
                model_id=model_id,
                action_taken="error",
                error_message=f"Unknown model status: {status.availability_status}"
            )
```

### EnhancedModelDownloader

Enhanced downloader with retry logic and progress tracking.

```python
class EnhancedModelDownloader(IModelDownloader):
    """Enhanced model downloader with retry logic"""

    def __init__(
        self,
        base_downloader: IModelDownloader,
        config: DownloadConfig,
        progress_tracker: IProgressTracker
    ):
        self.base_downloader = base_downloader
        self.config = config
        self.progress_tracker = progress_tracker
        self.active_downloads = {}
        self.download_queue = asyncio.Queue()

    async def download_model(
        self,
        model_id: str,
        max_retries: Optional[int] = None,
        priority: str = "normal",
        **kwargs
    ) -> DownloadResult:
        """Download model with enhanced retry logic"""
        max_retries = max_retries or self.config.max_retries

        download_context = DownloadContext(
            model_id=model_id,
            max_retries=max_retries,
            priority=priority,
            start_time=time.time(),
            **kwargs
        )

        # Add to active downloads
        self.active_downloads[model_id] = download_context

        try:
            result = await self._download_with_retry(download_context)
            return result
        finally:
            # Remove from active downloads
            self.active_downloads.pop(model_id, None)

    async def _download_with_retry(self, context: DownloadContext) -> DownloadResult:
        """Execute download with retry logic"""
        last_exception = None

        for attempt in range(context.max_retries + 1):
            try:
                # Update progress
                await self.progress_tracker.update_progress(
                    context.model_id,
                    DownloadProgress(
                        model_id=context.model_id,
                        status=DownloadStatus.DOWNLOADING,
                        retry_count=attempt,
                        max_retries=context.max_retries
                    )
                )

                # Attempt download
                result = await self.base_downloader.download_model(
                    context.model_id,
                    **context.kwargs
                )

                if result.success:
                    # Verify integrity
                    if await self._verify_download_integrity(context.model_id):
                        return DownloadResult(
                            success=True,
                            model_id=context.model_id,
                            total_retries=attempt,
                            total_time_seconds=time.time() - context.start_time
                        )
                    else:
                        # Integrity check failed, treat as failure
                        raise DownloadIntegrityError("Downloaded model failed integrity check")

            except Exception as e:
                last_exception = e

                if attempt < context.max_retries:
                    # Calculate backoff delay
                    delay = self._calculate_backoff_delay(attempt)
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Final attempt failed
                    break

        # All attempts failed
        return DownloadResult(
            success=False,
            model_id=context.model_id,
            total_retries=context.max_retries,
            total_time_seconds=time.time() - context.start_time,
            error_message=str(last_exception)
        )

    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay"""
        base_delay = self.config.retry_delay_seconds
        max_delay = self.config.max_retry_delay_seconds

        # Exponential backoff with jitter
        delay = min(base_delay * (2 ** attempt), max_delay)
        jitter = random.uniform(0.1, 0.3) * delay

        return delay + jitter
```

### ModelHealthMonitor

Health monitoring with corruption detection and repair.

```python
class ModelHealthMonitor(IModelHealthMonitor):
    """Model health monitoring and repair system"""

    def __init__(
        self,
        storage_manager: IStorageManager,
        integrity_checker: IIntegrityChecker,
        performance_monitor: IPerformanceMonitor,
        config: HealthConfig
    ):
        self.storage_manager = storage_manager
        self.integrity_checker = integrity_checker
        self.performance_monitor = performance_monitor
        self.config = config
        self.health_cache = {}
        self.scheduled_checks = {}

    async def check_model_health(self, model_id: str, force: bool = False) -> HealthResult:
        """Comprehensive health check for a model"""
        # Check cache first (unless forced)
        if not force and model_id in self.health_cache:
            cached_result = self.health_cache[model_id]
            if time.time() - cached_result.timestamp < self.config.cache_ttl:
                return cached_result

        health_result = HealthResult(
            model_id=model_id,
            timestamp=time.time(),
            checks_performed=[]
        )

        # File integrity check
        integrity_result = await self.integrity_checker.check_integrity(model_id)
        health_result.checks_performed.append("integrity")
        health_result.integrity_score = integrity_result.score
        health_result.integrity_issues = integrity_result.issues

        # Performance check (if model is loaded)
        if await self._is_model_loaded(model_id):
            perf_result = await self.performance_monitor.check_performance(model_id)
            health_result.checks_performed.append("performance")
            health_result.performance_score = perf_result.score
            health_result.performance_issues = perf_result.issues

        # Storage check
        storage_result = await self.storage_manager.check_storage_health(model_id)
        health_result.checks_performed.append("storage")
        health_result.storage_issues = storage_result.issues

        # Calculate overall health score
        health_result.overall_score = self._calculate_overall_score(health_result)
        health_result.is_healthy = health_result.overall_score >= self.config.healthy_threshold

        # Generate recommendations
        health_result.recommendations = self._generate_recommendations(health_result)

        # Cache result
        self.health_cache[model_id] = health_result

        # Trigger auto-repair if needed
        if self.config.auto_repair_enabled and not health_result.is_healthy:
            await self._trigger_auto_repair(model_id, health_result)

        return health_result

    async def repair_model(self, model_id: str, repair_type: str = "auto") -> RepairResult:
        """Attempt to repair a model"""
        repair_result = RepairResult(
            model_id=model_id,
            repair_type=repair_type,
            start_time=time.time(),
            actions_taken=[]
        )

        try:
            # Get current health status
            health = await self.check_model_health(model_id, force=True)

            if repair_type == "auto":
                # Determine best repair strategy
                repair_actions = self._determine_repair_actions(health)
            else:
                # Use specified repair type
                repair_actions = [repair_type]

            # Execute repair actions
            for action in repair_actions:
                action_result = await self._execute_repair_action(model_id, action)
                repair_result.actions_taken.append({
                    "action": action,
                    "success": action_result.success,
                    "details": action_result.details
                })

                if not action_result.success:
                    repair_result.success = False
                    repair_result.error_message = action_result.error_message
                    break

            # Verify repair success
            if repair_result.success:
                post_repair_health = await self.check_model_health(model_id, force=True)
                repair_result.success = post_repair_health.is_healthy
                repair_result.health_improvement = (
                    post_repair_health.overall_score - health.overall_score
                )

        except Exception as e:
            repair_result.success = False
            repair_result.error_message = str(e)

        repair_result.total_time_seconds = time.time() - repair_result.start_time
        return repair_result
```

## Extension Points

### Custom Model Providers

Implement custom model providers for different sources:

```python
class CustomModelProvider(IModelAvailabilityProvider):
    """Custom model provider implementation"""

    def __init__(self, provider_config: Dict[str, Any]):
        self.config = provider_config
        self.client = self._create_client()

    async def get_model_status(self, model_id: str) -> ModelStatus:
        """Get model status from custom source"""
        # Implement custom logic
        response = await self.client.get_model_info(model_id)

        return ModelStatus(
            model_id=model_id,
            is_available=response.get('available', False),
            availability_status=self._map_status(response.get('status')),
            size_mb=response.get('size_mb', 0),
            # ... map other fields
        )

    def _map_status(self, external_status: str) -> ModelAvailabilityStatus:
        """Map external status to internal enum"""
        mapping = {
            'ready': ModelAvailabilityStatus.AVAILABLE,
            'downloading': ModelAvailabilityStatus.DOWNLOADING,
            'missing': ModelAvailabilityStatus.MISSING,
            'corrupted': ModelAvailabilityStatus.CORRUPTED
        }
        return mapping.get(external_status, ModelAvailabilityStatus.MISSING)

# Register custom provider
def register_custom_provider():
    from enhanced_model_availability.registry import provider_registry

    provider_registry.register(
        'custom_provider',
        CustomModelProvider,
        config_schema={
            'api_endpoint': str,
            'api_key': str,
            'timeout_seconds': int
        }
    )
```

### Custom Health Checkers

Implement custom health checking logic:

```python
class CustomHealthChecker(IHealthChecker):
    """Custom health checker implementation"""

    def __init__(self, checker_config: Dict[str, Any]):
        self.config = checker_config

    async def check_health(self, model_id: str, model_path: str) -> HealthCheckResult:
        """Perform custom health check"""
        issues = []
        score = 1.0

        # Custom health check logic
        if await self._check_custom_condition(model_path):
            issues.append("Custom condition failed")
            score -= 0.3

        # Additional custom checks...

        return HealthCheckResult(
            checker_name="custom_checker",
            score=score,
            issues=issues,
            details={"custom_metric": await self._get_custom_metric(model_path)}
        )

    async def _check_custom_condition(self, model_path: str) -> bool:
        """Implement custom condition check"""
        # Custom logic here
        return False

# Register custom health checker
def register_custom_health_checker():
    from enhanced_model_availability.health import health_checker_registry

    health_checker_registry.register(
        'custom_checker',
        CustomHealthChecker,
        priority=10  # Higher priority runs first
    )
```

### Custom Fallback Strategies

Implement custom fallback strategies:

```python
class CustomFallbackStrategy(IFallbackStrategy):
    """Custom fallback strategy implementation"""

    def __init__(self, strategy_config: Dict[str, Any]):
        self.config = strategy_config

    async def suggest_fallback(
        self,
        requested_model: str,
        requirements: GenerationRequirements,
        available_models: List[str]
    ) -> FallbackSuggestion:
        """Suggest custom fallback strategy"""

        # Custom fallback logic
        if self._should_use_custom_strategy(requested_model, requirements):
            alternative = await self._find_custom_alternative(
                requested_model,
                available_models
            )

            return FallbackSuggestion(
                strategy_type=FallbackType.ALTERNATIVE_MODEL,
                suggested_model=alternative,
                compatibility_score=await self._calculate_compatibility(
                    requested_model,
                    alternative
                ),
                reason="Custom strategy recommendation"
            )

        # Fall back to default strategy
        return await self._default_fallback(requested_model, requirements, available_models)

# Register custom fallback strategy
def register_custom_fallback_strategy():
    from enhanced_model_availability.fallback import fallback_strategy_registry

    fallback_strategy_registry.register(
        'custom_strategy',
        CustomFallbackStrategy,
        conditions=['model_type:custom', 'priority:high']
    )
```

## API Development

### Adding New Endpoints

```python
from fastapi import APIRouter, Depends, HTTPException
from enhanced_model_availability.dependencies import get_availability_manager

router = APIRouter(prefix="/api/v1/custom", tags=["custom"])

@router.get("/models/{model_id}/custom-status")
async def get_custom_model_status(
    model_id: str,
    availability_manager: ModelAvailabilityManager = Depends(get_availability_manager)
):
    """Get custom model status information"""
    try:
        # Custom logic here
        custom_status = await availability_manager.get_custom_status(model_id)
        return {"model_id": model_id, "custom_status": custom_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/{model_id}/custom-action")
async def perform_custom_action(
    model_id: str,
    action_data: CustomActionRequest,
    availability_manager: ModelAvailabilityManager = Depends(get_availability_manager)
):
    """Perform custom action on model"""
    try:
        result = await availability_manager.perform_custom_action(
            model_id,
            action_data.action_type,
            action_data.parameters
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Register router
def register_custom_router(app):
    app.include_router(router)
```

### Custom Middleware

```python
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time

class PerformanceMiddleware(BaseHTTPMiddleware):
    """Custom middleware for performance monitoring"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Add custom headers
        request.state.start_time = start_time
        request.state.request_id = generate_request_id()

        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request.state.request_id

        # Log performance metrics
        await self._log_performance_metrics(request, response, process_time)

        return response

    async def _log_performance_metrics(self, request, response, process_time):
        """Log performance metrics"""
        # Custom logging logic
        pass

# Register middleware
def register_custom_middleware(app):
    app.add_middleware(PerformanceMiddleware)
```

## Testing Framework

### Unit Testing

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from enhanced_model_availability.core import ModelAvailabilityManager

@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for testing"""
    return {
        'model_manager': AsyncMock(),
        'downloader': AsyncMock(),
        'health_monitor': AsyncMock(),
        'analytics': AsyncMock(),
        'config': MagicMock()
    }

@pytest.fixture
def availability_manager(mock_dependencies):
    """Create ModelAvailabilityManager with mocked dependencies"""
    return ModelAvailabilityManager(**mock_dependencies)

@pytest.mark.asyncio
async def test_get_model_status_available(availability_manager, mock_dependencies):
    """Test getting status for available model"""
    # Setup mocks
    mock_dependencies['model_manager'].get_model_status.return_value = ModelStatus(
        model_id="test-model",
        is_available=True,
        availability_status=ModelAvailabilityStatus.AVAILABLE
    )

    # Execute
    result = await availability_manager.get_model_status("test-model")

    # Assert
    assert result.is_available
    assert result.availability_status == ModelAvailabilityStatus.AVAILABLE
    mock_dependencies['model_manager'].get_model_status.assert_called_once_with("test-model")

@pytest.mark.asyncio
async def test_handle_model_request_missing(availability_manager, mock_dependencies):
    """Test handling request for missing model"""
    # Setup mocks
    mock_dependencies['model_manager'].get_model_status.return_value = ModelStatus(
        model_id="missing-model",
        is_available=False,
        availability_status=ModelAvailabilityStatus.MISSING
    )
    mock_dependencies['downloader'].download_model.return_value = DownloadResult(
        success=True,
        model_id="missing-model",
        download_id="dl_123"
    )

    # Execute
    result = await availability_manager.handle_model_request("missing-model")

    # Assert
    assert result.action_taken == "download_started"
    assert result.download_id == "dl_123"
    mock_dependencies['downloader'].download_model.assert_called_once_with("missing-model")
```

### Integration Testing

```python
import pytest
import httpx
from fastapi.testclient import TestClient
from enhanced_model_availability.app import create_app

@pytest.fixture
def test_app():
    """Create test application"""
    app = create_app(config_path="tests/test_config.json")
    return app

@pytest.fixture
def client(test_app):
    """Create test client"""
    return TestClient(test_app)

def test_get_model_status_endpoint(client):
    """Test model status endpoint"""
    response = client.get("/api/v1/models/status/detailed")
    assert response.status_code == 200

    data = response.json()
    assert "models" in data
    assert "summary" in data

def test_start_download_endpoint(client):
    """Test download start endpoint"""
    response = client.post(
        "/api/v1/models/download/manage",
        json={
            "model_id": "test-model",
            "action": "start",
            "max_retries": 3
        }
    )
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "download_id" in data

@pytest.mark.asyncio
async def test_websocket_notifications(test_app):
    """Test WebSocket notifications"""
    async with httpx.AsyncClient(app=test_app, base_url="http://test") as client:
        async with client.websocket_connect("/ws") as websocket:
            # Trigger an event that should send notification
            await client.post("/api/v1/models/download/manage", json={
                "model_id": "test-model",
                "action": "start"
            })

            # Wait for WebSocket message
            message = await websocket.receive_json()
            assert message["type"] == "model_download_progress"
            assert message["model_id"] == "test-model"
```

### Performance Testing

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

async def performance_test_concurrent_downloads():
    """Test concurrent download performance"""
    model_ids = [f"test-model-{i}" for i in range(10)]

    start_time = time.time()

    # Start concurrent downloads
    tasks = []
    for model_id in model_ids:
        task = asyncio.create_task(start_download(model_id))
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    end_time = time.time()
    total_time = end_time - start_time

    # Analyze results
    successful_downloads = sum(1 for r in results if isinstance(r, DownloadResult) and r.success)

    print(f"Concurrent downloads: {len(model_ids)}")
    print(f"Successful: {successful_downloads}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per download: {total_time/len(model_ids):.2f}s")

    assert successful_downloads >= len(model_ids) * 0.8  # 80% success rate

def performance_test_api_throughput():
    """Test API throughput"""
    import requests
    import threading

    results = []

    def make_request():
        start = time.time()
        response = requests.get("http://localhost:8000/api/v1/models/status/detailed")
        end = time.time()
        results.append({
            'status_code': response.status_code,
            'response_time': end - start
        })

    # Create threads for concurrent requests
    threads = []
    for _ in range(100):
        thread = threading.Thread(target=make_request)
        threads.append(thread)

    # Start all threads
    start_time = time.time()
    for thread in threads:
        thread.start()

    # Wait for completion
    for thread in threads:
        thread.join()

    end_time = time.time()

    # Analyze results
    successful_requests = sum(1 for r in results if r['status_code'] == 200)
    average_response_time = sum(r['response_time'] for r in results) / len(results)
    requests_per_second = len(results) / (end_time - start_time)

    print(f"Total requests: {len(results)}")
    print(f"Successful: {successful_requests}")
    print(f"Average response time: {average_response_time:.3f}s")
    print(f"Requests per second: {requests_per_second:.2f}")

    assert successful_requests >= len(results) * 0.95  # 95% success rate
    assert average_response_time < 0.1  # Under 100ms average
```

## Debugging and Profiling

### Logging Configuration

```python
import logging
import sys
from enhanced_model_availability.logging import setup_logging

# Configure logging for development
def setup_development_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('debug.log')
        ]
    )

    # Set specific loggers
    logging.getLogger('enhanced_model_availability.downloads').setLevel(logging.DEBUG)
    logging.getLogger('enhanced_model_availability.health').setLevel(logging.INFO)
    logging.getLogger('httpx').setLevel(logging.WARNING)

# Custom logger for component
logger = logging.getLogger(__name__)

async def debug_download_process(model_id: str):
    """Debug download process with detailed logging"""
    logger.debug(f"Starting download debug for model: {model_id}")

    try:
        # Log each step
        logger.debug("Checking model status...")
        status = await get_model_status(model_id)
        logger.debug(f"Model status: {status}")

        logger.debug("Starting download...")
        result = await download_model(model_id)
        logger.debug(f"Download result: {result}")

    except Exception as e:
        logger.exception(f"Error during download debug: {e}")
        raise
```

### Performance Profiling

```python
import cProfile
import pstats
import io
from functools import wraps

def profile_function(func):
    """Decorator to profile function performance"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            profiler.disable()

            # Print profiling results
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions

            print(f"Profiling results for {func.__name__}:")
            print(s.getvalue())

    return wrapper

# Usage
@profile_function
async def download_model_with_profiling(model_id: str):
    return await download_model(model_id)
```

### Memory Profiling

```python
import tracemalloc
from memory_profiler import profile

def start_memory_profiling():
    """Start memory profiling"""
    tracemalloc.start()

def get_memory_snapshot():
    """Get current memory snapshot"""
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("Top 10 memory allocations:")
    for stat in top_stats[:10]:
        print(stat)

@profile
def memory_intensive_function():
    """Function with memory profiling"""
    # Function implementation
    pass
```

## Best Practices

### Code Organization

```
enhanced_model_availability/
├── core/                   # Core business logic
│   ├── __init__.py
│   ├── availability_manager.py
│   ├── downloader.py
│   ├── health_monitor.py
│   └── analytics.py
├── api/                    # API endpoints
│   ├── __init__.py
│   ├── models.py
│   ├── downloads.py
│   └── health.py
├── services/               # Service layer
│   ├── __init__.py
│   ├── model_service.py
│   └── notification_service.py
├── repositories/           # Data access layer
│   ├── __init__.py
│   ├── model_repository.py
│   └── analytics_repository.py
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── config.py
│   └── helpers.py
├── tests/                  # Test files
│   ├── unit/
│   ├── integration/
│   └── performance/
└── examples/               # Example code
    ├── basic_usage.py
    └── advanced_usage.py
```

### Error Handling

```python
from enhanced_model_availability.exceptions import (
    ModelNotFoundError,
    DownloadFailedError,
    HealthCheckFailedError
)

class EnhancedModelAvailabilityError(Exception):
    """Base exception for enhanced model availability"""
    pass

class ModelNotFoundError(EnhancedModelAvailabilityError):
    """Raised when a model is not found"""
    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model not found: {model_id}")

class DownloadFailedError(EnhancedModelAvailabilityError):
    """Raised when model download fails"""
    def __init__(self, model_id: str, reason: str, retry_count: int = 0):
        self.model_id = model_id
        self.reason = reason
        self.retry_count = retry_count
        super().__init__(f"Download failed for {model_id}: {reason} (retries: {retry_count})")

# Usage in code
async def download_model_safe(model_id: str) -> DownloadResult:
    """Safely download model with proper error handling"""
    try:
        result = await download_model(model_id)
        return result
    except ModelNotFoundError:
        logger.error(f"Model {model_id} not found in registry")
        raise
    except DownloadFailedError as e:
        logger.error(f"Download failed: {e}")
        # Attempt fallback strategy
        return await attempt_fallback_download(model_id)
    except Exception as e:
        logger.exception(f"Unexpected error downloading {model_id}")
        raise DownloadFailedError(model_id, str(e))
```

### Configuration Management

```python
from pydantic import BaseSettings, Field
from typing import Optional, Dict, Any

class DownloadConfig(BaseSettings):
    """Download configuration"""
    max_concurrent_downloads: int = Field(3, ge=1, le=10)
    max_retries: int = Field(3, ge=0, le=10)
    retry_delay_seconds: int = Field(30, ge=1)
    bandwidth_limit_mbps: int = Field(0, ge=0)

    class Config:
        env_prefix = "ENHANCED_MODEL_DOWNLOAD_"

class HealthConfig(BaseSettings):
    """Health monitoring configuration"""
    enabled: bool = Field(True)
    check_interval_hours: int = Field(24, ge=1)
    auto_repair_enabled: bool = Field(True)
    healthy_threshold: float = Field(0.8, ge=0.0, le=1.0)

    class Config:
        env_prefix = "ENHANCED_MODEL_HEALTH_"

class EnhancedModelConfig(BaseSettings):
    """Main configuration"""
    download: DownloadConfig = DownloadConfig()
    health: HealthConfig = HealthConfig()
    debug: bool = Field(False)
    log_level: str = Field("INFO")

    class Config:
        env_file = ".env"
        case_sensitive = False

# Usage
config = EnhancedModelConfig()
```

This comprehensive developer guide provides all the information needed to work with, extend, and contribute to the Enhanced Model Availability system.
