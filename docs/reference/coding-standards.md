---
category: reference
last_updated: '2025-09-15T22:50:00.829157'
original_path: tools\onboarding\docs\coding-standards.md
tags:
- configuration
- api
- troubleshooting
- installation
- security
- performance
title: Coding Standards and Best Practices
---

# Coding Standards and Best Practices

This document outlines the coding standards, conventions, and best practices for the WAN22 project. Following these guidelines ensures code consistency, maintainability, and team collaboration.

## 🐍 Python Standards

### Code Style

We follow **PEP 8** with some project-specific modifications:

```python
# Good: Clear, descriptive names
def generate_video_from_text(prompt: str, duration: int) -> VideoResult:
    """Generate a video from text prompt with specified duration."""
    pass

# Bad: Unclear, abbreviated names
def gen_vid(p: str, d: int) -> dict:
    pass
```

### Formatting Tools

- **Black**: Automatic code formatting
- **isort**: Import sorting
- **MyPy**: Type checking

```bash
# Format code
black backend/ core/ infrastructure/

# Sort imports
isort backend/ core/ infrastructure/

# Type checking
mypy backend/ core/ infrastructure/
```

### Naming Conventions

| Type      | Convention           | Example              |
| --------- | -------------------- | -------------------- |
| Variables | snake_case           | `video_duration`     |
| Functions | snake_case           | `generate_video()`   |
| Classes   | PascalCase           | `VideoGenerator`     |
| Constants | UPPER_SNAKE_CASE     | `MAX_VIDEO_LENGTH`   |
| Private   | \_leading_underscore | `_internal_method()` |
| Modules   | snake_case           | `video_generator.py` |

### Type Hints

Always use type hints for function parameters and return values:

```python
from typing import List, Optional, Dict, Any
from pathlib import Path

def process_video_queue(
    queue_items: List[VideoTask],
    output_dir: Path,
    max_concurrent: int = 3
) -> Dict[str, Any]:
    """Process video generation queue with specified concurrency."""
    results = {}
    # Implementation here
    return results

# Use Optional for nullable values
def get_user_preference(user_id: str) -> Optional[UserPreference]:
    """Get user preference, returns None if not found."""
    pass
```

### Documentation

Use **Google-style docstrings**:

```python
def generate_video(
    prompt: str,
    model_name: str,
    duration: int = 30,
    resolution: str = "1024x576"
) -> VideoResult:
    """Generate a video from text prompt using specified model.

    Args:
        prompt: Text description for video generation
        model_name: Name of the AI model to use
        duration: Video duration in seconds (default: 30)
        resolution: Output resolution in WxH format (default: "1024x576")

    Returns:
        VideoResult containing generated video path and metadata

    Raises:
        ModelNotFoundError: If specified model is not available
        GenerationError: If video generation fails

    Example:
        >>> result = generate_video("A cat playing piano", "t2v-model", 15)
        >>> print(result.video_path)
        "/outputs/video_123.mp4"
    """
    pass
```

### Error Handling

Use specific exception types and proper error handling:

```python
# Good: Specific exceptions
class ModelNotFoundError(Exception):
    """Raised when requested model is not available."""
    pass

class GenerationError(Exception):
    """Raised when video generation fails."""
    pass

def load_model(model_name: str) -> AIModel:
    """Load AI model by name."""
    try:
        model = ModelRegistry.get(model_name)
        if not model:
            raise ModelNotFoundError(f"Model '{model_name}' not found")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise GenerationError(f"Model loading failed: {e}") from e

# Bad: Generic exceptions
def load_model(model_name: str):
    try:
        return ModelRegistry.get(model_name)
    except:
        raise Exception("Something went wrong")
```

### Logging

Use structured logging with appropriate levels:

```python
import logging

logger = logging.getLogger(__name__)

def generate_video(prompt: str) -> VideoResult:
    """Generate video with proper logging."""
    logger.info(f"Starting video generation for prompt: {prompt[:50]}...")

    try:
        # Generation logic
        result = perform_generation(prompt)
        logger.info(f"Video generation completed: {result.video_path}")
        return result

    except Exception as e:
        logger.error(f"Video generation failed: {e}", exc_info=True)
        raise
```

## ⚛️ TypeScript/React Standards

### Code Style

We use **Prettier** and **ESLint** for consistent formatting:

```typescript
// Good: Clear component structure
interface VideoGeneratorProps {
  onVideoGenerated: (video: Video) => void;
  maxDuration?: number;
}

export const VideoGenerator: React.FC<VideoGeneratorProps> = ({
  onVideoGenerated,
  maxDuration = 60,
}) => {
  const [prompt, setPrompt] = useState<string>("");
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

  const handleGenerate = async (): Promise<void> => {
    setIsGenerating(true);
    try {
      const video = await generateVideo(prompt);
      onVideoGenerated(video);
    } catch (error) {
      console.error("Generation failed:", error);
    } finally {
      setIsGenerating(false);
    }
  };

  return <div className="video-generator">{/* Component JSX */}</div>;
};
```

### Naming Conventions

| Type       | Convention            | Example            |
| ---------- | --------------------- | ------------------ |
| Variables  | camelCase             | `videoUrl`         |
| Functions  | camelCase             | `generateVideo()`  |
| Components | PascalCase            | `VideoPlayer`      |
| Interfaces | PascalCase + I prefix | `IVideoConfig`     |
| Types      | PascalCase            | `VideoStatus`      |
| Constants  | UPPER_SNAKE_CASE      | `MAX_FILE_SIZE`    |
| Files      | kebab-case            | `video-player.tsx` |

### Component Structure

Follow this component structure:

```typescript
// 1. Imports
import React, { useState, useEffect } from "react";
import { VideoService } from "../services/video-service";
import { Button } from "./ui/button";

// 2. Types/Interfaces
interface VideoPlayerProps {
  videoUrl: string;
  autoPlay?: boolean;
  onEnded?: () => void;
}

// 3. Component
export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  videoUrl,
  autoPlay = false,
  onEnded,
}) => {
  // 4. State
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [duration, setDuration] = useState<number>(0);

  // 5. Effects
  useEffect(() => {
    // Effect logic
  }, [videoUrl]);

  // 6. Event handlers
  const handlePlay = (): void => {
    setIsPlaying(true);
  };

  const handlePause = (): void => {
    setIsPlaying(false);
  };

  // 7. Render
  return <div className="video-player">{/* JSX content */}</div>;
};
```

### State Management

Use appropriate state management patterns:

```typescript
// Local state for component-specific data
const [isLoading, setIsLoading] = useState<boolean>(false);

// Zustand for global state
interface AppState {
  videos: Video[];
  currentVideo: Video | null;
  addVideo: (video: Video) => void;
  setCurrentVideo: (video: Video) => void;
}

export const useAppStore = create<AppState>((set) => ({
  videos: [],
  currentVideo: null,
  addVideo: (video) =>
    set((state) => ({
      videos: [...state.videos, video],
    })),
  setCurrentVideo: (video) => set({ currentVideo: video }),
}));

// React Query for server state
const {
  data: videos,
  isLoading,
  error,
} = useQuery({
  queryKey: ["videos"],
  queryFn: () => videoService.getVideos(),
});
```

## 🎨 CSS/Styling Standards

### Tailwind CSS

We use Tailwind CSS for styling with these conventions:

```tsx
// Good: Semantic class grouping
<div className="
  flex items-center justify-between
  p-4 mb-6
  bg-white dark:bg-gray-800
  border border-gray-200 dark:border-gray-700
  rounded-lg shadow-sm
">
  <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
    Video Generator
  </h2>
</div>

// Bad: Random class order
<div className="rounded-lg flex bg-white p-4 border-gray-200 items-center shadow-sm border mb-6">
```

### Custom CSS

When custom CSS is needed, use CSS modules or styled-components:

```css
/* video-player.module.css */
.videoPlayer {
  @apply relative w-full h-auto;
}

.videoPlayer__controls {
  @apply absolute bottom-0 left-0 right-0;
  @apply flex items-center justify-between;
  @apply p-4 bg-black bg-opacity-50;
}

.videoPlayer__button {
  @apply px-4 py-2 text-white;
  @apply hover:bg-white hover:bg-opacity-20;
  @apply transition-colors duration-200;
}
```

## 🧪 Testing Standards

### Test Structure

Follow the **Arrange-Act-Assert** pattern:

```python
# Python test example
def test_video_generation_success():
    # Arrange
    prompt = "A cat playing piano"
    model_name = "test-model"
    expected_duration = 30

    # Act
    result = generate_video(prompt, model_name, expected_duration)

    # Assert
    assert result.success is True
    assert result.duration == expected_duration
    assert Path(result.video_path).exists()

# TypeScript test example
describe('VideoPlayer', () => {
  it('should play video when play button is clicked', async () => {
    // Arrange
    const mockOnPlay = jest.fn();
    render(<VideoPlayer videoUrl="test.mp4" onPlay={mockOnPlay} />);

    // Act
    const playButton = screen.getByRole('button', { name: /play/i });
    await user.click(playButton);

    // Assert
    expect(mockOnPlay).toHaveBeenCalledTimes(1);
  });
});
```

### Test Naming

Use descriptive test names that explain the scenario:

```python
# Good: Descriptive test names
def test_video_generation_fails_when_model_not_found():
def test_video_generation_respects_duration_limit():
def test_video_generation_creates_output_file():

# Bad: Vague test names
def test_generation():
def test_video():
def test_success():
```

### Mocking

Use appropriate mocking strategies:

```python
# Python mocking
@patch('video_service.AIModel')
def test_video_generation_with_mock_model(mock_model):
    # Setup mock
    mock_model.generate.return_value = VideoResult(
        success=True,
        video_path="/test/output.mp4"
    )

    # Test with mock
    result = generate_video("test prompt", "mock-model")
    assert result.success is True

# TypeScript mocking
const mockVideoService = {
  generateVideo: jest.fn().mockResolvedValue({
    id: '123',
    url: 'test-video.mp4',
    status: 'completed'
  })
};
```

## 📁 File Organization

### Directory Structure

Organize files by feature, not by type:

```
# Good: Feature-based organization
backend/
├── video_generation/
│   ├── __init__.py
│   ├── models.py
│   ├── services.py
│   ├── repositories.py
│   └── tests/
├── user_management/
│   ├── __init__.py
│   ├── models.py
│   ├── services.py
│   └── tests/
└── shared/
    ├── exceptions.py
    └── utils.py

# Bad: Type-based organization
backend/
├── models/
├── services/
├── repositories/
└── tests/
```

### Import Organization

Organize imports in this order:

```python
# 1. Standard library imports
import os
import sys
from pathlib import Path
from typing import List, Optional

# 2. Third-party imports
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 3. Local application imports
from core.models.video import Video
from infrastructure.config import settings
from .exceptions import GenerationError
```

## 🔧 Configuration Management

### Environment Variables

Use clear, descriptive environment variable names:

```bash
# Good: Clear variable names
WAN22_API_HOST=localhost
WAN22_API_PORT=8000
WAN22_DEBUG_MODE=true
WAN22_MODEL_CACHE_DIR=/models
WAN22_MAX_VIDEO_DURATION=300

# Bad: Unclear variable names
HOST=localhost
PORT=8000
DEBUG=1
DIR=/models
MAX=300
```

### Configuration Files

Use structured configuration with validation:

```yaml
# config/unified-config.yaml
system:
  debug: true
  log_level: INFO

backend:
  api:
    host: localhost
    port: 8000
    cors_origins:
      - "http://localhost:3000"

  video_generation:
    max_duration: 300
    default_resolution: "1024x576"
    models_directory: "./models"

frontend:
  api_url: "http://localhost:8000"
  theme: "light"
  features:
    advanced_mode: false
```

## 📝 Documentation Standards

### Code Comments

Write comments that explain **why**, not **what**:

```python
# Good: Explains reasoning
def calculate_optimal_batch_size(available_vram: int) -> int:
    """Calculate optimal batch size based on available VRAM.

    We use 80% of available VRAM to leave headroom for other operations
    and prevent out-of-memory errors during generation.
    """
    return int(available_vram * 0.8 / VRAM_PER_SAMPLE)

# Bad: States the obvious
def calculate_optimal_batch_size(available_vram: int) -> int:
    # Calculate batch size
    return int(available_vram * 0.8 / VRAM_PER_SAMPLE)
```

### API Documentation

Document all API endpoints with examples:

````python
@app.post("/api/v1/videos/generate")
async def generate_video(request: VideoGenerationRequest) -> VideoResponse:
    """Generate a video from text prompt.

    Args:
        request: Video generation parameters including prompt, model, and settings

    Returns:
        VideoResponse with generation task ID and status

    Raises:
        HTTPException: 400 if request parameters are invalid
        HTTPException: 503 if generation service is unavailable

    Example:
        ```
        POST /api/v1/videos/generate
        {
            "prompt": "A cat playing piano",
            "model": "t2v-model",
            "duration": 30,
            "resolution": "1024x576"
        }
        ```
    """
    pass
````

## 🚀 Performance Guidelines

### Python Performance

```python
# Good: Efficient list comprehension
active_tasks = [task for task in tasks if task.status == 'active']

# Bad: Inefficient loop
active_tasks = []
for task in tasks:
    if task.status == 'active':
        active_tasks.append(task)

# Good: Use generators for large datasets
def process_large_dataset(data):
    for item in data:
        yield process_item(item)

# Good: Cache expensive operations
@lru_cache(maxsize=128)
def get_model_config(model_name: str) -> ModelConfig:
    return load_model_config(model_name)
```

### React Performance

```typescript
// Good: Memoize expensive calculations
const expensiveValue = useMemo(() => {
  return calculateExpensiveValue(data);
}, [data]);

// Good: Memoize components
const VideoItem = React.memo<VideoItemProps>(({ video, onSelect }) => {
  return <div onClick={() => onSelect(video)}>{video.title}</div>;
});

// Good: Use callback for event handlers
const handleVideoSelect = useCallback((video: Video) => {
  setSelectedVideo(video);
}, []);
```

## 🔒 Security Guidelines

### Input Validation

Always validate and sanitize inputs:

```python
from pydantic import BaseModel, validator

class VideoGenerationRequest(BaseModel):
    prompt: str
    duration: int
    resolution: str

    @validator('prompt')
    def validate_prompt(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Prompt cannot be empty')
        if len(v) > 1000:
            raise ValueError('Prompt too long (max 1000 characters)')
        return v.strip()

    @validator('duration')
    def validate_duration(cls, v):
        if v < 1 or v > 300:
            raise ValueError('Duration must be between 1 and 300 seconds')
        return v
```

### File Handling

Secure file operations:

```python
from pathlib import Path

def save_generated_video(video_data: bytes, filename: str) -> Path:
    """Save video with secure filename handling."""
    # Sanitize filename
    safe_filename = secure_filename(filename)

    # Ensure file is saved in allowed directory
    output_dir = Path(settings.OUTPUT_DIRECTORY)
    output_path = output_dir / safe_filename

    # Prevent directory traversal
    if not str(output_path.resolve()).startswith(str(output_dir.resolve())):
        raise ValueError("Invalid file path")

    # Save file
    with open(output_path, 'wb') as f:
        f.write(video_data)

    return output_path
```

## 🔄 Git Workflow

### Commit Messages

Use conventional commit format:

```
feat: add video generation progress tracking
fix: resolve memory leak in model loading
docs: update API documentation for new endpoints
test: add integration tests for video generation
refactor: extract video processing into separate service
```

### Branch Naming

Use descriptive branch names:

```
feature/video-progress-tracking
bugfix/memory-leak-model-loading
hotfix/critical-generation-error
docs/api-documentation-update
```

## ✅ Pre-commit Checklist

Before committing code, ensure:

- [ ] Code follows style guidelines (run formatters)
- [ ] All tests pass
- [ ] Type checking passes (MyPy for Python, TypeScript compiler)
- [ ] No linting errors
- [ ] Documentation is updated
- [ ] Commit message follows conventions
- [ ] No sensitive data in commit

## 🛠️ Tools and Automation

### Development Tools

```bash
# Python tools
pip install black isort mypy pytest pre-commit

# Frontend tools
npm install -D prettier eslint @typescript-eslint/parser

# Pre-commit hooks
pre-commit install
```

### IDE Configuration

Recommended VS Code settings:

```json
{
  "python.formatting.provider": "black",
  "python.linting.mypyEnabled": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "typescript.preferences.importModuleSpecifier": "relative"
}
```

## 📚 Resources

### Style Guides

- [PEP 8 - Python Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript)

### Tools Documentation

- [Black - Python Formatter](https://black.readthedocs.io/)
- [Prettier - Code Formatter](https://prettier.io/)
- [ESLint - JavaScript Linter](https://eslint.org/)
- [MyPy - Python Type Checker](https://mypy.readthedocs.io/)

Remember: These standards exist to make our code more maintainable, readable, and collaborative. When in doubt, prioritize clarity and consistency! 🚀
