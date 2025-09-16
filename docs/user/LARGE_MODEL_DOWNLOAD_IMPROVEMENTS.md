---
category: user
last_updated: '2025-09-15T22:50:00.284292'
original_path: local_installation\LARGE_MODEL_DOWNLOAD_IMPROVEMENTS.md
tags:
- configuration
- api
- troubleshooting
- installation
title: Large Model Download Improvements
---

# Large Model Download Improvements

## Issue Identified

Large model downloads may appear successful but be incomplete, causing validation failures later in the installation process.

## Root Cause

- **Large file sizes** (multi-GB models) can timeout or fail partially
- **Network interruptions** during long downloads
- **Insufficient validation** of download completeness
- **Progress reporting** may show 100% even for incomplete downloads

## Proposed Improvements

### 1. Enhanced Download Validation

```python
def verify_download_integrity(self, model_path: Path, expected_size: int = None) -> bool:
    """Verify downloaded model integrity with multiple checks."""

    # Check file exists
    if not model_path.exists():
        return False

    # Check file size if expected size provided
    if expected_size and model_path.stat().st_size != expected_size:
        self.logger.warning(f"File size mismatch: expected {expected_size}, got {model_path.stat().st_size}")
        return False

    # Check if file is complete (not truncated)
    try:
        # For PyTorch models, try to load metadata
        if model_path.suffix == '.bin':
            import torch
            torch.load(model_path, map_location='cpu', weights_only=True)

        # For other formats, check file headers
        elif model_path.suffix == '.safetensors':
            # Validate safetensors format
            pass

        return True
    except Exception as e:
        self.logger.error(f"Model integrity check failed: {e}")
        return False
```

### 2. Robust Download with Retry

```python
def download_with_integrity_check(self, repo_id: str, local_dir: Path, max_retries: int = 3) -> bool:
    """Download with integrity verification and retry logic."""

    for attempt in range(max_retries):
        try:
            # Download with HF Hub
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                resume_download=True,
                local_dir_use_symlinks=False
            )

            # Verify all files are complete
            if self._verify_all_model_files(local_dir):
                return True
            else:
                self.logger.warning(f"Download verification failed, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    # Clean up incomplete download
                    shutil.rmtree(local_dir, ignore_errors=True)
                    local_dir.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            self.logger.error(f"Download attempt {attempt + 1} failed: {e}")

    return False
```

### 3. Better Progress Tracking

```python
def track_download_progress(self, repo_id: str, callback: Callable = None):
    """Enhanced progress tracking for large downloads."""

    # Get repository info first
    repo_info = self._get_repo_info(repo_id)
    total_size = sum(file.size for file in repo_info.files)

    # Track actual bytes downloaded
    downloaded_bytes = 0

    def progress_hook(downloaded: int, total: int):
        nonlocal downloaded_bytes
        downloaded_bytes = downloaded

        if callback:
            progress_percent = (downloaded / total_size) * 100
            callback(repo_id, progress_percent, f"Downloaded {downloaded_bytes / 1024**3:.1f}GB / {total_size / 1024**3:.1f}GB")

    return progress_hook
```

## Immediate Workarounds

### For Current Installation

Since you're manually downloading, here's how to integrate them:

1. **Place models in correct directories:**

```
models/
├── WAN2.2-T2V-A14B/
├── WAN2.2-I2V-A14B/
└── WAN2.2-TI2V-5B/
```

2. **Run validation only:**

```bash
install.bat --skip-models --force-reinstall
```

3. **Or validate existing models:**

```bash
python scripts/validate_installation.py --models-only
```

## Configuration Improvements

### Add Download Timeouts

```json
{
  "model_download": {
    "timeout_seconds": 3600,
    "chunk_size": 8192,
    "max_retries": 3,
    "verify_integrity": true,
    "resume_downloads": true
  }
}
```

### Add Size Validation

```python
MODEL_CONFIG = {
    "WAN2.2-T2V-A14B": ModelInfo(
        name="WAN2.2-T2V-A14B",
        repo_id="your-org/wan22-t2v-a14b",
        version="main",
        size_gb=28.5,
        expected_files={
            "pytorch_model.bin": 15_000_000_000,  # Expected size in bytes
            "config.json": 1024,
            # ... other files with expected sizes
        },
        required=True,
        local_dir="WAN2.2-T2V-A14B"
    )
}
```

## Testing Recommendations

### 1. Test with Smaller Models First

```bash
# Test with smaller models to verify the system works
install.bat --test-mode
```

### 2. Monitor Download Progress

```bash
# Use verbose mode to see detailed progress
install.bat --verbose
```

### 3. Check Available Disk Space

```bash
# Ensure sufficient space before starting
install.bat --check-space-only
```

## Production Recommendations

### 1. Pre-flight Checks

- Verify available disk space (3x model size recommended)
- Check network stability
- Estimate download time based on connection speed

### 2. Download Monitoring

- Real-time progress with ETA
- Bandwidth usage monitoring
- Automatic pause/resume on network issues

### 3. Integrity Verification

- File size validation
- Checksum verification (if available)
- Model loading test
- Metadata validation

## Next Steps

1. **Complete your manual download** - This will test the rest of the system
2. **Run validation** - Test that everything works with complete models
3. **Implement improvements** - Add the enhanced download validation
4. **Test with real WAN2.2 models** - Update repository IDs when ready

The installation system is working great - this download integrity issue is the final piece to make it production-ready for large models!
