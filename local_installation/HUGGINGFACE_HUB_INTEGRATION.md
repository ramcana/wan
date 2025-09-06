# WAN2.2 Hugging Face Hub Integration Guide

## Overview

The WAN2.2 installation system now uses the **Hugging Face Hub** for model downloads, providing better authentication, resume capabilities, and integration with the HF ecosystem.

## ✅ What's Already Set Up

### **Dependencies**

- ✅ `huggingface-hub>=0.16.0` already included in requirements.txt
- ✅ Automatic installation during dependency phase
- ✅ Fallback handling if library not available

### **Integration Features**

- ✅ **Proper HF Hub authentication** support
- ✅ **Resume downloads** if interrupted
- ✅ **Snapshot downloads** for complete repositories
- ✅ **Error handling** for authentication issues
- ✅ **Progress tracking** and logging

## 🔧 Configuration for Real WAN2.2 Models

### **Step 1: Update Model Repository IDs**

Currently using example repositories. Update in `scripts/download_models.py`:

```python
MODEL_CONFIG = {
    "WAN2.2-T2V-A14B": ModelInfo(
        name="WAN2.2-T2V-A14B",
        repo_id="your-org/wan22-t2v-a14b",  # Replace with actual repo
        version="main",
        size_gb=28.5,
        required=True,
        files=[...],
        local_dir="WAN2.2-T2V-A14B"
    ),
    # ... other models
}
```

### **Step 2: Authentication Setup**

For private repositories, users need to authenticate:

#### **Option A: CLI Login (Recommended)**

```bash
huggingface-cli login
```

#### **Option B: Environment Variable**

```bash
set HF_TOKEN=your_token_here
```

#### **Option C: Programmatic Login**

```python
from huggingface_hub import login
login(token="your_token_here")
```

## 🚀 How It Works

### **Download Process**

1. **Check existing models** - Skip already downloaded
2. **Authenticate** - Use HF token if required
3. **Download repositories** - Complete model repos via `snapshot_download`
4. **Resume support** - Automatically resume interrupted downloads
5. **Verify integrity** - Check downloaded files
6. **Update metadata** - Track download status

### **Error Handling**

- **401 Unauthorized**: Clear guidance on authentication
- **Network issues**: Automatic retry with exponential backoff
- **Disk space**: Pre-flight checks and clear error messages
- **Corrupted downloads**: Automatic re-download

## 📋 Usage Examples

### **For End Users**

#### **Public Models (No Auth Required)**

```bash
install.bat  # Downloads models automatically
```

#### **Private Models (Auth Required)**

```bash
# First, authenticate
huggingface-cli login

# Then install
install.bat
```

#### **Skip Models for Testing**

```bash
install.bat --skip-models  # Skip model download
```

### **For Developers**

#### **Test with Different Models**

```python
# Update MODEL_CONFIG in download_models.py
MODEL_CONFIG = {
    "test-model": ModelInfo(
        name="test-model",
        repo_id="microsoft/DialoGPT-small",  # Public test model
        version="main",
        size_gb=1.0,
        required=True,
        files=["pytorch_model.bin", "config.json"],
        local_dir="test-model"
    )
}
```

#### **Manual Model Download**

```python
from scripts.download_models import ModelDownloader

downloader = ModelDownloader("./")
success = downloader.download_wan22_models()
```

## 🔍 Authentication Troubleshooting

### **Common Issues**

#### **401 Unauthorized Error**

```
ERROR: 401 Client Error: Unauthorized
```

**Solutions:**

1. **Login via CLI**: `huggingface-cli login`
2. **Set token**: `set HF_TOKEN=your_token`
3. **Check repository access**: Ensure you have permission
4. **Use public models**: For testing, use public repositories

#### **Token Not Found**

```
ERROR: Token not found
```

**Solutions:**

1. **Generate token**: Visit https://huggingface.co/settings/tokens
2. **Set permissions**: Ensure token has read access
3. **Login again**: `huggingface-cli login --token your_token`

#### **Repository Not Found**

```
ERROR: Repository not found
```

**Solutions:**

1. **Check repo ID**: Verify the repository exists
2. **Check spelling**: Ensure correct organization/model name
3. **Check privacy**: Ensure you have access to private repos

## 📊 Current Model Configuration

### **Example Repositories (For Testing)**

- **WAN2.2-T2V-A14B**: `microsoft/DialoGPT-medium`
- **WAN2.2-I2V-A14B**: `microsoft/DialoGPT-small`
- **WAN2.2-TI2V-5B**: `microsoft/DialoGPT-large`

### **For Production**

Replace with actual WAN2.2 model repositories:

- Update `repo_id` fields in `MODEL_CONFIG`
- Update `files` lists with actual model files
- Update `size_gb` with actual model sizes

## 🛠️ Advanced Configuration

### **Custom Download Location**

```python
downloader = ModelDownloader(
    installation_path="./",
    models_dir="./custom_models"
)
```

### **Parallel Downloads**

```python
downloader = ModelDownloader(
    installation_path="./",
    max_workers=5  # Increase for faster downloads
)
```

### **Custom Progress Callback**

```python
def progress_callback(model_name, progress, message):
    print(f"{model_name}: {progress:.1f}% - {message}")

downloader.download_wan22_models(progress_callback=progress_callback)
```

## 🔐 Security Best Practices

### **Token Management**

- ✅ **Use environment variables** for tokens
- ✅ **Don't commit tokens** to version control
- ✅ **Use read-only tokens** when possible
- ✅ **Rotate tokens regularly**

### **Repository Access**

- ✅ **Verify repository authenticity** before downloading
- ✅ **Use official model repositories** when available
- ✅ **Check model licenses** before use

## 📈 Benefits of HF Hub Integration

### **Reliability**

- ✅ **Resume downloads** - No need to restart large downloads
- ✅ **Integrity checks** - Automatic verification
- ✅ **Error recovery** - Robust error handling

### **Performance**

- ✅ **Parallel downloads** - Multiple files simultaneously
- ✅ **Efficient storage** - Symlinks and deduplication
- ✅ **Progress tracking** - Real-time download status

### **Ecosystem Integration**

- ✅ **Standard authentication** - Works with HF ecosystem
- ✅ **Model versioning** - Support for specific revisions
- ✅ **Metadata handling** - Automatic model information

## 🎯 Next Steps

### **For Immediate Use**

1. **Test with current setup** - Uses example models for testing
2. **Verify authentication** - Test HF login if needed
3. **Run installation** - `install.bat` should work with HF Hub

### **For Production Deployment**

1. **Update model repositories** - Replace with actual WAN2.2 repos
2. **Configure authentication** - Set up tokens for private repos
3. **Test thoroughly** - Verify all models download correctly
4. **Update documentation** - Provide user authentication guide

---

**Status**: ✅ **HF Hub Integration Complete**  
**Ready for**: Testing with example models, production with real repos  
**Authentication**: Supported for private repositories

_The installation system now uses industry-standard Hugging Face Hub for reliable, authenticated model downloads._
