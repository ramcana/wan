---
category: reference
last_updated: '2025-09-15T22:49:59.931912'
original_path: docs\PHASE_1_MVP_GUIDE.md
tags:
- configuration
- api
- troubleshooting
- installation
- security
- performance
title: 'Phase 1 MVP Guide: WAN Models Integration'
---

# Phase 1 MVP Guide: WAN Models Integration

## Overview

Phase 1 delivers a fully functional MVP with integrated WAN models supporting T2V (text-to-video), I2V (image-to-video), and TI2V (text+image-to-video) generation with seamless model switching and enhanced user experience.

## 🎯 Phase 1 Goals Achieved

### ✅ Enhanced Model Functionality

- **Seamless Model Switching**: Auto-detection of optimal model based on input type
- **T2V Support**: Text-to-video generation with T2V-A14B model
- **I2V Support**: Image-to-video generation with I2V-A14B model
- **TI2V Support**: Text+image-to-video generation with TI2V-5B model
- **Prompt Enhancement**: Automatic prompt optimization for better results
- **LoRA Integration**: Quick style customizations with LoRA adapters
- **Hardware Optimization**: Memory and performance optimizations for RTX 4080

### ✅ API and Backend Refinements

- **Enhanced Generation Endpoint**: `/api/v1/generation/submit` with auto-detection
- **Model Detection API**: `/api/v1/generation/models/detect` for intelligent model selection
- **Prompt Enhancement API**: `/api/v1/generation/prompt/enhance` for better prompts
- **Capabilities API**: `/api/v1/generation/capabilities` for system information
- **Input Validation**: Comprehensive validation for all parameters
- **Queue Management**: Improved concurrent generation handling
- **WebSocket Progress**: Real-time generation progress updates

### ✅ Frontend Enhancements

- **Auto-Detection UI**: Intelligent model selection with user feedback
- **Enhanced Form**: Improved generation form with real-time validation
- **Image Upload**: Support for start and end image uploads
- **Progress Tracking**: Real-time progress with WebSocket integration
- **Mobile Responsive**: Optimized for all device sizes
- **Error Handling**: Comprehensive error display and recovery

### ✅ CLI Integration

- **WAN CLI Commands**: Complete CLI toolkit for model management
- **Testing Suite**: Comprehensive test commands for validation
- **Health Monitoring**: System health checks and optimization
- **Generation Tools**: CLI-based video generation capabilities

## 🚀 Quick Start

### 1. Installation and Setup

```bash
# Install dependencies
pip install -r requirements.txt
cd frontend && npm install

# Install CLI tools
python install_cli.py

# Validate Phase 1 setup
python scripts/deploy_phase1_mvp.py
```

### 2. Start the System

```bash
# Start backend (Terminal 1)
python backend/app.py

# Start frontend (Terminal 2)
cd frontend && npm run dev

# Verify system health
wan-cli wan health --detailed
```

### 3. Test Generation

```bash
# CLI generation
wan-cli wan generate "a beautiful sunset over mountains"

# Web interface
# Navigate to http://localhost:3000
```

## 📚 API Reference

### Enhanced Generation API

#### Auto-Detection Generation

```http
POST /api/v1/generation/submit
Content-Type: multipart/form-data

{
  "prompt": "a beautiful landscape",
  "enable_prompt_enhancement": true,
  "enable_optimization": true
}
```

**Response:**

```json
{
  "success": true,
  "task_id": "gen_abc123",
  "detected_model_type": "T2V-A14B",
  "estimated_time_minutes": 2.5,
  "enhanced_prompt": "a beautiful landscape, cinematic composition, high quality, detailed, HD quality",
  "applied_optimizations": [
    "Hardware-specific quantization",
    "Prompt enhancement"
  ]
}
```

#### Model Detection

```http
GET /api/v1/generation/models/detect?prompt=animate%20this%20image&has_image=true
```

**Response:**

```json
{
  "detected_model_type": "I2V-A14B",
  "confidence": 0.9,
  "explanation": [
    "Single image provided - I2V recommended for pure image animation"
  ],
  "requirements": {
    "requires_image": true,
    "estimated_vram_gb": 8.5,
    "estimated_time_per_frame": 1.4
  }
}
```

#### Prompt Enhancement

```http
POST /api/v1/generation/prompt/enhance
Content-Type: application/x-www-form-urlencoded

prompt=a cat&model_type=T2V-A14B&enhance_quality=true
```

**Response:**

```json
{
  "original_prompt": "a cat",
  "enhanced_prompt": "a cat, cinematic composition, smooth camera movement, high quality, detailed, HD quality",
  "enhancements_applied": [
    "cinematic composition",
    "smooth camera movement",
    "high quality, detailed",
    "HD quality"
  ]
}
```

## 🛠️ CLI Commands

### Model Management

```bash
# List all models with status
wan-cli wan models --detailed

# Check model health
wan-cli wan health --detailed --fix

# Run model tests
wan-cli wan test --pattern="test_wan_models*"

# Check test coverage
wan-cli wan coverage --min=85
```

### Video Generation

```bash
# Text-to-video
wan-cli wan generate "a cat walking in a garden"

# Image-to-video
wan-cli wan generate "animate this image" --image=input.jpg --model=I2V

# Text+Image-to-video with auto-detection
wan-cli wan generate "transform into a magical scene" --image=input.jpg

# Custom parameters
wan-cli wan generate "cinematic landscape" --resolution=1920x1080 --steps=75
```

### System Optimization

```bash
# Optimize for memory
wan-cli wan optimize --target=memory --gpu=RTX4080

# Optimize for speed
wan-cli wan optimize --target=speed --vram=16

# Dry run optimization
wan-cli wan optimize --target=quality --dry-run
```

### Testing and Validation

```bash
# Quick model tests
wan-cli wan test --quick

# Test specific model
wan-cli wan test --model=T2V --verbose

# Full audit
wan-cli wan audit --detailed --fix

# Coverage check
wan-cli wan coverage --min=85 --model=I2V-A14B
```

## 🎨 Frontend Usage

### Auto-Detection Mode (Recommended)

1. Select "Auto-Detect (Recommended)" as model type
2. Enter your prompt
3. Upload images if desired
4. System automatically selects optimal model
5. Real-time feedback shows detected model and reasoning

### Manual Model Selection

1. Choose specific model (T2V, I2V, or TI2V)
2. System validates requirements (e.g., image for I2V)
3. Upload required images
4. Configure advanced settings if needed

### Advanced Features

- **Prompt Enhancement**: Automatic prompt optimization
- **LoRA Selection**: Style customization with LoRA models
- **Resolution Options**: 480p to 1080p support
- **Steps Configuration**: Quality vs speed balance
- **Real-time Progress**: WebSocket-based progress updates

## 🔧 Configuration

### Model Configuration

```yaml
# config/models.yaml
models:
  t2v_a14b:
    path: "models/t2v-a14b"
    vram_usage: 8.0
    optimization: "bf16"

  i2v_a14b:
    path: "models/i2v-a14b"
    vram_usage: 8.5
    optimization: "bf16"

  ti2v_5b:
    path: "models/ti2v-5b"
    vram_usage: 6.0
    optimization: "bf16"
```

### Hardware Optimization

```yaml
# config/hardware.yaml
hardware:
  gpu: "RTX4080"
  vram_gb: 16
  quantization: "bf16"
  enable_offload: true
  enable_flash_attention: true
```

## 🧪 Testing

### Run Phase 1 Test Suite

```bash
# Full test suite
python -m pytest tests/test_wan_models_phase1.py -v

# Specific test categories
python -m pytest tests/test_wan_models_phase1.py::TestWANModelsPhase1 -v
python -m pytest tests/test_wan_models_phase1.py::TestWANModelsCLI -v
python -m pytest tests/test_wan_models_phase1.py::TestWANModelsIntegration -v

# Performance tests
python -m pytest tests/test_wan_models_phase1.py::TestWANModelsPerformance -v
```

### Coverage Requirements

- **Minimum Coverage**: 85% on WAN model paths
- **Test Categories**: Unit, Integration, Performance, CLI
- **Validation**: Automated via `wan-cli wan coverage`

## 📊 Performance Metrics

### Expected Performance (RTX 4080)

| Model    | Resolution | Steps | Est. Time   | VRAM Usage |
| -------- | ---------- | ----- | ----------- | ---------- |
| T2V-A14B | 720p       | 50    | 2-3 min     | ~8GB       |
| I2V-A14B | 720p       | 50    | 2.5-3.5 min | ~8.5GB     |
| TI2V-5B  | 720p       | 40    | 1.5-2.5 min | ~6GB       |

### Optimization Features

- **BF16 Quantization**: 20% memory reduction
- **Model Offloading**: 30% memory reduction when enabled
- **Pipeline Caching**: Faster subsequent generations
- **Flash Attention**: Improved memory efficiency

## 🚨 Troubleshooting

### Common Issues

#### Model Detection Not Working

```bash
# Check API connectivity
curl http://localhost:9000/api/v1/generation/capabilities

# Validate backend
wan-cli wan health --detailed
```

#### Generation Fails

```bash
# Check VRAM usage
wan-cli wan health --detailed

# Optimize for memory
wan-cli wan optimize --target=memory

# Check logs
tail -f backend/logs/generation.log
```

#### Frontend Not Connecting

```bash
# Check backend status
curl http://localhost:9000/health

# Validate CORS configuration
curl http://localhost:9000/api/v1/system/cors/validate
```

### Performance Issues

```bash
# Check system optimization
wan-cli wan optimize --dry-run

# Monitor performance
wan-cli health monitor --duration=300

# Run performance tests
wan-cli wan test --performance
```

## 🔄 Deployment

### Local Development

```bash
# Start development servers
python scripts/start_development.py

# Run validation
python scripts/deploy_phase1_mvp.py
```

### Staging Deployment

```bash
# Deploy to staging
wan-cli deploy staging --dry-run
wan-cli deploy staging --execute

# Validate deployment
wan-cli wan audit --detailed
```

### Production Readiness

```bash
# Full system check
wan-cli health comprehensive --fix

# Performance validation
wan-cli wan test --performance --min-score=85

# Security audit
wan-cli quality security-scan
```

## 📈 Next Steps (Phase 2)

Phase 1 provides the foundation for:

- **External API Integration**: Hugging Face, Replicate APIs
- **Advanced LoRA Management**: Custom training and fine-tuning
- **Batch Processing**: Multiple video generation
- **Cloud Deployment**: Scalable cloud infrastructure
- **Advanced UI Features**: Timeline editing, preview modes

## 🤝 Contributing

### Development Workflow

1. **Feature Development**: Create feature branch
2. **Testing**: Run `wan-cli wan test --comprehensive`
3. **Validation**: Execute `python scripts/deploy_phase1_mvp.py`
4. **Documentation**: Update relevant docs
5. **Pull Request**: Submit with test results

### Code Quality

```bash
# Code quality check
wan-cli quality check --fix

# Test coverage
wan-cli wan coverage --min=85

# Performance validation
wan-cli wan test --performance
```

---

**Phase 1 MVP Status**: ✅ **Production Ready**

For support and questions, see the troubleshooting section or check the comprehensive test suite results.
