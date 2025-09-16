---
category: reference
last_updated: '2025-09-15T22:50:01.133173'
original_path: working_curl_commands.md
tags:
- configuration
- api
- troubleshooting
title: WAN22 Backend - Working Curl Commands
---

# WAN22 Backend - Working Curl Commands

## Backend Status: ✅ OPERATIONAL

- **URL**: http://localhost:9000
- **API Docs**: http://localhost:9000/docs
- **Models Root**: D:\AI\models
- **Models Detected**: 3 (t2v-A14B, i2v-A14B, ti2v-5B)

## ✅ Working Endpoints

### Health Checks

```bash
# Basic health
curl http://localhost:9000/health

# System health with details
curl http://localhost:9000/api/v1/system/health
```

### Model Status

```bash
# Check model detection and status
curl http://localhost:9000/api/v1/models/status
```

### Prompt Enhancement

```bash
# Enhance a text prompt
curl -X POST http://localhost:9000/api/v1/prompt/enhance \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A beautiful mountain landscape at sunset"}'
```

### Queue Management

```bash
# Check generation queue
curl http://localhost:9000/api/v1/queue
```

## 🔧 Model Recognition Results

The backend successfully detects your models but reports them as "missing" due to configuration mismatches:

- **t2v-A14B@2.2.0**: ❌ Missing VAE and text encoder files
- **i2v-A14B@2.2.0**: ❌ File size mismatches, missing image encoder
- **ti2v-5b@2.2.0**: ❌ Config file size mismatch

### Actual vs Expected Files

**Your models have:**

- `configuration.json` (47 bytes) vs expected (2048 bytes)
- `models_t5_umt5-xxl-enc-bf16.pth` (11GB) vs expected (4GB)
- `Wan2.1_VAE.pth` (507MB) vs expected (335MB)
- Missing `image_encoder/pytorch_model.bin`

## 🎯 Next Steps

1. **✅ Backend is working** - All core endpoints operational
2. **🔧 Model config needs updating** - File sizes and paths in `config/models.toml`
3. **🐛 Generation endpoint** - Has parameter issue to fix
4. **📁 Model structure** - May need file mapping or config updates

## 🧪 Test Results Summary

| Endpoint           | Status             | Notes                                    |
| ------------------ | ------------------ | ---------------------------------------- |
| Health checks      | ✅ Working         | All health endpoints operational         |
| Model detection    | ✅ Working         | Finds 3 models, reports config issues    |
| Prompt enhancement | ✅ Working         | Enhances prompts successfully            |
| Queue management   | ✅ Working         | Returns empty queue as expected          |
| Generation submit  | ⚠️ Parameter issue | Endpoint responds but has param mismatch |

The backend is successfully running and the model orchestrator is working - it's correctly identifying that your models exist but don't match the expected configuration!
