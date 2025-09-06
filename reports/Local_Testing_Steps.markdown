# Local Testing Steps for Wan2.2 UI Variant

This document outlines the steps to run and test a local copy of the Wan2.2 UI Variant, validating the Task 15 performance optimizations (e.g., 80% VRAM reduction, 720p < 9 min, 1080p < 17 min) using the existing repository files. These steps assume the repository is already present in your development system, dependencies are installed, and configuration files are ready as per `DEPLOYMENT_GUIDE.md`.

## Prerequisites

- **Hardware**: NVIDIA GPU with ≥12GB VRAM (e.g., RTX 3060+), 16GB RAM (32GB recommended), 50GB+ free storage.
- **Software**: Python 3.8+, CUDA 11.8/12.1, cuDNN 8.6+, pip, virtual environment (venv).
- **Repository**: Wan2.2 UI Variant files present in your working directory.
- **Configuration**: `config.json` and `.env` files configured with local paths and Hugging Face token.

## Steps to Run and Test Locally

### 1. Verify Environment Setup

1. **Navigate to Project Directory**:
   ```bash
   cd /path/to/wan22-ui-variant
   ```

2. **Activate Virtual Environment**:
   ```bash
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Confirm CUDA Availability**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```
   Ensure CUDA is available. If not, verify NVIDIA drivers (`nvidia-smi`) and reinstall PyTorch:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Validate Configuration**:
   Ensure `config.json` is set up with:
   ```json
   {
     "system": {
       "default_quantization": "bf16",
       "enable_offload": true,
       "vae_tile_size": 256,
       "stats_refresh_interval": 5
     },
     "directories": {
       "output_directory": "outputs",
       "models_directory": "models",
       "loras_directory": "loras"
     },
     "optimization": {
       "max_vram_usage_gb": 12,
       "enable_memory_efficient_attention": true,
       "enable_attention_slicing": true
     },
     "performance": {
       "target_720p_time_minutes": 9,
       "target_1080p_time_minutes": 17,
       "vram_warning_threshold": 0.9
     }
   }
   ```
   Verify:
   ```bash
   python performance_profiler.py --validate-config
   ```

5. **Check Environment Variables**:
   Ensure `.env` includes:
   ```bash
   HF_TOKEN=your_huggingface_token_here
   CUDA_VISIBLE_DEVICES=0
   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

### 2. Test Performance Profiler

1. **Run Real-Time Monitoring**:
   ```bash
   python performance_profiler.py --monitor
   ```
   - Confirm metrics refresh every 5 seconds (Requirement 7.5).
   - Check for bottleneck warnings (e.g., high CPU/VRAM usage).

2. **Profile a Sample Task**:
   ```bash
   python performance_profiler.py --profile --input sample_video_task.json
   ```
   - If `sample_video_task.json` is missing, create one (see Step 4).
   - Verify bottleneck detection and performance recommendations.

3. **Generate Performance Report**:
   ```bash
   python -c "from performance_profiler import generate_performance_report; generate_performance_report('performance_report.json')"
   ```
   - Check `performance_report.json` for CPU, memory, GPU, and VRAM metrics.

### 3. Test Performance Optimizations

1. **Run Automated Optimization**:
   ```bash
   python optimize_performance.py --auto-optimize
   ```
   - Confirm optimal settings (e.g., VAE tiling, attention slicing) are applied.

2. **Execute Benchmark Tests**:
   ```bash
   python optimize_performance.py --benchmark
   ```
   - Validate performance targets:
     - 720p generation: < 9 minutes.
     - 1080p generation: < 17 minutes.
     - VRAM usage: < 12GB (Requirement 4.4).
   - Confirm up to 80% VRAM reduction.

3. **Test VRAM Optimizations**:
   ```bash
   python optimize_performance.py --test-vram --resolution 720p
   python optimize_performance.py --test-vram --resolution 1080p
   ```
   - Ensure VRAM usage stays within limits.

### 4. Test Video Generation

1. **Prepare Sample Input**:
   If `sample_input.json` is missing, create one:
   ```json
   {
     "input": "sample_video_prompt",
     "resolution": "720p",
     "output_path": "outputs/test_video.mp4"
   }
   ```
   Save as `sample_input.json` in the project root.

2. **Run Video Generation**:
   ```bash
   python generate_video.py --input sample_input.json --resolution 720p
   python generate_video.py --input sample_input.json --resolution 1080p
   ```
   - Verify generation times (720p < 9 min, 1080p < 17 min).
   - Check output video quality in `outputs/test_video.mp4`.
   - Monitor resources in another terminal:
     ```bash
     python performance_profiler.py --monitor
     ```

### 5. Run Integration Tests

1. **Execute Test Suite**:
   ```bash
   python -m unittest tests/test_integration.py
   ```
   - Validate generation timing, VRAM optimization, and monitoring accuracy.
   - Review test reports for failures.

2. **Debug Failures**:
   Check logs:
   ```bash
   tail -f wan22_ui.log
   tail -f wan22_errors.log
   ```
   Enable debug mode for detailed logs:
   ```bash
   python main.py --debug
   ```

### 6. Test UI (Optional)

1. **Launch UI**:
   ```bash
   python main.py --debug --port 7860
   ```
   - Access at `http://localhost:7860`.
   - Test video generation and monitoring via the UI.

2. **Validate Health Check**:
   ```bash
   curl http://localhost:7860/health
   ```
   - Confirm response includes `"status": "healthy"` and GPU availability.

### 7. Troubleshoot Issues

- **CUDA Out of Memory**:
  - Enable `enable_cpu_offload` and `enable_attention_slicing` in `config.json`.
  - Reduce resolution or batch size in `sample_input.json`.
- **Slow Generation**:
  - Verify `torch.compile` and `attention_slicing` are enabled.
  - Check GPU utilization: `nvidia-smi`.
- **Model Download Errors**:
  - Ensure `HF_TOKEN` is valid in `.env`.
  - Clear model cache:
    ```bash
    python -c "from utils import get_model_manager; get_model_manager().clear_cache()"
    ```
- **UI Not Accessible**:
  - Check firewall and port availability:
    ```bash
    netstat -tuln | grep 7860
    ```

### 8. Document Results

1. **Summarize Metrics**:
   - Record generation times, VRAM usage, and bottlenecks from `performance_report.json`.
   - Compare against Task 15 targets (720p < 9 min, 1080p < 17 min, VRAM < 12GB).

2. **Update Documentation**:
   - Add local testing notes or troubleshooting tips to `USER_GUIDE.md`.

### 9. Prepare for Production

1. **Simulate Production Workflow**:
   ```bash
   python pipeline.py --config config.json
   ```
   - Test the full pipeline (input preparation, optimization, generation, reporting).

2. **Backup Configurations**:
   ```bash
   tar -czf wan22-backup-$(date +%Y%m%d).tar.gz config.json .env outputs/ loras/
   ```

## Notes

- **Logs**: Monitor `wan22_ui.log` and `wan22_errors.log` for issues.
- **Multi-GPU**: For multiple GPUs, set `CUDA_VISIBLE_DEVICES=0,1` and update `config.json` (`multi_gpu: true`).
- **Sample Inputs**: If sample inputs are missing, refer to `USER_GUIDE.md` for format or use the provided `sample_input.json` example.
- **Performance Reports**: Use `performance_report.json` to verify Task 15 deliverables (e.g., 80% VRAM reduction).

For additional help, refer to `DEPLOYMENT_GUIDE.md` or run `python main.py --debug` for detailed logging.