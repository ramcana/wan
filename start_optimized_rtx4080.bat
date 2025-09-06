@echo off
echo 🚀 Starting Wan2.2 Optimized for RTX 4080
echo ==========================================

echo.
echo 🔧 System Specifications:
echo • GPU: RTX 4080 (16GB VRAM)
echo • CPU: Threadripper PRO 5995WX (128 threads)
echo • RAM: 128GB
echo.

echo 📋 Optimizations Applied:
echo • VRAM Limit: 14GB (2GB buffer)
echo • Quantization: bf16
echo • CPU Offload: Disabled
echo • VAE Tiling: 512px
echo • Max Duration: 10 seconds
echo • Max Resolution: 2560x1440
echo.

echo 🎯 Recommended Settings:
echo • Resolution: 1920x1080 for best balance
echo • Duration: 4-8 seconds
echo • Steps: 25-50 for quality
echo.

echo ⚡ Starting Backend Server...
cd /d "%~dp0backend"
start "WAN2.2 Backend (RTX 4080)" cmd /k "python start_server.py --host 127.0.0.1 --port 9000"

echo.
echo ⏳ Waiting 5 seconds for backend to initialize...
timeout /t 5 /nobreak > nul

echo.
echo 🎨 Starting Frontend Server...
cd /d "%~dp0frontend"
start "WAN2.2 Frontend (RTX 4080)" cmd /k "npm run dev"

echo.
echo ✅ Both servers starting...
echo.
echo 📖 Usage Instructions:
echo • Backend: http://127.0.0.1:9000
echo • Frontend: http://localhost:3000
echo • API Docs: http://127.0.0.1:9000/docs
echo • Performance Monitor: http://127.0.0.1:9000/api/v1/performance/status
echo.
echo 🔍 Monitoring:
echo • Watch VRAM usage in Task Manager
echo • Check generation logs in backend console
echo • Monitor performance at the API endpoint above
echo.
echo Press any key to exit this window...
pause > nul