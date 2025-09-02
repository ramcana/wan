# 🚀 WAN22 Quick Start Guide

## The Simple Way (Recommended for Everyone)

### Option 1: Desktop shortcut (easiest!)

1. **Double-click** `create_desktop_shortcut.bat` (one-time setup)
2. **Double-click** the WAN22 icon on your desktop anytime
3. **Wait** for setup to complete (first time takes longer)
4. **Open your browser** to the URL shown (usually http://localhost:3000)

### Option 2: Direct startup

1. **Double-click** `start.bat` in the project folder
2. **Wait** for the setup to complete (first time takes longer)
3. **Open your browser** to the URL shown (usually http://localhost:3000)

### Option 3: Command line

```bash
python start.py
```

That's it! The script handles everything automatically:

- ✅ Checks if you have Python and Node.js installed
- ✅ Installs missing dependencies
- ✅ Finds available ports if defaults are busy
- ✅ Starts both backend and frontend servers
- ✅ Opens your browser automatically
- ✅ Shows clear status messages

## What You'll See

```
==================================================
🚀 WAN22 Video Generation System
==================================================
📋 Checking requirements...
✅ Python version OK
✅ Project structure OK
✅ Node.js found

📦 Checking dependencies...
✅ Backend dependencies OK
✅ Frontend dependencies OK

🔧 Starting backend server on port 8000...
✅ Backend running at http://localhost:8000

⚛️  Starting frontend server on port 3000...
✅ Frontend running at http://localhost:3000

==================================================
🎉 WAN22 is now running!
==================================================
🌐 Open your browser to: http://localhost:3000
📚 API docs available at: http://localhost:8000/docs

💡 Tips:
   • Keep this window open while using WAN22
   • Press Ctrl+C to stop both servers
   • Check the other terminal windows if you see errors
==================================================
```

## Troubleshooting

### Common Issues (Quick Fixes)

#### "Python not found"

- Install Python 3.8-3.11 from [python.org](https://python.org)
- ✅ Check "Add Python to PATH" during installation
- Restart terminal after installation

#### "Node.js not found"

- Install Node.js 16-20 LTS from [nodejs.org](https://nodejs.org)
- Restart terminal after installation

#### "Permission denied" errors

- Right-click `start.bat` → "Run as administrator"
- Add firewall exceptions for Python and Node.js

#### Ports are busy

- The script automatically finds available ports
- Check output to see which ports are being used
- Kill processes: `netstat -ano | findstr :8000`

#### Slow performance or crashes

- Check system requirements (8+ GB RAM, GPU recommended)
- Close other resource-intensive applications
- Monitor temperatures and ensure adequate cooling

### System Requirements

**Minimum:**

- Windows 10/11, Python 3.8+, Node.js 16+
- 8 GB RAM, 50 GB free space
- Any GPU or CPU-only mode

**Recommended:**

- 16+ GB RAM, SSD storage
- NVIDIA RTX 3070+ or RTX 4080 (optimized)
- See [SYSTEM_REQUIREMENTS.md](SYSTEM_REQUIREMENTS.md) for details

### Need More Help?

📖 **Comprehensive guides:**

- [SYSTEM_REQUIREMENTS.md](SYSTEM_REQUIREMENTS.md) - Detailed hardware/software requirements
- [COMPREHENSIVE_TROUBLESHOOTING.md](COMPREHENSIVE_TROUBLESHOOTING.md) - Step-by-step problem solving
- [STARTUP_MIGRATION.md](STARTUP_MIGRATION.md) - Migration from other startup methods

🔧 **Quick diagnostic:**

```bash
python start.py --diagnostics  # Run system check
```

💡 **Before asking for help:**

- Run the diagnostic command above
- Check the comprehensive troubleshooting guide
- Include error messages and system info

## For Advanced Users

If you need more control, you can still use the advanced startup methods:

- `start_both_servers.bat` - Full startup manager with all features
- `main.py` - Application entry point with multiple modes
- Manual startup - See `START_SERVERS.md` for detailed instructions

But for 99% of users, just use `start.bat` or `python start.py`!
