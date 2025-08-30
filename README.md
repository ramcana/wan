# WAN2.2 Video Generation System

A modern, functionally organized video generation system with FastAPI backend and React frontend.

## 🚀 Quick Start (New Users Start Here!)

### The Simple Way - Just One Click!

1. **Double-click** `start.bat` in the project folder
2. **Wait** for setup to complete (first time takes a few minutes)
3. **Open your browser** to http://localhost:3000 when ready

**Or from command line:**

```bash
python start.py
```

That's it! The script automatically:

- ✅ Checks your system requirements
- ✅ Installs missing dependencies
- ✅ Starts both servers
- ✅ Opens your browser

**Having issues?** See [QUICK_START.md](QUICK_START.md) for troubleshooting.

---

## Architecture Overview

The codebase has been functionally organized into the following structure:

```
wan2.2/
├── backend/                    # FastAPI backend
│   ├── api/v1/endpoints/      # API endpoints
│   ├── repositories/          # Data access layer
│   ├── schemas/              # API schemas
│   └── services/             # Business logic services
├── core/                      # Core domain logic
│   ├── models/               # Domain models
│   ├── services/             # Core services
│   └── interfaces/           # Abstract interfaces
├── infrastructure/           # Infrastructure layer
│   ├── config/              # Configuration management
│   ├── storage/             # File and model storage
│   └── hardware/            # Hardware monitoring & optimization
├── frontend/                 # React frontend
│   └── src/                 # Frontend source code
├── utils_new/               # Utility functions
├── docs/                    # Documentation
└── scripts/                 # Build and deployment scripts
```

## Advanced Installation (For Developers)

### Prerequisites

- Python 3.8+
- Node.js 16+
- CUDA-compatible GPU (recommended)

### Manual Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd wan2.2
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

### Running the Application

#### Intelligent Startup Manager (Recommended)

```bash
# Enhanced startup with automatic port management and error recovery
start_both_servers.bat

# With verbose output for troubleshooting
start_both_servers.bat --verbose

# Force basic mode if needed
start_both_servers.bat --basic
```

#### Alternative Startup Methods

```bash
# Full stack using main.py
python main.py --mode full

# Backend only
python main.py --mode backend

# Frontend only
python main.py --mode frontend

# Legacy Gradio UI
python main.py --mode gradio
```

#### Manual Startup (Advanced Users)

```bash
# Backend server
cd backend
python start_server.py

# Frontend server (separate terminal)
cd frontend
npm run dev
```

### Access Points

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Gradio UI**: http://localhost:7860

## Development

### Project Structure

#### Backend (`/backend/`)

- **FastAPI** application with async support
- **SQLAlchemy** for database operations
- **Pydantic** for data validation
- **Background task processing** for video generation

#### Core (`/core/`)

- **Domain models** and business logic
- **Service layer** for core functionality
- **Interfaces** for dependency injection

#### Infrastructure (`/infrastructure/`)

- **Configuration management** with environment support
- **Hardware monitoring** and optimization
- **Storage management** for models and outputs

#### Frontend (`/frontend/`)

- **React** with TypeScript
- **Tailwind CSS** for styling
- **React Query** for API state management
- **Accessibility** and offline support

### Key Features

- **Multi-model support**: T2V-A14B, I2V-A14B, TI2V-5B
- **Intelligent startup management**: Automatic port conflict resolution, environment validation, and error recovery
- **Real-time progress tracking**
- **Hardware optimization** for RTX 4080
- **Queue management** for batch processing
- **LoRA support** for model customization
- **Responsive UI** with dark/light themes
- **Performance monitoring**: Startup metrics, resource usage tracking, and optimization suggestions

### Configuration

Configuration is managed through:

- `infrastructure/config/config.json` - Main configuration
- Environment variables for sensitive data
- Runtime configuration through the UI

### Testing

```bash
# Backend tests
cd backend
python -m pytest

# Frontend tests
cd frontend
npm test
```

### Contributing

1. Follow the functional organization structure
2. Update imports when moving files
3. Add tests for new functionality
4. Update documentation

## Troubleshooting

### Startup Issues

1. **Port conflicts**: The startup manager automatically resolves port conflicts

   ```bash
   # Use verbose mode to see port resolution details
   start_both_servers.bat --verbose
   ```

2. **Environment issues**: Run diagnostic mode for comprehensive system check

   ```bash
   python scripts/startup_manager.py --diagnostics
   ```

3. **Permission errors**: Run as administrator or use basic mode
   ```bash
   start_both_servers.bat --basic
   ```

### Common Issues

1. **Import errors**: Run the import update script:

   ```bash
   python utils_new/update_imports.py
   ```

2. **CUDA/GPU issues**: Check hardware compatibility in system settings

3. **Model loading**: Ensure models are downloaded to the `models/` directory

### Startup Manager Documentation

For detailed information about the intelligent startup system:

- **Integration Guide**: `docs/STARTUP_MANAGER_INTEGRATION_GUIDE.md`
- **Migration Guide**: `docs/STARTUP_MANAGER_MIGRATION_GUIDE.md`
- **User Guide**: `docs/STARTUP_MANAGER_USER_GUIDE.md`
- **Developer Guide**: `docs/STARTUP_MANAGER_DEVELOPER_GUIDE.md`

### Logs

- Application logs: Check console output
- Startup logs: `logs/startup_*.log`
- Error logs: `logs/` directory
- Database: `wan22_tasks.db`

## License

[License information]

## Support

[Support information]
