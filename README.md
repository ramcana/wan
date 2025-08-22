# WAN2.2 Video Generation System

A modern, functionally organized video generation system with FastAPI backend and React frontend.

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

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- CUDA-compatible GPU (recommended)

### Installation

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

#### Full Stack (Recommended)
```bash
python main.py --mode full
```

#### Backend Only
```bash
python main.py --mode backend
```

#### Frontend Only
```bash
python main.py --mode frontend
```

#### Legacy Gradio UI
```bash
python main.py --mode gradio
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
- **Real-time progress tracking**
- **Hardware optimization** for RTX 4080
- **Queue management** for batch processing
- **LoRA support** for model customization
- **Responsive UI** with dark/light themes

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

### Common Issues

1. **Import errors**: Run the import update script:
   ```bash
   python utils_new/update_imports.py
   ```

2. **CUDA/GPU issues**: Check hardware compatibility in system settings

3. **Model loading**: Ensure models are downloaded to the `models/` directory

### Logs

- Application logs: Check console output
- Error logs: `logs/` directory
- Database: `wan22_tasks.db`

## License

[License information]

## Support

[Support information]
