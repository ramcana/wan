# Wan2.2 React Frontend with FastAPI Backend

A modern web interface for the Wan2.2 video generation system, built with React and FastAPI.

## Project Structure

```
├── frontend/          # React frontend application
│   ├── src/
│   │   ├── components/    # Reusable UI components
│   │   ├── pages/         # Page components
│   │   ├── lib/           # Utilities and API client
│   │   └── ...
│   ├── package.json
│   └── vite.config.ts
├── backend/           # FastAPI backend application
│   ├── api/
│   │   └── routes/        # API route handlers
│   ├── models/            # Pydantic models
│   ├── database.py        # Database configuration
│   ├── main.py           # FastAPI application
│   └── requirements.txt
└── README.md
```

## Features

- **Modern React UI**: Built with React 18, TypeScript, and Tailwind CSS
- **Professional Design**: Using Radix UI components for accessibility and consistency
- **Real-time Updates**: HTTP polling for task progress and system stats
- **Responsive Layout**: Works on desktop, tablet, and mobile devices
- **FastAPI Backend**: RESTful API with automatic documentation
- **SQLite Database**: Persistent task storage and system stats
- **File Management**: Image upload and video output handling
- **Error Handling**: Comprehensive error handling with user-friendly messages

## Quick Start

### Prerequisites

- Node.js 18+ and npm/yarn
- Python 3.8+
- Existing Wan2.2 system files (utils.py, etc.)

### Backend Setup

1. Navigate to the backend directory:

   ```bash
   cd backend
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy environment file and configure:

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Initialize the database:

   ```bash
   python init_db.py
   ```

6. Start the FastAPI server:
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000` with documentation at `http://localhost:8000/docs`.

### Frontend Setup

1. Navigate to the frontend directory:

   ```bash
   cd frontend
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Copy environment file and configure:

   ```bash
   cp .env.example .env
   # Edit .env with your API URL
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

The frontend will be available at `http://localhost:3000`.

## Development

### Backend Development

- **API Documentation**: Available at `http://localhost:8000/docs`
- **Database**: SQLite database file will be created as `wan22_tasks.db`
- **File Uploads**: Images stored in `uploads/` directory
- **Generated Videos**: Stored in `outputs/` directory

### Frontend Development

- **Hot Reload**: Vite provides fast hot module replacement
- **API Proxy**: Configured to proxy `/api` requests to the backend
- **TypeScript**: Full type safety with TypeScript
- **Linting**: ESLint configured for code quality

### API Endpoints

#### Generation

- `POST /api/v1/generate` - Create new generation task
- `GET /api/v1/generate/{task_id}` - Get task information

#### Queue Management

- `GET /api/v1/queue` - Get queue status and all tasks
- `POST /api/v1/queue/{task_id}/cancel` - Cancel a task
- `DELETE /api/v1/queue/{task_id}` - Delete a task
- `POST /api/v1/queue/clear` - Clear completed tasks

#### System Monitoring

- `GET /api/v1/system/stats` - Get system resource statistics
- `GET /api/v1/system/health` - Get system health status
- `GET /api/v1/system/optimization` - Get optimization settings
- `POST /api/v1/system/optimization` - Update optimization settings

#### Outputs

- `GET /api/v1/outputs` - List generated videos
- `GET /api/v1/outputs/{video_id}` - Get video information
- `GET /api/v1/outputs/{video_id}/download` - Download video
- `DELETE /api/v1/outputs/{video_id}` - Delete video

## Building for Production

### Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend

```bash
cd frontend
npm run build
```

The built files will be in the `frontend/dist` directory.

## Integration with Existing System

This frontend is designed to integrate with the existing Wan2.2 Python system:

1. **Model Management**: Uses existing `utils.py` functions for model loading
2. **Generation**: Integrates with existing generation pipeline
3. **Configuration**: Reads from existing `config.json`
4. **Error Handling**: Uses existing error handling system

## Next Steps

1. **Complete Backend Integration**: Connect API endpoints to existing generation system
2. **Add WebSocket Support**: For real-time progress updates
3. **Implement Prompt Enhancement**: Add prompt enhancement features
4. **Add LoRA Support**: Implement LoRA file management
5. **Performance Optimization**: Add caching and optimization features

## Contributing

1. Follow the existing code style and conventions
2. Add tests for new features
3. Update documentation as needed
4. Test on multiple browsers and devices

## License

This project is part of the Wan2.2 video generation system.
