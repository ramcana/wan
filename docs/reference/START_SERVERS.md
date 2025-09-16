---
category: reference
last_updated: '2025-09-15T22:50:00.490365'
original_path: reports\START_SERVERS.md
tags:
- installation
- api
- troubleshooting
title: How to Start the WAN22 React + FastAPI Application
---

# How to Start the WAN22 React + FastAPI Application

## Prerequisites

- Python 3.8+ installed
- Node.js 16+ installed
- All dependencies installed

## Step 1: Start the FastAPI Backend Server

Open a terminal in the project root and run:

```bash
cd backend
python start_server.py
```

Or alternatively:

```bash
cd backend
python app.py
```

The backend server will start on: **http://localhost:8000**

- API Documentation: http://localhost:8000/docs
- WebSocket endpoint: ws://localhost:8000/ws

## Step 2: Start the React Frontend

Open a **second terminal** in the project root and run:

```bash
cd frontend
npm run dev
```

The frontend will start on: **http://localhost:3000**

## Step 3: Test the Connection

1. Open your browser to http://localhost:3000
2. You should see the React UI load without connection errors
3. The WebSocket indicator should show "Live Updates" (green)
4. Try submitting a video generation request

## Troubleshooting

### Backend Issues

- **Port 8000 already in use**: Kill any existing processes on port 8000
- **Import errors**: Make sure you're in the `backend` directory when running the server
- **Missing dependencies**: Run `pip install -r requirements.txt` in the backend directory

### Frontend Issues

- **Connection errors**: Make sure the backend server is running first
- **WebSocket failures**: Check that port 8000 is accessible
- **Form submission errors**: Check browser console for detailed error messages

### Both Servers Must Be Running

The React frontend (port 3000) communicates with the FastAPI backend (port 8000). Both must be running simultaneously for the application to work properly.
