#!/bin/bash

# Start backend in background
echo "Starting WAN22 Backend in background..."

# Activate virtual environment and start backend
source venv/bin/activate

# Start the backend in background and save PID
nohup python3 start_backend_simple.py > backend.log 2>&1 &
BACKEND_PID=$!

echo "Backend started with PID: $BACKEND_PID"
echo "Backend running at: http://localhost:9000"
echo "API docs at: http://localhost:9000/docs"
echo "Logs are being written to: backend.log"
echo "To stop the backend, run: kill $BACKEND_PID"
echo ""

# Save PID to file for easy stopping later
echo $BACKEND_PID > backend.pid
echo "Backend PID saved to backend.pid"