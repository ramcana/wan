#!/bin/bash

# Stop the backend server
if [ -f backend.pid ]; then
    PID=$(cat backend.pid)
    echo "Stopping backend server (PID: $PID)..."
    kill $PID
    rm backend.pid
    echo "Backend server stopped."
else
    echo "No backend.pid file found. Backend may not be running."
    echo "You can also manually stop it with: pkill -f start_backend_simple.py"
fi