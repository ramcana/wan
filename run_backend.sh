#!/bin/bash

# Activate virtual environment and start backend
source venv/bin/activate
echo "Starting WAN22 Backend Server..."
echo "Backend will be available at: http://localhost:9000"
echo "API Documentation: http://localhost:9000/docs"
echo "Press Ctrl+C to stop the server"
echo ""

python3 start_backend_simple.py