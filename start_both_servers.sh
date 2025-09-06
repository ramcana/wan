#!/bin/bash

echo "Starting WAN22 React + FastAPI Application"
echo ""

echo "Starting FastAPI Backend Server..."
cd backend
python start_server.py &
BACKEND_PID=$!
cd ..

echo "Waiting 3 seconds for backend to start..."
sleep 3

echo "Starting React Frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "Both servers are starting..."
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user interrupt
trap 'kill $BACKEND_PID $FRONTEND_PID; exit' INT
wait