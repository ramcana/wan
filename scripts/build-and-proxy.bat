@echo off
REM Script to build frontend and start reverse proxy setup

echo Building frontend...
cd frontend
npm run build
cd ..

echo Starting services with reverse proxy...
docker-compose -f backend/docker-compose.yml -f docker-compose.proxy.yml up -d

echo Services started!
echo Access the application at: http://app.localhost