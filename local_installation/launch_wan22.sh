#!/bin/bash

echo "========================================"
echo "WAN2.2 Video Generation System"
echo "========================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to installation directory
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run the installer again."
    echo ""
    read -p "Press any key to continue..."
    exit 1
fi

# Activate virtual environment
echo "Activating environment..."
source "venv/bin/activate"

# Check if main application exists
if [ ! -f "application/main.py" ]; then
    echo "Error: Main application not found!"
    echo "Please run the installer again."
    echo ""
    read -p "Press any key to continue..."
    exit 1
fi

# Launch main application
echo "Starting WAN2.2..."
echo ""
python "application/main.py"

# Check exit status
if [ $? -ne 0 ]; then
    echo ""
    echo "An error occurred. Check the logs for details."
    read -p "Press any key to close..."
fi