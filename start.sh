#!/bin/bash

# SweatHogs Medical AI Prediction System Startup Script
# Humber College Capstone Project
# 
# This script provides an easy way to start the medical prediction system
# with proper environment setup and error checking.

echo "=========================================="
echo "SweatHogs Medical AI Prediction System"
echo "Humber College Capstone Project"
echo "Team: Gurmat Singh, Minh Nhat Mai, Yuvraj Grover, Robert Seibel, Mohammed Hasnain Ali"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if virtual environment exists and create if needed
if [ ! -d "my_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv my_env
    echo "Virtual environment created successfully"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source my_env/bin/activate

# Check if dependencies need to be installed
echo "Checking and installing dependencies..."
if ! python3 -c "import flask" &> /dev/null; then
    echo "Installing Python packages from requirements.txt..."
    pip install -r requirements.txt
    echo "Dependencies installed successfully"
else
    echo "Dependencies already installed"
fi

# Check if .env file exists
if [[ ! -f ".env" ]]; then
    echo ""
    echo "Warning: .env file not found"
    echo "AI insights will not be available without GEMINI_API_KEY"
    echo "Create a .env file with your API keys for full functionality"
    echo ""
fi

# Check if required files exist
if [[ ! -f "models/best_model.pkt" ]]; then
    echo "Error: Model file not found at models/best_model.pkt"
    echo "Please ensure the trained model file exists"
    exit 1
fi

if [[ ! -f "data/processed/medical_data.pkl" ]]; then
    echo "Error: Processed data file not found at data/processed/medical_data.pkl"
    echo "Please ensure the preprocessed data file exists"
    exit 1
fi

echo ""
echo "Starting the medical prediction system..."
echo "Server will be available at: http://localhost:8080"
echo ""
echo "Available endpoints:"
echo "  - GET  /health         - System health check"
echo "  - POST /predict        - Get prediction with AI insights (JSON response)"
echo "  - POST /predict_with_report - Generate and download PDF report"
echo ""
echo "Example usage:"
echo "  For JSON: curl -X POST http://localhost:8080/predict -H 'Content-Type: application/json' -d '{...}'"
echo "  For PDF:  curl -X POST http://localhost:8080/predict_with_report -H 'Content-Type: application/json' -d '{...}' --output report.pdf"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the main application
python3 app.py
