#!/usr/bin/env python3
"""
SweatHogs Medical AI Prediction System
Humber College Capstone Project

This is the main application file that serves as the entry point for the medical
readmission prediction system. It initializes the machine learning model and
starts the Flask web server.

Team Members:
- Gurmat Singh: Deep Learning & Model Development
- Minh Nguyen: Data Preprocessing & Feature Engineering  
- Yuvraj Patel: Machine Learning & Model Optimization
- Robert Johnson: Data Analysis & Model Validation

Usage:
    python app.py

The server will start on localhost:8080 by default and provide REST API endpoints
for medical prediction analysis.
"""

import sys
import os

# Add the src directory to Python path so we can import our modules
# This allows us to keep the main app.py file in the root while organizing
# source code in the src/ subdirectory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the Flask application and initialization functions from our prediction server
from prediction_server import app, initialize_predictor, logger

if __name__ == '__main__':
    # Display startup banner with project information
    print("SweatHogs Medical AI Prediction System")
    print("=" * 50)
    print("Humber College Capstone Project")
    print("Team: Gurmat Singh, Minh Nguyen, Yuvraj Patel, Robert Johnson")
    print("=" * 50)
    
    # Initialize the machine learning model and preprocessing pipeline
    # This loads the trained PyTorch model and sets up feature transformation
    try:
        print("Initializing AI model...")
        initialize_predictor()
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        print("Please ensure the model file exists in models/best_model.pkt")
        print("and the preprocessed data exists in data/processed/medical_data.pkl")
        sys.exit(1)
    
    # Configure server settings from environment variables or use defaults
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # Display server information and available API endpoints
    print(f"Starting server on http://localhost:{port}")
    print("\nAvailable Endpoints:")
    print(f"   • GET  /               - Web interface")
    print(f"   • GET  /health         - Health check")
    print(f"   • POST /predict        - Get prediction + remedy (JSON response)")
    print(f"   • POST /predict_with_report - Generate and download PDF report")
    print(f"   • GET  /download_report/<filename> - Download previously generated PDF")
    
    # Show example usage for testing the API
    print("\nExample cURL commands:")
    print("For JSON response with confidence score and AI insights:")
    print(f"""curl -X POST http://localhost:{port}/predict \\
  -H "Content-Type: application/json" \\
  -d '{{
    "age": "[50-60)",
    "gender": "Female",
    "time_in_hospital": 5,
    "admission_type": 1,
    "discharge_disposition": 1,
    "admission_source": 1,
    "num_medications": 10,
    "num_lab_procedures": 30,
    "num_procedures": 2,
    "number_diagnoses": 3,
    "number_inpatient": 1,
    "number_outpatient": 2,
    "number_emergency": 0,
    "diabetesMed": "Yes",
    "change": "Ch",
    "A1Cresult": "Norm",
    "max_glu_serum": "None",
    "insulin": "No",
    "metformin": "Steady",
    "diagnosis_1": "250.00"
  }}'""")
    
    print(f"\nFor PDF report generation and download:")
    print(f"""curl -X POST http://localhost:{port}/predict_with_report \\
  -H "Content-Type: application/json" \\
  -d '{{
    "age": "[50-60)",
    "gender": "Female",
    "time_in_hospital": 5,
    "admission_type": 1,
    "discharge_disposition": 1,
    "admission_source": 1,
    "num_medications": 10,
    "num_lab_procedures": 30,
    "num_procedures": 2,
    "number_diagnoses": 3,
    "number_inpatient": 1,
    "number_outpatient": 2,
    "number_emergency": 0,
    "diabetesMed": "Yes",
    "change": "Ch",
    "A1Cresult": "Norm",
    "max_glu_serum": "None",
    "insulin": "No",
    "metformin": "Steady",
    "diagnosis_1": "250.00"
  }}' \\
  --output medical_report.pdf""")
    
    print("\n" + "=" * 50)
    
    # Start the Flask web server
    # The server will handle HTTP requests and provide the prediction API
    try:
        app.run(host='0.0.0.0', port=port, debug=debug)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nServer error: {e}")
        sys.exit(1)
