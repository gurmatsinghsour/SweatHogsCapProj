# SweatHogs Medical AI Prediction System
**Humber College Capstone Project - Computer Programming**

**Team Members:** 
- Gurmat Singh - Deep Learning & Model Development
- Minh Nhat Mai - Data Preprocessing & Feature Engineering  
- Yuvraj Grover - Machine Learning & Model Optimization
- Robert Seibel - Data Analysis & Model Validation
- Muhammed Hasnainali Saiyed - Data Preprocessing & Machine Learning

## Project Overview

This is a comprehensive medical readmission prediction system developed as a capstone project for Humber College's Computer Programming program. The system uses advanced deep learning techniques to predict patient readmission risk and integrates with Google's Gemini AI to generate detailed medical insights and recommendations.

### Key Features

- **Deep Learning Model**: Single-Shot Convolutional Neural Network for readmission prediction
- **AI Integration**: Google Gemini API for generating medical insights and recommendations
- **Professional Reports**: Automated PDF report generation with institutional branding
- **REST API**: Clean RESTful endpoints for JSON and PDF responses
- **Web Interface**: User-friendly web form for data input and testing
- **Comprehensive Documentation**: Extensive code documentation and usage examples

### Technology Stack

- **Backend**: Python Flask web framework
- **Machine Learning**: PyTorch for deep learning model implementation
- **Data Processing**: Pandas, NumPy, Scikit-learn for data preprocessing
- **AI Integration**: Google Generative AI (Gemini) for natural language insights
- **Report Generation**: ReportLab for professional PDF creation
- **Visualization**: Matplotlib for charts and graphs
- **Frontend**: HTML/CSS for web interface

## Project Structure

The project is organized into a clean, professional structure suitable for deployment and presentation:

```
SweatHogsCapProj/
├── app.py                          # MAIN APPLICATION - Run this to start the server
├── requirements.txt                # Python package dependencies
├── .env                           # Environment variables (API keys) - create this file
├── README.md                      # This documentation file
├── 
├── src/                           # Source code directory
│   ├── prediction_server.py      # Main Flask API server with prediction logic
│   └── pdf_report_generator.py   # Professional PDF report generation
├── 
├── notebooks/                     # Research and development notebooks
│   ├── data_preprocessing/        # Data cleaning and feature engineering notebooks
│   ├── machine_learning/          # Traditional ML model experiments
│   └── deep_learning/            # Neural network development and training
├── 
├── data/                         # Data storage directory
│   ├── raw/                     # Original CSV datasets from healthcare sources
│   └── processed/               # Preprocessed data files (pickle format)
├── 
├── models/                       # Trained model storage
│   └── best_model.pkt           # Best performing trained model file
├── 
├── reports/                      # Generated report storage
│   └── generated_pdfs/          # Automatically generated PDF reports
├── 
├── tests/                        # Test and validation files
│   ├── simple_test.py           # Basic API endpoint testing
│   └── test_pdf_report.py       # PDF generation functionality tests
├── 
└── static/                       # Web interface assets
    └── index.html               # Frontend web interface for user interaction
```

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for AI API calls)

### Step 1: Install Dependencies

Install all required Python packages using the requirements file:

```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables

Create a `.env` file in the project root directory with your API keys:

```bash
# Google Gemini AI API Key for medical insights generation
GEMINI_API_KEY=your_gemini_api_key_here

# Hugging Face API Token (optional, for alternative AI models)
HF_API_TOKEN=your_huggingface_token_here
```

**How to get a Gemini API Key:**
1. Visit [Google AI Studio](https://makersuite.google.com/)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key to your `.env` file

### Step 3: Run the Application

Start the medical prediction system with a single command:

```bash
python app.py
```

The server will start and display:
- System initialization status
- Available API endpoints
- Example usage commands
- Server URL: `http://localhost:8080`

### Step 4: Test the System

You can test the system using the provided web interface or API endpoints:

**Web Interface:** Open `http://localhost:8080` in your browser

**API Testing:** Use the provided cURL example or any HTTP client

## API Documentation

### Endpoint Overview

The system provides three main endpoints for different use cases:

| Endpoint | Method | Purpose | Response Format |
|----------|--------|---------|----------------|
| `/health` | GET | System status check | JSON |
| `/predict` | POST | Get prediction with AI insights | JSON |
| `/predict_with_report` | POST | Generate PDF report | PDF file |

### 1. Health Check Endpoint

**URL:** `GET /health`

**Purpose:** Verify that the system is running and the model is loaded

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "timestamp": "2025-07-14T10:30:00.000Z"
}
```

### 2. Prediction Endpoint

**URL:** `POST /predict`

**Purpose:** Get readmission risk prediction with AI-generated medical insights

**Request Body:** JSON with patient information

##  API Endpoints

### 1. Basic Prediction (JSON Response)
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Response:**
```json
{
  "status": "success",
  "confidence_score": 0.742,
  "remedy": "AI-generated medical insights and recommendations..."
}
```

### 2. Prediction with PDF Report
```bash
curl -X POST http://localhost:8080/predict_with_report \
  -H "Content-Type: application/json" \
  -d '{...same data as above...}'
```

**Response:**
```json
{
  "status": "success",
  "confidence_score": 0.742,
  "remedy": "AI-generated medical insights...",
  "pdf_filename": "medical_report_20250714_164135.pdf",
  "pdf_available": true
}
```

### 3. Download PDF Report
```bash
curl -o report.pdf http://localhost:8080/download_report/medical_report_20250714_164135.pdf
```

##  Technical Stack
- **Backend:** Flask, Python 3.8+
- **AI/ML:** PyTorch, scikit-learn, Google Gemini API
- **PDF Generation:** ReportLab, Matplotlib
- **Data Processing:** Pandas, NumPy

##  Model Information
- **Architecture:** Single-Shot CNN with genetic algorithm feature selection
- **Input Features:** 188+ medical and demographic features
- **Output:** Readmission risk probability (0-1) with calibrated confidence
- **Performance:** Optimized using Optuna hyperparameter tuning

##  Medical Features
The system analyzes:
- Patient demographics (age, gender)
- Hospital stay characteristics
- Medical procedures and lab tests
- Medication history and changes
- Previous healthcare utilization
- Diagnosis codes and medical conditions

## 📋 Sample Patient Data Format
```json
{
  "age": "[50-60)",                    // Age bracket
  "gender": "Female",                  // Gender
  "time_in_hospital": 5,              // Days in hospital
  "admission_type": 1,                // 1=Emergency, 2=Urgent, 3=Elective
  "discharge_disposition": 1,         // Discharge destination code
  "admission_source": 1,              // Admission source code
  "num_medications": 10,              // Number of medications
  "num_lab_procedures": 30,           // Number of lab procedures
  "num_procedures": 2,                // Number of medical procedures
  "number_diagnoses": 3,              // Number of diagnoses
  "number_inpatient": 1,              // Previous inpatient visits
  "number_outpatient": 2,             // Previous outpatient visits
  "number_emergency": 0,              // Previous emergency visits
  "diabetesMed": "Yes",               // Diabetes medication
  "change": "Ch",                     // Medication change
  "A1Cresult": "Norm",               // A1C test result
  "max_glu_serum": "None",           // Max glucose serum
  "insulin": "No",                    // Insulin usage
  "metformin": "Steady",             // Metformin usage
  "diagnosis_1": "250.00"            // Primary diagnosis code
}
```

##  Important Disclaimers
- This system is for educational and research purposes only
- NOT intended for actual medical diagnosis or treatment
- Always consult qualified healthcare professionals for medical decisions
- Predictions are based on statistical models and may not account for all factors

##  Development Team
- **Gurmat Singh** - Deep Learning & Model Development
- **Minh Nhat Mai** - Data Preprocessing & Feature Engineering  
- **Yuvraj Grover** - Machine Learning & Model Optimization
- **Robert Seibel** - Data Analysis & Model Validation

##  Academic Information
- **Institution:** Humber College Institute of Technology & Advanced Learning
- **Program:** Computer Programming
- **Project Type:** Capstone Project
- **Year:** 2025

## Notes
- Ensure paths are correct when accessing files across folders.
- Preprocessing must be completed before running any model notebooks.
