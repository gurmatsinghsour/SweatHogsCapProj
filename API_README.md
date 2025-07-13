# Medical Readmission Prediction API

A Flask-based API server that uses a deep learning CNN model (based on Gurmat's work) to predict patient readmission risk using the `best_model.pkt` trained model.

## Features

- **Deep Learning Model**: Uses a Single Shot CNN trained on medical data
- **REST API**: Easy-to-use JSON API endpoints
- **Production Ready**: Includes Gunicorn configuration for deployment
- **CORS Enabled**: Can be called from web applications
- **Comprehensive Predictions**: Returns confidence scores and risk levels

## Quick Start

### 1. Environment Setup

```bash
# Activate the conda environment
conda activate aienv

# Install required packages (if not already installed)
pip install flask flask-cors gunicorn requests
```

### 2. Start the Server

#### Development Mode

```bash
python prediction_server.py
```

#### Production Mode

```bash
python deploy_server.py
```

The server will start on `http://localhost:8080` by default.

### 3. Test the API

```bash
python test_api.py
```

## API Endpoints

### Health Check

```
GET /health
```

Returns server health status and model loading status.

### Model Information

```
GET /model-info
```

Returns information about the loaded model.

### Prediction

```
POST /predict
```

Predicts readmission risk for a patient.

## Sample Request

```json
POST /predict
Content-Type: application/json

{
    "encounter_id": 2278392,
    "patient_nbr": 8222157,
    "race": "Caucasian",
    "gender": "Female",
    "age": "[0-10)",
    "weight": "?",
    "admission_type_id": 6,
    "discharge_disposition_id": 25,
    "admission_source_id": 1,
    "time_in_hospital": 1,
    "payer_code": "?",
    "medical_specialty": "Pediatrics-Endocrinology",
    "num_lab_procedures": 41,
    "num_procedures": 0,
    "num_medications": 1,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "diag_1": "250.83",
    "diag_2": "?",
    "diag_3": "?",
    "number_diagnoses": 1,
    "max_glu_serum": "None",
    "A1Cresult": "None",
    "metformin": "No",
    "repaglinide": "No",
    "nateglinide": "No",
    "chlorpropamide": "No",
    "glimepiride": "No",
    "acetohexamide": "No",
    "glipizide": "No",
    "glyburide": "No",
    "tolbutamide": "No",
    "pioglitazone": "No",
    "rosiglitazone": "No",
    "acarbose": "No",
    "miglitol": "No",
    "troglitazone": "No",
    "tolazamide": "No",
    "examide": "No",
    "citoglipton": "No",
    "insulin": "No",
    "glyburide-metformin": "No",
    "glipizide-metformin": "No",
    "glimepiride-pioglitazone": "No",
    "metformin-rosiglitazone": "No",
    "metformin-pioglitazone": "No",
    "change": "No",
    "diabetesMed": "No",
    "readmitted": "NO"
}
```

## Sample Response

```json
{
  "status": "success",
  "patient_id": 8222157,
  "encounter_id": 2278392,
  "prediction": {
    "readmission_risk": 1,
    "confidence_score": 1.0,
    "risk_level": "HIGH",
    "detailed_scores": {
      "probability_not_readmitted": 0.0,
      "probability_readmitted": 1.0
    }
  },
  "model_info": {
    "model_type": "Deep Learning CNN",
    "based_on": "Gurmat_work",
    "features_used": 183
  }
}
```

## Risk Levels

- **LOW**: Confidence score ≤ 0.3
- **MEDIUM**: Confidence score 0.3 - 0.7
- **HIGH**: Confidence score > 0.7

## Deployment Options

### Local Deployment

The server runs locally and can be accessed on your network.

### Cloud Deployment (Recommended)

#### Option 1: Heroku

1. Create a Heroku app
2. Add the files to your repository
3. Deploy with Heroku's git integration
4. Set environment variables as needed

#### Option 2: Railway

1. Connect your GitHub repository to Railway
2. Railway will automatically detect the Python app
3. Set PORT environment variable if needed

#### Option 3: Google Cloud Run

1. Build a Docker container
2. Deploy to Cloud Run
3. Automatically scales and provides HTTPS

### Environment Variables

- `PORT`: Server port (default: 8080)
- `WORKERS`: Number of Gunicorn workers (default: 4)
- `DEBUG`: Enable debug mode (default: False)

## File Structure

```
├── prediction_server.py      # Main Flask application
├── deploy_server.py          # Production deployment script
├── test_api.py              # API testing script
├── requirements.txt         # Python dependencies
├── Models/
│   └── deepLearning/
│       └── best_model.pkt   # Trained model file
└── dataPreprocessing/
    └── medical_data.pkl     # Training data for preprocessing
```

## Model Details

- **Type**: Single Shot Convolutional Neural Network (CNN)
- **Input Features**: 183 features after preprocessing
- **Classes**: 2 (readmitted vs not readmitted)
- **Training**: Based on Gurmat's work with genetic algorithm feature selection
- **Device**: Automatically detects CUDA, MPS, or CPU

## Security Notes

⚠️ **Important**: This is a development/demo server. For production deployment:

1. Add authentication and authorization
2. Implement rate limiting
3. Add input validation and sanitization
4. Use HTTPS
5. Monitor and log all requests
6. Consider data privacy and HIPAA compliance

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the PORT environment variable
2. **Model not found**: Ensure `Models/deepLearning/best_model.pkt` exists
3. **Data not found**: Ensure `dataPreprocessing/medical_data.pkl` exists
4. **Memory issues**: Reduce number of workers or use CPU instead of GPU

### Logs

The server logs important information including:

- Model loading status
- Prediction requests and responses
- Error messages and stack traces

## Contact

For questions about the model or API, please refer to Gurmat's work documentation in `Models/deepLearning/Gurmat_work.ipynb`.
