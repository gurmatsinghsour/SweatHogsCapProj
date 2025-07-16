# Medical Readmission Prediction API Documentation

## Overview

The Medical Readmission Prediction API is a Flask-based web service that provides machine learning predictions for patient readmission risk. The API uses a trained deep learning model (SingleShotCNN) to analyze patient data and return confidence scores indicating the likelihood of hospital readmission.

## Base Information

- **Base URL**: `http://localhost:8080` (development)
- **Content Type**: `application/json`
- **Response Format**: JSON
- **Model**: SingleShotCNN with temperature scaling for confidence calibration

## Endpoints

### 1. Health Check

**GET** `/health`

Check if the API server and model are running properly.

#### Response
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### Status Codes
- `200`: Service is healthy
- `500`: Service error

---

### 2. Prediction Endpoint

**POST** `/predict`

Predict patient readmission risk based on medical data.

#### Request Headers
```
Content-Type: application/json
```

#### Request Body

**Required Fields:**
```json
{
  "age": "[50-60)",
  "gender": "Male",
  "time_in_hospital": 5,
  "admission_type": 1,
  "discharge_disposition": 1,
  "admission_source": 7,
  "num_medications": 15,
  "num_lab_procedures": 25,
  "num_procedures": 3,
  "number_diagnoses": 5,
  "number_inpatient": 2,
  "number_outpatient": 0,
  "number_emergency": 1,
  "diabetesMed": "Yes",
  "change": "Ch",
  "A1Cresult": ">8",
  "max_glu_serum": "Norm",
  "insulin": "Steady",
  "metformin": "No",
  "diagnosis_1": "250"
}
```

#### Field Specifications

| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| `age` | string | Patient age range | `[0-10)`, `[10-20)`, `[20-30)`, `[30-40)`, `[40-50)`, `[50-60)`, `[60-70)`, `[70-80)`, `[80-90)`, `[90-100)` |
| `gender` | string | Patient gender | `Male`, `Female` |
| `time_in_hospital` | integer | Days in hospital | 1-14 |
| `admission_type` | integer | Type of admission | 1-8 (see mapping below) |
| `discharge_disposition` | integer | Discharge type | 1-30 (see mapping below) |
| `admission_source` | integer | Admission source | 1-26 (see mapping below) |
| `num_medications` | integer | Number of medications | 0-81 |
| `num_lab_procedures` | integer | Number of lab procedures | 0-132 |
| `num_procedures` | integer | Number of procedures | 0-6 |
| `number_diagnoses` | integer | Number of diagnoses | 1-16 |
| `number_inpatient` | integer | Previous inpatient visits | 0-21 |
| `number_outpatient` | integer | Previous outpatient visits | 0-42 |
| `number_emergency` | integer | Previous emergency visits | 0-76 |
| `diabetesMed` | string | Diabetes medication prescribed | `Yes`, `No` |
| `change` | string | Change in diabetes medication | `Ch`, `No` |
| `A1Cresult` | string | A1C test result | `>7`, `>8`, `Norm`, `None` |
| `max_glu_serum` | string | Max glucose serum level | `>200`, `>300`, `Norm`, `None` |
| `insulin` | string | Insulin prescription status | `Down`, `Steady`, `Up`, `No` |
| `metformin` | string | Metformin prescription status | `Down`, `Steady`, `Up`, `No` |
| `diagnosis_1` | string | Primary diagnosis code | ICD-9 3-digit code (e.g., "250", "428") |

#### Response

**Success (200):**
```json
{
  "status": "success",
  "confidence_score": 0.67
}
```

**Error (400):**
```json
{
  "error": "Missing required fields: age, gender",
  "status": "error",
  "required_fields": ["age", "gender", "time_in_hospital", ...]
}
```

**Error (500):**
```json
{
  "error": "Model not initialized",
  "status": "error"
}
```

---

### 3. Static Files

**GET** `/`

Serves the web interface from `static/index.html`.

## Code Mappings

### Admission Type IDs
| ID | Description |
|----|-------------|
| 1 | Emergency |
| 2 | Urgent |
| 3 | Elective |
| 4 | Newborn |
| 5 | Not Available |
| 6 | NULL |
| 7 | Trauma Center |
| 8 | Not Mapped |

### Discharge Disposition IDs
| ID | Description |
|----|-------------|
| 1 | Discharged to home |
| 2 | Discharged/transferred to another short term hospital |
| 3 | Discharged/transferred to SNF |
| 6 | Discharged/transferred to home with home health service |
| 7 | Left AMA |
| 11 | Expired |
| 18 | NULL |
| 25 | Not Mapped |

### Admission Source IDs
| ID | Description |
|----|-------------|
| 1 | Physician Referral |
| 2 | Clinic Referral |
| 4 | Transfer from a hospital |
| 7 | Emergency Room |
| 9 | Not Available |
| 17 | NULL |
| 20 | Not Mapped |

## Model Details

### Architecture
- **Model Type**: SingleShotCNN (Convolutional Neural Network)
- **Input Features**: 188 engineered features
- **Filters**: 64
- **Kernel Size**: 7
- **Dropout**: 0.20
- **Output Classes**: 2 (not readmitted, readmitted)

### Confidence Calibration
The model uses temperature scaling with the following techniques:
- **Temperature**: 5.0 (reduces overconfidence)
- **Noise Injection**: 0.1 scale for uncertainty quantification
- **Extreme Confidence Dampening**: Maps 95%+ confidence to 70-97% range
- **Typical Range**: 0.55 - 0.92 confidence scores

### Feature Engineering
The API automatically generates 188 features from the 19 input parameters:
- Medical condition indicators (35 features)
- Diagnosis frequency calculations (108 features)
- Patient history aggregations (20 features)
- Admission/discharge groupings (10 features)
- Risk combination indicators (15 features)

## Usage Examples

### Python Example
```python
import requests
import json

# Example patient data
patient_data = {
    "age": "[60-70)",
    "gender": "Female",
    "time_in_hospital": 7,
    "admission_type": 1,
    "discharge_disposition": 1,
    "admission_source": 7,
    "num_medications": 20,
    "num_lab_procedures": 35,
    "num_procedures": 2,
    "number_diagnoses": 8,
    "number_inpatient": 3,
    "number_outpatient": 2,
    "number_emergency": 1,
    "diabetesMed": "Yes",
    "change": "Ch",
    "A1Cresult": ">8",
    "max_glu_serum": "Norm",
    "insulin": "Up",
    "metformin": "Steady",
    "diagnosis_1": "250"
}

# Make prediction request
response = requests.post(
    'http://localhost:8080/predict',
    headers={'Content-Type': 'application/json'},
    data=json.dumps(patient_data)
)

result = response.json()
print(f"Confidence Score: {result['confidence_score']}")
```

### JavaScript Example
```javascript
const patientData = {
    age: "[40-50)",
    gender: "Male",
    time_in_hospital: 4,
    admission_type: 3,
    discharge_disposition: 1,
    admission_source: 1,
    num_medications: 12,
    num_lab_procedures: 20,
    num_procedures: 1,
    number_diagnoses: 4,
    number_inpatient: 1,
    number_outpatient: 0,
    number_emergency: 0,
    diabetesMed: "No",
    change: "No",
    A1Cresult: "Norm",
    max_glu_serum: "Norm",
    insulin: "No",
    metformin: "No",
    diagnosis_1: "414"
};

fetch('http://localhost:8080/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(patientData)
})
.then(response => response.json())
.then(data => {
    console.log('Confidence Score:', data.confidence_score);
})
.catch(error => {
    console.error('Error:', error);
});
```

### cURL Example
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": "[70-80)",
    "gender": "Female",
    "time_in_hospital": 10,
    "admission_type": 1,
    "discharge_disposition": 3,
    "admission_source": 7,
    "num_medications": 25,
    "num_lab_procedures": 45,
    "num_procedures": 4,
    "number_diagnoses": 9,
    "number_inpatient": 5,
    "number_outpatient": 3,
    "number_emergency": 2,
    "diabetesMed": "Yes",
    "change": "Ch",
    "A1Cresult": ">8",
    "max_glu_serum": ">200",
    "insulin": "Up",
    "metformin": "Up",
    "diagnosis_1": "428"
  }'
```

## Confidence Score Interpretation

| Score Range | Risk Level | Clinical Interpretation |
|-------------|------------|------------------------|
| 0.55 - 0.65 | Low Risk | Routine discharge planning |
| 0.65 - 0.75 | Medium Risk | Enhanced discharge planning recommended |
| 0.75 - 0.85 | High Risk | Intensive discharge planning and follow-up |
| 0.85 - 0.92 | Very High Risk | Consider extended care or frequent monitoring |

## Error Handling

### Common Error Scenarios

1. **Missing Required Fields (400)**
   - Ensure all 19 required fields are provided
   - Check field names for typos

2. **Invalid Field Values (400)**
   - Verify age format follows `[X-Y)` pattern
   - Check numeric fields are within valid ranges
   - Ensure categorical fields use exact valid values

3. **Model Not Loaded (500)**
   - Restart the server
   - Check model file exists at `Models/deepLearning/best_model.pkt`

4. **Prediction Error (500)**
   - Check input data formatting
   - Verify all numeric fields are actually numeric

## Performance Considerations

- **Response Time**: Typically 100-300ms per prediction
- **Concurrent Requests**: Server handles multiple concurrent requests
- **Memory Usage**: ~500MB for model and preprocessing pipeline
- **CPU Usage**: Optimized for CPU inference (GPU optional)

## Security Considerations

- **Data Privacy**: No patient data is stored or logged
- **Input Validation**: All inputs are validated before processing
- **Error Handling**: Detailed errors in development, generic in production
- **CORS**: Enabled for cross-origin requests (configure for production)

## Deployment

### Development
```bash
python prediction_server.py
```

### Production
```bash
# Set environment variables
export PORT=8080
export DEBUG=false

# Run with gunicorn (recommended)
gunicorn -w 4 -b 0.0.0.0:8080 prediction_server:app
```

### Docker
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["python", "prediction_server.py"]
```

## Dependencies

- Flask: Web framework
- PyTorch: Deep learning model
- scikit-learn: Data preprocessing
- NumPy/Pandas: Data manipulation
- Flask-CORS: Cross-origin support

## Support

For technical support or questions about the API:
1. Check the logs for detailed error messages
2. Verify all input parameters match the specification
3. Ensure the model file is properly loaded
4. Test with the provided examples first

## Changelog

### Version 1.0
- Initial release with SingleShotCNN model
- 188-feature engineering pipeline
- Temperature scaling for confidence calibration
- Simplified response format with confidence score only
