# Medical Readmission Prediction Server Documentation

## Overview

The Medical Readmission Prediction Server is a Flask-based REST API that provides diabetes patient readmission risk assessment using a deep learning CNN model. The server processes patient medical data and returns confidence scores indicating the likelihood of hospital readmission within 30 days.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation & Setup](#installation--setup)
3. [API Endpoints](#api-endpoints)
4. [Input Parameters](#input-parameters)
5. [Response Format](#response-format)
6. [Model Architecture](#model-architecture)
7. [Feature Engineering](#feature-engineering)
8. [Confidence Calibration](#confidence-calibration)
9. [Error Handling](#error-handling)
10. [Integration Examples](#integration-examples)
11. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware
- **CPU**: Modern multi-core processor
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: 2GB free space for model and dependencies
- **GPU**: Optional (CUDA/MPS supported for acceleration)

### Software
- **Python**: 3.8+ (tested on 3.10)
- **Operating System**: macOS, Linux, Windows
- **Dependencies**: Listed in `requirements.txt`

### Key Dependencies
```
torch>=1.9.0
flask>=2.0.0
flask-cors>=3.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

---

## Installation & Setup

### 1. Environment Setup
```bash
# Create virtual environment
conda create -n aienv python=3.10
conda activate aienv

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Files
Ensure the following files are present:
- `Models/deepLearning/best_model.pkt` - Trained PyTorch model
- `dataPreprocessing/medical_data.pkl` - Training data for preprocessing

### 3. Start Server
```bash
python prediction_server.py
```

Default configuration:
- **Host**: `0.0.0.0` (all interfaces)
- **Port**: `8080`
- **Debug**: `False`

### 4. Environment Variables
```bash
export PORT=8080              # Custom port
export DEBUG=True             # Enable debug mode
```

---

## API Endpoints

### 1. Health Check
**Endpoint**: `GET /health`

**Purpose**: Verify server status and model initialization

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Home Page
**Endpoint**: `GET /`

**Purpose**: Serve web interface (if available)

**Response**: HTML page from `static/index.html`

### 3. Prediction Endpoint
**Endpoint**: `POST /predict`

**Purpose**: Generate readmission risk prediction

**Content-Type**: `application/json`

**Response**: Confidence score (see [Response Format](#response-format))

---

## Input Parameters

### Required Fields

All fields are **mandatory** unless marked as optional.

#### Core Demographics
| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| `age` | string | Patient age group | `"[20-30)"`, `"[25-35)"`, `"[30-40)"`, `"[35-45)"`, `"[45-55)"`, `"[55-65)"`, `"[65-75)"`, `"[75-85)"`, `"[80-90)"` |
| `gender` | string | Patient gender | `"Male"`, `"Female"` |

#### Hospital Stay Details
| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| `time_in_hospital` | integer | Days in hospital | 1-14+ |
| `admission_type` | integer | Type of admission | `1` (Emergency), `2` (Urgent), `3` (Elective) |
| `discharge_disposition` | integer | Discharge destination | `1` (Home), `3` (SNF), `6` (Home Health), etc. |
| `admission_source` | integer | Admission source | `1` (Physician), `7` (Emergency Room), etc. |

#### Medical Complexity
| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| `num_medications` | integer | Number of medications | 1-50+ |
| `num_lab_procedures` | integer | Number of lab tests | 1-120+ |
| `num_procedures` | integer | Number of procedures | 0-10+ |
| `number_diagnoses` | integer | Number of diagnoses | 1-15+ |

#### Patient History
| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| `number_inpatient` | integer | Previous inpatient visits | 0-10+ |
| `number_outpatient` | integer | Previous outpatient visits | 0-20+ |
| `number_emergency` | integer | Previous emergency visits | 0-10+ |

#### Diabetes Management
| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| `diabetesMed` | string | On diabetes medication | `"Yes"`, `"No"` |
| `change` | string | Medication changes | `"Ch"` (Changed), `"No"` (No change) |
| `A1Cresult` | string | A1C test result | `"Norm"`, `">7"`, `">8"`, `"None"` |
| `max_glu_serum` | string | Glucose serum level | `"Norm"`, `">200"`, `">300"`, `"None"` |

#### Key Medications
| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| `insulin` | string | Insulin management | `"No"`, `"Down"`, `"Steady"`, `"Up"` |
| `metformin` | string | Metformin management | `"No"`, `"Down"`, `"Steady"`, `"Up"` |

#### Diagnoses
| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| `diagnosis_1` | string | Primary diagnosis code | ICD-9 codes (e.g., `"250.00"`) |
| `diagnosis_2` | string | Secondary diagnosis *(optional)* | ICD-9 codes or omit field |
| `diagnosis_3` | string | Tertiary diagnosis *(optional)* | ICD-9 codes or omit field |

### Example Request
```json
{
  "age": "[50-60)",
  "gender": "Female",
  "time_in_hospital": 5,
  "admission_type": 2,
  "discharge_disposition": 1,
  "admission_source": 7,
  "num_medications": 12,
  "num_lab_procedures": 35,
  "num_procedures": 1,
  "number_diagnoses": 4,
  "number_inpatient": 1,
  "number_outpatient": 3,
  "number_emergency": 1,
  "diabetesMed": "Yes",
  "change": "No",
  "A1Cresult": ">7",
  "max_glu_serum": ">200",
  "insulin": "Steady",
  "metformin": "Steady",
  "diagnosis_1": "250.00",
  "diagnosis_2": "401.9"
}
```

---

## Response Format

### Success Response
```json
{
  "status": "success",
  "confidence_score": 0.634
}
```

**Fields**:
- `status`: Always `"success"` for successful predictions
- `confidence_score`: Float between 0.0-1.0 representing readmission probability

### Error Response
```json
{
  "status": "error",
  "error": "Missing required fields: age, gender",
  "required_fields": ["age", "gender", "..."]
}
```

**HTTP Status Codes**:
- `200`: Successful prediction
- `400`: Invalid input data
- `500`: Server/model error

---

## Model Architecture

### SingleShotCNN Architecture
The prediction model uses a 1D Convolutional Neural Network:

```
Input (188 features) 
    ↓
Conv1D (64 filters, kernel=7, padding='same')
    ↓
ReLU Activation
    ↓
AdaptiveMaxPool1D (global pooling)
    ↓
Dropout (p=0.20018027811185532)
    ↓
Linear (64 → 50)
    ↓
ReLU Activation
    ↓
Linear (50 → 2)
    ↓
Output (2 classes: readmitted/not readmitted)
```

### Model Parameters
- **Input Size**: 188 engineered features
- **Filters**: 64 convolutional filters
- **Kernel Size**: 7
- **Dropout Rate**: 0.200
- **Classes**: 2 (binary classification)
- **Device**: Auto-detected (CUDA/MPS/CPU)

---

## Feature Engineering

The server transforms input data into 188 engineered features:

### Feature Categories

#### 1. Base Features (15 features)
- Patient demographics
- Hospital stay metrics
- Medical history counts
- Medication information

#### 2. Medical Condition Indicators (35 features)
Categorical indicators for:
- **LTIS**: Life-Threatening Infections & Sepsis
- **CE**: Cardiovascular Emergencies
- **CMN**: Cancer (Malignant Neoplasms)
- **OF**: Organ Failure
- **NBD**: Neurological & Brain Disorders
- **STI**: Severe Trauma & Injuries
- **OCC**: Other Critical Conditions

#### 3. Diagnosis History Features (108 features)
- 54 specific diagnosis codes with max/sum indicators
- Diagnosis frequency calculations
- High-risk combination detection

#### 4. Patient-Level Aggregations (13 features)
- Encounter summaries
- Admission/discharge groupings
- Historical medical utilization

#### 5. Engineered Derived Features (17 features)
- Age encoding
- Admission type mappings
- Discharge disposition groupings
- Medical specialty indicators

### Feature Engineering Process
1. **Input Validation**: Verify required fields
2. **Code Mapping**: Convert IDs to descriptive text
3. **Diagnosis Processing**: Extract 3-character ICD codes
4. **History Simulation**: Generate patient-level aggregations
5. **Indicator Creation**: Binary flags for medical conditions
6. **Preprocessing**: MinMax scaling + One-hot encoding

---

## Confidence Calibration

### Problem Addressed
Original model suffered from severe overconfidence (returning 100% confidence scores).

### Calibration Techniques Applied

#### 1. Temperature Scaling
```python
temperature = 5.0
scaled_outputs = model_outputs / temperature
```

#### 2. Monte Carlo Noise
```python
noise_scale = 0.1
noise = torch.randn_like(outputs) * noise_scale
```

#### 3. Extreme Confidence Dampening
- Scores >95% mapped to 70-97% range
- Scores <5% mapped to 3-30% range

#### 4. Softmax Normalization
Final probabilities ensure proper probability distribution.

### Calibrated Score Ranges
- **0.36-0.50**: Lower readmission risk
- **0.50-0.70**: Moderate readmission risk
- **0.70-0.81**: Higher readmission risk

### Clinical Interpretation
The model maintains a **conservative bias**, which is clinically appropriate:
- False positives (over-prediction) are safer than false negatives
- Higher sensitivity helps identify at-risk patients
- Confidence scores provide nuanced risk stratification

---

## Error Handling

### Input Validation Errors
```json
{
  "status": "error",
  "error": "Missing required fields: age, gender",
  "required_fields": [...all required fields...]
}
```

### Model Initialization Errors
```json
{
  "status": "error",
  "error": "Model not initialized"
}
```

### Prediction Processing Errors
```json
{
  "status": "error",
  "error": "Prediction error: [specific error message]"
}
```

### Common Error Scenarios
1. **Missing Fields**: Required parameter not provided
2. **Invalid Values**: Parameters outside expected ranges
3. **Model Loading**: Missing model files or corruption
4. **Memory Issues**: Insufficient RAM for model processing
5. **Device Errors**: GPU/CUDA compatibility issues

---

## Integration Examples

### Python Integration
```python
import requests

# Patient data
patient = {
    "age": "[45-55)",
    "gender": "Male",
    "time_in_hospital": 4,
    "admission_type": 2,
    # ... all required fields
}

# Make prediction
response = requests.post(
    'http://localhost:8080/predict',
    json=patient,
    headers={'Content-Type': 'application/json'}
)

result = response.json()
confidence = result['confidence_score']
print(f"Readmission Risk: {confidence:.1%}")
```

### Node.js Integration
```javascript
const axios = require('axios');

async function predictReadmission(patientData) {
    try {
        const response = await axios.post(
            'http://localhost:8080/predict',
            patientData,
            {
                headers: { 'Content-Type': 'application/json' }
            }
        );
        
        return response.data.confidence_score;
    } catch (error) {
        console.error('Prediction error:', error.response.data);
        throw error;
    }
}

// Usage
const riskScore = await predictReadmission(patientData);
console.log(`Risk Score: ${(riskScore * 100).toFixed(1)}%`);
```

### cURL Integration
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": "[50-60)",
    "gender": "Female",
    "time_in_hospital": 5,
    "admission_type": 2,
    "discharge_disposition": 1,
    "admission_source": 7,
    "num_medications": 12,
    "num_lab_procedures": 35,
    "num_procedures": 1,
    "number_diagnoses": 4,
    "number_inpatient": 1,
    "number_outpatient": 3,
    "number_emergency": 1,
    "diabetesMed": "Yes",
    "change": "No",
    "A1Cresult": ">7",
    "max_glu_serum": ">200",
    "insulin": "Steady",
    "metformin": "Steady",
    "diagnosis_1": "250.00",
    "diagnosis_2": "401.9"
  }'
```

### Frontend JavaScript Integration
```javascript
async function assessReadmissionRisk(formData) {
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
    });
    
    const result = await response.json();
    
    if (result.status === 'success') {
        const riskLevel = categorizeRisk(result.confidence_score);
        displayRiskAssessment(riskLevel, result.confidence_score);
    } else {
        handleError(result.error);
    }
}

function categorizeRisk(confidence) {
    if (confidence < 0.4) return 'LOW';
    if (confidence < 0.7) return 'MEDIUM';
    return 'HIGH';
}
```

---

## Troubleshooting

### Common Issues & Solutions

#### 1. Server Won't Start
**Symptoms**: Server fails to initialize
**Causes**: 
- Missing model files
- Dependency issues
- Port conflicts

**Solutions**:
```bash
# Check model files exist
ls -la Models/deepLearning/best_model.pkt
ls -la dataPreprocessing/medical_data.pkl

# Check dependencies
pip install -r requirements.txt

# Try different port
export PORT=8081
python prediction_server.py
```

#### 2. Model Loading Errors
**Symptoms**: "Model not initialized" errors
**Causes**:
- Corrupted model file
- PyTorch version mismatch
- Insufficient memory

**Solutions**:
```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# Check available memory
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available/1e9:.1f}GB')"

# Re-download model if corrupted
```

#### 3. Prediction Errors
**Symptoms**: 500 errors during prediction
**Causes**:
- Invalid input format
- Missing required fields
- Data type mismatches

**Solutions**:
```python
# Validate input before sending
required_fields = [
    'age', 'gender', 'time_in_hospital', 'admission_type',
    # ... full list
]

missing = [f for f in required_fields if f not in patient_data]
if missing:
    print(f"Missing fields: {missing}")
```

#### 4. Performance Issues
**Symptoms**: Slow response times
**Causes**:
- CPU-only processing
- Large batch processing
- Memory leaks

**Solutions**:
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.mps.is_available()}')"

# Monitor memory usage
top -p $(pgrep -f prediction_server)

# Restart server periodically
```

#### 5. CORS Issues
**Symptoms**: Browser blocks requests
**Causes**: Cross-origin restrictions

**Solutions**:
- Server has CORS enabled by default
- Check browser console for specific errors
- Verify request headers are correct

### Debug Mode
Enable detailed logging:
```bash
export DEBUG=True
python prediction_server.py
```

### Health Check
Verify server status:
```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## Performance Characteristics

### Response Times
- **Typical**: 100-300ms per prediction
- **Cold start**: 1-3 seconds (first prediction)
- **Batch processing**: Not currently supported

### Resource Usage
- **RAM**: ~2-4GB during operation
- **CPU**: 1-2 cores during prediction
- **GPU**: Optional, provides 2-5x speedup

### Throughput
- **Single instance**: ~10-50 requests/second
- **Concurrent requests**: Limited by Flask development server
- **Production scaling**: Use WSGI server (Gunicorn, uWSGI)

---

## Security Considerations

### Input Validation
- All inputs are validated against expected ranges
- No SQL injection risks (no database queries)
- JSON parsing is safe (no eval/exec)

### Data Privacy
- No data is stored or logged by default
- Patient data is processed in memory only
- Consider adding request logging for auditing

### Network Security
- Server binds to all interfaces (`0.0.0.0`)
- Consider firewall rules for production
- Use HTTPS in production environments

### Recommendations for Production
1. **Use HTTPS**: Encrypt all communications
2. **Authentication**: Add API key or OAuth
3. **Rate Limiting**: Prevent abuse
4. **Input Sanitization**: Additional validation layers
5. **Logging**: Audit trail for medical predictions
6. **Monitoring**: Health checks and alerting

---

## Deployment Recommendations

### Development
```bash
python prediction_server.py
```

### Production
```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:8080 prediction_server:app

# Using Docker
docker build -t medical-prediction .
docker run -p 8080:8080 medical-prediction
```

### Environment Configuration
```bash
# Production settings
export FLASK_ENV=production
export DEBUG=False
export PORT=8080
export WORKERS=4
```

---

## API Versioning

Current version: **v1.0**

Future considerations:
- Version headers (`API-Version: 1.0`)
- Versioned endpoints (`/api/v1/predict`)
- Backward compatibility policies

---

## Support & Maintenance

### Logging
Server logs include:
- Model initialization status
- Request processing times
- Error details and stack traces
- Feature engineering warnings

### Monitoring Endpoints
- `GET /health` - Basic health check
- Consider adding: `/metrics`, `/status`, `/version`

### Model Updates
To update the model:
1. Replace `Models/deepLearning/best_model.pkt`
2. Restart server
3. Verify with health check

---

## License & Compliance

### Medical Device Considerations
- This is a decision support tool, not a medical device
- Clinical validation may be required for medical use
- Consider FDA/CE marking requirements for commercial deployment
- Maintain audit trails for medical predictions

### Data Compliance
- HIPAA compliance considerations for patient data
- GDPR compliance for EU patients
- Local healthcare data regulations

---

*Documentation generated for Medical Readmission Prediction Server v1.0*
*Last updated: July 13, 2025*
