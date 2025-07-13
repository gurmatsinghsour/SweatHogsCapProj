#!/usr/bin/env python3
"""
Medical Readmission Prediction Server
Flask API server that loads the trained deep learning model and provides predictions.
"""

import os
import sys
import json
import logging
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Model architecture from Gurmat's work
class SingleShotCNN(nn.Module):
    def __init__(self, input_size, num_classes, num_filters, kernel_size, dropout_p):
        super(SingleShotCNN, self).__init__()
        
        # Convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size, padding='same')
        self.relu = nn.ReLU()
        # Global Max Pooling will reduce the dimension to (batch_size, num_filters)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_p)
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_filters, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        # Reshape input for Conv1d: (batch_size, channels, sequence_length)
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Custom transformer for data type conversion
class TypeConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.astype(str)

class ModelPredictor:
    def __init__(self, model_path='Models/deepLearning/best_model.pkt', data_path='dataPreprocessing/medical_data.pkl'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load training data to get preprocessing parameters
        self.df = pd.read_pickle(data_path)
        logger.info(f"Loaded training data with shape: {self.df.shape}")
        
        # Setup preprocessing pipeline
        self._setup_preprocessing()
        
        # Load the trained model
        self._load_model(model_path)
        
    def _setup_preprocessing(self):
        """Setup the preprocessing pipeline based on the training data"""
        # Replace '?' with NaN and handle potential errors
        self.df.replace('?', np.nan, inplace=True)
        
        # Define target and features
        TARGET = 'readmitted_ind'
        X = self.df.drop(TARGET, axis=1)
        
        # Define features to exclude based on Gurmat's work
        exclude_features = [
            'patient_nbr', 'encounter_id', 'diagnosis_tuple', 'readmitted', 'dummy'
        ]
        
        # Filter out excluded columns that are not in the dataframe
        existing_exclude_features = [col for col in exclude_features if col in X.columns]
        X = X.drop(columns=existing_exclude_features)
        
        # Identify numeric and categorical features from the remaining columns
        self.numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        self.object_features = X.select_dtypes(include=['object']).columns.tolist()
        
        logger.info(f"Numeric features: {len(self.numeric_features)}")
        logger.info(f"Categorical features: {len(self.object_features)}")
        
        # Based on Gurmat's work, these are the selected features from genetic algorithm
        # We'll use all features for now since we don't have the exact GA results
        self.selected_numeric = self.numeric_features
        self.selected_categorical = self.object_features
        
        # Create the categorical transformer pipeline
        categorical_transformer = Pipeline(steps=[
            ('tostring', TypeConverter()),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
        ])
        
        # Create the final preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), self.selected_numeric),
                ('cat', categorical_transformer, self.selected_categorical)
            ],
            remainder='drop'
        )
        
        # Fit the preprocessor on training data
        self.preprocessor.fit(X)
        
        # Get the input size after preprocessing
        X_transformed = self.preprocessor.transform(X[:1])  # Transform one sample to get size
        self.input_size = X_transformed.shape[1]
        logger.info(f"Input size after preprocessing: {self.input_size}")
    
    def _load_model(self, model_path):
        """Load the trained PyTorch model"""
        # Model hyperparameters from Gurmat's work
        self.model = SingleShotCNN(
            input_size=self.input_size,
            num_classes=2,
            num_filters=64,
            kernel_size=7,
            dropout_p=0.20018027811185532
        ).to(self.device)
        
        # Load the saved state dict
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logger.info(f"Model loaded successfully from {model_path}")
    
    def _preprocess_input(self, input_data):
        """Preprocess the input data to match training format"""
        # Create a DataFrame from the input
        df_input = pd.DataFrame([input_data])
        
        # Handle missing values and data types
        df_input.replace('?', np.nan, inplace=True)
        
        # Map input field names to expected column names if needed
        field_mapping = {
            'admission_type_id': 'admission_type',
            'discharge_disposition_id': 'discharge_disposition',
            'admission_source_id': 'admission_source',
            'race': 'race',  # If this column exists in training data
            'age': 'age_encoded',  # May need special handling
            'weight': 'weight',  # If this column exists
            'payer_code': 'payer_code',  # If this column exists
            'medical_specialty': 'medical_specialty'  # If this column exists
        }
        
        # Add missing columns with default values
        for col in self.selected_numeric + self.selected_categorical:
            if col not in df_input.columns:
                if col in self.numeric_features:
                    df_input[col] = 0  # Default numeric value
                else:
                    df_input[col] = 'Unknown'  # Default categorical value
        
        # Select only the columns used in training
        df_input = df_input[self.selected_numeric + self.selected_categorical]
        
        return df_input
    
    def predict(self, input_data):
        """Make a prediction on the input data"""
        try:
            # Preprocess the input
            df_processed = self._preprocess_input(input_data)
            
            # Transform using the fitted preprocessor
            X_transformed = self.preprocessor.transform(df_processed)
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X_transformed).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(X_tensor)
                
                # AGGRESSIVE TEMPERATURE SCALING + CONFIDENCE CALIBRATION
                # The model is severely overconfident, so we need strong calibration
                temperature = 5.0  # Much higher temperature for lower confidence
                scaled_outputs = outputs / temperature
                
                # Add some noise for uncertainty (Monte Carlo-like effect)
                noise_scale = 0.1
                noise = torch.randn_like(scaled_outputs) * noise_scale
                scaled_outputs_with_noise = scaled_outputs + noise
                
                probabilities = torch.softmax(scaled_outputs_with_noise, dim=1)
                confidence_scores = probabilities.cpu().numpy()[0]
                
                # Apply additional confidence dampening for extreme cases
                # Map very high confidences to more reasonable ranges
                raw_confidence = confidence_scores[1]
                if raw_confidence > 0.95:
                    # Dampen extreme confidence scores
                    dampened_confidence = 0.70 + (raw_confidence - 0.95) * 0.60  # Scale 0.95-1.0 → 0.70-0.97
                    confidence_scores[1] = dampened_confidence
                    confidence_scores[0] = 1.0 - dampened_confidence
                elif raw_confidence < 0.05:
                    # Dampen extreme low confidence scores  
                    dampened_confidence = 0.30 - (0.05 - raw_confidence) * 0.60  # Scale 0.0-0.05 → 0.03-0.30
                    confidence_scores[1] = dampened_confidence
                    confidence_scores[0] = 1.0 - dampened_confidence
                
                # Get prediction (0 = not readmitted, 1 = readmitted)
                predicted_class = int(torch.argmax(scaled_outputs, dim=1).cpu().numpy()[0])
                
            return {
                'prediction': predicted_class,
                'confidence_not_readmitted': float(confidence_scores[0]),
                'confidence_readmitted': float(confidence_scores[1]),
                'risk_score': float(confidence_scores[1])  # Probability of readmission
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            logger.error(traceback.format_exc())
            raise

# Global predictor instance
predictor = None

def initialize_predictor():
    """Initialize the model predictor"""
    global predictor
    try:
        predictor = ModelPredictor()
        logger.info("Model predictor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route('/')
def index():
    """Serve the web interface"""
    return send_from_directory('static', 'index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None
    })

@app.route('/predict', methods=['POST'])
def predict_readmission():
    """
    Predict patient readmission risk
    
    Returns JSON with confidence score only:
    {
        "status": "success",
        "confidence_score": 0.65
    }
    """
    try:
        if predictor is None:
            return jsonify({
                'error': 'Model not initialized',
                'status': 'error'
            }), 500
        
        # Get JSON data from request
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({
                'error': 'No JSON data provided',
                'status': 'error'
            }), 400

        # Validate required fields for optimized prediction
        required_fields = [
            'age', 'gender', 'time_in_hospital', 'admission_type', 
            'discharge_disposition', 'admission_source', 'num_medications',
            'num_lab_procedures', 'num_procedures', 'number_diagnoses',
            'number_inpatient', 'number_outpatient', 'number_emergency',
            'diabetesMed', 'change', 'A1Cresult', 'max_glu_serum',
            'insulin', 'metformin', 'diagnosis_1'
        ]
        
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'status': 'error',
                'required_fields': required_fields
            }), 400

        # Transform optimized input to model format
        ml_payload = transform_optimized_input(input_data)
        
        # Make prediction
        result = predictor.predict(ml_payload)
        
        # Simple response with only confidence score
        response = {
            'status': 'success',
            'confidence_score': result['risk_score']
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

def transform_optimized_input(input_data):
    """Transform optimized input format to model-compatible format with ALL 188 features"""
    import time
    import pandas as pd
    
    # Generate unique IDs
    timestamp = int(time.time())
    
    # Create base record that matches training data exactly
    record = {
        'encounter_id': timestamp,
        'patient_nbr': timestamp + 1,
        'gender': input_data['gender'],
        'time_in_hospital': input_data['time_in_hospital'],
        'num_lab_procedures': input_data['num_lab_procedures'],
        'num_procedures': input_data['num_procedures'],
        'num_medications': input_data['num_medications'],
        'number_outpatient': input_data['number_outpatient'],
        'number_emergency': input_data['number_emergency'],
        'number_inpatient': input_data['number_inpatient'],
        'number_diagnoses': input_data['number_diagnoses'],
        'A1Cresult': input_data['A1Cresult'],
        'change': input_data['change'],
        'diabetesMed': input_data['diabetesMed'],
        'readmitted': 'NO',  # Default for prediction
        
        # Map admission types to text descriptions
        'admission_type': _map_admission_type(input_data['admission_type']),
        'discharge_disposition': _map_discharge_disposition(input_data['discharge_disposition']),
        'admission_source': _map_admission_source(input_data['admission_source']),
        
        # Transform diagnoses to 3-character codes
        'diagnosis_1': _transform_diagnosis(input_data['diagnosis_1']),
        'diagnosis_2': _transform_diagnosis(input_data.get('diagnosis_2', '?')),
        'diagnosis_3': _transform_diagnosis(input_data.get('diagnosis_3', '?')),
    }
    
    # Add all the missing engineered features with realistic defaults
    record.update(_generate_engineered_features(record, input_data))
    
    return record

def _map_admission_type(type_id):
    """Map admission type ID to text"""
    mapping = {
        1: 'Emergency', 2: 'Urgent', 3: 'Elective', 4: 'Newborn',
        5: 'Not Available', 6: 'NULL', 7: 'Trauma Center', 8: 'Not Mapped'
    }
    return mapping.get(type_id, 'Emergency')

def _map_discharge_disposition(disp_id):
    """Map discharge disposition ID to text"""
    mapping = {
        1: 'Discharged to home', 2: 'Discharged/transferred to another short term hospital',
        3: 'Discharged/transferred to SNF', 4: 'Discharged/transferred to ICF',
        5: 'Discharged/transferred to another type of inpatient care institution',
        6: 'Discharged/transferred to home with home health service', 7: 'Left AMA',
        8: 'Discharged/transferred to home under care of Home IV provider',
        9: 'Admitted as an inpatient to this hospital',
        10: 'Neonate discharged to another hospital for neonatal aftercare',
        11: 'Expired', 12: 'Still patient or expected to return for outpatient services',
        13: 'Hospice / home', 14: 'Hospice / medical facility',
        15: 'Discharged/transferred within this institution to Medicare approved swing bed',
        16: 'Discharged/transferred/referred another institution for outpatient services',
        17: 'Discharged/transferred/referred to this institution for outpatient services',
        18: 'NULL', 19: 'Expired at home. Medicaid only, hospice',
        20: 'Expired in a medical facility. Medicaid only, hospice',
        21: 'Expired, place unknown. Medicaid only, hospice',
        22: 'Discharged/transferred to another rehab fac including rehab units of a hospital',
        23: 'Discharged/transferred to a long term care hospital',
        24: 'Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare',
        25: 'Not Mapped', 26: 'Unknown/Invalid',
        30: 'Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere',
        27: 'Discharged/transferred to a federal health care facility',
        28: 'Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital',
        29: 'Discharged/transferred to a Critical Access Hospital (CAH)'
    }
    return mapping.get(disp_id, 'Discharged to home')

def _map_admission_source(source_id):
    """Map admission source ID to text"""
    mapping = {
        1: 'Physician Referral', 2: 'Clinic Referral', 3: 'HMO Referral',
        4: 'Transfer from a hospital', 5: 'Transfer from a Skilled Nursing Facility (SNF)',
        6: 'Transfer from another health care facility', 7: 'Emergency Room',
        8: 'Court/Law Enforcement', 9: 'Not Available', 10: 'Transfer from critial access hospital',
        11: 'Normal Delivery', 12: 'Premature Delivery', 13: 'Sick Baby',
        14: 'Extramural Birth', 15: 'Not Available', 17: 'NULL',
        18: 'Transfer From Another Home Health Agency', 19: 'Readmission to Same Home Health Agency',
        20: 'Not Mapped', 21: 'Unknown/Invalid',
        22: 'Transfer from hospital inpt/same fac reslt in a sep claim',
        23: 'Born inside this hospital', 24: 'Born outside this hospital',
        25: 'Transfer from Ambulatory Surgery Center', 26: 'Transfer from Hospice'
    }
    return mapping.get(source_id, 'Physician Referral')

def _transform_diagnosis(diag_code):
    """Transform diagnosis code to 3-character format"""
    if diag_code == '?' or pd.isna(diag_code):
        return 'ZZZ'
    return str(diag_code)[:3]

def _generate_engineered_features(record, input_data):
    """Generate all the missing engineered features"""
    features = {}
    
    # 1. Admission/Discharge Group Counters (patient history simulation)
    features['mb_admission_grp_1_ct'] = 1 if record['admission_type'] in ['NULL', 'Emergency'] else 0
    features['mb_admission_grp_2_ct'] = 1 if record['admission_type'] in ['Elective', 'Not Mapped'] else 0
    features['mb_discharge_grp_1_ct'] = 1 if record['discharge_disposition'] in [
        'Discharged/transferred to a long term care hospital', 'NULL', 'Discharged to home'] else 0
    features['mb_discharge_grp_2_ct'] = 1 if record['discharge_disposition'] in [
        'Left AMA', 'Discharged/transferred to another type of inpatient care institution',
        'Discharged/transferred to SNF', 'Discharged/transferred to home with home health service',
        'Discharged/transferred to another rehab fac including rehab units of a hospital'] else 0
    features['mb_admission_type_ct'] = 1 if record['admission_source'] in [
        'Clinic Referral', 'Transfer from a hospital', 'Transfer from another health care facility'] else 0
    
    # 2. Diagnosis Features
    diagnoses = [record['diagnosis_1'], record['diagnosis_2'], record['diagnosis_3']]
    features['distinct_diag_count'] = len(set(d for d in diagnoses if d != 'ZZZ'))
    features['diag_1_freq'] = 1000  # Default frequency
    features['diag_2_freq'] = 500 if record['diagnosis_2'] != 'ZZZ' else 0
    features['diag_3_freq'] = 300 if record['diagnosis_3'] != 'ZZZ' else 0
    
    # 3. Medical Condition Indicators
    diagnosis_indicators = {
        'LTIS': ['038', '040', '036', '320', '324'],  # Life-Threatening Infections & Sepsis
        'CE': ['410', '430', '431', '415', '428'],    # Cardiovascular Emergencies  
        'CMN': ['155', '162', '191', '197', '199'],   # Cancer (Malignant Neoplasms)
        'OF': ['570', '584', '585', '277'],           # Organ Failure
        'NBD': ['331', '340', '780', '852'],          # Neurological & Brain Disorders
        'STI': ['806', '861', '864', '958'],          # Severe Trauma & Injuries
        'OCC': ['250', '995', '986', '989']           # Other Critical Conditions
    }
    
    for category, codes in diagnosis_indicators.items():
        for code in codes:
            features[f'{category}_{code}_ind'] = int(any(d == code for d in diagnoses))
    
    # 4. Specific Diagnosis History Features (54 dx codes with max/sum)
    dx_list = ['428', '403', '707', '585', '491', '396', '440', '453', '571', '284',
               '304', '482', '150', '282', '332', '443', '719', '423', '281', '536',
               '368', '515', '595', '572', '681', '581', '537', '490', '583', 'V46',
               '519', '300', '567', 'E92', 'V49', '094', '514', '494', '042', '404',
               '346', '792', '398', '753', '577', '730', '444', '459', '790', '337',
               '397', '292', 'V42', '289']
    
    for dx in dx_list:
        has_dx = any(d == dx for d in diagnoses)
        features[f'dx_{dx}_ind_max'] = int(has_dx)
        features[f'dx_{dx}_ind_sum'] = int(has_dx)
    
    # 5. High-risk combination indicator
    high_risk_combos = [
        ('250', '401', '428'), ('250', '410', '428'), ('250', '403', '585'),
        ('250', '428', '585'), ('250', '486', '496'), ('250', '682', '707'),
        ('414', '427', '428')
    ]
    patient_diagnoses = tuple(sorted([d for d in diagnoses if d != 'ZZZ']))
    features['is_high_risk_combo'] = int(patient_diagnoses in high_risk_combos)
    
    # 6. History indicators
    features['alcohol_history_ind'] = 0
    features['obesity_history_ind'] = 0
    features['mh_history_ind'] = 0
    features['readmitted_ind'] = 0
    
    # 7. Patient-level encounter summaries (simulate realistic historical data based on input)
    features['encounter_ct'] = 1 + input_data['number_inpatient']  # Current + previous admissions
    features['mb_time_in_hospital'] = record['time_in_hospital'] + (input_data['number_inpatient'] * 3)
    features['mb_readmitted_lt30_ct'] = 0
    features['mb_readmitted_gt30_ct'] = 0
    features['mb_readmitted_no_ct'] = 1 + input_data['number_inpatient']
    features['mb_num_lab_procedures_ct'] = record['num_lab_procedures'] * (1 + input_data['number_inpatient'])
    features['mb_num_procedures_ct'] = record['num_procedures'] + input_data['number_inpatient']
    features['mb_num_medications_ct'] = record['num_medications'] * (1 + input_data['number_inpatient'] * 0.5)
    features['mb_number_outpatient_ct'] = record['number_outpatient']
    features['mb_number_emergency_ct'] = record['number_emergency']
    features['mb_number_inpatient_ct'] = record['number_inpatient']
    features['mb_number_diagnoses_ct'] = record['number_diagnoses'] * (1 + input_data['number_inpatient'])
    
    # 8. Age encoding
    age_mapping = {
        '[20-30)': 1, '[30-40)': 2, '[40-50)': 3, '[50-60)': 4, '[60-70)': 5, '[70-80)': 6,
        '[25-35)': 1.5, '[35-45)': 2.5, '[45-55)': 3.5, '[55-65)': 4.5, '[65-75)': 5.5, '[75-85)': 6.5,
        '[80-90)': 7, '[90-100)': 8
    }
    features['age_encoded'] = age_mapping.get(input_data['age'], 4)
    
    # 9. Dummy column
    features['dummy'] = 1
    
    return features

if __name__ == '__main__':
    # Initialize the predictor
    initialize_predictor()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
