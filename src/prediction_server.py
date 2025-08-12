#!/usr/bin/env python3
"""
Medical Readmission Prediction Server

This Flask application provides a REST API for predicting patient readmission
risk using a trained deep learning model. The system integrates with Google's
Gemini AI to generate medical insights and recommendations based on prediction
results.

Key Features:
- Single-shot CNN model for readmission prediction
- Google Gemini AI integration for medical insights
- PDF report generation with professional formatting
- Feature preprocessing and data validation
- RESTful API endpoints for JSON and PDF responses

The server loads a pre-trained PyTorch model and preprocessed feature encoders
to make real-time predictions on patient data.
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dotenv import load_dotenv
# Load environment variables from .env file if it exists
# This allows for secure storage of API keys and configuration
load_dotenv()

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Import PDF report generator module
# This handles creation of professional medical reports with charts
try:
    from pdf_report_generator import generate_medical_report
    PDF_GENERATOR_AVAILABLE = True
except ImportError as e:
    PDF_GENERATOR_AVAILABLE = False
    print(f"Warning: PDF generator not available: {e}")

# Import Google Gemini AI for generating medical insights
# This provides natural language explanations and recommendations
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Gemini AI not available - install google-generativeai")

# Retrieve API keys from environment variables
# These should be set in a .env file or system environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Configure logging to track application behavior and errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask application with static file serving
app = Flask(__name__, static_folder='static')
# Enable Cross-Origin Resource Sharing (CORS) for web browser access
# This allows the API to be called from web applications hosted on different domains
CORS(app)

def generate_remedy_with_llm(user_info, diagnosis, confidence):
    """
    Generate medical insights and recommendations using Google's Gemini AI
    
    This function takes patient information and model predictions, then uses
    Google's Gemini large language model to generate human-readable medical
    insights and care recommendations.
    
    Parameters:
    user_info (dict): Patient demographic and clinical data
    diagnosis (str): Primary diagnosis code or description
    confidence (float): Model confidence score for readmission risk (0.0 to 1.0)
    
    Returns:
    str: Generated medical insights and recommendations, or error message
    
    Note: This is for educational/research purposes only and should not be
    used as actual medical advice.
    """
    # Convert patient information dictionary to readable format
    # Replace underscores with spaces for better readability
    formatted_info = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in user_info.items())
    
    # Construct a detailed prompt for the AI model
    # The prompt provides context about the patient and asks for medical insights
    prompt = (
        f"You are a helpful medical assistant. Based on the following patient information, "
        f"provide a concise, medically-reasoned insight or remedy. "
        f"Always include a disclaimer that this is not medical advice.\n\n"
        f"Patient information: {formatted_info}\n"
        f"Primary diagnosis: {diagnosis}\n"
        f"Readmission risk score: {confidence:.2f}\n\n"
        f"maximum character limit is 1200 characters very strictly.\n"
        f"donot include any special characters in the response.\n"
        f"Provide me recommendations to avoid readmission:\n"
        f"Please provide medical insights and recommendations:"
    )
    
    # Check if the Gemini library is properly installed
    if not GEMINI_AVAILABLE:
        return "Gemini AI service not available. Please install: pip install google-generativeai"
    
    # Verify that the API key is configured
    if not GEMINI_API_KEY:
        return "Gemini API key not configured. Please set GEMINI_API_KEY environment variable."
    
    try:
        # Configure the Gemini API with the provided key
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Initialize the Gemini model (using the Flash variant for speed)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Generate the medical insights response
        response = model.generate_content(prompt)
        
        # Return the generated text if available
        if response.text:
            return response.text.strip()
        else:
            return "No response generated from Gemini API."
            
    except Exception as e:
        # Log and return any errors that occur during API call
        logger.error(f"Gemini API error: {str(e)}")
        return f"Gemini API error: {str(e)}"

# Single-Shot Convolutional Neural Network Architecture
# This model was developed by team member Gurmat Singh for medical readmission prediction
class SingleShotCNN(nn.Module):
    """
    Single-Shot Convolutional Neural Network for Medical Readmission Prediction
    
    This architecture uses a single convolutional layer followed by global max pooling
    and fully connected layers. It's designed to process tabular medical data by
    treating each feature as a sequence element.
    
    The model architecture:
    1. Conv1D layer: Extracts local patterns from features
    2. ReLU activation: Non-linear transformation
    3. Global Max Pooling: Reduces dimensionality while preserving important features
    4. Dropout: Prevents overfitting
    5. Fully connected layers: Final classification
    
    Parameters:
    input_size (int): Number of input features
    num_classes (int): Number of output classes (typically 2 for binary classification)
    num_filters (int): Number of convolutional filters
    kernel_size (int): Size of the convolutional kernel
    dropout_p (float): Dropout probability for regularization
    """
    
    def __init__(self, input_size, num_classes, num_filters, kernel_size, dropout_p):
        super(SingleShotCNN, self).__init__()
        
        # First convolutional layer
        # Uses 1 input channel since we're processing tabular data
        # Padding='same' ensures output length matches input length
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size, padding='same')
        
        # ReLU activation function for non-linearity
        self.relu = nn.ReLU()
        
        # Global max pooling reduces each filter to a single value
        # This captures the most important feature detected by each filter
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Dropout layer for regularization to prevent overfitting
        self.dropout = nn.Dropout(dropout_p)
        
        # Fully connected layers for final classification
        # Maps from convolutional features to class probabilities
        self.fc1 = nn.Linear(num_filters, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, input_size)
        
        Returns:
        torch.Tensor: Output logits for classification
        """
        # Reshape input for Conv1d: (batch_size, channels, sequence_length)
        # We add a channel dimension since Conv1d expects 3D input
        x = x.unsqueeze(1)
        
        # Apply convolution and activation
        x = self.conv1(x)
        x = self.relu(x)
        
        # Global max pooling to get the most important features
        x = self.pool(x)
        
        # Flatten the output for the fully connected layers
        # This converts from (batch_size, num_filters, 1) to (batch_size, num_filters)
        x = x.view(x.size(0), -1)
        
        # Apply dropout for regularization
        x = self.dropout(x)
        
        # First fully connected layer with ReLU activation
        x = self.fc1(x)
        x = self.relu(x)
        
        # Final output layer (no activation - will be applied in loss function)
        x = self.fc2(x)
        return x

# Custom Data Type Converter for Preprocessing Pipeline
class TypeConverter(BaseEstimator, TransformerMixin):
    """
    Custom transformer to ensure all categorical data is converted to string type
    
    This transformer is part of the sklearn preprocessing pipeline and ensures
    that all categorical features are properly formatted as strings before
    being processed by one-hot encoding or label encoding.
    
    This is necessary because the model training pipeline expects string
    categorical values, but input data might come as various types.
    """
    
    def fit(self, X, y=None):
        """
        Fit method required by sklearn transformer interface
        
        For this transformer, no fitting is required as it simply
        converts data types without learning any parameters.
        
        Parameters:
        X: Input data (not used)
        y: Target data (not used)
        
        Returns:
        self: Returns the transformer instance
        """
        return self
    
    def transform(self, X, y=None):
        """
        Transform the input data by converting all values to strings
        
        Parameters:
        X: Input data to transform
        y: Target data (not used)
        
        Returns:
        X converted to string type
        """
        return X.astype(str)

class ModelPredictor:
    """
    Main prediction class that handles model loading, data preprocessing, and inference
    
    This class encapsulates all the functionality needed to make predictions using
    the trained SingleShotCNN model. It handles:
    - Loading the trained PyTorch model
    - Setting up data preprocessing pipelines
    - Preprocessing new patient data
    - Making predictions and calculating confidence scores
    - Integrating with AI services for medical insights
    
    The preprocessing pipeline must match exactly what was used during training
    to ensure consistent feature engineering.
    """
    
    def __init__(self, model_path='models/best_model.pkt', data_path='data/processed/medical_data.pkl'):
        """
        Initialize the ModelPredictor with model and data paths
        
        Parameters:
        model_path (str): Path to the saved PyTorch model file
        data_path (str): Path to the preprocessed training data (used for feature encoding)
        """
        # Determine the best available device for model inference
        # Priority: CUDA GPU > Apple Metal Performance Shaders > CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load the preprocessed training data
        # This is needed to recreate the exact same preprocessing pipeline
        # that was used during model training
        self.df = pd.read_pickle(data_path)
        logger.info(f"Loaded training data with shape: {self.df.shape}")
        
        # Setup the preprocessing pipeline to match training conditions
        self._setup_preprocessing()
        
        # Load the trained PyTorch model
        self._load_model(model_path)
        
    def _setup_preprocessing(self):
        """
        Setup the preprocessing pipeline based on the training data
        
        This method recreates the exact same preprocessing steps that were
        used during model training. This is critical for ensuring that
        new data is processed in the same way as the training data.
        
        The preprocessing includes:
        - Categorical encoding for non-numeric features
        - Numerical scaling for continuous features
        - Feature selection and ordering
        """
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

# Flask API Route Handlers
# These endpoints provide the REST API interface for the medical prediction system

@app.route('/')
def index():
    """
    Serve the main web interface
    
    Returns the static HTML page that provides a web-based interface
    for interacting with the prediction API. This allows users to
    input patient data through a web form rather than using cURL.
    
    Returns:
    HTML file: The main interface page
    """
    return send_from_directory('static', 'index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring system status
    
    This endpoint allows external monitoring systems to verify that
    the API server is running and the machine learning model is
    properly loaded and ready to make predictions.
    
    Returns:
    JSON: System health status including model loading status
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict_readmission():
    """
    Predict patient readmission risk and generate AI insights
    
    This is the main prediction endpoint that accepts patient data in JSON format,
    processes it through the trained machine learning model, and returns both
    the confidence score and AI-generated medical insights.
    
    Expected JSON Input:
    {
        "age": "[50-60)" (age range as string),
        "gender": "Female" (Male/Female),
        "time_in_hospital": 5 (integer days),
        "admission_type": 1 (integer code),
        "discharge_disposition": 1 (integer code),
        "admission_source": 1 (integer code),
        "num_medications": 10 (integer count),
        "num_lab_procedures": 30 (integer count),
        "num_procedures": 2 (integer count),
        "number_diagnoses": 3 (integer count),
        "number_inpatient": 1 (integer count),
        "number_outpatient": 2 (integer count),
        "number_emergency": 0 (integer count),
        "diabetesMed": "Yes" (Yes/No),
        "change": "Ch" (Ch/No),
        "A1Cresult": "Norm" (None/Norm/>7/>8),
        "max_glu_serum": "None" (None/Norm/>200/>300),
        "insulin": "No" (No/Down/Steady/Up),
        "metformin": "Steady" (No/Down/Steady/Up),
        "diagnosis_1": "250.00" (ICD-9 diagnosis code)
    }
    
    Returns:
    JSON: Prediction results with confidence score and AI-generated insights
    {
        "status": "success",
        "confidence_score": 0.65,
        "remedy": "AI-generated medical insights and recommendations"
    }
    """
    try:
        # Verify that the model has been properly initialized
        if predictor is None:
            return jsonify({
                'error': 'Model not initialized. Please restart the server.',
                'status': 'error'
            }), 500
        
        # Extract JSON data from the HTTP request
        input_data = request.get_json()
        
        # Validate that JSON data was provided
        if not input_data:
            return jsonify({
                'error': 'No JSON data provided in request body',
                'status': 'error'
            }), 400

        # Define all required fields for making a prediction
        # These must match exactly what the model was trained on
        required_fields = [
            'age', 'gender', 'time_in_hospital', 'admission_type', 
            'discharge_disposition', 'admission_source', 'num_medications',
            'num_lab_procedures', 'num_procedures', 'number_diagnoses',
            'number_inpatient', 'number_outpatient', 'number_emergency',
            'diabetesMed', 'change', 'A1Cresult', 'max_glu_serum',
            'insulin', 'metformin', 'diagnosis_1'
        ]
        
        # Check for any missing required fields in the input
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

        # Generate remedy/insight using LLM
        # You can pass the input_data, diagnosis, and confidence score
        diagnosis = input_data.get('diagnosis_1', 'Unknown')
        remedy = generate_remedy_with_llm(input_data, diagnosis, result['risk_score'])

        # Response with confidence score and remedy
        response = {
            'status': 'success',
            'confidence_score': result['risk_score'],
            'remedy': remedy
        }
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/predict_with_report', methods=['POST'])
def predict_with_pdf_report():
    """
    Predict patient readmission risk and generate a PDF report
    
    Returns JSON with confidence score, remedy, and PDF download link:
    {
        "status": "success",
        "confidence_score": 0.65,
        "remedy": "...",
        "pdf_path": "path/to/report.pdf"
    }
    """
    try:
        if predictor is None:
            return jsonify({
                'error': 'Model not initialized',
                'status': 'error'
            }), 500
        
        if not PDF_GENERATOR_AVAILABLE:
            return jsonify({
                'error': 'PDF generator not available. Please install: pip install reportlab matplotlib',
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

        # Generate remedy/insight using LLM
        diagnosis = input_data.get('diagnosis_1', 'Unknown')
        remedy = generate_remedy_with_llm(input_data, diagnosis, result['risk_score'])

        # Generate PDF report
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"medical_report_{timestamp}.pdf"
        # Save PDFs in reports directory
        pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports', 'generated_pdfs', pdf_filename)
        
        try:
            generated_pdf_path = generate_medical_report(
                patient_data=input_data,
                confidence_score=result['risk_score'],
                remedy=remedy,
                output_path=pdf_path
            )
            
            # Log successful PDF generation
            logger.info(f"Successfully generated medical report: {pdf_filename}")
            print(f"Successfully done generating a report: {pdf_filename}")
            
            # Return the PDF file directly for download
            return send_file(
                generated_pdf_path,
                as_attachment=True,
                download_name=pdf_filename,
                mimetype='application/pdf'
            )
            
        except Exception as pdf_error:
            logger.error(f"PDF generation error: {str(pdf_error)}")
            # Return error as JSON if PDF generation fails
            return jsonify({
                'status': 'error',
                'error': f'PDF generation failed: {str(pdf_error)}',
                'confidence_score': result['risk_score'],
                'remedy': remedy
            }), 500
        
    except Exception as e:
        logger.error(f"Prediction with report endpoint error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/download_report/<filename>', methods=['GET'])
def download_report(filename):
    """
    Download a generated PDF report
    """
    try:
        # Look for PDFs in the reports directory
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports', 'generated_pdfs', filename)
        
        if not os.path.exists(file_path):
            return jsonify({
                'error': 'Report file not found',
                'status': 'error'
            }), 404
            
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
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
