# Dataset Setup Guide

## Required Datasets

The SweatHogs Medical AI system requires the following datasets:

### 1. Diabetic Data (diabetic_data.csv)
- **Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008
- **Description**: Hospital readmission data for diabetic patients
- **Size**: ~70MB
- **Location**: `data/raw/diabetic_data.csv`

### 2. Diagnosis Mapping (diagnosis_num_mapping.csv)
- **Source**: Generated from ICD-9 codes
- **Description**: Maps diagnosis codes to readable descriptions
- **Location**: `data/raw/diagnosis_num_mapping.csv`

### 3. Medical Data (medical_data.csv)
- **Source**: Preprocessed subset of diabetic_data.csv
- **Description**: Cleaned and feature-engineered dataset
- **Location**: `data/raw/medical_data.csv`

## Setup Instructions

### Option 1: Automatic Setup (Recommended)
```bash
python setup_data.py
```

### Option 2: Manual Download
1. Download the diabetes dataset from UCI:
   ```bash
   wget https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip
   unzip dataset_diabetes.zip
   mv diabetic_data.csv data/raw/
   ```

2. Run the preprocessing notebook:
   ```bash
   jupyter notebook notebooks/data_preprocessing/data_preprocessing.ipynb
   ```

### Option 3: Use Your Own Data
1. Place your CSV files in `data/raw/`
2. Ensure they follow the expected schema
3. Run preprocessing to generate `data/processed/medical_data.pkl`

## Data Privacy Notice

- These datasets contain anonymized medical information
- Use only for educational/research purposes
- Do not redistribute or use for commercial purposes
- Follow your institution's data usage policies

## Troubleshooting

### Missing Data Files Error
If you get an error about missing data files:
1. Run `python setup_data.py` to check data status
2. Ensure you have internet connectivity for downloads
3. Check file permissions in the data/ directory

### Preprocessing Errors
If preprocessing fails:
1. Verify CSV file format and headers
2. Check for missing or corrupted data
3. Review the preprocessing notebook for specific requirements

## File Structure
```
data/
├── raw/                    # Original datasets
│   ├── diabetic_data.csv  # Main hospital data
│   ├── diagnosis_num_mapping.csv
│   └── medical_data.csv
└── processed/             # Preprocessed data
    └── medical_data.pkl   # Ready for model training
```
