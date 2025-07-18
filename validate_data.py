#!/usr/bin/env python3
"""
Data Validation Script for SweatHogs Medical AI
Checks if all required data files exist and are properly formatted
"""

import os
import pandas as pd
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and return status info"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        return True, f"Found: {filepath} ({size:,} bytes)"
    else:
        return False, f"Missing: {filepath}"

def validate_csv_structure(filepath, expected_columns=None):
    """Validate CSV file structure"""
    try:
        df = pd.read_csv(filepath, nrows=5)  # Read just first 5 rows for validation
        rows, cols = df.shape
        
        status = f"Valid CSV: {rows}+ rows, {cols} columns"
        
        if expected_columns:
            missing_cols = set(expected_columns) - set(df.columns)
            if missing_cols:
                status += f" | Missing columns: {missing_cols}"
                return False, status
        
        return True, status
    except Exception as e:
        return False, f"CSV Error: {str(e)}"

def main():
    print("=" * 60)
    print("SweatHogs Medical AI - Data Validation")
    print("=" * 60)
    
    required_files = {
        "data/raw/diabetic_data.csv": "Main hospital readmission dataset",
        "data/raw/diagnosis_num_mapping.csv": "Diagnosis code mapping",
        "data/raw/medical_data.csv": "Preprocessed medical data",
        "data/processed/medical_data.pkl": "Model-ready processed data",
        "models/best_model.pkt": "Trained PyTorch model"
    }
    
    all_good = True
    
    for filepath, description in required_files.items():
        exists, message = check_file_exists(filepath, description)
        print(f"{'✓' if exists else '✗'} {message}")
        
        if exists and filepath.endswith('.csv'):
            valid, csv_status = validate_csv_structure(filepath)
            print(f"  └─ {csv_status}")
            if not valid:
                all_good = False
        elif not exists:
            all_good = False
    
    print("\n" + "=" * 60)
    
    if all_good:
        print("SUCCESS: All required files are present and valid!")
        print("You can now run: python app.py")
    else:
        print("ISSUES FOUND: Some required files are missing or invalid.")
        print("\nNext steps:")
        print("1. Run 'python setup_data.py' to download missing datasets")
        print("2. Check DATA_SETUP.md for detailed instructions")
        print("3. Ensure you have run the preprocessing notebooks")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
