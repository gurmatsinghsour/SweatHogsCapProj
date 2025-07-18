#!/usr/bin/env python3
"""
SweatHogs Medical AI - Dataset Download Script
Humber College Capstone Project

This script downloads the required datasets for the medical readmission prediction system.
Run this script if the data files are missing from your local installation.
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

def create_directories():
    """Create necessary data directories if they don't exist"""
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directories: {raw_dir}, {processed_dir}")
    return raw_dir, processed_dir

def download_sample_data(raw_dir):
    """Download sample datasets for the project"""
    
    # Note: These are placeholder URLs - you'll need to replace with actual data sources
    datasets = {
        "diabetic_data.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip",
        "diagnosis_num_mapping.csv": "https://raw.githubusercontent.com/your-repo/data/diagnosis_mapping.csv",
        "medical_data.csv": "https://raw.githubusercontent.com/your-repo/data/medical_data.csv"
    }
    
    print("Note: This is a template script. You need to:")
    print("1. Replace URLs with actual data sources")
    print("2. Add proper data download logic")
    print("3. Ensure compliance with data usage terms")
    print()
    
    # For now, create placeholder files with instructions
    for filename in datasets.keys():
        filepath = raw_dir / filename
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write("# Placeholder file - replace with actual dataset\n")
                f.write(f"# Original source: {datasets[filename]}\n")
                f.write("# This file was created by download_data.py\n")
            print(f"Created placeholder: {filepath}")

def check_existing_data():
    """Check if data files already exist"""
    required_files = [
        "data/raw/diabetic_data.csv",
        "data/raw/diagnosis_num_mapping.csv", 
        "data/raw/medical_data.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            size = os.path.getsize(file_path)
            print(f"Found: {file_path} ({size} bytes)")
    
    return missing_files

def main():
    print("=" * 60)
    print("SweatHogs Medical AI - Dataset Setup")
    print("=" * 60)
    print()
    
    # Check current data status
    missing_files = check_existing_data()
    
    if not missing_files:
        print("All required data files are present!")
        print("You can proceed with running the application.")
        return
    
    print(f"Missing {len(missing_files)} required data files:")
    for file in missing_files:
        print(f"  - {file}")
    print()
    
    # Create directories
    raw_dir, processed_dir = create_directories()
    
    # Attempt to download data
    print("Attempting to set up data files...")
    download_sample_data(raw_dir)
    
    print()
    print("IMPORTANT NEXT STEPS:")
    print("1. Replace placeholder files with actual datasets")
    print("2. Ensure you have permission to use the medical datasets")
    print("3. Run the data preprocessing notebooks to generate processed data")
    print("4. Verify data integrity before training models")
    print()
    print("For the diabetes dataset, you can download from:")
    print("https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008")

if __name__ == "__main__":
    main()
