#!/usr/bin/env python3
"""
Simple test for the simplified prediction API
"""

import requests
import json

BASE_URL = "http://localhost:8080"

def test_simple_api():
    """Test the simplified API that returns only confidence score"""
    
    # Test case
    test_data = {
        "age": "[45-55)",
        "gender": "Male",
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
    
    print("🔬 Testing Simplified API")
    print("=" * 40)
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"Status: {result['status']}")
            print(f"Confidence Score: {result['confidence_score']:.3f}")
            print(f"\n📝 Response format:")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_simple_api()
