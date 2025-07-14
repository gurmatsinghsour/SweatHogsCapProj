#!/usr/bin/env python3
"""
Test script for PDF report generation
This script tests the PDF report generator independently and via the API.
"""

import json
import requests
from pdf_report_generator import generate_medical_report

def test_pdf_generation_standalone():
    """Test PDF generation directly without API"""
    print("Testing standalone PDF generation...")
    
    # Sample patient data
    sample_patient_data = {
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
    }
    
    sample_remedy = """Despite well-controlled diabetes (normal A1C, metformin steady, no insulin, primary diagnosis 250.00), this female patient in her 50s has a high readmission risk score of 0.74. This suggests the elevated risk is likely driven by her other two diagnoses, the acute reason for her emergency admission, or the complexity of her overall care (10 medications, regimen changed, 30 lab procedures).

**Insight/Remedy:** Focus on robust post-discharge planning, including thorough medication reconciliation and close, multidisciplinary follow-up for all her medical conditions, not just diabetes, to mitigate this elevated readmission risk. Patient education on new medications and symptom monitoring is crucial.

**Disclaimer:** This information is for general insight and should not be considered medical advice. Always consult with a qualified healthcare professional for diagnosis and treatment."""
    
    try:
        output_file = generate_medical_report(
            patient_data=sample_patient_data, 
            confidence_score=0.7425894737243652, 
            remedy=sample_remedy,
            output_path="test_medical_report.pdf"
        )
        print(f" PDF generated successfully: {output_file}")
        return True
    except Exception as e:
        print(f" PDF generation failed: {e}")
        return False

def test_api_with_report():
    """Test PDF generation via API endpoint"""
    print("\nTesting API with PDF report generation...")
    
    # Sample request data
    sample_data = {
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
    }
    
    try:
        # Test the /predict_with_report endpoint
        response = requests.post(
            'http://localhost:8080/predict_with_report',
            json=sample_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"  API response successful")
            print(f"   Confidence Score: {result.get('confidence_score', 'N/A')}")
            print(f"   PDF Available: {result.get('pdf_available', False)}")
            print(f"   PDF Filename: {result.get('pdf_filename', 'N/A')}")
            
            # If PDF was generated, try to download it
            if result.get('pdf_available') and result.get('pdf_filename'):
                download_url = f"http://localhost:8080/download_report/{result['pdf_filename']}"
                print(f"   Download URL: {download_url}")
                
                download_response = requests.get(download_url)
                if download_response.status_code == 200:
                    with open(f"downloaded_{result['pdf_filename']}", 'wb') as f:
                        f.write(download_response.content)
                    print(f" PDF downloaded successfully as downloaded_{result['pdf_filename']}")
                else:
                    print(f" Failed to download PDF: {download_response.status_code}")
            
            return True
        else:
            print(f" API request failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(" Could not connect to API server. Make sure the server is running on localhost:8080")
        return False
    except Exception as e:
        print(f" API test failed: {e}")
        return False

def main():
    """Run all tests"""
    print(" Medical PDF Report Generator Test Suite")
    print("=" * 50)
    
    # Test standalone PDF generation
    standalone_success = test_pdf_generation_standalone()
    
    # Test API integration
    api_success = test_api_with_report()
    
    print("\n" + "=" * 50)
    print(" Test Results Summary:")
    print(f"   Standalone PDF: {' PASS' if standalone_success else ' FAIL'}")
    print(f"   API Integration: {' PASS' if api_success else ' FAIL'}")
    
    if standalone_success and api_success:
        print("\n All tests passed! Your PDF report system is working correctly.")
    elif standalone_success:
        print("\n  Standalone PDF works, but API integration failed. Check if server is running.")
    else:
        print("\n Tests failed. Check dependencies: pip install reportlab matplotlib")

if __name__ == "__main__":
    main()
