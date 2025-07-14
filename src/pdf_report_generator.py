#!/usr/bin/env python3
"""
PDF Report Generator for Medical Readmission Predictions

This module creates professional PDF reports for medical readmission predictions.
It integrates patient data, model predictions, AI-generated insights, and 
visualizations into a comprehensive medical report suitable for healthcare
professionals and academic presentation.

Features:
- Professional medical report formatting with institutional branding
- Patient information tables with clinical data
- Risk assessment visualizations and charts
- AI-generated medical insights and recommendations
- Humber College and team branding integration
- Compliance with medical report standards

The reports are generated using ReportLab for PDF creation and matplotlib
for data visualizations, ensuring high-quality professional output.
"""

import os
from datetime import datetime
from io import BytesIO
import base64

# ReportLab imports for PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.platypus import Image as ReportLabImage
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Matplotlib imports for creating charts and visualizations
# Configure matplotlib to use a non-GUI backend to prevent threading issues
import matplotlib
matplotlib.use('Agg')  # Use Anti-Grain Geometry backend (no GUI)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np


class MedicalReportPDF:
    """
    Professional PDF report generator for medical readmission predictions
    
    This class handles the creation of comprehensive medical reports that include:
    - Patient demographic and clinical information
    - Model prediction results and confidence scores
    - Risk assessment visualizations
    - AI-generated medical insights and recommendations
    - Professional formatting with institutional branding
    
    The reports follow medical documentation standards and include appropriate
    disclaimers for research and educational use.
    """
    
    def __init__(self):
        """
        Initialize the PDF report generator with custom styles
        
        Sets up the ReportLab document styles and creates custom formatting
        styles for different sections of the medical report.
        """
        # Load default ReportLab styles as a starting point
        self.styles = getSampleStyleSheet()
        
        # Create custom styles for medical report formatting
        self.setup_custom_styles()
        
    def setup_custom_styles(self):
        """
        Setup custom paragraph styles for the medical report
        
        Creates professional styling for different sections of the report:
        - Title and headers with institutional colors
        - Body text with appropriate spacing and formatting
        - Special styles for medical data and recommendations
        """
        # Main report title style with institutional blue color
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2C5282'),  # Professional blue
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Section header style for major report sections
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=16,
            spaceBefore=25,
            textColor=colors.HexColor('#2D3748'),
            fontName='Helvetica-Bold'
        ))
        
        # Patient info style
        self.styles.add(ParagraphStyle(
            name='PatientInfo',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            textColor=colors.HexColor('#4A5568'),
            fontName='Helvetica'
        ))
        
        # Remedy style
        self.styles.add(ParagraphStyle(
            name='RemedyText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            textColor=colors.HexColor('#2D3748'),
            fontName='Helvetica',
            alignment=TA_JUSTIFY,
            leftIndent=20,
            rightIndent=20
        ))
        
        # Warning style
        self.styles.add(ParagraphStyle(
            name='Warning',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            textColor=colors.HexColor('#E53E3E'),
            fontName='Helvetica-Bold',
            alignment=TA_CENTER
        ))
        
        # Institution style
        self.styles.add(ParagraphStyle(
            name='Institution',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            textColor=colors.HexColor('#2C5282'),
            fontName='Helvetica',
            alignment=TA_CENTER
        ))

    def create_confidence_chart(self, confidence_score):
        """Create a visual confidence score chart"""
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Create a gauge-like chart
        categories = ['Low Risk\n(0.0-0.3)', 'Medium Risk\n(0.3-0.7)', 'High Risk\n(0.7-1.0)']
        colors_list = ['#38A169', '#ECC94B', '#E53E3E']  # Green, Yellow, Red
        
        # Determine which category the score falls into
        if confidence_score < 0.3:
            highlight_idx = 0
        elif confidence_score < 0.7:
            highlight_idx = 1
        else:
            highlight_idx = 2
            
        # Create bar chart
        bars = ax.bar(categories, [0.3, 0.4, 0.3], color=colors_list, alpha=0.7)
        
        # Highlight the relevant bar
        bars[highlight_idx].set_alpha(1.0)
        bars[highlight_idx].set_edgecolor('black')
        bars[highlight_idx].set_linewidth(2)
        
        # Add score indicator
        ax.axhline(y=confidence_score, color='black', linestyle='--', linewidth=2)
        ax.text(1, confidence_score + 0.05, f'Score: {confidence_score:.3f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_ylabel('Risk Score', fontweight='bold')
        ax.set_title('Readmission Risk Assessment', fontweight='bold', fontsize=14)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to image for PDF
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return ImageReader(buf)

    def format_patient_data(self, patient_data):
        """Format patient data for display"""
        formatted_data = []
        
        # Define field mappings for better display
        field_mappings = {
            'age': 'Age Group',
            'gender': 'Gender',
            'time_in_hospital': 'Time in Hospital (days)',
            'admission_type': 'Admission Type',
            'discharge_disposition': 'Discharge Disposition',
            'admission_source': 'Admission Source',
            'num_medications': 'Number of Medications',
            'num_lab_procedures': 'Lab Procedures',
            'num_procedures': 'Medical Procedures',
            'number_diagnoses': 'Number of Diagnoses',
            'number_inpatient': 'Previous Inpatient Visits',
            'number_outpatient': 'Previous Outpatient Visits',
            'number_emergency': 'Previous Emergency Visits',
            'diabetesMed': 'Diabetes Medication',
            'change': 'Medication Change',
            'A1Cresult': 'A1C Test Result',
            'max_glu_serum': 'Max Glucose Serum',
            'insulin': 'Insulin',
            'metformin': 'Metformin',
            'diagnosis_1': 'Primary Diagnosis',
            'diagnosis_2': 'Secondary Diagnosis',
            'diagnosis_3': 'Tertiary Diagnosis'
        }
        
        for key, value in patient_data.items():
            if key in field_mappings:
                display_name = field_mappings[key]
                # Ensure we display the actual value, handle None and empty values
                display_value = str(value) if value is not None else 'N/A'
                if display_value == '' or display_value.strip() == '':
                    display_value = 'N/A'
                formatted_data.append([display_name, display_value])
                
        return formatted_data

    def create_patient_info_table(self, patient_data):
        """Create a formatted table with patient information"""
        data = self.format_patient_data(patient_data)
        
        # Create a simple two-column table (Field | Value)
        table_data = [['Field', 'Value']]  # Header row
        table_data.extend(data)
        
        # Create table with proper column widths
        table = Table(table_data, colWidths=[3*inch, 3*inch])
        table.setStyle(TableStyle([
            # Header row styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C5282')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            
            # Data rows styling
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),  # Bold field names
            ('FONTNAME', (1, 1), (1, -1), 'Helvetica'),       # Regular values
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            
            # General styling
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            
            # Alternating row colors for better readability
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')]),
        ]))
        
        return table

    def generate_report(self, patient_data, confidence_score, remedy, output_path=None):
        """Generate the complete PDF report"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"medical_report_{timestamp}.pdf"
        
        doc = SimpleDocTemplate(output_path, pagesize=letter, 
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        story = []
        
        # Header with institutional branding
        report_title = Paragraph("Medical Readmission Risk Assessment Report", 
                                self.styles['ReportTitle'])
        story.append(report_title)
        
        # Institution and project information
        institution_info = """
        <b>Humber College Institute of Technology & Advanced Learning</b><br/>
        <b>Capstone Project - Computer Programming</b><br/>
        <b>Team SweatHogs:</b> Gurmat Singh Sour, Minh Nhat Mai, Yuvraj Grover, Robert Seibel, Mohammed Hasnain Ali
        """
        institution_para = Paragraph(institution_info, self.styles['Institution'])
        story.append(institution_para)
        story.append(Spacer(1, 15))
        
        # Report metadata
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        date_para = Paragraph(f"<b>Generated:</b> {report_date}", 
                            self.styles['PatientInfo'])
        story.append(date_para)
        story.append(Spacer(1, 30))  # Increased spacing before patient info
        
        # Patient Information Section
        patient_header = Paragraph("Patient Information", self.styles['SectionHeader'])
        story.append(patient_header)
        story.append(Spacer(1, 12))  # Add more space after header
        
        # Patient info table
        patient_table = self.create_patient_info_table(patient_data)
        story.append(patient_table)
        story.append(Spacer(1, 25))
        
        # Risk Assessment Section
        risk_header = Paragraph("Risk Assessment Results", self.styles['SectionHeader'])
        story.append(risk_header)
        
        # Confidence score
        risk_level = "High" if confidence_score >= 0.7 else "Medium" if confidence_score >= 0.3 else "Low"
        risk_color = "#E53E3E" if confidence_score >= 0.7 else "#ECC94B" if confidence_score >= 0.3 else "#38A169"
        
        confidence_text = f"""
        <b>Readmission Risk Score:</b> {confidence_score:.3f}<br/>
        <b>Risk Level:</b> <font color="{risk_color}"><b>{risk_level} Risk</b></font><br/>
        <b>Model Confidence:</b> {(confidence_score * 100):.1f}%
        """
        confidence_para = Paragraph(confidence_text, self.styles['PatientInfo'])
        story.append(confidence_para)
        story.append(Spacer(1, 15))
        
        # Risk visualization
        try:
            chart_image = self.create_confidence_chart(confidence_score)
            chart = ReportLabImage(chart_image, width=5*inch, height=3*inch)
            story.append(chart)
        except Exception as e:
            error_text = f"Chart generation error: {str(e)}"
            error_para = Paragraph(error_text, self.styles['Warning'])
            story.append(error_para)
        
        story.append(Spacer(1, 20))
        
        # AI-Generated Insights Section
        insights_header = Paragraph("AI-Generated Medical Insights", self.styles['SectionHeader'])
        story.append(insights_header)
        
        # Clean and format the remedy text
        formatted_remedy = remedy.replace('\n\n', '<br/><br/>').replace('\n', '<br/>')
        remedy_para = Paragraph(formatted_remedy, self.styles['RemedyText'])
        story.append(remedy_para)
        story.append(Spacer(1, 20))
        
        # Risk Factors Explanation
        explanation_header = Paragraph("Understanding Your Risk Score", self.styles['SectionHeader'])
        story.append(explanation_header)
        
        explanation_text = """
        The readmission risk score is calculated using advanced machine learning algorithms that analyze 
        multiple patient factors including medical history, current medications, diagnosis codes, 
        hospital stay characteristics, and previous healthcare utilization patterns. 
        <br/><br/>
        <b>Risk Categories:</b><br/>
        • <font color="#38A169"><b>Low Risk (0.0-0.3):</b></font> Minimal likelihood of readmission<br/>
        • <font color="#ECC94B"><b>Medium Risk (0.3-0.7):</b></font> Moderate readmission risk requiring attention<br/>
        • <font color="#E53E3E"><b>High Risk (0.7-1.0):</b></font> Elevated readmission risk requiring intensive follow-up
        """
        explanation_para = Paragraph(explanation_text, self.styles['RemedyText'])
        story.append(explanation_para)
        story.append(Spacer(1, 30))
        
        # Disclaimer
        disclaimer_text = """
        <b>IMPORTANT MEDICAL DISCLAIMER:</b><br/>
        This report is generated by an AI system for informational purposes only and should not be considered 
        medical advice, diagnosis, or treatment recommendations. Always consult with qualified healthcare 
        professionals for medical decisions. The predictions are based on statistical models and may not 
        account for all individual patient factors.
        """
        disclaimer_para = Paragraph(disclaimer_text, self.styles['Warning'])
        story.append(disclaimer_para)
        
        # Footer with project information
        story.append(Spacer(1, 20))
        footer_text = """
        Generated by SweatHogs Medical AI Prediction System<br/>
        Humber College Capstone Project | Computer Programming Program<br/>
        Team Members: Gurmat Singh, Minh Nguyen, Yuvraj Patel, Robert Johnson
        """
        footer_para = Paragraph(footer_text, self.styles['PatientInfo'])
        story.append(footer_para)
        
        # Build PDF
        doc.build(story)
        return output_path


def generate_medical_report(patient_data, confidence_score, remedy, output_path=None):
    """
    Convenience function to generate a medical report PDF
    
    Args:
        patient_data (dict): Patient information dictionary
        confidence_score (float): Risk confidence score (0-1)
        remedy (str): AI-generated remedy/insights text
        output_path (str, optional): Output file path for PDF
        
    Returns:
        str: Path to generated PDF file
    """
    generator = MedicalReportPDF()
    return generator.generate_report(patient_data, confidence_score, remedy, output_path)


if __name__ == "__main__":
    # Example usage
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
    
    output_file = generate_medical_report(sample_patient_data, 0.7425894737243652, sample_remedy)
    print(f"Report generated: {output_file}")
