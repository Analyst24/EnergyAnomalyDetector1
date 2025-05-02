import pandas as pd
import numpy as np
import streamlit as st
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
import datetime
import plotly.io as pio

def get_binary_file_downloader_html(bin_data, file_label='File', file_name='file.csv'):
    """
    Generate HTML code for file download link
    
    Args:
        bin_data: Binary data to be downloaded
        file_label: Label for the download button
        file_name: Name of the file to be downloaded
        
    Returns:
        HTML string for download button
    """
    b64 = base64.b64encode(bin_data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">{file_label}</a>'
    return href

def export_csv(df, filename="anomaly_results.csv"):
    """
    Export dataframe to CSV and return download link
    
    Args:
        df: pandas DataFrame to export
        filename: Name of the CSV file
        
    Returns:
        HTML string for download button
    """
    csv = df.to_csv(index=False)
    csv_bytes = csv.encode()
    
    return get_binary_file_downloader_html(csv_bytes, "Download CSV", filename)

def export_figure_as_image(fig, filename="plot.png", format="png"):
    """
    Export plotly figure as image and return download link
    
    Args:
        fig: Plotly figure object
        filename: Name of the image file
        format: Image format ('png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf')
        
    Returns:
        HTML string for download button
    """
    img_bytes = pio.to_image(fig, format=format)
    
    return get_binary_file_downloader_html(img_bytes, f"Download {format.upper()}", filename)

def generate_pdf_report(df, detection_results, filename="anomaly_report.pdf"):
    """
    Generate PDF report with anomaly detection results
    
    Args:
        df: pandas DataFrame with anomaly results
        detection_results: Dict with detection statistics
        filename: Name of the PDF file
        
    Returns:
        PDF file as bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    elements.append(Paragraph("Anomaly Detection Report", styles['Title']))
    elements.append(Spacer(1, 12))
    
    # Date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Generated on: {current_time}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Summary statistics
    elements.append(Paragraph("Summary Statistics", styles['Heading2']))
    elements.append(Spacer(1, 6))
    
    # Create summary table
    summary_data = [
        ["Total Data Points", str(len(df))],
        ["Anomalies Detected", str(detection_results.get('anomaly_count', 0))],
        ["Normal Points", str(detection_results.get('normal_count', 0))],
        ["Anomaly Percentage", f"{detection_results.get('anomaly_percentage', 0):.2f}%"],
        ["Detection Model", detection_results.get('model_type', 'Unknown')],
        ["Threshold", f"{detection_results.get('threshold', 0):.3f}"]
    ]
    
    summary_table = Table(summary_data, colWidths=[200, 200])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    
    elements.append(summary_table)
    elements.append(Spacer(1, 12))
    
    # Time-based distribution
    if 'hour_anomalies' in detection_results:
        elements.append(Paragraph("Time-based Distribution", styles['Heading2']))
        elements.append(Spacer(1, 6))
        
        hour_data = [["Hour", "Anomaly Count"]]
        for hour, count in detection_results['hour_anomalies'].items():
            hour_data.append([str(hour), str(count)])
        
        hour_table = Table(hour_data, colWidths=[100, 100])
        hour_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ]))
        
        elements.append(hour_table)
        elements.append(Spacer(1, 12))
    
    # Feature importance
    if 'feature_importance' in detection_results:
        elements.append(Paragraph("Feature Importance", styles['Heading2']))
        elements.append(Spacer(1, 6))
        
        feature_data = [["Feature", "Importance"]]
        for feature, importance in detection_results['feature_importance'].items():
            feature_data.append([feature, f"{importance:.4f}"])
        
        feature_table = Table(feature_data, colWidths=[200, 100])
        feature_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ]))
        
        elements.append(feature_table)
        elements.append(Spacer(1, 12))
    
    # Sample anomalies
    elements.append(Paragraph("Sample Anomalies", styles['Heading2']))
    elements.append(Spacer(1, 6))
    
    # Get top 10 anomalies by score
    if 'anomaly' in df.columns and 'anomaly_score' in df.columns:
        anomalies = df[df['anomaly'] == 1].sort_values('anomaly_score', ascending=False).head(10)
        
        if not anomalies.empty:
            # Select columns to display
            display_cols = ['timestamp', 'consumption', 'anomaly_score']
            display_cols = [col for col in display_cols if col in anomalies.columns]
            
            if display_cols:
                # Create table data
                anomaly_data = [display_cols]
                for _, row in anomalies.iterrows():
                    row_data = []
                    for col in display_cols:
                        if col == 'timestamp':
                            row_data.append(str(row[col]))
                        elif col == 'anomaly_score':
                            row_data.append(f"{row[col]:.4f}")
                        else:
                            row_data.append(str(row[col]))
                    anomaly_data.append(row_data)
                
                col_widths = [400 // len(display_cols)] * len(display_cols)
                anomaly_table = Table(anomaly_data, colWidths=col_widths)
                anomaly_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ]))
                
                elements.append(anomaly_table)
            else:
                elements.append(Paragraph("No columns to display for anomalies", styles['Normal']))
        else:
            elements.append(Paragraph("No anomalies detected", styles['Normal']))
    else:
        elements.append(Paragraph("Anomaly data not available", styles['Normal']))
    
    elements.append(Spacer(1, 12))
    
    # Recommendations section
    elements.append(Paragraph("Recommendations", styles['Heading2']))
    elements.append(Spacer(1, 6))
    
    recommendations = [
        "Investigate the anomalies detected during unusual hours",
        "Check for equipment malfunctions in locations with high anomaly rates",
        "Compare anomalous patterns with maintenance records",
        "Consider adjusting sensitivity threshold for more precise detection"
    ]
    
    for rec in recommendations:
        elements.append(Paragraph(f"• {rec}", styles['Normal']))
        elements.append(Spacer(1, 3))
    
    # Footer
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("© 2025 Opulent Chikwiramakomo. All rights reserved.", styles['Normal']))
    
    # Build PDF
    doc.build(elements)
    
    # Get PDF data
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

def generate_recommendations(df):
    """
    Generate energy efficiency recommendations based on anomaly detection results
    
    Args:
        df: pandas DataFrame with anomaly results
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Count anomalies
    if 'anomaly' in df.columns:
        anomaly_count = df['anomaly'].sum()
        anomaly_percentage = (anomaly_count / len(df)) * 100
        
        # Basic recommendations
        recommendations.append("Monitor energy consumption patterns regularly")
        recommendations.append("Establish baseline consumption levels for different times of day")
        
        # Add time-based recommendations if timestamp is available
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Check for hourly patterns
            hourly_anomalies = df[df['anomaly'] == 1].groupby('hour')['anomaly'].count()
            if not hourly_anomalies.empty:
                peak_hour = hourly_anomalies.idxmax()
                
                if peak_hour in range(9, 18):  # Business hours
                    recommendations.append(f"Investigate equipment usage during peak anomaly hour ({peak_hour}:00)")
                    recommendations.append("Check for equipment malfunctions or unnecessary usage during business hours")
                elif peak_hour in range(0, 6) or peak_hour in range(22, 24):  # Night hours
                    recommendations.append(f"Check for unexpected energy usage during off-hours (peak anomalies at {peak_hour}:00)")
                    recommendations.append("Consider implementing automatic shutdown procedures for non-essential equipment")
            
            # Check for day-of-week patterns
            daily_anomalies = df[df['anomaly'] == 1].groupby('day_of_week')['anomaly'].count()
            if not daily_anomalies.empty:
                peak_day = daily_anomalies.idxmax()
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                if peak_day in [5, 6]:  # Weekend
                    recommendations.append(f"Review weekend energy usage patterns (high anomalies on {day_names[peak_day]})")
                else:
                    recommendations.append(f"Analyze workload distribution on {day_names[peak_day]} to understand high anomaly rate")
        
        # Add location-based recommendations if location is available
        if 'location' in df.columns:
            location_anomalies = df[df['anomaly'] == 1].groupby('location')['anomaly'].count().sort_values(ascending=False)
            
            if not location_anomalies.empty:
                top_location = location_anomalies.index[0]
                recommendations.append(f"Prioritize energy audit for location: {top_location}")
                recommendations.append("Compare energy efficiency between different locations to identify best practices")
        
        # Add consumption-based recommendations if available
        if 'consumption' in df.columns:
            # Check if there are extreme consumption anomalies
            if 'anomaly_score' in df.columns:
                extreme_anomalies = df[(df['anomaly'] == 1) & (df['anomaly_score'] > df['anomaly_score'].quantile(0.9))]
                
                if not extreme_anomalies.empty:
                    max_consumption = extreme_anomalies['consumption'].max()
                    min_consumption = extreme_anomalies['consumption'].min()
                    
                    if max_consumption > df['consumption'].median() * 2:
                        recommendations.append("Investigate instances of abnormally high energy consumption")
                        recommendations.append("Consider implementing consumption alerts for unexpected spikes")
                    
                    if min_consumption < df['consumption'].median() * 0.5:
                        recommendations.append("Review unusually low consumption events - may indicate meter malfunctions")
        
        # Add weather-based recommendations if available
        if all(col in df.columns for col in ['temperature', 'consumption']):
            # Check correlation between temperature and consumption
            temp_corr = df['temperature'].corr(df['consumption'])
            
            if abs(temp_corr) > 0.7:  # Strong correlation
                if temp_corr > 0:  # Positive correlation
                    recommendations.append("High correlation between temperature and consumption detected")
                    recommendations.append("Consider optimizing HVAC systems for better energy efficiency")
                    recommendations.append("Implement temperature setbacks during non-business hours")
                else:  # Negative correlation
                    recommendations.append("Inverse relationship between temperature and consumption detected")
                    recommendations.append("Review heating system efficiency and control settings")
    
    # Add general recommendations if list is short
    if len(recommendations) < 5:
        general_recommendations = [
            "Implement regular energy audits to identify efficiency opportunities",
            "Consider upgrading to energy-efficient equipment and lighting",
            "Train staff on energy conservation best practices",
            "Install smart meters for real-time energy monitoring",
            "Develop an energy management plan with clear efficiency targets"
        ]
        recommendations.extend(general_recommendations[:5 - len(recommendations)])
    
    return recommendations
