Epidemic Intelligence System

A machine learning-based system for outbreak prediction, risk assessment, and trend analysis

Overview

This project focuses on predicting the spread of infectious diseases using historical epidemiological data. It combines machine learning models with a web-based dashboard to provide insights into case growth, regional risk levels, and outbreak trends.

The system is designed as a decision-support tool that helps identify potential outbreaks early and highlight high-risk regions.


Tech Stack & Tools

Programming & Data Processing
	•	Python
	•	Pandas, NumPy

Machine Learning
	•	Scikit-learn
	•	XGBoost

Frontend
	•	HTML, CSS, JavaScript

Tools
	•	Git & GitHub
	•	Plotly


Features
	•	Prediction of future case counts (7–14 days)
	•	Risk classification (Low, Medium, High)
	•	Hotspot detection based on growth trends
	•	Anomaly detection for sudden spikes
	•	Interactive dashboard for visualization
	•	Model comparison for performance evaluation

Technical workflow

Data Collection
      ->
Data Cleaning & Preprocessing
      ->
Feature Engineering
      ->
Model Training (Linear Regression, XGBoost)
      ->
Prediction Generation
      ->
Risk Classification & Analysis
      ->
Visualization via Dashboard

Installation & Setup

1. Clone the repository
   
git clone https://github.com/KaustavKrishna/Epidemic-Intelligence-System
cd epidemic-intelligence

2. Install dependencies

pip install -r requirements.txt

3. Run the application
   
python epidemic_intelligence_model.py

4. Download and open epidemic_intelligence_website.html

