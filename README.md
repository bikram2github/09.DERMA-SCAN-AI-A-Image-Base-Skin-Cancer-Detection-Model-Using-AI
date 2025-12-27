# DERMA SCAN AI

CNN-based Skin Cancer Screening with Automated Diagnostic Report Generation

## Overview
DERMA SCAN AI is a deep learning–based system designed for binary skin cancer screening
(Benign vs Malignant) using dermoscopic images. The system uses a ResNet50 CNN model
for classification and integrates a Large Language Model (LLM) via the Groq API to
generate structured diagnostic reports.

## Features
- Binary skin lesion classification (Benign / Malignant)
- ResNet50-based CNN model
- Image preprocessing and data augmentation
- Automated diagnostic report generation
- Streamlit-based web interface
- Telemedicine-ready deployment

## Dataset
- HAM10000 Dataset
- ISIC Archive  
(Note: Dataset is not included due to size and license constraints.)

## Model & Training
- Input size: 150 × 150 RGB
- Optimizer: Adam
- Loss Function: Binary Cross-Entropy
- Train/Test Split: 90% / 10%
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

## Running the Application
```bash
pip install -r requirements.txt
streamlit run app/app.py
