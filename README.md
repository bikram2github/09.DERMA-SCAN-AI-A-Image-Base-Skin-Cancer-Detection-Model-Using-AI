# DERMA SCAN AI

CNN-based Skin Cancer Screening with Automated Diagnostic Report Generation

Web Link: https://derma-scan-ai.streamlit.app/

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

## Model & Training
- Input size: 150 × 150 RGB
- Optimizer: Adam
- Loss Function: Binary Cross-Entropy
- Train/Test Split: 90% / 10%
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

---

## 🧠 Tech Stack

**Programming:** Python  
**Deep Learning:** TensorFlow, Keras, CNN  
**Models:** ResNet, VGG19, DenseNet  
**Generative AI:** LangChain, Groq LLM  
**Web Framework:** Streamlit  
**Libraries:** NumPy, Pandas, Matplotlib  
**Tools:** Git, GitHub, Google Colab, VS Code  

---

## 🏗️ Workflow

1. Upload skin lesion image  
2. Image preprocessing and augmentation  
3. CNN-based lesion classification  
4. LLM-generated medical explanation  
5. Results displayed via Streamlit web interface  

---

## ▶️ How to Run the Application

```bash
pip install -r requirements.txt
# run from this folder
streamlit run app.py
