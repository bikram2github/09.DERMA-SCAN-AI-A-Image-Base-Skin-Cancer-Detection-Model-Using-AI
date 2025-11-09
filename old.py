import time
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image


@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="skin_cancer.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

st.warning("⚠️ This tool is for educational purposes only and should not be used for medical diagnosis.")


st.title("DERMA-SCAN AI : A Image-Base Skin Cancer Detection Model Using AI")

st.write("Upload an image of a skin lesion to check for potential skin cancer.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

class_names = ['Benign', 'Malignant']

@st.cache_data
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', width='content')
    st.write("")
    with st.spinner('Analyzing the image...'):
        time.sleep(1)
        input_data = preprocess_image(image)

        input_index = interpreter.get_input_details()[0]['index']
        output_index = interpreter.get_output_details()[0]['index']

        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        

        prediction = interpreter.get_tensor(output_index)
        if prediction.shape[-1] == 1:
            confidence = float(prediction[0][0])
            predicted_class = "Malignant" if confidence > 0.5 else "Benign"
            confidence = confidence * 100 if predicted_class == "Malignant" else (1 - confidence) * 100
        else:
            predicted_class = class_names[np.max(prediction)]
            confidence = np.max(prediction) * 100

        st.success(f"Prediction: **{predicted_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")