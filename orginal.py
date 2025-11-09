import time
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ✅ Load TFLite Model
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="skin_cancer.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# ✅ Header
st.warning("⚠️ This tool is for educational purposes only and should not be used for medical diagnosis.")
st.title("🩺 DERMA-SCAN AI: Image-Based Skin Cancer Detection Using AI")
st.write("Upload an image of a skin lesion to check for potential skin cancer.")

# ✅ File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

class_names = ['Benign', 'Malignant']

# ✅ Preprocess Image
@st.cache_data
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)

# ✅ Inference
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width='content')
    st.write("")

    with st.spinner("Analyzing the image..."):
        time.sleep(1)

        input_data = preprocess_image(image)

        input_index = interpreter.get_input_details()[0]['index']
        output_index = interpreter.get_output_details()[0]['index']

        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_index)

        # ✅ Sigmoid output (1 neuron)
        if prediction.shape[-1] == 1:
            confidence = float(prediction[0][0])
            predicted_class = "Malignant" if confidence > 0.5 else "Benign"
            confidence_malignant = confidence * 100
            confidence_benign = (1 - confidence) * 100

        # ✅ Softmax output (2 neurons)
        else:
            predicted_class = class_names[np.argmax(prediction)]
            confidence_malignant = float(prediction[0][1]) * 100
            confidence_benign = float(prediction[0][0]) * 100

    # ✅ Display Results
    st.markdown("### 📊 Confidence Breakdown:")
    col1, col2 = st.columns(2)
    col1.metric("Benign", f"{confidence_benign:.2f}%")
    col2.metric("Malignant", f"{confidence_malignant:.2f}%")

    if predicted_class == "Benign":
        st.success("✅ **Prediction: Benign (Non-cancerous)**")
    else:
        st.error("🚨 **Prediction: Malignant (Possible skin cancer)**")

    st.markdown("---")
    st.caption("Developed for educational purposes only. Not for medical use.")

else:
    st.info("👆 Please upload an image to analyze.")