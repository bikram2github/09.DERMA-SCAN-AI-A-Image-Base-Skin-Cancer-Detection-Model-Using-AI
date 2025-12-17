import time
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

api_key=st.secrets["GROQ_API_KEY"]

llm=ChatGroq(api_key=api_key, model="openai/gpt-oss-20b", temperature=0)



parser=StrOutputParser()


@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="Models/cnn_full_2model_resnet.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()


st.warning("⚠️ This tool is for educational purposes only and should not be used for medical diagnosis.")
st.title(" DERMA-SCAN AI: Image-Based Skin Cancer Detection Using AI")
st.write("Upload an image of a skin lesion to check for potential skin cancer.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

class_names = ['Benign', 'Malignant']


@st.cache_data
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width='content')
    st.write("")

    with st.spinner("Analyzing the image...",show_time=True):
        time.sleep(1)

        input_data = preprocess_image(image)

        input_index = interpreter.get_input_details()[0]['index']
        output_index = interpreter.get_output_details()[0]['index']

        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_index)

        confidence = float(prediction[0][0])
        predicted_class = "Malignant" if confidence > 0.5 else "Benign"
        confidence_malignant = confidence * 100
        confidence_benign = (1 - confidence) * 100

    
    
    
    col1, col2 = st.columns(2)
    col1.metric("Benign", f"{confidence_benign:.2f}%")
    col2.metric("Malignant", f"{confidence_malignant:.2f}%")

    if predicted_class == "Benign":
        st.success(" **Prediction: Benign (Non-cancerous)**")
    else:
        st.error(" **Prediction: Malignant (cancerous)**")


    st.divider() 
    space = st.empty()
    space.write("")


    prompt1 = ChatPromptTemplate.from_messages([
    ("system",
     "You explain model predictions for medical images in a short, human, and clear way. "
     "Describe only the visual features in the given image that could have influenced the model's prediction "
     "(such as color, texture, shape, borders, patterns, asymmetry)."
     "Try to give in column and should be short and simple to understand "
     "Do not add extra medical facts or generic AI phrases."),
    
    ("user",
     "Here is the image: {image}. The model predicted: {prediction} .\n"
     "In brief, explain which visual features in this image likely led to that prediction.")
])


    chain1=prompt1 | llm | parser
    explanation=chain1.invoke({
        "image": image,
        "prediction": predicted_class,
    })
    st.markdown("### Model Explanation:")
    st.warning(explanation)


    st.divider() 
    space = st.empty()
    space.write("")

    if st.button(" Generate Detailed Report"):
        with st.spinner("Generating report...",show_time=True):
            time.sleep(0.5)

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a medical AI assistant specialized in dermatology. Provide detailed analysis and recommendations based on skin lesion classifications."
                "don't use multiple fonts in your response. Stick to plain text formatting."
                "strictly the report should not look line ai generated report."),
                ("user", "The model has predicted that the skin lesion is {prediction} with a confidence of {confidence:.2f}%. Please provide a detailed report including possible implications, recommended next steps, and any precautions the user should take.")
            ] )

            chain=prompt | llm | parser
            report=chain.invoke({
                "prediction": predicted_class,
                "confidence": confidence_malignant if predicted_class == "Malignant" else confidence_benign
            })

            st.divider() 
            space = st.empty()
            space.write("")
            st.markdown("### Here is the Detailed Report:")
            st.write(report)
            st.divider() 
            space = st.empty()
            space.write("")


            st.download_button(
                label=" Download Report",
                data=report,
                file_name="derma_scan_report.txt",
                mime="text/plain"
            )

else:
    st.info("👆 Please upload an image to analyze.")

st.markdown("---")
st.caption("Developed for educational purposes only. Not for medical use.")