from pyexpat import features
import time
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

from auth import (
    create_user,
    authenticate_user,
    validate_email,
    init_db,
)

st.set_page_config(
    page_title="DermaScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None
if "auth_page" not in st.session_state:
    st.session_state.auth_page = "login"

                    
init_db()

                                                                                
st.markdown("""
<style>
/* ── Chrome ── */
#MainMenu {visibility: hidden;}
footer     {visibility: hidden;}
header     {visibility: hidden;}

/* ── Base ── */
.stApp {
    background-color: #0D1117;
    font-family: 'Segoe UI', Inter, system-ui, sans-serif;
}

/* ── Hero Header ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.8rem 1rem;
    border-bottom: 1px solid #21262D;
    margin-bottom: 2rem;
    background: linear-gradient(180deg, #0D1B2A 0%, #0D1117 100%);
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    color: #E6EDF3;
    margin: 0;
    line-height: 1.15;
}
.hero-title span { color: #00C5A7; }
.hero-subtitle {
    color: #8B949E;
    font-size: 1rem;
    margin-top: 0.55rem;
}
.hero-tags {
    margin-top: 0.9rem;
    display: flex;
    justify-content: center;
    gap: 0.45rem;
    flex-wrap: wrap;
}
.hero-tag {
    background: rgba(0,197,167,0.1);
    border: 1px solid rgba(0,197,167,0.25);
    color: #00C5A7;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.22rem 0.65rem;
    border-radius: 100px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── User Top Bar ── */
.user-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: #161B22;
    border-bottom: 1px solid #21262D;
    padding: 0.65rem 1.5rem;
}
.user-avatar {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: linear-gradient(135deg, #00C5A7, #0096A7);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: #0D1117;
    font-weight: 800;
    font-size: 0.85rem;
    flex-shrink: 0;
}
.user-name  { color: #E6EDF3; font-size: 0.9rem; font-weight: 600; }
.user-email { color: #8B949E; font-size: 0.78rem; }

/* ── Auth card ── */
.auth-wrapper {
    display: flex;
    justify-content: center;
    padding: 1.5rem 0 3rem 0;
}
.auth-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 18px;
    padding: 2.4rem 2rem 2rem 2rem;
    width: 100%;
}
.auth-icon  { font-size: 2rem; margin-bottom: 0.6rem; }
.auth-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #E6EDF3;
    margin-bottom: 0.3rem;
}
.auth-subtitle {
    color: #8B949E;
    font-size: 0.88rem;
    margin-bottom: 0.3rem;
}
.auth-divider {
    border: none;
    border-top: 1px solid #21262D;
    margin: 1.4rem 0;
}

/* ── Input fields ── */
[data-testid="stTextInput"] label {
    color: #8B949E !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
[data-testid="stTextInput"] input {
    background: #0D1117 !important;
    border: 1px solid #30363D !important;
    color: #E6EDF3 !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
}
[data-testid="stTextInput"] input::placeholder { color: #484F58 !important; }
[data-testid="stTextInput"] input:focus {
    border-color: #00C5A7 !important;
    box-shadow: 0 0 0 3px rgba(0,197,167,0.15) !important;
}

/* ── Form submit button ── */
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #00C5A7, #0096A7) !important;
    color: #0D1117 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.01em !important;
    transition: opacity 0.2s !important;
}
[data-testid="stFormSubmitButton"] > button:hover { opacity: 0.85 !important; }

/* ── Primary buttons (main app) ── */
.stButton > button {
    background: linear-gradient(135deg, #00C5A7, #0096A7) !important;
    color: #0D1117 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
    transform: translateY(-1px) !important;
}

/* ── Secondary (switch page) buttons ── */
[data-testid="baseButton-secondary"] {
    background: transparent !important;
    color: #8B949E !important;
    border: 1px solid #30363D !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
}
[data-testid="baseButton-secondary"]:hover {
    background: #21262D !important;
    color: #E6EDF3 !important;
    border-color: #484F58 !important;
}

/* ── Logout button ── */
.logout-btn-wrap .stButton > button {
    background: transparent !important;
    color: #F87171 !important;
    border: 1px solid rgba(218,54,51,0.35) !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    padding: 0.35rem 0.9rem !important;
    border-radius: 6px !important;
    transform: none !important;
}
.logout-btn-wrap .stButton > button:hover {
    background: rgba(218,54,51,0.08) !important;
    border-color: rgba(218,54,51,0.6) !important;
    opacity: 1 !important;
    transform: none !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: #21262D !important;
    color: #E6EDF3 !important;
    border: 1px solid #30363D !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #30363D !important;
    border-color: #484F58 !important;
}

/* ── Disclaimer ── */
.disclaimer-box {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: rgba(218,54,51,0.08);
    border: 1px solid rgba(218,54,51,0.3);
    border-radius: 10px;
    padding: 0.85rem 1.25rem;
    color: #F87171;
    font-size: 0.9rem;
    margin-bottom: 1.8rem;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #161B22;
    border: 2px dashed #30363D;
    border-radius: 14px;
    padding: 0.5rem 1rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #00C5A7; }

/* ── Section labels ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #00C5A7;
    margin-bottom: 0.3rem;
}
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #E6EDF3;
    margin-bottom: 1rem;
}

/* ── Prediction badge ── */
.prediction-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 1rem;
    font-weight: 700;
    padding: 0.5rem 1.1rem;
    border-radius: 100px;
    letter-spacing: 0.01em;
    margin-bottom: 1.25rem;
}
.prediction-badge.benign    { background: rgba(46,160,67,0.12);   color: #3FB950; border: 1px solid rgba(46,160,67,0.35); }
.prediction-badge.malignant { background: rgba(218,54,51,0.12);   color: #F85149; border: 1px solid rgba(218,54,51,0.35); }

/* ── Result card ── */
.result-card { background: #161B22; border-radius: 14px; padding: 1.35rem 1.5rem; border: 1px solid #21262D; margin-bottom: 1rem; }
.result-card.benign    { border-color: rgba(46,160,67,0.4);  background: rgba(46,160,67,0.06); }
.result-card.malignant { border-color: rgba(218,54,51,0.4);  background: rgba(218,54,51,0.06); }
.conf-sub-label { font-size: 0.73rem; font-weight: 700; letter-spacing: 0.1em; color: #8B949E; text-transform: uppercase; margin-bottom: 1rem; }
.conf-bar-container { margin-bottom: 0.9rem; }
.conf-label-row { display: flex; justify-content: space-between; margin-bottom: 0.35rem; }
.conf-name  { color: #C9D1D9; font-size: 0.88rem; font-weight: 600; }
.conf-value { color: #8B949E; font-size: 0.88rem; }
.conf-track { height: 7px; border-radius: 100px; background: #21262D; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 100px; }
.conf-fill.benign    { background: linear-gradient(90deg, #2EA043, #3FB950); }
.conf-fill.malignant { background: linear-gradient(90deg, #DA3633, #F85149); }

/* ── Img panel label ── */
.img-panel-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; color: #8B949E; text-transform: uppercase; padding: 0.7rem 0 0.5rem 0; }

/* ── Explanation & report boxes ── */
.explanation-box {
    background: #161B22;
    border: 1px solid #21262D;
    border-left: 4px solid #00C5A7;
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    color: #C9D1D9;
    font-size: 0.95rem;
    line-height: 1.8;
}

.explanation-box strong {
    color: #E6EDF3;
    font-size: 1rem;
}
.report-box {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 1.5rem;
    color: #C9D1D9;
    font-size: 0.93rem;
    line-height: 1.5;
    white-space: normal;
}

/* ── Empty state ── */
.empty-state { text-align: center; padding: 4.5rem 2rem; background: #161B22; border: 2px dashed #21262D; border-radius: 16px; margin-top: 1rem; }
.empty-state-icon  { font-size: 3.8rem; margin-bottom: 1rem; }
.empty-state-title { font-size: 1.15rem; font-weight: 700; color: #C9D1D9; margin-bottom: 0.5rem; }
.empty-state-sub   { color: #8B949E; font-size: 0.9rem; }

/* ── Footer ── */
.custom-footer { text-align: center; color: #484F58; font-size: 0.78rem; padding: 2rem 0 1rem 0; border-top: 1px solid #21262D; margin-top: 3rem; letter-spacing: 0.02em; }

/* ── Misc ── */
hr { border-color: #21262D !important; }
.stSpinner > div > div { border-top-color: #00C5A7 !important; }
[data-testid="metric-container"] { background: #161B22; border: 1px solid #21262D; border-radius: 10px; padding: 0.8rem 1rem !important; }

/* ── Alert overrides ── */
[data-testid="stAlert"] { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


                                                                                
            
                                                                                

def show_auth_hero():
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🔬 <span>DERMA</span>SCAN AI</div>
        <div class="hero-subtitle">Image-Based Skin Cancer Detection · CNN + Generative AI</div>
    </div>
    """, unsafe_allow_html=True)


def show_login_page():
    show_auth_hero()

    col_l, col_c, col_r = st.columns([1, 1.4, 1])
    with col_c:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown("""
            <div class="auth-icon">🔐</div>
            <div class="auth-title">Welcome back</div>
            <div class="auth-subtitle">Sign in to your DermaScan AI account</div>
            <hr class="auth-divider">
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            email = st.text_input(
                "Email Address",
                placeholder="you@email.com",
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
            )
            submitted = st.form_submit_button("Sign In →", use_container_width=True)

        if submitted:
            email_clean = email.strip().lower()
            password_clean = password.strip()

            if not email_clean or not password_clean:
                st.error("⚠️ Please fill in all fields.")
            elif not validate_email(email_clean):
                st.error("⚠️ Enter a valid email address.")
            else:
                ok, result = authenticate_user(email_clean, password_clean)
                if ok:
                    st.session_state.authenticated = True
                    st.session_state.user = result
                    st.rerun()
                else:
                    st.error(f"❌ {result}")


        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center;color:#8B949E;font-size:0.85rem;margin:0.4rem 0 0.2rem 0;'>"
            "or</div>",
            unsafe_allow_html=True,
        )

                                                 
                                                                               


        st.markdown(
            "<p style='text-align:center;color:#8B949E;font-size:0.85rem;margin-bottom:0.5rem;'>"
            "Don't have an account?</p>",
            unsafe_allow_html=True,
        )
        if st.button("Create a free account", type="secondary", use_container_width=True, key="go_signup"):
            st.session_state.auth_page = "signup"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="custom-footer">
        🔬 DermaScan AI &nbsp;·&nbsp; Educational purposes only &nbsp;·&nbsp; Not for medical use
    </div>
    """, unsafe_allow_html=True)


def show_signup_page():
    show_auth_hero()

    col_l, col_c, col_r = st.columns([1, 1.4, 1])
    with col_c:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown("""
            <div class="auth-icon">✨</div>
            <div class="auth-title">Create account</div>
            <div class="auth-subtitle">Join DermaScan AI — takes less than a minute</div>
            <hr class="auth-divider">
        """, unsafe_allow_html=True)

        with st.form("signup_form", clear_on_submit=False):
            name = st.text_input("Full Name", placeholder="John Doe")
            email = st.text_input("Email Address", placeholder="you@example.com")
            password = st.text_input("Password", type="password", placeholder="Min 6 characters")
            confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
            submitted = st.form_submit_button("Create Account →", use_container_width=True)

        if submitted:
            errors = []
            if not name.strip():
                errors.append("Full name is required.")
            if not validate_email(email):
                errors.append("Enter a valid email address.")
            if len(password) < 6:
                errors.append("Password must be at least 6 characters.")
            if password != confirm:
                errors.append("Passwords do not match.")

            if errors:
                for err in errors:
                    st.error(f"⚠️ {err}")
            else:
                ok, msg = create_user(name, email.strip().lower(), password)
                if ok:
                    st.success(f"✅ {msg} You can now sign in.")
                    st.session_state.auth_page = "login"
                    time.sleep(1.2)
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")


        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center;color:#8B949E;font-size:0.85rem;margin:0.4rem 0 0.2rem 0;'>"
            "or</div>",
            unsafe_allow_html=True,
        )

                                                 
                                                                               


        st.markdown(
            "<p style='text-align:center;color:#8B949E;font-size:0.85rem;margin-bottom:0.5rem;'>"
            "Already have an account?</p>",
            unsafe_allow_html=True,
        )
        if st.button("Sign in instead", type="secondary", use_container_width=True, key="go_login"):
            st.session_state.auth_page = "login"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="custom-footer">
        🔬 DermaScan AI &nbsp;·&nbsp; Educational purposes only &nbsp;·&nbsp; Not for medical use
    </div>
    """, unsafe_allow_html=True)


                                                                                
                          
                                                                                

def show_main_app():
    user = st.session_state.user

                                                                                
    initials = "".join(w[0].upper() for w in user["name"].split()[:2])
    col_info, col_logout = st.columns([5, 1])

    with col_info:
        st.markdown(
            f'<div class="user-bar">'
            f'  <div class="user-avatar">{initials}</div>'
            f'  <div>'
            f'    <div class="user-name">{user["name"]}</div>'
            f'    <div class="user-email">{user["email"]}</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_logout:
        st.markdown(
            '<div class="logout-btn-wrap" style="display:flex;align-items:center;height:100%;padding:0.3rem 0;">',
            unsafe_allow_html=True,
        )
        if st.button("⎋ Logout", key="logout_btn"):
                                                                            
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.auth_page = "login"
            st.session_state.pop("interpreter", None)
            st.session_state.pop("llm", None)
            st.session_state.pop("parser", None)
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


                                                                                
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🔬 <span>DERMA</span>SCAN AI</div>
        <div class="hero-subtitle">Image-Based Skin Cancer Detection · CNN + Generative AI</div>
        <div class="hero-tags">
            <span class="hero-tag">TFLite CNN Model</span>
            <span class="hero-tag">Groq LLM</span>
            <span class="hero-tag">HAM10000 Dataset</span>
            <span class="hero-tag">Binary Classification</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

                                                                                
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ &nbsp;<strong>Educational Use Only</strong> — This tool is not a substitute for professional
        medical advice, diagnosis, or treatment. Always consult a qualified dermatologist.
    </div>
    """, unsafe_allow_html=True)

                                                                                
    st.markdown(
        '<div class="section-label">STEP 1</div>'
        '<div class="section-title">Upload Skin Lesion Image</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Drag & drop or click to browse — supports JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )

                                                                            
    if uploaded_file is not None:
                                                             
        interpreter = st.session_state.get("interpreter")
        llm = st.session_state.get("llm")
        parser = st.session_state.get("parser")

        if interpreter is None or llm is None or parser is None:
            interpreter, llm, parser = load_runtime_components()

        image = Image.open(uploaded_file).convert("RGB")

        col_img, col_results = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown('<div class="img-panel-label">📷 UPLOADED IMAGE</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with col_results:
            st.markdown('<div class="section-label">ANALYSIS RESULTS</div>', unsafe_allow_html=True)

            with st.spinner("Analyzing image with CNN model…"):
                time.sleep(1)
                input_data   = preprocess_image(image)
                input_index  = interpreter.get_input_details()[0]["index"]
                output_index = interpreter.get_output_details()[0]["index"]
                interpreter.set_tensor(input_index, input_data)
                interpreter.invoke()
                prediction           = interpreter.get_tensor(output_index)
                confidence           = float(prediction[0][0])
                predicted_class      = "Malignant" if confidence > 0.5 else "Benign"
                confidence_malignant = confidence * 100
                confidence_benign    = (1 - confidence) * 100

            badge_cls   = "benign" if predicted_class == "Benign" else "malignant"
            badge_icon  = "✅" if predicted_class == "Benign" else "⚠️"
            badge_label = "Benign · Non-Cancerous" if predicted_class == "Benign" else "Malignant · Cancerous"

            st.markdown(
                f'<span class="prediction-badge {badge_cls}">{badge_icon} {badge_label}</span>',
                unsafe_allow_html=True,
            )

            st.markdown(f"""
            <div class="result-card {badge_cls}">
                <div class="conf-sub-label">Confidence Scores</div>
                <div class="conf-bar-container">
                    <div class="conf-label-row">
                        <span class="conf-name">🟢 Benign</span>
                        <span class="conf-value">{confidence_benign:.1f}%</span>
                    </div>
                    <div class="conf-track">
                        <div class="conf-fill benign" style="width:{confidence_benign:.1f}%"></div>
                    </div>
                </div>
                <div class="conf-bar-container" style="margin-bottom:0">
                    <div class="conf-label-row">
                        <span class="conf-name">🔴 Malignant</span>
                        <span class="conf-value">{confidence_malignant:.1f}%</span>
                    </div>
                    <div class="conf-track">
                        <div class="conf-fill malignant" style="width:{confidence_malignant:.1f}%"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

                                                                                
        st.markdown("---")
        st.markdown(
            '<div class="section-label">AI INSIGHT</div>'
            '<div class="section-title">Visual Feature Explanation</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Generating AI explanation…"):
            prompt1 = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """
            You explain model predictions for medical images.

            Describe only the visual features that may have influenced the prediction.

            Rules:
            - Use plain text only.
            - Do NOT use markdown.
            - Do NOT use tables.
            - Do NOT use **, *, #, |, or ---.
            - Keep each point short and clear.
            - Use bullet points.

            Format exactly:

            • Color: ...
            • Texture: ...
            • Shape: ...
            • Borders: ...
            • Pattern: ...
            • Symmetry: ...
                    """
                ),
                (
                    "user",
                    """
            The image is: {image}

            The model predicted: {prediction}

            Explain which visual features likely contributed to this prediction.
                    """
                )
            ])
            chain1      = prompt1 | llm | parser
            explanation = chain1.invoke({"image": image, "prediction": predicted_class}).strip()

                          
            features = ["Color", "Texture", "Shape", "Borders", "Pattern", "Symmetry"]

            for i, feature in enumerate(features, start=1):
                if i == 1:
                    explanation = explanation.replace(
                        f"• {feature}:",
                        f'<strong>{i}. {feature}:</strong>'
                    )
                else:
                    explanation = explanation.replace(
                        f"• {feature}:",
                        f'<br><br><strong>{i}. {feature}:</strong>'
                    )

            st.markdown(
                f'<div class="explanation-box">{explanation}</div>',
                unsafe_allow_html=True
)

                                                                                
        st.markdown("---")
        st.markdown(
            '<div class="section-label">STEP 2 · OPTIONAL</div>'
            '<div class="section-title">Detailed Clinical Report</div>',
            unsafe_allow_html=True,
        )

        if st.button("📋 Generate Detailed Report"):
            with st.spinner("Generating detailed clinical report…"):
                time.sleep(0.5)
                prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "You are a medical AI assistant specialized in dermatology. "
                     "Provide detailed analysis and recommendations based on skin lesion classifications. "
                     "Don't use multiple fonts in your response. Stick to plain text formatting. "
                     "Strictly the report should not look like an AI generated report."
                     "Give the fully detailed report"),
                    ("user",
                     "The model has predicted that the skin lesion is {prediction} with a confidence of {confidence:.2f}%. "
                     "Please provide a detailed report including possible implications, recommended next steps, "
                     "and any precautions the user should take."                     
                     "don't use multiple fonts in your response. Stick to plain text formatting.and strictly don't use any extra space"
                     ),
                ])
                chain  = prompt | llm | parser
                report = chain.invoke({
                    "prediction": predicted_class,
                    "confidence": confidence_malignant if predicted_class == "Malignant" else confidence_benign,
                })

            st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label="⬇️ Download Report",
                data=report,
                file_name="derma_scan_report.txt",
                mime="text/plain",
            )

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🩺</div>
            <div class="empty-state-title">No image uploaded yet</div>
            <div class="empty-state-sub">Upload a dermoscopic skin lesion image above to begin analysis.</div>
        </div>
        """, unsafe_allow_html=True)

                                                                                
    st.markdown("""
    <div class="custom-footer">
        🔬 DermaScan AI &nbsp;·&nbsp; Developed for educational purposes only &nbsp;·&nbsp; Not intended for medical use
    </div>
    """, unsafe_allow_html=True)


                                                                             
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="cnn_full_model.tflite")
    interpreter.allocate_tensors()
    return interpreter


@st.cache_resource
def load_llm_and_parser():
    api_key = st.secrets["GROQ_API_KEY"]
    llm = ChatGroq(api_key=api_key, model="openai/gpt-oss-120b", temperature=0)
    parser = StrOutputParser()
    return llm, parser


@st.cache_data
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)


def load_runtime_components():
    interpreter = load_tflite_model()
    llm, parser = load_llm_and_parser()
    st.session_state["interpreter"] = interpreter
    st.session_state["llm"] = llm
    st.session_state["parser"] = parser
    return interpreter, llm, parser


                                                                                
        
                                                                                
if st.session_state.authenticated:
    show_main_app()
else:
    if st.session_state.auth_page == "signup":
        show_signup_page()
    else:
        show_login_page()



