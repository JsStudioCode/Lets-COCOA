import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image

from src.feature_extraction import (
    extract_hsv_features,
    extract_rgb_features,
    extract_thermal_features,
)
from src.theme import CLASS_META, badge_html, inject_theme

st.set_page_config(page_title="Predict — CocoaPodAI", page_icon="🔬", layout="wide")
inject_theme()

CLASSES = ["CPB", "OR", "R", "UR"]


@st.cache_resource
def load_model():
    return joblib.load("models/svm_single.pkl")


model = load_model()

st.markdown('<div class="cp-display" style="font-size:2.4rem;">🔬 Classify a pod</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="cp-muted">Upload a matching RGB and thermal photo of the same pod.</p>',
    unsafe_allow_html=True,
)
st.write("")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="cp-card">', unsafe_allow_html=True)
    st.markdown("**RGB Image**")
    rgb_file = st.file_uploader("RGB Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="cp-card">', unsafe_allow_html=True)
    st.markdown("**Thermal Image (front or back)**")
    thermal_file = st.file_uploader(
        "Thermal Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
predict_clicked = st.button("Predict", type="primary")

if predict_clicked:
    if rgb_file is None:
        st.error("Please upload the RGB image.")
        st.stop()
    if thermal_file is None:
        st.error("Please upload a thermal image.")
        st.stop()

    with st.spinner("Reading pod signals..."):
        rgb_pil = Image.open(rgb_file)
        if rgb_pil.mode == "RGBA":
            rgb = cv2.cvtColor(np.array(rgb_pil), cv2.COLOR_RGBA2BGRA)
        else:
            rgb = cv2.cvtColor(np.array(rgb_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

        thermal = cv2.cvtColor(
            np.array(Image.open(thermal_file).convert("RGB")), cv2.COLOR_RGB2BGR
        )

        features = (
            extract_rgb_features(rgb) + extract_hsv_features(rgb) + extract_thermal_features(thermal)
        )

        prediction = model.predict([features])[0]
        confidence = np.max(model.predict_proba([features])[0]) * 100
        pred_class = CLASSES[prediction]

    st.divider()

    meta = CLASS_META[pred_class]
    r1, r2 = st.columns([1, 1], gap="large")
    with r1:
        st.markdown(
            f"""
            <div class="cp-card">
                <div class="cp-muted cp-mono" style="font-size:0.78rem; margin-bottom:0.6rem;">PREDICTION</div>
                {badge_html(pred_class)}
                <p style="margin-top:0.9rem; color:#B9A990;">{meta['blurb']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            f"""
            <div class="cp-card">
                <div class="cp-muted cp-mono" style="font-size:0.78rem; margin-bottom:0.6rem;">CONFIDENCE</div>
                <div class="cp-mono" style="font-size:2.2rem; color:{meta['color']};">{confidence:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.image(rgb_file, caption="RGB Image")
    with c2:
        st.image(thermal_file, caption="Thermal Image")
