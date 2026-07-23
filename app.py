import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

from src.feature_extraction import (
    extract_rgb_features,
    extract_hsv_features,
    extract_thermal_features,
    extract_all_features
)

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(
    page_title="CocoaPodAI",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 CocoaPodAI")
st.subheader("AI-Based Cocoa Pod Classification")

# --------------------------------
# Load Models
# --------------------------------
full_model = joblib.load("models/svm_full.pkl")
single_model = joblib.load("models/svm_single.pkl")

classes = ["CPB", "OR", "R", "UR"]

# --------------------------------
# Upload Images
# --------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    rgb_file = st.file_uploader(
        "RGB Image",
        type=["png", "jpg", "jpeg"]
    )

with col2:
    front_file = st.file_uploader(
        "Thermal Front (Optional)",
        type=["png", "jpg", "jpeg"]
    )

with col3:
    back_file = st.file_uploader(
        "Thermal Back (Optional)",
        type=["png", "jpg", "jpeg"]
    )

# --------------------------------
# Predict
# --------------------------------
if st.button("Predict"):

    # RGB required
    if rgb_file is None:
        st.error("Please upload the RGB image.")
        st.stop()

    # At least one thermal image required
    if front_file is None and back_file is None:
        st.error("Please upload at least one thermal image.")
        st.stop()

    rgb = cv2.cvtColor(
        np.array(Image.open(rgb_file)),
        cv2.COLOR_RGB2BGR
    )

    # ===========================
    # BOTH THERMAL IMAGES
    # ===========================
    if front_file is not None and back_file is not None:

        front = cv2.cvtColor(
            np.array(Image.open(front_file)),
            cv2.COLOR_RGB2BGR
        )

        back = cv2.cvtColor(
            np.array(Image.open(back_file)),
            cv2.COLOR_RGB2BGR
        )

        features = extract_all_features(
            rgb,
            front,
            back
        )

        model = full_model

    # ===========================
    # ONLY FRONT
    # ===========================
    elif front_file is not None:

        front = cv2.cvtColor(
            np.array(Image.open(front_file)),
            cv2.COLOR_RGB2BGR
        )

        features = (
            extract_rgb_features(rgb)
            + extract_hsv_features(rgb)
            + extract_thermal_features(front)
        )

        model = single_model

    # ===========================
    # ONLY BACK
    # ===========================
    else:

        back = cv2.cvtColor(
            np.array(Image.open(back_file)),
            cv2.COLOR_RGB2BGR
        )

        features = (
            extract_rgb_features(rgb)
            + extract_hsv_features(rgb)
            + extract_thermal_features(back)
        )

        model = single_model

    # --------------------------------
    # Prediction
    # --------------------------------
    prediction = model.predict([features])[0]

    confidence = np.max(
        model.predict_proba([features])[0]
    ) * 100

    st.success(f"Prediction: {classes[prediction]}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(rgb_file, caption="RGB Image")

    with c2:
        if front_file:
            st.image(front_file, caption="Thermal Front")

    with c3:
        if back_file:
            st.image(back_file, caption="Thermal Back")