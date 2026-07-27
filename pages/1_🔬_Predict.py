import re
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.feature_extraction import (
    extract_hsv_features,
    extract_rgb_features,
    extract_thermal_features,
)

from src.theme import (
    CLASS_META,
    badge_html,
    inject_theme,
)

st.set_page_config(
    page_title="Predict — CocoaPodAI",
    page_icon="🔬",
    layout="wide",
)

inject_theme()


CLASSES = ["CPB", "OR", "R", "UR"]


@st.cache_resource
def load_model():
    return joblib.load("models/svm_single.pkl")


model = load_model()


def preprocess_rgb(uploaded_file):
    rgb_pil = Image.open(uploaded_file)

    if rgb_pil.mode == "RGBA":
        rgb = cv2.cvtColor(
            np.array(rgb_pil),
            cv2.COLOR_RGBA2BGRA,
        )
    else:
        rgb = cv2.cvtColor(
            np.array(rgb_pil.convert("RGB")),
            cv2.COLOR_RGB2BGR,
        )

    return rgb


def preprocess_thermal(uploaded_file):
    thermal = cv2.cvtColor(
        np.array(Image.open(uploaded_file).convert("RGB")),
        cv2.COLOR_RGB2BGR,
    )

    return thermal


def predict_pod(rgb_file, thermal_file):

    rgb = preprocess_rgb(rgb_file)
    thermal = preprocess_thermal(thermal_file)

    features = (
        extract_rgb_features(rgb)
        + extract_hsv_features(rgb)
        + extract_thermal_features(thermal)
    )

    prediction = model.predict([features])[0]
    if hasattr(model, "predict_proba"):
        confidence = np.max(model.predict_proba([features])[0]) * 100
    else:
        confidence = 100.0
    return CLASSES[prediction], confidence


def normalize_name(filename):
    """
    CPB1.jpg  -> CPB1
    CPB1F.jpg -> CPB1
    CPB1B.jpg -> CPB1
    """

    name = filename.rsplit(".", 1)[0]
    name = re.sub(r"[FfBb]$", "", name)

    return name


st.markdown(
    '<div class="cp-display" style="font-size:2.4rem;">🔬 Classify a Pod</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="cp-muted">Choose single prediction or batch prediction.</p>',
    unsafe_allow_html=True,
)

st.write("")

tab_single, tab_batch = st.tabs(
    [
        "🖼 Single Image",
        "📁 Batch Folder",
    ]
)
with tab_single:

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="cp-card">', unsafe_allow_html=True)
        st.markdown("**RGB Image**")
        rgb_file = st.file_uploader(
            "RGB Image",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key="single_rgb",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="cp-card">', unsafe_allow_html=True)
        st.markdown("**Thermal Image (Front or Back)**")
        thermal_file = st.file_uploader(
            "Thermal Image",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key="single_thermal",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    predict_clicked = st.button(
        "Predict",
        type="primary",
        key="single_predict",
    )

    if predict_clicked:

        if rgb_file is None:
            st.error("Please upload the RGB image.")
            st.stop()

        if thermal_file is None:
            st.error("Please upload a thermal image.")
            st.stop()

        with st.spinner("Reading pod signals..."):

            pred_class, confidence = predict_pod(
                rgb_file,
                thermal_file,
            )

        st.divider()

        meta = CLASS_META[pred_class]

        r1, r2 = st.columns([1, 1], gap="large")

        with r1:
            st.html(
                f"""
<div class="cp-card">
    <div class="cp-muted cp-mono"
         style="font-size:0.78rem;margin-bottom:0.6rem;">
        PREDICTION
    </div>

    {badge_html(pred_class)}

    <p style="margin-top:0.9rem;color:#B9A990;">
        {meta['blurb']}
    </p>
</div>
"""
            )

        with r2:
            st.html(
                f"""
<div class="cp-card">
    <div class="cp-muted cp-mono"
         style="font-size:0.78rem;margin-bottom:0.6rem;">
        CONFIDENCE
    </div>

    <div class="cp-mono"
         style="font-size:2.2rem;color:{meta['color']};">
        {confidence:.2f}%
    </div>
</div>
"""
            )

        st.write("")

        c1, c2 = st.columns(2)

        with c1:
            st.image(
                rgb_file,
                caption="RGB Image",
            )

        with c2:
            st.image(
                thermal_file,
                caption="Thermal Image",
            )


with tab_batch:

    st.markdown("### 📁 Batch Folder Prediction")
    st.write("Upload all RGB images and all Thermal images.")

    rgb_files = st.file_uploader(
        "RGB Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_rgb",
    )

    thermal_files = st.file_uploader(
        "Thermal Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_thermal",
    )

    predict_all = st.button(
        "Predict All",
        type="primary",
        key="batch_predict",
    )

    if predict_all:

        if len(rgb_files) == 0:
            st.error("Please upload RGB images.")
            st.stop()

        if len(thermal_files) == 0:
            st.error("Please upload Thermal images.")
            st.stop()

        rgb_dict = {
            normalize_name(file.name): file
            for file in rgb_files
        }

        thermal_dict = {
            normalize_name(file.name): file
            for file in thermal_files
        }

        common = sorted(
            set(rgb_dict.keys()) &
            set(thermal_dict.keys())
        )

        if len(common) == 0:
            st.error("No matching RGB and Thermal image names found.")
            st.stop()

        progress = st.progress(0)

        results = []

        total = len(common)

        for i, name in enumerate(common):

            pred_class, confidence = predict_pod(
                rgb_dict[name],
                thermal_dict[name],
            )

            results.append(
                {
                    "Pod": name,
                    "Prediction": pred_class,
                    "Confidence (%)": round(confidence, 2),
                }
            )

            progress.progress((i + 1) / total)

        st.success(f"Finished predicting {total} cocoa pods.")

        df = pd.DataFrame(results)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download CSV",
            csv,
            file_name="batch_prediction_results.csv",
            mime="text/csv",
        )           