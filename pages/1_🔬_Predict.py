import base64
import re
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

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
from src.theme import CLASS_META, badge_html, inject_theme

st.set_page_config(
    page_title="Predict — CocoaPodAI",
    page_icon="🔬",
    layout="wide",
)

inject_theme()

CLASSES = ["CPB", "OR", "R", "UR"]
BASE_DIR = Path(__file__).resolve().parent.parent


@st.cache_resource
def load_model():
    model_path = BASE_DIR / "models" / "svm_single.pkl"
    if not model_path.exists():
        model_path = Path("models/svm_single.pkl")
    return joblib.load(model_path)


model = load_model()


def preprocess_rgb(uploaded_file):
    rgb_pil = Image.open(uploaded_file)

    if cv2 is not None:
        if rgb_pil.mode == "RGBA":
            rgb = cv2.cvtColor(np.array(rgb_pil), cv2.COLOR_RGBA2BGRA)
        else:
            rgb = cv2.cvtColor(np.array(rgb_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        rgb = np.array(rgb_pil.convert("RGB"))[:, :, ::-1]

    return rgb


def preprocess_thermal(uploaded_file):
    if cv2 is not None:
        thermal = cv2.cvtColor(
            np.array(Image.open(uploaded_file).convert("RGB")),
            cv2.COLOR_RGB2BGR,
        )
    else:
        thermal = np.array(Image.open(uploaded_file).convert("RGB"))[:, :, ::-1]

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
        proba = model.predict_proba([features])[0]
    else:
        proba = np.zeros(len(CLASSES))
        proba[prediction] = 1.0

    confidence = proba[prediction] * 100
    return CLASSES[prediction], confidence, proba


def second_best_class(proba, top_class):
    """Return (class_name, confidence_pct) for the runner-up prediction."""
    order = np.argsort(proba)[::-1]
    for idx in order:
        cls = CLASSES[idx]
        if cls != top_class:
            return cls, proba[idx] * 100
    return None, 0.0


def normalize_name(filename):
    """
    CPB1.jpg  -> CPB1
    CPB1F.jpg -> CPB1
    CPB1B.jpg -> CPB1
    """
    name = filename.rsplit(".", 1)[0]
    name = re.sub(r"[FfBb]$", "", name)
    return name


def image_box_html(uploaded_file, caption):
    """Render an image inside a fixed-height, fixed-width box (object-fit:
    cover) so RGB and thermal previews line up evenly regardless of their
    native aspect ratio."""
    uploaded_file.seek(0)
    b64 = base64.b64encode(uploaded_file.read()).decode()
    uploaded_file.seek(0)
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    return f"""
<div style="width:100%;">
    <div style="width:100%; height:300px; border-radius:16px; overflow:hidden;
                border:1px solid #3D2A1D;">
        <img src="data:image/{mime};base64,{b64}"
             style="width:100%; height:100%; object-fit:cover; display:block;" />
    </div>
    <div class="cp-muted" style="text-align:center; margin-top:0.5rem; font-size:0.88rem;">
        {caption}
    </div>
</div>
"""


st.markdown(
    '<div class="cp-display" style="font-size:2.4rem;">Classify A Pod!</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="cp-muted">Choose Single Prediction Or Batch Prediction.</p>',
    unsafe_allow_html=True,
)
st.write("")

tab_single, tab_batch = st.tabs(["🖼️ Single Cocoa Pod", "📁 Multiple Cocoa Pod"])

# =============================================================================
# SINGLE
# =============================================================================
with tab_single:

    st.markdown(
        '<p class="cp-muted" style="margin-top:0.6rem;">'
        "Upload A Cocoa Pod's RGB And Thermal Images</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("**RGB Image**")
        rgb_file = st.file_uploader(
            "RGB Image",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key="single_rgb",
        )

    with col2:
        st.markdown("**Thermal Image (Front Or Back)**")
        thermal_file = st.file_uploader(
            "Thermal Image",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key="single_thermal",
        )

    st.write("")
    predict_clicked = st.button("Predict", type="primary", key="single_predict")

    if predict_clicked:

        if rgb_file is None:
            st.error("Please Upload The RGB Image.")
            st.stop()

        if thermal_file is None:
            st.error("Please Upload A Thermal Image.")
            st.stop()

        with st.spinner("Reading Pod Signals..."):
            pred_class, confidence, proba = predict_pod(rgb_file, thermal_file)

        st.divider()

        meta = CLASS_META[pred_class]

        r1, r2 = st.columns([1, 1], gap="large")

        with r1:
            st.html(
                f"""
<div class="cp-card">
    <div class="cp-muted cp-mono" style="font-size:0.78rem;margin-bottom:0.6rem;">
        PREDICTION
    </div>
    {badge_html(pred_class)}
    <p class="cp-muted" style="margin-top:0.9rem;">
        {meta['blurb']}
    </p>
</div>
"""
            )

        with r2:
            st.html(
                f"""
<div class="cp-card">
    <div class="cp-muted cp-mono" style="font-size:0.78rem;margin-bottom:0.6rem;">
        CONFIDENCE SCORE
    </div>
    <div class="cp-mono" style="font-size:2.2rem;color:{meta['color']};">
        {confidence:.2f}%
    </div>
</div>
"""
            )

        if confidence < 85:
            alt_class, alt_conf = second_best_class(proba, pred_class)
            if alt_class:
                alt_meta = CLASS_META[alt_class]
                st.html(
                    f"""
<div class="cp-card">
    <div class="cp-muted cp-mono" style="font-size:0.78rem;margin-bottom:0.4rem;">
        COULD ALSO BE
    </div>
    <span style="color:{alt_meta['color']}; font-weight:600;">{alt_class} · {alt_meta['label']}</span>
    <span class="cp-muted"> — {alt_conf:.1f}% Confidence</span>
</div>
"""
                )

        st.write("")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.html(image_box_html(rgb_file, "RGB Image Of The Uploaded One"))
        with c2:
            st.html(image_box_html(thermal_file, "Thermal Image Of The Uploaded One"))

# =============================================================================
# BATCH
# =============================================================================
with tab_batch:

    st.markdown(
        '<p class="cp-muted" style="margin-top:0.6rem;">'
        "Upload RGB And Thermal Images From Multiple Cocoa Pods</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("**RGB Images**")
        rgb_files = st.file_uploader(
            "RGB Images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="batch_rgb",
        )

    with col2:
        st.markdown("**Thermal Images**")
        thermal_files = st.file_uploader(
            "Thermal Images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="batch_thermal",
        )

    st.write("")
    predict_all = st.button("Predict", type="primary", key="batch_predict")

    if predict_all:

        if len(rgb_files) == 0:
            st.error("Please Upload RGB Images.")
            st.stop()

        if len(thermal_files) == 0:
            st.error("Please Upload Thermal Images.")
            st.stop()

        if len(rgb_files) != len(thermal_files):
            if len(rgb_files) < len(thermal_files):
                st.warning(
                    f"Less Number Of RGB Images Has Been Uploaded. "
                    f"You Uploaded **{len(rgb_files)} RGB Image(s)** And "
                    f"**{len(thermal_files)} Thermal Image(s)**. Please Upload "
                    f"{len(thermal_files) - len(rgb_files)} More RGB Image(s) "
                    f"So Both Groups Match."
                )
            else:
                st.warning(
                    f"Less Number Of Thermal Images Has Been Uploaded. "
                    f"You Uploaded **{len(rgb_files)} RGB Image(s)** And "
                    f"**{len(thermal_files)} Thermal Image(s)**. Please Upload "
                    f"{len(rgb_files) - len(thermal_files)} More Thermal Image(s) "
                    f"So Both Groups Match."
                )
            st.stop()

        rgb_dict = {normalize_name(f.name): f for f in rgb_files}
        thermal_dict = {normalize_name(f.name): f for f in thermal_files}

        common = sorted(set(rgb_dict.keys()) & set(thermal_dict.keys()))

        if len(common) == 0:
            st.error("No Matching RGB And Thermal Image Names Found.")
            st.stop()

        progress = st.progress(0)
        results = []
        total = len(common)

        for i, name in enumerate(common):

            pred_class, confidence, proba = predict_pod(
                rgb_dict[name],
                thermal_dict[name],
            )

            if confidence < 85:
                alt_class, alt_conf = second_best_class(proba, pred_class)
                alt_text = f"{alt_class} ({alt_conf:.1f}%)" if alt_class else ""
            else:
                alt_text = ""

            results.append(
                {
                    "Pod": name,
                    "Classification": pred_class,
                    "Confidence (%)": round(confidence, 2),
                    "Second Likely Class": alt_text,
                }
            )

            progress.progress((i + 1) / total)

        st.success(f"Finished Predicting {total} Cocoa Pods.")

        df = pd.DataFrame(results)

        st.dataframe(df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download CSV",
            csv,
            file_name="batch_prediction_results.csv",
            mime="text/csv",
        )
