import streamlit as st
from src.theme import inject_theme, ripeness_spectrum_html, COLORS

st.set_page_config(
    page_title="CocoaPodAI — Let's Cocoa",
    page_icon="🌱",
    layout="wide",
)

inject_theme()

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
left, right = st.columns([1.3, 1], gap="large")

with left:
    st.markdown(
        """
        <div class="cp-mono" style="color:#E3A62F; letter-spacing:0.12em; font-size:0.85rem;
                    margin-bottom:0.6rem;">
            RGB + THERMAL COCOA POD CLASSIFICATION
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="cp-display" style="font-size:4.2rem; line-height:1.05; margin-bottom:0.6rem;">
            Let's Cocoa 🍫
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p style="font-size:1.15rem; color:#B9A990; max-width:36rem; line-height:1.6;">
        Point a camera at a pod. CocoaPodAI reads its rind color and its heat
        signature together, and tells you what it's looking at — unripe, ripe,
        overripe, or diseased — in seconds.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(ripeness_spectrum_html(), unsafe_allow_html=True)

    st.page_link("pages/1_🔬_Predict.py", label="Classify a pod →", icon="🔬")

with right:
    st.markdown(
        f"""
        <div class="cp-card" style="height:100%;">
            <div class="cp-mono" style="color:{COLORS['muted']}; font-size:0.8rem; margin-bottom:0.8rem;">
                HOW IT DECIDES
            </div>
            <div style="display:flex; flex-direction:column; gap:1rem;">
                <div>
                    <b>RGB image</b><br>
                    <span class="cp-muted">Rind color, texture, and shape — the same cues a farmer reads by eye.</span>
                </div>
                <div>
                    <b>Thermal image</b><br>
                    <span class="cp-muted">Surface heat pattern — signal the eye alone can't see.</span>
                </div>
                <div>
                    <b>Classifier</b><br>
                    <span class="cp-muted">An SVM trained on both signals together outputs a class and a confidence score.</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Feature strip
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns(3, gap="large")
features = [
    ("🩺", "Catches disease early", "CPB shows up in thermal patterns before it's obvious in the rind."),
    ("⚡", "Instant result", "Upload two photos, get a class and a confidence score back immediately."),
    ("🌾", "Built for the field", "Trained on real farm pods, not lab-perfect samples."),
]
for col, (icon, title, body) in zip((c1, c2, c3), features):
    with col:
        st.markdown(
            f"""
            <div class="cp-card">
                <div style="font-size:1.6rem;">{icon}</div>
                <div style="font-weight:600; margin:0.5rem 0 0.3rem 0;">{title}</div>
                <div class="cp-muted" style="font-size:0.92rem;">{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    f"""<p class="cp-muted" style="text-align:center; margin-top:2rem; font-size:0.85rem;">
    Want to know what each class actually looks like?
    </p>""",
    unsafe_allow_html=True,
)
mid = st.columns([1, 1, 1])[1]
with mid:
    st.page_link("pages/2_📖_Pod_Guide.py", label="See the pod guide →", icon="📖")
