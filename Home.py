import streamlit as st
from src.theme import inject_theme, COLORS

st.set_page_config(
    page_title="CocoaPodAI — Let's Cocoa",
    page_icon="🌱",
    layout="wide",
)

inject_theme()

# ---------------------------------------------------------------------------
# Nav
# ---------------------------------------------------------------------------
nav_l, nav_pod, nav_home = st.columns([5, 1.3, 1])
with nav_pod:
    st.page_link("pages/2_📖_Pod_Guide.py", label="Pod Guide", use_container_width=True)
with nav_home:
    st.page_link("Home.py", label="Home", use_container_width=True)

st.write("")
st.write("")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; max-width:44rem; margin:0 auto;">
        <div class="cp-display" style="font-size:3.6rem; margin-bottom:0.6rem;">
            🍫 Let's Cocoa!
        </div>
        <p class="cp-muted" style="font-size:1.15rem; line-height:1.6;">
            Ready To Discover Your Cocoa Pod's Story?<br>
            Upload Its RGB And Thermal Images And Let CocoaPodAI Reveal Its Stage.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
mid = st.columns([1, 1, 1])[1]
with mid:
    st.page_link(
        "pages/1_🔬_Predict.py",
        label="Predict Pod",
        use_container_width=True,
    )

st.write("")
st.write("")
st.divider()
st.write("")

# ---------------------------------------------------------------------------
# How It Decides
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="cp-display" style="font-size:1.9rem; text-align:center; margin-bottom:1.6rem;">'
    'How It Decides</div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3, gap="large")
steps = [
    ("🖼️", "RGB Image", "Colour, Texture, And Shape — Visual Cues That Reveal The Pod's Appearance And Stage Of Development."),
    ("🌡️", "Thermal Image", "Surface Heat Patterns — Thermal Clues That Reveal Differences Invisible To The Eye."),
    ("🧠", "Classifier", "The SVM Brings Both Sets Of Signals Together, Compares Their Learned Patterns, And Predicts The Pod's Class With A Confidence Score."),
]
for col, (icon, title, body) in zip((c1, c2, c3), steps):
    with col:
        st.markdown(
            f"""
            <div class="cp-card" style="height:100%;">
                <div style="font-size:1.7rem; margin-bottom:0.5rem;">{icon}</div>
                <div style="font-weight:600; font-size:1.05rem; margin-bottom:0.5rem;">{title}</div>
                <div class="cp-muted" style="font-size:0.92rem; line-height:1.55;">{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
st.write("")

# ---------------------------------------------------------------------------
# Feature strip
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns(3, gap="large")
features = [
    ("🩺", "Catches Disease Early", "CPB Shows Up In Thermal Patterns Before It's Obvious In The Rind."),
    ("⚡", "Instant Result", "Upload Two Photos, Get A Class And A Confidence Score Back Immediately."),
    ("🌾", "Built For The Field", "Trained On Real Farm Pods, Not Lab-Perfect Samples."),
]
for col, (icon, title, body) in zip((c1, c2, c3), features):
    with col:
        st.markdown(
            f"""
            <div class="cp-card">
                <div style="font-size:1.5rem;">{icon}</div>
                <div style="font-weight:600; margin:0.5rem 0 0.3rem 0;">{title}</div>
                <div class="cp-muted" style="font-size:0.9rem;">{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
st.markdown(
    '<p class="cp-muted" style="text-align:center; margin-top:1.2rem; font-size:0.9rem;">'
    "Want To Know What Each Class Actually Looks Like?</p>",
    unsafe_allow_html=True,
)
mid = st.columns([1, 1, 1])[1]
with mid:
    st.page_link("pages/2_📖_Pod_Guide.py", label="See The Pod Guide →", icon="📖", use_container_width=True)
