import streamlit as st
from src.theme import CLASS_META, inject_theme

st.set_page_config(page_title="Pod Guide — CocoaPodAI", page_icon="📖", layout="wide")
inject_theme()

st.markdown('<div class="cp-display" style="font-size:2.4rem;">Pod Guide 📖</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="cp-muted">What CocoaPodAI Is Actually Looking For In Each Class.</p>',
    unsafe_allow_html=True,
)
st.write("")

order = ["UR", "R", "OR", "CPB"]
cols = st.columns(4, gap="medium")

for col, code in zip(cols, order):
    meta = CLASS_META[code]
    with col:
        st.markdown(
            f"""
            <div class="cp-card" style="border-top: 5px solid {meta['color']}; min-height:230px;">
                <div class="cp-mono" style="color:{meta['color']}; font-size:0.85rem; margin-bottom:0.4rem;">
                    {code}
                </div>
                <div style="font-weight:600; font-size:1.1rem; margin-bottom:0.6rem;">
                    {meta['label']}
                </div>
                <div class="cp-muted" style="font-size:0.9rem; line-height:1.5;">
                    {meta['blurb']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
st.info(
    "Note: CPB (Cocoa Pod Borer / Black Pod Disease) Can Occur At Any Ripeness "
    "Stage — It's Shown Separately Because It's A Diagnosis, Not A Stage In The "
    "Ripening Timeline.",
    icon="🩺",
)
