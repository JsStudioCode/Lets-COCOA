import streamlit as st
from src.theme import CLASS_META, inject_theme

st.set_page_config(page_title="Pod Guide — CocoaPodAI", page_icon="📖", layout="wide")
inject_theme()

st.markdown('<div class="cp-display" style="font-size:2.4rem;">📖 Pod guide</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="cp-muted">What CocoaPodAI is actually looking for in each class.</p>',
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
            <div class="cp-card" style="border-top: 4px solid {meta['color']}; min-height:230px;">
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
    "Note: CPB (Cocoa Pod Borer / black pod disease) can occur at any ripeness "
    "stage — it's shown separately because it's a diagnosis, not a stage in the "
    "ripening timeline.",
    icon="🩺",
)
