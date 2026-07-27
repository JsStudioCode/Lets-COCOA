"""
Shared visual theme for CocoaPodAI.
Import inject_theme() at the top of every page, right after st.set_page_config().
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

COLORS = {
    "bg": "#17100B",           # page background — deep cacao brown, near black
    "surface": "#241811",      # card background
    "surface_hi": "#3D2A1D",   # borders / hover surfaces
    "cream": "#F5EDE0",        # primary text
    "muted": "#B9A990",        # secondary text
    "gold": "#E3A62F",         # ripe / primary accent
    "green": "#7C9A5C",        # unripe
    "rust": "#C1442E",         # diseased (CPB)
    "bark": "#8B5A2B",         # overripe
}

CLASS_META = {
    "UR": {
        "label": "Unripe",
        "color": COLORS["green"],
        "on_color": "#152008",
        "blurb": "Green rind, firm shell, higher surface temperature because of greater water content and active growth. Sugar accumulation is still ongoing.",
    },
    "R": {
        "label": "Ripe",
        "color": COLORS["gold"],
        "on_color": "#2A1B04",
        "blurb": "Golden-orange rind. Peak sugar content — ready to harvest.",
    },
    "OR": {
        "label": "Overripe",
        "color": COLORS["bark"],
        "on_color": "#F5EDE0",
        "blurb": "Deep brown rind, shell starting to dry and split. Past optimal harvest.",
    },
    "CPB": {
        "label": "Diseased (CPB)",
        "color": COLORS["rust"],
        "on_color": "#FBEAE4",
        "blurb": "Cocoa pod borer / black pod symptoms — lesions and irregular thermal signature.",
    },
}


def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Public+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {{
            --cp-bg: {COLORS["bg"]};
            --cp-surface: {COLORS["surface"]};
            --cp-surface-hi: {COLORS["surface_hi"]};
            --cp-cream: {COLORS["cream"]};
            --cp-muted: {COLORS["muted"]};
            --cp-gold: {COLORS["gold"]};
            --cp-green: {COLORS["green"]};
            --cp-rust: {COLORS["rust"]};
            --cp-bark: {COLORS["bark"]};
        }}

        html, body, [class*="css"] {{
            font-family: 'Public Sans', sans-serif;
        }}

        .stApp {{
            background: var(--cp-bg);
            color: var(--cp-cream);
        }}

        section[data-testid="stSidebar"] {{
            background: var(--cp-surface);
            border-right: 1px solid var(--cp-surface-hi);
        }}

        h1, h2, h3, .cp-display {{
            font-family: 'Fraunces', serif;
            font-weight: 600;
            color: var(--cp-cream);
            letter-spacing: -0.01em;
        }}

        p, li, span, label {{
            color: var(--cp-cream);
        }}

        .cp-muted {{ color: var(--cp-muted) !important; }}

        .cp-mono {{
            font-family: 'IBM Plex Mono', monospace;
        }}

        /* Buttons */
        .stButton > button, .stDownloadButton > button {{
            background: var(--cp-gold);
            color: #2A1B04;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.6rem 1.4rem;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(227, 166, 47, 0.25);
            color: #2A1B04;
        }}

        /* File uploader */
        [data-testid="stFileUploaderDropzone"] {{
            background: var(--cp-surface);
            border: 1.5px dashed var(--cp-surface-hi);
            border-radius: 12px;
        }}

        /* Generic card */
        .cp-card {{
            background: var(--cp-surface);
            border: 1px solid var(--cp-surface-hi);
            border-radius: 16px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1rem;
        }}

        /* Class badge pill */
        .cp-badge {{
            display: inline-block;
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 500;
            font-size: 0.85rem;
            padding: 0.3rem 0.85rem;
            border-radius: 999px;
            letter-spacing: 0.02em;
        }}

        hr {{ border-color: var(--cp-surface-hi); }}

        a, a:visited {{ color: var(--cp-gold); }}

        /* Reduced motion */
        @media (prefers-reduced-motion: reduce) {{
            * {{ transition: none !important; animation: none !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge_html(class_code: str) -> str:
    meta = CLASS_META[class_code]
    return (
        f'<span class="cp-badge" style="background:{meta["color"]};'
        f'color:{meta["on_color"]};">{class_code} · {meta["label"]}</span>'
    )


def ripeness_spectrum_html() -> str:
    """The signature element: a real ripening timeline, with disease shown
    as a branch off the sequence rather than a step within it."""
    ur, r, orp, cpb = (COLORS["green"], COLORS["gold"], COLORS["bark"], COLORS["rust"])
    return f"""
    <div style="margin: 2.2rem 0 1.4rem 0;">
      <div style="display:flex; align-items:center; gap:0; height:14px; border-radius:8px;
                  overflow:hidden; box-shadow: inset 0 0 0 1px {COLORS['surface_hi']};">
        <div style="flex:1; background:{ur};"></div>
        <div style="flex:1; background:{r};"></div>
        <div style="flex:1; background:{orp};"></div>
      </div>
      <div style="display:flex; justify-content:space-between; margin-top:0.5rem;
                  font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:{COLORS['muted']};">
        <span>UNRIPE</span><span>RIPE</span><span>OVERRIPE</span>
      </div>
      <div style="display:flex; align-items:center; gap:0.6rem; margin-top:1rem;
                  padding-top:1rem; border-top:1px dashed {COLORS['surface_hi']};">
        <div style="width:10px; height:10px; border-radius:50%; background:{cpb};
                    box-shadow:0 0 0 3px rgba(193,68,46,0.18);"></div>
        <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:{cpb};">
          CPB — off the timeline. Disease, not a ripening stage.
        </span>
      </div>
    </div>
    """
