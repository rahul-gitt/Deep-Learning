import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Fake Profile Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_resources():
    model = load_model("fake_profile_ann.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_resources()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,0.12), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(168,85,247,0.10), transparent 25%),
        #0b0f19;
}

/* Main content */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* =========================================================
   HEADER
   ========================================================= */

.hero {
    padding: 35px 35px 30px 35px;
    border-radius: 24px;
    background: linear-gradient(
        135deg,
        rgba(30,41,59,0.95),
        rgba(15,23,42,0.95)
    );
    border: 1px solid rgba(148,163,184,0.15);
    box-shadow: 0 20px 60px rgba(0,0,0,0.30);
    margin-bottom: 28px;
}

.hero-badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.25);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin-top: 14px;
    margin-bottom: 8px;
    letter-spacing: -1px;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 16px;
    line-height: 1.6;
}


/* =========================================================
   SECTION
   ========================================================= */

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 15px;
}


/* =========================================================
   INPUT CARDS
   ========================================================= */

div[data-testid="stNumberInput"] {
    background: rgba(15,23,42,0.75);
    padding: 14px;
    border-radius: 14px;
    border: 1px solid rgba(148,163,184,0.12);
}

div[data-testid="stNumberInput"] label {
    font-weight: 600;
}


/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    padding: 14px 20px;
    font-size: 16px;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    transition: 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(99,102,241,0.30);
}


/* =========================================================
   RESULT
   ========================================================= */

.result-card {
    padding: 30px;
    border-radius: 20px;
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(148,163,184,0.15);
    text-align: center;
    margin-top: 25px;
}

.result-title {
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 8px;
}

.result-confidence {
    font-size: 18px;
    color: #94a3b8;
}


/* =========================================================
   METRIC CARDS
   ========================================================= */

.metric-card {
    padding: 20px;
    border-radius: 16px;
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(148,163,184,0.12);
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
}

.metric-value {
    font-size: 25px;
    font-weight: 800;
    margin-top: 5px;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background: #080c14;
    border-right: 1px solid rgba(148,163,184,0.10);
}

.sidebar-title {
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 5px;
}

.sidebar-text {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.6;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    margin-top: 45px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🛡️ Detector</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-text">'
        'AI-powered social profile classification system.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 🤖 Model")

    st.write("**Architecture**  \nArtificial Neural Network")
    st.write("**Task**  \nBinary Classification")
    st.write("**Features**  \n17")
    st.write("**Optimizer**  \nAdam")
    st.write("**Activation**  \nReLU + Sigmoid")

    st.divider()

    st.markdown("### 📊 Performance")

    st.metric("Test Accuracy", "88.64%")

    st.divider()

    st.caption(
        "Built for educational and demonstration purposes."
    )


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<span class="hero-badge">ARTIFICIAL INTELLIGENCE • ANN</span>

<div class="hero-title">
🛡️ Fake Profile Detector
</div>

<div class="hero-subtitle">
Analyze social profile characteristics using a trained
Artificial Neural Network and classify the profile as
Fake or Authentic.
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# PROFILE FEATURES
# =========================================================

st.markdown(
    '<div class="section-title">👤 Profile Characteristics</div>',
    unsafe_allow_html=True
)


# -------- Row 1 --------

col1, col2, col3 = st.columns(3)

with col1:
    pos = st.number_input(
        "Posts",
        min_value=0.0,
        value=0.0
    )

with col2:
    flw = st.number_input(
        "Followers",
        min_value=0.0,
        value=0.0
    )

with col3:
    flg = st.number_input(
        "Following",
        min_value=0.0,
        value=0.0
    )


# -------- Row 2 --------

col1, col2, col3 = st.columns(3)

with col1:
    bl = st.number_input(
        "Bio Length",
        min_value=0.0,
        value=0.0
    )

with col2:
    pic = st.number_input(
        "Profile Picture",
        min_value=0.0,
        value=0.0
    )

with col3:
    lin = st.number_input(
        "External Link",
        min_value=0.0,
        value=0.0
    )


# -------- Row 3 --------

col1, col2, col3 = st.columns(3)

with col1:
    cl = st.number_input(
        "CL",
        min_value=0.0,
        value=0.0
    )

with col2:
    cz = st.number_input(
        "CZ",
        min_value=0.0,
        value=0.0
    )

with col3:
    ni = st.number_input(
        "NI",
        min_value=0.0,
        value=0.0
    )


# -------- Row 4 --------

col1, col2, col3 = st.columns(3)

with col1:
    erl = st.number_input(
        "ERL",
        min_value=0.0,
        value=0.0
    )

with col2:
    erc = st.number_input(
        "ERC",
        min_value=0.0,
        value=0.0
    )

with col3:
    lt = st.number_input(
        "LT",
        min_value=0.0,
        value=0.0
    )


# -------- Row 5 --------

col1, col2, col3 = st.columns(3)

with col1:
    hc = st.number_input(
        "HC",
        min_value=0.0,
        value=0.0
    )

with col2:
    pr = st.number_input(
        "Private",
        min_value=0.0,
        value=0.0
    )

with col3:
    fo = st.number_input(
        "Followers / Following",
        min_value=0.0,
        value=0.0
    )


# -------- Row 6 --------

col1, col2 = st.columns(2)

with col1:
    cs = st.number_input(
        "CS",
        min_value=0.0,
        value=0.0
    )

with col2:
    pi = st.number_input(
        "PI",
        min_value=0.0,
        value=0.0
    )


# =========================================================
# PREDICT BUTTON
# =========================================================

st.write("")

predict = st.button(
    "🔍  ANALYZE PROFILE",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict:

    features = np.array([[
        pos,
        flw,
        flg,
        bl,
        pic,
        lin,
        cl,
        cz,
        ni,
        erl,
        erc,
        lt,
        hc,
        pr,
        fo,
        cs,
        pi
    ]])

    # Scale
    features_scaled = scaler.transform(features)

    # Prediction
    probability = model.predict(
        features_scaled,
        verbose=0
    )[0][0]

    # Classification
    if probability >= 0.5:
        prediction = "AUTHENTIC"
        confidence = probability
        icon = "✅"
    else:
        prediction = "FAKE"
        confidence = 1 - probability
        icon = "🚨"


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown(
        '<div class="section-title">📋 Analysis Result</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="result-card">

        <div style="font-size:45px;">
        {icon}
        </div>

        <div class="result-title">
        {prediction}
        </div>

        <div class="result-confidence">
        Prediction confidence: <b>{confidence:.2%}</b>
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.write("")

    # =====================================================
    # RESULT METRICS
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
            <div class="metric-label">Prediction</div>
            <div class="metric-value">{prediction}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{confidence:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value">ANN</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")

    # Confidence bar

    st.write("**Prediction Confidence**")

    st.progress(float(confidence))


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
    Fake Profile Detector • Artificial Neural Network •
    Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)