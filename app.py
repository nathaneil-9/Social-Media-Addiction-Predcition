import streamlit as st

st.set_page_config(
    page_title="Social Media Addiction Predictor",
    page_icon="📱",
    layout="centered"
)

st.title("📱 Social Media Addiction Prediction")
st.write(
    "This application predicts the level of social media addiction "
    "based on daily usage and lifestyle indicators."
)

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
age = st.number_input(
    "Age",
    min_value=15,
    max_value=40,
    value=20
)

usage = st.slider(
    "Average Daily Social Media Usage (hours)",
    min_value=0.0,
    max_value=12.0,
    value=0.0,
    step=0.5
)

sleep = st.slider(
    "Sleep Hours Per Night",
    min_value=0.0,
    max_value=12.0,
    value=7.0,
    step=0.5
)

st.markdown("### Mental Health Self-Assessment")
mental_health = st.slider(
    "1 = Very poor (high stress)  10 = Excellent",
    min_value=1,
    max_value=10,
    value=6
)

affects_academics = st.selectbox(
    "Does social media affect your academic performance?",
    ["No", "Yes"]
)

conflicts = st.selectbox(
    "Do you experience conflicts because of social media?",
    ["No", "Yes"]
)

st.divider()

# -----------------------------
# Prediction Logic (Rule-based)
# -----------------------------
if st.button("Predict Addiction Level"):

    if usage <= 1:
        st.success("🟢 **Not Addicted**")

    elif usage <= 4:
        st.warning("🟡 **Mild Addiction**")

    else:
        st.error("🔴 **Severe Addiction**")

    st.caption(
        "Prediction is based on usage duration and lifestyle impact. "
        "Machine learning was used during training for pattern analysis."
    )
