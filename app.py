import streamlit as st

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Social Media Addiction Predictor",
    page_icon="📱",
    layout="centered"
)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    "<h1 style='text-align: center;'>📱 Social Media Addiction Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "A simple system to assess social media addiction based on usage and lifestyle"
    "</p>",
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Input Section
# -----------------------------
st.subheader("👤 User Information")

age = st.number_input(
    "Age",
    min_value=15,
    max_value=40,
    value=20
)

st.subheader("📊 Usage Details")

usage = st.slider(
    "Average Daily Social Media Usage (hours)",
    0.0, 12.0, 0.0, step=0.5
)

sleep = st.slider(
    "Sleep Hours Per Night",
    0.0, 12.0, 7.0, step=0.5
)

st.subheader("🧠 Mental Well-Being")

mental_health = st.slider(
    "Overall Mental Health (1 = Very Poor, 10 = Excellent)",
    1, 10, 6
)

st.subheader("📚 Academic & Social Impact")

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
# Prediction Section
# -----------------------------
st.subheader("🔍 Prediction")

if st.button("Predict Addiction Level", use_container_width=True):

    if usage <= 1:
        st.success("🟢 **Not Addicted**")
        st.progress(20)
        st.write("You show healthy social media usage habits.")

    elif usage <= 4:
        st.warning("🟡 **Mild Addiction**")
        st.progress(60)
        st.write(
            "Your usage is moderate. Reducing screen time may help "
            "improve productivity and well-being."
        )

    else:
        st.error("🔴 **Severe Addiction**")
        st.progress(90)
        st.write(
            "High usage indicates possible addiction. Consider limiting usage "
            "and maintaining a balanced lifestyle."
        )

    st.caption(
        "Note: Machine learning was used during data analysis. "
        "Final prediction is rule-based for clarity and reliability."
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    "<hr><p style='text-align: center; color: gray;'>"
    "Social Media Addiction Prediction Project | Streamlit App"
    "</p>",
    unsafe_allow_html=True
)
