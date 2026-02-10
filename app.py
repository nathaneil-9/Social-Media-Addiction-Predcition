import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Social Media Addiction Predictor")

st.title("📱 Social Media Addiction Prediction")

age = st.number_input("Age", 15, 40, 20)
daily_usage = st.slider("Average Daily Usage (hours)", 0.0, 12.0, 4.0)
sleep = st.slider("Sleep Hours Per Night", 0.0, 12.0, 7.0)
mental_health = st.slider("Mental Health Score", 1, 10, 5)

affects_academics = st.selectbox(
    "Does Social Media Affect Academic Performance?",
    ["No", "Yes"]
)

conflicts = st.selectbox(
    "Conflicts Over Social Media?",
    ["No", "Yes"]
)

affects_academics = 1 if affects_academics == "Yes" else 0
conflicts = 1 if conflicts == "Yes" else 0

if st.button("Predict"):
    input_data = np.array([[
        age,
        daily_usage,
        sleep,
        mental_health,
        affects_academics,
        conflicts
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        st.success("🟢 Not Addicted")
    elif prediction == 1:
        st.warning("🟡 Mild Addiction")
    else:
        st.error("🔴 Severe Addiction")
