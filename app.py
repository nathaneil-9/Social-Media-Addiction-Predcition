import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Social Media Addiction Predictor",
    page_icon="📱",
    layout="centered"
)


# -----------------------------
# Load Model and Encoders
# -----------------------------
@st.cache_resource
def load_model():

    model = joblib.load("addiction_model.pkl")
    encoders = joblib.load("encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    return model, encoders, feature_columns


model, encoders, feature_columns = load_model()


# -----------------------------
# Header
# -----------------------------
st.markdown(
    "<h1 style='text-align: center;'>📱 Social Media Addiction Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "A machine learning system that predicts social media addiction "
    "levels using usage, lifestyle and behavioral factors."
    "</p>",
    unsafe_allow_html=True
)

st.divider()


# -----------------------------
# User Information
# -----------------------------
st.subheader("👤 User Information")

age = st.number_input(
    "Age",
    min_value=15,
    max_value=40,
    value=20,
    step=1
)


# -----------------------------
# Categorical Information
# -----------------------------
st.subheader("📋 Personal Information")

gender = st.selectbox(
    "Gender",
    encoders["Gender"].classes_.tolist()
)

academic_level = st.selectbox(
    "Academic Level",
    encoders["Academic_Level"].classes_.tolist()
)

country = st.selectbox(
    "Country",
    encoders["Country"].classes_.tolist()
)

platform = st.selectbox(
    "Most Used Social Media Platform",
    encoders["Most_Used_Platform"].classes_.tolist()
)

relationship_status = st.selectbox(
    "Relationship Status",
    encoders["Relationship_Status"].classes_.tolist()
)


# -----------------------------
# Usage Details
# -----------------------------
st.subheader("📊 Usage Details")

usage = st.slider(
    "Average Daily Social Media Usage (hours)",
    min_value=0.0,
    max_value=12.0,
    value=4.0,
    step=0.5
)

sleep = st.slider(
    "Sleep Hours Per Night",
    min_value=0.0,
    max_value=12.0,
    value=7.0,
    step=0.5
)


# -----------------------------
# Mental Health
# -----------------------------
st.subheader("🧠 Mental Well-Being")

mental_health = st.slider(
    "Overall Mental Health (1 = Very Poor, 10 = Excellent)",
    min_value=1,
    max_value=10,
    value=6,
    step=1
)


# -----------------------------
# Academic & Social Impact
# -----------------------------
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
# Prediction
# -----------------------------
st.subheader("🔍 Prediction")


if st.button(
    "Predict Addiction Level",
    use_container_width=True
):

    try:

        # -----------------------------
        # Encode User Inputs
        # -----------------------------

        gender_encoded = encoders["Gender"].transform(
            [gender]
        )[0]

        academic_level_encoded = encoders["Academic_Level"].transform(
            [academic_level]
        )[0]

        country_encoded = encoders["Country"].transform(
            [country]
        )[0]

        platform_encoded = encoders["Most_Used_Platform"].transform(
            [platform]
        )[0]

        relationship_encoded = encoders["Relationship_Status"].transform(
            [relationship_status]
        )[0]


        academic_encoded = (
            1 if affects_academics == "Yes" else 0
        )

        conflicts_encoded = (
            1 if conflicts == "Yes" else 0
        )


        # -----------------------------
        # Create Input DataFrame
        # -----------------------------
        input_data = pd.DataFrame({

            "Age": [age],

            "Gender": [gender_encoded],

            "Academic_Level": [
                academic_level_encoded
            ],

            "Country": [
                country_encoded
            ],

            "Most_Used_Platform": [
                platform_encoded
            ],

            "Avg_Daily_Usage_Hours": [
                usage
            ],

            "Sleep_Hours_Per_Night": [
                sleep
            ],

            "Mental_Health_Score": [
                mental_health
            ],

            "Relationship_Status": [
                relationship_encoded
            ],

            "Affects_Academic_Performance": [
                academic_encoded
            ],

            "Conflicts_Over_Social_Media": [
                conflicts_encoded
            ]

        })


        # -----------------------------
        # Ensure Correct Feature Order
        # -----------------------------
        input_data = input_data[
            feature_columns
        ]


        # -----------------------------
        # Make Prediction
        # -----------------------------
        prediction = model.predict(
            input_data
        )[0]

        probabilities = model.predict_proba(
            input_data
        )[0]


        # -----------------------------
        # Prediction Labels
        # -----------------------------
        labels = {
            0: "Not Addicted",
            1: "Mild Addiction",
            2: "Severe Addiction"
        }

        prediction_label = labels[prediction]


        # -----------------------------
        # Display Result
        # -----------------------------
        st.subheader("📊 Prediction Result")

        confidence = probabilities[prediction] * 100


        if prediction == 0:

            st.success(
                f"🟢 **{prediction_label}**"
            )

        elif prediction == 1:

            st.warning(
                f"🟡 **{prediction_label}**"
            )

        else:

            st.error(
                f"🔴 **{prediction_label}**"
            )


        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )


        # -----------------------------
        # Probability Breakdown
        # -----------------------------
        st.subheader("📈 Prediction Probabilities")

        probability_data = pd.DataFrame({

            "Addiction Level": [
                "Not Addicted",
                "Mild Addiction",
                "Severe Addiction"
            ],

            "Probability": [
                probabilities[0],
                probabilities[1],
                probabilities[2]
            ]

        })

        probability_data["Probability"] = (
            probability_data["Probability"] * 100
        )

        st.bar_chart(
            probability_data.set_index(
                "Addiction Level"
            )
        )


        # -----------------------------
        # Explanation
        # -----------------------------
        st.info(
            "The prediction is generated using a trained "
            "Random Forest classification model based on "
            "the information provided above."
        )


    except Exception as e:

        st.error(
            "Unable to generate prediction."
        )

        st.exception(e)


# -----------------------------
# Footer
# -----------------------------
st.divider()

st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Social Media Addiction Prediction Project | "
    "Machine Learning + Streamlit"
    "</p>",
    unsafe_allow_html=True
)

st.caption(
    "This application is intended for educational purposes "
    "and does not provide medical or clinical diagnosis."
)
