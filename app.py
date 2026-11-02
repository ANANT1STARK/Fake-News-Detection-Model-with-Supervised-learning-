import streamlit as st
import joblib

# Load trained model
vectorizer = joblib.load("models/vectorizer.jb")
model = joblib.load("models/lr_model.jb")

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article below to analyze whether "
    "it is classified as Fake or Real."
)

st.divider()

# Input
inputn = st.text_area(
    "Enter News Article:",
    height=250,
    placeholder="Paste your news article here..."
)

if st.button("Analyze News", use_container_width=True):

    if inputn.strip():

        # Transform input
        transformed_input = vectorizer.transform([inputn])

        # Predict
        prediction = model.predict(transformed_input)[0]

        # Get prediction probabilities
        probabilities = model.predict_proba(transformed_input)[0]

        confidence = max(probabilities) * 100

        st.divider()

        if prediction == 1:
            st.success("Prediction: REAL NEWS")
        else:
            st.error("Prediction: FAKE NEWS")

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        st.caption(
            "Confidence represents the model's predicted "
            "probability, not the verified truth of the article."
        )

    else:
        st.warning("Please enter some text to analyze.")
