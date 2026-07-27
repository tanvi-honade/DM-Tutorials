import os
import streamlit as st
import joblib

from preprocess import clean_text
from train_model import train_model

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Election Sentiment Analysis",
    page_icon="🗳️",
    layout="wide"
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")


# -----------------------------
# Load or Train Model
# -----------------------------
@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):

        with st.spinner("Training model for first time... Please wait..."):
            train_model()

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


model, vectorizer = load_model()

# -----------------------------
# Title
# -----------------------------
st.title("🗳️ Election Sentiment Analysis")

st.markdown("---")

st.write(
    "Analyze election-related social media posts using Machine Learning "
    "and classify them as **Positive**, **Negative**, or **Neutral**."
)

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area(
    "Enter Election Related Text",
    height=150
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        cleaned = clean_text(user_input)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)

        confidence = probability.max() * 100

        st.markdown("---")

        st.subheader("Prediction")

        if prediction == "Positive":
            st.success(f"😊 Positive ({confidence:.2f}%)")

        elif prediction == "Negative":
            st.error(f"😠 Negative ({confidence:.2f}%)")

        else:
            st.info(f"😐 Neutral ({confidence:.2f}%)")

        st.write("### Cleaned Text")

        st.code(cleaned)

        st.write(f"### Confidence : {confidence:.2f}%")