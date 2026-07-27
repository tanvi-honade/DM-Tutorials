from pathlib import Path
import streamlit as st
import plotly.express as px

from model import train_model
from utils import load_data

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

with open(BASE_DIR / "style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

model = train_model()
df = load_data()

st.title("🎓 Intelligent Placement Prediction System")

tab1, tab2, tab3 = st.tabs(
    ["Prediction", "Analytics", "Dataset"]
)

# ---------------- Prediction ----------------

with tab1:

    st.subheader("Student Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        cgpa = st.slider("CGPA", 5.0, 10.0, 7.5)
        iq = st.slider("IQ", 70, 150, 110)

    with c2:
        projects = st.slider("Projects", 0, 6, 2)
        internships = st.slider("Internships", 0, 3, 1)

    with c3:
        communication = st.slider("Communication Skills", 1, 10, 7)

    if st.button("Predict Placement"):

        features = [[
            cgpa,
            iq,
            projects,
            internships,
            communication
        ]]

        prediction = model.predict(features)
        probability = model.predict_proba(features)

        confidence = probability.max() * 100

        if prediction[0] == 1:
            st.success("✅ Student is likely to get Placed")
        else:
            st.error("❌ Student may not get Placed")

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

# ---------------- Analytics ----------------

with tab2:

    c1, c2 = st.columns(2)

    with c1:

        fig = px.scatter(
            df,
            x="CGPA",
            y="IQ",
            color=df["Placement"].astype(str),
            size="Projects",
            title="CGPA vs IQ"
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:

        fig2 = px.histogram(
            df,
            x="CGPA",
            color=df["Placement"].astype(str),
            title="CGPA Distribution"
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Placement Statistics")

    col1, col2, col3 = st.columns(3)

    placed = df[df["Placement"] == 1].shape[0]
    total = df.shape[0]
    rate = placed / total * 100

    col1.metric("Students", total)
    col2.metric("Placed", placed)
    col3.metric("Placement Rate", f"{rate:.1f}%")

# ---------------- Dataset ----------------

with tab3:
    st.dataframe(df, use_container_width=True)