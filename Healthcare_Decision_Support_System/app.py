import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Healthcare Decision Support System",
    page_icon="🏥",
    layout="wide"
)

# ---------------- File Paths ----------------
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "heart.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

# ---------------- Train Model if Not Found ----------------
if not MODEL_PATH.exists():

    MODEL_PATH.parent.mkdir(exist_ok=True)

    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        st.stop()

    df = pd.read_csv(DATA_PATH)

    X = df.drop("target", axis=1)
    y = df["target"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


# ---------------- Load Model ----------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>

.main{
    background-color:#f5f9ff;
}

.title{
    text-align:center;
    color:#1565C0;
    font-size:42px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.block-container{
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("<div class='title'>🏥 Healthcare Decision Support System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Based Heart Disease Prediction using Data Mining</div>", unsafe_allow_html=True)

st.write("")

# ---------------- Sidebar ----------------
st.sidebar.title("🩺 About")

st.sidebar.success("""
Healthcare Decision Support System

✔ Random Forest Classifier

✔ Predicts Heart Disease Risk

✔ Built using Python & Streamlit
""")

st.sidebar.info("""
**Dataset**

Heart Disease Dataset

**Algorithm**

Random Forest Classification
""")

# ---------------- Input ----------------
left, right = st.columns(2)

with left:

    st.subheader("👤 Patient Information")

    age = st.slider("Age",20,100,40)

    sex = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [0,1,2,3]
    )

    trestbps = st.slider(
        "Resting Blood Pressure",
        80,200,120
    )

    chol = st.slider(
        "Cholesterol",
        100,600,200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar >120",
        ["No","Yes"]
    )

with right:

    st.subheader("❤️ Medical Information")

    restecg = st.selectbox(
        "Rest ECG",
        [0,1,2]
    )

    thalach = st.slider(
        "Maximum Heart Rate",
        60,220,150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        ["No","Yes"]
    )

    oldpeak = st.slider(
        "Old Peak",
        0.0,6.5,1.0
    )

    slope = st.selectbox(
        "Slope",
        [0,1,2]
    )

    ca = st.selectbox(
        "Major Vessels",
        [0,1,2,3,4]
    )

    thal = st.selectbox(
        "Thal",
        [0,1,2,3]
    )

# ---------------- Convert ----------------
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# ---------------- Predict ----------------
st.write("")

if st.button("🔍 Predict Heart Disease", use_container_width=True):

    sample = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Healthy Probability",
            f"{probability[0]*100:.2f}%"
        )

    with c2:
        st.metric(
            "Disease Probability",
            f"{probability[1]*100:.2f}%"
        )

    st.write("")

    if prediction == 1:

        st.error("⚠ High Risk of Heart Disease")

        st.progress(float(probability[1]))

        st.markdown("### 🩺 Recommendations")

        st.warning("""
- Consult a Cardiologist
- Exercise Regularly
- Reduce Cholesterol
- Monitor Blood Pressure
- Stop Smoking
- Follow Healthy Diet
""")

    else:

        st.success("✅ Low Risk of Heart Disease")

        st.progress(float(probability[0]))

        st.markdown("### 💚 Recommendations")

        st.info("""
- Continue Healthy Lifestyle
- Regular Exercise
- Balanced Diet
- Annual Health Checkup
- Maintain Healthy Weight
""")

st.divider()

st.caption("© 2026 Healthcare Decision Support System | Data Mining Mini Project")