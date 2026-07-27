import os
import sys
import subprocess
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>

.main{
    background-color:#f8f9fa;
}

h1{
    color:#1565C0;
    text-align:center;
}

.stButton>button{
    width:100%;
    height:50px;
    background:#1565C0;
    color:white;
    border-radius:10px;
    font-size:18px;
}

.metric-card{
    background:white;
    padding:15px;
    border-radius:12px;
    box-shadow:0px 0px 8px rgba(0,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# PROJECT DIRECTORY
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

python = sys.executable

dataset_path = os.path.join(BASE_DIR, "student_dropout.csv")
model_path = os.path.join(BASE_DIR, "model.pkl")
encoder_path = os.path.join(BASE_DIR, "label_encoders.pkl")

# -------------------------------------------------
# GENERATE DATASET
# -------------------------------------------------



# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------

# CHECK REQUIRED FILES

if not os.path.exists(dataset_path):
    st.error("student_dropout.csv not found.")
    st.stop()

if not os.path.exists(model_path):
    st.error("model.pkl not found.")
    st.stop()

if not os.path.exists(encoder_path):
    st.error("label_encoders.pkl not found.")
    st.stop()

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

model = joblib.load(model_path)

encoders = joblib.load(encoder_path)

df = pd.read_csv(dataset_path)

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("🎓 Student Dropout Prediction System")

st.markdown(
"""
This application predicts whether a student is **likely to continue**
or **drop out** using a Machine Learning Random Forest Classifier.
"""
)

st.divider()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select",
    [
        "Dashboard",
        "Prediction",
        "Batch Prediction"
    ]
)

st.sidebar.divider()

st.sidebar.success("Model : Random Forest")

st.sidebar.info(
"""
Features

✔ Attendance

✔ Grades

✔ Study Hours

✔ Backlogs

✔ Family Income

✔ Scholarship

✔ Internet

✔ Gender
"""
)

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------

if page == "Dashboard":

    st.header("📊 Dashboard")

    c1,c2,c3,c4 = st.columns(4)

    total = len(df)

    dropout = int(df["Dropout"].sum())

    continue_students = total-dropout

    rate = dropout/total*100

    c1.metric(
        "Total Students",
        total
    )

    c2.metric(
        "Continue",
        continue_students
    )

    c3.metric(
        "Dropouts",
        dropout
    )

    c4.metric(
        "Dropout %",
        f"{rate:.2f}%"
    )

    st.divider()

    col1,col2 = st.columns(2)

    with col1:

        fig = px.pie(
            df,
            names=df["Dropout"].map({
                0:"Continue",
                1:"Dropout"
            }),
            title="Student Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig2 = px.histogram(
            df,
            x="Attendance",
            nbins=20,
            title="Attendance Distribution"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    fig3 = px.histogram(
        df,
        x="Grades",
        nbins=20,
        title="Grades Distribution"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

    # -------------------------------------------------
# PREDICTION PAGE
# -------------------------------------------------

elif page == "Prediction":

    st.header("🎯 Student Prediction")

    st.write("Enter the student's details below.")

    col1, col2 = st.columns(2)

    with col1:

        attendance = st.slider(
            "Attendance (%)",
            0,
            100,
            75
        )

        grades = st.slider(
            "Average Grades",
            0,
            100,
            70
        )

        study_hours = st.slider(
            "Study Hours Per Day",
            1,
            10,
            4
        )

        backlogs = st.slider(
            "Number of Backlogs",
            0,
            10,
            0
        )

    with col2:

        family_income = st.selectbox(
            "Family Income",
            ["Low", "Medium", "High"]
        )

        scholarship = st.selectbox(
            "Scholarship",
            ["Yes", "No"]
        )

        internet = st.selectbox(
            "Internet Access",
            ["Yes", "No"]
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    st.divider()

    if st.button("Predict Student Status"):

        income = encoders["FamilyIncome"].transform([family_income])[0]
        scholar = encoders["Scholarship"].transform([scholarship])[0]
        net = encoders["Internet"].transform([internet])[0]
        gen = encoders["Gender"].transform([gender])[0]

        sample = pd.DataFrame(
            [[
                attendance,
                grades,
                study_hours,
                backlogs,
                income,
                scholar,
                net,
                gen
            ]],
            columns=[
                "Attendance",
                "Grades",
                "StudyHours",
                "Backlogs",
                "FamilyIncome",
                "Scholarship",
                "Internet",
                "Gender"
            ]
        )

        prediction = model.predict(sample)[0]

        probability = model.predict_proba(sample)[0]

        continue_prob = probability[0] * 100
        dropout_prob = probability[1] * 100

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 0:

            st.success("✅ Student is likely to CONTINUE")

        else:

            st.error("⚠ Student is likely to DROP OUT")

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Continue Probability",
                f"{continue_prob:.2f}%"
            )

        with c2:

            st.metric(
                "Dropout Probability",
                f"{dropout_prob:.2f}%"
            )

        st.progress(int(max(continue_prob, dropout_prob)))

        st.divider()

        st.subheader("Risk Level")

        if dropout_prob < 30:

            st.success("🟢 LOW RISK")

        elif dropout_prob < 70:

            st.warning("🟡 MEDIUM RISK")

        else:

            st.error("🔴 HIGH RISK")

        st.divider()

        st.subheader("Student Details")

        display = pd.DataFrame({

            "Feature":[
                "Attendance",
                "Grades",
                "Study Hours",
                "Backlogs",
                "Family Income",
                "Scholarship",
                "Internet",
                "Gender"
            ],

            "Value":[
                attendance,
                grades,
                study_hours,
                backlogs,
                family_income,
                scholarship,
                internet,
                gender
            ]

        })

        st.table(display)

        st.divider()

        st.subheader("Recommendations")

        if prediction == 1:

            st.error("Immediate academic intervention recommended.")

            st.write("• Improve attendance above 75%.")
            st.write("• Reduce the number of backlogs.")
            st.write("• Increase study hours to at least 4 hours/day.")
            st.write("• Attend counseling sessions.")
            st.write("• Meet academic mentor weekly.")
            st.write("• Participate in remedial classes.")

        else:

            st.success("Student is performing well.")

            st.write("• Maintain attendance.")
            st.write("• Continue regular study.")
            st.write("• Participate in extracurricular activities.")
            st.write("• Aim to improve grades further.")