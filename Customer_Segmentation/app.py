import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from pathlib import Path

from utils import load_data
from clustering import *

# ---------------- Base Directory ----------------
BASE_DIR = Path(__file__).resolve().parent

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🎯",
    layout="wide"
)

# ---------------- Load CSS ----------------
css_path = BASE_DIR / "style.css"

if css_path.exists():
    with open(css_path) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# ---------------- Title ----------------
st.title("🎯 Customer Segmentation for Marketing")

# ---------------- Upload CSV ----------------
uploaded = st.sidebar.file_uploader(
    "Upload CSV",
    type="csv"
)

if uploaded is not None:
    df = load_data(uploaded)
else:
    data_path = BASE_DIR / "dataset" / "customers.csv"
    df = load_data(data_path)

# ---------------- Dataset ----------------
st.subheader("Dataset")
st.dataframe(df, use_container_width=True)

# ---------------- Slider ----------------
clusters = st.sidebar.slider(
    "Number of Clusters",
    2,
    8,
    5
)

# ---------------- Run Model ----------------
if st.button("Run Segmentation"):

    result, scaled, model = perform_clustering(df, clusters)

    c1, c2, c3 = st.columns(3)

    c1.metric("Customers", len(result))
    c2.metric("Clusters", clusters)
    c3.metric("Avg Income", round(result["AnnualIncome"].mean(), 2))

    # ---------------- 3D Plot ----------------
    st.subheader("Customer Segments")

    fig = px.scatter_3d(
        result,
        x="Age",
        y="AnnualIncome",
        z="SpendingScore",
        color=result["Cluster"].astype(str),
        hover_data=["CustomerID"]
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- 2D Plot ----------------
    st.subheader("2D Visualization")

    fig2 = px.scatter(
        result,
        x="AnnualIncome",
        y="SpendingScore",
        color=result["Cluster"].astype(str),
        size="Age",
        hover_data=["CustomerID"]
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ---------------- Table ----------------
    st.subheader("Segmented Customers")
    st.dataframe(result, use_container_width=True)

    # ---------------- Suggestions ----------------
    st.subheader("Marketing Suggestions")

    for cluster in sorted(result.Cluster.unique()):

        group = result[result.Cluster == cluster]

        income = group["AnnualIncome"].mean()
        spend = group["SpendingScore"].mean()

        if income > 60 and spend > 60:
            st.success(
                f"Cluster {cluster}: Premium Customers → Luxury products, VIP memberships."
            )

        elif income > 60:
            st.info(
                f"Cluster {cluster}: High Income but Low Spending → Discount offers."
            )

        elif spend > 60:
            st.warning(
                f"Cluster {cluster}: Impulse Buyers → Flash sales and coupons."
            )

        else:
            st.error(
                f"Cluster {cluster}: Budget Customers → Affordable products."
            )

# ---------------- Elbow Method ----------------
st.subheader("Elbow Method")

inertia = elbow_method(df)

fig, ax = plt.subplots()

ax.plot(range(1, 11), inertia, marker="o")

ax.set_xlabel("Number of Clusters")
ax.set_ylabel("Inertia")

st.pyplot(fig)