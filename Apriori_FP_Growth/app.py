import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils import *

st.set_page_config(
    page_title="Apriori vs FP-Growth",
    layout="wide"
)

st.title("Apriori vs FP-Growth Comparison")

uploaded = st.file_uploader(
    "Upload Transaction Dataset (CSV)",
    type=["csv"]
)

support = st.slider(
    "Minimum Support",
    0.01,
    0.50,
    0.05
)

if uploaded:

    transactions = load_transactions(uploaded)

    encoded = encode_transactions(transactions)

    st.success("Dataset Loaded Successfully")

    st.write("Transactions:", len(transactions))
    st.write("Unique Items:", len(encoded.columns))

    if st.button("Run Comparison"):

        apriori_items, apriori_rules, apriori_time = run_apriori(
            encoded,
            support
        )

        fp_items, fp_rules, fp_time = run_fpgrowth(
            encoded,
            support
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Apriori")

            st.metric(
                "Execution Time",
                f"{apriori_time:.4f} sec"
            )

            st.metric(
                "Frequent Itemsets",
                len(apriori_items)
            )

            st.metric(
                "Association Rules",
                len(apriori_rules)
            )

            st.dataframe(apriori_items.head())

        with col2:

            st.subheader("FP-Growth")

            st.metric(
                "Execution Time",
                f"{fp_time:.4f} sec"
            )

            st.metric(
                "Frequent Itemsets",
                len(fp_items)
            )

            st.metric(
                "Association Rules",
                len(fp_rules)
            )

            st.dataframe(fp_items.head())

        fig, ax = plt.subplots()

        algorithms = ["Apriori", "FP-Growth"]

        times = [apriori_time, fp_time]

        ax.bar(algorithms, times)

        ax.set_ylabel("Execution Time (sec)")

        ax.set_title("Execution Time Comparison")

        st.pyplot(fig)

        fig2, ax2 = plt.subplots()

        itemsets = [
            len(apriori_items),
            len(fp_items)
        ]

        ax2.bar(algorithms, itemsets)

        ax2.set_ylabel("Frequent Itemsets")

        ax2.set_title("Frequent Itemsets Comparison")

        st.pyplot(fig2)