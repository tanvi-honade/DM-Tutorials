import streamlit as st
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# ----------------------------
# Sample Transactions
# ----------------------------

transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread'],
    ['Milk', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread', 'Eggs'],
    ['Bread', 'Butter', 'Eggs'],
    ['Milk', 'Bread', 'Butter'],
    ['Milk', 'Eggs'],
    ['Bread', 'Butter', 'Jam']
]

# ----------------------------
# Data Preparation
# ----------------------------

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

frequent_itemsets = apriori(
    df,
    min_support=0.30,
    use_colnames=True
)

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.60
)

# ----------------------------
# Dashboard
# ----------------------------

st.set_page_config(
    page_title="E-Commerce Recommendation System",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 E-Commerce Recommendation System")
st.write("Association Rule Mining using Apriori Algorithm")

# Sidebar

st.sidebar.header("Settings")

selected_product = st.sidebar.selectbox(
    "Choose Product",
    sorted(df.columns)
)

# Metrics

col1, col2, col3 = st.columns(3)

col1.metric("Transactions", len(transactions))
col2.metric("Products", len(df.columns))
col3.metric("Association Rules", len(rules))

st.divider()

# Frequent Itemsets

st.subheader("Frequent Itemsets")

st.dataframe(frequent_itemsets)

st.divider()

# Association Rules

st.subheader("Association Rules")

display_rules = rules[
    [
        'antecedents',
        'consequents',
        'support',
        'confidence',
        'lift'
    ]
].copy()

display_rules['antecedents'] = display_rules['antecedents'].astype(str)
display_rules['consequents'] = display_rules['consequents'].astype(str)

st.dataframe(display_rules)

st.divider()

# Recommendation

st.subheader("Recommended Products")

found = False

for _, row in rules.iterrows():

    antecedent = list(row['antecedents'])

    consequent = list(row['consequents'])

    if selected_product in antecedent:

        st.success(f"Customer buys **{selected_product}**")

        st.info(f"Recommend: **{', '.join(consequent)}**")

        st.write(f"Confidence : {row['confidence']:.2f}")

        st.write(f"Lift : {row['lift']:.2f}")

        st.progress(min(row['confidence'],1.0))

        st.write("---")

        found = True

if not found:
    st.warning("No recommendation available.")

# Raw Data

with st.expander("View Encoded Dataset"):
    st.dataframe(df)


    ##  How to run the app
    # cd Recommendation_System
    # dir
    # streamlit run app.py