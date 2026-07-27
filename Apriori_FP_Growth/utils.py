import pandas as pd
import time

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


def load_transactions(file):

    df = pd.read_csv(file)

    transactions = []

    for row in df.values:
        row = [str(item) for item in row if pd.notna(item)]
        transactions.append(row)

    return transactions


def encode_transactions(transactions):

    te = TransactionEncoder()

    te_array = te.fit(transactions).transform(transactions)

    return pd.DataFrame(te_array, columns=te.columns_)


def run_apriori(df, support):

    start = time.time()

    frequent = apriori(df,
                       min_support=support,
                       use_colnames=True)

    rules = association_rules(
        frequent,
        metric="confidence",
        min_threshold=0.5
    )

    end = time.time()

    return frequent, rules, end-start


def run_fpgrowth(df, support):

    start = time.time()

    frequent = fpgrowth(df,
                        min_support=support,
                        use_colnames=True)

    rules = association_rules(
        frequent,
        metric="confidence",
        min_threshold=0.5
    )

    end = time.time()

    return frequent, rules, end-start