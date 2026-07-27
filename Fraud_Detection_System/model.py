from sklearn.ensemble import IsolationForest

def train_model(df):

    X = df[[
        "Amount",
        "Time",
        "TransactionType",
        "LocationRisk"
    ]]

    model = IsolationForest(
        contamination=0.2,
        random_state=42
    )

    model.fit(X)

    return model