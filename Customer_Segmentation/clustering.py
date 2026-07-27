from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_clustering(df,n_clusters):

    features=df[[
        "Age",
        "AnnualIncome",
        "SpendingScore"
    ]]

    scaler=StandardScaler()

    scaled=scaler.fit_transform(features)

    model=KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels=model.fit_predict(scaled)

    df["Cluster"]=labels

    return df,scaled,model


def elbow_method(df):

    features=df[[
        "Age",
        "AnnualIncome",
        "SpendingScore"
    ]]

    scaler=StandardScaler()

    scaled=scaler.fit_transform(features)

    inertia=[]

    for k in range(1,11):

        km=KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        km.fit(scaled)

        inertia.append(km.inertia_)

    return inertia