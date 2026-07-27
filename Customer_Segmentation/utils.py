import pandas as pd

def load_data(file):

    df = pd.read_csv(file)

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({
            "Male":0,
            "Female":1
        })

    return df