from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "placement_model.pkl"
DATA_PATH = BASE_DIR / "dataset" / "placement.csv"


def train_model():

    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)

    X = df.drop("Placement", axis=1)
    y = df["Placement"]

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    model.fit(X, y)

    MODEL_PATH.parent.mkdir(exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    return model