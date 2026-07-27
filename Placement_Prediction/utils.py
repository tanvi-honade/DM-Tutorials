from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset" / "placement.csv"


def load_data():
    return pd.read_csv(DATA_PATH)