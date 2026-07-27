import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from preprocess import clean_text


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train_model():

    csv_path = os.path.join(BASE_DIR, "election_sentiment.csv")

    df = pd.read_csv(csv_path)

    df["clean_text"] = df["text"].apply(clean_text)

    X = df["clean_text"]
    y = df["sentiment"]

    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)

    model.fit(X, y)

    joblib.dump(model, os.path.join(BASE_DIR, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(BASE_DIR, "vectorizer.pkl"))

    print("Model Trained Successfully")


if __name__ == "__main__":
    train_model()