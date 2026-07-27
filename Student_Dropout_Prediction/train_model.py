import os
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------
# Project Directory
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "student_dropout.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
ACCURACY_PATH = os.path.join(BASE_DIR, "accuracy.pkl")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

df = pd.read_csv(DATASET_PATH)

# -------------------------------------------------
# Encode Categorical Columns
# -------------------------------------------------

encoders = {}

categorical_columns = [
    "FamilyIncome",
    "Scholarship",
    "Internet",
    "Gender"
]

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# -------------------------------------------------
# Features and Target
# -------------------------------------------------

X = df.drop("Dropout", axis=1)
y = df["Dropout"]

# -------------------------------------------------
# Train-Test Split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# Train Random Forest
# -------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------------------------
# Evaluate Model
# -------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%\n")
print(classification_report(y_test, predictions))

# -------------------------------------------------
# Save Files
# -------------------------------------------------

joblib.dump(model, MODEL_PATH)
joblib.dump(encoders, ENCODER_PATH)
joblib.dump(accuracy, ACCURACY_PATH)

print("\nModel Saved Successfully!")
print(f"Dataset  : {DATASET_PATH}")
print(f"Model    : {MODEL_PATH}")
print(f"Encoders : {ENCODER_PATH}")
print(f"Accuracy : {ACCURACY_PATH}")