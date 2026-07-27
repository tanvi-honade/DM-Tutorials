import os
import random
import pandas as pd

# -------------------------------------------------
# Project Directory
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "student_dropout.csv")

# -------------------------------------------------
# Generate Dataset
# -------------------------------------------------

rows = []

for _ in range(1000):

    attendance = random.randint(40, 100)
    grades = random.randint(35, 100)
    study_hours = random.randint(1, 8)
    backlogs = random.randint(0, 6)

    income = random.choice(["Low", "Medium", "High"])
    scholarship = random.choice(["Yes", "No"])
    internet = random.choice(["Yes", "No"])
    gender = random.choice(["Male", "Female"])

    dropout = 0

    if (
        attendance < 60
        or grades < 50
        or backlogs >= 3
        or study_hours <= 2
    ):
        dropout = 1

    rows.append([
        attendance,
        grades,
        study_hours,
        backlogs,
        income,
        scholarship,
        internet,
        gender,
        dropout
    ])

columns = [
    "Attendance",
    "Grades",
    "StudyHours",
    "Backlogs",
    "FamilyIncome",
    "Scholarship",
    "Internet",
    "Gender",
    "Dropout"
]

df = pd.DataFrame(rows, columns=columns)

# -------------------------------------------------
# Save Dataset
# -------------------------------------------------

df.to_csv(DATASET_PATH, index=False)

print(f"Dataset generated successfully: {DATASET_PATH}")