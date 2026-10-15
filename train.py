
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Generate sample student data
data = []

for hours in range(1, 9):
    for attendance in range(40, 101, 10):
        for marks in range(10, 101, 10):
            passed = int(
                marks >= 40
                and attendance >= 75
                and hours >= 2
            )

            data.append([
                hours, attendance, marks, passed
            ])

df = pd.DataFrame(
    data,
    columns=[
        "study_hours",
        "attendance",
        "internal_marks",
        "result"
    ]
)

X = df[[
    "study_hours",
    "attendance",
    "internal_marks"
]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Model training completed!")
print("Accuracy:", round(
    accuracy_score(y_test, predictions) * 100, 2
), "%")

joblib.dump(model, "student_model.pkl")
print("Model saved as student_model.pkl")
