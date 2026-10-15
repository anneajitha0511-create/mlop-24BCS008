
from flask import Flask, request, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

MODEL_VERSION = "1.0"
model = joblib.load("student_model.pkl")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Student Result Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #eef2ff;
            margin: 0;
            padding: 40px 15px;
        }
        .container {
            max-width: 450px;
            margin: auto;
            background: white;
            padding: 28px;
            border-radius: 14px;
            box-shadow: 0 5px 20px #ccd0df;
        }
        h1 { color: #4338ca; text-align: center; }
        label { display: block; margin-top: 16px; }
        input {
            box-sizing: border-box;
            width: 100%;
            padding: 10px;
            margin-top: 6px;
        }
        button {
            margin-top: 22px;
            width: 100%;
            padding: 12px;
            background: #4338ca;
            color: white;
            border: 0;
            border-radius: 6px;
            cursor: pointer;
        }
        .result {
            margin-top: 20px;
            padding: 12px;
            background: #e0e7ff;
            text-align: center;
            font-weight: bold;
        }
        .version {
            text-align: center;
            color: #555;
            margin-top: 20px;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>Student Result Predictor</h1>

    <form method="POST">
        <label>Study Hours (1-8)</label>
        <input type="number" name="hours"
               min="1" max="8" step="1"
               required value="{{ hours }}">

        <label>Attendance (%)</label>
        <input type="number" name="attendance"
               min="0" max="100" step="any"
               required value="{{ attendance }}">

        <label>Internal Marks (out of 100)</label>
        <input type="number" name="marks"
               min="0" max="100" step="any"
               required value="{{ marks }}">

        <button type="submit">Predict Result</button>
    </form>

    {% if prediction %}
    <div class="result">
        Predicted Result: {{ prediction }}
    </div>
    {% endif %}

    {% if error %}
    <div class="result">{{ error }}</div>
    {% endif %}

    <p class="version">Model Version: {{ version }}</p>
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    hours = 3
    attendance = 80
    marks = 60

    if request.method == "POST":
        try:
            hours = float(request.form["hours"])
            attendance = float(request.form["attendance"])
            marks = float(request.form["marks"])

            if not (
                1 <= hours <= 8
                and 0 <= attendance <= 100
                and 0 <= marks <= 100
            ):
                raise ValueError("Values are outside the allowed range.")

            features = pd.DataFrame([{
                "study_hours": hours,
                "attendance": attendance,
                "internal_marks": marks
            }])

            result = model.predict(features)[0]
            prediction = "PASS" if result == 1 else "FAIL"

        except (ValueError, KeyError):
            error = "Enter valid values within the specified ranges."

    return render_template_string(
        HTML,
        prediction=prediction,
        error=error,
        hours=hours,
        attendance=attendance,
        marks=marks,
        version=MODEL_VERSION
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
