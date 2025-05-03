from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import json
from pymongo import MongoClient

# Load model and label encodings
model = pickle.load(open("linear_regression.pkl", "rb"))
with open("label_enc_data.json", "r") as f:
    label_data = json.load(f)

# Column names (must match training order)
column_names = np.array(['age', 'gender', 'bmi', 'children', 'smoker', 
                         'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest'])

app = Flask(__name__)

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["insurance_db"]
collection = db["predictions"]

@app.route('/')
def home():
    return render_template("predict.html")

@app.route('/predict', methods=["POST"])
def predict():
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    age = int(data['age'])
    gender = data['gender']
    bmi = float(data['bmi'])
    children = int(data['children'])
    smoker = data['smoker']
    region = data['region']

    # Prepare input array
    test_array = np.zeros((1, column_names.size))
    test_array[0, 0] = age
    test_array[0, 1] = label_data['gender'][gender]
    test_array[0, 2] = bmi
    test_array[0, 3] = children
    test_array[0, 4] = label_data['smoker'][smoker]

    region_col = f"region_{region}"
    region_index = np.where(column_names == region_col)[0][0]
    test_array[0, region_index] = 1

    # Make prediction
    prediction = model.predict(test_array)[0]

    # Save to MongoDB
    record = {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "predicted_charges": float(prediction)
    }
    collection.insert_one(record)

    if request.is_json:
        return jsonify({"predicted_charges": prediction})
    else:
        return render_template("predict.html", prediction=prediction)

# @app.route("/history", methods=["GET"])
# def history():
#     records = list(collection.find({}, {"_id": 0}))
#     return jsonify(records)

@app.route("/history", methods=["GET"])
def history():
    records = list(collection.find({}, {"_id": 0}))
    return render_template("history.html", records=records)


if __name__ == "__main__":
    app.run(debug=True)
