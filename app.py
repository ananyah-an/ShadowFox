import os
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

MODEL_PATH = "car_price_model.pkl"

# Train and save the model
def train_model():
    df = pd.read_csv("car.csv")  # Replace with actual dataset
    df.dropna(inplace=True)
    df['Years_Since_Manufacture'] = 2025 - df['Year']
    df.drop(columns=['Year'], inplace=True)
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(columns=['Selling_Price'])
    y = df['Selling_Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    return X_train.columns  # Return feature names for consistency

FEATURE_NAMES = train_model()

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Predict API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Get JSON input
    sample_data = pd.DataFrame([data])  # Convert to DataFrame

    # Ensure matching feature columns
    sample_data = sample_data.reindex(columns=FEATURE_NAMES, fill_value=0)

    # Load model & make prediction
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    predicted_price = model.predict(sample_data)[0]

    return jsonify({"predicted_price": f"₹{predicted_price:,.2f}"})

if __name__ == "__main__":
    app.run(debug=True)
