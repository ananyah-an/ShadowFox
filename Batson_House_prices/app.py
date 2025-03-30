import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset from the same directory
dataset_path = "HousingData.csv"
df = pd.read_csv(dataset_path)

# Handle missing values
for col in ["CRIM", "ZN", "INDUS", "AGE", "LSTAT"]:
    df[col].fillna(df[col].median(), inplace=True)
df["CHAS"].fillna(df["CHAS"].mode()[0], inplace=True)

# Split data into features (X) and target (y)
X = df.drop(columns=["MEDV"])
y = df["MEDV"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.1)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R²": r2}
    print(f"\n{name} Model:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.4f}")

# Feature importance for Gradient Boosting
feature_importance = models["Gradient Boosting"].feature_importances_
feature_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importance})
feature_df = feature_df.sort_values(by="Importance", ascending=False)

# Print feature importance
print("\nFeature Importance - Gradient Boosting:")
print(feature_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_df["Importance"], y=feature_df["Feature"], palette="viridis")
plt.title("Feature Importance - Gradient Boosting")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

