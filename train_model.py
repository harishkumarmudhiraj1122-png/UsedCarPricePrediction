
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Reproducible synthetic dataset for the practical
np.random.seed(42)
n = 2000

# Expanded ranges to include older, high-mileage cars
years = np.random.randint(2005, 2027, n)
kms = np.random.randint(0, 250001, n)
owners = np.random.randint(1, 4, n)

fuel = np.random.choice(["Petrol", "Diesel", "CNG"], n)
transmission = np.random.choice(["Manual", "Automatic"], n)
brand = np.random.choice(
    ["Maruti", "Hyundai", "Honda", "Tata", "Toyota"], n
)

brand_values = {
    "Maruti": 0,
    "Hyundai": 40000,
    "Honda": 70000,
    "Tata": 20000,
    "Toyota": 100000
}

fuel_values = {"Petrol": 0, "Diesel": 40000, "CNG": -15000}
transmission_values = {"Manual": 0, "Automatic": 40000}

# Illustrative price formula (not real market data)
age = 2026 - years

prices = (
    900000
    - age * 25000
    - kms * 1.0
    - (owners - 1) * 20000
    + pd.Series(brand).map(brand_values).to_numpy()
    + pd.Series(fuel).map(fuel_values).to_numpy()
    + pd.Series(transmission).map(transmission_values).to_numpy()
    + np.random.normal(0, 25000, n)
)

data = pd.DataFrame({
    "year": years,
    "km_driven": kms,
    "owner": owners,
    "fuel": fuel,
    "transmission": transmission,
    "brand": brand,
    "selling_price": np.maximum(prices, 30000)
})

X = data.drop(columns=["selling_price"])
y = data["selling_price"]

numeric_features = ["year", "km_driven", "owner"]
categorical_features = ["fuel", "transmission", "brand"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("USED CAR PRICE PREDICTION - MODEL EVALUATION")
print("-" * 45)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"MAE: Rs. {mae:,.2f}")
print(f"RMSE: Rs. {rmse:,.2f}")
print(f"R-squared: {r2:.4f}")

model_path = Path(__file__).resolve().parent / "model.pkl"
joblib.dump(model, model_path)
print(f"\nModel saved successfully: {model_path}")
