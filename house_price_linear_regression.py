#!/usr/bin/env python3
"""
house_price_linear_regression.py

A beginner-friendly standalone script that:
- Generates a synthetic housing dataset (if no file is provided)
- Preprocesses the data (handles missing values)
- Splits into train/test (80/20)
- Trains a Linear Regression model (scikit-learn)
- Evaluates using R^2 and Mean Squared Error (MSE)
- Visualizes Actual vs Predicted prices and feature coefficients

Requirements:
- pandas
- numpy
- scikit-learn
- matplotlib

Run:
    python3 house_price_linear_regression.py
"""

import sys

# Try to import required packages and give a helpful message if any are missing.
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
except Exception as e:  # ImportError or other import-time errors
    print("Missing or failed import:", e)
    print("Please install the required packages before running this script:")
    print("  python3 -m pip install --user numpy pandas scikit-learn matplotlib")
    sys.exit(1)

# For reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def generate_synthetic_data(n_samples: int = 500, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generate a synthetic dataset for house prices.

    Columns:
    - square_feet: continuous, roughly 400 - 4500
    - bedrooms: integer, 1 - 6
    - bathrooms: integer, 1 - 4
    - price: target, linear combination of features + noise

    Returns a pandas DataFrame.
    """
    rng = np.random.default_rng(random_state)

    # square_feet: skewed distribution (more smaller houses)
    square_feet = rng.normal(loc=1800, scale=600, size=n_samples)
    square_feet = np.clip(square_feet, 300, 7000)  # ensure positive, reasonable upper bound

    # bedrooms: 1 to 6, more often 2-4
    bedrooms = rng.choice([1, 2, 3, 4, 5, 6], size=n_samples, p=[0.03, 0.25, 0.45, 0.2, 0.05, 0.02])

    # bathrooms: 1 to 4, integer
    bathrooms = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.35, 0.45, 0.15, 0.05])

    # True underlying coefficients we simulate
    base_price = 50_000  # base house price in dollars
    coef_sqft = 150      # price per square foot
    coef_bedroom = 12_000
    coef_bathroom = 8_000

    # Noise
    noise = rng.normal(loc=0, scale=30_000, size=n_samples)  # some houses have large noise

    price = base_price + coef_sqft * square_feet + coef_bedroom * bedrooms + coef_bathroom * bathrooms + noise
    price = np.clip(price, a_min=20_000, a_max=None)  # avoid unrealistic negative prices

    df = pd.DataFrame({
        "square_feet": square_feet.round(0).astype(int),
        "bedrooms": bedrooms.astype(int),
        "bathrooms": bathrooms.astype(int),
        "price": price.round(2)
    })

    # Introduce a few missing values to demonstrate handling (about 1% per column)
    n_missing = max(1, int(0.01 * n_samples))
    for col in ["square_feet", "bedrooms", "bathrooms"]:
        missing_indices = rng.choice(n_samples, size=n_missing, replace=False)
        df.loc[missing_indices, col] = np.nan

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset:
    - Show basic info
    - Handle missing values by imputing medians for numeric features
    - (No categorical variables here)
    Returns cleaned DataFrame.
    """
    print("\nData snapshot (first 5 rows):")
    print(df.head())

    print("\nMissing values per column before imputation:")
    print(df.isna().sum())

    # Simple imputation: fill numeric missing values with median
    numeric_cols = ["square_feet", "bedrooms", "bathrooms"]
    medians = df[numeric_cols].median()
    df[numeric_cols] = df[numeric_cols].fillna(medians)

    print("\nMissing values per column after imputation:")
    print(df.isna().sum())

    # Ensure types are numeric ints for features
    df["square_feet"] = df["square_feet"].astype(int)
    df["bedrooms"] = df["bedrooms"].astype(int)
    df["bathrooms"] = df["bathrooms"].astype(int)

    return df


def train_and_evaluate(df: pd.DataFrame):
    """
    Split the data, train Linear Regression, evaluate and visualize results.
    """
    features = ["square_feet", "bedrooms", "bathrooms"]
    target = "price"

    X = df[features]
    y = df[target]

    # Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("\nModel coefficients (feature importances):")
    coef_df = pd.DataFrame({
        "feature": features,
        "coefficient": model.coef_
    }).sort_values(by="coefficient", ascending=False).reset_index(drop=True)
    print(coef_df)

    print(f"\nIntercept (base price): {model.intercept_:.2f}")
    print(f"R^2 on test set: {r2:.4f}")
    print(f"Mean Squared Error on test set: {mse:,.2f}")

    # Visualization 1: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor="k")
    min_price = min(y_test.min(), y_pred.min())
    max_price = max(y_test.max(), y_pred.max())
    plt.plot([min_price, max_price], [min_price, max_price], color="red", linestyle="--", label="Perfect prediction")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs. Predicted House Prices")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Visualization 2: Feature coefficients
    plt.figure(figsize=(6, 4))
    plt.bar(coef_df["feature"], coef_df["coefficient"], color="skyblue", edgecolor="k")
    plt.ylabel("Coefficient value (USD per unit)")
    plt.title("Linear Regression Feature Coefficients")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Return model and metrics if further use is needed
    return {
        "model": model,
        "r2": r2,
        "mse": mse,
        "coef_df": coef_df
    }


def main():
    print("House Price Prediction using Linear Regression")
    print("=============================================")

    # 1) Generate a synthetic dataset (500 samples) as no external dataset was provided.
    df = generate_synthetic_data(n_samples=500)

    # 2) Preprocess the data (handle missing values etc.)
    df = preprocess_data(df)

    # 3) Train the model and visualize/evaluate
    results = train_and_evaluate(df)

    # Print a short summary
    print("\nSummary:")
    print(f"- Test R^2: {results['r2']:.4f}")
    print(f"- Test MSE: {results['mse']:.2f}")
    print("- Feature coefficients:")
    print(results["coef_df"].to_string(index=False))


if __name__ == "__main__":
    main()