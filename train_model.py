"""
International Car Price Prediction
----------------------------------
End-to-end Machine Learning pipeline for predicting international car prices
using Decision Tree Regressor.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def categorize_make(make: str) -> str:
    """Categorize car manufacturers based on segment and market origin."""
    luxury_european = [
        'mazda', 'mg', 'rover', 'alfa-romeo', 'audi', 'peugeot',
        'chrysler', 'bmw', 'aston-martin', 'jaguar', 'land-rover'
    ]
    mainstream_european = [
        'renault', 'dacia', 'citroen', 'volvo', 'fiat', 'opel',
        'seat', 'volkswagen', 'skoda', 'mini', 'smart'
    ]
    eastern_european = [
        'gaz', 'aro', 'lada-vaz', 'izh', 'raf', 'bogdan', 'moskvich',
        'uaz', 'luaz', 'wartburg', 'trabant', 'proton', 'fso', 'jac',
        'iran-khodro', 'zotye', 'tagaz', 'saipa', 'brilliance'
    ]
    asian = [
        'toyota', 'nissan', 'asia', 'mitsubishi', 'chery', 'hyundai',
        'honda', 'ssangyong', 'suzuki', 'daihatsu', 'kia', 'changan',
        'lexus', 'isuzu', 'great-wall', 'daewoo', 'vortex', 'infiniti',
        'byd', 'geely', 'haval', 'acura', 'scion', 'tata', 'datsun',
        'ravon'
    ]
    american = [
        'oldsmobile', 'gmc', 'plymouth', 'ford', 'cadillac', 'jeep',
        'mercury', 'lincoln', 'buick', 'saturn', 'pontiac', 'chevrolet'
    ]
    specialty = ['porsche', 'bentley', 'maserati', 'tesla', 'mclaren']

    if make in luxury_european:
        return 'Luxury European'
    elif make in mainstream_european:
        return 'Mainstream European'
    elif make in eastern_european:
        return 'Russian/Eastern European'
    elif make in asian:
        return 'Asian'
    elif make in american:
        return 'American'
    elif make in specialty:
        return 'Specialty'
    else:
        return 'Other'


def run_pipeline(data_path: str = "cars.csv"):
    print("=" * 60)
    print("Starting International Car Price Prediction Pipeline")
    print("=" * 60)

    # 1. Load Data
    print(f"\n[1/7] Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Preprocessing Part 1
    print("\n[2/7] Preprocessing and Categorization...")
    df.drop(columns=['model', 'segment'], inplace=True, errors='ignore')
    df['make_segment'] = df['make'].apply(categorize_make)

    # Filter years after 1980
    df = df[df['year'] > 1980]
    print(f"Shape after filtering year > 1980: {df.shape}")

    # 3. Preprocessing Part 2 - Missing values & Encoding
    print("\n[3/7] Handling missing values and label encoding...")
    df.dropna(inplace=True)
    df.drop(columns=['make'], inplace=True, errors='ignore')

    cols_to_encode = ['condition', 'fuel_type', 'transmission', 'color', 'drive_unit', 'make_segment']
    encoders = {}
    for col in cols_to_encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # 4. Outlier Removal via Z-score
    print("\n[4/7] Removing outliers using Z-score...")
    z = np.abs(stats.zscore(df))
    df = df[(z < 3).all(axis=1)]
    print(f"Shape after outlier removal: {df.shape}")

    # 5. Train-Test Split
    print("\n[5/7] Splitting dataset (80% train, 20% test)...")
    X = df.drop(columns=['priceUSD'])
    y = df['priceUSD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # 6. Model Training (Decision Tree Regressor)
    print("\n[6/7] Training Decision Tree Regressor...")
    # Best parameters identified from GridSearchCV
    dtr = DecisionTreeRegressor(
        max_depth=8,
        max_features=None,  # 'auto' is deprecated in newer scikit-learn, None maintains equivalent behavior
        min_samples_leaf=4,
        min_samples_split=2,
        random_state=0
    )
    dtr.fit(X_train, y_train)

    # 7. Model Evaluation
    print("\n[7/7] Model Evaluation:")
    train_score = dtr.score(X_train, y_train)
    y_pred = dtr.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("-" * 40)
    print(f"Training R² Score : {train_score:.4f}")
    print(f"Testing R² Score  : {r2:.4f}")
    print(f"Mean Absolute Error (MAE)       : ${mae:,.2f}")
    print(f"Mean Squared Error (MSE)        : {mse:,.2f}")
    print(f"Root Mean Squared Error (RMSE)  : ${rmse:,.2f}")
    print("-" * 40)

    # Feature Importance
    feat_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': dtr.feature_importances_})
    feat_df = feat_df.sort_values(by='Importance', ascending=False)
    print("\nFeature Importances:")
    print(feat_df.to_string(index=False))

    return dtr, feat_df


if __name__ == "__main__":
    run_pipeline()
