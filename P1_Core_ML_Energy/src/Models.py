from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

import joblib


# Project Base Directory

BASE_DIR = Path(__file__).resolve().parents[1]


# Data File Paths

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "train.csv"
)

VALIDATION_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "validation.csv"
)


# Model Directory

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Load Datasets

print("Loading train and validation datasets...")

train_df = pd.read_csv(
    TRAIN_FILE,
    low_memory=False
)

validation_df = pd.read_csv(
    VALIDATION_FILE,
    low_memory=False
)

print("Datasets loaded successfully.")


# Define Target

target_column = "Global_active_power"


# Define Features

feature_columns = [
    "Hour",
    "Day",
    "DayOfWeek",
    "Month",
    "Year",
    "IsWeekend",
    "IsPeakHour",
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Lag_24",
    "Lag_48",
    "Lag_60",
    "Lag_1440",
    "Lag_10080",
    "Rolling_Mean_15",
    "Rolling_Mean_60",
    "Rolling_Mean_1440",
    "Rolling_Std_15",
    "Rolling_Std_60",
    "Rolling_Std_1440"
]


# Combine Training and Validation Data

final_training_df = pd.concat(
    [
        train_df,
        validation_df
    ],
    ignore_index=True
)


# Create Features and Target

X_final = final_training_df[
    feature_columns
]

y_final = final_training_df[
    target_column
]


# Create Final Random Forest Model

final_model = RandomForestRegressor(
    n_estimators=150,
    min_samples_split=7,
    min_samples_leaf=5,
    max_depth=40,
    max_features="log2",
    random_state=42,
    n_jobs=-1
)


# Train Final Model

print("\nTraining Final Random Forest Model...")

final_model.fit(
    X_final,
    y_final
)

print("Final Random Forest Model trained successfully.")


# Save Forecasting Model

forecasting_model_file = (
    MODEL_DIR
    / "forecasting_model.pkl"
)

joblib.dump(
    final_model,
    forecasting_model_file
)


# Save Anomaly Configuration

anomaly_config = {
    "anomaly_threshold": 0.14389535964468322,
    "feature_columns": feature_columns,
    "target_column": target_column
}


anomaly_config_file = (
    MODEL_DIR
    / "anomaly_config.pkl"
)

joblib.dump(
    anomaly_config,
    anomaly_config_file
)


# Display Saved Files

print("\nModel Files Saved Successfully.")

print(
    "Forecasting Model :- ",
    forecasting_model_file
)

print(
    "Anomaly Configuration :- ",
    anomaly_config_file
)