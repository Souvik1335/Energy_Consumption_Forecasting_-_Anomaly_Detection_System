from pathlib import Path

import joblib


# Project Base Directory

BASE_DIR = Path(__file__).resolve().parents[1]


# Model File Paths

FORECASTING_MODEL_FILE = (
    BASE_DIR
    / "models"
    / "forecasting_model.pkl"
)

ANOMALY_CONFIG_FILE = (
    BASE_DIR
    / "models"
    / "anomaly_config.pkl"
)


# Load Forecasting Model

print("Loading forecasting model...")
forecasting_model = joblib.load(FORECASTING_MODEL_FILE)
print("Forecasting model loaded successfully.")


# Load Anomaly Configuration

print("Loading anomaly configuration...")
anomaly_config = joblib.load(ANOMALY_CONFIG_FILE)
print("Anomaly configuration loaded successfully.")


# Load Anomaly Threshold

anomaly_threshold = anomaly_config["anomaly_threshold"]


# Load Feature Columns

feature_columns = anomaly_config["feature_columns"]


# Load Target Column

target_column = anomaly_config["target_column"]


# Display Model Information

print("\nP2 Model Loading Completed.")
print("Forecasting Model :- ", type(forecasting_model))
print("Anomaly Threshold :- ", anomaly_threshold)
print("Target Column :- ", target_column)
print("Number of Features :- ", len(feature_columns))