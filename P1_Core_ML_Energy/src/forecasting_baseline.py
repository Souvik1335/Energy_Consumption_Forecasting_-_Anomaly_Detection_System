from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Project Base Directory

BASE_DIR = Path(__file__).resolve().parents[1]


# Data File Paths

TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.csv"
VALIDATION_FILE = BASE_DIR / "data" / "processed" / "validation.csv"
TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"


# Load Datasets

print("Loading train, validation and test datasets...")

train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
validation_df = pd.read_csv(VALIDATION_FILE, low_memory=False)
test_df = pd.read_csv(TEST_FILE, low_memory=False)

print("Datasets loaded successfully.")


# Define Target

target_column = "Global_active_power"


# Define Baseline Feature

baseline_feature = "Lag_1"


# Remove Missing Values

validation_baseline = validation_df[
    [baseline_feature, target_column]
].dropna()

test_baseline = test_df[
    [baseline_feature, target_column]
].dropna()


# Generate Validation Predictions

print("\nGenerating validation baseline predictions...")

y_validation = validation_baseline[target_column]
y_validation_pred = validation_baseline[baseline_feature]


# Generate Test Predictions

print("Generating test baseline predictions...")

y_test = test_baseline[target_column]
y_test_pred = test_baseline[baseline_feature]


# Calculate Validation Metrics

validation_mae = mean_absolute_error(
    y_validation,
    y_validation_pred
)

validation_mse = mean_squared_error(
    y_validation,
    y_validation_pred
)

validation_rmse = validation_mse ** 0.5

validation_r2 = r2_score(
    y_validation,
    y_validation_pred
)


# Calculate Test Metrics

test_mae = mean_absolute_error(
    y_test,
    y_test_pred
)

test_mse = mean_squared_error(
    y_test,
    y_test_pred
)

test_rmse = test_mse ** 0.5

test_r2 = r2_score(
    y_test,
    y_test_pred
)


# Display Validation Results

print("\nBaseline Validation Results")

print(f"MAE  : {validation_mae:.4f}")
print(f"MSE  : {validation_mse:.4f}")
print(f"RMSE : {validation_rmse:.4f}")
print(f"R²   : {validation_r2:.4f}")


# Display Test Results

print("\nBaseline Test Results")

print(f"MAE  : {test_mae:.4f}")
print(f"MSE  : {test_mse:.4f}")
print(f"RMSE : {test_rmse:.4f}")
print(f"R²   : {test_r2:.4f}")


# Save Baseline Predictions

OUTPUT_DIR = BASE_DIR / "outputs" / "predictions"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


validation_predictions = validation_baseline.copy()

validation_predictions["Predicted_Global_active_power"] = (
    y_validation_pred.values
)


test_predictions = test_baseline.copy()

test_predictions["Predicted_Global_active_power"] = (
    y_test_pred.values
)


validation_predictions.to_csv(
    OUTPUT_DIR / "baseline_validation_predictions.csv",
    index=False
)

test_predictions.to_csv(
    OUTPUT_DIR / "baseline_test_predictions.csv",
    index=False
)


# Completion Message

print("\nBaseline predictions saved successfully.")

print(
    f"Validation Predictions: "
    f"{OUTPUT_DIR / 'baseline_validation_predictions.csv'}"
)

print(
    f"Test Predictions: "
    f"{OUTPUT_DIR / 'baseline_test_predictions.csv'}"
)