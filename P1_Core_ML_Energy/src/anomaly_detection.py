from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import (RandomizedSearchCV, TimeSeriesSplit)
import warnings
warnings.filterwarnings("ignore")


# Project Base Directory

BASE_DIR = Path(__file__).resolve().parents[1]


# Data File Paths

TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.csv"

VALIDATION_FILE = BASE_DIR / "data" / "processed" / "validation.csv"

TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"


# Load Datasets

print("Loading train, validation and test datasets...")

train_df = pd.read_csv(
    TRAIN_FILE,
    low_memory=False
)

validation_df = pd.read_csv(
    VALIDATION_FILE,
    low_memory=False
)

test_df = pd.read_csv(
    TEST_FILE,
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


# Create Features and Target

X_train = train_df[feature_columns]
y_train = train_df[target_column]

X_validation = validation_df[feature_columns]
y_validation = validation_df[target_column]

X_test = test_df[feature_columns]
y_test = test_df[target_column]

# Forecasting Model Creation

forecasting_model = RandomForestRegressor( n_estimators=150, min_samples_split=7,
                                           min_samples_leaf=5, max_depth=40, 
                                           max_features="log2", random_state=42, n_jobs=-1 )

# Train Forecasting Model 

forecasting_model.fit( X_train, y_train )

# Generate Predictions

validation_pred = forecasting_model.predict( X_validation )

# Calculate Residuals 

validation_residual = ( y_validation - validation_pred )

# Calculate Absolute Residuals 

validation_absolute_residual = ( validation_residual.abs() )

# Display Residual Information 

print("\nResidual Analysis") 
print( "Mean Absolute Residual is :- ", validation_absolute_residual.mean() )
print( "Maximum Absolute Residual is :- ", validation_absolute_residual.max() )


# Calculate Q1 and Q3

Q1 = validation_absolute_residual.quantile(0.25)

Q3 = validation_absolute_residual.quantile(0.75)


# Calculate IQR

IQR = Q3 - Q1


# Calculate Anomaly Threshold

anomaly_threshold = Q3 + (1.5 * IQR)


# Display Threshold Information

print("\nAnomaly Threshold Analysis")

print("Q1 is :- ", Q1)
print("Q3 is :- ", Q3)
print("IQR is :- ", IQR)

print("Anomaly Threshold is :- ", anomaly_threshold)

# Create Anomaly Flag

validation_anomaly = (
    validation_absolute_residual > anomaly_threshold
).astype(int)


# Count Anomalies

anomaly_count = validation_anomaly.sum()

normal_count = (validation_anomaly == 0).sum()

# Calculate Anomaly Percentage

anomaly_percentage = (anomaly_count / len(validation_anomaly)) * 100

# Display Anomaly Information

print("\nAnomaly Detection Results")

print("Total Validation Records is :- ", len(validation_anomaly))
print("Normal Records is :- ", normal_count)
print("Anomaly Records is :- ", anomaly_count)
print("Anomaly Percentage is :- ", anomaly_percentage)

# Create Anomaly Analysis DataFrame

anomaly_analysis_df = validation_df[
    ["Datetime", target_column]
].copy()

anomaly_analysis_df["Predicted_Global_Active_Power"] = validation_pred

anomaly_analysis_df["Residual"] = validation_residual

anomaly_analysis_df["Absolute_Residual"] = validation_absolute_residual

anomaly_analysis_df["Anomaly"] = validation_anomaly


# Sort Anomalies by Absolute Residual

anomaly_records = anomaly_analysis_df[
    anomaly_analysis_df["Anomaly"] == 1
].sort_values(
    "Absolute_Residual",
    ascending=False
)


# Display Top Anomalies

print("\nTop 10 Anomalies")

print(
    anomaly_records[
        [
            "Datetime",
            target_column,
            "Predicted_Global_Active_Power",
            "Residual",
            "Absolute_Residual",
            "Anomaly"
        ]
    ].head(10)
)

# Save Anomaly Results

ANOMALY_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "validation_anomaly_results.csv"
)


anomaly_analysis_df.to_csv(
    ANOMALY_FILE,
    index=False
)


print("\nAnomaly Results Saved Successfully.")

print(
    "Saved File :- ",
    ANOMALY_FILE
)