from pathlib import Path
import pandas as pd


# Project Directory
BASE_DIR = Path(__file__).resolve().parents[2]

# File Paths
INPUT_FILE = BASE_DIR / "data" / "processed" / "household_power_consumption_time_features.csv"


# Load Dataset
print("Loading time feature dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print(f"Dataset Shape: {df.shape}")


# Convert Datetime
print("\nConverting Datetime column...")

df["Datetime"] = pd.to_datetime(df["Datetime"])

df = df.sort_values("Datetime").reset_index(drop=True)

print("Datetime conversion completed.")


# Create Lag Features
print("\nCreating lag features...")

df["Lag_1"] = df["Global_active_power"].shift(1)
df["Lag_2"] = df["Global_active_power"].shift(2)
df["Lag_3"] = df["Global_active_power"].shift(3)

df["Lag_24"] = df["Global_active_power"].shift(24)
df["Lag_48"] = df["Global_active_power"].shift(48)

df["Lag_60"] = df["Global_active_power"].shift(60)

df["Lag_1440"] = df["Global_active_power"].shift(1440)

df["Lag_10080"] = df["Global_active_power"].shift(10080)


# Display Created Features
print("\nLag features created successfully.")

print("\nCreated Lag Features:")
print([
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Lag_24",
    "Lag_48",
    "Lag_60",
    "Lag_1440",
    "Lag_10080"
])


# Missing Value Analysis
print("\nMissing Values Created by Lag Features:")

lag_columns = [
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Lag_24",
    "Lag_48",
    "Lag_60",
    "Lag_1440",
    "Lag_10080"
]

print(df[lag_columns].isnull().sum())


# Feature Preview
print("\nLag Feature Preview:")

print(
    df[
        [
            "Datetime",
            "Global_active_power",
            "Lag_1",
            "Lag_2",
            "Lag_3",
            "Lag_24",
            "Lag_48",
            "Lag_60",
            "Lag_1440",
            "Lag_10080"
        ]
    ].head(15)
)


# Dataset Information
print("\nDataset Shape After Lag Feature Engineering:")
print(df.shape)


# Save Dataset
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "household_power_consumption_lag_features.csv"

print("\nSaving dataset...")

df.to_csv(OUTPUT_FILE, index=False)

print(f"Lag feature dataset saved successfully to: {OUTPUT_FILE}")