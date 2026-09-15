from pathlib import Path
import pandas as pd


# Project Directory
BASE_DIR = Path(__file__).resolve().parents[2]

# File Paths
INPUT_FILE = BASE_DIR / "data" / "processed" / "household_power_consumption_lag_features.csv"


# Load Dataset
print("Loading lag feature dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print(f"Dataset Shape: {df.shape}")


# Convert Datetime
print("\nConverting Datetime column...")

df["Datetime"] = pd.to_datetime(df["Datetime"])

df = df.sort_values("Datetime").reset_index(drop=True)

print("Datetime conversion completed.")


# Create Rolling Features
print("\nCreating rolling features...")

df["Rolling_Mean_15"] = (
    df["Global_active_power"]
    .shift(1)
    .rolling(15)
    .mean()
)

df["Rolling_Mean_60"] = (
    df["Global_active_power"]
    .shift(1)
    .rolling(60)
    .mean()
)

df["Rolling_Mean_1440"] = (
    df["Global_active_power"]
    .shift(1)
    .rolling(1440)
    .mean()
)


# Create Rolling Standard Deviation
df["Rolling_Std_15"] = (
    df["Global_active_power"]
    .shift(1)
    .rolling(15)
    .std()
)

df["Rolling_Std_60"] = (
    df["Global_active_power"]
    .shift(1)
    .rolling(60)
    .std()
)

df["Rolling_Std_1440"] = (
    df["Global_active_power"]
    .shift(1)
    .rolling(1440)
    .std()
)


# Display Created Features
print("\nRolling features created successfully.")

rolling_columns = [
    "Rolling_Mean_15",
    "Rolling_Mean_60",
    "Rolling_Mean_1440",
    "Rolling_Std_15",
    "Rolling_Std_60",
    "Rolling_Std_1440"
]

print("\nCreated Rolling Features:")
print(rolling_columns)


# Missing Value Analysis
print("\nMissing Values Created by Rolling Features:")

print(df[rolling_columns].isnull().sum())


# Feature Preview
print("\nRolling Feature Preview:")

print(
    df[
        [
            "Datetime",
            "Global_active_power",
            "Rolling_Mean_15",
            "Rolling_Mean_60",
            "Rolling_Mean_1440",
            "Rolling_Std_15",
            "Rolling_Std_60",
            "Rolling_Std_1440"
        ]
    ].head(15)
)


# Dataset Information
print("\nDataset Shape After Rolling Feature Engineering:")
print(df.shape)


# Save Dataset
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "household_power_consumption_rolling_features.csv"

print("\nSaving dataset...")

df.to_csv(OUTPUT_FILE, index=False)

print(f"Rolling feature dataset saved successfully to: {OUTPUT_FILE}")