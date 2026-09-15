from pathlib import Path
import pandas as pd


# Project Directory
BASE_DIR = Path(__file__).resolve().parents[2]

# File Paths
INPUT_FILE = BASE_DIR / "data" / "processed" / "household_power_consumption_rolling_features.csv"


# Load Dataset
print("Loading rolling feature dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print(f"Dataset Shape: {df.shape}")


# Convert Datetime
print("\nConverting Datetime column...")

df["Datetime"] = pd.to_datetime(df["Datetime"])

df = df.sort_values("Datetime").reset_index(drop=True)

print("Datetime conversion completed.")


# Define Feature Columns
print("\nDefining feature columns...")

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

print("Feature columns defined successfully.")


# Target Column
target_column = "Global_active_power"

print(f"\nTarget Column: {target_column}")


# Missing Value Analysis
print("\nMissing Value Analysis Before Cleaning:")

print(df[feature_columns].isnull().sum())


# Remove Rows With Insufficient Historical Features
print("\nRemoving rows with insufficient historical data...")

initial_shape = df.shape

df = df.dropna(subset=feature_columns).reset_index(drop=True)

final_shape = df.shape

print(f"Initial Shape: {initial_shape}")
print(f"Final Shape: {final_shape}")

print(f"Rows Removed: {initial_shape[0] - final_shape[0]}")


# Check Missing Values
print("\nMissing Value Analysis After Cleaning:")

print(df[feature_columns].isnull().sum().sum())

print("Total Missing Values in Features:", df[feature_columns].isnull().sum().sum())


# Select Final Columns
print("\nSelecting final modeling columns...")

final_columns = [
    "Datetime",
    target_column
] + feature_columns

df_final = df[final_columns].copy()


# Final Dataset Information
print("\nFinal Feature Dataset Shape:")
print(df_final.shape)

print("\nFinal Dataset Columns:")
print(df_final.columns.tolist())


# Feature Preview
print("\nFinal Feature Dataset Preview:")

print(df_final.head())


# Datetime Range
print("\nFinal Datetime Range:")

print("Start:", df_final["Datetime"].min())
print("End:", df_final["Datetime"].max())


# Save Final Dataset
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "household_power_consumption_final_features.csv"

print("\nSaving final feature dataset...")

df_final.to_csv(OUTPUT_FILE, index=False)

print(f"Final feature dataset saved successfully to: {OUTPUT_FILE}")


# Completion Message
print("\nFinal Feature Engineering Completed Successfully.")