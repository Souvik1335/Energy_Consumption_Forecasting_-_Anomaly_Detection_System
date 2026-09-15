from pathlib import Path
import pandas as pd


# Project Directory
BASE_DIR = Path(__file__).resolve().parents[2]

# File Paths
INPUT_FILE = BASE_DIR / "data" / "processed" / "household_power_consumption_cleaned.csv"


# Load Dataset
print("Loading cleaned dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print(f"Dataset Shape: {df.shape}")


# Convert Datetime
print("\nConverting Datetime column...")

df["Datetime"] = pd.to_datetime(df["Datetime"])

df = df.sort_values("Datetime").reset_index(drop=True)

print("Datetime conversion completed.")


# Create Time Features
print("\nCreating time-based features...")

df["Hour"] = df["Datetime"].dt.hour
df["Day"] = df["Datetime"].dt.day
df["DayOfWeek"] = df["Datetime"].dt.dayofweek
df["Month"] = df["Datetime"].dt.month
df["Year"] = df["Datetime"].dt.year


# Create Weekend Feature
df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)


# Create Peak Hour Feature
df["IsPeakHour"] = df["Hour"].isin([7, 8, 9, 18, 19, 20, 21]).astype(int)


# Display Created Features
print("\nTime features created successfully.")

print("\nNew Features:")
print([
    "Hour",
    "Day",
    "DayOfWeek",
    "Month",
    "Year",
    "IsWeekend",
    "IsPeakHour"
])


# Dataset Information
print("\nDataset Shape After Feature Engineering:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())


# Feature Preview
print("\nFeature Preview:")
print(df[
    [
        "Datetime",
        "Hour",
        "Day",
        "DayOfWeek",
        "Month",
        "Year",
        "IsWeekend",
        "IsPeakHour"
    ]
].head(10))


# Save Dataset
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "household_power_consumption_time_features.csv"

print("\nSaving dataset...")

df.to_csv(OUTPUT_FILE, index=False)

print(f"Time feature dataset saved successfully to: {OUTPUT_FILE}")