from pathlib import Path
import pandas as pd


# Data File Path
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "household_power_consumption_final_features.csv"
)


# Data Loading
print("Loading final feature dataset...")

df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

print("Final feature dataset loaded successfully.")
print(f"Dataset Shape: {df.shape}")


# Convert Datetime
print("\nConverting Datetime column...")

df["Datetime"] = pd.to_datetime(df["Datetime"])

df = df.sort_values("Datetime").reset_index(drop=True)

print("Datetime conversion completed.")


# Define Target
target_column = "Global_active_power"

print(f"\nTarget Column: {target_column}")


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
X = df[feature_columns]
y = df[target_column]


# Chronological Train Validation Test Split
print("\nPerforming chronological train-validation-test split...")

total_rows = len(df)

train_end = int(total_rows * 0.70)
validation_end = int(total_rows * 0.85)

X_train = X.iloc[:train_end]
X_validation = X.iloc[train_end:validation_end]
X_test = X.iloc[validation_end:]

y_train = y.iloc[:train_end]
y_validation = y.iloc[train_end:validation_end]
y_test = y.iloc[validation_end:]


# Display Split Sizes
print("\nSplit completed successfully.")

print(f"Total Rows: {total_rows}")

print(f"Training Rows: {len(X_train)}")
print(f"Validation Rows: {len(X_validation)}")
print(f"Testing Rows: {len(X_test)}")


# Display Date Ranges
print("\nTraining Date Range:")
print(df["Datetime"].iloc[0])
print("to")
print(df["Datetime"].iloc[train_end - 1])


print("\nValidation Date Range:")
print(df["Datetime"].iloc[train_end])
print("to")
print(df["Datetime"].iloc[validation_end - 1])


print("\nTesting Date Range:")
print(df["Datetime"].iloc[validation_end])
print("to")
print(df["Datetime"].iloc[-1])


# Save Split Datasets
print("\nSaving split datasets...")

TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.csv"
VALIDATION_FILE = BASE_DIR / "data" / "processed" / "validation.csv"
TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"


train_df = df.iloc[:train_end].copy()
validation_df = df.iloc[train_end:validation_end].copy()
test_df = df.iloc[validation_end:].copy()


train_df.to_csv(TRAIN_FILE, index=False)
validation_df.to_csv(VALIDATION_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)


print("\nDatasets saved successfully.")

print(f"Training Dataset: {TRAIN_FILE}")
print(f"Validation Dataset: {VALIDATION_FILE}")
print(f"Testing Dataset: {TEST_FILE}")