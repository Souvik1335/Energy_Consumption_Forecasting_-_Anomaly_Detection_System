from pathlib import Path

import pandas as pd
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "household_power_consumption.csv"

print("Loading dataset...")

df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

print("Dataset loaded successfully.")

df = df.replace("?", pd.NA)

print("Missing values converted to NaN.")

print("\nDataset shape:")
print(df.shape)

print("\nDataset columns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nFirst 5 rows:")
print(df.head())

# Numerical columns
numeric_columns = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

print("\nConverting numerical columns...")

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("Numerical columns converted successfully.")

print("\nUpdated data types:")
print(df.dtypes)

print("\nCreating datetime columns")

df['Datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'], format="%d/%m/%Y %H:%M:%S", errors="coerce")

print("\nDatetime preview:")
print(df[["Date", "Time", "Datetime"]].head())

print("\nDatetime data type:")
print(df["Datetime"].dtype)

print("\nChecking Datetime values...")

print("Invalid Datetime values:", df["Datetime"].isna().sum())

print("Minimum Datetime:", df["Datetime"].min())
print("Maximum Datetime:", df["Datetime"].max())

print("\nChecking duplicate Datetime values...")
print("Duplicate Datetime rows:", df["Datetime"].duplicated().sum())

print("\nSorting dataset by Datetime...")

df = df.sort_values("Datetime").reset_index(drop=True)

print("Dataset sorted successfully.")

print("\nFirst Datetime:", df["Datetime"].iloc[0])
print("Last Datetime:", df["Datetime"].iloc[-1])

print("\nMissing values after numeric conversion:")

missing_values = df.isna().sum()

print(missing_values)

print("\nTotal missing values:", missing_values.sum())

print("\nMissing value percentage:")

missing_percentage = (df.isna().sum() / len(df)) * 100

print(missing_percentage)

print("\nRows containing missing values:")

missing_rows = df[df[numeric_columns].isna().any(axis=1)]

print("Number of rows with missing values:", len(missing_rows))

print("\nFirst 10 missing-value rows:")
print(missing_rows.head(10))

print("Missing Value per row")

missing_value_pre_rows = df[numeric_columns].isna().sum()
print(missing_value_pre_rows.value_counts().sort_index())

print("Checking Time Interval")
time_difference = df['Datetime'].diff()

print("Checking the most common Time Interval")
print(time_difference.value_counts().head(10))

print("Handling the missings values using time-based interlocation")
df = df.sort_values("Datetime")
df = df.set_index("Datetime")

df[numeric_columns] = df[numeric_columns].interpolate(method="time")
print("Time-based interpolation completed.")

print("Missing value after interlocation")
print(df[numeric_columns].isna().sum())

print('Total remaining missing values :- \n', df[numeric_columns].isna().sum().sum())

print("\nSaving preprocessed dataset...")

OUTPUT_FILE = BASE_DIR / "data" / "processed" / "household_power_consumption_cleaned.csv"

df.to_csv(OUTPUT_FILE, index=True)

print(f"Preprocessed dataset saved successfully to: {OUTPUT_FILE}")