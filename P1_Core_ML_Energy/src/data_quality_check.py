from pathlib import Path
import pandas as pd

# Load Dataset Path
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "processed" / "household_power_consumption.csv"

print("ENERGY CONSUMPTION DATA QUALITY CHECK")
print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE, sep=",", low_memory=False)

# Dataset Shape
print("DATASET SHAPE")
print("Rows   :", df.shape[0])
print("Columns:", df.shape[1])


print("COLUMNS")
for column in df.columns:
    print(column)


print("DATA TYPES")
print(df.dtypes)

print("MISSING VALUES")
missing = df.isna().sum()
print(missing)

print("FIRST 5 ROWS")
print(df.head())


print("LAST 5 ROWS")
print(df.tail())

print("7. DUPLICATE ROWS")
print("Duplicate rows:", df.duplicated().sum())

print("TARGET COLUMN INFORMATION")

print("Target:", "Global_active_power")
print(
    df["Global_active_power"]
    .value_counts(dropna=False)
    .head(10)
)
