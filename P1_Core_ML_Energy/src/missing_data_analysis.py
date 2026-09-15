from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "household_power_consumption.csv"


print("Loading dataset...")

df = pd.read_csv(DATA_FILE, low_memory=False)

print("Dataset loaded successfully.")


print("\nMissing values represented by '?'")
question_mark_counts = (df == "?").sum()
print(question_mark_counts)


print("\nMissing values represented by NaN")
nan_counts = df.isna().sum()
print(nan_counts)


total_missing = question_mark_counts + nan_counts

print("\nTotal missing values")
print(total_missing)


rows_with_question_mark = (df == "?").any(axis=1).sum()

print("\nRows containing '?'")
print(rows_with_question_mark)


rows_with_nan = df.isna().any(axis=1).sum()

print("\nRows containing NaN")
print(rows_with_nan)


missing_rows = df[
    (df == "?").any(axis=1) |
    df.isna().any(axis=1)
]

print("\nFirst 10 rows with missing values")
print(missing_rows.head(10).to_string(index=False))


missing_percentage = total_missing / len(df) * 100

print("\nMissing percentage")
print(missing_percentage.round(4))


print("\nMissing data analysis completed.")