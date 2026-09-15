from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "household_power_consumption_cleaned.csv"
)


# Load Dataset

print("Loading cleaned dataset...")

df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

print("Cleaned dataset loaded successfully.")


# Dataset Shape
print("\nDataset shape:")

print(df.shape)


# Dataset Columns

print("\nDataset columns:")

print(df.columns.tolist())


# Data Types

print("\nData types:")

print(df.dtypes)


# First 5 Rows

print("\nFirst 5 rows:")

print(df.head())


# Last 5 Rows
print("\nLast 5 rows:")

print(df.tail())


# Missing Value Analysis

print("\nChecking missing values:")

missing_values = df.isna().sum()

print(missing_values)


print("\nTotal missing values:")

print(missing_values.sum())


# Duplicate Row Analysis

print("\nChecking duplicate rows...")

duplicate_rows = df.duplicated().sum()

print(
    "Duplicate rows:",
    duplicate_rows
)


# Datetime Analysis

print("\nConverting Datetime column...")

df["Datetime"] = pd.to_datetime(
    df["Datetime"],
    errors="coerce"
)

print("Datetime converted successfully.")


print("\nDatetime data type:")

print(df["Datetime"].dtype)


print("\nChecking Datetime values...")

print(
    "Invalid Datetime values:",
    df["Datetime"].isna().sum()
)

print(
    "Minimum Datetime:",
    df["Datetime"].min()
)

print(
    "Maximum Datetime:",
    df["Datetime"].max()
)


#  Duplicate Datetime Analysis

print("\nChecking duplicate Datetime values...")

duplicate_datetime = df["Datetime"].duplicated().sum()

print(
    "Duplicate Datetime values:",
    duplicate_datetime
)


# Numericalb Column

print("\nChecking numerical columns...")

numeric_columns = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

print("\nNumerical columns:")

print(numeric_columns)


# Numerical Data Types

print("\nNumerical column data types:")

print(
    df[numeric_columns].dtypes
)


# Descriptive Statistics

print("\nDescriptive Statistics:")

print(
    df[numeric_columns].describe()
)


print("\nTransposed Descriptive Statistics:")

print(
    df[numeric_columns].describe().T
)


# Unique Value Analysis

print("\nChecking unique values...")

for column in numeric_columns:

    print(
        column,
        "unique values:",
        df[column].nunique()
    )


# Zero Value Analysis

print("\nChecking zero values...")

for column in numeric_columns:

    print(
        column,
        "zero values:",
        (df[column] == 0).sum()
    )


# Basic Global Active Power Analysis

print("\nChecking Global Active Power statistics...")

print(
    df["Global_active_power"].describe()
)


print("\nAverage Global Active Power:")

print(
    df["Global_active_power"].mean()
)


print("\nMedian Global Active Power:")

print(
    df["Global_active_power"].median()
)


print("\nMinimum Global Active Power:")

print(
    df["Global_active_power"].min()
)


print("\nMaximum Global Active Power:")

print(
    df["Global_active_power"].max()
)


print("\nStandard Deviation:")

print(
    df["Global_active_power"].std()
)


# basic EDA Summery

print("\n" + "=" * 60)

print("BASIC EDA COMPLETED SUCCESSFULLY")

print("=" * 60)

print("\nDataset Shape:")

print(df.shape)

print("\nDate Range:")

print(
    df["Datetime"].min(),
    "to",
    df["Datetime"].max()
)

print("\nTotal Missing Values:")

print(
    df.isna().sum().sum()
)

print("\nDuplicate Rows:")

print(
    df.duplicated().sum()
)

print("\nDuplicate Datetime Values:")

print(
    df["Datetime"].duplicated().sum()
)

print("\nBasic EDA analysis completed successfully.")