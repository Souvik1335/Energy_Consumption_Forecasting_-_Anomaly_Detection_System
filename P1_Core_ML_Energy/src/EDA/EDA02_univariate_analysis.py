from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


# Convert Datetime

print("\nConverting Datetime column...")

df["Datetime"] = pd.to_datetime(
    df["Datetime"],
    errors="coerce"
)

print("Datetime converted successfully.")


# Numerical Columns

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


# Descriptive Statistics

print("\nDescriptive Statistics:")

print(
    df[numeric_columns].describe().T
)


# Standard Deviation

print("\nStandard Deviation:")

print(
    df[numeric_columns].std()
)


# Variance

print("\nVariance:")

print(
    df[numeric_columns].var()
)


# Skewness

print("\nSkewness:")

skewness = df[numeric_columns].skew()

print(skewness)


# Kurtosis

print("\nKurtosis:")

kurtosis = df[numeric_columns].kurt()

print(kurtosis)


# Zero Value Analysis

print("\nChecking zero values...")

for column in numeric_columns:

    zero_values = (
        df[column] == 0
    ).sum()

    zero_percentage = (
        zero_values / len(df)
    ) * 100

    print("\nColumn:", column)

    print(
        "Zero values:",
        zero_values
    )

    print(
        "Zero percentage:",
        zero_percentage
    )


# Quartile Analysis

print("\nQuartile Analysis:")

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)

    Q2 = df[column].quantile(0.50)

    Q3 = df[column].quantile(0.75)

    print("\nColumn:", column)

    print("Q1:", Q1)

    print("Median:", Q2)

    print("Q3:", Q3)


# Global Active Power Analysis

print("\nGlobal Active Power Analysis:")

print(
    df["Global_active_power"].describe()
)

print(
    "\nGlobal Active Power Skewness:",
    df["Global_active_power"].skew()
)

print(
    "Global Active Power Kurtosis:",
    df["Global_active_power"].kurt()
)


# Global Reactive Power Analysis

print("\nGlobal Reactive Power Analysis:")

print(
    df["Global_reactive_power"].describe()
)

print(
    "\nGlobal Reactive Power Skewness:",
    df["Global_reactive_power"].skew()
)


# Voltage Analysis

print("\nVoltage Analysis:")

print(
    df["Voltage"].describe()
)

print(
    "\nVoltage Skewness:",
    df["Voltage"].skew()
)


# Global Intensity Analysis

print("\nGlobal Intensity Analysis:")

print(
    df["Global_intensity"].describe()
)

print(
    "\nGlobal Intensity Skewness:",
    df["Global_intensity"].skew()
)


# Sub Metering Analysis

print("\nSub Metering Analysis:")

for column in [
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]:

    print("\nColumn:", column)

    print(
        df[column].describe()
    )

    print(
        "Skewness:",
        df[column].skew()
    )


# Global Active Power Distribution

print("\nPlotting Global Active Power distribution...")

plt.figure(figsize=(10, 6))

plt.hist(
    df["Global_active_power"],
    bins=100
)

plt.xlabel(
    "Global Active Power"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Global Active Power Distribution"
)

plt.grid()

plt.show()


# Global Reactive Power Distribution

print("\nPlotting Global Reactive Power distribution...")

plt.figure(figsize=(10, 6))

plt.hist(
    df["Global_reactive_power"],
    bins=100
)

plt.xlabel(
    "Global Reactive Power"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Global Reactive Power Distribution"
)

plt.grid()

plt.show()


# Voltage Distribution

print("\nPlotting Voltage distribution...")

plt.figure(figsize=(10, 6))

plt.hist(
    df["Voltage"],
    bins=100
)

plt.xlabel(
    "Voltage"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Voltage Distribution"
)

plt.grid()

plt.show()


# Global Intensity Distribution

print("\nPlotting Global Intensity distribution...")

plt.figure(figsize=(10, 6))

plt.hist(
    df["Global_intensity"],
    bins=100
)

plt.xlabel(
    "Global Intensity"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Global Intensity Distribution"
)

plt.grid()

plt.show()


# Sub Metering Distributions

print("\nPlotting Sub Metering distributions...")

for column in [
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]:

    plt.figure(figsize=(10, 6))

    plt.hist(
        df[column],
        bins=50
    )

    plt.xlabel(column)

    plt.ylabel("Frequency")

    plt.title(
        f"{column} Distribution"
    )

    plt.grid()

    plt.show()


# Global Active Power Boxplot

print("\nPlotting Global Active Power boxplot...")

plt.figure(figsize=(10, 6))

plt.boxplot(
    df["Global_active_power"]
)

plt.ylabel(
    "Global Active Power"
)

plt.title(
    "Global Active Power Boxplot"
)

plt.grid()

plt.show()


# Global Reactive Power Boxplot

print("\nPlotting Global Reactive Power boxplot...")

plt.figure(figsize=(10, 6))

plt.boxplot(
    df["Global_reactive_power"]
)

plt.ylabel(
    "Global Reactive Power"
)

plt.title(
    "Global Reactive Power Boxplot"
)

plt.grid()

plt.show()


# Voltage Boxplot

print("\nPlotting Voltage boxplot...")

plt.figure(figsize=(10, 6))

plt.boxplot(
    df["Voltage"]
)

plt.ylabel(
    "Voltage"
)

plt.title(
    "Voltage Boxplot"
)

plt.grid()

plt.show()


# Global Intensity Boxplot

print("\nPlotting Global Intensity boxplot...")

plt.figure(figsize=(10, 6))

plt.boxplot(
    df["Global_intensity"]
)

plt.ylabel(
    "Global Intensity"
)

plt.title(
    "Global Intensity Boxplot"
)

plt.grid()

plt.show()


# Sub Metering Boxplots

print("\nPlotting Sub Metering boxplots...")

for column in [
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]:

    plt.figure(figsize=(10, 6))

    plt.boxplot(
        df[column]
    )

    plt.ylabel(column)

    plt.title(
        f"{column} Boxplot"
    )

    plt.grid()

    plt.show()


# Univariate Analysis Summary

print("\nUnivariate analysis completed successfully.")

print("\nMost positively skewed features:")

print(
    skewness.sort_values(
        ascending=False
    )
)

print("\nHighest variance features:")

print(
    df[numeric_columns]
    .var()
    .sort_values(
        ascending=False
    )
)