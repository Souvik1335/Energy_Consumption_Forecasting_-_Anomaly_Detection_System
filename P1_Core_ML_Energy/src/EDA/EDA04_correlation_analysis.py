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


# Checking Data Types

print("\nChecking numerical column data types...")

print(
    df[numeric_columns].dtypes
)


# Checking Missing Values

print("\nChecking missing values...")

print(
    df[numeric_columns].isna().sum()
)

print(
    "\nTotal missing values:",
    df[numeric_columns].isna().sum().sum()
)


# Pearson Correlation

print("\nCalculating Pearson correlation...")

pearson_correlation = (
    df[numeric_columns]
    .corr(method="pearson")
)

print("\nPearson Correlation Matrix:")

print(pearson_correlation)


# Correlation With Global Active Power

print("\nCorrelation with Global Active Power:")

global_active_power_correlation = (
    pearson_correlation[
        "Global_active_power"
    ]
    .sort_values(
        ascending=False
    )
)

print(
    global_active_power_correlation
)


# Spearman Correlation

print("\nCalculating Spearman correlation...")

spearman_correlation = (
    df[numeric_columns]
    .corr(method="spearman")
)

print("\nSpearman Correlation Matrix:")

print(spearman_correlation)


# Spearman Correlation With Global Active Power

print("\nSpearman correlation with Global Active Power:")

spearman_global_active_power = (
    spearman_correlation[
        "Global_active_power"
    ]
    .sort_values(
        ascending=False
    )
)

print(
    spearman_global_active_power
)


# Strong Correlations

print("\nChecking strong correlations...")

correlation_values = (
    pearson_correlation[
        "Global_active_power"
    ]
    .drop("Global_active_power")
)

strong_correlations = (
    correlation_values[
        correlation_values.abs() >= 0.5
    ]
    .sort_values(
        ascending=False
    )
)

print("\nFeatures with absolute correlation >= 0.5:")

print(
    strong_correlations
)


# Correlation Difference

print("\nComparing Pearson and Spearman correlations...")

correlation_comparison = pd.DataFrame(
    {
        "Pearson": pearson_correlation[
            "Global_active_power"
        ],
        "Spearman": spearman_correlation[
            "Global_active_power"
        ]
    }
)

print(
    correlation_comparison
)


# Global Active Power Correlation Plot

print("\nPlotting correlation with Global Active Power...")

correlation_plot = (
    pearson_correlation[
        "Global_active_power"
    ]
    .drop("Global_active_power")
    .sort_values()
)

plt.figure(figsize=(10, 6))

plt.barh(
    correlation_plot.index,
    correlation_plot.values
)

plt.xlabel(
    "Pearson Correlation"
)

plt.ylabel(
    "Features"
)

plt.title(
    "Feature Correlation with Global Active Power"
)

plt.grid(
    axis="x"
)

plt.show()


# Correlation Matrix

print("\nPlotting correlation matrix...")

plt.figure(figsize=(12, 8))

plt.imshow(
    pearson_correlation,
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(numeric_columns)),
    numeric_columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(numeric_columns)),
    numeric_columns
)

plt.title(
    "Pearson Correlation Matrix"
)

plt.tight_layout()

plt.show()


# Correlation Analysis Summary

print("\nCorrelation analysis completed successfully.")

print("\nStrongest positive correlation with Global Active Power:")

print(
    correlation_values.idxmax()
)

print(
    correlation_values.max()
)

print("\nStrongest negative correlation with Global Active Power:")

print(
    correlation_values.idxmin()
)

print(
    correlation_values.min()
)

print("\nCorrelation analysis completed successfully.")