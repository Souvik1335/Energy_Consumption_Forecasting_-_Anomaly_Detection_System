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


# Checking Missing Values

print("\nChecking missing values...")

print(
    df[numeric_columns].isna().sum()
)

print(
    "\nTotal missing values:",
    df[numeric_columns].isna().sum().sum()
)


# IQR Outlier Analysis

print("\nPerforming IQR outlier analysis...")

outlier_summary = []

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)

    upper_bound = Q3 + (1.5 * IQR)

    outliers = df[
        (df[column] < lower_bound)
        |
        (df[column] > upper_bound)
    ]

    outlier_count = len(outliers)

    outlier_percentage = (
        outlier_count / len(df)
    ) * 100

    print("\nColumn:", column)

    print("Q1:", Q1)

    print("Q3:", Q3)

    print("IQR:", IQR)

    print("Lower Bound:", lower_bound)

    print("Upper Bound:", upper_bound)

    print("Number of Outliers:", outlier_count)

    print(
        "Outlier Percentage:",
        outlier_percentage
    )

    outlier_summary.append(
        [
            column,
            Q1,
            Q3,
            IQR,
            lower_bound,
            upper_bound,
            outlier_count,
            outlier_percentage
        ]
    )


# Outlier Summary DataFrame

print("\nCreating outlier summary...")

outlier_summary = pd.DataFrame(
    outlier_summary,
    columns=[
        "Column",
        "Q1",
        "Q3",
        "IQR",
        "Lower_Bound",
        "Upper_Bound",
        "Outlier_Count",
        "Outlier_Percentage"
    ]
)

print("\nOutlier Summary:")

print(outlier_summary)


# Global Active Power Outliers

print("\nAnalyzing Global Active Power outliers...")

Q1 = df["Global_active_power"].quantile(0.25)

Q3 = df["Global_active_power"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - (1.5 * IQR)

upper_bound = Q3 + (1.5 * IQR)

global_active_power_outliers = df[
    (df["Global_active_power"] < lower_bound)
    |
    (df["Global_active_power"] > upper_bound)
]

print("\nGlobal Active Power Lower Bound:")

print(lower_bound)

print("\nGlobal Active Power Upper Bound:")

print(upper_bound)

print("\nNumber of Global Active Power Outliers:")

print(
    len(global_active_power_outliers)
)

print("\nGlobal Active Power Outlier Percentage:")

print(
    (
        len(global_active_power_outliers)
        / len(df)
    ) * 100
)


# Highest Global Active Power Values

print("\nFinding highest Global Active Power values...")

highest_consumption = (
    df[
        [
            "Datetime",
            "Global_active_power"
        ]
    ]
    .sort_values(
        "Global_active_power",
        ascending=False
    )
    .head(20)
)

print("\nTop 20 Global Active Power Values:")

print(highest_consumption)


# Lowest Global Active Power Values

print("\nFinding lowest Global Active Power values...")

lowest_consumption = (
    df[
        [
            "Datetime",
            "Global_active_power"
        ]
    ]
    .sort_values(
        "Global_active_power",
        ascending=True
    )
    .head(20)
)

print("\nBottom 20 Global Active Power Values:")

print(lowest_consumption)


# Z-Score Analysis

print("\nPerforming Z-Score analysis...")

z_score_summary = []

for column in numeric_columns:

    mean = df[column].mean()

    std = df[column].std()

    z_scores = (
        (df[column] - mean)
        / std
    )

    extreme_values = (
        z_scores.abs() > 3
    )

    extreme_count = extreme_values.sum()

    extreme_percentage = (
        extreme_count / len(df)
    ) * 100

    print("\nColumn:", column)

    print(
        "Mean:",
        mean
    )

    print(
        "Standard Deviation:",
        std
    )

    print(
        "Values with |Z-Score| > 3:",
        extreme_count
    )

    print(
        "Extreme Value Percentage:",
        extreme_percentage
    )

    z_score_summary.append(
        [
            column,
            mean,
            std,
            extreme_count,
            extreme_percentage
        ]
    )


# Z-Score Summary

print("\nCreating Z-Score summary...")

z_score_summary = pd.DataFrame(
    z_score_summary,
    columns=[
        "Column",
        "Mean",
        "Standard_Deviation",
        "Extreme_Value_Count",
        "Extreme_Value_Percentage"
    ]
)

print("\nZ-Score Summary:")

print(z_score_summary)


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
    "Global Active Power Outlier Analysis"
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
    "Global Reactive Power Outlier Analysis"
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
    "Voltage Outlier Analysis"
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
    "Global Intensity Outlier Analysis"
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
        f"{column} Outlier Analysis"
    )

    plt.grid()

    plt.show()


# Extreme Global Active Power Analysis

print("\nChecking extreme Global Active Power values...")

extreme_consumption = df[
    df["Global_active_power"]
    >= df["Global_active_power"].quantile(0.99)
]

print(
    "\n99th Percentile:"
)

print(
    df["Global_active_power"].quantile(0.99)
)

print(
    "\nNumber of values above 99th percentile:"
)

print(
    len(extreme_consumption)
)

print(
    "\nPercentage of values above 99th percentile:"
)

print(
    (
        len(extreme_consumption)
        / len(df)
    ) * 100
)


# Outlier Analysis Summary

print("\nOutlier analysis completed successfully.")

print("\nTotal Dataset Rows:")

print(len(df))

print("\nGlobal Active Power Outliers:")

print(
    len(global_active_power_outliers)
)

print("\nGlobal Active Power Outlier Percentage:")

print(
    (
        len(global_active_power_outliers)
        / len(df)
    ) * 100
)

print("\nOutlier analysis completed successfully.")