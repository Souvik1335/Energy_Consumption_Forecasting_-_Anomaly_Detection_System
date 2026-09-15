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


# Sort Dataset

print("\nSorting dataset by Datetime...")

df = df.sort_values(
    "Datetime"
).reset_index(drop=True)

print("Dataset sorted successfully.")


# Create Time Based Features

print("\nCreating time based features...")

df["Year"] = df["Datetime"].dt.year

df["Month"] = df["Datetime"].dt.month

df["Day"] = df["Datetime"].dt.day

df["Hour"] = df["Datetime"].dt.hour

df["DayOfWeek"] = df["Datetime"].dt.dayofweek

print("Time based features created successfully.")


# Global Active Power Trend

print("\nCalculating Global Active Power trend...")

daily_consumption = (
    df.set_index("Datetime")
    ["Global_active_power"]
    .resample("D")
    .sum()
)

print("Daily consumption calculated successfully.")


print("\nPlotting daily energy consumption trend...")

plt.figure(figsize=(14, 6))

plt.plot(
    daily_consumption.index,
    daily_consumption.values
)

plt.xlabel("Date")

plt.ylabel("Daily Global Active Power")

plt.title(
    "Daily Energy Consumption Trend"
)

plt.grid()

plt.tight_layout()

plt.show()


# Hourly Consumption Pattern

print("\nCalculating hourly consumption pattern...")

hourly_consumption = (
    df.groupby("Hour")["Global_active_power"]
    .mean()
)

print("Hourly consumption calculated successfully.")


print("\nPlotting hourly consumption pattern...")

plt.figure(figsize=(10, 6))

plt.plot(
    hourly_consumption.index,
    hourly_consumption.values,
    marker="o"
)

plt.xlabel("Hour")

plt.ylabel("Average Global Active Power")

plt.title(
    "Average Energy Consumption by Hour"
)

plt.xticks(range(24))

plt.grid()

plt.tight_layout()

plt.show()


# Day Of Week Consumption Pattern

print("\nCalculating day-of-week consumption pattern...")

day_of_week_consumption = (
    df.groupby("DayOfWeek")["Global_active_power"]
    .mean()
)

print("Day-of-week consumption calculated successfully.")


print("\nPlotting day-of-week consumption pattern...")

plt.figure(figsize=(10, 6))

plt.plot(
    day_of_week_consumption.index,
    day_of_week_consumption.values,
    marker="o"
)

plt.xlabel("Day of Week")

plt.ylabel("Average Global Active Power")

plt.title(
    "Average Energy Consumption by Day of Week"
)

plt.xticks(range(7))

plt.grid()

plt.tight_layout()

plt.show()


# Monthly Consumption Pattern

print("\nCalculating monthly consumption pattern...")

monthly_consumption = (
    df.groupby("Month")["Global_active_power"]
    .mean()
)

print("Monthly consumption calculated successfully.")


print("\nPlotting monthly consumption pattern...")

plt.figure(figsize=(10, 6))

plt.plot(
    monthly_consumption.index,
    monthly_consumption.values,
    marker="o"
)

plt.xlabel("Month")

plt.ylabel("Average Global Active Power")

plt.title(
    "Average Energy Consumption by Month"
)

plt.xticks(range(1, 13))

plt.grid()

plt.tight_layout()

plt.show()


# Yearly Consumption Pattern

print("\nCalculating yearly consumption pattern...")

yearly_consumption = (
    df.groupby("Year")["Global_active_power"]
    .mean()
)

print("Yearly consumption calculated successfully.")


print("\nPlotting yearly consumption pattern...")

plt.figure(figsize=(10, 6))

plt.plot(
    yearly_consumption.index,
    yearly_consumption.values,
    marker="o"
)

plt.xlabel("Year")

plt.ylabel("Average Global Active Power")

plt.title(
    "Average Energy Consumption by Year"
)

plt.grid()

plt.tight_layout()

plt.show()


# Global Active Power Distribution

print("\nPlotting Global Active Power distribution...")

plt.figure(figsize=(10, 6))

plt.hist(
    df["Global_active_power"],
    bins=100
)

plt.xlabel("Global Active Power")

plt.ylabel("Frequency")

plt.title(
    "Global Active Power Distribution"
)

plt.grid()

plt.tight_layout()

plt.show()


# Global Active Power Boxplot

print("\nPlotting Global Active Power boxplot...")

plt.figure(figsize=(10, 6))

plt.boxplot(
    df["Global_active_power"]
)

plt.ylabel("Global Active Power")

plt.title(
    "Global Active Power Boxplot"
)

plt.grid()

plt.tight_layout()

plt.show()


# Sub Metering Comparison

print("\nCalculating average sub-metering consumption...")

sub_metering_columns = [
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

sub_metering_average = (
    df[sub_metering_columns].mean()
)

print("\nAverage Sub-Metering Consumption:")

print(sub_metering_average)


print("\nPlotting sub-metering comparison...")

plt.figure(figsize=(10, 6))

plt.bar(
    sub_metering_average.index,
    sub_metering_average.values
)

plt.xlabel("Sub-Metering")

plt.ylabel("Average Consumption")

plt.title(
    "Average Consumption by Sub-Metering"
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.show()


# Voltage Distribution

print("\nPlotting Voltage distribution...")

plt.figure(figsize=(10, 6))

plt.hist(
    df["Voltage"],
    bins=100
)

plt.xlabel("Voltage")

plt.ylabel("Frequency")

plt.title(
    "Voltage Distribution"
)

plt.grid()

plt.tight_layout()

plt.show()


# Global Intensity Distribution

print("\nPlotting Global Intensity distribution...")

plt.figure(figsize=(10, 6))

plt.hist(
    df["Global_intensity"],
    bins=100
)

plt.xlabel("Global Intensity")

plt.ylabel("Frequency")

plt.title(
    "Global Intensity Distribution"
)

plt.grid()

plt.tight_layout()

plt.show()


# Correlation Analysis

print("\nCalculating correlation matrix...")

numeric_columns = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

correlation_matrix = (
    df[numeric_columns].corr()
)

print("\nCorrelation Matrix:")

print(correlation_matrix)


# Correlation Heatmap

print("\nPlotting correlation heatmap...")

plt.figure(figsize=(12, 8))

plt.imshow(
    correlation_matrix,
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
    "Correlation Matrix of Energy Features"
)

plt.tight_layout()

plt.show()


# Hourly Consumption Heatmap

print("\nCalculating hourly consumption by day of week...")

hour_day_consumption = (
    df.groupby(
        ["DayOfWeek", "Hour"]
    )["Global_active_power"]
    .mean()
    .unstack()
)

print("\nHourly Consumption by Day of Week:")

print(hour_day_consumption)


print("\nPlotting hourly consumption heatmap...")

plt.figure(figsize=(14, 6))

plt.imshow(
    hour_day_consumption,
    aspect="auto",
    interpolation="nearest"
)

plt.colorbar()

plt.xlabel("Hour")

plt.ylabel("Day of Week")

plt.title(
    "Average Energy Consumption by Day and Hour"
)

plt.xticks(
    range(24)
)

plt.yticks(
    range(7)
)

plt.tight_layout()

plt.show()


# Final Visualization Summary

print("\n" + "=" * 50)

print("FINAL EDA VISUALIZATION COMPLETED")

print("=" * 50)

print("\nTotal Rows:")

print(len(df))

print("\nDate Range:")

print(
    df["Datetime"].min(),
    "to",
    df["Datetime"].max()
)

print("\nAverage Global Active Power:")

print(
    df["Global_active_power"].mean()
)

print("\nMaximum Global Active Power:")

print(
    df["Global_active_power"].max()
)

print("\nVisualization analysis completed successfully.")