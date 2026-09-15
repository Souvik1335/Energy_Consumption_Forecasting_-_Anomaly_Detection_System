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

df["Minute"] = df["Datetime"].dt.minute

df["DayOfWeek"] = df["Datetime"].dt.dayofweek

df["DayName"] = df["Datetime"].dt.day_name()

df["MonthName"] = df["Datetime"].dt.month_name()

print("Time based features created successfully.")


# Time Based Feature Preview

print("\nTime based feature preview:")

print(
    df[
        [
            "Datetime",
            "Year",
            "Month",
            "Day",
            "Hour",
            "Minute",
            "DayOfWeek",
            "DayName",
            "MonthName"
        ]
    ].head()
)


# Hourly Consumption Analysis

print("\nCalculating hourly consumption...")

hourly_consumption = (
    df.groupby("Hour")["Global_active_power"]
    .mean()
)

print("\nHourly Average Global Active Power:")

print(hourly_consumption)


print("\nPeak Consumption Hour:")

print(
    hourly_consumption.idxmax()
)

print(
    "Peak Hour Average Consumption:",
    hourly_consumption.max()
)


print("\nLowest Consumption Hour:")

print(
    hourly_consumption.idxmin()
)

print(
    "Lowest Hour Average Consumption:",
    hourly_consumption.min()
)


# Day Of Week Consumption Analysis

print("\nCalculating day-of-week consumption...")

day_of_week_consumption = (
    df.groupby(
        ["DayOfWeek", "DayName"]
    )["Global_active_power"]
    .mean()
    .sort_index()
)

print("\nDay-of-Week Average Global Active Power:")

print(day_of_week_consumption)


print("\nHighest Consumption Day:")

print(
    day_of_week_consumption.idxmax()
)


print("\nLowest Consumption Day:")

print(
    day_of_week_consumption.idxmin()
)


# Daily Consumption Analysis

print("\nCalculating daily consumption...")

daily_consumption = (
    df.set_index("Datetime")
    ["Global_active_power"]
    .resample("D")
    .sum()
)

print("\nFirst 10 Daily Consumption Values:")

print(
    daily_consumption.head(10)
)


print("\nAverage Daily Consumption:")

print(
    daily_consumption.mean()
)


print("\nMaximum Daily Consumption:")

print(
    daily_consumption.max()
)


print("\nMinimum Daily Consumption:")

print(
    daily_consumption.min()
)


print("\nDate with Maximum Consumption:")

print(
    daily_consumption.idxmax()
)


print("\nDate with Minimum Consumption:")

print(
    daily_consumption.idxmin()
)


# Weekly Consumption Analysis

print("\nCalculating weekly consumption...")

weekly_consumption = (
    df.set_index("Datetime")
    ["Global_active_power"]
    .resample("W")
    .sum()
)

print("\nFirst 10 Weekly Consumption Values:")

print(
    weekly_consumption.head(10)
)


print("\nAverage Weekly Consumption:")

print(
    weekly_consumption.mean()
)


print("\nMaximum Weekly Consumption:")

print(
    weekly_consumption.max()
)


print("\nMinimum Weekly Consumption:")

print(
    weekly_consumption.min()
)


# Monthly Consumption Analysis

print("\nCalculating monthly consumption...")

monthly_consumption = (
    df.set_index("Datetime")
    ["Global_active_power"]
    .resample("ME")
    .sum()
)

print("\nMonthly Consumption:")

print(monthly_consumption)


print("\nAverage Monthly Consumption:")

print(
    monthly_consumption.mean()
)


print("\nMaximum Monthly Consumption:")

print(
    monthly_consumption.max()
)


print("\nMinimum Monthly Consumption:")

print(
    monthly_consumption.min()
)


# Yearly Consumption Analysis

print("\nCalculating yearly consumption...")

yearly_consumption = (
    df.set_index("Datetime")
    ["Global_active_power"]
    .resample("YE")
    .sum()
)

print("\nYearly Consumption:")

print(yearly_consumption)


print("\nAverage Yearly Consumption:")

print(
    yearly_consumption.mean()
)


# Monthly Average Pattern

print("\nCalculating monthly average consumption...")

monthly_average_consumption = (
    df.groupby(
        ["Month", "MonthName"]
    )["Global_active_power"]
    .mean()
    .sort_index()
)

print("\nMonthly Average Global Active Power:")

print(monthly_average_consumption)


print("\nHighest Consumption Month:")

print(
    monthly_average_consumption.idxmax()
)


print("\nLowest Consumption Month:")

print(
    monthly_average_consumption.idxmin()
)


# Yearly Average Pattern

print("\nCalculating yearly average consumption...")

yearly_average_consumption = (
    df.groupby("Year")["Global_active_power"]
    .mean()
)

print("\nYearly Average Global Active Power:")

print(yearly_average_consumption)


# Hourly Consumption Plot

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

plt.show()


# Day Of Week Consumption Plot

print("\nPlotting day-of-week consumption pattern...")

day_plot = (
    df.groupby("DayOfWeek")["Global_active_power"]
    .mean()
)

plt.figure(figsize=(10, 6))

plt.plot(
    day_plot.index,
    day_plot.values,
    marker="o"
)

plt.xlabel("Day of Week")

plt.ylabel("Average Global Active Power")

plt.title(
    "Average Energy Consumption by Day of Week"
)

plt.xticks(range(7))

plt.grid()

plt.show()


# Monthly Average Consumption Plot

print("\nPlotting monthly average consumption pattern...")

month_plot = (
    df.groupby("Month")["Global_active_power"]
    .mean()
)

plt.figure(figsize=(10, 6))

plt.plot(
    month_plot.index,
    month_plot.values,
    marker="o"
)

plt.xlabel("Month")

plt.ylabel("Average Global Active Power")

plt.title(
    "Average Energy Consumption by Month"
)

plt.xticks(range(1, 13))

plt.grid()

plt.show()


# Daily Consumption Trend

print("\nPlotting daily consumption trend...")

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

plt.show()


# Weekly Consumption Trend

print("\nPlotting weekly consumption trend...")

plt.figure(figsize=(14, 6))

plt.plot(
    weekly_consumption.index,
    weekly_consumption.values
)

plt.xlabel("Date")

plt.ylabel("Weekly Global Active Power")

plt.title(
    "Weekly Energy Consumption Trend"
)

plt.grid()

plt.show()


# Monthly Consumption Trend

print("\nPlotting monthly consumption trend...")

plt.figure(figsize=(14, 6))

plt.plot(
    monthly_consumption.index,
    monthly_consumption.values,
    marker="o"
)

plt.xlabel("Date")

plt.ylabel("Monthly Global Active Power")

plt.title(
    "Monthly Energy Consumption Trend"
)

plt.grid()

plt.show()


# Yearly Consumption Trend

print("\nPlotting yearly consumption trend...")

plt.figure(figsize=(10, 6))

plt.plot(
    yearly_consumption.index,
    yearly_consumption.values,
    marker="o"
)

plt.xlabel("Year")

plt.ylabel("Yearly Global Active Power")

plt.title(
    "Yearly Energy Consumption Trend"
)

plt.grid()

plt.show()


# Hourly Consumption by Day of Week

print("\nCalculating hourly consumption by day of week...")

hour_day_consumption = (
    df.groupby(
        ["DayOfWeek", "Hour"]
    )["Global_active_power"]
    .mean()
)

print("\nHourly Consumption by Day of Week:")

print(
    hour_day_consumption.head(20)
)


# Hourly Pattern by Year

print("\nCalculating yearly hourly consumption...")

year_hour_consumption = (
    df.groupby(
        ["Year", "Hour"]
    )["Global_active_power"]
    .mean()
)

print("\nYearly Hourly Consumption:")

print(
    year_hour_consumption.head(20)
)


# Temporal Analysis Summary

print("\nTemporal analysis completed successfully.")

print("\nDataset Date Range:")

print(
    df["Datetime"].min(),
    "to",
    df["Datetime"].max()
)

print("\nPeak Consumption Hour:")

print(
    hourly_consumption.idxmax()
)

print("\nLowest Consumption Hour:")

print(
    hourly_consumption.idxmin()
)

print("\nHighest Consumption Day:")

print(
    day_of_week_consumption.idxmax()
)

print("\nLowest Consumption Day:")

print(
    day_of_week_consumption.idxmin()
)

print("\nHighest Consumption Month:")

print(
    monthly_average_consumption.idxmax()
)

print("\nLowest Consumption Month:")

print(
    monthly_average_consumption.idxmin()
)

print("\nTemporal analysis completed successfully.")