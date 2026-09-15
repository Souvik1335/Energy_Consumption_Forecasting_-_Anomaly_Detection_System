from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit
)

import warnings

warnings.filterwarnings("ignore")


# Project Base Directory

BASE_DIR = Path(__file__).resolve().parents[1]


# Data File Paths

TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.csv"

VALIDATION_FILE = BASE_DIR / "data" / "processed" / "validation.csv"

TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"


# Load Datasets

print("Loading train, validation and test datasets...")

train_df = pd.read_csv(
    TRAIN_FILE,
    low_memory=False
)

validation_df = pd.read_csv(
    VALIDATION_FILE,
    low_memory=False
)

test_df = pd.read_csv(
    TEST_FILE,
    low_memory=False
)

print("Datasets loaded successfully.")


# Define Target

target_column = "Global_active_power"


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

X_train = train_df[feature_columns]

y_train = train_df[target_column]

X_validation = validation_df[feature_columns]

y_validation = validation_df[target_column]

X_test = test_df[feature_columns]

y_test = test_df[target_column]


# First Model Creation & Model Train

model2 = RandomForestRegressor(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

model2.fit(
    X_train,
    y_train
)


# Model Prediction

model2_validation_pred = model2.predict(
    X_validation
)


# Model Metrics Evolution

model2_mse = mean_squared_error(
    y_validation,
    model2_validation_pred
)

print(
    "The model2's (Random Forest Model) Mean Squared Error is :- ",
    model2_mse
)


model2_mae = mean_absolute_error(
    y_validation,
    model2_validation_pred
)

print(
    "The model2's (Random Forest Model) Mean Absolute Error is :- ",
    model2_mae
)


model2_r2_score = r2_score(
    y_validation,
    model2_validation_pred
)

print(
    "The model2's (Random Forest Model) R2 Score is :- ",
    model2_r2_score
)


print("\nTune Our Random Forest Model")


# Tuning the Parameter

param = {
    "n_estimators": [50, 75, 100, 150],
    "min_samples_split": [3, 5, 7],
    "min_samples_leaf": [3, 5, 7],
    "max_depth": [None, 10, 20, 30, 40, 50],
    "max_features": [
        "sqrt",
        "log2",
        0.7,
        1.0
    ]
}


# Define Time Series Cross Validation

time_series_cv = TimeSeriesSplit(
    n_splits=3
)


# Create Tuning Model

tuning_model2 = RandomForestRegressor(
    random_state=42,
    n_jobs=1
)


# Creation of Hyper Model

RandomCV = RandomizedSearchCV(
    estimator=tuning_model2,
    param_distributions=param,
    n_iter=10,
    n_jobs=2,
    scoring="neg_root_mean_squared_error",
    cv=time_series_cv,
    random_state=42,
    verbose=1
)


# Hyperparameter Tuning

print("\nStarting Hyperparameter Tuning...")

RandomCV.fit(
    X_train,
    y_train
)

print("\nHyperparameter Tuning Completed.")


# Get Best Parameters

print("\nBest Hyperparameters:")

print(
    RandomCV.best_params_
)


# Get Best Cross Validation Score

print("\nBest Cross Validation RMSE:")

print(
    -RandomCV.best_score_
)


# Get Best Model

best_model2 = RandomCV.best_estimator_


# Hyper Model Prediction

print("\nGenerating Tuned Random Forest Predictions...")

Hyper_Model2_pred = best_model2.predict(
    X_validation
)


# Model Metrics Evolution

Hyper_Model2_mse = mean_squared_error(
    y_validation,
    Hyper_Model2_pred
)

print(
    "\nThe Hyper Model2's (Tuned Random Forest Model) Mean Squared Error is :- ",
    Hyper_Model2_mse
)


Hyper_Model2_mae = mean_absolute_error(
    y_validation,
    Hyper_Model2_pred
)

print(
    "The Hyper Model2's (Tuned Random Forest Model) Mean Absolute Error is :- ",
    Hyper_Model2_mae
)


Hyper_Model2_r2_score = r2_score(
    y_validation,
    Hyper_Model2_pred
)

print(
    "The Hyper Model2's (Tuned Random Forest Model) R2 Score is :- ",
    Hyper_Model2_r2_score
)

# Final Test Prediction

final_test_pred = best_model2.predict(
    X_test
)


# Final Test Metrics

final_test_mse = mean_squared_error(
    y_test,
    final_test_pred
)

final_test_mae = mean_absolute_error(
    y_test,
    final_test_pred
)

final_test_rmse = final_test_mse ** 0.5

final_test_r2_score = r2_score(
    y_test,
    final_test_pred
)


# Display Final Test Metrics

print("\nFinal Test Metrics")

print(
    "Final Test Mean Squared Error is :- ",
    final_test_mse
)

print(
    "Final Test Mean Absolute Error is :- ",
    final_test_mae
)

print(
    "Final Test Root Mean Squared Error is :- ",
    final_test_rmse
)

print(
    "Final Test R2 Score is :- ",
    final_test_r2_score
)
