import os
import time
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VALIDATION_FILE = os.path.join(DATA_DIR, "validation.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

OUTPUT_MODEL = os.path.join(
    MODEL_DIR,
    "forecasting_model_deployment.pkl"
)

# FEATURES
FEATURES = [
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
    "Rolling_Std_1440",
]

TARGET = "Global_active_power"

# LOAD DATA

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

start_time = time.time()

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)
test = pd.read_csv(TEST_FILE)

print(f"Train shape:      {train.shape}")
print(f"Validation shape: {validation.shape}")
print(f"Test shape:       {test.shape}")

print(f"\nData loading time: {time.time() - start_time:.2f} seconds")

# PREPARE DATA

print("\n" + "=" * 60)
print("PREPARING DATA")
print("=" * 60)

X_train = train[FEATURES]
y_train = train[TARGET]

X_validation = validation[FEATURES]
y_validation = validation[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

print(f"Number of features: {len(FEATURES)}")
print(f"Training samples:   {len(X_train)}")
print(f"Validation samples: {len(X_validation)}")
print(f"Test samples:       {len(X_test)}")

# TRAIN SMALLER RANDOM FOREST
print("\n" + "=" * 60)
print("TRAINING DEPLOYMENT RANDOM FOREST")
print("=" * 60)

model = RandomForestRegressor(
    n_estimators=75,
    max_depth=20,
    min_samples_split=7,
    min_samples_leaf=5,
    max_features="log2",
    random_state=42,
    n_jobs=-1
)

print("\nModel configuration:")
print(f"n_estimators      : {model.n_estimators}")
print(f"max_depth         : {model.max_depth}")
print(f"min_samples_split : {model.min_samples_split}")
print(f"min_samples_leaf  : {model.min_samples_leaf}")
print(f"max_features      : {model.max_features}")

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"\nTraining completed in: {training_time / 60:.2f} minutes")

# VALIDATION
print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

validation_predictions = model.predict(X_validation)

validation_mae = mean_absolute_error(
    y_validation,
    validation_predictions
)

validation_mse = mean_squared_error(
    y_validation,
    validation_predictions
)

validation_rmse = validation_mse ** 0.5

validation_r2 = r2_score(
    y_validation,
    validation_predictions
)

print(f"MAE  : {validation_mae:.6f}")
print(f"MSE  : {validation_mse:.6f}")
print(f"RMSE : {validation_rmse:.6f}")
print(f"R²   : {validation_r2:.6f}")

# TEST
print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

test_predictions = model.predict(X_test)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_mse = mean_squared_error(
    y_test,
    test_predictions
)

test_rmse = test_mse ** 0.5

test_r2 = r2_score(
    y_test,
    test_predictions
)

print(f"MAE  : {test_mae:.6f}")
print(f"MSE  : {test_mse:.6f}")
print(f"RMSE : {test_rmse:.6f}")
print(f"R²   : {test_r2:.6f}")

# SAVE MODEL
print("\n" + "=" * 60)
print("SAVING DEPLOYMENT MODEL")
print("=" * 60)

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    model,
    OUTPUT_MODEL,
    compress=3
)

model_size_mb = os.path.getsize(OUTPUT_MODEL) / (1024 * 1024)

print(f"\nModel saved to:")
print(OUTPUT_MODEL)

print(f"\nModel size: {model_size_mb:.2f} MB")

# SUMMARY
print("\n" + "=" * 60)
print("DEPLOYMENT MODEL SUMMARY")
print("=" * 60)

print(f"Model              : RandomForestRegressor")
print(f"Trees              : {model.n_estimators}")
print(f"Max depth          : {model.max_depth}")
print(f"Features           : {len(FEATURES)}")
print(f"Validation R²      : {validation_r2:.6f}")
print(f"Test R²            : {test_r2:.6f}")
print(f"Test MAE           : {test_mae:.6f}")
print(f"Test MSE           : {test_mse:.6f}")
print(f"Test RMSE          : {test_rmse:.6f}")
print(f"Model size         : {model_size_mb:.2f} MB")

print("\nTraining completed successfully.")