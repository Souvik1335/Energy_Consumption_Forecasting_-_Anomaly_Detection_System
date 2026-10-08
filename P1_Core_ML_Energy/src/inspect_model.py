import os
import joblib

MODEL_PATH = r"P1_Core_ML_Energy\models\forecasting_model.pkl"

print("Loading model...")
model = joblib.load(MODEL_PATH)

print("\n========== MODEL INFO ==========")
print("Model type:", type(model))
print(
    "Model size:",
    round(os.path.getsize(MODEL_PATH) / (1024 * 1024), 2),
    "MB"
)

if hasattr(model, "n_estimators"):
    print("Number of trees:", model.n_estimators)

if hasattr(model, "max_depth"):
    print("Maximum depth:", model.max_depth)

if hasattr(model, "n_features_in_"):
    print("Number of features:", model.n_features_in_)

if hasattr(model, "min_samples_split"):
    print("Min samples split:", model.min_samples_split)

if hasattr(model, "min_samples_leaf"):
    print("Min samples leaf:", model.min_samples_leaf)

print("\nModel inspection completed successfully.")