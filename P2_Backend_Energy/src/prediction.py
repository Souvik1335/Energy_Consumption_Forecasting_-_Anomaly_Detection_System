import pandas as pd
from model_loader import (forecasting_model, feature_columns, anomaly_threshold)


# Generate Prediction

def generate_prediction(input_data, actual_value):
    input_df = pd.DataFrame([input_data], columns=feature_columns)

    # Generate Forecast

    prediction = forecasting_model.predict(input_df)[0]

    # Calsulate Residural

    residual = actual_value - prediction

    # Absolute Residural

    absolute_residual = abs(residual)

    # Detect Anomaly

    if absolute_residual > anomaly_threshold :
        anomaly = 1
        anomaly_status = 'Anomaly'
    else :
        anomaly = 0
        anomaly_status = 'Normal'


    # Return Prediction Result

    return {
        "actual_global_active_power": actual_value,
        "predicted_global_active_power": prediction,
        "residual": residual,
        "absolute_residual": absolute_residual,
        "anomaly": anomaly,
        "anomaly_status": anomaly_status
    }