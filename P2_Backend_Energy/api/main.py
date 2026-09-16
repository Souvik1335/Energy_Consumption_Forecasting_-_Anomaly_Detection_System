from pathlib import Path
import sys
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel


# Add Source Directory

SRC_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
)

sys.path.append(
    str(SRC_DIR)
)


from src.prediction import generate_prediction


# Create FastAPI Application

app = FastAPI(
    title="Energy Consumption Forecasting API",
    description="API for energy consumption forecasting and anomaly detection",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Define Prediction Request

class PredictionRequest(BaseModel):

    Hour: int
    Day: int
    DayOfWeek: int
    Month: int
    Year: int

    IsWeekend: int
    IsPeakHour: int

    Lag_1: float
    Lag_2: float
    Lag_3: float
    Lag_24: float
    Lag_48: float
    Lag_60: float
    Lag_1440: float
    Lag_10080: float

    Rolling_Mean_15: float
    Rolling_Mean_60: float
    Rolling_Mean_1440: float

    Rolling_Std_15: float
    Rolling_Std_60: float
    Rolling_Std_1440: float

    Global_active_power: float


# API Health

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Energy Backend"
    }

# Root Endpoint

@app.get("/")
def root():

    return {
        "message": "Energy Consumption Forecasting API is running"
    }


# Prediction Endpoint

@app.post("/predict")
def predict(request: PredictionRequest):

    input_data = request.model_dump()

    actual_value = input_data.pop(
        "Global_active_power"
    )

    result = generate_prediction(
        input_data,
        actual_value
    )

    return result