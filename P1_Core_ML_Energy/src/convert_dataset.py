from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "raw" / "household_power_consumption.txt"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "household_power_consumption.csv"


print("Loading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    sep=";",
    low_memory=False
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Conversion completed!")
print("Shape:", df.shape)
print("CSV saved to:", OUTPUT_FILE)