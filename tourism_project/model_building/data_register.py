
import pandas as pd
import os
import sys

# ---------------------------------------------------
# Define dataset path
# ---------------------------------------------------
DATA_PATH = "tourism_project/data/tourism.csv"

# ---------------------------------------------------
# Expected columns in the Tourism dataset
# ---------------------------------------------------
EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "ProductPitched",
    "NumberOfFollowups",
    "DurationOfPitch"
]

# ---------------------------------------------------
# Check whether dataset exists
# ---------------------------------------------------
if not os.path.exists(DATA_PATH):
    print(f"ERROR: Dataset not found at {DATA_PATH}")
    sys.exit(1)

# ---------------------------------------------------
# Load dataset
# ---------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")

# ---------------------------------------------------
# Validate expected columns
# ---------------------------------------------------
missing_columns = [
    col for col in EXPECTED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    print("Dataset validation FAILED.")
    print("Missing columns:", missing_columns)
    sys.exit(1)

print("Dataset validation PASSED.")
print("All expected columns are present.")

# ---------------------------------------------------
# Dataset summary
# ---------------------------------------------------
print("\n----- DATASET SUMMARY -----")
print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution (ProdTaken):")
print(df["ProdTaken"].value_counts())

print("\nDataset registration completed successfully.")
