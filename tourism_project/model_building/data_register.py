
import pandas as pd
import os

def register_data(data_path):
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    df = pd.read_csv(data_path)

    # Expected columns based on the problem description and training script
    expected_columns = [
        'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier', 'Occupation', 
        'Gender', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'MaritalStatus', 
        'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting', 
        'Designation', 'MonthlyIncome', 'PitchSatisfactionScore', 'ProductPitched', 
        'NumberOfFollowups', 'DurationOfPitch'
    ]

    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing expected columns in the dataset: {missing_columns}")

    print("Dataset registered successfully!")
    print(f"Shape of the dataset: {df.shape}")
    print("First 5 rows:")
    print(df.head())
    print("Column information:")
    print(df.info())

if __name__ == '__main__':
    register_data('tourism_project/data/tourism.csv')
