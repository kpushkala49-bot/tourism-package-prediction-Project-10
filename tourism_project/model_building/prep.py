
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(data_path):
    df = pd.read_csv(data_path)

    # Remove unnecessary columns
    df = df.drop(columns=['CustomerID'], errors='ignore')

    # Split data into features (X) and target (y)
    X = df.drop('ProdTaken', axis=1)
    y = df['ProdTaken']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Save the splits locally as CSV files
    X_train.to_csv('Xtrain.csv', index=False)
    X_test.to_csv('Xtest.csv', index=False)
    y_train.to_csv('ytrain.csv', index=False)
    y_test.to_csv('ytest.csv', index=False)

    print("Data preparation complete. Train and test sets saved.")

if __name__ == '__main__':
    # This part will be executed when the script is run directly
    # For the pipeline, we'll assume the data_path is passed or known.
    # For local testing, you might need to adjust this.
    prepare_data('tourism_project/data/tourism.csv') # Assuming data is in this path
