
# ============================================================
# Tourism Package Prediction - Model Training
# ============================================================

# Data manipulation
import pandas as pd
import os
import joblib

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Model training and tuning
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Model evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Experiment tracking
import mlflow


# ============================================================
# 1. MLflow Configuration
# ============================================================

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-prediction-experiment")


# ============================================================
# 2. Define Data Paths
# ============================================================

DATA_DIR = "tourism_project/model_building"

Xtrain_path = os.path.join(DATA_DIR, "Xtrain.csv")
Xtest_path = os.path.join(DATA_DIR, "Xtest.csv")
ytrain_path = os.path.join(DATA_DIR, "ytrain.csv")
ytest_path = os.path.join(DATA_DIR, "ytest.csv")


# ============================================================
# 3. Load Train and Test Data
# ============================================================

print("Loading training and testing data...")

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Convert target DataFrames to 1-D Series
ytrain = ytrain.squeeze()
ytest = ytest.squeeze()

print("Data loaded successfully.")

print("Xtrain shape:", Xtrain.shape)
print("Xtest shape :", Xtest.shape)
print("ytrain shape:", ytrain.shape)
print("ytest shape :", ytest.shape)


# ============================================================
# 4. Define Numerical and Categorical Features
# ============================================================

numeric_features = [
    "Age",
    "CityTier",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch"
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched"
]

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 5. Create Preprocessing Pipeline
# ============================================================

preprocessor = make_column_transformer(
    (
        StandardScaler(),
        numeric_features
    ),
    (
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    )
)


# ============================================================
# 6. Define XGBoost Classification Model
# ============================================================

xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 7. Create Complete ML Pipeline
# ============================================================

model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)


# ============================================================
# 8. Define Hyperparameter Grid
# ============================================================

param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.8, 1.0]
}


# ============================================================
# 9. Start MLflow Experiment
# ============================================================

with mlflow.start_run(run_name="XGBoost_Tourism_Classification"):

    print("\nStarting GridSearchCV...")

    # --------------------------------------------------------
    # Hyperparameter tuning
    # --------------------------------------------------------

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(Xtrain, ytrain)

    print("\nGrid Search completed successfully.")


    # ========================================================
    # 10. Log Tuned Parameter Sets
    # ========================================================

    results = grid_search.cv_results_

    for i in range(len(results["params"])):

        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]

        with mlflow.start_run(
            run_name=f"parameter_set_{i+1}",
            nested=True
        ):

            mlflow.log_params(param_set)

            mlflow.log_metric(
                "mean_cv_f1_score",
                float(mean_score)
            )


    # ========================================================
    # 11. Get Best Model
    # ========================================================

    best_model = grid_search.best_estimator_

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print(
        "\nBest Cross Validation F1 Score:",
        grid_search.best_score_
    )

    # Log best parameters
    mlflow.log_params(grid_search.best_params_)

    mlflow.log_metric(
        "best_cv_f1_score",
        float(grid_search.best_score_)
    )


    # ========================================================
    # 12. Make Predictions
    # ========================================================

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    y_prob_train = best_model.predict_proba(Xtrain)[:, 1]
    y_prob_test = best_model.predict_proba(Xtest)[:, 1]


    # ========================================================
    # 13. Calculate Training Metrics
    # ========================================================

    train_accuracy = accuracy_score(
        ytrain,
        y_pred_train
    )

    train_precision = precision_score(
        ytrain,
        y_pred_train,
        zero_division=0
    )

    train_recall = recall_score(
        ytrain,
        y_pred_train,
        zero_division=0
    )

    train_f1 = f1_score(
        ytrain,
        y_pred_train,
        zero_division=0
    )

    train_roc_auc = roc_auc_score(
        ytrain,
        y_prob_train
    )


    # ========================================================
    # 14. Calculate Test Metrics
    # ========================================================

    test_accuracy = accuracy_score(
        ytest,
        y_pred_test
    )

    test_precision = precision_score(
        ytest,
        y_pred_test,
        zero_division=0
    )

    test_recall = recall_score(
        ytest,
        y_pred_test,
        zero_division=0
    )

    test_f1 = f1_score(
        ytest,
        y_pred_test,
        zero_division=0
    )

    test_roc_auc = roc_auc_score(
        ytest,
        y_prob_test
    )


    # ========================================================
    # 15. Print Model Performance
    # ========================================================

    print("\n========================================")
    print("TRAINING PERFORMANCE")
    print("========================================")

    print(f"Accuracy : {train_accuracy:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall   : {train_recall:.4f}")
    print(f"F1 Score : {train_f1:.4f}")
    print(f"ROC-AUC  : {train_roc_auc:.4f}")


    print("\n========================================")
    print("TEST PERFORMANCE")
    print("========================================")

    print(f"Accuracy : {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall   : {test_recall:.4f}")
    print(f"F1 Score : {test_f1:.4f}")
    print(f"ROC-AUC  : {test_roc_auc:.4f}")


    # ========================================================
    # 16. Classification Report
    # ========================================================

    print("\nClassification Report:")
    print(
        classification_report(
            ytest,
            y_pred_test,
            zero_division=0
        )
    )


    # ========================================================
    # 17. Confusion Matrix
    # ========================================================

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            ytest,
            y_pred_test
        )
    )


    # ========================================================
    # 18. Log Evaluation Metrics to MLflow
    # ========================================================

    mlflow.log_metrics({

        "train_accuracy": train_accuracy,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_f1": train_f1,
        "train_roc_auc": train_roc_auc,

        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc

    })


    # ========================================================
    # 19. Save Best Model
    # ========================================================

    DEPLOYMENT_DIR = "tourism_project/deployment"

    os.makedirs(
        DEPLOYMENT_DIR,
        exist_ok=True
    )

    model_path = os.path.join(
        DEPLOYMENT_DIR,
        "best_model.joblib"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print(
        f"\nBest model saved successfully at: {model_path}"
    )


    # ========================================================
    # 20. Log Model as MLflow Artifact
    # ========================================================

    mlflow.log_artifact(
        model_path,
        artifact_path="model"
    )

    print(
        "Model artifact logged successfully in MLflow."
    )


print("\n========================================")
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("========================================")
