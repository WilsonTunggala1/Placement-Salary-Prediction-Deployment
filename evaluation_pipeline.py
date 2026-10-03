import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

def evaluate_clf(x_test, y_test, run_id):
    """
    Evaluates the classification model and logs metrics to the existing MLflow run.
    """
    print(f"Loading Classification model from MLflow (Run ID: {run_id})...")
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    # Make predictions
    preds = model.predict(x_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)

    # FIX: Set the exact same experiment name used in train_clf_pipeline.py
    mlflow.set_experiment("Placement_Classification_Pipeline")

    # Re-open the same MLflow run to log the evaluation metrics
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision_macro", prec)
        mlflow.log_metric("test_recall_macro", rec)
        mlflow.log_metric("test_f1_macro", f1)

    print(f"Classification Evaluation: Accuracy={acc:.4f} | F1-Score (Macro)={f1:.4f}")

    return acc, f1


def evaluate_reg(x_test, y_test, run_id):
    """
    Evaluates the regression model and logs metrics to the existing MLflow run.
    """
    print(f"Loading Regression model from MLflow (Run ID: {run_id})...")
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    # Make predictions
    preds = model.predict(x_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # FIX: Set the exact same experiment name used in train_reg_pipeline.py
    mlflow.set_experiment("Salary_Regression_Pipeline")

    # Re-open the same MLflow run to log the evaluation metrics
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)

    print(f"Regression Evaluation: MAE={mae:.4f} LPA | RMSE={rmse:.4f} LPA | R2={r2:.4f}")

    return mae, r2