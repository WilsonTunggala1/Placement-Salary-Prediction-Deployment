import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from lightgbm import LGBMRegressor

def train_reg(x_train, y_train):
    # Ensure artifact directory exists
    os.makedirs("artifacts", exist_ok=True)

    # Features to scale or encode (Exact same as Classification)
    onehot_features = ['gender', 'branch', 'part_time_job', 'internet_access']
    numeric_features = [
        'cgpa', 'backlogs', 'study_hours_per_day', 'attendance_percentage', 
        'projects_completed', 'internships_completed', 'coding_skill_rating', 
        'communication_skill_rating', 'aptitude_skill_rating', 
        'hackathons_participated', 'certifications_count', 'sleep_hours', 'stress_level'
    ]
    ordinal_features = ['family_income_level', 'city_tier']
    extra_feature = ['extracurricular_involvement']

    onehot_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    robust_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=[
            ['Low', 'Medium', 'High'],      # For family_income_level
            ['Tier 3', 'Tier 2', 'Tier 1']  # For city_tier
        ], handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    extra_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('encoder', OrdinalEncoder(categories=[
            ['None', 'Low', 'Medium', 'High'] 
        ], handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ('onehot', onehot_pipeline, onehot_features),
        ('ordinal', ordinal_pipeline, ordinal_features),
        ('extra', extra_pipeline, extra_feature),
        ('num', robust_pipeline, numeric_features)
    ], remainder='drop')

    # The Regression Pipeline
    reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), 
        ('regressor', LGBMRegressor(random_state=42, verbose=-1))
    ])

    # MLflow Tracking - Separate Experiment for Regression
    mlflow.set_experiment("Salary_Regression_Pipeline")

    with mlflow.start_run() as run:
        # Log model parameters
        mlflow.log_param("model_type", "LightGBM Regressor")
        mlflow.log_param("random_state", 42)
       
        # Train the model
        print("Training Regression Pipeline...")
        reg_pipeline.fit(x_train, y_train)

        # Save model locally with a distinct name
        joblib.dump(reg_pipeline, "artifacts/placement_reg_pipeline.pkl")
        
        # Log model artifact to MLflow
        mlflow.sklearn.log_model(reg_pipeline, artifact_path="model")
        
        print(f"Regression Training Complete. MLflow Run ID: {run.info.run_id}")
        
        return run.info.run_id

if __name__ == "__main__":
    train_reg()