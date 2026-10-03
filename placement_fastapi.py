from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the machine learning models from the artifacts folder
clf_pipeline = joblib.load('artifacts/placement_clf_pipeline.pkl')
reg_pipeline = joblib.load('artifacts/placement_reg_pipeline.pkl')

# Define the input schema matching the required pipeline features
class StudentFeatures(BaseModel):
    gender: str
    branch: str
    part_time_job: str
    internet_access: str
    cgpa: float
    backlogs: int
    study_hours_per_day: float
    attendance_percentage: float
    projects_completed: int
    internships_completed: int
    coding_skill_rating: int
    communication_skill_rating: int
    aptitude_skill_rating: int
    hackathons_participated: int
    certifications_count: int
    sleep_hours: float
    stress_level: int
    family_income_level: str
    city_tier: str
    extracurricular_involvement: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Student Placement Prediction API"}

@app.post('/predict')
def predict(student: StudentFeatures):
   
    df = pd.DataFrame([student.dict()])
    
    placement_pred = clf_pipeline.predict(df)[0]
    
    placement_proba = clf_pipeline.predict_proba(df)[0][1]

    if placement_pred == 1:
        salary_pred = reg_pipeline.predict(df)[0]
    else:
        salary_pred = 0.0
    
    # Return the predictions
    return {
        'placement_status': int(placement_pred),
        'placement_probability': float(placement_proba),
        'estimated_salary_lpa': float(salary_pred)
    }