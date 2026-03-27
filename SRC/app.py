from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel
import sqlite3

# Import functions from your allocation engine
from task_allocation_engine import run_allocation, get_top_3_recommendations

# --- 1. DATABASE INITIALIZATION ---
def init_db():
    conn = sqlite3.connect("workforce.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS predictions 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                      employee_id INTEGER, 
                      risk_level TEXT, 
                      probability REAL, 
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

app = FastAPI(title="AI_Workforce_Management_System")

# --- 2. LOAD AI ASSETS ---
# Loading specific @ version files from your project directory
model = joblib.load("Models/AI_workforce_management@.pkl") 
scaler = joblib.load("Models/scaler.pkl")

# --- 3. DATA MODELS ---
class EmployeeData(BaseModel):
    employee_id: int
    Age: int
    Department: str
    JobRole: str
    OverTime: str 
    JobLevel: int
    PerformanceRating: int
    YearsWithCurrManager: int
    MonthlyIncome: int

class TaskRequest(BaseModel):
    description: str
    department: str

# --- 4. API ROUTES ---

@app.get("/")
def home():
    return {"message": "AI Workforce Management API is Active"}

@app.post("/predict")
def predict(data: EmployeeData):
    try:
        defaults = {
            'BusinessTravel': 'Travel_Rarely', 'DailyRate': 800, 'DistanceFromHome': 5,
            'Education': 3, 'EducationField': 'Life Sciences', 'EnvironmentSatisfaction': 3,
            'Gender': 'Male', 'JobInvolvement': 3, 'JobSatisfaction': 3,
            'MaritalStatus': 'Married', 'MonthlyIncome': 5000, 'NumCompaniesWorked': 1,
            'PercentSalaryHike': 11, 'RelationshipSatisfaction': 3, 'StockOptionLevel:': 0,
            'TotalWorkingYears': 10, 'TrainingTimesLastYear': 3, 'WorkLifeBalance': 3,
            'YearsAtCompany': 5, 'YearsInCurrentRole': 2, 'YearsSinceLastPromotion': 1,
            'YearsWithCurrManager': 2
        }
        # 1. Convert to DataFrame and extract ID
        input_dict = data.dict()
        employee_id = input_dict.pop('employee_id', 0)
        full_data = {**defaults, **input_dict}
        df = pd.DataFrame([full_data])

        # 2. Preprocessing & Feature Engineering
        df['OverTime'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
        df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
        
        # Creating buckets to match training data structure
        df['TenureBucket'] = pd.cut(df['YearsAtCompany'], bins=[-1, 1, 3, 5, 10, 40], labels=['<1', '1-3', '3-5', '5-10', '10+'])
        df['MonthlyGroup'] = pd.cut(df['MonthlyIncome'], bins=[0, 3000, 6000, 9000, 12000, 20000], labels=['0-3k', '3k-6k', '6k-9k', '9k-12k', '12k-20k'])
        df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 60], labels=['18-25', '26-35', '36-45', '46+'])

        df.drop(['YearsAtCompany', 'MonthlyIncome', 'Age'], axis=1, inplace=True, errors='ignore')

        # 3. One-Hot Encoding
        categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus', 'TenureBucket', 'MonthlyGroup', 'AgeGroup']
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # 4. Align Features with Scaler (Ensures exact 53+ features)
        model_features = scaler.feature_names_in_
        final_df = pd.DataFrame(0, index=[0], columns=model_features)
        
        for col in df_encoded.columns:
            if col in final_df.columns:
                final_df[col] = df_encoded[col]

        # 5. Prediction Logic
        X_scaled = scaler.transform(final_df)
        probability = model.predict_proba(X_scaled)[0][1]
        risk_level = "High" if probability > 0.5 else "Low"

        # 6. Database Logging
        conn = sqlite3.connect("workforce.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (employee_id, risk_level, probability) VALUES (?, ?, ?)", 
            (int(employee_id), str(risk_level), float(probability))
        )
        conn.commit()
        conn.close()

        return {
            "employee_id": employee_id,
            "attrition_risk": risk_level,
            "probability_score": round(float(probability), 4),
            "status": "Success"
        }
    except Exception as e:
        return {"status": "Error", "message": str(e)}

@app.post("/allocate-tasks")
def trigger_bulk_allocation():
    try:
        results = run_allocation() 
        return {
            "status": "Success",
            "tasks_assigned": len(results),
            "message": "Bulk allocation complete. Results saved to allocation_results.csv"
        }
    except Exception as e:
        return {"status": "Error", "message": str(e)}

@app.post("/recommend-employees")
def recommend_employees(request: TaskRequest):
    try:
        # Calls the function from task_allocation_engine.py
        recommendations = get_top_3_recommendations(request.description, request.department)
        
        if recommendations.empty:
            return {"status": "Error", "message": f"No candidates found for {request.department}"}
            
        results = recommendations.to_dict(orient="records")
        return {
            "status": "Success",
            "top_candidates": results
        }
    except Exception as e:
        return {"status": "Error", "message": str(e)}