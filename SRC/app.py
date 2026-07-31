import joblib
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel, Field,ValidationError
from typing import Annotated, Literal
from .task_allocation_engine import task_allocation,top_recommendation

# Load model
model = joblib.load("Models/attrition_model.pkl")
scaler = joblib.load("Models/scaler.pkl")
feature_columns = joblib.load("Models/feature_columns.pkl")

app = FastAPI(title="Attrition Prediction Model")

# Pydantic model for data validation (incoming data validation)

# Restrict Department to an allowed list.
# Replace the values below with the exact department names used in your training data.
AllowedDepartment = Literal[ "HR","Research & Development","Sales",]
overtime = Literal["Yes","No"]
BusinessTravel = Literal["Non-Travel","Travel_Frequently","Travel_Rarely"]
Gender = Literal["Male","Female"]
MaritalStatus = Literal["Single","Married","Divorced"]
stock_option_level = Literal[0,1,2,3]
class UserInput(BaseModel):
    EmployeeNumber: Annotated[int, Field(..., description="Employee ID")]
    Age: Annotated[int, Field(..., ge=18, le=60, description="Age")]
    Department: Annotated[AllowedDepartment,Field(..., description="Department (allowed values only)"),]
    JobRole: Annotated[str,Field(..., description="Job Role")]
    OverTime: Annotated[overtime, Field(..., description="Overtime")]
    JobLevel: Annotated[int,Field(...,ge=1,le=5, description="Job Level")]
    PerformanceRating: Annotated[int, Field(...,ge=1,le=4,  description="Performance Rating")]
    MonthlyIncome: Annotated[float, Field(...,ge=2000, le=20000, description="Monthly Income")]
    BusinessTravel: Annotated[BusinessTravel, Field(..., description="Business Travel")]
    DailyRate: Annotated[int, Field(..., ge=102,le=1500, description="Daily Rate")]
    DistanceFromHome: Annotated[int, Field(..., description="Distance from Home")]
    Education: Annotated[int, Field(...,ge=1,le=5, description="Education")]
    EducationField: Annotated[str, Field(..., description="Education Field")]
    EnvironmentSatisfaction: Annotated[int, Field(...,ge=1,le=4,description="Environment Satisfaction")]
    Gender: Annotated[Gender, Field(..., description="Gender")]
    JobInvolvement: Annotated[int, Field(...,ge=1,le=4, description="Job Involvement")]
    JobSatisfaction: Annotated[int, Field(..., ge=1,le=4, description="Job Satisfaction")]
    MaritalStatus: Annotated[MaritalStatus, Field(..., description="Marital Status")]
    NumCompaniesWorked: Annotated[int, Field(...,ge=-1,le=10, description="Number of Companies Worked")]
    PercentSalaryHike: Annotated[int, Field(...,ge=10,le=26, description="Percent Salary Hike")]
    RelationshipSatisfaction: Annotated[int, Field(...,ge=1,le=4, description="Relationship Satisfaction")]
    StockOptionLevel: Annotated[stock_option_level, Field(..., description="Stock Option Level")]
    TotalWorkingYears: Annotated[int, Field(...,ge=-1,le=41, description="Total Working Years")]
    TrainingTimesLastYear: Annotated[int, Field(...,ge=0,le=6, description="Training Times Last Year")]
    WorkLifeBalance: Annotated[int, Field(..., ge=1,le=4,description="Work-Life Balance")]
    YearsAtCompany: Annotated[int, Field(...,ge=-1,le=40, description="Years at Company")]
    YearsInCurrentRole: Annotated[int, Field(...,ge=-1,le=20, description="Years in Current Role")]
    YearsSinceLastPromotion: Annotated[int, Field(...,ge=-1,le=15, description="Years Since Last Promotion")]
    YearsWithCurrManager: Annotated[int, Field(..., ge=0,le=17,description="Years with Current Manager")] 
    skills: Annotated[str, Field(...,description="skills of Employee")]

Task_complex=Literal["Low","Medium","High"]
Task_dura=Literal['Long', 'Medium', 'Short']
criticality=Literal["Critical","Non-critical"]
class TaskUserInput(BaseModel):
    task_id:Annotated[str, Field(..., description="T_1001")]
    task_name:Annotated[str, Field(..., description="Task Name")]
    department:Annotated[AllowedDepartment,Field(..., description="Department")]
    target_role: Annotated[str, Field(..., description="target_role")]
    required_skills: Annotated[str, Field(..., description="required skills")]
    task_complexity: Annotated[Task_complex, Field(..., description="Low,Medium,High")]
    estimated_hours:Annotated[int, Field(..., gt=2, lt=101, description="1")]
    task_priority: Annotated[Task_complex, Field(...,description="low,Medium,High")]
    task_duration:Annotated[Task_dura,Field(..., description="['Long', 'Midterm', 'Short']")]
    criticality:Annotated[criticality,Field(..., description="[Critical,Non-critical]")]
    
class AllocationRequest(BaseModel):
    employee_df: list[UserInput]
    task_df: list[TaskUserInput]

@app.get("/")
def home():
    return {"Message": " Attrition prediction Model is working correctly"}


@app.post("/predict")
def predict(Data: UserInput):

    # Convert Pydantic model to dictionary
    input_data = Data.model_dump()
    EmployeeNumber = input_data.pop("EmployeeNumber")

    # Create DataFrame
    df = pd.DataFrame([input_data])

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    df["OverTime"] = df["OverTime"].apply(lambda x: 1 if x == "Yes" else 0)
    df["Gender"] = df["Gender"].apply(lambda x: 1 if x == "Male" else 0)

    # Create Buckets
    df["TenureBucket"] = pd.cut(
        df["YearsAtCompany"],
        bins=[-1, 1, 3, 5, 10, 40],
        labels=["<1", "1-3", "3-5", "5-10", "10+"]
    )

    df["MonthlyIncomeBucket"] = pd.cut(
        df["MonthlyIncome"],
        bins=[-1, 3000, 5000, 10000, 15000, 20000],
        labels=["<3k", "3k-5k", "5k-10k", "10k-15k", "15k+"]
    )

    df["AgeBucket"] = pd.cut(
        df["Age"],
        bins=[-1, 25, 35, 45, 55, 65],
        labels=["<25", "25-35", "35-45", "45-55", "55+"]
    )

    # One Hot Encoding
    categorical_features = [
        "Department",
        "TenureBucket",
        "MonthlyIncomeBucket",
        "AgeBucket",
        "JobRole",
        "BusinessTravel",
        "EducationField",
        "MaritalStatus"
    ]

    df_encoded = pd.get_dummies(
        df,
        columns=categorical_features,
        drop_first=True
    )

    # Drop original columns
    df_encoded.drop(
        ["YearsAtCompany", "MonthlyIncome", "Age"],
        axis=1,
        inplace=True,
        errors="ignore"
    )

    # -----------------------------
    # Align Features
    # -----------------------------
    final_df = pd.DataFrame(0, index=[0], columns=feature_columns)

    for col in df_encoded.columns:
        if col in final_df.columns:
            final_df[col] = df_encoded[col]

    final_df = final_df.astype(float)

    # -----------------------------
    # Debug Prints
    # -----------------------------
    print("\n========== INPUT ==========")
    print(df)

    print("\n========== ENCODED ==========")
    print(df_encoded)

    print("\n========== FEATURE COLUMNS ==========")
    print(feature_columns)

    print("\n========== FINAL DF ==========")
    print(final_df)

    # Scale
    scaled_data = scaler.transform(final_df)

    print("\n========== SCALED DATA ==========")
    print(scaled_data)

    # Prediction
    prediction = model.predict_proba(scaled_data)[0][1]

    print("\nPrediction:", prediction)

    return {
        "EmployeeNumber": EmployeeNumber,
        "prediction": round(float(prediction), 4),
        "attrition": "Yes" if prediction >= 0.5 else "No"
    }
    
@app.post("/task_allocation")
def allocation(data: AllocationRequest):
    data.employee_df=pd.DataFrame(
        [employee.model_dump() for employee in data.employee_df]
    )

    data.task_df = pd.DataFrame(
        [task.model_dump() for task in data.task_df])
    try:
        result= task_allocation(data.employee_df,data.task_df)
        return{
            "Status":"Success",
            "Message": "Task allocated to best fitted employee",
            "Data": result
        }
    except Exception as e:
        return {"status": "Error", "message": str(e)}
    
@app.post("/Recommed Employee")
def recommendation(emp_df: list[UserInput],desc: str,dpt: AllowedDepartment):
    try:
        result = top_recommendation(emp_df,dpt,desc)
        return {
            "status": "Success",
            "message": "Best employee according to dataset",
            "data": result
}
    except Exception as e:
        return {"status": "Error", "message": str(e)}
    
   