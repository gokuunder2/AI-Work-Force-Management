import os
import joblib
import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
import io
from fastapi.middleware.cors import CORSMiddleware
from pandas import DataFrame
from pydantic import BaseModel, Field, ValidationError
from typing import Annotated, Literal, Optional

from .task_allocation_engine import task_allocation, top_recommendation, predict_attrition

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "Models")
DATA_DIR=os.path.join(BASE_DIR,"Datasets")
DATABASE_DIR = os.path.join(BASE_DIR, "Database")
os.makedirs(DATABASE_DIR, exist_ok=True)

# Load model artifacts (paths resolved relative to the repo root, not cwd)
model = joblib.load(os.path.join(MODELS_DIR, "attrition_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))

app = FastAPI(title="AI Workforce Management API")

# Allow the React (Vite) frontend to call this API from the browser.
# NOTE: browsers reject `Access-Control-Allow-Origin: *` combined with
# credentials, so allow_credentials must be False while origins is "*".
# If you switch to explicit origins in production, you can turn credentials
# back on.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your deployed frontend origin in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------
# Data loading: the frontend should never have to send full 25-field
# employee/task records. We load the source datasets once at startup and
# the allocation/recommendation ML endpoints work off that in-memory copy.
# Individually uploaded employees/tasks (via /Employee/upload and
# /task/upload) live in their own SQLite tables and back the lookup /
# CRUD-style endpoints (/employees, /tasks, /predict, /recommend).
# -----------------------------------------------------------------------
_employee_df = pd.read_csv(os.path.join(DATA_DIR, "Employee_Attrition_with_Skills.csv"))
_employee_df = _employee_df.rename(columns={"Skills": "skills"})
_employee_df["skills"] = _employee_df["skills"].fillna("")

_task_df = pd.read_csv(os.path.join(DATA_DIR, "Task_Dataset.csv"))

REQUIRED_EMPLOYEE_COLUMNS = [
    "EmployeeNumber", "Age", "Department", "JobRole", "OverTime", "JobLevel",
    "PerformanceRating", "MonthlyIncome", "BusinessTravel", "DailyRate",
    "DistanceFromHome", "Education", "EducationField", "EnvironmentSatisfaction",
    "Gender", "JobInvolvement", "JobSatisfaction", "MaritalStatus",
    "NumCompaniesWorked", "PercentSalaryHike", "RelationshipSatisfaction",
    "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear",
    "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager", "skills",
]
_employee_df: DataFrame = _employee_df[[c for c in REQUIRED_EMPLOYEE_COLUMNS if c in _employee_df.columns]]


AllowedDepartment = Literal["Human Resources", "Research & Development", "Sales"]
overtime = Literal["Yes", "No"]
BusinessTravel = Literal["Non-Travel", "Travel_Frequently", "Travel_Rarely"]
Gender = Literal["Male", "Female"]
MaritalStatus = Literal["Single", "Married", "Divorced"]
stock_option_level = Literal[0, 1, 2, 3]


class UserInput(BaseModel):
    EmployeeNumber: Annotated[int, Field(..., description="Employee ID")]
    Age: Annotated[int, Field(..., ge=18, le=60, description="Age")]
    Department: Annotated[AllowedDepartment, Field(..., description="Department (allowed values only)")]
    JobRole: Annotated[str, Field(..., description="Job Role")]
    OverTime: Annotated[overtime, Field(..., description="Overtime")]
    JobLevel: Annotated[int, Field(..., ge=1, le=5, description="Job Level")]
    PerformanceRating: Annotated[int, Field(..., ge=1, le=4, description="Performance Rating")]
    MonthlyIncome: Annotated[float, Field(..., ge=0, le=20000, description="Monthly Income")]
    BusinessTravel: Annotated[BusinessTravel, Field(..., description="Business Travel")]
    DailyRate: Annotated[int, Field(..., ge=102, le=1500, description="Daily Rate")]
    DistanceFromHome: Annotated[int, Field(..., description="Distance from Home")]
    Education: Annotated[int, Field(..., ge=1, le=5, description="Education")]
    EducationField: Annotated[str, Field(..., description="Education Field")]
    EnvironmentSatisfaction: Annotated[int, Field(..., ge=1, le=4, description="Environment Satisfaction")]
    Gender: Annotated[Gender, Field(..., description="Gender")]
    JobInvolvement: Annotated[int, Field(..., ge=1, le=4, description="Job Involvement")]
    JobSatisfaction: Annotated[int, Field(..., ge=1, le=4, description="Job Satisfaction")]
    MaritalStatus: Annotated[MaritalStatus, Field(..., description="Marital Status")]
    NumCompaniesWorked: Annotated[int, Field(..., ge=-1, le=10, description="Number of Companies Worked")]
    PercentSalaryHike: Annotated[int, Field(..., ge=10, le=26, description="Percent Salary Hike")]
    RelationshipSatisfaction: Annotated[int, Field(..., ge=1, le=4, description="Relationship Satisfaction")]
    StockOptionLevel: Annotated[stock_option_level, Field(..., description="Stock Option Level")]
    TotalWorkingYears: Annotated[int, Field(..., ge=-1, le=41, description="Total Working Years")]
    TrainingTimesLastYear: Annotated[int, Field(..., ge=0, le=6, description="Training Times Last Year")]
    WorkLifeBalance: Annotated[int, Field(..., ge=1, le=4, description="Work-Life Balance")]
    YearsAtCompany: Annotated[int, Field(..., ge=-1, le=40, description="Years at Company")]
    YearsInCurrentRole: Annotated[int, Field(..., ge=-1, le=20, description="Years in Current Role")]
    YearsSinceLastPromotion: Annotated[int, Field(..., ge=-1, le=15, description="Years Since Last Promotion")]
    YearsWithCurrManager: Annotated[int, Field(..., ge=0, le=17, description="Years with Current Manager")]
    skills: Annotated[str, Field(..., description="skills of Employee")]


# Task-side Literal types. `Task_dura` intentionally includes "Midterm" (not
# "Medium" -- that value belongs to task_complexity/task_priority) since a
# duplicate "Medium" here doesn't exist in the task dataset and would reject
# every real upload row.
Task_complex = Literal["Low", "Medium", "High"]
Task_dura = Literal["Long", "Medium", "Short"]
criticality = Literal["Critical", "Non-critical"]


class TaskUserInput(BaseModel):
    task_id: Annotated[str, Field(..., description="T_1001")]
    task_name: Annotated[str, Field(..., description="Task Name")]
    department: Annotated[AllowedDepartment, Field(..., description="Department")]
    target_role: Annotated[str, Field(..., description="target_role")]
    # NOTE: standardized on "required_skills" (correct spelling). If your
    # Task_Dataset.csv actually has a column literally spelled
    # "required_skils", rename this field (and REQUIRED_TASK_COLUMNS below
    # will follow automatically since it's derived from this model).
    required_skills: Annotated[str, Field(..., description="required skills")]
    task_complexity: Annotated[Task_complex, Field(..., description="Low,Medium,High")]
    estimated_hours: Annotated[int, Field(..., gt=0, lt=101, description="1")]
    task_priority: Annotated[Task_complex, Field(..., description="Low,Medium,High")]
    task_duration: Annotated[Task_dura, Field(..., description="Long, Midterm, Short")]
    criticality: Annotated[criticality, Field(..., description="Critical, Non-critical")]


# Single source of truth for the task schema -- derived from the Pydantic
# model so it can never drift out of sync with it (this also fixes a
# previous typo where the hand-written column list said "required_skils").
REQUIRED_TASK_COLUMNS = list(TaskUserInput.model_fields.keys())


class AllocationRequest(BaseModel):
    employee_df: list[UserInput]
    task_df: list[TaskUserInput]


class TaskAllocationBody(BaseModel):
    task_ids: Optional[list[str]] = Field(
        default=None,
        description="Subset of task_id values to allocate. Omit to allocate every task on file.",
    )


class RecommendBody(BaseModel):
    department: AllowedDepartment
    description: str


@app.get("/")
def home():
    return {"Message": "AI Workforce Management API is running"}


USER_DB_PATH = os.path.join(DATABASE_DIR, "user.db")
TASK_DB_PATH = os.path.join(DATABASE_DIR, "task.db")


# ------------------------------ employee helpers ------------------------------

def _normalize_employee_df(df: DataFrame) -> DataFrame:
    df = df.copy()
    if "Skills" in df.columns and "skills" not in df.columns:
        df = df.rename(columns={"Skills": "skills"})
    df = df[[c for c in REQUIRED_EMPLOYEE_COLUMNS if c in df.columns]]
    df = df.reindex(columns=REQUIRED_EMPLOYEE_COLUMNS)
    if "skills" in df.columns:
        df["skills"] = df["skills"].fillna("")
    return df


def save_employee_to_db(df: DataFrame):
    if df is None or df.empty:
        raise ValueError("No employee data provided")

    df = _normalize_employee_df(df)

    conn = sqlite3.connect(USER_DB_PATH)
    try:
        df.to_sql("employees", conn, if_exists="append", index=False)
    finally:
        conn.close()


def delete_employee_from_db(employee_numbers: Optional[list[int] | int] = None) -> DataFrame:
    if employee_numbers is None:
        raise ValueError("employee_numbers must be provided for delete")
    if isinstance(employee_numbers, int):
        employee_numbers = [employee_numbers]

    current = fetch_employees_from_db()
    result = current[~current["EmployeeNumber"].isin(employee_numbers)].copy()

    # Write back into the SQLite table (was previously writing CSV text
    # over the .db file, which corrupted the database for every other
    # endpoint that opens it with sqlite3.connect()).
    conn = sqlite3.connect(USER_DB_PATH)
    try:
        result.to_sql("employees", conn, if_exists="replace", index=False)
    finally:
        conn.close()
    return result


def update_employee_in_db(employee_number: int, updates: dict) -> DataFrame:
    if not isinstance(employee_number, int):
        raise ValueError("employee_number must be an integer")
    if not updates:
        raise ValueError("updates must be provided")

    current = fetch_employees_from_db()
    if current.empty or employee_number not in current["EmployeeNumber"].values:
        raise KeyError(f"EmployeeNumber {employee_number} not found")

    row_idx = current.index[current["EmployeeNumber"] == employee_number]
    for col, value in updates.items():
        if col not in current.columns:
            raise ValueError(f"Invalid column: {col}")
        current.loc[row_idx, col] = value

    conn = sqlite3.connect(USER_DB_PATH)
    try:
        current.to_sql("employees", conn, if_exists="replace", index=False)
    finally:
        conn.close()
    return current


def fetch_employees_from_db() -> DataFrame:
    conn = sqlite3.connect(USER_DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM employees", conn)
    except Exception:
        df = pd.DataFrame(columns=REQUIRED_EMPLOYEE_COLUMNS)
    finally:
        conn.close()

    return _normalize_employee_df(df)


def _load_employees_from_db():
    return fetch_employees_from_db()


# -------------------------------- task helpers --------------------------------
def _normalize_task_df(df: DataFrame) -> DataFrame:
    df = df.copy()
    df = df[[c for c in REQUIRED_TASK_COLUMNS if c in df.columns]]
    df = df.reindex(columns=REQUIRED_TASK_COLUMNS)
    return df


def save_task_to_db(df: DataFrame):
    if df is None or df.empty:
        raise ValueError("No task data provided")

    df = _normalize_task_df(df)

    conn = sqlite3.connect(TASK_DB_PATH)
    try:
        df.to_sql("tasks", conn, if_exists="append", index=False)
    finally:
        conn.close()


def fetch_task_db() -> DataFrame:
    conn = sqlite3.connect(TASK_DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM tasks", conn)
    except Exception:
        df = pd.DataFrame(columns=REQUIRED_TASK_COLUMNS)
    finally:
        conn.close()

    return _normalize_task_df(df)


def _load_task_from_db():
    return fetch_task_db()


# -----------------------------------------------------------------------
# Upload endpoints
# -----------------------------------------------------------------------
@app.post("/Employee/upload")
async def emp_details(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="only CSV files are accepted")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {e}")

    if "Skills" in df.columns and "skills" not in df.columns:
        df = df.rename(columns={"Skills": "skills"})

    missing = [c for c in REQUIRED_EMPLOYEE_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    # row by row data validation using pydantic model
    records = df.to_dict(orient="records")
    validated = []
    row_error = []

    for idx, row in enumerate(records):
        try:
            emp = UserInput.model_validate(row)
            validated.append(emp)
        except ValidationError as e:
            row_error.append({
                "row": idx + 2,  # +2 because row 1 is header
                "errors": e.errors()
            })
    if row_error:
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"{len(row_error)} row(s) failed validation. File was NOT saved.",
                "errors": row_error[:30]
            }
        )

    clean_df = pd.DataFrame([emp.model_dump() for emp in validated])
    save_employee_to_db(clean_df)

    return {
        "status": "Success",
        "message": f"Uploaded and saved {len(clean_df)} employees",
        "count": len(clean_df)
    }


@app.get("/employees")
def list_employees(department: Optional[AllowedDepartment] = None):
    df = fetch_employees_from_db()
    if department:
        df = df[df["Department"] == department]
    records = df.to_dict(orient="records")
    return {"count": len(records), "data": records}


@app.get("/employees/{employee_number}")
def get_employee(employee_number: int):
    df = fetch_employees_from_db()
    row = df[df["EmployeeNumber"] == employee_number]
    if row.empty:
        raise HTTPException(status_code=404, detail="Employee not found")
    return row.iloc[0].to_dict()


@app.post("/task/upload")
async def task_details(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="only csv file is present")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to read csv file: {e}")

    missing = [c for c in REQUIRED_TASK_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"missing required columns: {missing}")

    records = df.to_dict(orient="records")
    validated = []
    row_error = []

    for idx, row in enumerate(records):
        try:
            task = TaskUserInput.model_validate(row)
            # Save the validated/coerced record, not the raw CSV row --
            # otherwise pydantic's type coercion is discarded and stray
            # extra columns from the CSV leak into the saved table.
            validated.append(task.model_dump())
        except ValidationError as e:
            row_error.append({
                "row": idx + 2,
                "error": e.errors()
            })
    if row_error:
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"{len(row_error)} row(s) failed validation. File was NOT saved.",
                "errors": row_error[:30]
            }
        )

    clean_df = pd.DataFrame(validated)
    save_task_to_db(clean_df)

    return {
        "status": "Success",
        "message": f"Uploaded and saved {len(clean_df)} tasks",
        "count": len(clean_df)
    }


@app.get("/tasks")
def list_tasks(department: Optional[AllowedDepartment] = None):
    df = fetch_task_db()
    if department:
        df = df[df["department"] == department]
    records = df.to_dict(orient="records")
    return {"count": len(records), "data": records}


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    df = fetch_task_db()
    row = df[df["task_id"] == task_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Task not found")
    return row.iloc[0].to_dict()


# -----------------------------------------------------------------------
# ML endpoints
# -----------------------------------------------------------------------
@app.post("/predict/{employee_number}")
def predict(employee_number: int):
    df = fetch_employees_from_db()
    row = df[df["EmployeeNumber"] == employee_number]
    if row.empty:
        raise HTTPException(status_code=404, detail="Employee not found")

    try:
        scored = predict_attrition(row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    result = scored.iloc[0]

    return {
        "EmployeeNumber": int(result["EmployeeNumber"]),
        "prediction": round(float(result["Attrition_Probability"]), 4),
        "attrition": "Yes" if result["Attrition_Probability"] >= 0.5 else "No",
        "risk_level": result["Attrition_Risk_Level"],
    }


@app.post("/task-allocation")
def allocation(body: TaskAllocationBody = TaskAllocationBody()):
    task_df = _task_df
    if body.task_ids:
        task_df = task_df[task_df["task_id"].isin(body.task_ids)]
        if task_df.empty:
            raise HTTPException(status_code=404, detail="None of the given task_ids were found")

    try:
        result = task_allocation(_employee_df, task_df)
        return {
            "status": "Success",
            "message": "Tasks allocated to the best-fitted employees",
            "data": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Allocation failed: {e}")


@app.post("/recommend")
def recommendation(body: RecommendBody):
    row = fetch_employees_from_db()
    try:
        result = top_recommendation(row, body.department, body.description)
        return {
            "status": "Success",
            "message": "Best-matched employees for the description",
            "data": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}")