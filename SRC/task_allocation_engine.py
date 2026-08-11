import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Resolve model paths relative to the repo root (parent of this SRC/ folder),
# so the API works no matter what directory it's launched from.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "Models")

model = joblib.load(os.path.join(MODELS_DIR, "attrition_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))


def _score_employees(employee_df: pd.DataFrame) -> pd.DataFrame:
    """Shared feature engineering + scoring used by both prediction and allocation."""
    df = employee_df.copy()

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    df["OverTime"] = df["OverTime"].apply(lambda x: 1 if x == "Yes" else 0)
    df["Gender"] = df["Gender"].apply(lambda x: 1 if x == "Male" else 0)

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

    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    df_encoded.drop(
        ["YearsAtCompany", "MonthlyIncome", "Age"],
        axis=1,
        inplace=True,
        errors="ignore"
    )

    final_df = pd.DataFrame(0, index=df_encoded.index, columns=feature_columns)
    for col in df_encoded.columns:
        if col in final_df.columns:
            final_df[col] = df_encoded[col]
    final_df = final_df.astype(float)

    scaled_data = scaler.transform(final_df)
    prediction = model.predict_proba(scaled_data)[:, 1]

    df["Attrition_Probability"] = prediction

    def risk_level(p):
        if p < 0.30:
            return "Low"
        elif p < 0.60:
            return "Medium"
        else:
            return "High"

    df["Attrition_Risk_Level"] = df["Attrition_Probability"].apply(risk_level)
    df["OverTime_Flag"] = df["OverTime"]
    df["Capacity_Score"] = (1 - df["OverTime_Flag"]).astype(float)
    df["recent_performance_score"] = 0.6 + ((df["PerformanceRating"] - 3) / 1) * 0.3
    df["pervious_performance_score"] = (df["PercentSalaryHike"] / 25.0)
    df["performance_trend"] = df["recent_performance_score"] - df["pervious_performance_score"]
    df["performance_score"] = df["recent_performance_score"] * 0.7 + df["performance_trend"] * 0.3
    df["performance_score"] += df["JobSatisfaction"] / 100
    df["Engagement_Score"] = (df["JobSatisfaction"] + df["EnvironmentSatisfaction"] + df["WorkLifeBalance"]) / 3
    eng_scaler = MinMaxScaler()
    df["Engagement_Score"] = eng_scaler.fit_transform(df[["Engagement_Score"]])
    df["Burnout_Risk"] = 0.4 * df["OverTime_Flag"] + 0.3 * (1 - df["Engagement_Score"]) + 0.3 * (1 - df["performance_score"])

    df["Allocation_score"] = (
        0.30 * df["performance_score"]
        + 0.20 * df["Capacity_Score"]
        + 0.15 * df["Engagement_Score"]
        + 0.20 * (1 - df["Burnout_Risk"])
        + 0.15 * (1 - df["Attrition_Probability"])
    )

    return df


def predict_attrition(employee_df: pd.DataFrame) -> pd.DataFrame:
    """Run attrition prediction for one or more employees; returns df with predictions."""
    return _score_employees(employee_df)


def task_allocation(employee_df: pd.DataFrame, task_df: pd.DataFrame):
    df = _score_employees(employee_df)

    task_df = task_df.copy()

    employee_skill_matrix = vectorizer.transform(df["skills"].fillna(""))
    task_skill_matrix = vectorizer.transform(task_df["required_skills"].fillna(""))

    task_df_sorted = task_df.sort_values(
        by=["task_priority", "criticality"],
        ascending=[True, False]
    )

    assignments = []
    unassigned = []
    working_hours = 160

    for task in task_df_sorted.itertuples():
        eligible_employees = df.copy()

        eligible_employees = eligible_employees[
            eligible_employees["Department"] == task.department
        ]

        eligible_vectors = employee_skill_matrix[eligible_employees.index]
        task_vector = task_skill_matrix[task.Index]
        skill_scores = cosine_similarity(eligible_vectors, task_vector)
        eligible_employees["Skill_Match_Score"] = skill_scores.flatten()

        eligible_employees = eligible_employees[eligible_employees["Skill_Match_Score"] > 0.5]
        eligible_employees = eligible_employees[eligible_employees["Attrition_Probability"] <= 0.7]
        eligible_employees = eligible_employees[eligible_employees["Burnout_Risk"] < 0.7]
        eligible_employees = eligible_employees[eligible_employees["Capacity_Score"] > 0.2]

        # No one qualifies for this task -- skip it instead of crashing the whole batch.
        if eligible_employees.empty:
            unassigned.append({
                "Task_id": task.task_id,
                "Task_name": task.task_name,
                "Reason": "No eligible employee (department/skill/attrition/burnout/capacity filters)"
            })
            continue

        eligible_employees["Task_Allocation_score"] = (
            0.25 * eligible_employees["Skill_Match_Score"]
            + 0.20 * eligible_employees["performance_score"]
            + 0.15 * eligible_employees["Capacity_Score"]
            + 0.10 * eligible_employees["Engagement_Score"]
            + 0.10 * (1 - eligible_employees["Attrition_Probability"])
            + 0.10 * (1 - eligible_employees["Burnout_Risk"])
        )

        # Fairness penalty: employees already assigned tasks this run get penalized,
        # so work spreads across the team instead of piling onto one person.
        if len(assignments) > 0:
            temp_df = pd.DataFrame(assignments)
            task_count = temp_df["EmployeeNumber"].value_counts()
            eligible_employees["current_task_count"] = (
                eligible_employees["EmployeeNumber"].map(task_count).fillna(0)
            )
        else:
            eligible_employees["current_task_count"] = 0

        eligible_employees["Task_Allocation_score"] -= (
            0.7 * eligible_employees["current_task_count"]
        )

        best_employee = eligible_employees.sort_values(
            by="Task_Allocation_score", ascending=False
        ).iloc[0]

        assignments.append({
            "Task_id": task.task_id,
            "Task_name": task.task_name,
            "Task_skills": task.required_skills,
            "Complexity": task.task_complexity,
            "EmployeeNumber": int(best_employee["EmployeeNumber"]),
            "Task_Allocation_score": round(float(best_employee["Task_Allocation_score"]), 4),
        })

        reduction = task.estimated_hours / working_hours
        df.loc[best_employee.name, "Capacity_Score"] -= reduction
        df["Capacity_Score"] = df["Capacity_Score"].clip(lower=0)

    return {
        "assignments": assignments,
        "unassigned": unassigned,
    }


def top_recommendation(employee_df: pd.DataFrame, department: str, description: str):
    """Return the top 3 employees in a department matching a free-text task description."""
    if department not in employee_df["Department"].unique():
        return {"status": "Error", "message": "Invalid Department"}

    dept_pool = employee_df[employee_df["Department"] == department].copy()
    if dept_pool.empty:
        return {"status": "Error", "message": "No employees found in that department"}

    vector_desc = vectorizer.transform([description])
    vector_emp_skill = vectorizer.transform(dept_pool["skills"].fillna(""))
    skill_score = cosine_similarity(vector_desc, vector_emp_skill).flatten()

    dept_pool["skill_score"] = skill_score
    dept_pool["Total_Score"] = 0.7 * dept_pool["skill_score"] + 0.3 * (dept_pool["PerformanceRating"] / 4.0)

    top_emp = dept_pool.sort_values(by="Total_Score", ascending=False).head(3)

    return top_emp.to_dict(orient="records")
