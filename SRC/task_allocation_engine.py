import pandas as pd
import numpy as np
import joblib
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity

# 1. LOAD ASSETS
model = joblib.load("Models/AI_workforce_management@.pkl")
scaler = joblib.load("Models/scaler.pkl")
vectorizer = joblib.load("Models/tfidf_vectorizer.pkl")

def run_allocation():
    # Load Datasets
    df = pd.read_csv("Datasets/Employee_Attrition_with_Skills.csv")
    task_df = pd.read_csv("Datasets/Task_Dataset.csv")

    # --- PREPARE EMPLOYEE SCORES ---
    df['OverTime_Flag'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
    df["Attrition_Probability"] = 0.15 
    df["Burnout_Risk"] = ((df['OverTime_Flag'] * 0.4) + 0.3).clip(0.1, 1.0)
    df["Performance_Score"] = (df["PerformanceRating"] / 4.0).clip(0.1, 1.0)
    df["Capacity_Score"] = 1.0 

    # --- ALLOCATION LOGIC ---
    task_df_sorted = task_df.sort_values(by=["task_priority", "criticality"], ascending=[True, False])
    assignments = []
    
    emp_skill_matrix = vectorizer.transform(df["Skills"].astype(str))
    task_skill_matrix = vectorizer.transform(task_df["required_skills"].astype(str))

    for task in task_df_sorted.itertuples():
        eligible = df.copy()
        
        # Filtering
        if task.criticality == "Critical":
            eligible = eligible[eligible["Attrition_Probability"] < 0.6]
        eligible = eligible[(eligible["Burnout_Risk"] < 0.7) & (eligible["Capacity_Score"] > 0.1)]
        
        if eligible.empty: continue

        # Scoring
        task_vec = task_skill_matrix[task.Index]
        eligible_idx = [df.index.get_loc(i) for i in eligible.index]
        skills_sim = cosine_similarity(emp_skill_matrix[eligible_idx], task_vec)
        eligible["Skill_Match"] = skills_sim.flatten()

        eligible["Total_Score"] = (
            0.35 * eligible["Skill_Match"] + 
            0.20 * eligible["Performance_Score"] + 
            0.15 * eligible["Capacity_Score"] + 
            0.30 * (1 - eligible["Burnout_Risk"])
        )

        # Fairness Penalty
        if assignments:
            counts = pd.DataFrame(assignments)["employee_id"].value_counts()
            eligible["Total_Score"] -= (0.07 * eligible.index.map(counts).fillna(0))

        # Select Top 3
        top_3 = eligible.sort_values("Total_Score", ascending=False).head(3)
        
        for rank, (idx, emp) in enumerate(top_3.iterrows(), 1):
            assignments.append({
                "task_id": task.task_id,
                "task_name": task.task_name,
                "rank": rank,
                "employee_id": idx,
                "skill_match_score": round(emp["Skill_Match"] * 100, 2),
                "allocation_score": round(emp["Total_Score"] * 100, 2)
            })

        # Update Capacity for #1
        df.at[top_3.index[0], "Capacity_Score"] -= (task.estimated_hours / 160)

    # FIXED INDENTATION: These lines now run AFTER the loop finishes
    results = pd.DataFrame(assignments)
    results.to_csv("allocation_results.csv", index=False)
    return results

def get_top_3_recommendations(task_description, department_name):
    # 1. Load data first so we can check columns
    df = pd.read_csv("Datasets/Employee_Attrition_with_Skills.csv")
    if department_name not in df['Department'].unique():
        print(f"Error:{department_name} not found in csv.")
        return pd.DataFrame()
    print("Task Received:", task_description)
    print("Dataset Columns found:", df.columns.tolist())
    
    # 2. Filter by department
    dept_pool = df[df['Department'] == department_name].copy()
    
    if dept_pool.empty:
        print(f"Warning: No employees found in department: {department_name}")
        return pd.DataFrame()

    # 3. Vectorization and Scoring
    desc_vector = vectorizer.transform([task_description])
    emp_skill_matrix = vectorizer.transform(dept_pool["Skills"].astype(str))
    
    dept_pool["Skill_Match"] = cosine_similarity(emp_skill_matrix, desc_vector).flatten()
    
    # We use 'Total_Score' here
    dept_pool["Total_Score"] = (0.50 * dept_pool["Skill_Match"] + 0.50 * (dept_pool["PerformanceRating"] / 4))
    
    print("Scores calculated successfully.")
    
    # 4. Get Top 3 using the CORRECT column name (Total_Score)
    # Also ensure we are using dept_pool, not the original df
    top_three = dept_pool.nlargest(3, 'Total_Score')
    
    # Use .get() or check if 'Name' exists to prevent crashes
    name_list = top_three['EmployeeNumber'].tolist() # Change to 'Name' if that column exists
    print("Top 3 identified IDs:", name_list)
    
    return top_three

if __name__ == "__main__":
    run_allocation()