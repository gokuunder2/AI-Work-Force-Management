import pandas as pd
import numpy as np
import joblib 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

model = joblib.load("Models/AI_workforce_management@.pkl")
scaler = joblib.load("Models/scaler.pkl")
vectorizer = joblib.load("Models/tfidf_vectorizer.pkl")
model = joblib.load("Models/attrition_model.pkl")
scaler = joblib.load("Models/scaler.pkl")
feature_columns = joblib.load("Models/feature_columns.pkl")


def task_allocation(employee_df, task_df):
    
    # Employee Data preprocessing
    df = employee_df.copy()
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
    final_df = pd.DataFrame(0, index=df_encoded.index, columns=feature_columns)

    for col in df_encoded.columns:
        if col in final_df.columns:
            final_df[col] = df_encoded[col]

    final_df = final_df.astype(float)
    scaler_data=scaler.transform(final_df) #convert the data into sclare
    prediction = model.predict_proba(scaler_data)[:, 1]
    
    df["Attrition_Probability"] = prediction
    
    def risk_level(p):
        if p < 0.30:
            return "Low"
        elif p < 0.60:
            return "Medium"
        else:
            return "High"

    #create new columns for allocation
    df["Attrition_Risk_Level"] = df["Attrition_Probability"].apply(risk_level)
    df["OverTime_Flag"] = df["OverTime"]
    df["Capacity_Score"] = (1 - df["OverTime_Flag"]).astype(float)
    df["recent_performance_score"] = 0.6 + ((df["PerformanceRating"] - 3) / (1)) * 0.3
    df["pervious_performance_score"] =(df["PercentSalaryHike"] / 25.0)
    df["performance_trend"]= df["recent_performance_score"] - df["pervious_performance_score"]
    df["performance_score"]= df["recent_performance_score"]*0.7+df["performance_trend"]*0.3
    df["performance_score"]+=df["JobSatisfaction"]/100
    df['Engagement_Score'] = (df["JobSatisfaction"] + df["EnvironmentSatisfaction"] + df["WorkLifeBalance"])/3 
    eng_scaler = MinMaxScaler()# use minmax fit_transform funcation to make the data in binary
    df["Engagement_Score"] = eng_scaler.fit_transform(df[["Engagement_Score"]])
    df["Burnout_Risk"]= 0.4* df["OverTime_Flag"] + 0.3*(1-df["Engagement_Score"]) + 0.3*(1-df["performance_score"])
    

    #main value for task allocation
    df["Allocation_score"]= 0.30*df["performance_score"]+0.20*df["Capacity_Score"]+0.15*df["Engagement_Score"]+0.20*(1-df["Burnout_Risk"])+0.15*(1-df["Attrition_Probability"])

    print("Task dataset preprocessing started")

    task_df = task_df.copy()

    # Create skill vectors
    employee_skill_matrix = vectorizer.transform(
        df["skills"].fillna("")
    )

    task_skill_matrix = vectorizer.transform(
        task_df["required_skills"].fillna("")
    )

    task_df_sorted = task_df.sort_values(
        by=["task_priority", "criticality"],
        ascending=[True, False]
    )
    
    assignments=[]
    working_hours=160
    for task in task_df_sorted.itertuples():
        eligible_employees=df.copy()
        
        eligible_employees = eligible_employees[
        eligible_employees["Department"] == task.department]
        
        
        
        # Get vectors of eligible employees
        eligible_vectors = employee_skill_matrix[eligible_employees.index]
        # Get current task vector
        task_vector = task_skill_matrix[task.Index]
        # Calculate cosine similarity
        skill_scores = cosine_similarity(eligible_vectors,task_vector)
        # Store skill match score
        eligible_employees["Skill_Match_Score"] = skill_scores.flatten()
        #flatten is used because cousine similarity returns a 2D array, and we want to convert it to a 1D array for easier handling.
        print("\nTASK:", task.task_id)

        print("After Department:",
            len(eligible_employees))

        print("Skill scores:")
        print(eligible_employees[
            ["EmployeeNumber", "Skill_Match_Score"]
        ])
        
        eligible_employees = eligible_employees[
        eligible_employees["Skill_Match_Score"] > 0.5]
        print("After Skill:", len(eligible_employees))    
        
        eligible_employees = eligible_employees[
        eligible_employees["Attrition_Probability"] <= 0.7]
        print("After Attrition:", len(eligible_employees))
        
        eligible_employees = eligible_employees[
        eligible_employees["Burnout_Risk"] < 0.7]
        print("After Burnout:", len(eligible_employees))
       
        eligible_employees = eligible_employees[
        eligible_employees["Capacity_Score"] > 0.2]
        print("After Capacity:", len(eligible_employees))
         
        #Calculating base allocation score for eligible employees
        eligible_employees["Task_Allocation_score"] = (
        0.25 * eligible_employees["Skill_Match_Score"]
        + 0.20 * eligible_employees["performance_score"]
        + 0.15 * eligible_employees["Capacity_Score"]
        + 0.10 * eligible_employees["Engagement_Score"]
        + 0.10 * (1 - eligible_employees["Attrition_Probability"])
        + 0.10 * (1 - eligible_employees["Burnout_Risk"])
        )
        
    
        #fainess Penalty
        if len(assignments) > 0:

            temp_df = pd.DataFrame(assignments)

            task_count = temp_df["EmployeeNumber"].value_counts()

            eligible_employees["current_task_count"] = (
                eligible_employees["EmployeeNumber"]
                .map(task_count)
                .fillna(0)
            )

    else:
        eligible_employees["current_task_count"] = 0

        eligible_employees["Task_Allocation_score"] -= (
            0.7 * eligible_employees["current_task_count"]
            )
        
        #selecting top employees
        best_employee=eligible_employees.sort_values(by="Task_Allocation_score",ascending=False).iloc[0]#used to select row by with the help of indexing
        assignments.append(
            {
                'Task_id': task.task_id,
                'Task_name':task.task_name,
                'Task_skills': task.required_skills,
                'Complexity': task.task_complexity,
                'EmployeeNumber.': best_employee["EmployeeNumber"],
                'Task_Allocation_score': best_employee["Task_Allocation_score"]
            }
        )
        
        #reduce capacity
        reduction = task.estimated_hours / working_hours

        df.loc[
            best_employee.name,
            "Capacity_Score"
        ] -= reduction

        df["Capacity_Score"] = df["Capacity_Score"].clip(lower=0)
        #clip is used for rollback value to zero after crossing zero  
        
    Allocation_result = pd.DataFrame(assignments)

    return Allocation_result.to_dict(orient="records")

#top3_employee for task
def top_recommendation(employe_df,department,description):
    try:
        employee_df = pd.DataFrame([emp.model_dump() for emp in employe_df])
        print(employee_df.columns.tolist())
        if department not in employee_df["Department"].unique():
            return {"status": "Error", "message": "Invalid Department"}
            
        print("Task_description:", description)
        print("Department:", department)
        #Boolean mask:- filter data on department
        dept_pool=employee_df[employee_df["Department"]==department].copy()

        #similarity score
        vector_desc=vectorizer.transform([description])
        vector_emp_skill = vectorizer.transform(dept_pool["skills"])
        skill_score=cosine_similarity(vector_desc,vector_emp_skill).flatten()
        
        dept_pool["skill_score"]=skill_score
        
        dept_pool["Total_Score"]=0.7*dept_pool["skill_score"]+0.3*dept_pool["PerformanceRating"]
        
        top_emp = dept_pool.sort_values(by=["Total_Score"], ascendingending=False).head(3)
        
        return top_emp.to_dict(orient="records")
    except Exception as e:
        return {"status": "Error", "message": str(e)}

if __name__ == "__main__":
    task_allocation() 
    top_recommendation()    
    
    