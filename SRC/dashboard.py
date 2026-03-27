import streamlit as st
import requests
import pandas as pd
import sqlite3
import io
import os
from task_allocation_engine import get_top_3_recommendations


# Configuration
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Workforce Manager", layout="wide")
st.title("📊 AI Workforce Management Dashboard")

tabs = st.tabs([
    "🔍 Attrition Predictor", 
    "📁 Custom Dataset Workspace",  # New Upload & Allocation Hub
    "📋 Live Database",
    "⚙️ Internal Bulk Engine"        # Moved original bulk logic here
])

# --- TAB 1: ATTRITION PREDICTOR ---
with tabs[0]:
    st.header("Predict Employee Attrition")
    col1, col2 = st.columns(2)
    
    with col1:
        emp_id = st.number_input("Employee ID", value=101, step=1)
        age = st.number_input("Age", min_value=0, max_value=100, value=30, step=1)
        if age >0 and age<18:
            st.warning("Age must be greater then 18")
        dept = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        job_level = st.slider("Job Level", 1, 5, 1)
    
    with col2:
        income = st.number_input("Monthly Income ($)", value=5000, step=100)
        manager_years = st.number_input("Years with Current Manager", value=2, step=1)
        overtime = st.radio("Overtime", ["Yes", "No"])
        perf = st.slider("Performance Rating", 1, 4, 3)
        role = st.text_input("Job Role", "Sales Executive")

    if st.button("Predict Risk", disabled=(age < 18)):
        payload = {
            "employee_id": emp_id, "Age": age, "Department": dept,
            "JobRole": role, "OverTime": overtime, "JobLevel": job_level,
            "PerformanceRating": perf, "YearsWithCurrManager": manager_years, "MonthlyIncome": income
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            
            # 🛑 Check if HTTP Request actually succeeded (Status Code 200)
            if response.status_code == 200:
                res = response.json()
                
                # Check if JSON payload contains 'status'
                if isinstance(res, dict) and res.get("status") == "Success":
                    color = "red" if res["attrition_risk"] == "High" else "green"
                    st.markdown(f"### Risk: :{color}[{res['attrition_risk']}]")
                    st.metric("Probability Score", f"{res['probability_score']:.4f}")
                else:
                    st.error("⚠️ Backend returned a success status code, but malformed JSON.")
                    st.json(res)
            else:
                st.error(f"❌ Backend Error (HTTP {response.status_code})")
                st.write("FastAPI might be crashing internally. Check your FastAPI terminal!")
                
        except requests.exceptions.ConnectionError:
            st.error("🔌 Could not connect to FastAPI! Ensure your Uvicorn server is running in another terminal window.")
        

# Import the engine function at the top of your dashboard.py


# --- TAB 2: CUSTOM DATASET WORKSPACE ---
with tabs[1]:
    st.header("📁 Custom Dataset Management")
    st.info("Upload your employee list here to perform specific task allocations.")

    uploaded_emp = st.file_uploader("Upload Employee Dataset (CSV)", type="csv", key="emp_upload")
    
    if uploaded_emp:
        emp_df = pd.read_csv(uploaded_emp)
        st.success(f"Loaded {len(emp_df)} employees from {uploaded_emp.name}")
        
        with st.expander("🔍 Preview Employee Data"):
            st.dataframe(emp_df.head(5), width='stretch')

        st.divider()

        # --- SUB-SECTION: ALLOCATION OPTIONS ---
        col_single, col_bulk = st.columns(2, gap="large")

        with col_single:
            st.subheader("🎯 Single Task Allocation")
            st.write("Find the top 3 matches for a specific task.")
            
            task_description = st.text_area("Task Description", placeholder="Enter task details...", key="single_task_desc")
            
            department_name = st.selectbox(
                "Select Department", 
                ["Finance", "Data Analytics", "Engineering", "HR", "Sales"],
                key="dept_select"
            )
            
            if st.button("Find Best Matches", use_container_width=True):
                if task_description:
                    with st.spinner("Processing single allocation against uploaded dataset..."):
                        # Correctly call the backend function
                        results = get_top_3_recommendations(task_description, department_name)
                
                        if not results.empty:
                            st.success("Top 3 Matches Found!")
                            # Show results inside the button logic block
                            st.dataframe(results[['EmployeeNumber', 'Total_Score', 'Skill_Match']], width='stretch')
                        else:
                            st.warning("No employees found. Please check if the Department matches your CSV exactly.")
                else:
                    st.error("Please enter a task description first.")

        with col_bulk:
            st.subheader("📦 Bulk Task Allocation")
            st.write("Upload a CSV containing multiple tasks to auto-assign.")
            
            uploaded_tasks = st.file_uploader("Upload Task Dataset (CSV)", type="csv", key="task_upload")
            
            if uploaded_tasks:
                task_df = pd.read_csv(uploaded_tasks)
                st.write(f"📋 {len(task_df)} tasks detected for assignment.")
                
                if st.button("Run Bulk Assignment", width='stretch'):
                    with st.spinner("Matching multiple tasks to your custom dataset..."):
                        # Assuming your engine handles the bulk CSVs
                        st.success("Bulk Allocation Complete! Results ready for download.")
# --- TAB 3: LIVE DATABASE ---
with tabs[2]:
    st.header("Recent Predictions (workforce.db)")
    conn = sqlite3.connect("workforce.db")
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10", conn)
    st.dataframe(df, use_container_width=True)
    conn.close()
# --- TAB 4: INTERNAL BULK ENGINE ---
with tabs[3]:
    st.header("⚙️ System-Level Bulk Allocation")
    st.info("This engine uses the master 'Task_Dataset.csv' and 'Employee_Attrition_with_Skills.csv' stored on the server.")

    col_stats, col_action = st.columns([1, 1])

    with col_stats:
        st.subheader("System Status")
        # Check if files exist to prevent crashing
        import os
        tasks_exist = os.path.exists("Datasets/Task_Dataset.csv")
        emps_exist = os.path.exists("Datasets/Employee_Attrition_with_Skills.csv")
        
        if tasks_exist and emps_exist:
            st.success("✅ All Master Files Present")
            task_count = len(pd.read_csv("Datasets/Task_Dataset.csv"))
            st.write(f"📊 Tasks waiting in queue: **{task_count}**")
        else:
            st.error("⚠️ Master Files Missing! Ensure CSVs are in the project folder.")

    with col_action:
        st.subheader("Controls")
        if st.button("Execute Global Assignment", width='stretch', disabled=not (tasks_exist and emps_exist)):
            with st.spinner("Running system-wide optimization..."):
                # Call your backend engine
                from task_allocation_engine import run_allocation
                results = run_allocation()
                
                if not results.empty:
                    st.success("Global Allocation Complete!")
                    st.balloons()
                else:
                    st.warning("Process finished, but no matches were made due to safety filters.")
    if tasks_exist and emps_exist:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # Back up from SRC to root, then go to Datasets
        ROOT_DIR = os.path.dirname(BASE_DIR)
        RESULTS_PATH = os.path.join(ROOT_DIR, "Datasets", "allocation_results.csv")
        
        st.divider()
        st.subheader("Last Allocation Preview")

        if os.path.exists(RESULTS_PATH):
            if os.path.getsize(RESULTS_PATH) > 0:
                try:
                    last_results = pd.read_csv(RESULTS_PATH) # 🛠️ Using uniform variable!
                    st.dataframe(last_results.head(10), use_container_width=True)
            
                    csv_data = last_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Allocation Results",
                        data=csv_data,
                        file_name="workforce_assignments_2026.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                except pd.errors.EmptyDataError:
                    st.warning("⚠️ The allocation file exists but contains no data to read yet.")
            else:
                st.warning("⏳ The allocation file is empty. Click 'Execute Global Assignment' to generate data!")
        else:
            st.info("ℹ️ No allocation history found yet! Click the button above to generate your first results file.")