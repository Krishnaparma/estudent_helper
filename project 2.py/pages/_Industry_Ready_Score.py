import streamlit as st
import pandas as pd
import datetime
import numpy as np
import plotly.express as px

#      Placeholder task generator based on field 
def get_tasks_for_field(field):
    if field == "IT":
        return [
            {"task": "Learn Python basics", "link": "https://youtu.be/_uQrJ0TkZlc"},
            {"task": "Master Git & GitHub", "link": "https://youtu.be/apGV9Kg7ics"},
            {"task": "Understand Operating Systems", "link": "https://youtu.be/26QPDBe-NB8"},
            {"task": "Build mini projects in Python", "link": "https://youtu.be/8ext9G7xspg"},
        ]
    elif field == "Data Science / AI / ML":
        return [
            {"task": "Learn Python for Data Science", "link": "https://youtu.be/rfscVS0vtbw"},
            {"task": "Learn Pandas & NumPy", "link": "https://youtu.be/ZyhVh-qRZPA"},
            {"task": "Explore Scikit-learn & ML models", "link": "https://youtu.be/0Lt9w-BxKFQ"},
            {"task": "Complete end-to-end ML project", "link": "https://youtu.be/wgk0YvJGnVc"},
        ]
    else:
        return [
            {"task": "Explore basics in your field", "link": "https://youtu.be/Z1Yd7upQsXY"},
        ]

# ------------------ ML Model Placeholder ------------------
def predict_salary(tasks_completed, total_tasks):
    if total_tasks == 0:
        return 0
    percent = tasks_completed / total_tasks
    predicted_lpa = round(3 + percent * 15, 2)  # Base 3 LPA + growth
    return predicted_lpa

# ------------------ App Layout ------------------
st.set_page_config(page_title="Industry Ready Score", layout="wide")
st.title("📈 Industry Ready Score & Career Planner")

# ------------------ Select Field ------------------
st.subheader("Step 1: Choose Your Career Field")
field = st.selectbox("Select your field of interest:", ["IT", "Data Science / AI / ML", "Cybersecurity", "Other"])

# ------------------ Load or Generate Tasks ------------------
tasks = get_tasks_for_field(field)
df = pd.DataFrame(tasks)

# Add columns for completion status
if "status" not in df.columns:
    df["status"] = False

# ------------------ Display and Edit Tasks ------------------
st.subheader("Step 2: Complete the Suggested Tasks")
for i, row in df.iterrows():
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown(f"**{i+1}. {row['task']}** — [Watch here]({row['link']})")
    with col2:
        df.at[i, "status"] = st.checkbox("Done", key=f"task_{i}", value=row["status"])

# ------------------ Show Progress and Prediction ------------------
completed_tasks = df["status"].sum()
total_tasks = len(df)
completion_percent = int((completed_tasks / total_tasks) * 100) if total_tasks else 0

st.subheader("Step 3: Your Progress")
st.progress(completion_percent)
st.write(f"**{completed_tasks} / {total_tasks} tasks completed ({completion_percent}%)**")

# ------------------ Predict Salary ------------------
predicted_lpa = predict_salary(completed_tasks, total_tasks)
st.metric("🎯 Predicted Job Offer (LPA)", f"₹ {predicted_lpa} LPA")

# ------------------ Chart ------------------
st.subheader("Step 4: Progress Chart")
chart_df = pd.DataFrame({
    "Tasks": ["Completed", "Pending"],
    "Count": [completed_tasks, total_tasks - completed_tasks]
})
st.plotly_chart(px.pie(chart_df, names="Tasks", values="Count", title="Task Completion Status"))

# ------------------ Download Progress ------------------
st.subheader("Step 5: Save Your Progress")
if st.button("Download Progress CSV"):
    progress_df = df.copy()
    progress_df["completed"] = progress_df["status"].apply(lambda x: "Yes" if x else "No")
    csv = progress_df[["task", "link", "completed"]].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Progress File", data=csv, file_name="career_progress.csv", mime='text/csv')