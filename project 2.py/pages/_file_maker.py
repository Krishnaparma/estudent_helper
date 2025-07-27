# pages/file_maker.py
import streamlit as st
import pandas as pd
import datetime
import io

st.title("📋 AI-Assisted Student Data Sheet Creator")
st.write("Fill in the student data below. You can add up to 60 students. Once done, export it as a CSV file and upload it to the main app.")

# Define the columns required
data_columns = [
    "student_id", "student_name", "class", "internal_assessment", "attendance",
    "previous_gpa", "study_hours", "final_score", "pass_fail", "risk_level", "date_recorded"
]

# Initialize an editable dataframe with empty rows
max_students = 60
empty_data = pd.DataFrame(
    [["" for _ in data_columns] for _ in range(max_students)],
    columns=data_columns
)

# Prefill today's date for the date_recorded column
empty_data["date_recorded"] = [datetime.date.today()] * max_students

# Display editable dataframe
edited_df = st.data_editor(
    empty_data,
    use_container_width=True,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "pass_fail": st.column_config.SelectboxColumn("pass_fail", options=["pass", "fail"]),
        "risk_level": st.column_config.SelectboxColumn("risk_level", options=["Low", "Medium", "High"]),
        "date_recorded": st.column_config.DateColumn("date_recorded")
    },
    key="student_data_editor"
)

# Columns that must not be NaN or blank
required_columns = [
    "student_id", "student_name", "class",
    "internal_assessment", "attendance", "previous_gpa", "study_hours", "final_score"
]

# Remove rows where student_id or student_name are missing
filtered_df = edited_df.dropna(subset=["student_id", "student_name"], how="any")

# Convert numeric fields to proper numbers
for col in ["internal_assessment", "attendance", "previous_gpa", "study_hours", "final_score"]:
    filtered_df[col] = pd.to_numeric(filtered_df[col], errors="coerce")

# Drop rows that have NaNs in any of the required numeric fields
final_df = filtered_df.dropna(subset=required_columns)

if not final_df.empty:
    st.success(f"✅ You have filled in data for {len(final_df)} students.")

    # Convert dataframe to CSV and offer download
    csv_buffer = io.StringIO()
    final_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download CSV File",
        data=csv_buffer.getvalue(),
        file_name="student_data.csv",
        mime="text/csv"
    )

    st.info("After downloading, upload this CSV into your main app directory or merge it with `master.csv`.")
else:
    st.warning("Please fill in at least one complete student's data to enable export.")

