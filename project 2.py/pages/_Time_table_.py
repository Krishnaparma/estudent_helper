# import streamlit as st
# import pandas as pd
# import os
# from datetime import datetime
# import matplotlib.pyplot as plt


# if 'play_sound' not in st.session_state:
#     st.session_state['play_sound'] = False

# # --------- Setup ---------
# if 'timetable' not in st.session_state:
#     st.session_state['timetable'] = []
# if 'progress' not in st.session_state:
#     st.session_state['progress'] = []

# st.title("📅 AI Goal-Based Timetable & Progress Tracker")

# # --------- Generate AI Goals ---------
# st.subheader("🎯 Auto-Generate Timetable")
# default_goals = [
#     {"task": "Revise Physics - Ch.1", "deadline": "2025-07-28", "completed": False},
#     {"task": "Complete Math Assignment", "deadline": "2025-07-29", "completed": False},
#     {"task": "Watch Chemistry Video", "deadline": "2025-07-30", "completed": False},
# ]

# if st.button("⚡ Generate AI Timetable"):
#     st.session_state['timetable'] = default_goals.copy()
#     st.success("Timetable generated!")

# # --------- Add New Goal ---------
# st.subheader("➕ Add New Goal")
# with st.expander("Add Goal Manually"):
#      new_task = st.text_input("Enter your goal/task")
#      new_deadline = st.date_input("Deadline", value=datetime.today())
#      if st.button("Add Goal"):
#          if new_task:
#              st.session_state['timetable'].append({
#                  "task": new_task,
#                  "deadline": str(new_deadline),
#                 "completed": False
#             })
#              st.success("Goal added!")
# for i, goal in enumerate(st.session_state.goals):
#     cols = st.columns([6, 2, 2])
#     task = cols[0].text_input("🎯 Goal", value=goal["task"], key=f"task_{i}")
#     completed = cols[1].checkbox("✅ Completed", value=goal["completed"], key=f"completed_{i}")
    
#     if completed and not goal["completed"]:
#         st.balloons()
#         st.toast(f"Great job completing: {task} 🎉")
#         st.session_state["goals"][i]["completed"] = True
#         st.session_state["play_sound"] = True
#     else:
#         st.session_state["goals"][i]["completed"] = completed

# # Sound player appears only after task is completed
# if st.session_state.get("play_sound"):
#     st.markdown("🔊 **Click the button below to celebrate your progress!**")
#     st.audio("assets/success.mp3", format="audio/mp3")
#     st.session_state["play_sound"] = False  # Reset it after showing once

# # --------- Editable Timetable ---------
# if st.session_state['timetable']:
#     st.subheader("📝 Your Editable Timetable")

#     updated_goals = []
#     for i, goal in enumerate(st.session_state['timetable']):
#         st.markdown(f"**Goal {i+1}**")
#         task = st.text_input("Task", goal['task'], key=f"task_{i}")
#         deadline = st.date_input("Deadline", pd.to_datetime(goal['deadline']), key=f"deadline_{i}")
#         completed = st.checkbox("Completed", value=goal['completed'], key=f"completed_{i}")

#         # If newly marked complete
#         if completed and not goal['completed']:
#             st.balloons()
#             st.audio("assets/success.mp3", format="audio/mp3", start_time=0)
#             st.toast(f"Great job completing: {task} 🎉")

#         updated_goals.append({
#             "task": task,
#             "deadline": str(deadline),
#             "completed": completed
#         })
#         st.markdown("---")

#     st.session_state['timetable'] = updated_goals

# # --------- Progress Tracking ---------
# if st.session_state['timetable']:
#     st.subheader("📊 Your Progress Tracker")

#     total = len(st.session_state['timetable'])
#     completed = sum([1 for g in st.session_state['timetable'] if g['completed']])
#     progress_percent = int((completed / total) * 100) if total > 0 else 0

#     st.metric("Total Goals", total)
#     st.metric("Completed", completed)
#     st.progress(progress_percent / 100.0)

#     today = datetime.today().strftime('%Y-%m-%d')
#     if not any(p['date'] == today for p in st.session_state['progress']):
#         st.session_state['progress'].append({
#             "date": today,
#             "progress": progress_percent
#         })

#     # Trend Chart
#     progress_df = pd.DataFrame(st.session_state['progress'])
#     if not progress_df.empty:
#         st.line_chart(progress_df.set_index("date"))

# # --------- Save Timetable ---------
# save_path = f"data/timetable_{st.session_state.get('user_id', 'demo')}.csv"
# if st.button("💾 Save Timetable"):
#     pd.DataFrame(st.session_state['timetable']).to_csv(save_path, index=False)
#     st.success("Timetable saved successfully!")
import streamlit as st
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import os

# Initialize session state for goals
if 'goals' not in st.session_state:
    st.session_state.goals = []

if 'play_sound' not in st.session_state:
    st.session_state.play_sound = False

st.title("🎯 AI Goal-based Study Timetable")
st.write("Generate a smart, editable study timetable and track your progress!")

# Generate a default timetable
if st.button("🧠 Generate AI Goal Timetable"):
    st.session_state.goals = [
        {"task": "Revise Physics - Chapter 1", "completed": False},
        {"task": "Solve Maths - Integration problems", "completed": False},
        {"task": "Read English literature lesson", "completed": False},
    ]
    st.success("Timetable generated! Edit as per your needs.")

# Editable timetable
if st.session_state.goals:
    st.subheader("✏️ Your Study Goals")
    for i, goal in enumerate(st.session_state.goals):
        cols = st.columns([6, 2, 2])
        task = cols[0].text_input("🎯 Goal", value=goal["task"], key=f"task_{i}")
        completed = cols[1].checkbox("✅ Completed", value=goal["completed"], key=f"completed_{i}")

        if completed and not goal["completed"]:
            st.balloons()
            st.toast(f"Great job completing: {task} 🎉")
            st.session_state.goals[i]["completed"] = True
            st.session_state.play_sound = True
        else:
            st.session_state.goals[i]["completed"] = completed

    # Option to add more goals
    if st.button("➕ Add More Goal"):
        st.session_state.goals.append({"task": "", "completed": False})

    # Show audio if task was completed
    if st.session_state.get("play_sound"):
        if os.path.exists("assets/success.mp3"):
            st.markdown("🔊 **Click the button below to celebrate your progress!**")
            st.audio("assets/success.mp3", format="audio/mp3")
        else:
            st.warning("Sound file not found! Please place 'success.mp3' in the 'assets' folder.")
        st.session_state.play_sound = False

    # Show progress chart
    st.subheader("📈 Progress Tracker")
    completed_count = sum([1 for g in st.session_state.goals if g["completed"]])
    total_goals = len(st.session_state.goals)
    progress_percent = (completed_count / total_goals) * 100 if total_goals > 0 else 0

    progress_df = pd.DataFrame({
        'Status': ['Completed', 'Remaining'],
        'Count': [completed_count, total_goals - completed_count]
    })

    fig, ax = plt.subplots()
    ax.pie(progress_df['Count'], labels=progress_df['Status'], autopct='%1.1f%%', startangle=90, colors=['#00C49F', '#FF8042'])
    ax.axis('equal')
    st.pyplot(fig)
else:
    st.info("Click the button above to generate your AI-based timetable.")
