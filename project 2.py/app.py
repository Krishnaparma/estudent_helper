import os
import json
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import time
from datetime import datetime


# CONFIGURATION & SIMPLIFIED AUTHENTICATION

st.set_page_config(
    page_title="Academic Performance AI Suite",
    layout="wide",
    page_icon="🎓"
)

# Simple session-based authentication (no external dependencies)
def init_auth():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'name' not in st.session_state:
        st.session_state.name = None

def login_page():
    st.title("🎓 Academic Performance AI Suite")
    st.subheader("Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login", use_container_width=True):
            # Simple authentication (replace with real auth in production)
            valid_users = {
                "teacher": {"name": "Mr.dhan raj", "role": "Teacher"},
                "student": {"name": "krishna maurya", "role": "Student"},
                "admin": {"name": "Admin", "role": "Admin"}
            }
            
            if username in valid_users and password == "demo123":
                st.session_state.authenticated = True
                st.session_state.role = valid_users[username]["role"]
                st.session_state.name = valid_users[username]["name"]
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Use: teacher/student/admin with password: demo123")
        
        st.info("💡 Demo credentials: teacher/student/admin, password: demo123")


# 1. DATA LOADING & GENERATION

@st.cache_data
def generate_sample_data():
    """Generate sample student data if no file is uploaded"""
    np.random.seed(42)
    n_students = 150
    
    data = []
    for i in range(n_students):
        student_id = f"STU{i+1:03d}"
        student_name = f"Student {i+1}"
        class_name = np.random.choice(['A', 'B', 'C', 'D'])
        
        # Generate correlated data
        internal_assessment = np.random.normal(75, 12)
        attendance = np.random.normal(85, 10)
        previous_gpa = np.random.normal(3.0, 0.4)
        study_hours = np.random.normal(15, 4)
        
        # Clip values to realistic ranges
        internal_assessment = np.clip(internal_assessment, 0, 100)
        attendance = np.clip(attendance, 50, 100)
        previous_gpa = np.clip(previous_gpa, 1.0, 4.0)
        study_hours = np.clip(study_hours, 5, 40)
        
        # Calculate final score with some noise
        final_score = (
            internal_assessment * 0.4 +
            attendance * 0.3 +
            previous_gpa * 50 * 0.2 +
            study_hours * 0.1 +
            np.random.normal(0, 3)
        )
        final_score = np.clip(final_score, 0, 100)
        
        # Determine pass/fail and risk level
        pass_fail = "Pass" if final_score >= 60 else "Fail"
        if final_score >= 80:
            risk_level = "Low Risk"
        elif final_score >= 60:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
        
        data.append({
            'student_id': student_id,
            'student_name': student_name,
            'class': class_name,
            'internal_assessment': round(internal_assessment, 1),
            'attendance': round(attendance, 1),
            'previous_gpa': round(previous_gpa, 2),
            'study_hours': round(study_hours, 1),
            'final_score': round(final_score, 1),
            'pass_fail': pass_fail,
            'risk_level': risk_level,
            'date_recorded': datetime.now().strftime('%Y-%m-%d')
        })
    
    return pd.DataFrame(data)

def load_data():
    """Load data from uploaded file or generate sample data"""
    st.sidebar.title("📂 Data Management")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV/Excel file", 
        type=['csv', 'xlsx'],
        help="Upload your student data file"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.sidebar.success("✅ File uploaded successfully!")
            return df
        except Exception as e:
            st.sidebar.error(f"❌ Error loading file: {str(e)}")
            return generate_sample_data()
    else:
        st.sidebar.info("📋 Using demo data. Upload your file to override.")
        return generate_sample_data()


# MODEL TRAINING

@st.cache_resource
def train_model(X, y, task_type="regression"):
    """Train machine learning model"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        if task_type == "regression":
            model = GradientBoostingRegressor(random_state=42, n_estimators=100)
        else:
            model = GradientBoostingClassifier(random_state=42, n_estimators=100)
        
        model.fit(X_train, y_train)
        
        # Calculate metrics
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        if task_type == "regression":
            train_score = r2_score(y_train, train_pred)
            test_score = r2_score(y_test, test_pred)
        else:
            train_score = accuracy_score(y_train, train_pred)
            test_score = accuracy_score(y_test, test_pred)
        
        return model, train_score, test_score, X_train, X_test, y_train, y_test
    except Exception as e:
        st.error(f"Model training error: {str(e)}")
        return None, 0, 0, None, None, None, None


# VISUALIZATION FUNCTIONS

def create_performance_overview(df):
    """Create performance overview charts"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Score distribution
        fig = px.histogram(
            df, x='final_score', 
            title="Final Score Distribution",
            nbins=20,
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk level distribution
        risk_counts = df['risk_level'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Level Distribution",
            color_discrete_map={
                'Low Risk': '#28a745',
                'Medium Risk': '#ffc107',
                'High Risk': '#dc3545'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        correlation_matrix,
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)

def create_feature_importance_chart(model, feature_names):
    """Create feature importance chart"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(
            importance_df, 
            x='importance', 
            y='feature',
            orientation='h',
            title="Feature Importance",
            color='importance',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        return importance_df
    return None


# PDF REPORT GENERATION

def generate_student_report(student_data, prediction=None):
    """Generate PDF report for a student"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Student Performance Report", 0, 1, "C")
        
        pdf.set_font("Arial", "", 12)
        pdf.ln(10)
        
        # Student information
        for key, value in student_data.items():
            if key != 'date_recorded':
                pdf.cell(0, 8, f"{key.replace('_', ' ').title()}: {value}", 0, 1)
        
        if prediction is not None:
            pdf.ln(5)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, f"Predicted Score: {prediction:.1f}", 0, 1)
        
        pdf.ln(5)
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 8, f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
        
        return pdf.output(dest='S').encode('latin-1')
    except Exception as e:
        st.error(f"PDF generation error: {str(e)}")
        return None


# MAIN APPLICATION

def main_app():
    """Main application logic"""
    # Header
    st.title("🎓 Academic Performance AI Suite")
    st.subheader(f"Welcome {st.session_state.name} ({st.session_state.role})")
    
    # Load data
    df = load_data()
    
    if df is None or df.empty:
        st.error("No data available. Please upload a valid file.")
        return
    
    # Data overview
    with st.expander("📊 Data Overview", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Students", len(df))
        with col2:
            st.metric("Average Score", f"{df['final_score'].mean():.1f}")
        with col3:
            pass_rate = (df['pass_fail'] == 'Pass').mean() * 100
            st.metric("Pass Rate", f"{pass_rate:.1f}%")
        with col4:
            high_risk = (df['risk_level'] == 'High Risk').sum()
            st.metric("High Risk Students", high_risk)
        
        st.dataframe(df.head(10), use_container_width=True)
    
    # Performance overview
    st.header("📈 Performance Overview")
    create_performance_overview(df)
    
    # Model training and prediction
    st.header("🤖 Predictive Analytics")
    
    # Prepare data for modeling
    try:
        # Select numeric features for modeling
        feature_cols = ['internal_assessment', 'attendance', 'previous_gpa', 'study_hours']
        available_features = [col for col in feature_cols if col in df.columns]
        
        if len(available_features) < 2:
            st.warning("Insufficient numeric features for modeling.")
            return
        
        X = df[available_features]
        y = df['final_score']
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model, train_score, test_score, X_train, X_test, y_train, y_test = train_model(
            X_scaled, y, "regression"
        )
        
        if model is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training R²", f"{train_score:.3f}")
            with col2:
                st.metric("Testing R²", f"{test_score:.3f}")
            with col3:
                st.metric("Model Status", "✅ Ready")
            
            # Feature importance
            st.subheader("🎯 Feature Importance")
            importance_df = create_feature_importance_chart(model, available_features)
            
            # Correlation analysis
            st.subheader("🔗 Feature Correlations")
            create_correlation_heatmap(df[available_features + ['final_score']])
            
            # Individual prediction
            st.subheader("🔮 Individual Prediction")
            
            with st.form("prediction_form"):
                st.write("Enter student information for prediction:")
                
                col1, col2 = st.columns(2)
                with col1:
                    internal_score = st.slider("Internal Assessment", 0, 100, 75)
                    attendance_pct = st.slider("Attendance %", 50, 100, 85)
                with col2:
                    gpa = st.slider("Previous GPA", 1.0, 4.0, 3.0, 0.1)
                    study_hrs = st.slider("Study Hours/Week", 5, 40, 15)
                
                predict_button = st.form_submit_button("🎯 Predict Performance")
                
                if predict_button:
                    # Make prediction
                    input_data = np.array([[internal_score, attendance_pct, gpa, study_hrs]])
                    input_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_scaled)[0]
                    
                    # Display prediction
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Score", f"{prediction:.1f}")
                    with col2:
                        pred_status = "Pass" if prediction >= 60 else "Fail"
                        color = "normal" if pred_status == "Pass" else "inverse"
                        st.metric("Predicted Result", pred_status, delta_color=color)
                    with col3:
                        if prediction >= 80:
                            risk = "Low Risk"
                            risk_color = "normal"
                        elif prediction >= 60:
                            risk = "Medium Risk" 
                            risk_color = "normal"
                        else:
                            risk = "High Risk"
                            risk_color = "inverse"
                        st.metric("Risk Level", risk, delta_color=risk_color)
                    
                    # Recommendations
                    if prediction < 70:
                        st.subheader("📋 Recommendations")
                        recommendations = []
                        
                        if internal_score < 70:
                            recommendations.append("💡 Focus on improving internal assessment scores through regular practice")
                        if attendance_pct < 80:
                            recommendations.append("📅 Improve attendance - aim for 90%+ attendance rate")
                        if gpa < 2.5:
                            recommendations.append("📚 Strengthen foundational knowledge through additional study")
                        if study_hrs < 10:
                            recommendations.append("⏰ Increase study time - recommend 15+ hours per week")
                        
                        if not recommendations:
                            recommendations.append("👍 Keep up the good work! Focus on consistency")
                        
                        for rec in recommendations:
                            st.write(rec)
    
    except Exception as e:
        st.error(f"Error in predictive analytics: {str(e)}")
    
    # Role-specific features
    if st.session_state.role == "Teacher":
        st.header("👩‍🏫 Teacher Tools")
        
        with st.expander("📄 Generate Student Reports"):
            selected_student = st.selectbox(
                "Select Student", 
                df['student_id'].tolist()
            )
            
            if st.button("Generate PDF Report"):
                student_data = df[df['student_id'] == selected_student].iloc[0].to_dict()
                pdf_bytes = generate_student_report(student_data)
                
                if pdf_bytes:
                    st.download_button(
                        "📥 Download Report",
                        data=pdf_bytes,
                        file_name=f"{selected_student}_report.pdf",
                        mime="application/pdf"
                    )
    
    elif st.session_state.role == "Student":
        st.header("🧑‍🎓 Student Portal")
        
        student_id = st.selectbox("Select Your ID", df['student_id'].tolist())
        if student_id:
            student_data = df[df['student_id'] == student_id].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Your Score", f"{student_data['final_score']:.1f}")
            with col2:
                st.metric("Class Average", f"{df['final_score'].mean():.1f}")
            with col3:
                percentile = (df['final_score'] < student_data['final_score']).mean() * 100
                st.metric("Your Percentile", f"{percentile:.0f}%")
    
    elif st.session_state.role == "Admin":
        st.header("⚙️ Admin Dashboard")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Export Data"):
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    data=csv,
                    file_name="student_data_export.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()
    
    # Logout
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.role = None
        st.session_state.name = None
        st.rerun()


# RUN APPLICATION

def main():
    """Main function to run the application"""
    init_auth()
    
    if not st.session_state.authenticated:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()