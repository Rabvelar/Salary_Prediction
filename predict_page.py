pip install -U scikit-learn
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Check if the CSV file exists
csv_file_path = "jobs_in_data.csv"
if not os.path.exists(csv_file_path):
    st.error(f"CSV file '{csv_file_path}' not found.")
    st.stop()

# Check if the pickled model file exists
model_file_path = "saved_steps.pkl"
if not os.path.exists(model_file_path):
    st.error(f"Pickled model file '{model_file_path}' not found.")
    st.stop()

# Load data from CSV
df = pd.read_csv(csv_file_path)

# Load the trained model and label encoders
def load_model():
    try:
        with open(model_file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        st.error(f"Error loading model or encoders: {e}")
        st.stop()

data = load_model()
regressor_loaded = data.get("model")
le_job_title = data.get("le_job_title")
le_experience_level = data.get("le_experience_level")
le_work_setting = data.get("le_work_setting")
le_company_location = data.get("le_company_location")

# Check if all required objects are loaded
if not all([regressor_loaded, le_job_title, le_experience_level, le_work_setting, le_company_location]):
    st.error("Model or encoders are missing or corrupted.")
    st.stop()

# Define the function to display the prediction page
def show_predict_page():
    st.title("Data Jobs Salary Prediction")
    st.write("### We need some information to predict the salary")

    job_titles = (
        'Data Analyst',
        'Data Scientist',
        'Data Engineer',
        'Machine Learning Engineer',
        'Data Architect',
        'Analytics Engineer',
        'Applied Scientist',
        'Research Scientist',
    )
    experience_levels = (
        'Entry-level',
        'Mid-level',
        'Senior',
        'Executive',
    )
    work_settings = (
        'Office',
        'Hybrid',
        'Remote',
    )
    company_locations = (
        'United States',
        'United Kingdom',
        'Canada',
        'Spain',
        'Germany',
        'France',
        'Netherlands',
        'Portugal',
        'Australia',
        'Other',
    )

    job_title = st.selectbox("Job Title", job_titles)
    experience_level = st.selectbox("Experience Level", experience_levels)
    work_setting = st.selectbox("Work Type", work_settings)
    company_location = st.selectbox("Company Country", company_locations)

    if st.button("Calculate Salary"):
        try:
            x = np.array([[job_title, experience_level, work_setting, company_location]])
            x[:, 0] = le_job_title.transform(x[:, 0])
            x[:, 1] = le_experience_level.transform(x[:, 1])
            x[:, 2] = le_work_setting.transform(x[:, 2])
            x[:, 3] = le_company_location.transform(x[:, 3])
            x = x.astype(float)

            salary = regressor_loaded.predict(x)
            st.subheader(f"The estimated salary is ${salary[0]:.2f}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")

show_predict_page()
