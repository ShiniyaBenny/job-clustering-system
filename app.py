import streamlit as st
import pandas as pd
import joblib

# Load the data and model
jobs_df = pd.read_csv('jobs_with_clusters.csv')
model = joblib.load('kmeans_model.pkl')

st.title("Job Posting Classification Based on Skills")

st.subheader("All Jobs with Cluster Labels")
st.write(jobs_df)

# User inputs skills
user_skills = st.text_input("Enter your skills (comma-separated):")

if user_skills:
    user_skills_list = [skill.strip().lower() for skill in user_skills.split(',')]

    # Filter jobs where any user skill matches job skills
    def match_skills(row):
        skills = row['skills'].lower()
        return any(skill in skills for skill in user_skills_list)

    filtered_jobs = jobs_df[jobs_df.apply(match_skills, axis=1)]

    st.subheader("Jobs Matching Your Skills")
    if not filtered_jobs.empty:
        st.write(filtered_jobs)
    else:
        st.write("No matching jobs found for the entered skills.")
