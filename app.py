import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import IsolationForest
from utils import load_data

# Streamlit UI
st.title("Financial Transactions Analysis")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.write("Data Preview", df.head())

    # Automatically detect numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Summary Statistics
    st.write("Summary Statistics", df[numeric_cols].describe())

    # Allow user to select column for visualization
    selected_column = st.selectbox("Select a column for analysis", numeric_cols)

    # Histogram of selected column
    fig, ax = plt.subplots()
    sns.histplot(df[selected_column], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    # Time-series analysis (only if a time column exists)
    time_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    if time_cols:
        time_col = st.selectbox("Select a time column", time_cols)
        fig = px.line(df, x=time_col, y=selected_column, title=f"{selected_column} Trends Over Time")
        st.plotly_chart(fig)

    # Anomaly Detection
    contamination_level = st.slider("Set anomaly contamination level", 0.01, 0.2, 0.05, 0.01)
    model = IsolationForest(n_estimators=100, contamination=contamination_level, random_state=42)
    df['Anomaly'] = model.fit_predict(df[[selected_column]])
    anomalies = df[df['Anomaly'] == -1]

    st.write("Suspicious Transactions", anomalies)
