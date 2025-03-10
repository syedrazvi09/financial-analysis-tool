import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import IsolationForest
from utils import load_data

# Streamlit UI
st.set_page_config(page_title="Financial Transactions Analysis", layout="wide")

st.title("📊 Financial Transactions Analysis Tool")
st.markdown("Upload transaction data, analyze trends, and detect anomalies using machine learning.")

# Sidebar for file upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)

    # UI Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Data Analysis", "📸 Media Uploads", "⚙️ Anomaly Detection"])

    with tab1:
        st.subheader("📊 Data Overview")
        st.write("Preview of Uploaded Data", df.head())

        # Automatically detect numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        # Summary Statistics
        st.subheader("📉 Summary Statistics")
        st.write(df[numeric_cols].describe())

        # Allow user to select column for visualization
        selected_column = st.selectbox("🔍 Select a column for analysis", numeric_cols)

        # Histogram of selected column
        st.subheader(f"📊 Distribution of {selected_column}")
        fig, ax = plt.subplots()
        sns.histplot(df[selected_column], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

        # Time-series analysis (only if a time column exists)
        time_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        if time_cols:
            time_col = st.selectbox("🕒 Select a time column", time_cols)
            st.subheader(f"📈 {selected_column} Trends Over Time")
            fig = px.line(df, x=time_col, y=selected_column, title=f"{selected_column} Trends Over Time")
            st.plotly_chart(fig)

    with tab2:
        st.subheader("📸 Media Uploads")
        st.markdown("Upload screenshots or videos related to the analysis.")

        # Upload images
        uploaded_images = st.file_uploader("Upload Screenshots (PNG, JPG)", type=["png", "jpg"],
                                           accept_multiple_files=True)
        if uploaded_images:
            for img in uploaded_images:
                st.image(img, caption=img.name, use_column_width=True)

        # Upload videos
        uploaded_videos = st.file_uploader("Upload Videos (MP4, MOV)", type=["mp4", "mov"], accept_multiple_files=True)
        if uploaded_videos:
            for vid in uploaded_videos:
                st.video(vid)

    with tab3:
        st.subheader("⚙️ Anomaly Detection")
        st.markdown("Use **Isolation Forest** to detect anomalies in transaction data.")

        # Anomaly Detection
        contamination_level = st.slider("Set anomaly contamination level", 0.01, 0.2, 0.05, 0.01)
        model = IsolationForest(n_estimators=100, contamination=contamination_level, random_state=42)
        df['Anomaly'] = model.fit_predict(df[[selected_column]])
        anomalies = df[df['Anomaly'] == -1]

        st.subheader("🔴 Suspicious Transactions")
        st.write(anomalies)

        # Plot anomalies
        if time_cols:
            st.subheader("📉 Anomalies Over Time")
            fig = px.scatter(df, x=time_col, y=selected_column, color=df["Anomaly"].map({1: "Normal", -1: "Anomaly"}))
            st.plotly_chart(fig)
