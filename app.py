import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import io
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from prophet import Prophet
import warnings
from metrics import MetricsTracker
warnings.filterwarnings('ignore')

# Initialize metrics tracker
metrics = MetricsTracker()

# Set page config
st.set_page_config(
    page_title="Pattern Recognition Analysis",
    page_icon="📊",
    layout="wide"
)

# Initialize session state for visit tracking
if 'visit_counted' not in st.session_state:
    st.session_state.visit_counted = False
    # Log visit only once per session
    metrics.log_event('page_view')
    st.session_state.visit_counted = True

# Title and description
st.title("Pattern Recognition Analysis")
st.markdown("""
This application uses machine learning to analyze patterns and detect anomalies in your manufacturing data.
Upload your Excel or CSV file to get started.
""")

# Add metrics display in sidebar
st.sidebar.header("Usage Metrics")
if st.sidebar.checkbox("Show Usage Statistics"):
    total_stats = metrics.get_total_stats()
    st.sidebar.metric("Total Site Visits", total_stats['total_views'])
    st.sidebar.metric("Total Analyses", total_stats['total_analyses'])
    st.sidebar.metric("Total Files Uploaded", total_stats['total_uploads'])
    
    # Show daily stats
    st.sidebar.subheader("Last 7 Days")
    daily_stats = metrics.get_daily_stats(7)
    for stat in daily_stats:
        st.sidebar.write(f"**{stat['date']}**")
        st.sidebar.write(f"Visits: {stat['page_views']}")
        st.sidebar.write(f"Analyses: {stat['analyses_performed']}")
        st.sidebar.write(f"Uploads: {stat['files_uploaded']}")
        st.sidebar.write("---")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'csv'])

# Log file upload when a new file is selected
if uploaded_file is not None and 'last_uploaded_file' not in st.session_state:
    metrics.log_event('file_upload', {'filename': uploaded_file.name})
    st.session_state.last_uploaded_file = uploaded_file.name
elif uploaded_file is None:
    # Reset the last uploaded file when no file is selected
    if 'last_uploaded_file' in st.session_state:
        del st.session_state.last_uploaded_file

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Display the data
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Sidebar for analysis options
        st.sidebar.header("Analysis Options")
        
        # Select columns for analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        selected_numeric = st.sidebar.multiselect(
            "Select numeric columns for analysis",
            numeric_columns,
            default=numeric_columns[:3] if len(numeric_columns) > 3 else numeric_columns
        )
        
        selected_categorical = st.sidebar.multiselect(
            "Select categorical columns for analysis",
            categorical_columns,
            default=categorical_columns[:2] if len(categorical_columns) > 2 else categorical_columns
        )
        
        # Analysis buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Analyze Patterns"):
                # Log analysis event
                metrics.log_event('analysis', {'type': 'pattern_analysis'})
                
                st.subheader("Pattern Analysis Results")
                
                if len(selected_numeric) > 1:
                    # Prepare data for clustering
                    X = df[selected_numeric].fillna(df[selected_numeric].mean())
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Perform PCA for visualization
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    # Perform K-means clustering
                    n_clusters = min(5, len(df))  # Limit clusters to 5 or less
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(X_scaled)
                    
                    # Plot clusters
                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
                    ax.set_title('Pattern Clusters (PCA Visualization)')
                    ax.set_xlabel('First Principal Component')
                    ax.set_ylabel('Second Principal Component')
                    plt.colorbar(scatter, label='Cluster')
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show cluster statistics
                    st.subheader("Cluster Statistics")
                    cluster_stats = pd.DataFrame(X, columns=selected_numeric)
                    cluster_stats['Cluster'] = clusters
                    st.write(cluster_stats.groupby('Cluster').mean())
                
                # Show correlation matrix
                if len(selected_numeric) > 1:
                    st.subheader("Correlation Analysis")
                    corr_matrix = df[selected_numeric].corr()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
                    plt.close()
        
        with col2:
            if st.button("Detect Anomalies"):
                # Log analysis event
                metrics.log_event('analysis', {'type': 'anomaly_detection'})
                
                st.subheader("Anomaly Detection Results")
                
                for col in selected_numeric:
                    # Prepare data
                    X = df[[col]].fillna(df[col].mean())
                    
                    # Use Isolation Forest for anomaly detection
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    anomalies = iso_forest.fit_predict(X)
                    
                    # Convert predictions to boolean (1 for normal, -1 for anomaly)
                    is_anomaly = anomalies == -1
                    
                    if is_anomaly.any():
                        st.write(f"Anomalies detected in {col}:")
                        anomaly_data = df[is_anomaly]
                        st.dataframe(anomaly_data)
                        
                        # Plot with anomalies highlighted
                        fig, ax = plt.subplots(figsize=(10, 4))
                        sns.lineplot(data=df, x=df.index, y=col, ax=ax, label='Normal')
                        sns.scatterplot(data=anomaly_data, x=anomaly_data.index, y=col, 
                                      color='red', ax=ax, label='Anomaly')
                        ax.set_title(f"{col} with Detected Anomalies")
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.write(f"No significant anomalies detected in {col}")
        
        with col3:
            if st.button("Forecast Trends"):
                # Log analysis event
                metrics.log_event('analysis', {'type': 'trend_forecasting'})
                
                st.subheader("Trend Forecasting")
                
                for col in selected_numeric:
                    # Prepare data for Prophet
                    if 'Date' in df.columns:
                        date_col = 'Date'
                    else:
                        date_col = df.index.name if df.index.name else 'ds'
                        df[date_col] = df.index
                    
                    prophet_df = df[[date_col, col]].rename(columns={date_col: 'ds', col: 'y'})
                    
                    # Fit Prophet model
                    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
                    model.fit(prophet_df)
                    
                    # Make future predictions
                    future = model.make_future_dataframe(periods=30)
                    forecast = model.predict(future)
                    
                    # Plot forecast
                    fig = model.plot(forecast)
                    plt.title(f"{col} Forecast")
                    st.pyplot(fig)
                    plt.close()
                    
                    # Plot components
                    fig = model.plot_components(forecast)
                    st.pyplot(fig)
                    plt.close()
        
        # Additional analysis options
        st.sidebar.subheader("Advanced Analysis")
        if st.sidebar.checkbox("Show Statistical Summary"):
            st.subheader("Statistical Summary")
            st.write(df[selected_numeric].describe())
        
        if st.sidebar.checkbox("Show Distribution Plots"):
            st.subheader("Distribution Analysis")
            for col in selected_numeric:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(data=df, x=col, ax=ax)
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)
                plt.close()
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        # Don't log the upload if there was an error
else:
    st.info("Please upload a file to begin analysis.") 