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
from sklearn.feature_extraction.text import TfidfVectorizer
from prophet import Prophet
import warnings
from metrics import MetricsTracker
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

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

# Create tabs
tab1, tab2 = st.tabs(["Numeric Analysis", "Text Analysis"])

# Get the selected tab
selected_tab = "Numeric Analysis" if st.session_state.get('active_tab') is None else st.session_state.get('active_tab')

# Clear sidebar when switching tabs
if selected_tab != st.session_state.get('active_tab'):
    # Clear any existing sidebar elements
    st.sidebar.empty()
    # Reset any tab-specific session state variables
    if 'selected_numeric' in st.session_state:
        del st.session_state.selected_numeric
    if 'selected_categorical' in st.session_state:
        del st.session_state.selected_categorical

with tab1:
    # Update selected tab
    st.session_state.active_tab = "Numeric Analysis"
    
    # File uploader for numeric analysis
    uploaded_file = st.file_uploader("Choose a file for numeric analysis", type=['xlsx', 'csv'], key="numeric_analysis_uploader")

    # Log file upload when a new file is selected
    if uploaded_file is not None and 'last_uploaded_file' not in st.session_state:
        metrics.log_event('file_upload', {'filename': uploaded_file.name})
        st.session_state.last_uploaded_file = uploaded_file.name
    elif uploaded_file is None:
        # Reset the last uploaded file when no file is selected
        if 'last_uploaded_file' in st.session_state:
            del st.session_state.last_uploaded_file
    
    # Main content area
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
                default=numeric_columns[:3] if len(numeric_columns) > 3 else numeric_columns,
                key="numeric_columns"
            )
            
            selected_categorical = st.sidebar.multiselect(
                "Select categorical columns for analysis",
                categorical_columns,
                default=categorical_columns[:2] if len(categorical_columns) > 2 else categorical_columns,
                key="categorical_columns"
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
        st.info("Please upload a file to begin numeric analysis.")

with tab2:
    # Update selected tab
    st.session_state.active_tab = "Text Analysis"
    
    st.header("Text Pattern Analysis")
    
    # File uploader for text analysis
    text_file = st.file_uploader("Choose a file for text analysis", type=['xlsx', 'csv'], key="text_analysis_uploader")

    # Log file upload when a new file is selected
    if text_file is not None and 'last_text_file' not in st.session_state:
        metrics.log_event('file_upload', {'filename': text_file.name})
        st.session_state.last_text_file = text_file.name
    elif text_file is None:
        # Reset the last uploaded file when no file is selected
        if 'last_text_file' in st.session_state:
            del st.session_state.last_text_file

    if text_file is not None:
        try:
            if text_file.name.endswith('.csv'):
                text_df = pd.read_csv(text_file)
            else:
                text_df = pd.read_excel(text_file)
            
            # Display the data
            st.subheader("Data Preview")
            st.dataframe(text_df.head())
            
            # Get text columns
            text_columns = text_df.select_dtypes(include=['object']).columns.tolist()
            
            if not text_columns:
                st.warning("No text columns found in the uploaded file.")
            else:
                # Select text column for analysis
                selected_column = st.selectbox(
                    "Select text column for analysis",
                    text_columns,
                    key="text_column"
                )
                
                # Get text data from selected column
                text_data = text_df[selected_column].dropna().astype(str)
                
                # Analysis buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Analyze Text Patterns"):
                        # Log analysis event
                        metrics.log_event('analysis', {'type': 'text_pattern_analysis'})
                        
                        st.subheader("Text Pattern Analysis Results")
                        
                        # Convert text to numerical features using TF-IDF
                        vectorizer = TfidfVectorizer(max_features=100)
                        X = vectorizer.fit_transform(text_data)
                        
                        # Perform PCA for visualization
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X.toarray())
                        
                        # Perform K-means clustering
                        n_clusters = min(5, len(text_data))
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        clusters = kmeans.fit_predict(X.toarray())
                        
                        # Plot clusters
                        fig, ax = plt.subplots(figsize=(10, 6))
                        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
                        ax.set_title('Text Pattern Clusters')
                        ax.set_xlabel('First Principal Component')
                        ax.set_ylabel('Second Principal Component')
                        plt.colorbar(scatter, label='Cluster')
                        st.pyplot(fig)
                        plt.close()
                        
                        # Show cluster characteristics
                        st.subheader("Cluster Characteristics")
                        for cluster in range(n_clusters):
                            cluster_texts = text_data[clusters == cluster]
                            st.write(f"Cluster {cluster + 1} ({len(cluster_texts)} entries):")
                            st.write("Sample entries:")
                            st.write(cluster_texts.head(3).tolist())
                            st.write("---")
                
                with col2:
                    if st.button("Detect Text Anomalies"):
                        # Log analysis event
                        metrics.log_event('analysis', {'type': 'text_anomaly_detection'})
                        
                        st.subheader("Text Anomaly Detection Results")
                        
                        # Convert text to numerical features
                        vectorizer = TfidfVectorizer(max_features=100)
                        X = vectorizer.fit_transform(text_data)
                        
                        # Use Isolation Forest for anomaly detection
                        iso_forest = IsolationForest(contamination=0.1, random_state=42)
                        anomalies = iso_forest.fit_predict(X.toarray())
                        
                        # Convert predictions to boolean (1 for normal, -1 for anomaly)
                        is_anomaly = anomalies == -1
                        
                        if is_anomaly.any():
                            st.write("Anomalous text entries detected:")
                            anomaly_data = text_data[is_anomaly]
                            st.dataframe(anomaly_data)
                            
                            # Plot anomaly distribution
                            fig, ax = plt.subplots(figsize=(10, 4))
                            anomaly_counts = pd.Series(anomalies).value_counts()
                            anomaly_counts.plot(kind='bar', ax=ax)
                            ax.set_title("Anomaly Distribution")
                            ax.set_xlabel("Anomaly Status (-1: Anomaly, 1: Normal)")
                            ax.set_ylabel("Count")
                            st.pyplot(fig)
                            plt.close()
                        else:
                            st.write("No significant anomalies detected in the text data")
                
                with col3:
                    if st.button("Forecast Text Trends"):
                        # Log analysis event
                        metrics.log_event('analysis', {'type': 'text_trend_forecasting'})
                        
                        st.subheader("Text Trend Forecasting")
                        
                        # Check if there's a date column
                        date_columns = text_df.select_dtypes(include=['datetime64']).columns.tolist()
                        if not date_columns:
                            st.warning("No date column found for trend analysis. Please ensure your data has a date column.")
                        else:
                            date_col = st.selectbox("Select date column", date_columns)
                            
                            # Prepare data for trend analysis
                            text_df['date'] = pd.to_datetime(text_df[date_col])
                            text_df['text_length'] = text_df[selected_column].str.len()
                            
                            # Group by date and calculate metrics
                            daily_metrics = text_df.groupby('date').agg({
                                'text_length': ['mean', 'count']
                            }).reset_index()
                            
                            # Plot text length trend
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ax.plot(daily_metrics['date'], daily_metrics[('text_length', 'mean')], label='Average Text Length')
                            ax.set_title("Text Length Trend Over Time")
                            ax.set_xlabel("Date")
                            ax.set_ylabel("Average Text Length")
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                            plt.close()
                            
                            # Plot text frequency trend
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ax.plot(daily_metrics['date'], daily_metrics[('text_length', 'count')], label='Number of Entries')
                            ax.set_title("Text Entry Frequency Over Time")
                            ax.set_xlabel("Date")
                            ax.set_ylabel("Number of Entries")
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                            plt.close()
                            
                            # Show trend statistics
                            st.write("Trend Statistics:")
                            st.write(f"Average daily entries: {daily_metrics[('text_length', 'count')].mean():.2f}")
                            st.write(f"Average text length: {daily_metrics[('text_length', 'mean')].mean():.2f}")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Please upload a file to begin text analysis.") 