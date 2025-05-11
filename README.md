# Pattern Recognition Analysis Tool

A user-friendly web application for analyzing patterns and detecting anomalies in manufacturing data using machine learning.

## Features

- Upload Excel or CSV files
- Interactive data visualization
- Machine learning-based pattern analysis
- Anomaly detection using Isolation Forest
- Time series forecasting with Prophet
- Statistical analysis
- Distribution analysis

## Local Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Local Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Cloud Deployment

This application is designed to be deployed on Streamlit Cloud. To deploy:

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app"
5. Select this repository
6. Set the main file path to `app.py`
7. Click "Deploy"

## Data Format

The application works with Excel (.xlsx) or CSV files containing:
- Numeric columns for quantitative analysis
- Categorical columns for grouping and comparison
- Optional date column for time series analysis

## Machine Learning Features

1. **Pattern Analysis**
   - K-means clustering for pattern identification
   - PCA visualization of clusters
   - Cluster statistics and characteristics

2. **Anomaly Detection**
   - Isolation Forest algorithm
   - Contextual anomaly detection
   - Visual anomaly highlighting

3. **Trend Forecasting**
   - Time series forecasting with Prophet
   - Trend decomposition
   - Future predictions with confidence intervals

## Requirements

- Python 3.8+
- See requirements.txt for package dependencies

## License

MIT License

## Author

[Your Name]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 