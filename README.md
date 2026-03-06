# E-Commerce AI Engine & Analytics Dashboard 🛒🚀

An end-to-end Data Science and Artificial Intelligence portfolio project. This repository contains the code to generate a realistic e-commerce dataset and a suite of Machine Learning modules (Recommendation Systems, Customer Segmentation, Time-Series Forecasting, and NLP Sentiment Analysis), all bundled into an interactive Streamlit Web Dashboard.

## 🌟 Modules Included

1. **Synthetic Data Generator** (`generate_data.py`)
   - Generates 10,000+ realistic transactions, 2,000 customers with hidden personas, 150 products, and 3,000+ text reviews, complete with geospatial and seasonal buying patterns.

2. **Exploratory Data Analysis (EDA)** (`eda_analysis.py`)
   - Cleans the raw data and generates baseline business metrics and visualizations (Heatmaps, Time-Series charts).

3. **Recommendation Engine** (`recommendation_engine.py`)
   - An Apriori/Market Basket Analysis algorithm utilizing Collaborative Filtering to recommend "Frequently Bought Together" items.

4. **Customer Segmentation** (`customer_segmentation.py`)
   - A K-Means Clustering Machine Learning model that groups customers into Profiles (e.g., VIP High Rollers, Deal Seekers) based on their RFM (Recency, Frequency, Monetary) behavior.

5. **NLP Review Analyzer** (`nlp_analyzer.py`)
   - Uses the TextBlob library to conduct Sentiment Analysis on text reviews and extract key phrases (e.g., "zipper broke") to understand customer satisfaction at scale.

6. **Time-Series Sales Forecasting** (`sales_forecasting.py`)
   - Uses the SARIMA (Seasonal AutoRegressive Integrated Moving Average) model to predict the next 30 days of daily revenue with a 95% Confidence Interval.

7. **Streamlit Web Dashboard** (`app.py`)
   - A fully interactive web application combining the EDA, Recommendation Engine, and Customer Segmentation into an easy-to-use GUI.

## 🛠️ Tech Stack
- **Python** (Pandas, NumPy)
- **Machine Learning**: Scikit-Learn (K-Means), Statsmodels (SARIMA)
- **Natural Language Processing**: TextBlob
- **Data Visualization**: Matplotlib, Seaborn
- **Frontend / Deployment**: Streamlit

## 🚀 How to Run Locally

1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit Dashboard:
   ```bash
   streamlit run app.py
   ```
   Navigate to `http://localhost:8501` to view.

## 📈 Deployment
This application is designed to be easily deployed on [Streamlit Community Cloud](https://streamlit.io/cloud). Simply connect your GitHub repository to Streamlit and select `app.py` as the main file.
