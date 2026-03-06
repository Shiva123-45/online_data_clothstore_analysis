import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from collections import Counter
import os

# --- Page Configuration ---
st.set_page_config(page_title="E-Commerce AI Engine", page_icon="🛒", layout="wide")
st.title("🛒 E-Commerce AI Analytics Dashboard")
st.markdown("Welcome to the Ultimate Retail Intelligence Engine. We've combined EDA, Machine Learning, and NLP into one platform.")

# --- Load Data Engine ---
@st.cache_data
def load_data():
    # Helper to check paths (GitHub sometimes flattens folders)
    paths = ['data/', '']
    customers, products, transactions = None, None, None
    
    for p in paths:
        try:
            customers = pd.read_csv(f'{p}customers.csv')
            products = pd.read_csv(f'{p}products.csv')
            transactions = pd.read_csv(f'{p}transactions.csv')
            break # Found them!
        except FileNotFoundError:
            continue
            
    if customers is not None:
        transactions['Date'] = pd.to_datetime(transactions['Date'])
        # Handle discounts
        if transactions['Discount_Applied'].dtype == 'O':
            transactions['Discount_Applied'] = transactions['Discount_Applied'].str.replace('%', '', regex=False).astype(float) / 100
        transactions['Total_Revenue'] = transactions['Quantity'] * transactions['Total_Amount_Paid']
        return customers, products, transactions
    
    st.error("🚨 Data files (customers.csv, etc.) not found! Please ensure they are uploaded to your GitHub repository.")
    return None, None, None

customers, products, transactions = load_data()

if customers is not None:
    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to module:", ["Business Overview (EDA)", "Recommendation Engine", "Customer Segments"])

    # Merge core data
    df = pd.merge(transactions, customers, on='Customer_ID', how='inner')
    df = pd.merge(df, products, on='Product_ID', how='inner')

    # ==========================================
    # PAGE 1: EDA DASHBOARD
    # ==========================================
    if page == "Business Overview (EDA)":
        st.header("📈 Business Overview")
        
        # High level metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"₹{df['Total_Revenue'].sum():,.0f}")
        col2.metric("Total Orders", f"{len(df):,}")
        col3.metric("Avg Order Value", f"₹{df['Total_Revenue'].mean():,.0f}")
        col4.metric("Top City", f"{df.groupby('City')['Total_Revenue'].sum().idxmax()}")
        
        st.markdown("---")
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Revenue by Category")
            fig, ax = plt.subplots(figsize=(8, 5))
            cat_sales = df.groupby('Category')['Total_Revenue'].sum().sort_values(ascending=False)
            sns.barplot(x=cat_sales.index, y=cat_sales.values, palette="viridis", ax=ax)
            ax.set_ylabel('Revenue (₹)')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        with col_chart2:
            st.subheader("Monthly Sales Trend")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            df['Month'] = df['Date'].dt.to_period('M').astype(str)
            monthly_sales = df.groupby('Month')['Total_Revenue'].sum()
            monthly_sales.plot(kind='line', marker='o', color='coral', ax=ax2, linewidth=2)
            ax2.set_ylabel('Revenue (₹)')
            plt.xticks(rotation=45)
            st.pyplot(fig2)


    # ==========================================
    # PAGE 2: RECOMMENDATION ENGINE
    # ==========================================
    elif page == "Recommendation Engine":
        st.header("🧠 AI Recommendation Engine")
        st.markdown("Select a product to see what the Market Basket AI predicts the customer will buy next:")
        
        df['Product_Name'] = df['Category'] + " (" + df['Style_Tag'] + ")"
        all_products = sorted(df['Product_Name'].unique())
        
        selected_product = st.selectbox("Customer is currently viewing:", all_products)
        
        if st.button("Generate Recommendations"):
            with st.spinner('Analyzing 10,000+ purchase patterns...'):
                # Apriori Engine Logic
                customer_purchases = df.groupby('Customer_ID')['Product_Name'].apply(list)
                purchase_pairs = Counter()
                for items in customer_purchases:
                    unique_items = list(set(items))
                    if len(unique_items) > 1:
                        pairs = combinations(sorted(unique_items), 2)
                        purchase_pairs.update(pairs)
                
                recs = []
                for pair, count in purchase_pairs.most_common():
                    if selected_product in pair:
                        other = pair[0] if pair[1] == selected_product else pair[1]
                        recs.append((other, count))
                        if len(recs) == 3: break
                
                if recs:
                    st.success("Frequently Bought Together Analysis Complete!")
                    st.subheader(f"If they buy **{selected_product}**, they are highly likely to also buy:")
                    for i, (rec_item, count) in enumerate(recs, 1):
                        st.info(f"**#{i}: {rec_item}** (Bought together {count} times previously)")
                else:
                    st.warning("Not enough data to form strong pairings for this exact item yet.")


    # ==========================================
    # PAGE 3: CUSTOMER SEGMENTATION
    # ==========================================
    elif page == "Customer Segments":
        st.header("🎯 Customer Segmentation (K-Means)")
        st.markdown("The AI has automatically grouped your customers into 4 distinct behavioral profiles.")
        
        with st.spinner('Running K-Means Clustering on RFM Data...'):
            current_date = transactions['Date'].max()
            rfm = transactions.groupby('Customer_ID').agg({
                'Date': lambda x: (current_date - x.max()).days,
                'Transaction_ID': 'count',
                'Total_Revenue': 'sum',
                'Discount_Applied': 'mean'
            }).reset_index()
            
            rfm.columns = ['Customer_ID', 'Recency', 'Frequency', 'Monetary', 'Avg_Discount']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary', 'Avg_Discount']])
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            rfm['AI_Cluster'] = kmeans.fit_predict(X_scaled)
            
            # Simple naming logic
            cluster_means = rfm.groupby('AI_Cluster').mean()
            names = {}
            for idx, row in cluster_means.iterrows():
                if row['Avg_Discount'] > 0.15: names[idx] = "Deal Seekers"
                elif row['Monetary'] > cluster_means['Monetary'].mean() and row['Frequency'] > cluster_means['Frequency'].mean(): names[idx] = "VIP High Rollers"
                elif row['Recency'] > 150: names[idx] = "At-Risk / Churning"
                else: names[idx] = "Standard Shoppers"
            
            rfm['Profile'] = rfm['AI_Cluster'].map(names)
            
            # Show summary
            st.subheader("Profile Breakdown")
            summary_df = rfm.groupby('Profile').agg({'Customer_ID': 'count', 'Monetary': 'mean', 'Recency':'mean'}).round(0)
            summary_df.columns = ['Total Customers', 'Avg ₹ Spent', 'Avg Days Since Last Sale']
            st.dataframe(summary_df, use_container_width=True)
            
            # 3D Chart 
            st.subheader("AI Clustering Map")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Profile', size='Frequency', sizes=(20, 200), alpha=0.7, palette="Set1", ax=ax)
            ax.set_ylabel('Total Spent (₹)')
            ax.set_xlabel('Recency (Days)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            st.pyplot(fig)
