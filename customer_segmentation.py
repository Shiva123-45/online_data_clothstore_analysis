import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
import sys

print("🎯 Starting Phase 4: Customer Segmentation (AI Clustering)...\n")
os.makedirs('visualizations', exist_ok=True)

# ==========================================
# 1. LOAD DATA
# ==========================================
try:
    transactions = pd.read_csv('data/transactions.csv')
    customers = pd.read_csv('data/customers.csv')
except FileNotFoundError:
    print("❌ Error: Data not found. Run generate_data.py first.")
    sys.exit(1)

# Clean dates and discounts
transactions['Date'] = pd.to_datetime(transactions['Date'])
if transactions['Discount_Applied'].dtype == 'O': # string object
    transactions['Discount_Applied'] = transactions['Discount_Applied'].str.replace('%', '', regex=False).astype(float) / 100

transactions['Total_Revenue'] = transactions['Quantity'] * transactions['Total_Amount_Paid']

# We simulate today's date as the MAX date in our dataset for "Recency" calculations
current_date = transactions['Date'].max()

# ==========================================
# 2. RFM FEATURE ENGINEERING (Recency, Frequency, Monetary)
# ==========================================
print("Calculating RFM metrics for each customer...")

# Aggregate transaction data to the Customer Level
rfm_df = transactions.groupby('Customer_ID').agg({
    'Date': lambda x: (current_date - x.max()).days, # Recency: Days since last purchase
    'Transaction_ID': 'count',                       # Frequency: Total number of transactions
    'Total_Revenue': 'sum',                          # Monetary: Total money spent
    'Discount_Applied': 'mean'                       # Discount affinity: Average discount they use
}).reset_index()

# Rename columns for clarity
rfm_df.rename(columns={
    'Date': 'Recency',
    'Transaction_ID': 'Frequency',
    'Total_Revenue': 'Monetary',
    'Discount_Applied': 'Avg_Discount_Used'
}, inplace=True)

# Merge back with customer demographic data (Age)
rfm_df = pd.merge(rfm_df, customers[['Customer_ID', 'Age', 'Persona']], on='Customer_ID')
# We keep the "True Persona" just to evaluate how well our AI groups them later!

print(f"✅ RFM Profile built for {len(rfm_df)} customers.\n")

# ==========================================
# 3. K-MEANS CLUSTERING (The AI Model)
# ==========================================
print("🤖 Feeding data to K-Means Clustering Algorithm...")

# Step A: Select the features the AI will learn from
features = ['Recency', 'Frequency', 'Monetary', 'Avg_Discount_Used', 'Age']
X = rfm_df[features]

# Step B: Scale the data. (AI needs Age (e.g. 25) and Monetary (e.g. 50000) on the same scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step C: Train the K-Means Model
# We ask the AI to find 4 distinct groups (clusters) in the data
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm_df['AI_Cluster'] = kmeans.fit_predict(X_scaled)

print("✅ AI Clustering Complete!\n")

# ==========================================
# 4. ANALYZE RESULTS & NAME PROFILES
# ==========================================
print("--- 📊 AI CLUSTER PROFILES ---")

# Let's look at the average stats for each cluster to understand who they are
cluster_summary = rfm_df.groupby('AI_Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'Avg_Discount_Used': 'mean',
    'Age': 'mean',
    'Customer_ID': 'count'
}).round(2)

cluster_summary.rename(columns={'Customer_ID': 'Num_Customers'}, inplace=True)

# We can automatically assign human-readable names based on their Monetary/Discount behavior
cluster_names = {}
for idx, row in cluster_summary.iterrows():
    name = f"Cluster {idx}"
    if row['Avg_Discount_Used'] > 0.15:
        name = "Deal Seekers"
    elif row['Monetary'] > cluster_summary['Monetary'].mean() and row['Frequency'] > cluster_summary['Frequency'].mean():
        name = "VIP High Rollers"
    elif row['Recency'] > 150:
        name = "At-Risk / Churning"
    else:
        name = "Standard Shoppers"
    
    cluster_names[idx] = name
    
rfm_df['Profile_Name'] = rfm_df['AI_Cluster'].map(cluster_names)
cluster_summary.index = cluster_summary.index.map(cluster_names)

print(cluster_summary.to_string())

# ==========================================
# 5. VISUALIZATION
# ==========================================
print("\nSaving 3D Segmentation Chart...")

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=rfm_df, x='Recency', y='Monetary', hue='Profile_Name', 
    size='Frequency', sizes=(20, 200), alpha=0.7, palette="Set1"
)
plt.title('AI Customer Segmentation (Monetary vs Recency)', fontsize=16)
plt.ylabel('Total Spent (₹)')
plt.xlabel('Days Since Last Purchase (Recency)')
# Move legend out of the way
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
plt.savefig('visualizations/5_customer_segments.png')
plt.close()

print("\n✅ Phase 4 Complete! Chart saved to visualizations folder.")
