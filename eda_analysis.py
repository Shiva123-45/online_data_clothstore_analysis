import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create an output directory for our charts
os.makedirs('visualizations', exist_ok=True)

print("📊 Starting Phase 2: Exploratory Data Analysis (EDA)...\n")

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading datasets...")
try:
    customers = pd.read_csv('data/customers.csv')
    products = pd.read_csv('data/products.csv')
    transactions = pd.read_csv('data/transactions.csv')
    reviews = pd.read_csv('data/reviews.csv')
except FileNotFoundError:
    print("❌ Error: Data files not found. Please run generate_data.py first.")
    exit(1)

# ==========================================
# 2. DATA CLEANING & PREPROCESSING
# ==========================================
print("Cleaning Data...")

# Convert 'Date' column to proper datetime objects for time-series analysis
transactions['Date'] = pd.to_datetime(transactions['Date'])

# Clean 'Discount_Applied' (e.g., "15%" -> 0.15)
transactions['Discount_Applied'] = transactions['Discount_Applied'].str.replace('%', '', regex=False).astype(float) / 100

# Merge datasets together to create a "Master" view of our sales
# Merge Transactions with Customer info
df = pd.merge(transactions, customers, on='Customer_ID', how='inner')
# Merge the result with Product info
df = pd.merge(df, products, on='Product_ID', how='inner')

# Calculate Total Revenue per transaction (Quantity * Amount Paid)
df['Total_Revenue'] = df['Quantity'] * df['Total_Amount_Paid']

print(f"✅ Data merged successfully! Total merged records: {len(df)}")


# ==========================================
# 3. DATA VISUALIZATION (Generating Charts)
# ==========================================
print("Generating Visualizations...")
sns.set_theme(style="whitegrid")

# --- Chart 1: Top Selling Categories ---
plt.figure(figsize=(10, 6))
cat_sales = df.groupby('Category')['Total_Revenue'].sum().sort_values(ascending=False)
sns.barplot(x=cat_sales.index, y=cat_sales.values, palette="viridis")
plt.title('Total Revenue by Product Category', fontsize=16)
plt.ylabel('Revenue (₹)')
plt.xlabel('Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/1_revenue_by_category.png')
plt.close()

# --- Chart 2: Regional Sales Heatmap (Category vs Region) ---
plt.figure(figsize=(12, 8))
# Create pivot table: Rows = Cities, Columns = Categories, Values = Total quantity sold
city_cat_pivot = df.pivot_table(index='City', columns='Category', values='Quantity', aggfunc='sum')
sns.heatmap(city_cat_pivot, cmap="YlGnBu", annot=True, fmt=".0f")
plt.title('Sales Volume Heatmap: City vs. Product Category', fontsize=16)
plt.ylabel('City')
plt.xlabel('Category')
plt.tight_layout()
plt.savefig('visualizations/2_sales_heatmap.png')
plt.close()

# --- Chart 3: Customer Age Distribution by Persona ---
plt.figure(figsize=(10, 6))
sns.histplot(data=customers, x='Age', hue='Persona', multiple="stack", bins=20, palette="husl")
plt.title('Customer Demographics: Age Distribution by Persona', fontsize=16)
plt.xlabel('Age')
plt.ylabel('Count of Customers')
plt.tight_layout()
plt.savefig('visualizations/3_customer_demographics.png')
plt.close()

# --- Chart 4: Monthly Sales Trend (Time Series) ---
plt.figure(figsize=(12, 6))
# Extract Month-Year for grouping
df['Month'] = df['Date'].dt.to_period('M')
monthly_sales = df.groupby('Month')['Total_Revenue'].sum()
# Plot
monthly_sales.plot(kind='line', marker='o', color='coral', linewidth=2.5)
plt.title('Monthly Revenue Trend (2025)', fontsize=16)
plt.ylabel('Revenue (₹)')
plt.xlabel('Month')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('visualizations/4_monthly_sales_trend.png')
plt.close()


# ==========================================
# 4. BASIC ANALYTICS SUMMARY
# ==========================================
print("\n--- 📈 KEY METRICS SUMMARY ---")
print(f"Total Revenue Generated:   ₹{df['Total_Revenue'].sum():,.2f}")
print(f"Total Items Sold:          {df['Quantity'].sum():,}")
print(f"Average Order Value (AOV): ₹{df['Total_Revenue'].mean():,.2f}")
print(f"Top City by Revenue:       {df.groupby('City')['Total_Revenue'].sum().idxmax()}")
print(f"Top Persona by Revenue:    {df.groupby('Persona')['Total_Revenue'].sum().idxmax()}")
print("------------------------------")

print("\n✅ Phase 2 Complete!")
print("Charts saved to the 'visualizations' folder.")
