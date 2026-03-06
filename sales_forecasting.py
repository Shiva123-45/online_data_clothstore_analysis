import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import sys
import os

print("📈 Starting Phase 6: Time-Series Sales Forecasting...\n")

# Ensure the visualizations folder exists
os.makedirs('visualizations', exist_ok=True)

# ==========================================
# 1. LOAD DATA
# ==========================================
try:
    transactions = pd.read_csv('data/transactions.csv')
except FileNotFoundError:
    print("❌ Error: transactions.csv not found. Run generate_data.py first.")
    sys.exit(1)

# Clean dates
transactions['Date'] = pd.to_datetime(transactions['Date'])

# Calculate daily total revenue
transactions['Total_Revenue'] = transactions['Quantity'] * transactions['Total_Amount_Paid']
daily_sales = transactions.groupby('Date')['Total_Revenue'].sum().reset_index()

# Sort by date just to be safe
daily_sales = daily_sales.sort_values('Date').reset_index(drop=True)
daily_sales.set_index('Date', inplace=True)

# If there are any missing days, fill them with 0 sales
# (Our synthetic data is quite dense, but this is good practice in the real world)
full_date_range = pd.date_range(start=daily_sales.index.min(), end=daily_sales.index.max(), freq='D')
daily_sales = daily_sales.reindex(full_date_range, fill_value=0)

print(f"✅ Loaded {len(daily_sales)} days of historical daily sales data.")

# ==========================================
# 2. TRAIN THE FORECASTING MODEL (SARIMA)
# ==========================================
print("\n🤖 AI is analyzing historical trends and seasonality...")

# We use SARIMA (Seasonal AutoRegressive Integrated Moving Average)
# Order: (p=1, d=1, q=1) -> Basic ARIMA
# Seasonal Order: (P=1, D=1, Q=1, m=7) -> We tell the model there is a weekly (7-day) cycle
model = SARIMAX(daily_sales['Total_Revenue'], 
                order=(1, 1, 1), 
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False)

# Train the model
results = model.fit(disp=False)
print("✅ Time-Series Model Trained Successfully!")


# ==========================================
# 3. FORECAST THE FUTURE
# ==========================================
FORECAST_DAYS = 30
print(f"\n🔮 Predicting sales for the next {FORECAST_DAYS} days...")

# Tell the model to predict the next 30 steps (days)
forecast = results.get_forecast(steps=FORECAST_DAYS)
predicted_mean = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# Ensure we don't predict negative sales (a mathematical possibility, but practically impossible)
predicted_mean = predicted_mean.clip(lower=0)
confidence_intervals['lower Total_Revenue'] = confidence_intervals['lower Total_Revenue'].clip(lower=0)

# ==========================================
# 4. VISUALIZE THE PREDICTION
# ==========================================
print("Generating 30-Day Forecast Chart...")

plt.figure(figsize=(14, 7))

# Plot the last 90 days of ACTUAL historical data for context
last_90_days = daily_sales.iloc[-90:]
plt.plot(last_90_days.index, last_90_days['Total_Revenue'], label='Actual Past Sales', color='blue', linewidth=2)

# Plot the PREDICTED future data
future_dates = [daily_sales.index[-1] + timedelta(days=i) for i in range(1, FORECAST_DAYS + 1)]
plt.plot(future_dates, predicted_mean.values, label='AI Forecasted Sales', color='orange', linestyle='--', linewidth=2.5)

# Add the "Confidence Band" (The AI is saying: "I am 95% sure sales will fall in this shaded area")
plt.fill_between(future_dates, 
                 confidence_intervals.iloc[:, 0], # Lower bound
                 confidence_intervals.iloc[:, 1], # Upper bound
                 color='orange', alpha=0.2, label='95% Confidence Interval')

plt.title('30-Day Sales Revenue Forecast (SARIMA Model)', fontsize=18)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Daily Revenue (₹)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the chart
plt.savefig('visualizations/6_sales_forecast.png')
plt.close()

# ==========================================
# 5. SUMMARY OUTPUT
# ==========================================
projected_revenue = predicted_mean.sum()
print("\n--- 30-DAY BUSINESS PROJECTION ---")
print(f"Projected Total Revenue (Next 30 Days): ₹{projected_revenue:,.2f}")
print(f"Projected Best Sales Day: {future_dates[predicted_mean.argmax()].strftime('%Y-%m-%d')} (₹{predicted_mean.max():,.2f})")
print(f"Projected Worst Sales Day: {future_dates[predicted_mean.argmin()].strftime('%Y-%m-%d')} (₹{predicted_mean.min():,.2f})")

print("\n✅ Phase 6 Complete! Forecast chart saved to the visualizations folder.")
