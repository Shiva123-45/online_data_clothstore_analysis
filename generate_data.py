import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Ensure the output directory exists
os.makedirs('data', exist_ok=True)

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

NUM_CUSTOMERS = 2000
NUM_PRODUCTS = 150
NUM_TRANSACTIONS = 10000

print("⏳ Starting E-Commerce AI Engine Data Generation...")

# ==========================================
# 1. GENERATING CUSTOMERS
# ==========================================
print("Generating Customers...")
ages = np.random.randint(18, 70, NUM_CUSTOMERS)
genders = np.random.choice(['Male', 'Female', 'Other'], NUM_CUSTOMERS, p=[0.48, 0.48, 0.04])
cities = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami', 'Seattle', 'Denver', 'Boston', 'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad'], NUM_CUSTOMERS)

personas = []
for age in ages:
    if age < 30:
        personas.append(np.random.choice(['Trend_Setter', 'Bargain_Hunter'], p=[0.6, 0.4]))
    else:
        personas.append(np.random.choice(['Premium_Loyalist', 'Practical_Buyer'], p=[0.7, 0.3]))

customers = pd.DataFrame({
    'Customer_ID': range(1, NUM_CUSTOMERS + 1),
    'Age': ages,
    'Gender': genders,
    'City': cities,
    'Persona': personas # This will be dropped later when we start ML, but needed for generating realistic data
})


# ==========================================
# 2. GENERATING PRODUCTS
# ==========================================
print("Generating Products...")
categories = ['T-Shirt', 'Jeans', 'Jacket', 'Sneakers', 'Dress', 'Accessories', 'Activewear']
style_tags = ['Casual', 'Formal', 'Sport', 'Vintage', 'Streetwear', 'Athleisure']

products_data = []
for i in range(1, NUM_PRODUCTS + 1):
    cat = random.choice(categories)
    style = random.choice(style_tags)
    
    # Logic to make prices realistic in INR (multiply approx by 83)
    base_price = round(random.uniform(1200.0, 8300.0), 2)
    if cat == 'Jacket': base_price = round(random.uniform(6600.0, 20700.0), 2)
    elif cat == 'Sneakers': base_price = round(random.uniform(4150.0, 16600.0), 2)
    elif cat == 'Accessories': base_price = round(random.uniform(830.0, 3700.0), 2)

    # Specific brands / styles cost more
    if style in ['Streetwear', 'Formal']: base_price *= 1.3
    base_price = round(base_price, 2)
    
    products_data.append({
        'Product_ID': i,
        'Category': cat,
        'Style_Tag': style,
        'Base_Price': base_price
    })
products = pd.DataFrame(products_data)


# ==========================================
# 3. GENERATING TRANSACTIONS
# ==========================================
print("Generating Transactions (this might take a few seconds)...")
start_date = datetime(2025, 1, 1)

transactions_data = []
for i in range(1, NUM_TRANSACTIONS + 1):
    customer = customers.iloc[random.randint(0, NUM_CUSTOMERS - 1)]
    product = products.iloc[random.randint(0, NUM_PRODUCTS - 1)]
    
    # How much of a discount are they getting?
    discount_rate = 0.0
    if customer['Persona'] == 'Bargain_Hunter':
        discount_rate = random.choice([0.15, 0.25, 0.40]) # Will only buy if on discount
    elif customer['Persona'] == 'Premium_Loyalist':
        discount_rate = 0.0 # Will pay full price for what they want
    else:
        discount_rate = random.choice([0.0, 0.0, 0.10, 0.20])
        
    final_price = round(product['Base_Price'] * (1 - discount_rate), 2)
    
    # Random date within the year with slight bias toward weekend shopping
    days_offset = random.randint(0, 365)
    tx_date_raw = start_date + timedelta(days=days_offset)
    while tx_date_raw.weekday() < 5 and random.random() < 0.3:
        # 30% chance to reroll if it's not a weekend, biasing weekends
        days_offset = random.randint(0, 365)
        tx_date_raw = start_date + timedelta(days=days_offset)
    
    tx_date = tx_date_raw.strftime('%Y-%m-%d')
    tx_time = f"{random.randint(8, 23):02d}:{random.randint(0, 59):02d}:00"
    
    # Adding Geospatial/Seasonal realism logic
    # E.g., Jackets sell drastically less in Miami and significantly less in Summer everywhere
    if product['Category'] == 'Jacket':
        if customer['City'] in ['Miami', 'Chennai'] and random.random() > 0.05: continue
        if tx_date_raw.month in [6, 7, 8] and customer['City'] not in ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad'] and random.random() > 0.1: continue
        if tx_date_raw.month in [4, 5, 6, 7] and customer['City'] in ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad'] and random.random() > 0.1: continue # Indian summer months
        
    # Floral/Summer dresses sell well in Miami year-round, but only summer in Chicago
    if product['Category'] == 'Dress' and product['Style_Tag'] == 'Casual':
        if customer['City'] in ['Chicago', 'Denver'] and tx_date_raw.month not in [5, 6, 7, 8, 9] and random.random() > 0.2: continue

    transactions_data.append({
        'Transaction_ID': i,
        'Customer_ID': customer['Customer_ID'],
        'Product_ID': product['Product_ID'],
        'Date': tx_date,
        'Time': tx_time,
        'Quantity': random.randint(1, 3), # Buying 1 to 3 of the item (or sizes)
        'Total_Amount_Paid': final_price,
        'Discount_Applied': f"{int(discount_rate * 100)}%"
    })

transactions = pd.DataFrame(transactions_data)
# Fix overlapping IDs from skipped seasonality checks
transactions['Transaction_ID'] = range(1, len(transactions) + 1)


# ==========================================
# 4. GENERATING REVIEWS (For NLP Analysis)
# ==========================================
print("Generating Customer Reviews...")
reviews_data = []

# Text bank to simulate NLP targets
positive_phrases = ["Absolutely love this!", "The fit is perfect.", "Highly recommend this item.", "Great material and super soft.", "Looks exactly like the picture."]
neutral_phrases = ["It's okay.", "Runs a bit small but works.", "Decent for the price.", "Not bad, but shipping took a while.", "Average quality."]
negative_phrases_fit = ["Way too small, I can't even zip it.", "Sizes are completely wrong.", "Shrinks instantly in the wash."]
negative_phrases_quality = ["Zipper broke on the second day.", "Terrible material, feels cheap.", "Colors faded fast.", "Stitching came undone."]

# Sample a random subset of transactions to get a review (approx ~35% write reviews)
reviewed_transactions = transactions.sample(frac=0.35, random_state=42)

for index, row in reviewed_transactions.iterrows():
    product_id = row['Product_ID']
    product_cat = products[products['Product_ID'] == product_id]['Category'].values[0]
    
    # Decide a realistic rating based on discount and category
    base_rating_prob = [0.05, 0.10, 0.15, 0.40, 0.30] # Weights for 1 star, 2 star, 3 star, 4 star, 5 star
    rating = np.random.choice([1, 2, 3, 4, 5], p=base_rating_prob)
    
    # Generate Review Text based on rating
    if rating >= 4:
        text = random.choice(positive_phrases)
    elif rating == 3:
        text = random.choice(neutral_phrases)
    else:
        # If negative, randomly decide if it's a fit issue or quality issue
        if random.random() > 0.5:
            text = random.choice(negative_phrases_fit)
        else:
            text = random.choice(negative_phrases_quality)
            
        # Add some product-specific context sometimes
        if product_cat == 'Jacket' and 'Zipper' not in text and random.random() > 0.5:
            text += " Also, the pockets are weird."
            
    reviews_data.append({
        'Review_ID': len(reviews_data) + 1,
        'Transaction_ID': row['Transaction_ID'],
        'Product_ID': row['Product_ID'],
        'Rating': rating,
        'Review_Text': text
    })

reviews = pd.DataFrame(reviews_data)

# ==========================================
# 5. SAVE TO CSV
# ==========================================
print("Saving datasets...")
customers.to_csv('data/customers.csv', index=False)
products.to_csv('data/products.csv', index=False)
transactions.to_csv('data/transactions.csv', index=False)
reviews.to_csv('data/reviews.csv', index=False)

print("\n✅ SUCCESS! Mock data generation complete.")
print(f"Total Rows Saved:")
print(f" - Customers:    {len(customers):,}")
print(f" - Products:     {len(products):,}")
print(f" - Transactions: {len(transactions):,}")
print(f" - Reviews:      {len(reviews):,}")
print("\nYou are now ready for Phase 2!")
