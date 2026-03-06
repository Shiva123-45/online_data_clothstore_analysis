import pandas as pd
from itertools import combinations
from collections import Counter
import sys

print("🛍️ Starting Phase 3: Recommendation Engine (Market Basket Analysis)\n")

# ==========================================
# 1. LOAD DATA
# ==========================================
try:
    transactions = pd.read_csv('data/transactions.csv')
    products = pd.read_csv('data/products.csv')
except FileNotFoundError:
    print("❌ Error: Data not found. Run generate_data.py first.")
    sys.exit(1)

# Merge to get product names (Categories & Styles) easily
df = pd.merge(transactions, products, on='Product_ID')
df['Product_Name'] = df['Category'] + " (" + df['Style_Tag'] + ")"

print(f"Loaded {len(df)} transactions.")

# ==========================================
# 2. MARKET BASKET ANALYSIS (Apriori Logic)
# ==========================================
# We want to find out what items are frequently bought *by the same customer*
print("\n🔍 Analyzing customer purchase history to find patterns...")

# Group by Customer_ID and get a list of all products they have bought
customer_purchases = df.groupby('Customer_ID')['Product_Name'].apply(list).reset_index()

# For every customer, we find all pairs of items they bought
purchase_pairs = Counter()

for items in customer_purchases['Product_Name']:
    # Remove duplicates within the same customer's history for this analysis
    unique_items = list(set(items)) 
    # Create all possible pairs (combinations of 2) from their purchase list
    if len(unique_items) > 1:
        pairs = combinations(sorted(unique_items), 2)
        purchase_pairs.update(pairs)

print("✅ Analysis Complete!")

# ==========================================
# 3. RECOMMENDATION FUNCTION
# ==========================================
def get_recommendations(target_item, top_n=3):
    """
    Given a Target Item, look through our pairs and find the items
    most commonly bought alongside it.
    """
    recommendations = []
    
    # Search through all pairs for the target_item
    for pair, count in purchase_pairs.most_common():
        if target_item in pair:
            # Get the *other* item in the pair
            other_item = pair[0] if pair[1] == target_item else pair[1]
            recommendations.append((other_item, count))
            
            if len(recommendations) == top_n:
                break
                
    return recommendations

# ==========================================
# 4. TESTING THE ENGINE
# ==========================================
print("\n--- 🛒 TESTING RECOMMENDATION ENGINE ---")

# Let's test it on a few different items
test_items = [
    "Jacket (Streetwear)",
    "Sneakers (Sport)",
    "Dress (Casual)"
]

for item in test_items:
    print(f"\nIf a customer views: [{item}]")
    recs = get_recommendations(item, top_n=3)
    
    if recs:
        print("  AI Recommends 'Frequently Bought Together':")
        for i, (rec_item, count) in enumerate(recs, 1):
            print(f"    {i}. {rec_item} (Bought together {count} times)")
    else:
        print("  No strong recommendations found for this item yet.")

print("\n----------------------------------------")
print("\n✅ Phase 3 Complete! We have built a basic Collaborative Filtering / Market Basket Engine.")
