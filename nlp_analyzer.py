import pandas as pd
from textblob import TextBlob
from collections import Counter
import re
import sys

print("Starting Phase 5: NLP Review Analyzer...\n")

# ==========================================
# 1. LOAD DATA
# ==========================================
try:
    reviews = pd.read_csv('data/reviews.csv')
    products = pd.read_csv('data/products.csv')
except FileNotFoundError:
    print("❌ Error: Data not found. Run generate_data.py first.")
    sys.exit(1)

# Merge to get product categories
df = pd.merge(reviews, products[['Product_ID', 'Category', 'Style_Tag']], on='Product_ID')
print(f"Loaded {len(df)} customer text reviews.")

# ==========================================
# 2. SENTIMENT ANALYSIS
# ==========================================
print("\nAI is reading reviews to calculate Sentiment Scores...")

def calculate_sentiment(text):
    """
    Uses TextBlob NLP to get a polarity score from -1.0 (Very Negative) to 1.0 (Very Positive)
    """
    blob = TextBlob(str(text))
    # Return polarity (-1 to 1) and subjectivity (0 to 1)
    return blob.sentiment.polarity

# Apply NLP to every review
df['Sentiment_Score'] = df['Review_Text'].apply(calculate_sentiment)

# Categorize into human-readable labels
def get_sentiment_label(score):
    if score > 0.1: return "Positive"
    elif score < -0.1: return "Negative"
    else: return "Neutral"

df['Sentiment_Label'] = df['Sentiment_Score'].apply(get_sentiment_label)

print("\n--- GLOBAL SENTIMENT BREAKDOWN ---")
print(df['Sentiment_Label'].value_counts().to_string())

# ==========================================
# 3. KEYWORD EXTRACTION (Finding the "Why")
# ==========================================
print("\nExtracting critical issues from Negative reviews...")

negative_reviews = df[df['Sentiment_Label'] == 'Negative']

# Let's extract common 'complaint' noun phrases or bigrams
# For simplicity in this script, we'll look for common combinations of words
def get_keywords(text_series):
    words = []
    for text in text_series:
        # Simple cleanup
        clean_text = re.sub(r'[^\w\s]', '', str(text).lower())
        tokens = clean_text.split()
        
        # Create bigrams (pairs of 2 words)
        bigrams = [" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
        words.extend(bigrams)
        
    return Counter(words).most_common(5)

# Find top issues specifically for Jackets
jacket_neg_reviews = negative_reviews[negative_reviews['Category'] == 'Jacket']['Review_Text']
print("\nTop Complaints for JACKETS:")
for issue, count in get_keywords(jacket_neg_reviews):
    print(f" - '{issue}' (Mentioned {count} times)")

# Find top issues for Dresses
dress_neg_reviews = negative_reviews[negative_reviews['Category'] == 'Dress']['Review_Text']
print("\nTop Complaints for DRESSES:")
for issue, count in get_keywords(dress_neg_reviews):
    print(f" - '{issue}' (Mentioned {count} times)")

# ==========================================
# 4. REPORTING BY CATEGORY
# ==========================================
print("\n--- SENTIMENT BY CATEGORY ---")
# Let's see which category of clothing is performing the best in customer satisfaction
category_sentiment = df.groupby('Category')['Sentiment_Score'].mean().sort_values(ascending=False)

for cat, score in category_sentiment.items():
    # Convert -1 to 1 score to a 0-100% satisfaction rating roughly
    satisfaction = round(((score + 1) / 2) * 100, 1)
    print(f"{cat.ljust(15)} : {satisfaction}% Avg Satisfaction")

print("\nPhase 5 Complete! The AI has processed all text reviews.")
