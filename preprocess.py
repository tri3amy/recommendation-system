import os
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack, csr_matrix, save_npz
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# ============ CONFIGURATION ============
MIN_REVIEWS = 10          # Minimum reviews per product
MAX_TEXT_LENGTH = 2000    # Maximum combined text length per product
TFIDF_MAX_FEATURES = 1000 # TF-IDF dimensions
N_NEIGHBORS = 6           # Number of nearest neighbors (including self)
# =======================================

path_ulasan = os.path.join('data', 'Reviews.csv')
df = pd.read_csv(path_ulasan)
print(f"Initial data shape: {df.shape}")

df = df.dropna(subset=['Summary', 'Text'], how='all')
df['Summary'] = df['Summary'].fillna('')
df['Text'] = df['Text'].fillna('')

df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
df = df.dropna(subset=['Score'])
df = df[(df['Score'] >= 1) & (df['Score'] <= 5)]

df = df[df['HelpfulnessDenominator'] >= 0]
df['HelpfulnessNumerator'] = df[['HelpfulnessNumerator', 'HelpfulnessDenominator']].min(axis=1)

df = df.drop_duplicates(subset='Id')

review_counts = df['ProductId'].value_counts()
valid_products = review_counts[review_counts >= MIN_REVIEWS].index
df = df[df['ProductId'].isin(valid_products)]
df = df.reset_index(drop=True)

print(f"After cleaning: {df.shape}")
print(f"Unique products: {df['ProductId'].nunique()}")

df['review'] = df['Summary'] + ' ' + df['Text']
product_text = df.groupby('ProductId')['review'].apply(
    lambda x: ' '.join(x)[:MAX_TEXT_LENGTH]
).reset_index(name='combined_text')

product_score = df.groupby('ProductId')['Score'].mean().reset_index(name='avg_score')

df['helpfulness_ratio'] = np.where(
    df['HelpfulnessDenominator'] > 0,
    df['HelpfulnessNumerator'] / df['HelpfulnessDenominator'],
    0.0
)
product_helpfulness = df.groupby('ProductId')['helpfulness_ratio'].mean().reset_index(name='avg_helpfulness')

idx_best = df.groupby('ProductId')['HelpfulnessNumerator'].idxmax()
best_reviews = df.loc[idx_best, ['ProductId', 'Summary']].copy()
best_reviews = best_reviews.drop_duplicates(subset='ProductId', keep='first')
best_reviews['Summary'] = best_reviews['Summary'].replace('', 'No summary')
best_reviews = best_reviews.rename(columns={'Summary': 'BestSummary'})


products = product_text.merge(product_score, on='ProductId')
products = products.merge(product_helpfulness, on='ProductId')
products = products.merge(best_reviews, on='ProductId')

print(f"Final number of products: {products.shape[0]}")

stemmer = SnowballStemmer('english')
english_stopwords = stopwords.words('english')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in english_stopwords]
    return ' '.join(words)

products['preprocessed_text'] = products['combined_text'].apply(preprocess_text)

tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
tfidf_matrix = tfidf.fit_transform(products['preprocessed_text'])

numeric_features = products[['avg_score', 'avg_helpfulness']].values
scaler = MinMaxScaler()
normalized_numeric = scaler.fit_transform(numeric_features)

normalized_sparse = csr_matrix(normalized_numeric)
combined_features = hstack((tfidf_matrix, normalized_sparse)).tocsr()

print("Computing nearest neighbors...")
nn = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric='cosine', algorithm='brute')
nn.fit(combined_features)
distances, indices = nn.kneighbors(combined_features)

# Save neighbors
np.savez_compressed('neighbors.npz', indices=indices, distances=distances)

# Save product info (only required columns)
products[['ProductId', 'BestSummary', 'avg_score']].to_csv('products.csv', index=False)

print("Preprocessing completed! Files saved: products.csv and neighbors.npz")