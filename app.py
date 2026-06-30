import os
import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def load_products():
    path_produk = os.path.join('data', 'products.csv')
    return pd.read_csv(path_produk)

@st.cache_resource
def load_neighbors():
    path_model = os.path.join('models', 'neighbors.npz')
    data = np.load(path_model)
    return data['indices'], data['distances']

products = load_products()
indices, distances = load_neighbors()

def get_recommendations(product_id, product_df, top_n=5):
    """Return a DataFrame of recommended products based on nearest neighbors."""
    try:
        idx = product_df[product_df['ProductId'] == product_id].index[0]
    except IndexError:
        return pd.DataFrame()

    neighbor_indices = indices[idx][1:top_n+1]

    recommendations = product_df.iloc[neighbor_indices][['ProductId', 'BestSummary', 'avg_score']].copy()
    recommendations = recommendations.reset_index(drop=True)
    recommendations.index = range(1, len(recommendations) + 1)
    return recommendations

st.title('🛍️ Product Recommendation System')
st.write('Product recommendations based on review text similarity, rating, and helpfulness.')

def make_label(row):
    summary_preview = str(row['BestSummary'])[:80] + ('...' if len(str(row['BestSummary'])) > 80 else '')
    return f"{row['ProductId']} – {summary_preview}"

product_list = products[['ProductId', 'BestSummary']].drop_duplicates()
options = product_list.apply(make_label, axis=1).tolist()
selected_label = st.selectbox('Select a Product:', options)

if selected_label:
    selected_id = selected_label.split(' – ')[0] 

if st.button('Get Recommendations'):
    if selected_id:
        results = get_recommendations(selected_id, products, top_n=5)
        if not results.empty:
            # Show details of the selected product
            selected_product = products[products['ProductId'] == selected_id].iloc[0]
            st.subheader(f"Recommendations for **{selected_id}**")
            st.caption(f"📝 {selected_product['BestSummary']}")

            st.dataframe(
                results[['ProductId', 'BestSummary', 'avg_score']],
                column_config={
                    'ProductId': 'Product ID',
                    'BestSummary': 'Best Review Summary',
                    'avg_score': st.column_config.NumberColumn('Average Rating', format="%.2f")
                },
                use_container_width=True
            )
        else:
            st.warning("Product not found.")
    else:
        st.warning("Please select a product first.")