import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load the pre-trained model and data
model = pickle.load(open('ourmodel/knn_modell.pkl', 'rb'))
books_name = pickle.load(open('ourmodel/books_name.pkl', 'rb'))
final_ratings = pickle.load(open('ourmodel/final_ratings.pkl', 'rb'))
book_pivot = pickle.load(open('ourmodel/book_pivot.pkl', 'rb'))

def recommend(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distances, indices = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6)
    
    recommendations = []
    for i in range(1, len(distances.flatten())):  # Skip first result (itself)
        recommendations.append(book_pivot.index[indices.flatten()[i]])
    return recommendations

# Streamlit UI
st.title('Book Recommender System')

selected_book = st.selectbox(
    'Select a book you like:',
    books_name
)

if st.button('Show Recommendations'):
    try:
        recommendations = recommend(selected_book)
        st.subheader(f"Recommended books similar to '{selected_book}':")
        
        # Display recommendations with book details
        for book in recommendations:
            book_details = final_ratings[final_ratings['Title'] == book].iloc[0]
            st.markdown(f"**{book}**")
            st.write(f"Author: {book_details['Author']}")
            st.write(f"Year: {book_details['Year']}")
            st.write(f"Publisher: {book_details['Publisher']}")
            st.image(book_details['Image-URL'], width=150)
            st.write("---")
    except IndexError:
        st.error("Could not generate recommendations for this book. Please try another selection.")