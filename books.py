import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the books data from CSV
books_df = pd.read_csv(r'C:\Users\wjrtsf\Desktop\test\books.csv')

# Select relevant columns
books_df = books_df[['title', 'authors', 'average_rating', 'language_code', 'num_pages']]

# Drop rows with missing values
books_df.dropna(subset=['title', 'authors', 'average_rating'], inplace=True)

# Create a TF-IDF matrix for book attributes
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(books_df['title'] + ' ' + books_df['authors'] + ' ' + books_df['language_code'])

# Calculate the cosine similarity between book attributes
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get book recommendations based on book title
def get_book_recommendations(book_title, num_recommendations=5):
    try:
        book_index = books_df.index[books_df['title'] == book_title].tolist()[0]
        similar_books = list(enumerate(cosine_sim[book_index]))
        sorted_similar_books = sorted(similar_books, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
        recommended_books = [books_df['title'][i[0]] for i in sorted_similar_books]
        return recommended_books
    except IndexError:
        return []

# Test the recommendation system
print("Enter book name: ")
book_title = input()
recommended_books = get_book_recommendations(book_title)
if recommended_books:
    print(f"Recommended books for '{book_title}':")
    for book in recommended_books:
        print(book)
        #input()
else:
    print(f"'{book_title}' not found in the dataset.")
    input()
input()
