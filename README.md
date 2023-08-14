# Book Recommendation System using Cosine Similarity and TF-IDF

This project implements a simple book recommendation system using Python, pandas, numpy, and scikit-learn. The system utilizes cosine similarity and TF-IDF (Term Frequency-Inverse Document Frequency) vectors to suggest similar books based on their attributes such as title, authors, and language code.

## Features

- **Data Loading and Preprocessing**: Load book data from a CSV file, select relevant columns, and clean the data by removing rows with missing values.

- **TF-IDF Vectorization**: Create TF-IDF matrices for book attributes (title, authors, and language code) to represent the textual information of each book.

- **Cosine Similarity Calculation**: Calculate the cosine similarity between book attributes to determine how similar each book is to others.

- **Recommendation Function**: Provide a function to input a book title and get a list of recommended book titles based on cosine similarity.

## Usage

1. Install the required libraries using `pip install pandas numpy scikit-learn`.

2. Prepare your book data in a CSV file with columns 'title', 'authors', 'average_rating', 'language_code', and 'num_pages'.

3. Update the file path in the script to point to your CSV file.

4. Run the script and follow the prompts to test the recommendation system.

## Note

This is a basic implementation of a book recommendation system. The accuracy and performance can be improved by exploring more advanced techniques and incorporating user feedback.
