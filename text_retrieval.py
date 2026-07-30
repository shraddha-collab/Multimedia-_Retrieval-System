import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("dataset/text/BBC news dataset.csv")

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["description"].fillna(""))

def search(query, top_n=5):
    query_vector = vectorizer.transform([query])
    similarity = cosine_similarity(query_vector, tfidf_matrix).flatten()

    top_indices = similarity.argsort()[-top_n:][::-1]

    print("\nTop Matching Results:\n")

    for i in top_indices:
        print("Description:", df.iloc[i]["description"])
        print("Similarity Score:", round(similarity[i], 4))
        print("-" * 50)

query = input("Enter your search query: ")
search(query)