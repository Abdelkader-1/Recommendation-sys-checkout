import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.preprocessing import normalize
import re

app = Flask(__name__)

class ProductRecommendationSystem:
    def __init__(self, model_name, categories_file):
        self.model_name = model_name
        self.categories_file = categories_file
        self.model = SentenceTransformer(self.model_name)
        self.load_categories()

    def load_categories(self):
        try:
            self.categories = pd.read_csv(self.categories_file, encoding='latin1')
            self.categories = self.categories.dropna()
            self.categories['category'] = self.categories['category'].str.lower()
            self.categories['Demographics'] = self.categories['Demographics'].str.lower()
            self.categories['category'] = self.categories['category'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
            self.categories['Demographics'] = self.categories['Demographics'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
            self.categories['category'] = self.categories['category'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
            self.categories['Demographics'] = self.categories['Demographics'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        except FileNotFoundError:
            print(f"Error: File '{self.categories_file}' not found.")
            self.categories = pd.DataFrame()

    def embed_categories(self):
        if hasattr(self, 'categories'):
            self.embedded_categories = self.model.encode(self.categories['Demographics'].astype(str))
            self.embedded_categories = normalize(self.embedded_categories)
        else:
            print("No categories loaded.")

    def embed_query(self, query):
        query = re.sub(r'[^a-zA-Z0-9\s]', '', query)
        query = re.sub(r'\s+', ' ', query).strip().lower()
        category_row = self.categories[self.categories['category'] == query]
        if not category_row.empty:
            demographics = category_row['Demographics'].values[0]
            self.embedded_query = self.model.encode([demographics])
            self.embedded_query = normalize(self.embedded_query).flatten()
        else:
            print(f"Category '{query}' not found in dataset.")
            self.embedded_query = None

    def recommend_categories(self, query, top_n=10):
        self.embed_query(query)
        if self.embedded_query is not None:
            distances = manhattan_distances(self.embedded_categories, self.embedded_query.reshape(1, -1)).flatten()
            sorted_indices = np.argsort(distances)
            recommended_categories = self.categories.iloc[sorted_indices[:top_n]]
            recommended_distances = distances[sorted_indices[:top_n]]
            return recommended_categories['category'].tolist(), recommended_categories['Demographics'].tolist(), recommended_distances.tolist()
        else:
            return [], [], []

recommendation_system = ProductRecommendationSystem('paraphrase-multilingual-MiniLM-L12-v2', 'categories_demographics.csv')
recommendation_system.embed_categories()

@app.route('/embed_query', methods=['POST'])
def embed_query():
    data = request.get_json()
    query = data.get('query', '')
    recommendation_system.embed_query(query)
    if recommendation_system.embedded_query is not None:
        return jsonify({'embedded_query': recommendation_system.embedded_query.tolist()})
    else:
        return jsonify({'error': f"Category '{query}' not found in dataset."}), 404

@app.route('/recommend_categories', methods=['POST'])
def recommend_categories():
    data = request.get_json()
    query = data.get('query', '')
    top_n = data.get('top_n', 10)
    categories, demographics, distances = recommendation_system.recommend_categories(query, top_n)
    return jsonify({
        'categories': categories,
        'demographics': demographics,
        'distances': distances
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
