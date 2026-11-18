"""Content-based recommendations from movie overviews, genres and keywords."""
import gzip
import json
from pathlib import Path
import pickle
import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'recommendation_model.pkl.gz'


def clean_text(value):
    if pd.isna(value):
        return ''
    words = re.sub(r'[^a-z0-9\s]', ' ', str(value).lower()).split()
    return ' '.join(word for word in words if word not in ENGLISH_STOP_WORDS)


def names_from_json(value):
    if pd.isna(value):
        return ''
    return ' '.join(item.get('name', '') for item in json.loads(value))


def build_model(source=ROOT / 'tmdb_5000_movies.csv'):
    raw = pd.read_csv(source)
    movies = raw[['id', 'title', 'release_date', 'overview', 'genres', 'keywords']].copy()
    movies['overview'] = movies['overview'].fillna('')
    movies['genre_text'] = movies['genres'].apply(names_from_json)
    movies['keyword_text'] = movies['keywords'].apply(names_from_json)
    combined = movies['overview'] + ' ' + movies['genre_text'] + ' ' + movies['keyword_text']
    movies['clean_text'] = combined.apply(clean_text)
    years = movies['release_date'].fillna('').str[:4].replace('', 'unknown year')
    movies['item_name'] = movies['title'] + ' (' + years + ') [' + movies['id'].astype(str) + ']'
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    vectors = vectorizer.fit_transform(movies['clean_text']).astype('float32')
    similarity = cosine_similarity(vectors).astype('float16')
    return {'movies': movies, 'vectorizer': vectorizer, 'vectors': vectors,
            'similarity': similarity}


def save_model(model, path=MODEL_PATH):
    with gzip.open(path, 'wb', compresslevel=6) as handle:
        pickle.dump(model, handle, protocol=5)


def load_model(path=MODEL_PATH):
    # Load only the project-generated trusted local artifact.
    if not Path(path).exists():
        model = build_model()
        save_model(model, path)
        return model
    with gzip.open(path, 'rb') as handle:
        return pickle.load(handle)


class MovieRecommender:
    def __init__(self, model):
        self.movies = model['movies']
        self.similarity = model['similarity']

    def recommend(self, item_name, top_n=5):
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
            raise ValueError('top_n must be a positive integer.')
        exact = self.movies.index[self.movies['item_name'] == item_name].tolist()
        if not exact:
            exact = self.movies.index[self.movies['title'].str.casefold() == str(item_name).casefold()].tolist()
        if not exact:
            raise ValueError('Movie not found. Select a movie from the dropdown.')
        if len(exact) > 1:
            raise ValueError('This title has multiple movies. Use its full dropdown label.')
        index = exact[0]
        ranked = sorted(((i, float(score)) for i, score in enumerate(self.similarity[index])
                         if i != index), key=lambda pair: (-pair[1], pair[0]))[:top_n]
        result = self.movies.loc[[i for i, _ in ranked],
                                 ['id', 'title', 'release_date', 'genre_text', 'overview']].copy()
        result['similarity'] = [score for _, score in ranked]
        return result.reset_index(drop=True)


_default_recommender = None


def recommend(item_name, top_n=5):
    global _default_recommender
    if _default_recommender is None:
        _default_recommender = MovieRecommender(load_model())
    return _default_recommender.recommend(item_name, top_n)
