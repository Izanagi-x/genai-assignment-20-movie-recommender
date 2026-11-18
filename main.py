"""Run preprocessing, TF-IDF, all-item cosine similarity and example queries."""
import json
import pandas as pd
from recommender import ROOT, MovieRecommender, build_model, save_model

if __name__ == '__main__':
    raw = pd.read_csv(ROOT / 'tmdb_5000_movies.csv')
    print('Dataset shape:', raw.shape)
    print('Columns:', raw.columns.tolist())
    print('First five rows:\n', raw.head().to_string(index=False))
    print('Data types:\n', raw.dtypes.to_string())
    print('Missing values:\n', raw.isna().sum().to_string())
    print('Duplicate IDs:', raw['id'].duplicated().sum())
    print('Text inputs: overview + genre names + keyword names')
    model = build_model()
    save_model(model)
    model['movies'].to_csv(ROOT / 'prepared_movies.csv', index=False)
    print('TF-IDF matrix:', model['vectors'].shape)
    print('Stored cosine matrix:', model['similarity'].shape)
    recommender = MovieRecommender(model)
    examples = []
    for title in ['Avatar', 'Toy Story', 'Fight Club']:
        result = recommender.recommend(title)
        print('\nQuery:', title)
        print(result[['title', 'similarity']].to_string(index=False))
        result.insert(0, 'query', title)
        examples.append(result)
    pd.concat(examples, ignore_index=True).to_csv(ROOT / 'example_recommendations.csv', index=False)
    details = {'raw_shape': list(raw.shape), 'tfidf_shape': list(model['vectors'].shape),
               'similarity_shape': list(model['similarity'].shape),
               'missing_overviews_replaced': int(raw['overview'].isna().sum()),
               'duplicate_titles_disambiguated': int(raw['title'].duplicated().sum()),
               'max_features': 5000, 'ngram_range': [1, 2], 'min_df': 2}
    (ROOT / 'dataset_summary.json').write_text(json.dumps(details, indent=2) + '\n')
