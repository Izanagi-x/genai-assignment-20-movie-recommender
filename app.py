"""Simple Streamlit interface for the content-based movie recommender."""
import streamlit as st
from recommender import MovieRecommender, load_model

st.set_page_config(page_title='Find a similar movie', page_icon='🎬')
st.title('Find a similar movie')
st.write('Choose a movie to find five titles with similar descriptions, genres and keywords.')

@st.cache_resource
def get_recommender():
    return MovieRecommender(load_model())

try:
    engine = get_recommender()
    options = engine.movies.sort_values(['title', 'id'])['item_name'].tolist()
    selection = st.selectbox('Movie', options)
    if st.button('Get recommendations', type='primary'):
        results = engine.recommend(selection)
        st.subheader('Recommended movies')
        for position, movie in enumerate(results.itertuples(), 1):
            st.markdown(f'**{position}. {movie.title}** · {str(movie.release_date)[:4]}')
            st.caption(f'{movie.genre_text} · Similarity: {movie.similarity:.3f}')
            st.write(movie.overview or 'No overview available.')
    st.caption(f'{len(engine.movies):,} movies · Content similarity, based on the Kaggle TMDB 5000 dataset.')
except (OSError, ValueError) as exc:
    st.error(f'Unable to load recommendations: {exc}')
