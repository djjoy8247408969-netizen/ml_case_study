"""
ML Capstone — interactive web app (Review 2 bonus: GUI + public deployment).

Run locally from the repository root:
    streamlit run app/app.py
"""
import streamlit as st

from utils import inject_css

st.set_page_config(
    page_title='ML Capstone — Predict · Classify · Segment',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='auto',
)
inject_css()

pages = {
    'Overview': [
        st.Page('views/home.py', title='Home', icon='🏠', default=True),
    ],
    'Live models': [
        st.Page('views/bike.py', title='Bike Demand Forecaster', icon='🚲'),
        st.Page('views/income.py', title='Income Classifier', icon='💼'),
        st.Page('views/shoppers.py', title='Shopper Segmentation', icon='🛒'),
    ],
    'Results': [
        st.Page('views/leaderboard.py', title='Model Leaderboards', icon='🏆'),
        st.Page('views/about.py', title='Method & About', icon='📘'),
    ],
}

with st.sidebar:
    st.markdown('### 🧠 ML Capstone')
    st.caption('23CSE301 · Regression · Classification · Clustering')

nav = st.navigation(pages)

with st.sidebar:
    st.divider()
    st.caption('Models are trained in the project notebooks and loaded here as saved scikit-learn pipelines — '
               'the app never retrains, it only predicts.')

nav.run()
