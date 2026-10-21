import streamlit as st

from utils import hero

hero('📘 Method & About',
     'How the three tracks were built, what each design decision protects against, and how to run everything yourself.')

c1, c2 = st.columns([1.3, 1], gap='large')
with c1:
    st.markdown("""
#### Datasets (UCI Machine Learning Repository)
| Track | Dataset | Rows | Target / goal |
|---|---|---|---|
| Regression | [Bike Sharing (hourly)](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) | 17,379 | `cnt` — rentals per hour |
| Classification | [Adult / Census Income](https://archive.ics.uci.edu/dataset/2/adult) | 48,842 | `income` > $50K (binary, 24% positive) |
| Clustering | [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) | 12,330 | behavioural segments (no label used) |

#### Key design decisions
- **No leakage:** every imputer, encoder and scaler lives *inside* a scikit-learn `Pipeline`, so it is fitted on the training split
  only. `casual`/`registered` were dropped from the bike data because they sum to the target.
- **Fair comparison:** one fixed 80/20 split per track (`random_state=42`, stratified for classification), identical
  preprocessing for every algorithm, and one results table per track.
- **Hyperparameters on training data only:** GridSearchCV / RandomizedSearchCV / cross-validation on the training split; the test
  set is touched once per model. The classification decision threshold is also chosen on out-of-fold training predictions.
- **Feature engineering:** cyclical `hr_sin`/`hr_cos` + `is_rush_hour` (bike), `capital_net` (income), and session volume / depth /
  product-focus features (shoppers).
- **Clustering without labels:** `Revenue` is used only after fitting, to validate that the behavioural segments matter.
""")
with c2:
    st.markdown('#### Algorithms trained')
    st.markdown("""
**Regression (10):** Linear, Ridge, Lasso, ElasticNet, Polynomial, Decision Tree, Random Forest, Gradient Boosting, SVR, KNN

**Classification (10):** Logistic Regression, KNN, Gaussian NB, Decision Tree, SVC *(Part A)* · Random Forest, AdaBoost,
Gradient Boosting, Bagging, MLP *(Part B)*

**Clustering (2):** K-Means (Elbow + Silhouette), Agglomerative (Dendrogram; ward / complete / average / single linkage)
""")
    st.markdown('#### Run it yourself')
    st.code("""git clone <your-repo-url>
cd <repo>
pip install -r requirements.txt
streamlit run app/app.py""", language='bash')
    st.markdown('#### Tech stack')
    st.markdown('Python · pandas · NumPy · scikit-learn · SciPy · Matplotlib · Seaborn · Plotly · Streamlit · joblib')

st.divider()
st.caption('23CSE301 Machine Learning — Capstone Project, Academic Year 2026-27. '
           'The source notebooks (`notebooks/`) contain every step, plot and insight behind the numbers shown in this app.')
