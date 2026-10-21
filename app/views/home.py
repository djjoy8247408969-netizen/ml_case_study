import streamlit as st

from utils import hero, load_csv, load_json

reg = load_json('regression_meta.json')
clf = load_json('classification_meta.json')
clu = load_json('clustering_meta.json')
clu_res = load_csv('clustering_results.csv')

hero('Predict · Classify · Segment',
     'One end-to-end machine-learning pipeline applied to three real UCI datasets — from raw CSV to tuned, saved models '
     'you can query live below. 22 algorithms trained and compared; the winners power this app.',
     chips=('🚲 Regression · 17,379 hours', '💼 Classification · 48,842 people', '🛒 Clustering · 12,330 sessions',
            'scikit-learn pipelines', 'random_state = 42'))

km_row = clu_res[clu_res['Algorithm'].str.startswith('K-Means')].iloc[0]

c1, c2, c3 = st.columns(3, gap='medium')
with c1:
    st.markdown(f"""<div class="track-card">
        <div class="muted">REGRESSION · UCI Bike Sharing</div>
        <h3>🚲 How many bikes will be rented this hour?</h3>
        <div class="big">R² = {reg['metrics']['R2']:.3f}</div>
        <div class="muted">{reg['model_name']} · RMSE {reg['metrics']['RMSE']:.1f} bikes · MAE {reg['metrics']['MAE']:.1f} bikes<br>
        best of 10 algorithms + 2 tuned variants</div></div>""", unsafe_allow_html=True)
    st.page_link('views/bike.py', label='Open the forecaster', icon='➡️')
with c2:
    m = clf['metrics_default_threshold']
    st.markdown(f"""<div class="track-card">
        <div class="muted">CLASSIFICATION · UCI Adult Census</div>
        <h3>💼 Does this person earn more than $50K?</h3>
        <div class="big">AUC = {m['ROC-AUC']:.3f}</div>
        <div class="muted">{clf['model_name']} · accuracy {m['Accuracy']:.1%} · weighted F1 {m['F1 (weighted)']:.3f}<br>
        best of 10 algorithms, then tuned</div></div>""", unsafe_allow_html=True)
    st.page_link('views/income.py', label='Open the classifier', icon='➡️')
with c3:
    st.markdown(f"""<div class="track-card">
        <div class="muted">CLUSTERING · UCI Online Shoppers</div>
        <h3>🛒 What kind of visitor is this?</h3>
        <div class="big">{clu['k']} segments</div>
        <div class="muted">K-Means · Silhouette {km_row['Silhouette ↑']:.3f} · Davies-Bouldin {km_row['Davies-Bouldin ↓']:.3f}<br>
        conversion ranges from {min(clu['conversion'].values()):.0%} to {max(clu['conversion'].values()):.0%} across segments</div></div>""",
                unsafe_allow_html=True)
    st.page_link('views/shoppers.py', label='Open the segmenter', icon='➡️')

st.markdown('#### The pipeline behind every model')
steps = [('📥', 'Load & audit'), ('🔎', 'EDA'), ('🧹', 'Clean & engineer'), ('⚙️', 'Train-only preprocessing'),
         ('🤖', 'Train all algorithms'), ('📊', 'Compare on one test set'), ('🎛️', 'Tune with CV'), ('💾', 'Save pipeline → app')]
cols = st.columns(len(steps))
for col, (icon, label) in zip(cols, steps):
    col.markdown(f'<div class="step"><span class="n">{icon}</span>{label}</div>', unsafe_allow_html=True)

st.write('')
left, right = st.columns([1.15, 1], gap='large')
with left:
    st.markdown('#### What you can do here')
    st.markdown("""
- **🚲 Bike Demand Forecaster** — pick a date, hour and weather; get the predicted rentals plus a full 24-hour demand curve and a
  temperature what-if.
- **💼 Income Classifier** — describe a person (or load an example profile); see the probability of earning >$50K, how each
  attribute moves it, and score a whole CSV in one go.
- **🛒 Shopper Segmentation** — describe a browsing session; see which of the 4 behavioural segments it falls into, where it lands
  on the PCA map, and the marketing action that segment calls for.
- **🏆 Model Leaderboards** — every algorithm's test-set scores, exactly as produced in the notebooks.
""")
with right:
    st.markdown('#### Leakage-safe by design')
    st.info('Each saved model is a complete scikit-learn **Pipeline** (imputer → encoder → scaler → model). Every preprocessing '
            'step was fitted on the training split only, and the app passes raw inputs through that same fitted pipeline — so '
            'what you see here is exactly what was evaluated in the notebooks.', icon='🛡️')
