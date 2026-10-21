import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (ACCENT, BAD, GOOD, INCOME_INPUT_COLS, MUTED, PLOTLY_LAYOUT, PRIMARY, hero, income_frame, load_json,
                   load_model, pill, result_card)

meta = load_json('classification_meta.json')
model = load_model('income_classifier.joblib')
opts = meta['category_options']
edu_map = {int(k): v for k, v in meta['education_map'].items()}
edu_names = [edu_map[k] for k in sorted(edu_map)]
edu_to_num = {v: k for k, v in edu_map.items()}
m_def, m_thr = meta['metrics_default_threshold'], meta['metrics_tuned_threshold']

hero('💼 Income Classifier',
     'Estimate the probability that a U.S. adult earns more than $50K a year from census attributes (UCI Adult, 1994 census). '
     f"Model: <b>{meta['model_name']}</b> — ROC-AUC {m_def['ROC-AUC']:.3f}, accuracy {m_def['Accuracy']:.1%} on "
     f"{meta['n_test']:,} unseen people.",
     chips=('10 algorithms compared', 'RandomizedSearchCV-tuned', 'CV-optimised decision threshold', 'batch scoring'))

# ------------------------------------------------------------------ example profiles
PRESETS = {
    '👩‍💼 Senior manager': dict(age=46, workclass='Private', education='Masters', **{'marital-status': 'Married-civ-spouse'},
                               occupation='Exec-managerial', relationship='Wife', race='White', sex='Female',
                               **{'capital-gain': 7688, 'capital-loss': 0, 'hours-per-week': 50, 'native-country': 'United-States'}),
    '🧑‍🔧 Mid-career technician': dict(age=38, workclass='Private', education='Some-college', **{'marital-status': 'Married-civ-spouse'},
                                     occupation='Craft-repair', relationship='Husband', race='Black', sex='Male',
                                     **{'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 40, 'native-country': 'United-States'}),
    '🎓 Young graduate': dict(age=24, workclass='Private', education='Bachelors', **{'marital-status': 'Never-married'},
                             occupation='Prof-specialty', relationship='Own-child', race='Asian-Pac-Islander', sex='Female',
                             **{'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 38, 'native-country': 'India'}),
    '🧑‍🏫 Married professor': dict(age=52, workclass='State-gov', education='Doctorate', **{'marital-status': 'Married-civ-spouse'},
                                  occupation='Prof-specialty', relationship='Husband', race='White', sex='Male',
                                  **{'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 45, 'native-country': 'United-States'}),
}
DEFAULT = PRESETS['🧑‍🔧 Mid-career technician']


def apply_preset(p):
    for k, v in p.items():
        st.session_state[f'in_{k}'] = v


if 'in_age' not in st.session_state:
    apply_preset(DEFAULT)

st.markdown('**Try an example profile:**')
pcols = st.columns(len(PRESETS))
for col, (name, p) in zip(pcols, PRESETS.items()):
    col.button(name, on_click=apply_preset, args=(p,), width='stretch')

tab_single, tab_batch = st.tabs(['👤 Single prediction', '📄 Batch scoring (CSV)'])

with tab_single:
    with st.container(border=True):
        a1, a2, a3, a4 = st.columns(4)
        age = a1.slider('Age', 17, 90, key='in_age')
        sex = a2.selectbox('Sex', opts['sex'], key='in_sex')
        race = a3.selectbox('Race', opts['race'], key='in_race')
        country = a4.selectbox('Native country', opts['native-country'], key='in_native-country')

        b1, b2, b3, b4 = st.columns(4)
        education = b1.selectbox('Education', edu_names, key='in_education')
        marital = b2.selectbox('Marital status', opts['marital-status'], key='in_marital-status')
        relationship = b3.selectbox('Relationship in household', opts['relationship'], key='in_relationship')
        hours = b4.slider('Hours worked per week', 1, 99, key='in_hours-per-week')

        c1, c2, c3, c4 = st.columns(4)
        workclass = c1.selectbox('Work class', opts['workclass'], key='in_workclass')
        occupation = c2.selectbox('Occupation', opts['occupation'], key='in_occupation')
        gain = c3.number_input('Capital gain ($/yr)', 0, 99999, step=500, key='in_capital-gain')
        loss = c4.number_input('Capital loss ($/yr)', 0, 4356, step=100, key='in_capital-loss')

    def make_row(**over):
        base = {'age': age, 'workclass': workclass, 'education-num': edu_to_num[education], 'marital-status': marital,
                'occupation': occupation, 'relationship': relationship, 'race': race, 'sex': sex,
                'capital-gain': gain, 'capital-loss': loss, 'hours-per-week': hours, 'native-country': country}
        base.update(over)
        return base

    X_one = income_frame(pd.DataFrame([make_row()]), meta)
    proba = float(model.predict_proba(X_one)[0, 1])

    use_tuned = st.toggle(f"Use the CV-optimised decision threshold ({meta['threshold']:.2f}) instead of 0.50", value=True,
                          help='Chosen in the notebook on training-set cross-validation to maximise F1 for the >50K class. '
                               f"F1 for the >50K class: {m_def['F1 (>50K class)']:.3f} at 0.50 → {m_thr['F1 (>50K class)']:.3f} F1 at "
                               f"{meta['threshold']:.2f}.")
    thr = meta['threshold'] if use_tuned else 0.5
    is_high = proba >= thr

    r1, r2 = st.columns([1, 1.25], gap='large')
    with r1:
        verdict = 'Earns > $50K' if is_high else 'Earns ≤ $50K'
        result_card('Prediction', verdict,
                    f'Probability of >$50K: <b>{proba:.1%}</b> · threshold {thr:.2f}<br>'
                    + pill('HIGH INCOME' if is_high else 'LOWER INCOME', GOOD if is_high else MUTED),
                    color=GOOD if is_high else PRIMARY)
        fig = go.Figure(go.Indicator(
            mode='gauge+number', value=proba * 100, number={'suffix': '%', 'font': {'size': 40}},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': GOOD if is_high else PRIMARY, 'thickness': .3},
                   'steps': [{'range': [0, thr * 100], 'color': 'rgba(100,116,139,.12)'},
                             {'range': [thr * 100, 100], 'color': 'rgba(5,150,105,.12)'}],
                   'threshold': {'line': {'color': BAD, 'width': 4}, 'thickness': .9, 'value': thr * 100}}))
        fig.update_layout(**{**PLOTLY_LAYOUT, 'margin': dict(l=35, r=45, t=50, b=0)}, height=250,
                          title='P(income > $50K) — red line = decision threshold')
        st.plotly_chart(fig)

    with r2:
        st.markdown('##### What would change the prediction?')
        st.caption('Each curve varies **one** attribute while everything else stays as entered (model-agnostic what-if analysis).')
        wt1, wt2, wt3 = st.tabs(['Age', 'Hours / week', 'Education'])

        def whatif(values, col, title, xlab, current, ticktext=None):
            rows = income_frame(pd.DataFrame([make_row(**{col: v}) for v in values]), meta)
            p = model.predict_proba(rows)[:, 1]
            f = go.Figure(go.Scatter(x=list(range(len(values))) if ticktext else values, y=p * 100, mode='lines+markers',
                                     line=dict(color=ACCENT, width=3), marker=dict(size=5)))
            f.add_hline(y=thr * 100, line=dict(color=BAD, dash='dash'), annotation_text='threshold')
            cx = values.index(current) if ticktext else current
            f.add_trace(go.Scatter(x=[cx], y=[proba * 100], mode='markers', marker=dict(size=14, color=PRIMARY,
                                   line=dict(color='white', width=2)), name='You'))
            f.update_layout(**PLOTLY_LAYOUT, height=300, title=title, showlegend=False,
                            yaxis=dict(title='P(>50K) %', range=[0, 100]), xaxis=dict(title=xlab))
            if ticktext:
                f.update_xaxes(tickmode='array', tickvals=list(range(len(values))), ticktext=ticktext, tickangle=-40)
            st.plotly_chart(f)

        with wt1:
            whatif(list(range(18, 81, 2)), 'age', 'Probability vs. age', 'Age', age - age % 2 if age >= 18 else 18)
        with wt2:
            whatif(list(range(5, 81, 5)), 'hours-per-week', 'Probability vs. weekly hours', 'Hours per week',
                   min(max(5, int(round(hours / 5) * 5)), 80))
        with wt3:
            nums = sorted(edu_map)
            whatif(nums, 'education-num', 'Probability vs. education level', '', edu_to_num[education],
                   ticktext=[edu_map[n] for n in nums])

    with st.expander('🔍 What does the model rely on overall? (global feature importance)'):
        imp = pd.DataFrame(meta['feature_importance']).sort_values('importance')
        f = go.Figure(go.Bar(x=imp['importance'], y=imp['feature'], orientation='h', marker_color=PRIMARY))
        f.update_layout(**PLOTLY_LAYOUT, height=470, title=f"Top 15 features — {meta['model_name']}",
                        xaxis=dict(title='Importance'))
        st.plotly_chart(f)
        st.caption('One-hot columns appear as `column_value` (e.g. `marital-status_Married-civ-spouse`).')

with tab_batch:
    st.markdown('Upload a CSV with the 12 raw input columns below (same names as the UCI file). `education-num` is the numeric '
                'education level (1 = Preschool … 16 = Doctorate). Missing categoricals can be left blank or `?` — the pipeline '
                'imputes them with the training-set mode.')
    template = pd.DataFrame([{**{k: v for k, v in p.items() if k != 'education'}, 'education-num': edu_to_num[p['education']]}
                             for p in PRESETS.values()])[INCOME_INPUT_COLS]
    st.download_button('⬇️ Download a template CSV', template.to_csv(index=False), 'income_template.csv', 'text/csv')
    up = st.file_uploader('Upload CSV', type='csv')
    if up is not None:
        try:
            raw = pd.read_csv(up, skipinitialspace=True)
            missing = [c for c in INCOME_INPUT_COLS if c not in raw.columns]
            if missing:
                st.error(f'Missing columns: {missing}')
            else:
                Xb = income_frame(raw[INCOME_INPUT_COLS], meta)
                pb = model.predict_proba(Xb)[:, 1]
                out = raw.copy()
                out['P(>50K)'] = pb.round(4)
                out['prediction'] = np.where(pb >= meta['threshold'], '>50K', '<=50K')
                k1, k2, k3 = st.columns(3)
                k1.metric('Rows scored', f'{len(out):,}', border=True)
                k2.metric('Predicted >50K', f"{(out['prediction'] == '>50K').mean():.1%}", border=True)
                k3.metric('Mean probability', f'{pb.mean():.1%}', border=True)
                st.dataframe(out, hide_index=True, column_config={
                    'P(>50K)': st.column_config.ProgressColumn('P(>50K)', min_value=0, max_value=1, format='percent')})
                st.download_button('⬇️ Download predictions', out.to_csv(index=False), 'income_predictions.csv', 'text/csv')
        except Exception as e:  # noqa: BLE001 — show any parsing problem to the user instead of crashing the page
            st.error(f'Could not score this file: {e}')

st.info('**Responsible use:** the model learns from the 1994 U.S. census, so it reproduces the income gaps of that time '
        '(e.g. by sex and race). It is a teaching demonstration of classification — not a tool for decisions about real people.',
        icon='⚖️')
