import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (PLOTLY_LAYOUT, PRIMARY, SEGMENT_COLORS, SEGMENT_PLAYBOOK, SHOPPER_RAW_COLS, add_session_features, hero,
                   load_csv, load_json, load_model, pill, result_card)

meta = load_json('clustering_meta.json')
bundle = load_model('shopper_segmenter.joblib')
pipe, pca = bundle['pipeline'], bundle['pca']
pre, kmeans = pipe.named_steps['pre'], pipe.named_steps['kmeans']
names = {int(k): v for k, v in meta['segment_names'].items()}
FEATS = meta['cluster_features']
pca_sample = load_csv('clustering_pca_sample.csv')
res = load_csv('clustering_results.csv')
km = res[res['Algorithm'].str.startswith('K-Means')].iloc[0]

hero('🛒 Shopper Segmentation',
     'Describe one browsing session and see which behavioural segment it belongs to. Segments were discovered with K-Means on '
     '12,330 real e-commerce sessions — <b>without</b> ever looking at whether the visitor bought anything.',
     chips=(f"K-Means · k = {meta['k']}", f"Silhouette {km['Silhouette ↑']:.3f}", f"Davies-Bouldin {km['Davies-Bouldin ↓']:.3f}",
            'log1p + StandardScaler', 'PCA map'))

PRESETS = {
    '🚪 Instant bounce': dict(adm=0, adm_t=0.0, inf=0, inf_t=0.0, prod=1, prod_t=0.0, bounce=0.2, exit=0.2, pv=0.0),
    '🪟 Window shopping': dict(adm=1, adm_t=0.5, inf=0, inf_t=0.0, prod=35, prod_t=21.0, bounce=0.01, exit=0.035, pv=0.0),
    '🧾 Checking my account': dict(adm=7, adm_t=4.0, inf=2, inf_t=2.0, prod=6, prod_t=5.0, bounce=0.005, exit=0.03, pv=0.0),
    '🎯 Ready to buy': dict(adm=4, adm_t=2.0, inf=1, inf_t=0.5, prod=55, prod_t=38.0, bounce=0.004, exit=0.017, pv=35.0),
}


def apply_preset(p):
    for k, v in p.items():
        st.session_state[f'sh_{k}'] = v


if 'sh_adm' not in st.session_state:
    apply_preset(PRESETS['🪟 Window shopping'])

st.markdown('**Load an example session:**')
for col, (name, p) in zip(st.columns(len(PRESETS)), PRESETS.items()):
    col.button(name, on_click=apply_preset, args=(p,), width='stretch')

tab_one, tab_explore, tab_batch = st.tabs(['🔮 Classify a session', '🗺️ Segment explorer', '📄 Batch segmentation (CSV)'])

with tab_one:
    with st.container(border=True):
        st.markdown('##### Pages visited in the session')
        c1, c2, c3 = st.columns(3)
        with c1:
            adm = st.number_input('Account / admin pages', 0, 30, key='sh_adm')
            adm_t = st.number_input('…minutes on them', 0.0, 60.0, step=0.5, key='sh_adm_t')
        with c2:
            inf = st.number_input('Information pages', 0, 25, key='sh_inf')
            inf_t = st.number_input('…minutes on them ', 0.0, 45.0, step=0.5, key='sh_inf_t')
        with c3:
            prod = st.number_input('Product pages', 0, 750, key='sh_prod')
            prod_t = st.number_input('…minutes on them  ', 0.0, 600.0, step=1.0, key='sh_prod_t')
        st.markdown('##### Engagement signals (Google-Analytics style)')
        d1, d2, d3 = st.columns(3)
        bounce = d1.slider('Bounce rate of pages visited', 0.0, 0.2, step=0.001, format='%.3f', key='sh_bounce')
        exit_ = d2.slider('Exit rate of pages visited', 0.0, 0.2, step=0.001, format='%.3f', key='sh_exit')
        pv = d3.number_input('Page value (avg. value of pages seen before a purchase)', 0.0, 400.0, step=1.0, key='sh_pv')

    raw = pd.DataFrame([{'Administrative': adm, 'Administrative_Duration': adm_t * 60, 'Informational': inf,
                         'Informational_Duration': inf_t * 60, 'ProductRelated': prod, 'ProductRelated_Duration': prod_t * 60,
                         'BounceRates': bounce, 'ExitRates': exit_, 'PageValues': pv}])
    feat = add_session_features(raw)
    Xt = pre.transform(feat[FEATS])
    cluster = int(kmeans.predict(Xt)[0])
    seg = names[cluster]
    dists = np.linalg.norm(kmeans.cluster_centers_ - Xt, axis=1)
    pc = pca.transform(Xt)[0]
    book = SEGMENT_PLAYBOOK[seg]
    conv = meta['conversion'][seg]

    r1, r2 = st.columns([1, 1.35], gap='large')
    with r1:
        result_card('This session belongs to', f"{book['icon']} {seg}",
                    f"{book['who']}<br><br>{pill(f'Historical conversion {conv:.1%}', SEGMENT_COLORS[seg])} "
                    f"{pill(str(meta['sessions'][seg]) + ' similar sessions', PRIMARY)}",
                    color=SEGMENT_COLORS[seg])
        st.write('')
        st.success(f"**Recommended action:** {book['action']}", icon='💡')

        dd = pd.DataFrame({'segment': [names[i] for i in range(len(dists))], 'distance': dists}).sort_values('distance')
        f = go.Figure(go.Bar(x=dd['distance'], y=dd['segment'], orientation='h',
                             marker_color=[SEGMENT_COLORS[s] for s in dd['segment']],
                             text=[f'{v:.2f}' for v in dd['distance']], textposition='outside'))
        f.update_layout(**PLOTLY_LAYOUT, height=230, title='Distance to each K-Means centroid (nearest wins)',
                        xaxis=dict(title='Distance (scaled space)', range=[0, dd['distance'].max() * 1.25]), yaxis=dict(autorange='reversed'))
        st.plotly_chart(f)

    with r2:
        fig = px.scatter(pca_sample, x='PC1', y='PC2', color='segment', color_discrete_map=SEGMENT_COLORS, opacity=.45,
                         category_orders={'segment': list(SEGMENT_COLORS)})
        fig.update_traces(marker=dict(size=5))
        fig.add_trace(go.Scatter(x=[pc[0]], y=[pc[1]], mode='markers+text', name='Your session', text=[''],
                                 textposition='middle right', textfont=dict(size=14, color='black'),
                                 marker=dict(symbol='star', size=34, color='gold', line=dict(color='black', width=2))))
        fig.add_annotation(x=pc[0], y=pc[1], text='<b>Your session</b>', showarrow=True, arrowhead=2, ax=60, ay=-55,
                           bgcolor='white', bordercolor='black', borderwidth=1, font=dict(color='black'))
        ev = meta['pca_explained_variance']
        fig.update_layout(**PLOTLY_LAYOUT, height=440, title='Where your session lands on the PCA map (3,000 real sessions)',
                          xaxis_title=f'PC1 — disengagement ({ev[0]:.0%} of variance)',
                          yaxis_title=f'PC2 — product focus & page value ({ev[1]:.0%})',
                          legend=dict(orientation='h', y=-0.2, title=None))
        st.plotly_chart(fig)

        cent = kmeans.cluster_centers_[cluster]
        nice = {'total_pages': 'pages', 'avg_time_per_page': 'sec / page', 'PageValues': 'page value',
                'BounceRates': 'bounce', 'ExitRates': 'exit', 'product_share': 'product share'}
        comp = pd.DataFrame({'feature': [nice[f] for f in FEATS] * 2, 'z': np.r_[Xt[0], cent],
                             'who': ['Your session'] * len(FEATS) + [f'{seg} centroid'] * len(FEATS)})
        f2 = px.bar(comp, x='feature', y='z', color='who', barmode='group',
                    color_discrete_sequence=['#F5B700', SEGMENT_COLORS[seg]])
        f2.update_layout(**PLOTLY_LAYOUT, height=300, title='Your session vs. its segment centre (standardised features)',
                         yaxis_title='z-score', xaxis_title='', legend=dict(orientation='h', y=1.0, yanchor='bottom', x=1, xanchor='right', title=None))
        f2.update_xaxes(tickangle=0)
        st.plotly_chart(f2)

with tab_explore:
    prof = pd.DataFrame(meta['profile']).set_index('segment')
    order = sorted(meta['conversion'], key=meta['conversion'].get, reverse=True)
    cols = st.columns(4)
    for col, s in zip(cols, order):
        b = SEGMENT_PLAYBOOK[s]
        col.markdown(f"""<div class="track-card" style="border-top: 5px solid {SEGMENT_COLORS[s]}">
            <div style="font-size:1.6rem">{b['icon']}</div><h3>{s}</h3>
            <div class="big" style="color:{SEGMENT_COLORS[s]}">{meta['conversion'][s]:.1%}</div>
            <div class="muted">conversion · {meta['sessions'][s]:,} sessions<br><br>{b['who']}</div></div>""",
                     unsafe_allow_html=True)
    st.write('')
    e1, e2 = st.columns(2, gap='large')
    with e1:
        cv = pd.DataFrame({'segment': order, 'conversion': [meta['conversion'][s] * 100 for s in order]})
        f = px.bar(cv, x='segment', y='conversion', color='segment', color_discrete_map=SEGMENT_COLORS, text_auto='.1f')
        f.add_hline(y=meta['overall_conversion'] * 100, line_dash='dash', line_color='red',
                    annotation_text=f"site average {meta['overall_conversion']:.1%}")
        f.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False, title='Purchase rate by segment (post-hoc validation)',
                        yaxis_title='Conversion %', xaxis_title='')
        st.plotly_chart(f)
    with e2:
        elbow = load_csv('clustering_elbow.csv')
        f = go.Figure()
        f.add_trace(go.Scatter(x=elbow['k'], y=elbow['Inertia'], name='Inertia', line=dict(color=PRIMARY, width=3)))
        f.add_trace(go.Scatter(x=elbow['k'], y=elbow['Silhouette'], name='Silhouette', yaxis='y2',
                               line=dict(color='#D55E00', width=3, dash='dash')))
        f.add_vline(x=meta['k'], line_dash='dot', annotation_text=f"chosen k = {meta['k']}")
        f.update_layout(**PLOTLY_LAYOUT, height=380, title='How k was chosen: Elbow + Silhouette',
                        xaxis_title='k', yaxis=dict(title='Inertia'),
                        yaxis2=dict(title='Silhouette', overlaying='y', side='right', showgrid=False),
                        legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(f)
    show = prof[['sessions', 'total_pages', 'total_duration', 'avg_time_per_page', 'PageValues', 'BounceRates', 'ExitRates',
                 'product_share']].loc[order].copy()
    show['total_duration'] = show['total_duration'] / 60
    st.dataframe(show.rename(columns={'total_duration': 'total minutes', 'avg_time_per_page': 'sec / page'}),
                 column_config={c: st.column_config.NumberColumn(format='%.2f') for c in show.columns if c != 'sessions'})

with tab_batch:
    st.markdown('Upload sessions in the **UCI Online Shoppers format** (durations in seconds). Only these columns are needed: '
                + ', '.join(f'`{c}`' for c in SHOPPER_RAW_COLS) + '. Extra columns are kept in the output.')
    tmpl = pd.DataFrame([{'Administrative': p['adm'], 'Administrative_Duration': p['adm_t'] * 60, 'Informational': p['inf'],
                          'Informational_Duration': p['inf_t'] * 60, 'ProductRelated': p['prod'],
                          'ProductRelated_Duration': p['prod_t'] * 60, 'BounceRates': p['bounce'], 'ExitRates': p['exit'],
                          'PageValues': p['pv']} for p in PRESETS.values()])
    st.download_button('⬇️ Download a template CSV', tmpl.to_csv(index=False), 'sessions_template.csv', 'text/csv')
    up = st.file_uploader('Upload CSV', type='csv', key='shop_up')
    if up is not None:
        try:
            df = pd.read_csv(up)
            missing = [c for c in SHOPPER_RAW_COLS if c not in df.columns]
            if missing:
                st.error(f'Missing columns: {missing}')
            else:
                labels = pipe.predict(add_session_features(df[SHOPPER_RAW_COLS])[FEATS])
                df['segment'] = [names[int(c)] for c in labels]
                counts = df['segment'].value_counts().reindex(list(SEGMENT_COLORS)).fillna(0).reset_index()
                counts.columns = ['segment', 'sessions']
                f = px.pie(counts, names='segment', values='sessions', color='segment', color_discrete_map=SEGMENT_COLORS, hole=.5)
                f.update_layout(**PLOTLY_LAYOUT, height=340, title=f'{len(df):,} sessions segmented')
                a, b = st.columns([1, 1.4])
                a.plotly_chart(f)
                b.dataframe(df, hide_index=True, height=340)
                st.download_button('⬇️ Download segmented sessions', df.to_csv(index=False), 'segmented_sessions.csv', 'text/csv')
        except Exception as e:  # noqa: BLE001
            st.error(f'Could not segment this file: {e}')
