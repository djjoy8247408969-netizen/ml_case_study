import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import BAD, GOOD, MUTED, PLOTLY_LAYOUT, PRIMARY, hero, load_csv, load_json

hero('🏆 Model Leaderboards',
     'Every algorithm, scored on the same held-out test set within each track. These tables are written by the notebooks '
     '(Section "Saving the final model") — nothing here is re-computed or hand-typed.',
     chips=('same split per track', 'random_state = 42', 'train-only preprocessing'))

t1, t2, t3 = st.tabs(['🚲 Regression (12)', '💼 Classification (10 + tuned)', '🛒 Clustering (5)'])

with t1:
    reg = load_csv('regression_results.csv')
    best = reg.iloc[0]
    st.markdown(f"**Best model:** {best['Model']}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Models compared', len(reg), border=True)
    k2.metric('R²', f"{best['R2']:.4f}", border=True)
    k3.metric('RMSE (bikes)', f"{best['RMSE']:.1f}", border=True)
    k4.metric('MAE (bikes)', f"{best['MAE']:.1f}", border=True)
    c1, c2 = st.columns([1.1, 1], gap='large')
    with c1:
        d = reg.sort_values('R2')
        colors = [GOOD if m == best['Model'] else (PRIMARY if '(tuned)' in m else MUTED) for m in d['Model']]
        f = go.Figure(go.Bar(x=d['R2'], y=d['Model'], orientation='h', marker_color=colors,
                             text=[f'{v:.3f}' for v in d['R2']], textposition='outside'))
        f.update_layout(**PLOTLY_LAYOUT, height=480, title='Test-set R² (green = best, indigo = GridSearchCV-tuned)',
                        xaxis=dict(range=[0.55, 1.04], title='R²'))
        st.plotly_chart(f)
    with c2:
        st.dataframe(reg, hide_index=True, height=480, column_config={
            'R2': st.column_config.ProgressColumn('R²', min_value=0, max_value=1, format='%.4f'),
            'RMSE': st.column_config.NumberColumn(format='%.2f'), 'MAE': st.column_config.NumberColumn(format='%.2f')})
    st.caption('Linear models plateau at R² ≈ 0.65 because the hour/temperature effects are strongly non-linear; tree ensembles '
               'capture them. Tuning the Random Forest (deeper trees, 200 estimators) added +0.017 R².')

with t2:
    clf = load_csv('classification_results.csv')
    meta = load_json('classification_meta.json')
    tuned = clf.iloc[0]
    st.markdown(f"**Final model:** {tuned['Model']} — chosen as the best of 10, then tuned with RandomizedSearchCV")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Models compared', len(clf) - 1, border=True)
    k2.metric('ROC-AUC', f"{tuned['ROC-AUC']:.4f}", border=True)
    k3.metric('Weighted F1', f"{tuned['F1 (weighted)']:.4f}", border=True)
    k4.metric('Accuracy', f"{tuned['Accuracy']:.4f}", border=True)
    metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1 (weighted)', 'ROC-AUC']
    hm = clf.set_index('Model')[metric_cols]
    f = px.imshow(hm, text_auto='.3f', color_continuous_scale='Greens', aspect='auto',
                  zmin=hm.values.min() - 0.02, zmax=hm.values.max())
    f.update_layout(**PLOTLY_LAYOUT, height=520, title='All classification models × all mandatory metrics (sorted by weighted F1)',
                    coloraxis_showscale=False, xaxis_title='', yaxis_title='')
    st.plotly_chart(f)
    clf_show = clf.copy()
    clf_show['Train time (s)'] = clf_show['Train time (s)'].map(lambda v: '—' if v != v else f'{v:.2f}')
    st.dataframe(clf_show, hide_index=True, column_config={c: st.column_config.NumberColumn(format='%.4f') for c in metric_cols})
    st.caption('Part A = Review 1 algorithms, Part B = Review 2 algorithms, Final = the selected model after RandomizedSearchCV. '
               'SVC was trained on an 8,000-row stratified subsample (documented in the notebook) for tractability.')

with t3:
    clu = load_csv('clustering_results.csv')
    cm = load_json('clustering_meta.json')
    st.markdown('**Deployed model:** K-Means (k = 4) — the best non-degenerate segmentation, and the only one that can label new sessions')
    k1, k2, k3, k4 = st.columns(4)
    km = clu.iloc[0]
    k1.metric('Configurations compared', len(clu), border=True)
    k2.metric('Silhouette ↑', f"{km['Silhouette ↑']:.4f}", border=True)
    k3.metric('Davies-Bouldin ↓', f"{km['Davies-Bouldin ↓']:.4f}", border=True)
    k4.metric('Calinski-Harabasz ↑', f"{km['Calinski-Harabasz ↑']:,.0f}", border=True)
    st.dataframe(clu, hide_index=True, column_config={
        'Silhouette ↑': st.column_config.NumberColumn(format='%.4f'),
        'Davies-Bouldin ↓': st.column_config.NumberColumn(format='%.4f'),
        'Calinski-Harabasz ↑': st.column_config.NumberColumn(format='%.0f')})
    st.warning('**Read the metrics with the cluster sizes.** Single, average and complete linkage get the "best" Silhouette and '
               'Davies-Bouldin scores only because they leave 1–19 points in some clusters and dump almost everything into one '
               'giant cluster (chaining). Among the non-degenerate solutions, K-Means beats Ward-linkage on all three metrics.',
               icon='⚠️')
    cols = st.columns(3)
    for col, (m, better) in zip(cols, [('Silhouette ↑', 'higher'), ('Davies-Bouldin ↓', 'lower'), ('Calinski-Harabasz ↑', 'higher')]):
        colors = [BAD if deg == 'yes' else PRIMARY for deg in clu['Degenerate?']]
        f = go.Figure(go.Bar(x=clu[m], y=clu['Algorithm'], orientation='h', marker_color=colors))
        f.update_layout(**PLOTLY_LAYOUT, height=330, title=f'{m} ({better} = better)', yaxis=dict(autorange='reversed'))
        col.plotly_chart(f)
    st.caption('Red bars = degenerate segmentations (a cluster with < 2% of sessions).')
