from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (ACCENT, BAD, GOOD, MUTED, PLOTLY_LAYOUT, PRIMARY, SEASON_SHORT, WARN, WEATHER_LABELS, bike_rows,
                   hero, load_json, load_model, pill, result_card, season_from_date)

meta = load_json('regression_meta.json')
hero('🚲 Bike Demand Forecaster',
     'Predict hourly Capital Bikeshare rentals (Washington D.C.) from the calendar and the weather. '
     f"Model: <b>{meta['model_name']}</b> — R² {meta['metrics']['R2']:.3f}, typical error ±{meta['metrics']['MAE']:.0f} bikes (MAE).",
     chips=('10 algorithms compared', 'GridSearchCV-tuned', 'cyclical hour encoding', 'rush-hour feature'))

model = load_model('bike_regressor.joblib')
profile = pd.DataFrame(meta['hourly_profile'])

# ------------------------------------------------------------------ inputs
with st.container(border=True):
    st.markdown('##### 1 · When?')
    c1, c2, c3 = st.columns([1.2, 2, 1])
    d = c1.date_input('Date', value=date(2012, 6, 15), min_value=date(2011, 1, 1), max_value=date(2030, 12, 31),
                      help='The model was trained on 2011–2012. Dates after 2012 use the 2012 demand level (yr = 1).')
    hour = c2.slider('Hour of day', 0, 23, 17, format='%d:00')
    holiday = c3.toggle('Public holiday', value=False)

    st.markdown('##### 2 · Weather')
    w1, w2, w3, w4, w5 = st.columns([1.6, 1, 1, 1, 1])
    weathersit = w1.selectbox('Conditions', options=list(WEATHER_LABELS), format_func=WEATHER_LABELS.get, index=0)
    temp_c = w2.slider('Temperature (°C)', 0.0, 41.0, 26.0, 0.5)
    feels_c = w3.slider('Feels like (°C)', 0.0, 50.0, 28.0, 0.5)
    hum = w4.slider('Humidity (%)', 0, 100, 55)
    wind = w5.slider('Wind (km/h)', 0.0, 57.0, 13.0, 0.5)

weekday_name = d.strftime('%A')
is_working = d.weekday() < 5 and not holiday
day_type = 'Working day' if is_working else ('Holiday' if holiday else 'Weekend')

# ------------------------------------------------------------------ predict the full day (24 rows) in one call
day_rows = bike_rows(d, range(24), weathersit, temp_c, feels_c, hum, wind, holiday, meta)
day_pred = np.clip(model.predict(day_rows), 0, None)
pred = float(day_pred[hour])
hist = profile[(profile['workingday'] == int(is_working)) & (profile['hr'] == hour)]['cnt'].iloc[0]
delta = pred - hist

if pred < 50:
    level, color = 'Low', MUTED
elif pred < 200:
    level, color = 'Moderate', ACCENT
elif pred < 450:
    level, color = 'High', WARN
else:
    level, color = 'Peak', BAD

st.write('')
r1, r2 = st.columns([1, 2], gap='large')
with r1:
    result_card('Predicted rentals this hour', f'{pred:,.0f} bikes',
                f'{weekday_name}, {d:%d %b %Y} · {hour:02d}:00–{hour:02d}:59<br>'
                f'{pill(level + " demand", color)}&nbsp; {pill(day_type, PRIMARY)}', color=PRIMARY)
    st.write('')
    m1, m2 = st.columns(2)
    m1.metric('2-yr avg, this hour', f'{hist:,.0f}', f'{delta:+,.0f} bikes', border=True)
    m2.metric('Predicted day total', f'{day_pred.sum():,.0f}', border=True)
    st.caption(f'Derived inputs → {SEASON_SHORT[season_from_date(d)]}, weekday code {(d.weekday() + 1) % 7}, '
               f'working day = {int(is_working)}, rush hour = {int(hour in meta["rush_hours"])}.')

with r2:
    tab1, tab2, tab3 = st.tabs(['📈 24-hour forecast', '🌡️ Temperature what-if', '🌦️ Weather what-if'])
    with tab1:
        hist_curve = profile[profile['workingday'] == int(is_working)].sort_values('hr')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_curve['hr'], y=hist_curve['cnt'], name=f'Historical average ({day_type.lower()}s)'
                                 if day_type != 'Holiday' else 'Historical average (non-working days)',
                                 line=dict(color=MUTED, dash='dot', width=2)))
        fig.add_trace(go.Scatter(x=list(range(24)), y=day_pred, name='Model forecast', fill='tozeroy',
                                 line=dict(color=PRIMARY, width=3), fillcolor='rgba(67,56,202,.12)'))
        fig.add_trace(go.Scatter(x=[hour], y=[pred], mode='markers', name='Selected hour',
                                 marker=dict(size=14, color=BAD, line=dict(color='white', width=2))))
        for h in meta['rush_hours']:
            fig.add_vrect(x0=h - .5, x1=h + .5, fillcolor=WARN, opacity=.07, line_width=0)
        fig.update_layout(**PLOTLY_LAYOUT, height=380, title='Predicted rentals for every hour of the selected day',
                          xaxis=dict(title='Hour of day', dtick=2), yaxis=dict(title='Bikes rented per hour'),
                          legend=dict(orientation='h', y=-0.22))
        st.plotly_chart(fig)
        st.caption('Shaded bands = commute rush hours (the engineered `is_rush_hour` feature). Working days show two commute '
                   'peaks; weekends show one broad midday peak.')
    with tab2:
        temps = np.arange(0, 41.5, 1.0)
        rows = pd.concat([bike_rows(d, hour, weathersit, t, t + (feels_c - temp_c), hum, wind, holiday, meta) for t in temps])
        curve = np.clip(model.predict(rows), 0, None)
        fig = go.Figure(go.Scatter(x=temps, y=curve, line=dict(color=ACCENT, width=3), name='Prediction'))
        fig.add_trace(go.Scatter(x=[temp_c], y=[pred], mode='markers', marker=dict(size=13, color=BAD), name='Current input'))
        fig.update_layout(**PLOTLY_LAYOUT, height=380, title=f'How temperature changes demand at {hour:02d}:00 (other inputs fixed)',
                          xaxis=dict(title='Temperature (°C)'), yaxis=dict(title='Predicted bikes'), showlegend=False)
        st.plotly_chart(fig)
        st.caption('Demand rises with warmth and flattens or dips at the hottest temperatures — the non-linear pattern found in '
                   'the EDA, and the reason tree ensembles beat linear regression.')
    with tab3:
        rows = pd.concat([bike_rows(d, hour, w, temp_c, feels_c, hum, wind, holiday, meta) for w in WEATHER_LABELS])
        vals = np.clip(model.predict(rows), 0, None)
        fig = go.Figure(go.Bar(x=[WEATHER_LABELS[w] for w in WEATHER_LABELS], y=vals, text=[f'{v:,.0f}' for v in vals],
                               textposition='outside', marker_color=[GOOD, ACCENT, WARN, BAD]))
        fig.update_layout(**PLOTLY_LAYOUT, height=380, title=f'Same hour, different weather ({hour:02d}:00)',
                          yaxis=dict(title='Predicted bikes'), xaxis=dict(title=''))
        st.plotly_chart(fig)
        st.caption('Category 4 (heavy rain/storm) appears only 3 times in the training data, so predictions for it are the '
                   'least reliable.')

with st.expander('ℹ️ About this model'):
    st.markdown(f"""
- **Algorithm:** {meta['model_name']} — chosen as the highest test-set R² among 10 algorithms and 2 GridSearchCV-tuned variants.
- **Test-set performance** ({meta['n_test']:,} held-out hours): R² = **{meta['metrics']['R2']:.4f}**, RMSE = **{meta['metrics']['RMSE']:.1f}**,
  MAE = **{meta['metrics']['MAE']:.1f}** bikes.
- **Inputs used:** season, year, month, weekday (one-hot) + holiday, working day, temperature, feels-like, humidity, wind,
  `hr_sin`/`hr_cos` (cyclical hour) and `is_rush_hour` (standardised). `casual` / `registered` were dropped because they sum to the target.
- **Unit conversion:** the UCI data stores weather normalised — temperature ÷ 41, feels-like ÷ 50, humidity ÷ 100, wind ÷ 67 — so the
  app converts your real-world values the same way before predicting.
""")
