"""
Shared helpers for the Streamlit app: paths, cached loaders, feature engineering that mirrors the notebooks,
and small UI components. Every transformation here is a copy of the corresponding notebook step so that the
app feeds the saved pipelines exactly the columns they were trained on.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'
DATA = ROOT / 'data'

# ------------------------------------------------------------------ palette (colour-blind safe, Okabe-Ito based)
PRIMARY = '#4338CA'
ACCENT = '#0EA5E9'
GOOD = '#059669'
WARN = '#D97706'
BAD = '#DC2626'
MUTED = '#64748B'
SEGMENT_COLORS = {
    'High-Intent Shoppers': '#009E73',
    'Window Shoppers': '#0072B2',
    'Account & Info Seekers': '#E69F00',
    'Bouncers': '#D55E00',
}
PLOTLY_LAYOUT = dict(
    template='plotly_white',
    font=dict(family='Inter, system-ui, sans-serif', size=13),
    margin=dict(l=10, r=10, t=50, b=10),
    hoverlabel=dict(font_size=13),
)


# ------------------------------------------------------------------ loaders (cached once per server process)
@st.cache_resource(show_spinner='Loading the trained model…')
def load_model(name: str):
    return joblib.load(MODELS / name)


@st.cache_data
def load_json(name: str) -> dict:
    with open(MODELS / name) as f:
        return json.load(f)


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(MODELS / name)


# ------------------------------------------------------------------ REGRESSION (bike sharing) — mirrors regression.ipynb B2
WEATHER_LABELS = {
    1: '☀️ Clear / partly cloudy',
    2: '🌫️ Mist / cloudy',
    3: '🌧️ Light rain or snow',
    4: '⛈️ Heavy rain / storm / fog',
}
SEASON_LABELS = {1: 'Winter (Spring in UCI coding)', 2: 'Spring/Summer', 3: 'Summer/Fall', 4: 'Fall/Winter'}
SEASON_SHORT = {1: 'Season 1 (Dec 21 – Mar 20)', 2: 'Season 2 (Mar 21 – Jun 20)',
                3: 'Season 3 (Jun 21 – Sep 22)', 4: 'Season 4 (Sep 23 – Dec 20)'}


def season_from_date(d: date) -> int:
    """UCI season coding, boundaries verified against hour.csv (season changes on 21 Mar, 21 Jun, 23 Sep, 21 Dec)."""
    md = (d.month, d.day)
    if (3, 21) <= md < (6, 21):
        return 2
    if (6, 21) <= md < (9, 23):
        return 3
    if (9, 23) <= md < (12, 21):
        return 4
    return 1


def bike_rows(d: date, hours, weathersit: int, temp_c: float, feels_c: float, hum_pct: float,
              wind_kmh: float, holiday: bool, meta: dict) -> pd.DataFrame:
    """Build model-ready rows (one per hour) exactly like the notebook's df_clean after feature engineering."""
    sc = meta['scale_constants']
    weekday = (d.weekday() + 1) % 7            # UCI: 0 = Sunday … 6 = Saturday; Python: 0 = Monday
    workingday = int(weekday not in (0, 6) and not holiday)
    hours = np.atleast_1d(hours)
    rows = pd.DataFrame({
        'season': season_from_date(d),
        'yr': 0 if d.year <= 2011 else 1,      # model only knows 2011 (0) and 2012 (1); later dates use the 2012 demand level
        'mnth': d.month,
        'holiday': int(holiday),
        'weekday': weekday,
        'workingday': workingday,
        'weathersit': weathersit,
        'temp': np.clip(temp_c / sc['temp'], 0, 1),
        'atemp': np.clip(feels_c / sc['atemp'], 0, 1),
        'hum': np.clip(hum_pct / sc['hum'], 0, 1),
        'windspeed': np.clip(wind_kmh / sc['windspeed'], 0, 1),
        'hr': hours,
    })
    rows['hr_sin'] = np.sin(2 * np.pi * rows['hr'] / 24)
    rows['hr_cos'] = np.cos(2 * np.pi * rows['hr'] / 24)
    rows['is_rush_hour'] = rows['hr'].isin(meta['rush_hours']).astype(int)
    return rows.drop(columns=['hr'])


# ------------------------------------------------------------------ CLASSIFICATION (adult income) — mirrors classification.ipynb B1–B3
INCOME_INPUT_COLS = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                     'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']


def income_frame(raw: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Apply the notebook's cleaning to raw rows: hours clipping (train-set bounds) + capital_net feature."""
    X = raw.copy()
    lo, hi = meta['hours_clip']
    X['hours-per-week'] = X['hours-per-week'].clip(lo, hi)
    X['capital_net'] = X['capital-gain'] - X['capital-loss']
    for c in meta['categorical_cols']:
        X[c] = X[c].replace({'?': np.nan, '': np.nan})
    return X


# ------------------------------------------------------------------ CLUSTERING (online shoppers) — mirrors clustering.ipynb B2
SHOPPER_RAW_COLS = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                    'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']


def add_session_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Verbatim copy of add_session_features() from clustering.ipynb."""
    out = frame.copy()
    out['total_pages'] = out['Administrative'] + out['Informational'] + out['ProductRelated']
    out['total_duration'] = out['Administrative_Duration'] + out['Informational_Duration'] + out['ProductRelated_Duration']
    has_pages = out['total_pages'] > 0
    out['avg_time_per_page'] = np.where(has_pages, out['total_duration'] / out['total_pages'].where(has_pages, 1), 0.0)
    out['product_share'] = np.where(has_pages, out['ProductRelated'] / out['total_pages'].where(has_pages, 1), 0.0)
    return out


SEGMENT_PLAYBOOK = {
    'High-Intent Shoppers': dict(
        icon='🎯',
        who='Reach high-value pages (product detail, cart, checkout) and spend real time on product pages.',
        action='Remove friction: one-click checkout, free-shipping threshold nudges, cart-abandonment email within 1 hour.'),
    'Window Shoppers': dict(
        icon='🪟',
        who='Long, product-focused visits (dozens of product pages, ~20 minutes) that almost never reach checkout-adjacent pages.',
        action='Turn interest into intent: personalised recommendations, comparison tools, time-limited offers, retargeting ads.'),
    'Account & Info Seekers': dict(
        icon='🧾',
        who='Shorter visits dominated by account and information pages (policies, shipping, sign-up); many are first-time visitors.',
        action='Make the info journey sell: product links on help/FAQ pages, welcome discount after sign-up, clear shipping & returns.'),
    'Bouncers': dict(
        icon='🚪',
        who='Leave after a single page almost immediately (bounce & exit rates near the 0.2 maximum).',
        action='Fix the first impression: landing-page speed, ad-to-page relevance, clearer above-the-fold value proposition.'),
}


# ------------------------------------------------------------------ UI helpers
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea { font-family: 'Inter', system-ui, sans-serif; }
    .block-container { padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1250px; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    .hero {
        padding: 1.8rem 2rem; border-radius: 18px; color: #fff; margin-bottom: 1.4rem;
        background: radial-gradient(1200px 300px at 10% -20%, rgba(255,255,255,.18), transparent),
                    linear-gradient(120deg, #312E81 0%, #4338CA 45%, #0EA5E9 100%);
        box-shadow: 0 10px 30px rgba(67,56,202,.25);
    }
    .hero h1 { color: #fff; margin: 0 0 .35rem 0; font-size: 2.1rem; font-weight: 800; }
    .hero p { color: rgba(255,255,255,.92); margin: 0; font-size: 1.02rem; line-height: 1.5; }
    .hero .chips { margin-top: .9rem; }
    .chip { display: inline-block; padding: .22rem .7rem; margin: 0 .35rem .35rem 0; border-radius: 999px;
            background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28); font-size: .82rem; color: #fff; }
    .result-card { border-radius: 16px; padding: 1.3rem 1.4rem; border: 1px solid rgba(100,116,139,.22);
                   background: linear-gradient(180deg, rgba(67,56,202,.06), rgba(14,165,233,.04)); }
    .result-card .label { font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; color: #64748B; font-weight: 600; }
    .result-card .value { font-size: 2.6rem; font-weight: 800; line-height: 1.1; margin: .25rem 0; }
    .result-card .sub { color: #475569; font-size: .95rem; }
    .pill { display: inline-block; padding: .2rem .65rem; border-radius: 999px; font-size: .8rem; font-weight: 700; color: #fff; }
    .track-card { border-radius: 16px; padding: 1.2rem 1.3rem; height: 100%; min-height: 285px;
                  border: 1px solid rgba(100,116,139,.22); background: rgba(148,163,184,.06); }
    .track-card h3 { margin: .2rem 0 .3rem 0; font-size: 1.15rem; }
    .track-card .big { font-size: 1.9rem; font-weight: 800; color: #4338CA; }
    .track-card .muted { color: #64748B; font-size: .88rem; }
    .step { text-align: center; padding: .7rem .4rem; border-radius: 12px; background: rgba(67,56,202,.07);
            border: 1px solid rgba(67,56,202,.15); font-size: .85rem; font-weight: 600; }
    .step .n { display:block; font-size: 1.3rem; }
    div[data-testid="stMetric"] { border-radius: 14px; }
    @media (prefers-color-scheme: dark) {
        .result-card .sub { color: #CBD5E1; }
        .track-card .big { color: #A5B4FC; }
    }
    </style>
    """, unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips=()):
    chip_html = ''.join(f'<span class="chip">{c}</span>' for c in chips)
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>'
                f'<div class="chips">{chip_html}</div></div>', unsafe_allow_html=True)


def result_card(label: str, value: str, sub: str = '', color: str = PRIMARY):
    st.markdown(f'<div class="result-card"><div class="label">{label}</div>'
                f'<div class="value" style="color:{color}">{value}</div><div class="sub">{sub}</div></div>',
                unsafe_allow_html=True)


def pill(text: str, color: str) -> str:
    return f'<span class="pill" style="background:{color}">{text}</span>'
