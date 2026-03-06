#!/usr/bin/env python3
"""
WaxalNLP Dataset Explorer — Interactive Gradio Application
===========================================================
Explore Google's WAXAL multilingual African language speech corpus.
Analyze languages, data coverage, cross-lingual transfer potential,
and plan Igisha integration priorities.

Usage:
    pip install gradio duckdb plotly pandas numpy
    python waxal_gradio_explorer.py

Then open http://localhost:7860 in your browser.
"""

import gradio as gr
import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# ============================================================
# DATA CATALOG
# ============================================================

COLORS = {
    'primary': '#1B4332', 'accent': '#2D6A4F', 'light': '#40916C',
    'highlight': '#52B788', 'pale': '#B7E4C7', 'bg': '#F0F7F4',
}
PALETTE = ['#1B4332', '#2D6A4F', '#40916C', '#52B788', '#74C69D', '#95D5B2', '#B7E4C7', '#D8F3DC']

ASR_DATA = [
    {'language': 'Acholi',    'code': 'ach', 'config': 'ach_asr', 'provider': 'Makerere University',  'family': 'Nilotic',        'subfamily': 'Western Nilotic',   'country': 'Uganda',           'speakers_m': 2.0,  'est_hours': 55,  'license': 'CC-BY-SA-4.0', 'region': 'East Africa'},
    {'language': 'Luganda',   'code': 'lug', 'config': 'lug_asr', 'provider': 'Makerere University',  'family': 'Bantu',          'subfamily': 'Bantu J (J10)',     'country': 'Uganda',           'speakers_m': 10.0, 'est_hours': 85,  'license': 'CC-BY-SA-4.0', 'region': 'East Africa'},
    {'language': 'Masaaba',   'code': 'myx', 'config': 'myx_asr', 'provider': 'Makerere University',  'family': 'Bantu',          'subfamily': 'Bantu J (J30)',     'country': 'Uganda',           'speakers_m': 1.5,  'est_hours': 45,  'license': 'CC-BY-SA-4.0', 'region': 'East Africa'},
    {'language': 'Nyankole',  'code': 'nyn', 'config': 'nyn_asr', 'provider': 'Makerere University',  'family': 'Bantu',          'subfamily': 'Bantu J (J10)',     'country': 'Uganda',           'speakers_m': 3.0,  'est_hours': 60,  'license': 'CC-BY-SA-4.0', 'region': 'East Africa'},
    {'language': 'Soga',      'code': 'sog', 'config': 'sog_asr', 'provider': 'Makerere University',  'family': 'Bantu',          'subfamily': 'Bantu J (J10)',     'country': 'Uganda',           'speakers_m': 3.0,  'est_hours': 50,  'license': 'CC-BY-SA-4.0', 'region': 'East Africa'},
    {'language': 'Akan',      'code': 'aka', 'config': 'aka_asr', 'provider': 'University of Ghana',  'family': 'Kwa',            'subfamily': 'Volta-Niger',       'country': 'Ghana',            'speakers_m': 11.0, 'est_hours': 80,  'license': 'CC-BY-4.0',    'region': 'West Africa'},
    {'language': 'Ewe',       'code': 'ewe', 'config': 'ewe_asr', 'provider': 'University of Ghana',  'family': 'Kwa',            'subfamily': 'Gbe',              'country': 'Ghana',            'speakers_m': 7.0,  'est_hours': 65,  'license': 'CC-BY-4.0',    'region': 'West Africa'},
    {'language': 'Dagbani',   'code': 'dag', 'config': 'dag_asr', 'provider': 'University of Ghana',  'family': 'Gur',            'subfamily': 'Oti-Volta',         'country': 'Ghana',            'speakers_m': 1.2,  'est_hours': 50,  'license': 'CC-BY-4.0',    'region': 'West Africa'},
    {'language': 'Dagaare',   'code': 'dga', 'config': 'dga_asr', 'provider': 'University of Ghana',  'family': 'Gur',            'subfamily': 'Oti-Volta',         'country': 'Ghana',            'speakers_m': 1.1,  'est_hours': 45,  'license': 'CC-BY-4.0',    'region': 'West Africa'},
    {'language': 'Ikposo',    'code': 'kpo', 'config': 'kpo_asr', 'provider': 'University of Ghana',  'family': 'Kwa',            'subfamily': 'Ka-Togo',           'country': 'Togo/Ghana',       'speakers_m': 0.15, 'est_hours': 30,  'license': 'CC-BY-4.0',    'region': 'West Africa'},
    {'language': 'Fula',      'code': 'ful', 'config': 'ful_asr', 'provider': 'Digital Umuganda',     'family': 'Atlantic-Congo', 'subfamily': 'Senegambian',       'country': 'West Africa',      'speakers_m': 40.0, 'est_hours': 75,  'license': 'CC-BY-SA-4.0', 'region': 'West Africa'},
    {'language': 'Lingala',   'code': 'lin', 'config': 'lin_asr', 'provider': 'Digital Umuganda',     'family': 'Bantu',          'subfamily': 'Bantu C',           'country': 'DRC/Congo',        'speakers_m': 25.0, 'est_hours': 70,  'license': 'CC-BY-SA-4.0', 'region': 'Central Africa'},
    {'language': 'Shona',     'code': 'sna', 'config': 'sna_asr', 'provider': 'Digital Umuganda',     'family': 'Bantu',          'subfamily': 'Bantu S',           'country': 'Zimbabwe',         'speakers_m': 12.0, 'est_hours': 80,  'license': 'CC-BY-SA-4.0', 'region': 'Southern Africa'},
    {'language': 'Malagasy',  'code': 'mlg', 'config': 'mlg_asr', 'provider': 'Digital Umuganda',     'family': 'Austronesian',   'subfamily': 'Malayo-Polynesian', 'country': 'Madagascar',       'speakers_m': 25.0, 'est_hours': 65,  'license': 'CC-BY-SA-4.0', 'region': 'East Africa'},
    {'language': 'Amharic',   'code': 'amh', 'config': 'amh_asr', 'provider': 'Digital Umuganda',     'family': 'Afroasiatic',    'subfamily': 'Semitic',           'country': 'Ethiopia',         'speakers_m': 32.0, 'est_hours': 90,  'license': 'CC-BY-SA-4.0', 'region': 'East Africa'},
    {'language': 'Oromo',     'code': 'orm', 'config': 'orm_asr', 'provider': 'Digital Umuganda',     'family': 'Afroasiatic',    'subfamily': 'Cushitic',          'country': 'Ethiopia',         'speakers_m': 36.0, 'est_hours': 85,  'license': 'CC-BY-SA-4.0', 'region': 'East Africa'},
    {'language': 'Sidama',    'code': 'sid', 'config': 'sid_asr', 'provider': 'Digital Umuganda',     'family': 'Afroasiatic',    'subfamily': 'Cushitic',          'country': 'Ethiopia',         'speakers_m': 4.0,  'est_hours': 45,  'license': 'CC-BY-SA-4.0', 'region': 'East Africa'},
    {'language': 'Tigrinya',  'code': 'tir', 'config': 'tir_asr', 'provider': 'Digital Umuganda',     'family': 'Afroasiatic',    'subfamily': 'Semitic',           'country': 'Eritrea/Ethiopia', 'speakers_m': 9.0,  'est_hours': 60,  'license': 'CC-BY-SA-4.0', 'region': 'East Africa'},
    {'language': 'Wolaytta',  'code': 'wal', 'config': 'wal_asr', 'provider': 'Digital Umuganda',     'family': 'Omotic',         'subfamily': 'North Omotic',      'country': 'Ethiopia',         'speakers_m': 2.5,  'est_hours': 40,  'license': 'CC-BY-SA-4.0', 'region': 'East Africa'},
]

TTS_DATA = [
    {'language': 'Acholi',          'code': 'ach', 'config': 'ach_tts', 'provider': 'Makerere University',  'family': 'Nilotic',        'country': 'Uganda',         'est_hours': 10, 'license': 'CC-BY-SA-4.0', 'region': 'East Africa'},
    {'language': 'Luganda',         'code': 'lug', 'config': 'lug_tts', 'provider': 'Makerere University',  'family': 'Bantu',          'country': 'Uganda',         'est_hours': 15, 'license': 'CC-BY-SA-4.0', 'region': 'East Africa'},
    {'language': 'Kiswahili',       'code': 'swa', 'config': 'swa_tts', 'provider': 'Makerere University',  'family': 'Bantu',          'country': 'Tanzania/Kenya', 'est_hours': 18, 'license': 'CC-BY-SA-4.0', 'region': 'East Africa'},
    {'language': 'Nyankole',        'code': 'nyn', 'config': 'nyn_tts', 'provider': 'Makerere University',  'family': 'Bantu',          'country': 'Uganda',         'est_hours': 12, 'license': 'CC-BY-SA-4.0', 'region': 'East Africa'},
    {'language': 'Akan (Fante)',    'code': 'fat', 'config': 'fat_tts', 'provider': 'University of Ghana',  'family': 'Kwa',            'country': 'Ghana',          'est_hours': 12, 'license': 'CC-BY-4.0',    'region': 'West Africa'},
    {'language': 'Akan (Twi)',      'code': 'twi', 'config': 'twi_tts', 'provider': 'University of Ghana',  'family': 'Kwa',            'country': 'Ghana',          'est_hours': 12, 'license': 'CC-BY-4.0',    'region': 'West Africa'},
    {'language': 'Fula',            'code': 'ful', 'config': 'ful_tts', 'provider': 'Media Trust',          'family': 'Atlantic-Congo', 'country': 'Nigeria',        'est_hours': 12, 'license': 'CC-BY-SA-4.0', 'region': 'West Africa'},
    {'language': 'Igbo',            'code': 'ibo', 'config': 'ibo_tts', 'provider': 'Media Trust',          'family': 'Volta-Niger',    'country': 'Nigeria',        'est_hours': 15, 'license': 'CC-BY-SA-4.0', 'region': 'West Africa'},
    {'language': 'Hausa',           'code': 'hau', 'config': 'hau_tts', 'provider': 'Media Trust',          'family': 'Afroasiatic',    'country': 'Nigeria',        'est_hours': 18, 'license': 'CC-BY-SA-4.0', 'region': 'West Africa'},
    {'language': 'Yoruba',          'code': 'yor', 'config': 'yor_tts', 'provider': 'Media Trust',          'family': 'Volta-Niger',    'country': 'Nigeria',        'est_hours': 15, 'license': 'CC-BY-SA-4.0', 'region': 'West Africa'},
    {'language': 'Nigerian Pidgin', 'code': 'pcm', 'config': 'pcm_tts', 'provider': 'Media Trust',          'family': 'Creole',         'country': 'Nigeria',        'est_hours': 15, 'license': 'CC-BY-SA-4.0', 'region': 'West Africa'},
    {'language': 'Kikuyu',          'code': 'kik', 'config': 'kik_tts', 'provider': 'Loud and Clear',       'family': 'Bantu',          'country': 'Kenya',          'est_hours': 10, 'license': 'CC-BY-SA-4.0', 'region': 'East Africa'},
    {'language': 'Luo',             'code': 'luo', 'config': 'luo_tts', 'provider': 'Loud and Clear',       'family': 'Nilotic',        'country': 'Kenya',          'est_hours': 10, 'license': 'CC-BY-SA-4.0', 'region': 'East Africa'},
]

TRANSFER_SCORES = [
    {'language': 'Nyankole',  'phoneme': 90, 'morphology': 94, 'tonal': 80, 'geographic': 98, 'overall': 91},
    {'language': 'Luganda',   'phoneme': 88, 'morphology': 92, 'tonal': 75, 'geographic': 95, 'overall': 88},
    {'language': 'Soga',      'phoneme': 85, 'morphology': 88, 'tonal': 72, 'geographic': 90, 'overall': 84},
    {'language': 'Masaaba',   'phoneme': 70, 'morphology': 75, 'tonal': 65, 'geographic': 85, 'overall': 74},
    {'language': 'Kiswahili', 'phoneme': 65, 'morphology': 70, 'tonal': 40, 'geographic': 80, 'overall': 64},
    {'language': 'Lingala',   'phoneme': 60, 'morphology': 65, 'tonal': 55, 'geographic': 70, 'overall': 63},
    {'language': 'Shona',     'phoneme': 55, 'morphology': 62, 'tonal': 50, 'geographic': 50, 'overall': 54},
    {'language': 'Acholi',    'phoneme': 35, 'morphology': 20, 'tonal': 30, 'geographic': 90, 'overall': 44},
    {'language': 'Akan',      'phoneme': 40, 'morphology': 30, 'tonal': 35, 'geographic': 20, 'overall': 31},
    {'language': 'Amharic',   'phoneme': 25, 'morphology': 15, 'tonal': 10, 'geographic': 55, 'overall': 26},
    {'language': 'Yoruba',    'phoneme': 30, 'morphology': 18, 'tonal': 40, 'geographic': 10, 'overall': 25},
    {'language': 'Hausa',     'phoneme': 20, 'morphology': 10, 'tonal': 25, 'geographic': 15, 'overall': 18},
]

asr_df = pd.DataFrame(ASR_DATA)
tts_df = pd.DataFrame(TTS_DATA)
transfer_df = pd.DataFrame(TRANSFER_SCORES)

con = duckdb.connect()
con.execute("CREATE TABLE asr AS SELECT * FROM asr_df")
con.execute("CREATE TABLE tts AS SELECT * FROM tts_df")


# ============================================================
# TAB FUNCTIONS
# ============================================================

def overview_tab():
    """Dataset overview statistics."""
    total_asr = asr_df['est_hours'].sum()
    total_tts = tts_df['est_hours'].sum()
    total_speakers = asr_df['speakers_m'].sum()

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=['ASR Languages', 'TTS Languages', 'Speaker Coverage']
    )
    fig.add_trace(go.Indicator(mode='number+delta', value=len(asr_df),
                               delta={'reference': 15, 'suffix': ' vs prev'},
                               number={'font': {'size': 60, 'color': COLORS['primary']}}), row=1, col=1)
    fig.add_trace(go.Indicator(mode='number', value=len(tts_df),
                               number={'font': {'size': 60, 'color': COLORS['accent']}}), row=1, col=2)
    fig.add_trace(go.Indicator(mode='number', value=total_speakers,
                               number={'font': {'size': 60, 'color': COLORS['light']}, 'suffix': 'M'}), row=1, col=3)
    fig.update_layout(height=250, margin=dict(t=50, b=20))

    summary = f"""### WaxalNLP Dataset Summary
- **ASR**: {len(asr_df)} languages, ~{total_asr} hours of transcribed speech
- **TTS**: {len(tts_df)} languages, ~{total_tts} hours of studio recordings
- **Total**: ~{total_asr + total_tts} hours across {len(set(asr_df['code']) | set(tts_df['code']))} unique languages
- **Coverage**: ~{total_speakers:.0f} million speakers across 12+ countries
- **Size**: ~821 GB (Parquet), ~2.59M samples
- **License**: CC-BY-SA-4.0 / CC-BY-4.0
"""
    return fig, summary


def asr_analysis(sort_by, filter_family):
    """ASR dataset analysis with filters."""
    df = asr_df.copy()
    if filter_family != "All":
        df = df[df['family'] == filter_family]

    sort_col = {'Hours': 'est_hours', 'Speakers (M)': 'speakers_m', 'Language': 'language'}[sort_by]
    df = df.sort_values(sort_col, ascending=sort_col == 'language')

    fig = px.bar(
        df, x='language', y='est_hours', color='provider',
        color_discrete_sequence=PALETTE,
        title=f'ASR Hours by Language (Family: {filter_family})',
        labels={'est_hours': 'Estimated Hours', 'language': 'Language'},
        hover_data=['country', 'family', 'speakers_m']
    )
    fig.update_layout(height=450, xaxis_tickangle=45)

    table = df[['language', 'code', 'est_hours', 'speakers_m', 'family', 'provider', 'license']].reset_index(drop=True)
    return fig, table


def tts_analysis(sort_by):
    """TTS dataset analysis."""
    df = tts_df.copy()
    sort_col = {'Hours': 'est_hours', 'Language': 'language', 'Provider': 'provider'}[sort_by]
    df = df.sort_values(sort_col, ascending=sort_col != 'est_hours')

    fig = px.bar(
        df, x='language', y='est_hours', color='family',
        color_discrete_sequence=PALETTE,
        title='TTS Hours by Language (Single-Speaker Studio Recordings)',
        labels={'est_hours': 'Estimated Hours', 'language': 'Language'},
        hover_data=['provider', 'country']
    )
    fig.update_layout(height=450, xaxis_tickangle=45)

    table = df[['language', 'code', 'est_hours', 'family', 'provider', 'license']].reset_index(drop=True)
    return fig, table


def coverage_analysis():
    """ASR vs TTS coverage gap analysis."""
    all_codes = sorted(set(asr_df['code'].tolist() + tts_df['code'].tolist()))
    rows = []
    for code in all_codes:
        asr_row = asr_df[asr_df['code'] == code]
        tts_row = tts_df[tts_df['code'] == code]
        lang = asr_row['language'].values[0] if len(asr_row) > 0 else tts_row['language'].values[0]
        rows.append({
            'language': lang, 'code': code,
            'asr_hours': asr_row['est_hours'].values[0] if len(asr_row) > 0 else 0,
            'tts_hours': tts_row['est_hours'].values[0] if len(tts_row) > 0 else 0,
        })
    cov = pd.DataFrame(rows)
    cov['total'] = cov['asr_hours'] + cov['tts_hours']
    cov['status'] = cov.apply(
        lambda r: 'Both' if r['asr_hours'] > 0 and r['tts_hours'] > 0
        else 'ASR Only' if r['asr_hours'] > 0 else 'TTS Only', axis=1)
    cov = cov.sort_values('total', ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(name='ASR', x=cov['language'], y=cov['asr_hours'], marker_color=COLORS['primary']))
    fig.add_trace(go.Bar(name='TTS', x=cov['language'], y=cov['tts_hours'], marker_color=COLORS['highlight']))
    fig.update_layout(barmode='stack', height=500, title='ASR + TTS Coverage (Stacked)', xaxis_tickangle=45)

    summary = f"**Both ASR+TTS**: {(cov['status']=='Both').sum()} | **ASR Only**: {(cov['status']=='ASR Only').sum()} | **TTS Only**: {(cov['status']=='TTS Only').sum()}"
    return fig, cov[['language', 'code', 'asr_hours', 'tts_hours', 'total', 'status']], summary


def family_analysis():
    """Language family clustering."""
    fig = px.sunburst(
        asr_df, path=['family', 'subfamily', 'language'], values='est_hours',
        color='family', color_discrete_sequence=px.colors.qualitative.Set2,
        title='Language Family Hierarchy — ASR Data Hours',
    )
    fig.update_layout(height=550)

    stats = con.execute("""
        SELECT family, COUNT(*) as n_langs, SUM(est_hours) as total_hours,
               ROUND(SUM(speakers_m), 1) as total_speakers_m,
               STRING_AGG(language, ', ' ORDER BY language) as languages
        FROM asr GROUP BY family ORDER BY total_hours DESC
    """).fetchdf()
    return fig, stats


def transfer_analysis(dimension):
    """Cross-lingual transfer potential analysis."""
    if dimension == "Overall Comparison":
        fig = px.bar(
            transfer_df.sort_values('overall'), x='overall', y='language',
            orientation='h', color='overall',
            color_continuous_scale='YlGn',
            title='Cross-Lingual Transfer Potential to Kinyarwanda (Overall Score)',
            labels={'overall': 'Similarity Score (0-100)', 'language': ''}
        )
        fig.update_layout(height=500)
    elif dimension == "Radar — Top 5":
        cats = ['Phoneme', 'Morphology', 'Tonal', 'Geographic']
        fig = go.Figure()
        for _, row in transfer_df.head(5).iterrows():
            vals = [row['phoneme'], row['morphology'], row['tonal'], row['geographic']]
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill='toself', name=row['language'], opacity=0.6
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                         title='Top 5 Languages — Transfer Dimensions', height=500)
    else:  # Heatmap
        cols = ['phoneme', 'morphology', 'tonal', 'geographic', 'overall']
        matrix = transfer_df.set_index('language')[cols].sort_values('overall', ascending=False)
        fig = px.imshow(
            matrix, color_continuous_scale='YlGn', aspect='auto',
            labels=dict(x='Dimension', y='Language', color='Score'),
            title='Transfer Potential Heatmap'
        )
        fig.update_layout(height=500)

    return fig, transfer_df.sort_values('overall', ascending=False)


def igisha_scorecard(w_demand, w_similarity, w_quality, w_effort):
    """Dynamic integration priority scorecard with adjustable weights."""
    total_w = w_demand + w_similarity + w_quality + w_effort
    if total_w == 0:
        total_w = 1

    scores = pd.DataFrame([
        {'language': 'Swahili (TTS)',  'similarity': 64, 'quality': 90, 'demand': 95, 'effort': 90, 'has_asr': False, 'has_tts': True},
        {'language': 'Luganda',        'similarity': 88, 'quality': 85, 'demand': 85, 'effort': 65, 'has_asr': True,  'has_tts': True},
        {'language': 'Nyankole',       'similarity': 91, 'quality': 80, 'demand': 60, 'effort': 60, 'has_asr': True,  'has_tts': True},
        {'language': 'Lingala',        'similarity': 63, 'quality': 75, 'demand': 70, 'effort': 55, 'has_asr': True,  'has_tts': False},
        {'language': 'Acholi',         'similarity': 44, 'quality': 80, 'demand': 40, 'effort': 55, 'has_asr': True,  'has_tts': True},
        {'language': 'Amharic',        'similarity': 26, 'quality': 85, 'demand': 55, 'effort': 50, 'has_asr': True,  'has_tts': False},
        {'language': 'Hausa',          'similarity': 18, 'quality': 85, 'demand': 50, 'effort': 45, 'has_asr': False, 'has_tts': True},
        {'language': 'Yoruba',         'similarity': 25, 'quality': 80, 'demand': 45, 'effort': 45, 'has_asr': False, 'has_tts': True},
    ])

    scores['composite'] = (
        scores['demand'] * (w_demand / total_w) +
        scores['similarity'] * (w_similarity / total_w) +
        scores['quality'] * (w_quality / total_w) +
        scores['effort'] * (w_effort / total_w)
    ).round(1)

    scores = scores.sort_values('composite', ascending=False)
    scores['rank'] = range(1, len(scores) + 1)
    scores['priority'] = scores['rank'].map(lambda r: 'P0' if r <= 2 else 'P1' if r <= 4 else 'P2' if r <= 6 else 'P3')

    fig = px.bar(
        scores.sort_values('composite'),
        x='composite', y='language', orientation='h',
        color='priority',
        color_discrete_map={'P0': '#1B4332', 'P1': '#2D6A4F', 'P2': '#74C69D', 'P3': '#B7E4C7'},
        title='Integration Priority Scorecard (Weighted)',
        labels={'composite': 'Composite Score', 'language': ''},
        hover_data=['similarity', 'quality', 'demand', 'effort']
    )
    fig.update_layout(height=450)

    return fig, scores[['rank', 'priority', 'language', 'composite', 'similarity', 'quality', 'demand', 'effort', 'has_asr', 'has_tts']]


def run_duckdb_query(query):
    """Execute a custom DuckDB query against the catalog."""
    try:
        result = con.execute(query).fetchdf()
        return result, f"Query returned {len(result)} rows."
    except Exception as e:
        return pd.DataFrame(), f"Error: {str(e)}"


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(title="WaxalNLP Explorer") as app:

    gr.Markdown("""
    # WaxalNLP Dataset Explorer
    ### Google Research Africa's Multilingual Speech Corpus — Interactive Analysis for Igisha Integration
    ---
    """)

    with gr.Tab("Overview"):
        overview_btn = gr.Button("Load Overview", variant="primary")
        overview_plot = gr.Plot()
        overview_md = gr.Markdown()
        overview_btn.click(overview_tab, outputs=[overview_plot, overview_md])

    with gr.Tab("ASR Analysis"):
        with gr.Row():
            asr_sort = gr.Dropdown(["Hours", "Speakers (M)", "Language"], value="Hours", label="Sort by")
            asr_family = gr.Dropdown(["All"] + sorted(asr_df['family'].unique().tolist()), value="All", label="Filter Family")
        asr_btn = gr.Button("Analyze ASR Data", variant="primary")
        asr_plot = gr.Plot()
        asr_table = gr.Dataframe(label="ASR Dataset Catalog")
        asr_btn.click(asr_analysis, inputs=[asr_sort, asr_family], outputs=[asr_plot, asr_table])

    with gr.Tab("TTS Analysis"):
        tts_sort = gr.Dropdown(["Hours", "Language", "Provider"], value="Hours", label="Sort by")
        tts_btn = gr.Button("Analyze TTS Data", variant="primary")
        tts_plot = gr.Plot()
        tts_table = gr.Dataframe(label="TTS Dataset Catalog")
        tts_btn.click(tts_analysis, inputs=[tts_sort], outputs=[tts_plot, tts_table])

    with gr.Tab("Coverage Gaps"):
        cov_btn = gr.Button("Analyze Coverage", variant="primary")
        cov_plot = gr.Plot()
        cov_table = gr.Dataframe(label="Coverage Matrix")
        cov_md = gr.Markdown()
        cov_btn.click(coverage_analysis, outputs=[cov_plot, cov_table, cov_md])

    with gr.Tab("Language Families"):
        fam_btn = gr.Button("Explore Families", variant="primary")
        fam_plot = gr.Plot()
        fam_table = gr.Dataframe(label="Family Statistics")
        fam_btn.click(family_analysis, outputs=[fam_plot, fam_table])

    with gr.Tab("Transfer Potential"):
        transfer_dim = gr.Dropdown(
            ["Overall Comparison", "Radar — Top 5", "Heatmap"],
            value="Overall Comparison", label="Visualization"
        )
        transfer_btn = gr.Button("Analyze Transfer", variant="primary")
        transfer_plot = gr.Plot()
        transfer_table = gr.Dataframe(label="Transfer Scores (to Kinyarwanda)")
        transfer_btn.click(transfer_analysis, inputs=[transfer_dim], outputs=[transfer_plot, transfer_table])

    with gr.Tab("Igisha Scorecard"):
        gr.Markdown("### Adjust weights to reprioritize integration candidates:")
        with gr.Row():
            w1 = gr.Slider(0, 100, value=35, step=5, label="User Demand")
            w2 = gr.Slider(0, 100, value=25, step=5, label="Linguistic Similarity")
            w3 = gr.Slider(0, 100, value=20, step=5, label="Data Quality")
            w4 = gr.Slider(0, 100, value=20, step=5, label="Implementation Ease")
        score_btn = gr.Button("Calculate Priorities", variant="primary")
        score_plot = gr.Plot()
        score_table = gr.Dataframe(label="Priority Scorecard")
        score_btn.click(igisha_scorecard, inputs=[w1, w2, w3, w4], outputs=[score_plot, score_table])

    with gr.Tab("DuckDB Query"):
        gr.Markdown("### Run custom SQL against the dataset catalog\nAvailable tables: `asr`, `tts`")
        query_input = gr.Textbox(
            value="SELECT language, est_hours, family, provider\nFROM asr\nWHERE family = 'Bantu'\nORDER BY est_hours DESC",
            lines=5, label="SQL Query"
        )
        query_btn = gr.Button("Execute Query", variant="primary")
        query_table = gr.Dataframe(label="Results")
        query_status = gr.Markdown()
        query_btn.click(run_duckdb_query, inputs=[query_input], outputs=[query_table, query_status])

    gr.Markdown("""
    ---
    *Data from [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP) |
    Paper: [arXiv:2602.02734](https://arxiv.org/abs/2602.02734) |
    Built for [Igisha](https://igisha.org) language learning platform*
    """)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0", server_port=7860, share=False,
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald",
                             font=["Inter", "system-ui", "sans-serif"]),
        css=".gradio-container { max-width: 1200px !important; }"
    )
