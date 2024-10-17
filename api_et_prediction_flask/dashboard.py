import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Obtenez le répertoire courant du script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construisez le chemin approprié pour vos fichiers CSV
path_df_train = os.path.join(current_directory, "nouveaux_clients.csv")
path_definition_features_df = os.path.join(
    current_directory, "definition_features.csv"
)

df_train = pd.read_csv(path_df_train)
definition_features_df = pd.read_csv(path_definition_features_df)

st.set_page_config(layout="wide")

def get_title_font_size(height):
    base_size = 12
    scale_factor = height / 600.0
    return base_size * scale_factor

def generate_figure(df, title_text, x_anchor, yaxis_categoryorder, yaxis_side):
    fig = go.Figure(data=[go.Bar(y=df["Feature"], x=df["SHAP Value"], orientation="h")])
    annotations = generate_annotations(df, x_anchor)

    title_font_size = get_title_font_size(600)
    fig.update_layout(
        annotations=annotations,
        title_text=title_text,
        title_x=0.25,
        title_y=0.88,
        title_font=dict(size=title_font_size),
        yaxis=dict(
            categoryorder=yaxis_categoryorder, side=yaxis_side, tickfont=dict(size=14)
        ),
        height=600,
    )
    fig.update_xaxes(title_text="Impact des fonctionnalités")
    return fig

def generate_annotations(df, x_anchor):
    annotations = []
    for y_val, x_val, feat_val in zip(
        df["Feature"], df["SHAP Value"], df["Feature Value"]
    ):
        formatted_feat_val = (
            feat_val
            if pd.isna(feat_val)
            else (int(feat_val) if feat_val == int(feat_val) else feat_val)
        )
        annotations.append(
            dict(
                x=x_val,
                y=y_val,
                text=f"<b>{formatted_feat_val}</b>",
                showarrow=False,
                xanchor=x_anchor,
                yanchor="middle",
                font=dict(color="white"),
            )
        )
    return annotations

def compute_color(value):
    return "green" if value < 48 else "red"

def format_value(val):
    if pd.isna(val):
        return val
    if isinstance(val, (float, int)):
        if val == int(val):
            return int(val)
        return round(val, 2)
    return val

def find_closest_description(feature_name, definitions_df):
    for index, row in definitions_df.iterrows():
        if row["Row"] in feature_name:
            return row["Description"]
    return None

# État de la session pour garder les données
def get_state():
    if "state" not in st.session_state:
        st.session_state["state"] = {
            "data_received": False,
            "data": None,
            "last_sk_id_curr": None,
        }
    return st.session_state["state"]

state = get_state()

st.markdown(
    "<h1 style='text-align: center; color: black;'>Estimation du risque de non-remboursement</h1>",
    unsafe_allow_html=True,
)
sk_id_curr = st.text_input("Entrez le SK_ID_CURR:", on_change=lambda: state.update(run=True))
col1, col2 = st.columns([1, 20])

if col1.button("Run") or state["data_received"]:
    if state["last_sk_id_curr"] != sk_id_curr:
        state["data_received"] = False
        state["last_sk_id_curr"] = sk_id_curr

    if not state["data_received"]:
        response = requests.post(
            "http://localhost:5000/predict", json={"SK_ID_CURR": int(sk_id_curr)}
        )
        if response.status_code != 200:
            st.error(f"Erreur lors de l'appel à l'API: {response.status_code}")
            st.stop()

        state["data"] = response.json()
        state["data_received"] = True

    data = state["data"]

    proba = data["probability"]
    feature_names = data["feature_names"]
    shap_values = data["shap_values"]
    feature_values = data["feature_values"]

    shap_df = pd.DataFrame(
        list(zip(feature_names, shap_values, [format_value(val) for val in feature_values])),
        columns=["Feature", "SHAP Value", "Feature Value"],
    )

    color = compute_color(proba)
    st.empty()
    col2.markdown(
        f"<p style='margin: 10px;'>La probabilité que ce client ne puisse pas rembourser son crédit est de <span style='color:{color}; font-weight:bold;'>{proba:.2f}%</span> (tolérance max: <strong>48%</strong>)</p>",
        unsafe_allow_html=True,
    )

    decision_message = "Le prêt sera accordé." if proba < 48 else "Le prêt ne sera pas accordé."
    st.markdown(
        f"<div style='text-align: center; color:{color}; font-size:30px; border:2px solid {color}; padding:10px;'>{decision_message}</div>",
        unsafe_allow_html=True,
    )

    top_positive_shap = shap_df.sort_values(by="SHAP Value", ascending=False).head(10)
    top_negative_shap = shap_df.sort_values(by="SHAP Value").head(10)

    fig_positive = generate_figure(
        top_positive_shap,
        "Top 10 des fonctionnalités augmentant le risque de non-remboursement",
        "right",
        "total ascending",
        "left",
    )
    fig_negative = generate_figure(
        top_negative_shap,
        "Top 10 des fonctionnalités réduisant le risque de non-remboursement",
        "left",
        "total descending",
        "right",
    )

    col1.plotly_chart(fig_positive, use_container_width=True)
    col2.plotly_chart(fig_negative, use_container_width=True)
