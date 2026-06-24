import streamlit as st
import pandas as pd
import os

st.title("Redrob Candidate Ranking System")

st.markdown("### Top Ranked Candidates")

@st.cache_data
def load_data():
    filepath = os.path.join(os.path.dirname(__file__), '../outputs/team_antigravity.csv')
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return pd.DataFrame()

df = load_data()

if not df.empty:
    st.dataframe(df.head(100))
else:
    st.warning("No ranking outputs found yet. Please run the rank.py script.")
