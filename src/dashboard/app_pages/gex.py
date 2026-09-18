import streamlit as st

from src.dashboard.tabs.gex import _tab_gex

_tab_gex(
    st.session_state["snap"],
    st.session_state["gex_by_strike"],
    st.session_state["merged_df"],
)
