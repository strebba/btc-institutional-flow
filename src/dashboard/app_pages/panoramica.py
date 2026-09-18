import streamlit as st

from src.dashboard.tabs.panoramica import _tab_panoramica

_tab_panoramica(
    st.session_state["snap"],
    st.session_state["merged_df"],
    st.session_state["barriers"],
)
